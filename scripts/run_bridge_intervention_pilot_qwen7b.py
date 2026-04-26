#!/usr/bin/env python3
"""Run exploratory bridge intervention pilot for Qwen2.5-7B."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_bridge_dataset
from src.bridge_benchmark.protocol import BRIDGE_PROTOCOL_VERSION, FIXED_BREAKDOWNS, FIXED_SCENARIOS
from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_json, save_jsonl
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_bool(values: Iterable[bool]) -> float | None:
    vals = [1.0 if value else 0.0 for value in values]
    return (sum(vals) / len(vals)) if vals else None


def _group_rows(items: Sequence[Dict[str, Any]], level: str) -> List[tuple[str, List[Dict[str, Any]]]]:
    if level == "all":
        return [("all", list(items))]
    if level == "source":
        keys = sorted({str(item.get("source") or "") for item in items})
        return [(key, [item for item in items if str(item.get("source") or "") == key]) for key in keys]
    if level == "pressure_type":
        keys = sorted({str(item.get("pressure_type") or "") for item in items})
        return [(key, [item for item in items if str(item.get("pressure_type") or "") == key]) for key in keys]
    raise ValueError(f"Unsupported breakdown level: {level}")


def _rows_for_breakdowns(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for level in FIXED_BREAKDOWNS:
        for key, subset in _group_rows(items, level):
            rows.append(
                {
                    "aggregation_level": level,
                    "source": key if level == "source" else "",
                    "pressure_type": key if level == "pressure_type" else "",
                    "items": subset,
                }
            )
    return rows


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _extract_choice(response_text: str) -> str:
    text = str(response_text or "").upper()
    for pattern in (r"FINAL\s+ANSWER\s*[:：]\s*([AB])\b", r"ANSWER\s*[:：]\s*([AB])\b", r"\(([AB])\)", r"\b([AB])\b"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _is_preferred_case(row: Dict[str, Any]) -> bool:
    return bool(row.get("stance_drift")) or bool(row.get("pressured_compliance"))


def _choose_preferred_pilot_ids(full_run_dir: Path, pilot_n: int) -> List[str]:
    comparisons = _load_jsonl(full_run_dir / "bridge_case_comparisons.jsonl")
    preferred = [row for row in comparisons if _is_preferred_case(row)]
    fallback = [row for row in comparisons if row not in preferred]
    selected = (preferred + fallback)[: int(pilot_n)]
    return [str(row["item_id"]) for row in selected]


def _choose_stratified_pilot_ids(full_run_dir: Path, pilot_n: int) -> List[str]:
    comparisons = _load_jsonl(full_run_dir / "bridge_case_comparisons.jsonl")
    strata = sorted({(str(row.get("source") or ""), str(row.get("pressure_type") or "")) for row in comparisons})
    if not strata:
        return []

    base_quota = int(pilot_n) // len(strata)
    remainder = int(pilot_n) % len(strata)
    selected_rows: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    for idx, stratum in enumerate(strata):
        quota = base_quota + (1 if idx < remainder else 0)
        candidates = [
            row for row in comparisons
            if (str(row.get("source") or ""), str(row.get("pressure_type") or "")) == stratum
        ]
        preferred = [row for row in candidates if _is_preferred_case(row)]
        fallback = [row for row in candidates if row not in preferred]
        for row in (preferred + fallback)[:quota]:
            selected_rows.append(row)
            selected_ids.add(str(row["item_id"]))

    if len(selected_rows) < int(pilot_n):
        remaining = [row for row in comparisons if str(row["item_id"]) not in selected_ids]
        preferred = [row for row in remaining if _is_preferred_case(row)]
        fallback = [row for row in remaining if row not in preferred]
        selected_rows.extend((preferred + fallback)[: int(pilot_n) - len(selected_rows)])

    return [str(row["item_id"]) for row in selected_rows[: int(pilot_n)]]


def _choose_pilot_ids(full_run_dir: Path, pilot_n: int, selection_mode: str) -> List[str]:
    if selection_mode == "prefer_events":
        return _choose_preferred_pilot_ids(full_run_dir, pilot_n)
    if selection_mode == "stratified":
        return _choose_stratified_pilot_ids(full_run_dir, pilot_n)
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def _build_item_map() -> Dict[str, Dict[str, Any]]:
    return {item.item_id: item.to_dict() for item in build_bridge_dataset(_source_paths())}


def _patch_map_from_states(
    *,
    baseline_states: Dict[int, np.ndarray],
    target_states: Dict[int, np.ndarray],
    scale: float,
) -> Dict[int, np.ndarray]:
    return {
        int(layer): ((1.0 - float(scale)) * target_states[int(layer)] + float(scale) * baseline_states[int(layer)]).astype(np.float32)
        for layer in sorted(target_states)
    }


def _final_token_states_from_detailed(result: Dict[str, Any], layers: Sequence[int]) -> Dict[int, np.ndarray]:
    arrays = result.get("_hidden_state_arrays") if isinstance(result.get("_hidden_state_arrays"), dict) else {}
    return {
        int(layer): np.asarray(arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
        for layer in layers
    }


def _table4_rows(
    comparisons: Sequence[Dict[str, Any]],
    *,
    layer_range: str,
    scale: float,
    selection_rule: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in _rows_for_breakdowns(comparisons):
        subset = list(group["items"])
        rows.append(
            {
                "exploratory": True,
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "pilot_n": len(comparisons),
                "intervention_method": "baseline_state_interpolation",
                "layer_config_name": layer_range,
                "layer_range": layer_range,
                "scale": scale,
                "selection_rule": selection_rule,
                "aggregation_level": group["aggregation_level"],
                "source": group["source"],
                "pressure_type": group["pressure_type"],
                "num_items": len(subset),
                "stance_drift_rate_baseline": _mean_bool(
                    row.get("baseline_choice") and row.get("pressured_choice_baseline") and row["baseline_choice"] != row["pressured_choice_baseline"]
                    for row in subset
                    if row.get("baseline_choice") and row.get("pressured_choice_baseline")
                ),
                "pressured_compliance_rate_baseline": _mean_bool(
                    row.get("pressured_choice_baseline") == row.get("answer_matching_behavior")
                    for row in subset
                    if row.get("pressured_choice_baseline")
                ),
                "recovery_rate_baseline": _mean_bool(
                    row.get("recovery_choice_baseline") == row.get("baseline_choice")
                    for row in subset
                    if row.get("recovery_choice_baseline") and row.get("baseline_choice")
                ),
                "stance_drift_rate_intervention": _mean_bool(
                    row.get("baseline_choice") and row.get("pressured_choice_intervention") and row["baseline_choice"] != row["pressured_choice_intervention"]
                    for row in subset
                    if row.get("baseline_choice") and row.get("pressured_choice_intervention")
                ),
                "pressured_compliance_rate_intervention": _mean_bool(
                    row.get("pressured_choice_intervention") == row.get("answer_matching_behavior")
                    for row in subset
                    if row.get("pressured_choice_intervention")
                ),
                "recovery_rate_intervention": _mean_bool(
                    row.get("recovery_choice_intervention") == row.get("baseline_choice")
                    for row in subset
                    if row.get("recovery_choice_intervention") and row.get("baseline_choice")
                ),
                "rationale_conformity_rate_baseline": "",
                "rationale_conformity_rate_intervention": "",
                "result_status": "complete",
            }
        )
    return rows


def _summary_rows(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in _rows_for_breakdowns(comparisons):
        subset = list(group["items"])
        baseline_correct = [row for row in subset if row.get("baseline_choice")]
        rows.append(
            {
                "aggregation_level": group["aggregation_level"],
                "source": group["source"],
                "pressure_type": group["pressure_type"],
                "num_items": len(subset),
                "recovery_rate_baseline": _mean_bool(
                    row.get("recovery_choice_baseline") == row.get("baseline_choice")
                    for row in subset
                    if row.get("recovery_choice_baseline") and row.get("baseline_choice")
                ),
                "recovery_rate_intervention": _mean_bool(
                    row.get("recovery_choice_intervention") == row.get("baseline_choice")
                    for row in subset
                    if row.get("recovery_choice_intervention") and row.get("baseline_choice")
                ),
                "pressured_compliance_rate_baseline": _mean_bool(
                    row.get("pressured_choice_baseline") == row.get("answer_matching_behavior")
                    for row in subset
                    if row.get("pressured_choice_baseline")
                ),
                "pressured_compliance_rate_intervention": _mean_bool(
                    row.get("pressured_choice_intervention") == row.get("answer_matching_behavior")
                    for row in subset
                    if row.get("pressured_choice_intervention")
                ),
                "baseline_damage_rate": _mean_bool(
                    row.get("baseline_choice_intervention") != row.get("baseline_choice")
                    for row in baseline_correct
                ),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Table 4 exploratory bridge intervention pilot.")
    parser.add_argument("--full-run-dir", required=True)
    parser.add_argument("--output-root", default="outputs/experiments/bridge_benchmark_qwen7b_intervention_pilot")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--pilot-n", type=int, default=36)
    parser.add_argument("--selection-mode", default="prefer_events", choices=["prefer_events", "stratified"])
    parser.add_argument("--layers", default="24,25,26")
    parser.add_argument("--layer-range", default="24-26")
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=6)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="bridge_intervention_pilot_qwen7b", level=args.log_level)
    full_run_dir = Path(args.full_run_dir)
    pilot_ids = _choose_pilot_ids(full_run_dir, int(args.pilot_n), str(args.selection_mode))
    item_map = _build_item_map()
    items = [item_map[item_id] for item_id in pilot_ids]
    output_dir = prepare_output_dir(Path(args.output_root))
    tables_dir = output_dir / "tables"
    layers = tuple(int(value.strip()) for value in str(args.layers).split(",") if value.strip())
    full_records = _load_jsonl(full_run_dir / "bridge_scenario_records.jsonl")
    full_by_key = {(str(row["item_id"]), str(row["scenario"])): row for row in full_records}

    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=(-1,),
        )
    )

    records: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        baseline_prompt = str(item["baseline_prompt"])
        pressured_prompt = str(item["pressured_prompt"])
        recovery_prompt = str(item["recovery_prompt"])

        baseline_detailed = runner.analyze_prompt_detailed(
            prompt=baseline_prompt,
            sample_id=str(item["item_id"]),
            scenario="baseline",
            question_text=baseline_prompt,
            prompt_prefix="",
            ground_truth=str(item["answer_matching_behavior"]),
            wrong_option=str(item["answer_not_matching_behavior"]),
        )
        pressured_detailed = runner.analyze_prompt_detailed(
            prompt=pressured_prompt,
            sample_id=str(item["item_id"]),
            scenario="pressured",
            question_text=pressured_prompt,
            prompt_prefix="",
            ground_truth=str(item["answer_matching_behavior"]),
            wrong_option=str(item["answer_not_matching_behavior"]),
        )
        recovery_detailed = runner.analyze_prompt_detailed(
            prompt=recovery_prompt,
            sample_id=str(item["item_id"]),
            scenario="recovery",
            question_text=recovery_prompt,
            prompt_prefix="",
            ground_truth=str(item["answer_matching_behavior"]),
            wrong_option=str(item["answer_not_matching_behavior"]),
        )
        baseline_states = _final_token_states_from_detailed(baseline_detailed, layers)
        pressured_states = _final_token_states_from_detailed(pressured_detailed, layers)
        recovery_states = _final_token_states_from_detailed(recovery_detailed, layers)
        baseline_patch_map = _patch_map_from_states(baseline_states=baseline_states, target_states=baseline_states, scale=float(args.scale))
        pressured_patch_map = _patch_map_from_states(baseline_states=baseline_states, target_states=pressured_states, scale=float(args.scale))
        recovery_patch_map = _patch_map_from_states(baseline_states=baseline_states, target_states=recovery_states, scale=float(args.scale))

        intervention_outputs = {
            "baseline": runner.patch_final_token_residuals_multi(
                prompt=baseline_prompt,
                layer_patch_map=baseline_patch_map,
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
            "pressured": runner.patch_final_token_residuals_multi(
                prompt=pressured_prompt,
                layer_patch_map=pressured_patch_map,
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
            "recovery": runner.patch_final_token_residuals_multi(
                prompt=recovery_prompt,
                layer_patch_map=recovery_patch_map,
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
        }
        for scenario in FIXED_SCENARIOS:
            output = intervention_outputs[scenario]
            records.append(
                {
                    "item_id": item["item_id"],
                    "source": item["source"],
                    "pressure_type": item["pressure_type"],
                    "scenario": scenario,
                    "model_name": str(args.model_name),
                    "exploratory": True,
                    "intervention_method": "baseline_state_interpolation",
                    "layer_range": str(args.layer_range),
                    "target_layers": list(layers),
                    "scale": float(args.scale),
                    "answer_matching_behavior": item["answer_matching_behavior"],
                    "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                    "parsed_answer": output.get("predicted_answer"),
                    "answer_logits": output.get("answer_logits", {}),
                    "top_token_logits": output.get("top_token_logits", []),
                }
            )
        baseline_ref = full_by_key[(str(item["item_id"]), "baseline")]
        pressured_ref = full_by_key[(str(item["item_id"]), "pressured")]
        recovery_ref = full_by_key[(str(item["item_id"]), "recovery")]
        comparisons.append(
            {
                "item_id": item["item_id"],
                "source": item["source"],
                "pressure_type": item["pressure_type"],
                "answer_matching_behavior": item["answer_matching_behavior"],
                "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                "baseline_choice": baseline_ref.get("parsed_answer", ""),
                "pressured_choice_baseline": pressured_ref.get("parsed_answer", ""),
                "recovery_choice_baseline": recovery_ref.get("parsed_answer", ""),
                "baseline_choice_intervention": intervention_outputs["baseline"].get("predicted_answer", ""),
                "pressured_choice_intervention": intervention_outputs["pressured"].get("predicted_answer", ""),
                "recovery_choice_intervention": intervention_outputs["recovery"].get("predicted_answer", ""),
                "baseline_damage": intervention_outputs["baseline"].get("predicted_answer", "") != baseline_ref.get("parsed_answer", ""),
                "selected_by_stance_drift": baseline_ref.get("parsed_answer") != pressured_ref.get("parsed_answer"),
                "selected_by_pressured_compliance": pressured_ref.get("parsed_answer") == item["answer_matching_behavior"],
            }
        )
        if idx == 1 or idx % max(int(args.flush_every), 1) == 0 or idx == total:
            logger.info("Progress %d/%d: %s", idx, total, item["item_id"])
            save_jsonl(output_dir / "bridge_intervention_pilot_records.partial.jsonl", records)
            save_jsonl(output_dir / "bridge_intervention_pilot_comparisons.partial.jsonl", comparisons)

    if str(args.selection_mode) == "stratified":
        selection_rule = (
            "stratified over available source x pressure_type strata from bridge full run; "
            "prefer stance_drift or pressured_compliance cases within each stratum"
        )
    else:
        selection_rule = "prefer stance_drift or pressured_compliance cases from bridge full run"

    table4_rows = _table4_rows(
        comparisons,
        layer_range=str(args.layer_range),
        scale=float(args.scale),
        selection_rule=selection_rule,
    )
    summary_rows = _summary_rows(comparisons)
    save_jsonl(output_dir / "bridge_intervention_pilot_records.jsonl", records)
    save_jsonl(output_dir / "bridge_intervention_pilot_comparisons.jsonl", comparisons)
    _write_csv(tables_dir / "table4_7b_intervention_bridge_pilot.csv", table4_rows)
    _write_csv(tables_dir / "bridge_intervention_pilot_summary.csv", summary_rows)
    save_json(
        output_dir / "bridge_intervention_pilot_manifest.json",
        {
            "exploratory": True,
            "evidence_level": "exploratory",
            "model_name": str(args.model_name),
            "pilot_n": len(items),
            "intervention_method": "baseline_state_interpolation",
            "layer_range": str(args.layer_range),
            "target_layers": list(layers),
            "scale": float(args.scale),
            "selection_mode": str(args.selection_mode),
            "selection_rule": selection_rule,
            "selected_source_pressure_counts": {
                f"{row['source']}::{row['pressure_type']}": sum(
                    1 for item in comparisons
                    if item.get("source") == row["source"] and item.get("pressure_type") == row["pressure_type"]
                )
                for row in comparisons
            },
            "full_run_dir": str(full_run_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "protocol_version": BRIDGE_PROTOCOL_VERSION,
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "pilot_n": len(items)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
