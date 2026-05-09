#!/usr/bin/env python3
"""Run authority-pressure behavioral closure with Qwen baseline-state interpolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_hidden_state_arrays, save_json, save_jsonl
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.pressure_subspace import mean_bool, parse_layers, read_jsonl, write_csv


def _patch_map_from_states(
    *,
    baseline_states: Dict[int, np.ndarray],
    target_states: Dict[int, np.ndarray],
    scale: float,
) -> Dict[int, np.ndarray]:
    return {
        int(layer): (
            (1.0 - float(scale)) * np.asarray(target_states[int(layer)], dtype=np.float32)
            + float(scale) * np.asarray(baseline_states[int(layer)], dtype=np.float32)
        ).astype(np.float32)
        for layer in sorted(target_states)
    }


def _final_token_states_from_detailed(result: Dict[str, Any], layers: Sequence[int]) -> Dict[int, np.ndarray]:
    arrays = result.get("_hidden_state_arrays") if isinstance(result.get("_hidden_state_arrays"), dict) else {}
    return {
        int(layer): np.asarray(arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
        for layer in layers
    }


def _record_row(item: Dict[str, Any], scenario: str, method: str, output: Dict[str, Any], layers: Sequence[int], beta: float) -> Dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "source": item["source"],
        "pressure_type": item["pressure_type"],
        "scenario": scenario,
        "method": method,
        "layers": list(layers),
        "beta": float(beta),
        "parsed_answer": output.get("predicted_answer", ""),
        "answer_logits": output.get("answer_logits", {}),
        "correct_wrong_margin": output.get("correct_wrong_margin"),
    }


def _comparison_row(item: Dict[str, Any], outputs_by_method: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    no_outputs = outputs_by_method["no_intervention"]
    patched_outputs = outputs_by_method["baseline_state_interpolation"]
    baseline_choice = no_outputs["baseline"].get("predicted_answer", "")
    rows: List[Dict[str, Any]] = []
    for method, outputs in outputs_by_method.items():
        rows.append(
            {
                "item_id": item["item_id"],
                "source": item["source"],
                "pressure_type": item["pressure_type"],
                "method": method,
                "answer_matching_behavior": item["answer_matching_behavior"],
                "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                "baseline_choice": outputs["baseline"].get("predicted_answer", ""),
                "pressured_choice": outputs["pressured"].get("predicted_answer", ""),
                "recovery_choice": outputs["recovery"].get("predicted_answer", ""),
                "stance_drift": bool(
                    outputs["baseline"].get("predicted_answer", "")
                    and outputs["pressured"].get("predicted_answer", "")
                    and outputs["baseline"].get("predicted_answer", "") != outputs["pressured"].get("predicted_answer", "")
                ),
                "pressured_compliance": bool(outputs["pressured"].get("predicted_answer", "") == item["answer_matching_behavior"]),
                "recovered": bool(
                    baseline_choice
                    and outputs["recovery"].get("predicted_answer", "")
                    and outputs["recovery"].get("predicted_answer", "") == baseline_choice
                ),
                "baseline_damage": bool(
                    method != "no_intervention"
                    and outputs["baseline"].get("predicted_answer", "")
                    and no_outputs["baseline"].get("predicted_answer", "")
                    and outputs["baseline"].get("predicted_answer", "") != no_outputs["baseline"].get("predicted_answer", "")
                ),
            }
        )
    return rows


def _summary_rows(comparisons: Sequence[Dict[str, Any]], beta: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in ("no_intervention", "baseline_state_interpolation"):
        subset = [row for row in comparisons if row["method"] == method]
        rows.append(
            {
                "beta": float(beta),
                "method": method,
                "num_items": len(subset),
                "stance_drift_rate": mean_bool(row["stance_drift"] for row in subset),
                "pressured_compliance_rate": mean_bool(row["pressured_compliance"] for row in subset),
                "recovery_rate": mean_bool(row["recovered"] for row in subset),
                "baseline_damage_rate": mean_bool(row["baseline_damage"] for row in subset),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run authority-pressure behavioral closure with baseline-state interpolation.")
    parser.add_argument("--sample-file", default="data/authority_pressure/authority_pressure_pairs.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layers", default="24,25,26")
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--output-root", default="outputs/experiments/authority_pressure_behavioral_closure/qwen7b")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="authority_pressure_behavioral_closure", level=args.log_level)
    items = read_jsonl(Path(args.sample_file))
    layers = parse_layers(str(args.layers))
    output_dir = prepare_output_dir(Path(args.output_root), run_name=str(args.model_name).replace("/", "_"))
    hidden_dir = output_dir / "hidden_state_arrays"
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
        prompts = {
            "baseline": str(item["baseline_prompt"]),
            "pressured": str(item["pressured_prompt"]),
            "recovery": str(item["recovery_prompt"]),
        }
        detailed: Dict[str, Dict[str, Any]] = {}
        scenario_states: Dict[str, Dict[int, np.ndarray]] = {}
        for scenario, prompt in prompts.items():
            result = runner.analyze_prompt_detailed(
                prompt=prompt,
                sample_id=str(item["item_id"]),
                scenario=scenario,
                question_text=prompt,
                prompt_prefix="",
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            )
            detailed[scenario] = result
            states = _final_token_states_from_detailed(result, layers)
            scenario_states[scenario] = states
            save_hidden_state_arrays(
                hidden_dir,
                str(item["item_id"]),
                scenario,
                {f"layer_{int(layer)}_final_token": np.asarray(state, dtype=np.float32) for layer, state in states.items()},
            )

        no_outputs = {
            scenario: {
                "predicted_answer": detailed[scenario].get("predicted_answer", ""),
                "answer_logits": detailed[scenario].get("answer_logits", {}),
                "correct_wrong_margin": detailed[scenario].get("correct_wrong_margin"),
            }
            for scenario in prompts
        }
        patched_outputs = {
            "baseline": runner.patch_final_token_residuals_multi(
                prompt=prompts["baseline"],
                layer_patch_map=_patch_map_from_states(
                    baseline_states=scenario_states["baseline"],
                    target_states=scenario_states["baseline"],
                    scale=float(args.beta),
                ),
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
            "pressured": runner.patch_final_token_residuals_multi(
                prompt=prompts["pressured"],
                layer_patch_map=_patch_map_from_states(
                    baseline_states=scenario_states["baseline"],
                    target_states=scenario_states["pressured"],
                    scale=float(args.beta),
                ),
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
            "recovery": runner.patch_final_token_residuals_multi(
                prompt=prompts["recovery"],
                layer_patch_map=_patch_map_from_states(
                    baseline_states=scenario_states["baseline"],
                    target_states=scenario_states["recovery"],
                    scale=float(args.beta),
                ),
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
            ),
        }
        for scenario, output in no_outputs.items():
            records.append(_record_row(item, scenario, "no_intervention", output, layers, float(args.beta)))
        for scenario, output in patched_outputs.items():
            records.append(_record_row(item, scenario, "baseline_state_interpolation", output, layers, float(args.beta)))
        comparisons.extend(
            _comparison_row(
                item,
                {
                    "no_intervention": no_outputs,
                    "baseline_state_interpolation": patched_outputs,
                },
            )
        )
        if idx == 1 or idx % max(int(args.flush_every), 1) == 0 or idx == total:
            logger.info("Progress %d/%d: %s", idx, total, item["item_id"])
            save_jsonl(output_dir / "belief_causal_records.partial.jsonl", records)
            save_jsonl(output_dir / "belief_causal_comparisons.partial.jsonl", comparisons)

    summary_rows = _summary_rows(comparisons, float(args.beta))
    save_jsonl(output_dir / "belief_causal_records.jsonl", records)
    save_jsonl(output_dir / "belief_causal_comparisons.jsonl", comparisons)
    write_csv(output_dir / "belief_causal_summary.csv", summary_rows)
    save_json(
        output_dir / "authority_pressure_behavioral_manifest.json",
        {
            "pipeline": "authority_pressure_behavioral_closure_v1",
            "sample_file": str(Path(args.sample_file).resolve()),
            "model_name": str(args.model_name),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layers": list(layers),
            "beta": float(args.beta),
            "num_items": len(items),
            "outputs": {
                "summary_csv": str((output_dir / "belief_causal_summary.csv").resolve()),
                "records_jsonl": str((output_dir / "belief_causal_records.jsonl").resolve()),
                "comparisons_jsonl": str((output_dir / "belief_causal_comparisons.jsonl").resolve()),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_items": len(items)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
