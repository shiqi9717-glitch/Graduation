#!/usr/bin/env python3
"""Run exploratory component-level intervention comparisons."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_bridge_dataset
from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_json, save_jsonl
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.pressure_subspace import mean_bool, write_csv

import scripts.run_belief_causal_transfer as belief_transfer


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _parse_layers(raw: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return tuple(values)


def _component_patch_map(
    *,
    baseline_states: Dict[int, np.ndarray],
    target_states: Dict[int, np.ndarray],
    beta: float,
) -> Dict[int, np.ndarray]:
    return {
        int(layer): (
            (1.0 - float(beta)) * np.asarray(target_states[int(layer)], dtype=np.float32)
            + float(beta) * np.asarray(baseline_states[int(layer)], dtype=np.float32)
        ).astype(np.float32)
        for layer in sorted(target_states)
    }


def _summary_rows(comparisons: Sequence[Dict[str, Any]], beta: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for component in ("no_intervention", "full_residual", "attention_only", "mlp_only"):
        subset = [row for row in comparisons if row["component"] == component]
        rows.append(
            {
                "beta": float(beta),
                "component": component,
                "num_items": len(subset),
                "stance_drift_rate": mean_bool(row["stance_drift"] for row in subset),
                "pressured_compliance_rate": mean_bool(row["pressured_compliance"] for row in subset),
                "recovery_rate": mean_bool(row["recovered"] for row in subset),
                "baseline_damage_rate": mean_bool(row["baseline_damage"] for row in subset),
            }
        )
    return rows


def _delta_rows(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    no_row = next((row for row in summary_rows if row["component"] == "no_intervention"), None)
    if no_row is None:
        return []
    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        if row["component"] == "no_intervention":
            continue
        drift_delta = float(row.get("stance_drift_rate") or 0.0) - float(no_row.get("stance_drift_rate") or 0.0)
        compliance_delta = float(row.get("pressured_compliance_rate") or 0.0) - float(no_row.get("pressured_compliance_rate") or 0.0)
        recovery_delta = float(row.get("recovery_rate") or 0.0) - float(no_row.get("recovery_rate") or 0.0)
        damage = float(row.get("baseline_damage_rate") or 0.0)
        out.append(
            {
                "component": row["component"],
                "beta": row["beta"],
                "num_items": row["num_items"],
                "drift_delta": drift_delta,
                "compliance_delta": compliance_delta,
                "recovery_delta": recovery_delta,
                "baseline_damage_rate": damage,
                "clean_control_score": (-(drift_delta + compliance_delta) + recovery_delta - 10.0 * damage),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exploratory component-level intervention comparisons.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--eval-source", default="nlp_survey")
    parser.add_argument("--pressure-type", default="belief_argument")
    parser.add_argument("--prompt-variant", default="original", choices=["original", "english", "en", "chinese", "zh", "zh_instruction"])
    parser.add_argument("--eval-n", type=int, default=24)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=6)
    parser.add_argument("--output-root", default="outputs/experiments/component_level_intervention")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="component_level_intervention", level=args.log_level)

    layers = _parse_layers(str(args.layers))
    all_items = [item.to_dict() for item in build_bridge_dataset(_source_paths())]
    eval_pairs = belief_transfer.base._sample_source(
        items=all_items,
        source=str(args.eval_source),
        n=int(args.eval_n),
        split="eval",
        seed=int(args.seed),
    )
    if str(args.pressure_type) != "belief_argument":
        eval_pairs = [row for row in eval_pairs if row.get("pressure_type") == str(args.pressure_type)]
    eval_pairs = belief_transfer._apply_prompt_variant(eval_pairs, str(args.prompt_variant))

    output_dir = prepare_output_dir(Path(args.output_root), run_name=str(args.model_name).replace("/", "_"))
    save_jsonl(output_dir / "pressure_pairs_eval.jsonl", eval_pairs)
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
    total = len(eval_pairs)
    for idx, item in enumerate(eval_pairs, start=1):
        prompts = {
            "baseline": str(item["baseline_prompt"]),
            "pressured": str(item["pressured_prompt"]),
            "recovery": str(item["recovery_prompt"]),
        }
        residual_states: Dict[str, Dict[int, np.ndarray]] = {}
        attention_states: Dict[str, Dict[int, np.ndarray]] = {}
        mlp_states: Dict[str, Dict[int, np.ndarray]] = {}
        no_outputs: Dict[str, Dict[str, Any]] = {}
        for scenario, prompt in prompts.items():
            result = runner.analyze_prompt_selected_layers(
                prompt=prompt,
                sample_id=str(item["item_id"]),
                scenario=scenario,
                question_text=prompt,
                prompt_prefix="",
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
                selected_layers=layers,
            )
            residual_states[scenario] = {
                int(layer): np.asarray(result["_hidden_state_arrays"][f"layer_{int(layer)}_final_token"], dtype=np.float32)
                for layer in layers
            }
            attention_states[scenario] = runner.capture_attention_output(prompt=prompt, target_layers=layers)
            mlp_states[scenario] = runner.capture_mlp_output(prompt=prompt, target_layers=layers)
            no_outputs[scenario] = {
                "predicted_answer": result.get("predicted_answer", ""),
                "answer_logits": result.get("answer_logits", {}),
                "correct_wrong_margin": result.get("correct_wrong_margin"),
                "top_token_logits": result.get("top_token_logits", []),
            }

        component_outputs = {
            "no_intervention": no_outputs,
            "full_residual": {
                "baseline": runner.patch_final_token_residuals_multi(
                    prompt=prompts["baseline"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=residual_states["baseline"],
                        target_states=residual_states["baseline"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "pressured": runner.patch_final_token_residuals_multi(
                    prompt=prompts["pressured"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=residual_states["baseline"],
                        target_states=residual_states["pressured"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "recovery": runner.patch_final_token_residuals_multi(
                    prompt=prompts["recovery"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=residual_states["baseline"],
                        target_states=residual_states["recovery"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
            },
            "attention_only": {
                "baseline": runner.patch_attention_output_multi(
                    prompt=prompts["baseline"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=attention_states["baseline"],
                        target_states=attention_states["baseline"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "pressured": runner.patch_attention_output_multi(
                    prompt=prompts["pressured"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=attention_states["baseline"],
                        target_states=attention_states["pressured"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "recovery": runner.patch_attention_output_multi(
                    prompt=prompts["recovery"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=attention_states["baseline"],
                        target_states=attention_states["recovery"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
            },
            "mlp_only": {
                "baseline": runner.patch_mlp_output_multi(
                    prompt=prompts["baseline"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=mlp_states["baseline"],
                        target_states=mlp_states["baseline"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "pressured": runner.patch_mlp_output_multi(
                    prompt=prompts["pressured"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=mlp_states["baseline"],
                        target_states=mlp_states["pressured"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
                "recovery": runner.patch_mlp_output_multi(
                    prompt=prompts["recovery"],
                    layer_patch_map=_component_patch_map(
                        baseline_states=mlp_states["baseline"],
                        target_states=mlp_states["recovery"],
                        beta=float(args.beta),
                    ),
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                ),
            },
        }

        baseline_choice_ref = no_outputs["baseline"].get("predicted_answer", "")
        for component, outputs in component_outputs.items():
            for scenario, output in outputs.items():
                records.append(
                    {
                        "item_id": item["item_id"],
                        "source": item["source"],
                        "pressure_type": item["pressure_type"],
                        "component": component,
                        "scenario": scenario,
                        "beta": float(args.beta),
                        "layers": list(layers),
                        "parsed_answer": output.get("predicted_answer", ""),
                        "answer_logits": output.get("answer_logits", {}),
                        "correct_wrong_margin": output.get("correct_wrong_margin"),
                        "top_token_logits": output.get("top_token_logits", []),
                    }
                )
            comparisons.append(
                {
                    "item_id": item["item_id"],
                    "source": item["source"],
                    "pressure_type": item["pressure_type"],
                    "component": component,
                    "beta": float(args.beta),
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
                        baseline_choice_ref
                        and outputs["recovery"].get("predicted_answer", "")
                        and outputs["recovery"].get("predicted_answer", "") == baseline_choice_ref
                    ),
                    "baseline_damage": bool(
                        component != "no_intervention"
                        and outputs["baseline"].get("predicted_answer", "")
                        and baseline_choice_ref
                        and outputs["baseline"].get("predicted_answer", "") != baseline_choice_ref
                    ),
                }
            )
        if idx == 1 or idx % max(int(args.flush_every), 1) == 0 or idx == total:
            logger.info("Progress %d/%d: %s", idx, total, item["item_id"])
            save_jsonl(output_dir / "component_level_records.partial.jsonl", records)
            save_jsonl(output_dir / "component_level_comparisons.partial.jsonl", comparisons)

    summary_rows = _summary_rows(comparisons, float(args.beta))
    delta_rows = _delta_rows(summary_rows)
    save_jsonl(output_dir / "component_level_records.jsonl", records)
    save_jsonl(output_dir / "component_level_comparisons.jsonl", comparisons)
    write_csv(output_dir / "component_level_summary.csv", summary_rows)
    write_csv(output_dir / "component_level_delta_summary.csv", delta_rows)
    save_json(
        output_dir / "component_level_manifest.json",
        {
            "pipeline": "component_level_intervention_v1",
            "model_name": str(args.model_name),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "eval_source": str(args.eval_source),
            "pressure_type": str(args.pressure_type),
            "prompt_variant": str(args.prompt_variant),
            "eval_n": len(eval_pairs),
            "layers": list(layers),
            "beta": float(args.beta),
            "outputs": {
                "summary_csv": str((output_dir / "component_level_summary.csv").resolve()),
                "delta_summary_csv": str((output_dir / "component_level_delta_summary.csv").resolve()),
                "records_jsonl": str((output_dir / "component_level_records.jsonl").resolve()),
                "comparisons_jsonl": str((output_dir / "component_level_comparisons.jsonl").resolve()),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_items": len(eval_pairs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
