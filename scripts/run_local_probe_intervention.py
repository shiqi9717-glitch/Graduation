#!/usr/bin/env python3
"""Run minimal mechanism-informed intervention prototypes on local probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.intervention import (
    InterventionSpec,
    build_layer_patch_map,
    build_mean_residual_subtraction_directions,
    format_layer_config_name,
    load_mechanistic_sample_cases,
    load_mechanistic_scenario_records,
    select_target_layers,
    summarize_intervention_records,
)
from src.open_model_probe.io_utils import load_sample_file, prepare_output_dir, save_json, save_jsonl, sanitize_filename
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.prompt_builder import build_prompt


def _parse_layers(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(int(item) for item in values)


def _parse_methods(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(values or ("late_layer_residual_subtraction", "baseline_state_interpolation"))


def _parse_sample_types(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(values or ("strict_positive", "high_pressure_wrong_option"))


def _parse_scales(raw: str, fallback: float) -> tuple[float, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return (float(fallback),)
    return tuple(float(item) for item in values)


def _parse_layer_configs(raw: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    configs: list[tuple[str, tuple[int, ...]]] = []
    for chunk in str(raw or "").split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" in piece:
            label, values_raw = piece.split("=", 1)
            values = _parse_layers(values_raw)
            config_name = label.strip()
        else:
            values = _parse_layers(piece)
            config_name = format_layer_config_name(values)
        if values:
            configs.append((config_name, values))
    return tuple(configs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal intervention runner for local probe models.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--reference-mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_intervention")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--methods", default="late_layer_residual_subtraction,baseline_state_interpolation")
    parser.add_argument("--sample-types", default="strict_positive,high_pressure_wrong_option")
    parser.add_argument("--direction-sample-types", default="strict_positive,high_pressure_wrong_option")
    parser.add_argument("--top-layers", type=int, default=3)
    parser.add_argument("--explicit-layers", default="")
    parser.add_argument("--layer-configs", default="")
    parser.add_argument("--subtraction-scale", type=float, default=0.5)
    parser.add_argument("--subtraction-scales", default="")
    parser.add_argument("--interpolation-scale", type=float, default=0.5)
    parser.add_argument("--interpolation-scales", default="")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _default_explicit_layers(model_name: str) -> tuple[int, ...]:
    model = str(model_name)
    if "7B" in model or "7b" in model:
        return (24, 25, 26, 27)
    if "3B" in model or "3b" in model:
        return (31, 32, 33, 34, 35)
    return ()


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_intervention", level=args.log_level)

    sample_types = _parse_sample_types(args.sample_types)
    direction_sample_types = _parse_sample_types(args.direction_sample_types)
    methods = _parse_methods(args.methods)
    mech_dir = Path(args.reference_mechanistic_run_dir)
    sample_cases = load_mechanistic_sample_cases(mech_dir)
    scenario_record_map = load_mechanistic_scenario_records(mech_dir)
    explicit_layers = _parse_layers(args.explicit_layers)
    layer_configs = _parse_layer_configs(args.layer_configs)
    if explicit_layers:
        layer_configs = ((format_layer_config_name(explicit_layers), explicit_layers),)
    if not layer_configs:
        default_layers = _default_explicit_layers(args.model_name)
        selected_layers = select_target_layers(
            layer_summary_csv=mech_dir / "layer_summary.csv",
            sample_types=direction_sample_types,
            top_k=int(args.top_layers),
            explicit_layers=default_layers,
        )
        layer_configs = ((format_layer_config_name(selected_layers), tuple(selected_layers)),)

    subtraction_scales = _parse_scales(args.subtraction_scales, args.subtraction_scale)
    interpolation_scales = _parse_scales(args.interpolation_scales, args.interpolation_scale)

    specs: list[InterventionSpec] = []
    for method in methods:
        scales = interpolation_scales if method == "baseline_state_interpolation" else subtraction_scales
        for layer_config_name, layer_config in layer_configs:
            for scale in scales:
                specs.append(
                    InterventionSpec(
                        method=method,
                        target_layers=tuple(layer_config),
                        layer_config_name=layer_config_name,
                        subtraction_scale=float(scale if method == "late_layer_residual_subtraction" else args.subtraction_scale),
                        interpolation_scale=float(scale if method == "baseline_state_interpolation" else args.interpolation_scale),
                        target_sample_types=tuple(direction_sample_types),
                    )
                )

    subtraction_directions = build_mean_residual_subtraction_directions(
        sample_cases=sample_cases,
        scenario_record_map=scenario_record_map,
        target_layers=sorted({layer for spec in specs for layer in spec.target_layers}),
        sample_types=direction_sample_types,
    )

    samples = [
        sample for sample in load_sample_file(Path(args.sample_file))
        if str(sample.get("sample_type")) in set(sample_types)
    ]
    if int(args.limit_samples) > 0:
        samples = samples[: int(args.limit_samples)]

    output_dir = prepare_output_dir(Path(args.output_dir), run_name=sanitize_filename(args.model_name))
    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=(-1, -2, -3, -4),
        )
    )

    records: list[dict] = []
    comparisons: list[dict] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        baseline_ref = scenario_record_map[(sample_id, "baseline")]
        interference_ref = scenario_record_map[(sample_id, "interference")]
        baseline_prompt = build_prompt(sample, "baseline")
        interference_prompt = build_prompt(sample, "interference")

        for spec in specs:
            baseline_patch_map = build_layer_patch_map(
                method=spec.method,
                sample_id=sample_id,
                scenario="baseline",
                target_layers=spec.target_layers,
                scenario_record_map=scenario_record_map,
                subtraction_directions=subtraction_directions,
                subtraction_scale=spec.subtraction_scale,
                interpolation_scale=spec.interpolation_scale,
            )
            interference_patch_map = build_layer_patch_map(
                method=spec.method,
                sample_id=sample_id,
                scenario="interference",
                target_layers=spec.target_layers,
                scenario_record_map=scenario_record_map,
                subtraction_directions=subtraction_directions,
                subtraction_scale=spec.subtraction_scale,
                interpolation_scale=spec.interpolation_scale,
            )

            baseline_intervened = runner.patch_final_token_residuals_multi(
                prompt=baseline_prompt,
                layer_patch_map=baseline_patch_map,
                ground_truth=str(sample.get("ground_truth") or ""),
                wrong_option=str(sample.get("wrong_option") or ""),
            )
            interference_intervened = runner.patch_final_token_residuals_multi(
                prompt=interference_prompt,
                layer_patch_map=interference_patch_map,
                ground_truth=str(sample.get("ground_truth") or ""),
                wrong_option=str(sample.get("wrong_option") or ""),
            )
            ground_truth = str(sample.get("ground_truth") or "").strip().upper()
            wrong_option = str(sample.get("wrong_option") or "").strip().upper()
            records.append(
                {
                    "sample_id": sample_id,
                    "model_name": str(args.model_name),
                    "sample_type": sample.get("sample_type"),
                    "condition_id": sample.get("condition_id"),
                    "method": spec.method,
                    "layer_config_name": spec.layer_config_name,
                    "target_layers": list(spec.target_layers),
                    "active_scale": spec.active_scale,
                    "subtraction_scale": spec.subtraction_scale,
                    "interpolation_scale": spec.interpolation_scale,
                    "ground_truth": ground_truth,
                    "wrong_option": wrong_option,
                    "baseline_reference_answer": baseline_ref.get("predicted_answer"),
                    "interference_reference_answer": interference_ref.get("predicted_answer"),
                    "baseline_intervened_answer": baseline_intervened.get("predicted_answer"),
                    "interference_intervened_answer": interference_intervened.get("predicted_answer"),
                    "baseline_reference_correct": baseline_ref.get("predicted_answer") == ground_truth,
                    "interference_reference_correct": interference_ref.get("predicted_answer") == ground_truth,
                    "baseline_intervened_correct": baseline_intervened.get("predicted_answer") == ground_truth,
                    "interference_intervened_correct": interference_intervened.get("predicted_answer") == ground_truth,
                    "interference_reference_wrong_option_follow": interference_ref.get("predicted_answer") == wrong_option,
                    "interference_intervened_wrong_option_follow": interference_intervened.get("predicted_answer") == wrong_option,
                    "baseline_reference_margin": baseline_ref.get("correct_wrong_margin"),
                    "interference_reference_margin": interference_ref.get("correct_wrong_margin"),
                    "baseline_intervened_margin": baseline_intervened.get("correct_wrong_margin"),
                    "interference_intervened_margin": interference_intervened.get("correct_wrong_margin"),
                }
            )
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "model_name": str(args.model_name),
                    "sample_type": sample.get("sample_type"),
                    "condition_id": sample.get("condition_id"),
                    "method": spec.method,
                    "layer_config_name": spec.layer_config_name,
                    "target_layers": list(spec.target_layers),
                    "active_scale": spec.active_scale,
                    "baseline_reference_correct": baseline_ref.get("predicted_answer") == ground_truth,
                    "interference_reference_correct": interference_ref.get("predicted_answer") == ground_truth,
                    "baseline_intervened_correct": baseline_intervened.get("predicted_answer") == ground_truth,
                    "interference_intervened_correct": interference_intervened.get("predicted_answer") == ground_truth,
                    "interference_induced_error_reference": (
                        baseline_ref.get("predicted_answer") == ground_truth and interference_ref.get("predicted_answer") != ground_truth
                    ),
                    "intervention_recovers_error": (
                        baseline_ref.get("predicted_answer") == ground_truth
                        and interference_ref.get("predicted_answer") != ground_truth
                        and interference_intervened.get("predicted_answer") == ground_truth
                    ),
                    "baseline_damage": (
                        baseline_ref.get("predicted_answer") == ground_truth
                        and baseline_intervened.get("predicted_answer") != ground_truth
                    ),
                    "wrong_option_follow_reference": interference_ref.get("predicted_answer") == wrong_option,
                    "wrong_option_follow_intervened": interference_intervened.get("predicted_answer") == wrong_option,
                    "baseline_reference_margin": baseline_ref.get("correct_wrong_margin"),
                    "interference_reference_margin": interference_ref.get("correct_wrong_margin"),
                    "baseline_intervened_margin": baseline_intervened.get("correct_wrong_margin"),
                    "interference_intervened_margin": interference_intervened.get("correct_wrong_margin"),
                }
            )

    intervention_summary = summarize_intervention_records(records)
    run_payload = {
        "model_name": str(args.model_name),
        "sample_file": str(Path(args.sample_file).resolve()),
        "reference_mechanistic_run_dir": str(mech_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_samples": len(samples),
        "sample_types": list(sample_types),
        "direction_sample_types": list(direction_sample_types),
        "methods": [spec.method for spec in specs],
        "tested_settings": [
            {
                "method": spec.method,
                "layer_config_name": spec.layer_config_name,
                "target_layers": list(spec.target_layers),
                "active_scale": spec.active_scale,
                "subtraction_scale": spec.subtraction_scale,
                "interpolation_scale": spec.interpolation_scale,
            }
            for spec in specs
        ],
        "subtraction_direction_layers": sorted(int(layer) for layer in subtraction_directions),
        "intervention_summary": intervention_summary,
    }

    save_jsonl(output_dir / "intervention_records.jsonl", records)
    save_jsonl(output_dir / "intervention_comparisons.jsonl", comparisons)
    save_json(output_dir / "intervention_run.json", run_payload)
    save_json(output_dir / "intervention_summary.json", intervention_summary)
    print(json.dumps(run_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
