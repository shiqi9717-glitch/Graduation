#!/usr/bin/env python3
"""Run mechanistic analysis on local probe samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.io_utils import load_sample_file, prepare_output_dir, save_hidden_state_arrays, save_json, sanitize_filename
from src.open_model_probe.mechanistic import (
    build_sample_case_analysis,
    focused_subset,
    layer_summary_dataframe,
    save_markdown_report,
    transition_label,
    write_jsonl,
)
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.prompt_builder import build_prompt

RESEARCH_METADATA_FIELDS = (
    "sample_type",
    "condition_id",
    "authority_level",
    "confidence_level",
    "explicit_wrong_option",
    "is_control",
    "is_hard_negative",
    "subject",
    "category",
    "task_id",
)


def _parse_hidden_state_layers(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return (-1, -2, -3, -4)
    return tuple(int(item) for item in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mechanistic analysis for local Qwen white-box probing.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_mechanistic")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-state-layers", default="-1,-2,-3,-4")
    parser.add_argument("--patching-max-samples", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _attach_metadata(record: dict, sample: dict) -> None:
    for field in RESEARCH_METADATA_FIELDS:
        if field in sample:
            record[field] = sample.get(field)


def _save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_mechanistic", level=args.log_level)

    samples = load_sample_file(Path(args.sample_file))
    output_dir = prepare_output_dir(Path(args.output_dir), run_name=sanitize_filename(args.model_name))
    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=_parse_hidden_state_layers(args.hidden_state_layers),
        )
    )

    scenario_rows: list[dict] = []
    sample_rows: list[dict] = []
    patching_rows: list[dict] = []
    patched_sample_count = 0

    detail_dir = output_dir / "details"
    hidden_dir = output_dir / "layer_hidden_states"
    attention_dir = output_dir / "attention_arrays"

    for sample in samples:
        working_sample = dict(sample)
        scenario_records: dict[str, dict] = {}
        scenario_prompts: dict[str, str] = {}

        for scenario in ("baseline", "interference", "recheck"):
            prompt = build_prompt(working_sample, scenario)
            scenario_prompts[scenario] = prompt
            record = runner.analyze_prompt_detailed(
                prompt=prompt,
                sample_id=str(working_sample.get("sample_id") or "sample"),
                scenario=scenario,
                question_text=str(working_sample.get("question_text") or ""),
                prompt_prefix=str(working_sample.get("prompt_prefix") or ""),
                ground_truth=str(working_sample.get("ground_truth") or ""),
                wrong_option=str(working_sample.get("wrong_option") or ""),
            )
            _attach_metadata(record, working_sample)

            hidden_arrays = record.pop("_hidden_state_arrays", {}) or {}
            attention_arrays = record.pop("_attention_arrays", {}) or {}
            hidden_path = save_hidden_state_arrays(hidden_dir, record["sample_id"], scenario, hidden_arrays)
            attention_path = save_hidden_state_arrays(attention_dir, record["sample_id"], scenario, attention_arrays)
            detail_path = _save_json(
                detail_dir / f"{sanitize_filename(record['sample_id'])}__{scenario}_details.json",
                {
                    "layer_logit_lens": record.get("layer_logit_lens", []),
                    "attention_summary": record.get("attention_summary", []),
                },
            )
            record["layer_hidden_state_path"] = str(hidden_path.resolve())
            record["attention_array_path"] = str(attention_path.resolve())
            record["detail_json_path"] = str(detail_path.resolve())
            scenario_records[scenario] = {
                **record,
                "_hidden_state_arrays": hidden_arrays,
            }
            scenario_rows.append(record)

            if scenario == "interference":
                working_sample["interference_predicted_answer"] = record.get("predicted_answer")
                working_sample.setdefault("recheck_first_answer", record.get("predicted_answer"))

        sample_case = build_sample_case_analysis(scenario_records)
        sample_rows.append(sample_case)

        if (
            sample_case["transition_label"] == "baseline_correct_to_interference_wrong"
            and patched_sample_count < int(args.patching_max_samples)
        ):
            baseline_hidden_arrays = scenario_records["baseline"].get("_hidden_state_arrays", {}) or {}
            for key, patch_tensor in baseline_hidden_arrays.items():
                if not key.endswith("_final_token"):
                    continue
                layer_index = int(key.split("_")[1])
                patched = runner.patch_final_token_residual(
                    prompt=scenario_prompts["interference"],
                    patch_layer_index=layer_index,
                    patched_final_token=patch_tensor,
                    ground_truth=str(sample.get("ground_truth") or ""),
                    wrong_option=str(sample.get("wrong_option") or ""),
                )
                patching_rows.append(
                    {
                        "sample_id": sample_case["sample_id"],
                        "transition_label": sample_case["transition_label"],
                        "sample_type": sample_case.get("sample_type"),
                        "patch_layer_index": layer_index,
                        "baseline_answer": sample_case.get("baseline_answer"),
                        "interference_answer": sample_case.get("interference_answer"),
                        "patched_answer": patched.get("predicted_answer"),
                        "ground_truth": sample_case.get("ground_truth"),
                        "wrong_option": sample_case.get("wrong_option"),
                        "restored_baseline_answer": patched.get("predicted_answer") == sample_case.get("baseline_answer"),
                        "restored_correct": patched.get("predicted_answer") == sample_case.get("ground_truth"),
                        "patched_correct_option_logit": patched.get("correct_option_logit"),
                        "patched_wrong_option_logit": patched.get("wrong_option_logit"),
                        "patched_correct_wrong_margin": patched.get("correct_wrong_margin"),
                        "patched_margin_gain_vs_interference": float(
                            patched.get("correct_wrong_margin", 0.0)
                            - scenario_records["interference"].get("correct_wrong_margin", 0.0)
                        ),
                    }
                )
            patched_sample_count += 1

    layer_summary = layer_summary_dataframe(sample_rows)

    write_jsonl(output_dir / "mechanistic_scenario_records.jsonl", scenario_rows)
    write_jsonl(output_dir / "mechanistic_sample_cases.jsonl", sample_rows)
    write_jsonl(
        output_dir / "baseline_correct_to_interference_wrong.jsonl",
        focused_subset(sample_rows, "baseline_correct_to_interference_wrong"),
    )
    write_jsonl(
        output_dir / "baseline_wrong_to_interference_wrong.jsonl",
        focused_subset(sample_rows, "baseline_wrong_to_interference_wrong"),
    )
    write_jsonl(
        output_dir / "baseline_correct_to_interference_correct.jsonl",
        focused_subset(sample_rows, "baseline_correct_to_interference_correct"),
    )
    write_jsonl(output_dir / "patching_results.jsonl", patching_rows)

    if not layer_summary.empty:
        layer_summary.to_csv(output_dir / "layer_summary.csv", index=False)

    if patching_rows:
        (
            pd.DataFrame(patching_rows)
            .groupby(["transition_label", "patch_layer_index"], dropna=False)
            .agg(
                num_trials=("sample_id", "count"),
                restore_correct_rate=("restored_correct", "mean"),
                restore_baseline_rate=("restored_baseline_answer", "mean"),
                mean_margin_gain=("patched_margin_gain_vs_interference", "mean"),
            )
            .reset_index()
            .sort_values(["transition_label", "patch_layer_index"])
            .to_csv(output_dir / "patching_summary.csv", index=False)
        )

    save_markdown_report(
        output_dir / "mechanistic_report.md",
        sample_rows=sample_rows,
        layer_summary=layer_summary,
        patching_rows=patching_rows,
    )

    summary = {
        "model_name": str(args.model_name),
        "sample_file": str(Path(args.sample_file).resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_samples": len(sample_rows),
        "transition_counts": {
            label: int(sum(1 for row in sample_rows if row.get("transition_label") == label))
            for label in sorted({row.get("transition_label") for row in sample_rows})
        },
        "files": {
            "scenario_records": str((output_dir / "mechanistic_scenario_records.jsonl").resolve()),
            "sample_cases": str((output_dir / "mechanistic_sample_cases.jsonl").resolve()),
            "layer_summary": str((output_dir / "layer_summary.csv").resolve()) if (output_dir / "layer_summary.csv").exists() else None,
            "patching_results": str((output_dir / "patching_results.jsonl").resolve()),
            "report": str((output_dir / "mechanistic_report.md").resolve()),
        },
    }
    save_json(output_dir / "mechanistic_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
