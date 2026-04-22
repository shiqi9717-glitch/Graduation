#!/usr/bin/env python3
"""Late-layer mechanistic analysis for local Qwen probing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.io_utils import load_sample_file, prepare_output_dir, save_json, sanitize_filename
from src.open_model_probe.mechanistic import (
    build_anchor_head_expansion_sets,
    build_drop_wrong_option_case_rows,
    select_anchor_expansion_heads,
    summarize_drop_wrong_option_cases,
)
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.prompt_builder import build_prompt


def _parse_layers(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join([header, sep, *body])


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def _ablate_prefix(prefix: str, ablation: str) -> str:
    text = str(prefix or "")
    if ablation == "none":
        return text
    lines = [line for line in text.splitlines() if line.strip()]
    if ablation == "drop_authority":
        lines = [line for line in lines if ("专家" not in line and "接触过一些相关内容" not in line)]
    elif ablation == "drop_confidence":
        patterns = [
            r"我对这道题有比较强的把握，?",
            r"我对这道题有比较明确的判断，?",
            r"不过这道题我也不能完全确定，只是有一个初步判断，?",
            r"不过我也不太确定，只是有一个初步判断，?",
            r"我只是随便说一下自己的第一反应，但并不确定，?",
        ]
        updated = []
        for line in lines:
            for pattern in patterns:
                line = re.sub(pattern, "", line)
            updated.append(line.strip())
        lines = [line for line in updated if line]
    elif ablation == "drop_wrong_option":
        patterns = [
            r"我认为应该选\s*[A-D]。",
            r"我第一反应有点像是\s*[A-D]。",
        ]
        updated = []
        for line in lines:
            for pattern in patterns:
                line = re.sub(pattern, "", line)
            updated.append(line.strip(" ，。"))
        lines = [line for line in updated if line]
    elif ablation == "drop_instruction":
        lines = [line for line in lines if "请你独立判断后作答" not in line]
    else:
        raise ValueError(f"Unsupported ablation: {ablation}")
    return "\n".join(lines) + ("\n\n" if lines else "")


def _late_layer_margin_delta(record: Dict[str, Any], baseline_record: Dict[str, Any], late_layers: Iterable[int]) -> float | None:
    baseline_map = {int(row["layer_index"]): row for row in baseline_record.get("layer_logit_lens", [])}
    record_map = {int(row["layer_index"]): row for row in record.get("layer_logit_lens", [])}
    correct = str(record.get("ground_truth") or "").strip().upper()
    wrong = str(record.get("wrong_option") or "").strip().upper()
    values = []
    for layer_index in late_layers:
        base_row = baseline_map.get(int(layer_index))
        cur_row = record_map.get(int(layer_index))
        if not base_row or not cur_row:
            continue
        base_logits = base_row.get("answer_logits", {})
        cur_logits = cur_row.get("answer_logits", {})
        base_margin = float(base_logits.get(correct, 0.0) - base_logits.get(wrong, 0.0))
        cur_margin = float(cur_logits.get(correct, 0.0) - cur_logits.get(wrong, 0.0))
        values.append(cur_margin - base_margin)
    if not values:
        return None
    return float(np.mean(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Late-layer head/token/prefix analysis for local probe.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_late_layer")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--late-layers", default="27,28,29,30,31,32,33,34,35")
    parser.add_argument("--head-focus-layers", default="30,31,32,33,34,35")
    parser.add_argument("--expansion-focus-layers", default="31,32,34")
    parser.add_argument("--anchor-head", default="31:7")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_late_layer", level=args.log_level)

    late_layers = _parse_layers(args.late_layers)
    head_focus_layers = _parse_layers(args.head_focus_layers)
    expansion_focus_layers = _parse_layers(args.expansion_focus_layers)
    anchor_head = tuple(int(item) for item in str(args.anchor_head).split(":", maxsplit=1))
    if len(anchor_head) != 2:
        raise ValueError("--anchor-head must be formatted like 31:7")
    samples = load_sample_file(Path(args.sample_file))
    sample_map = {str(sample["sample_id"]): dict(sample) for sample in samples}

    mech_dir = Path(args.mechanistic_run_dir)
    focus_rows = _load_jsonl(mech_dir / "baseline_correct_to_interference_wrong.jsonl")
    focus_sample_ids = [str(row["sample_id"]) for row in focus_rows]

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

    head_rows: List[Dict[str, Any]] = []
    expansion_rows: List[Dict[str, Any]] = []
    wrong_token_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    sample_notes: List[Dict[str, Any]] = []

    ablations = ["none", "drop_authority", "drop_confidence", "drop_wrong_option", "drop_instruction"]

    for sample_id in focus_sample_ids:
        sample = dict(sample_map[sample_id])
        baseline_prompt = build_prompt(sample, "baseline")
        interference_prompt = build_prompt(sample, "interference")
        baseline_record = runner.analyze_prompt_detailed(
            prompt=baseline_prompt,
            sample_id=sample_id,
            scenario="baseline",
            question_text=str(sample.get("question_text") or ""),
            prompt_prefix=str(sample.get("prompt_prefix") or ""),
            ground_truth=str(sample.get("ground_truth") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )
        interference_record = runner.analyze_prompt_detailed(
            prompt=interference_prompt,
            sample_id=sample_id,
            scenario="interference",
            question_text=str(sample.get("question_text") or ""),
            prompt_prefix=str(sample.get("prompt_prefix") or ""),
            ground_truth=str(sample.get("ground_truth") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )

        sample_notes.append(
            {
                "sample_id": sample_id,
                "sample_type": sample.get("sample_type"),
                "condition_id": sample.get("condition_id"),
                "baseline_answer": baseline_record.get("predicted_answer"),
                "interference_answer": interference_record.get("predicted_answer"),
                "ground_truth": sample.get("ground_truth"),
                "wrong_option": sample.get("wrong_option"),
            }
        )

        baseline_head_outputs = runner.capture_attention_head_outputs(
            prompt=baseline_prompt,
            target_layers=head_focus_layers,
        )
        for layer_index in head_focus_layers:
            if layer_index not in baseline_head_outputs:
                continue
            layout = runner.attention_head_layout(layer_index)
            num_heads = int(layout["num_heads"])
            head_dim = int(layout["head_dim"])
            layer_tensor = baseline_head_outputs[layer_index]
            final_token_hidden = layer_tensor[-1]
            for head_index in range(num_heads):
                start = head_index * head_dim
                end = start + head_dim
                patched = runner.patch_attention_head_output(
                    prompt=interference_prompt,
                    patch_layer_index=layer_index,
                    head_index=head_index,
                    patched_head_vector=final_token_hidden[start:end],
                    patch_token_position=-1,
                    ground_truth=str(sample.get("ground_truth") or ""),
                    wrong_option=str(sample.get("wrong_option") or ""),
                )
                head_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_type": sample.get("sample_type"),
                        "condition_id": sample.get("condition_id"),
                        "patch_layer_index": int(layer_index),
                        "head_index": int(head_index),
                        "baseline_answer": baseline_record.get("predicted_answer"),
                        "interference_answer": interference_record.get("predicted_answer"),
                        "patched_answer": patched.get("predicted_answer"),
                        "ground_truth": sample.get("ground_truth"),
                        "wrong_option": sample.get("wrong_option"),
                        "restored_correct": patched.get("predicted_answer") == sample.get("ground_truth"),
                        "mean_margin_gain": float(
                            patched.get("correct_wrong_margin", 0.0)
                            - interference_record.get("correct_wrong_margin", 0.0)
                        ),
                    }
                )

        baseline_wrong_positions = runner.locate_wrong_option_question_positions(
            prompt=baseline_prompt,
            question_text=str(sample.get("question_text") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )
        interference_wrong_positions = runner.locate_wrong_option_question_positions(
            prompt=interference_prompt,
            question_text=str(sample.get("question_text") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )
        if baseline_wrong_positions and interference_wrong_positions:
            baseline_vectors = runner.capture_layer_token_residuals(
                prompt=baseline_prompt,
                target_layers=head_focus_layers,
                token_positions=baseline_wrong_positions,
            )
            for layer_index in head_focus_layers:
                base_positions = sorted(baseline_vectors.get(layer_index, {}).keys())
                int_positions = sorted(interference_wrong_positions)
                patch_positions = {}
                for base_pos, int_pos in zip(base_positions, int_positions):
                    patch_positions[int_pos] = baseline_vectors[layer_index][base_pos]
                if not patch_positions:
                    continue
                patched = runner.patch_residual_positions(
                    prompt=interference_prompt,
                    patch_layer_index=layer_index,
                    patch_positions=patch_positions,
                    ground_truth=str(sample.get("ground_truth") or ""),
                    wrong_option=str(sample.get("wrong_option") or ""),
                )
                final_patch_margin_gain = float(
                    baseline_record.get("correct_wrong_margin", 0.0)
                    - interference_record.get("correct_wrong_margin", 0.0)
                )
                wrong_token_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_type": sample.get("sample_type"),
                        "condition_id": sample.get("condition_id"),
                        "patch_layer_index": int(layer_index),
                        "num_wrong_option_positions": len(patch_positions),
                        "patched_answer": patched.get("predicted_answer"),
                        "ground_truth": sample.get("ground_truth"),
                        "wrong_option": sample.get("wrong_option"),
                        "restored_correct": patched.get("predicted_answer") == sample.get("ground_truth"),
                        "mean_margin_gain": float(
                            patched.get("correct_wrong_margin", 0.0)
                            - interference_record.get("correct_wrong_margin", 0.0)
                        ),
                        "full_final_token_patch_margin_gap": final_patch_margin_gain,
                    }
                )

        for ablation in ablations:
            ablated_sample = dict(sample)
            ablated_sample["prompt_prefix"] = _ablate_prefix(str(sample.get("prompt_prefix") or ""), ablation)
            ablated_prompt = build_prompt(ablated_sample, "interference")
            ablated_record = runner.analyze_prompt_detailed(
                prompt=ablated_prompt,
                sample_id=sample_id,
                scenario=f"interference_{ablation}",
                question_text=str(sample.get("question_text") or ""),
                prompt_prefix=str(ablated_sample.get("prompt_prefix") or ""),
                ground_truth=str(sample.get("ground_truth") or ""),
                wrong_option=str(sample.get("wrong_option") or ""),
            )
            ablation_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_type": sample.get("sample_type"),
                    "condition_id": sample.get("condition_id"),
                    "ablation": ablation,
                    "predicted_answer": ablated_record.get("predicted_answer"),
                    "is_correct": ablated_record.get("predicted_answer") == sample.get("ground_truth"),
                    "wrong_option_follow": ablated_record.get("predicted_answer") == sample.get("wrong_option"),
                    "late_layer_margin_delta": _late_layer_margin_delta(ablated_record, baseline_record, late_layers),
                }
            )

    head_df = pd.DataFrame(head_rows)
    wrong_df = pd.DataFrame(wrong_token_rows)
    ablation_df = pd.DataFrame(ablation_rows)

    if head_df.empty:
        head_summary = pd.DataFrame(columns=["patch_layer_index", "head_index", "num_trials", "restore_correct_rate", "mean_margin_gain"])
    else:
        head_summary = (
            head_df.groupby(["patch_layer_index", "head_index"], dropna=False)
            .agg(
                num_trials=("sample_id", "count"),
                restore_correct_rate=("restored_correct", "mean"),
                mean_margin_gain=("mean_margin_gain", "mean"),
            )
            .reset_index()
            .sort_values(["restore_correct_rate", "mean_margin_gain"], ascending=[False, False])
            .reset_index(drop=True)
        )
    if wrong_df.empty:
        wrong_summary = pd.DataFrame(columns=["patch_layer_index", "num_trials", "restore_correct_rate", "mean_margin_gain", "mean_final_patch_gap"])
    else:
        wrong_summary = (
            wrong_df.groupby(["patch_layer_index"], dropna=False)
            .agg(
                num_trials=("sample_id", "count"),
                restore_correct_rate=("restored_correct", "mean"),
                mean_margin_gain=("mean_margin_gain", "mean"),
                mean_final_patch_gap=("full_final_token_patch_margin_gap", "mean"),
            )
            .reset_index()
            .sort_values(["restore_correct_rate", "mean_margin_gain"], ascending=[False, False])
            .reset_index(drop=True)
        )
    ablation_summary = (
        ablation_df.groupby("ablation", dropna=False)
        .agg(
            num_samples=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            wrong_option_follow_rate=("wrong_option_follow", "mean"),
            mean_late_layer_margin_delta=("late_layer_margin_delta", "mean"),
        )
        .reset_index()
            .sort_values("accuracy", ascending=False)
            .reset_index(drop=True)
    )

    ordered_heads = select_anchor_expansion_heads(
        head_summary,
        anchor_head=(int(anchor_head[0]), int(anchor_head[1])),
        allowed_layers=expansion_focus_layers,
        max_heads=8,
    )
    expansion_sets = build_anchor_head_expansion_sets(ordered_heads, target_sizes=(1, 2, 4, 8))

    for sample_id in focus_sample_ids:
        sample = dict(sample_map[sample_id])
        baseline_prompt = build_prompt(sample, "baseline")
        interference_prompt = build_prompt(sample, "interference")
        baseline_record = runner.analyze_prompt_detailed(
            prompt=baseline_prompt,
            sample_id=sample_id,
            scenario="baseline",
            question_text=str(sample.get("question_text") or ""),
            prompt_prefix=str(sample.get("prompt_prefix") or ""),
            ground_truth=str(sample.get("ground_truth") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )
        interference_record = runner.analyze_prompt_detailed(
            prompt=interference_prompt,
            sample_id=sample_id,
            scenario="interference",
            question_text=str(sample.get("question_text") or ""),
            prompt_prefix=str(sample.get("prompt_prefix") or ""),
            ground_truth=str(sample.get("ground_truth") or ""),
            wrong_option=str(sample.get("wrong_option") or ""),
        )
        baseline_head_outputs = runner.capture_attention_head_outputs(
            prompt=baseline_prompt,
            target_layers=sorted({int(layer) for layer, _head in ordered_heads}),
        )
        spec_map: Dict[tuple[int, int], Dict[str, Any]] = {}
        for layer_index, head_index in ordered_heads:
            if layer_index not in baseline_head_outputs:
                continue
            layout = runner.attention_head_layout(layer_index)
            num_heads = int(layout["num_heads"])
            head_dim = int(layout["head_dim"])
            if head_index >= num_heads:
                continue
            final_token_hidden = baseline_head_outputs[layer_index][-1]
            start = head_index * head_dim
            end = start + head_dim
            spec_map[(layer_index, head_index)] = {
                "patch_layer_index": int(layer_index),
                "head_index": int(head_index),
                "patch_token_position": -1,
                "patched_head_vector": final_token_hidden[start:end],
            }
        for expansion in expansion_sets:
            head_set = [tuple(item) for item in expansion["head_set"]]
            patch_specs = [spec_map[head] for head in head_set if head in spec_map]
            if len(patch_specs) != len(head_set):
                continue
            patched = runner.patch_attention_head_outputs_multi(
                prompt=interference_prompt,
                patch_specs=patch_specs,
                ground_truth=str(sample.get("ground_truth") or ""),
                wrong_option=str(sample.get("wrong_option") or ""),
            )
            expansion_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_type": sample.get("sample_type"),
                    "condition_id": sample.get("condition_id"),
                    "expansion_size": int(expansion["expansion_size"]),
                    "head_set_label": str(expansion["head_set_label"]),
                    "patched_answer": patched.get("predicted_answer"),
                    "ground_truth": sample.get("ground_truth"),
                    "wrong_option": sample.get("wrong_option"),
                    "restored_correct": patched.get("predicted_answer") == sample.get("ground_truth"),
                    "restored_baseline": patched.get("predicted_answer") == baseline_record.get("predicted_answer"),
                    "mean_margin_gain": float(
                        patched.get("correct_wrong_margin", 0.0)
                        - interference_record.get("correct_wrong_margin", 0.0)
                    ),
                }
            )

    expansion_df = pd.DataFrame(expansion_rows)
    if expansion_df.empty:
        expansion_summary = pd.DataFrame(
            columns=["expansion_size", "head_set_label", "num_trials", "restore_correct_rate", "restore_baseline_rate", "mean_margin_gain"]
        )
    else:
        expansion_summary = (
            expansion_df.groupby(["expansion_size", "head_set_label"], dropna=False)
            .agg(
                num_trials=("sample_id", "count"),
                restore_correct_rate=("restored_correct", "mean"),
                restore_baseline_rate=("restored_baseline", "mean"),
                mean_margin_gain=("mean_margin_gain", "mean"),
            )
            .reset_index()
            .sort_values(["expansion_size"])
            .reset_index(drop=True)
        )

    case_df = build_drop_wrong_option_case_rows(
        sample_notes,
        ablation_df=ablation_df,
        head_df=head_df,
        expansion_df=expansion_df,
    )
    case_summary = summarize_drop_wrong_option_cases(case_df)

    head_df.to_csv(output_dir / "head_patch_details.csv", index=False)
    expansion_df.to_csv(output_dir / "anchored_head_expansion_details.csv", index=False)
    wrong_df.to_csv(output_dir / "wrong_option_token_patch_details.csv", index=False)
    ablation_df.to_csv(output_dir / "prefix_ablation_details.csv", index=False)
    head_summary.to_csv(output_dir / "head_patch_summary.csv", index=False)
    expansion_summary.to_csv(output_dir / "anchored_head_expansion_summary.csv", index=False)
    wrong_summary.to_csv(output_dir / "wrong_option_token_patch_summary.csv", index=False)
    ablation_summary.to_csv(output_dir / "prefix_ablation_summary.csv", index=False)
    case_df.to_csv(output_dir / "drop_wrong_option_case_study.csv", index=False)
    case_summary.to_csv(output_dir / "drop_wrong_option_case_summary.csv", index=False)
    _write_jsonl(output_dir / "head_patch_details.jsonl", head_df.to_dict(orient="records"))
    _write_jsonl(output_dir / "anchored_head_expansion_details.jsonl", expansion_df.to_dict(orient="records"))
    _write_jsonl(output_dir / "wrong_option_token_patch_details.jsonl", wrong_df.to_dict(orient="records"))
    _write_jsonl(output_dir / "prefix_ablation_details.jsonl", ablation_df.to_dict(orient="records"))
    _write_jsonl(output_dir / "drop_wrong_option_case_study.jsonl", case_df.to_dict(orient="records"))
    save_json(output_dir / "focus_samples.json", sample_notes)

    report_lines = [
        "# Late-Layer Local Probe Report",
        "",
        f"- Focus samples: {len(focus_sample_ids)}",
        f"- Late layers: {late_layers}",
        f"- Head focus layers: {head_focus_layers}",
        "",
        "## Top Heads",
        "",
        _markdown_table(head_summary.head(15)),
        "",
        "## Anchored Head Expansion",
        "",
        _markdown_table(expansion_summary),
        "",
        "## Wrong-Option Token Patching",
        "",
        _markdown_table(wrong_summary.head(15)),
        "",
        "## Prefix Ablation",
        "",
        _markdown_table(ablation_summary),
        "",
        "## Drop-Wrong-Option Case Study",
        "",
        _markdown_table(case_df[[
            "sample_id",
            "sample_type",
            "condition_id",
            "ground_truth",
            "wrong_option",
            "none_predicted_answer",
            "drop_wrong_option_answer",
            "drop_wrong_option_recovers_correct",
            "drop_wrong_option_margin_improvement",
            "best_single_head_label",
            "best_expansion_label",
            "best_expansion_margin_gain",
            "explicit_wrong_option_cue_dependence",
        ]]),
        "",
    ]
    (output_dir / "late_layer_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "model_name": str(args.model_name),
        "mechanistic_run_dir": str(mech_dir.resolve()),
        "sample_file": str(Path(args.sample_file).resolve()),
        "output_dir": str(output_dir.resolve()),
        "focus_samples": len(focus_sample_ids),
        "anchor_head": {"layer_index": int(anchor_head[0]), "head_index": int(anchor_head[1])},
        "ordered_expansion_heads": [{"layer_index": int(layer), "head_index": int(head)} for layer, head in ordered_heads],
        "expansion_sizes": [int(item["expansion_size"]) for item in expansion_sets],
        "files": {
            "head_patch_summary": str((output_dir / "head_patch_summary.csv").resolve()),
            "anchored_head_expansion_summary": str((output_dir / "anchored_head_expansion_summary.csv").resolve()),
            "wrong_option_token_patch_summary": str((output_dir / "wrong_option_token_patch_summary.csv").resolve()),
            "prefix_ablation_summary": str((output_dir / "prefix_ablation_summary.csv").resolve()),
            "head_patch_details_jsonl": str((output_dir / "head_patch_details.jsonl").resolve()),
            "anchored_head_expansion_details_jsonl": str((output_dir / "anchored_head_expansion_details.jsonl").resolve()),
            "wrong_option_token_patch_details_jsonl": str((output_dir / "wrong_option_token_patch_details.jsonl").resolve()),
            "prefix_ablation_details_jsonl": str((output_dir / "prefix_ablation_details.jsonl").resolve()),
            "drop_wrong_option_case_study_jsonl": str((output_dir / "drop_wrong_option_case_study.jsonl").resolve()),
            "report": str((output_dir / "late_layer_report.md").resolve()),
        },
    }
    save_json(output_dir / "late_layer_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
