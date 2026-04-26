#!/usr/bin/env python3
"""Export DeepSeek-chat new15 rows in a V4 behavior-summary-compatible shape."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


DEFAULT_SOURCE = (
    "/Users/shiqi/code/代码/毕业代码/outputs/experiments/"
    "interference_detector_new15_full_detector_grid/interference_strict_new15_full.csv"
)

ARM_TO_CONDITION = {
    "ctrl_base": "baseline",
    "a1_c1_w1": "strict_positive",
    "a2_c1_w1": "high_pressure_wrong_option",
}


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def _condition_summary(df: pd.DataFrame, condition: str) -> Dict[str, Any]:
    subset = df[df["condition"] == condition].copy()
    if subset.empty:
        return {
            "model_name": "deepseek-chat",
            "aggregation_level": "condition",
            "condition": condition,
            "arm_id": "",
            "num_responses": 0,
            "num_items": 0,
            "accuracy": 0.0,
            "wrong_option_follow_rate": 0.0,
            "invalid_rate": 0.0,
            "refusal_or_abstention_rate": "",
            "mean_latency_ms": "",
            "baseline_to_pressured_changed_rate": "",
            "baseline_accuracy_reference": "",
        }
    pressured = condition in {"strict_positive", "high_pressure_wrong_option"}
    return {
        "model_name": "deepseek-chat",
        "aggregation_level": "condition",
        "condition": condition,
        "arm_id": subset["arm_id"].iloc[0],
        "num_responses": int(len(subset)),
        "num_items": int(subset["task_id"].nunique()),
        "accuracy": _safe_mean(subset["answer_equals_ground_truth"]),
        "wrong_option_follow_rate": _safe_mean(subset["answer_equals_wrong_option"]),
        "invalid_rate": _safe_mean(subset["is_invalid"]),
        "refusal_or_abstention_rate": "",
        "mean_latency_ms": "",
        "baseline_to_pressured_changed_rate": _safe_mean(subset["answer_changed_from_baseline"]) if pressured else "",
        "baseline_accuracy_reference": _safe_mean(subset["baseline_accuracy_prob"]) if pressured else "",
    }


def _comparison_summary(df: pd.DataFrame, condition: str) -> Dict[str, Any]:
    subset = df[df["condition"] == condition].copy()
    return {
        "model_name": "deepseek-chat",
        "aggregation_level": "comparison",
        "condition": condition,
        "arm_id": subset["arm_id"].iloc[0] if not subset.empty else "",
        "num_responses": int(len(subset)),
        "num_items": int(subset["task_id"].nunique()) if not subset.empty else 0,
        "accuracy": _safe_mean(subset["answer_equals_ground_truth"]),
        "wrong_option_follow_rate": _safe_mean(subset["answer_equals_wrong_option"]),
        "invalid_rate": "",
        "refusal_or_abstention_rate": "",
        "mean_latency_ms": "",
        "baseline_to_pressured_changed_rate": _safe_mean(subset["answer_changed_from_baseline"]),
        "baseline_accuracy_reference": _safe_mean(subset["baseline_accuracy_prob"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export V3 DeepSeek-chat summary aligned to V4 behavior_summary.csv.")
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", default="outputs/experiments/deepseek_v3_v4_aligned_behavior_summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source_csv).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_columns = [
        "pair_key",
        "task_id",
        "model_name",
        "split",
        "sample_index",
        "arm_id",
        "ground_truth",
        "wrong_option",
        "predicted_answer",
        "is_invalid",
        "subject",
        "category",
        "baseline_answer",
        "baseline_accuracy_prob",
        "answer_equals_wrong_option",
        "answer_equals_ground_truth",
        "answer_equals_baseline_answer",
        "answer_changed_from_baseline",
        "arm_correct_prob",
        "arm_wrong_option_follow_prob",
    ]
    df = pd.read_csv(source, usecols=lambda col: col in use_columns)
    df = df[df["model_name"].fillna("").astype(str).eq("deepseek-chat")].copy()
    df = df[df["arm_id"].isin(ARM_TO_CONDITION)].copy()
    df["condition"] = df["arm_id"].map(ARM_TO_CONDITION)
    df["v4_condition_name"] = df["condition"]

    baseline = df[df["condition"] == "baseline"].copy()
    strict = df[df["condition"] == "strict_positive"].copy()
    high = df[df["condition"] == "high_pressure_wrong_option"].copy()

    key_cols = ["task_id", "sample_index"]
    alignment = baseline[key_cols + ["predicted_answer", "answer_equals_ground_truth"]].rename(
        columns={
            "predicted_answer": "baseline_answer_row",
            "answer_equals_ground_truth": "baseline_correct_row",
        }
    )
    for name, subset in [("strict_positive", strict), ("high_pressure_wrong_option", high)]:
        cols = key_cols + [
            "predicted_answer",
            "answer_equals_ground_truth",
            "answer_equals_wrong_option",
            "answer_changed_from_baseline",
        ]
        alignment = alignment.merge(
            subset[cols].rename(
                columns={
                    "predicted_answer": f"{name}_answer",
                    "answer_equals_ground_truth": f"{name}_correct",
                    "answer_equals_wrong_option": f"{name}_wrong_option_follow",
                    "answer_changed_from_baseline": f"{name}_changed_from_baseline",
                }
            ),
            on=key_cols,
            how="left",
        )

    rows: List[Dict[str, Any]] = [
        _condition_summary(df, "baseline"),
        _condition_summary(df, "strict_positive"),
        _condition_summary(df, "high_pressure_wrong_option"),
        _comparison_summary(df, "strict_positive"),
        _comparison_summary(df, "high_pressure_wrong_option"),
    ]
    summary = pd.DataFrame(rows)

    summary_path = output_dir / "deepseek_chat_new15_v4_aligned_behavior_summary.csv"
    alignment_path = output_dir / "deepseek_chat_new15_v4_aligned_item_comparisons.csv"
    records_path = output_dir / "deepseek_chat_new15_v4_aligned_records.csv"
    manifest_path = output_dir / "deepseek_chat_new15_v4_aligned_manifest.json"
    summary.to_csv(summary_path, index=False)
    alignment.to_csv(alignment_path, index=False)
    df.to_csv(records_path, index=False)
    manifest = {
        "pipeline": "deepseek_v3_v4_aligned_behavior_summary_v1",
        "source_csv": str(source.resolve()),
        "output_dir": str(output_dir.resolve()),
        "arm_mapping": ARM_TO_CONDITION,
        "outputs": {
            "summary_csv": str(summary_path.resolve()),
            "item_comparisons_csv": str(alignment_path.resolve()),
            "records_csv": str(records_path.resolve()),
        },
        "notes": [
            "num_responses is scenario-level rows; V3 includes repeated sample_index responses per item.",
            "baseline_to_pressured_changed_rate uses answer_changed_from_baseline from the new15 strict table.",
            "wrong_option_follow_rate uses answer_equals_wrong_option, aligned with V4 behavior_summary wrong_option_follow_rate.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
