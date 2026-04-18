#!/usr/bin/env python3
"""Rebuild trusted full-data detector + real recheck results from raw sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCORED_DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "interference_detector_new15_full_sentence_embedding"
    / "sentence_embedding_logreg_strict_new15_full_scored.csv"
)
RUN_SPECS = [
    {
        "run_name": "same_model_full_real_deepseek_chat",
        "run_dir": PROJECT_ROOT / "outputs" / "experiments" / "same_model_full_real" / "deepseek_chat" / "20260413_210903",
    },
    {
        "run_name": "same_model_full_real_qwen_max",
        "run_dir": PROJECT_ROOT / "outputs" / "experiments" / "same_model_full_real" / "qwen_max" / "20260413_210903",
    },
    {
        "run_name": "same_model_full_real_qwen3_max",
        "run_dir": PROJECT_ROOT / "outputs" / "experiments" / "same_model_full_real" / "qwen3_max" / "20260413_210903",
    },
]

for chunk_idx in range(1, 9):
    RUN_SPECS.append(
        {
            "run_name": f"reasoner_full_real_chunk_{chunk_idx:02d}",
            "run_dir": PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "reasoner_full_real"
            / f"chunk_{chunk_idx:02d}"
            / ("20260413_210525" if chunk_idx == 1 else "20260413_211206"),
        }
    )

GROUP_KEY_COLUMNS = ["model_name", "task_id", "arm_id"]
SAMPLE_KEY_COLUMNS = ["model_name", "task_id", "arm_id", "sample_index"]
STABLE_KEY_COLUMNS = SAMPLE_KEY_COLUMNS + ["occurrence_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild trusted full-data detector + real recheck results.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments" / "full_real_recheck_rebuilt",
        help="Destination directory for rebuilt outputs.",
    )
    return parser.parse_args()


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _assign_occurrence_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["occurrence_index"] = out.groupby(SAMPLE_KEY_COLUMNS).cumcount().astype(int)
    return out


def _assign_group_occurrence_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_occurrence_index"] = out.groupby(GROUP_KEY_COLUMNS).cumcount().astype(int)
    return out


def _stable_row_key(df: pd.DataFrame) -> pd.Series:
    return (
        _text_series(df, "model_name").str.strip()
        + "::"
        + _text_series(df, "task_id").str.strip()
        + "::"
        + _text_series(df, "arm_id").str.strip()
        + "::"
        + _numeric_series(df, "sample_index", 0).astype(int).astype(str)
        + "::"
        + _numeric_series(df, "occurrence_index", 0).astype(int).astype(str)
    )


def _normalize_answer_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def _preferred_column(df: pd.DataFrame, base_name: str) -> str:
    for candidate in (f"{base_name}_judge", base_name):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Missing expected column: {base_name}")


@dataclass
class RunAlignmentReport:
    run_name: str
    input_rows: int
    judge_rows: int
    exact_matches: int
    aligned_rows: int
    unmatched_input_rows: int
    unmatched_judge_rows: int
    duplicate_stable_keys: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_run_level_table(run_name: str, input_path: Path, judge_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, RunAlignmentReport]:
    input_df = pd.read_csv(input_path).reset_index(drop=True)
    judge_df = _load_jsonl(judge_path).reset_index(drop=True)

    input_df = _assign_group_occurrence_index(input_df)
    judge_df = _assign_group_occurrence_index(judge_df)

    aligned = input_df.merge(
        judge_df,
        on=GROUP_KEY_COLUMNS + ["group_occurrence_index"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_judge"),
    )
    aligned["_merge"] = aligned["_merge"].astype(str)

    aligned["sample_index"] = _numeric_series(aligned, "sample_index", 0).astype(int)
    aligned = _assign_occurrence_index(aligned)
    aligned["stable_row_key"] = _stable_row_key(aligned)

    judge_answer_col = _preferred_column(aligned, "recheck_answer")
    finish_reason_col = _preferred_column(aligned, "recheck_finish_reason")
    success_col = _preferred_column(aligned, "success")

    aligned["raw_answer"] = _normalize_answer_series(_text_series(aligned, "predicted_answer"))
    aligned["recheck_answer"] = _normalize_answer_series(_text_series(aligned, judge_answer_col))
    aligned["final_answer"] = aligned["raw_answer"]
    valid_recheck = aligned["recheck_answer"].isin(["A", "B", "C", "D"])
    aligned.loc[valid_recheck, "final_answer"] = aligned.loc[valid_recheck, "recheck_answer"]
    aligned["success"] = _numeric_series(aligned, success_col, 0).astype(bool)
    aligned["finish_reason"] = _text_series(aligned, finish_reason_col)
    aligned["run_name"] = run_name

    report = RunAlignmentReport(
        run_name=run_name,
        input_rows=int(len(input_df)),
        judge_rows=int(len(judge_df)),
        exact_matches=int((aligned["_merge"] == "both").sum()),
        aligned_rows=int(len(aligned)),
        unmatched_input_rows=int((aligned["_merge"] == "left_only").sum()),
        unmatched_judge_rows=int((aligned["_merge"] == "right_only").sum()),
        duplicate_stable_keys=int(aligned["stable_row_key"].duplicated().sum()),
    )

    keep_columns = [
        "run_name",
        "model_name",
        "task_id",
        "arm_id",
        "sample_index",
        "occurrence_index",
        "stable_row_key",
        "raw_answer",
        "recheck_answer",
        "final_answer",
        "success",
        "finish_reason",
        "ground_truth",
        "wrong_option",
        "predicted_answer",
        "interference_score",
        "explicit_wrong_option",
        "is_control",
        "authority_level",
        "confidence_level",
        "_merge",
    ]
    matched = aligned.loc[aligned["_merge"] == "both", keep_columns].copy()
    unmatched = aligned.loc[aligned["_merge"] != "both"].copy()
    return matched, unmatched, report


def compute_metrics(df: pd.DataFrame, final_answer_column: str, triggered_column: str) -> Dict[str, Any]:
    working = df.copy()
    final_answer = _normalize_answer_series(_text_series(working, final_answer_column))
    raw_answer = _normalize_answer_series(_text_series(working, "predicted_answer"))
    ground_truth = _normalize_answer_series(_text_series(working, "ground_truth"))
    wrong_option = _normalize_answer_series(_text_series(working, "wrong_option"))
    triggered = _numeric_series(working, triggered_column, 0).astype(int)

    raw_correct = raw_answer.eq(ground_truth)
    final_correct = final_answer.eq(ground_truth)
    raw_wrong_follow = wrong_option.isin(["A", "B", "C", "D"]) & raw_answer.eq(wrong_option)
    final_wrong_follow = wrong_option.isin(["A", "B", "C", "D"]) & final_answer.eq(wrong_option)
    changed_to_correct = (~raw_correct) & final_correct & final_answer.ne(raw_answer)
    correct_to_wrong = raw_correct & (~final_correct) & final_answer.ne(raw_answer)

    return {
        "n": int(len(working)),
        "full_accuracy": float(final_correct.mean()) if len(working) else 0.0,
        "full_wrong_option_follow_rate": float(final_wrong_follow.mean()) if len(working) else 0.0,
        "trigger_rate": float(triggered.mean()) if len(working) else 0.0,
        "changed_to_correct_count": int(changed_to_correct.sum()),
        "changed_to_correct_rate": float(changed_to_correct.mean()) if len(working) else 0.0,
        "correct_to_wrong_count": int(correct_to_wrong.sum()),
        "correct_to_wrong_rate": float(correct_to_wrong.mean()) if len(working) else 0.0,
        "raw_accuracy": float(raw_correct.mean()) if len(working) else 0.0,
        "raw_wrong_option_follow_rate": float(raw_wrong_follow.mean()) if len(working) else 0.0,
    }


def per_model_metrics(df: pd.DataFrame, final_answer_column: str, triggered_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, model_df in df.groupby("model_name", dropna=False):
        metrics = compute_metrics(model_df, final_answer_column=final_answer_column, triggered_column=triggered_column)
        metrics["model_name"] = str(model_name)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(SCORED_DATASET_PATH).reset_index(drop=True)
    full_df = _assign_occurrence_index(full_df)
    full_df["stable_row_key"] = _stable_row_key(full_df)

    all_run_tables: List[pd.DataFrame] = []
    all_unmatched_tables: List[pd.DataFrame] = []
    run_reports: List[Dict[str, Any]] = []
    for spec in RUN_SPECS:
        run_table, unmatched_table, report = build_run_level_table(
            run_name=str(spec["run_name"]),
            input_path=Path(spec["run_dir"]) / "sample_manifest.csv",
            judge_path=Path(spec["run_dir"]) / "judge_recheck_results.jsonl",
        )
        all_run_tables.append(run_table)
        all_unmatched_tables.append(unmatched_table)
        run_reports.append(report.to_dict())
        run_table.to_csv(output_dir / f"{spec['run_name']}_run_level_results.csv", index=False)
        unmatched_table.to_csv(output_dir / f"{spec['run_name']}_unmatched_alignment_rows.csv", index=False)

    combined_runs = pd.concat(all_run_tables, axis=0, ignore_index=True)
    combined_runs.to_csv(output_dir / "combined_run_level_results.csv", index=False)
    combined_unmatched = pd.concat(all_unmatched_tables, axis=0, ignore_index=True)
    combined_unmatched.to_csv(output_dir / "combined_unmatched_alignment_rows.csv", index=False)
    raw_judge_total_rows = int(sum(item["judge_rows"] for item in run_reports))

    duplicate_run_keys = int(combined_runs["stable_row_key"].duplicated().sum())
    duplicate_full_keys = int(full_df["stable_row_key"].duplicated().sum())

    run_key_match = combined_runs.merge(
        full_df[["stable_row_key"]],
        on="stable_row_key",
        how="left",
        indicator="_full_match",
        validate="many_to_one",
    )
    unmatched_run_results = run_key_match.loc[
        run_key_match["_full_match"] != "both",
        ["run_name", "model_name", "task_id", "arm_id", "sample_index", "occurrence_index", "stable_row_key"],
    ]
    unmatched_run_results.to_csv(output_dir / "unmatched_run_results.csv", index=False)

    merged = full_df.merge(
        combined_runs[
            [
                "run_name",
                "stable_row_key",
                "raw_answer",
                "recheck_answer",
                "final_answer",
                "success",
                "finish_reason",
            ]
        ],
        on="stable_row_key",
        how="left",
        validate="one_to_one",
    )

    merged["raw_answer"] = _normalize_answer_series(_text_series(merged, "predicted_answer"))
    merged["recheck_answer"] = _normalize_answer_series(_text_series(merged, "recheck_answer"))
    merged["finish_reason"] = _text_series(merged, "finish_reason")
    merged["has_real_recheck"] = _text_series(merged, "run_name").ne("").astype(int)

    merged["trigger_b"] = (_numeric_series(merged, "interference_score", 0.0) >= 0.55).astype(int)
    merged["final_answer_a"] = merged["raw_answer"]
    merged["final_answer_b"] = merged["raw_answer"]
    valid_b = merged["trigger_b"].eq(1) & merged["recheck_answer"].isin(["A", "B", "C", "D"])
    merged.loc[valid_b, "final_answer_b"] = merged.loc[valid_b, "recheck_answer"]

    merged["trigger_c"] = merged["trigger_b"]
    merged.loc[_text_series(merged, "model_name").str.strip().str.lower().eq("deepseek-reasoner"), "trigger_c"] = 0
    merged["final_answer_c"] = merged["raw_answer"]
    valid_c = merged["trigger_c"].eq(1) & merged["recheck_answer"].isin(["A", "B", "C", "D"])
    merged.loc[valid_c, "final_answer_c"] = merged.loc[valid_c, "recheck_answer"]

    merged.to_csv(output_dir / "full_dataset_with_recheck_backfill.csv", index=False)

    summary = {
        "sources": {
            "scored_dataset": str(SCORED_DATASET_PATH.resolve()),
            "run_specs": [
                {
                    "run_name": str(spec["run_name"]),
                    "run_dir": str(Path(spec["run_dir"]).resolve()),
                    "input_source": str((Path(spec["run_dir"]) / "sample_manifest.csv").resolve()),
                }
                for spec in RUN_SPECS
            ],
        },
        "validation": {
            "scored_total_rows": int(len(full_df)),
            "backfilled_total_rows": int(len(merged)),
            "raw_judge_total_rows": raw_judge_total_rows,
            "recoverable_real_recheck_rows": int(len(combined_runs)),
            "triggered_with_real_recheck_count_b": int((merged["trigger_b"].eq(1) & merged["has_real_recheck"].eq(1)).sum()),
            "triggered_with_real_recheck_count_c": int((merged["trigger_c"].eq(1) & merged["has_real_recheck"].eq(1)).sum()),
            "duplicate_merge_keys_full_dataset": duplicate_full_keys,
            "duplicate_merge_keys_run_results": duplicate_run_keys,
            "unmatched_run_results": int(len(unmatched_run_results)),
            "unmatched_alignment_rows": int(len(combined_unmatched)),
            "run_alignment_reports": run_reports,
        },
        "metrics": {
            "A_baseline": compute_metrics(merged.assign(trigger_a=0), final_answer_column="final_answer_a", triggered_column="trigger_a"),
            "B_all_models_detector_plus_recheck": compute_metrics(merged, final_answer_column="final_answer_b", triggered_column="trigger_b"),
            "C_nonreasoner_detector_plus_recheck_reasoner_no_detector_no_recheck": compute_metrics(
                merged,
                final_answer_column="final_answer_c",
                triggered_column="trigger_c",
            ),
        },
    }

    for label, final_col, trigger_col in [
        ("A_baseline", "final_answer_a", "trigger_a"),
        ("B_all_models_detector_plus_recheck", "final_answer_b", "trigger_b"),
        ("C_nonreasoner_detector_plus_recheck_reasoner_no_detector_no_recheck", "final_answer_c", "trigger_c"),
    ]:
        working = merged.copy()
        if trigger_col == "trigger_a":
            working["trigger_a"] = 0
        model_df = per_model_metrics(working, final_answer_column=final_col, triggered_column=trigger_col)
        model_df.to_csv(output_dir / f"{label}_by_model.csv", index=False)
        summary["metrics"][label]["by_model_path"] = str((output_dir / f"{label}_by_model.csv").resolve())

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
