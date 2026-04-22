#!/usr/bin/env python3
"""Runtime-safe internal predictor evaluation on existing local probe outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.internal_signal_predictor import (
    TARGET_LABEL,
    build_runtime_safe_dataset,
    cross_validated_predictor_baseline,
    evaluate_predictions,
    fit_full_model_coefficients,
    fit_predictor_model,
    load_jsonl,
    predict_with_model,
    runtime_safe_feature_set_definitions,
)
from src.open_model_probe.io_utils import prepare_output_dir, save_json


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())


def _result_row(result, *, eval_split: str, num_features: int) -> dict:
    return {
        "feature_set_name": result.feature_set_name,
        "eval_split": eval_split,
        "num_samples": result.num_samples,
        "num_positive": result.num_positive,
        "num_features": int(num_features),
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "average_precision": result.average_precision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime-safe predictor evaluation for local probe outputs.")
    parser.add_argument("--train-probe-run-dir", required=True)
    parser.add_argument("--train-mechanistic-run-dir", required=True)
    parser.add_argument("--generalization-probe-run-dir", required=True)
    parser.add_argument("--generalization-mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_runtime_safe_predictor")
    parser.add_argument("--focus-layers", default="31,32,33,34,35")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_runtime_safe_predictor", level=args.log_level)

    focus_layers = _parse_layers(args.focus_layers)
    output_dir = prepare_output_dir(Path(args.output_dir), run_name="runtime_safe_predictor")

    train_probe_run_payload = json.loads((Path(args.train_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))
    generalization_probe_run_payload = json.loads((Path(args.generalization_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))

    train_probe_rows = load_jsonl(Path(args.train_probe_run_dir) / "probe_comparisons.jsonl")
    train_case_rows = load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl")
    train_scenario_rows = load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl")
    gen_probe_rows = load_jsonl(Path(args.generalization_probe_run_dir) / "probe_comparisons.jsonl")
    gen_case_rows = load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl")
    gen_scenario_rows = load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl")

    train_dataset = build_runtime_safe_dataset(
        probe_comparisons=train_probe_rows,
        sample_cases=train_case_rows,
        scenario_records=train_scenario_rows,
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    generalization_dataset = build_runtime_safe_dataset(
        probe_comparisons=gen_probe_rows,
        sample_cases=gen_case_rows,
        scenario_records=gen_scenario_rows,
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    train_dataset["eval_split"] = "train"
    generalization_dataset["eval_split"] = "generalization"
    combined_dataset = pd.concat([train_dataset, generalization_dataset], ignore_index=True)
    combined_dataset.to_csv(output_dir / "generalization_dataset.csv", index=False)

    metrics_rows = []
    prediction_frames = []
    coefficient_frames = []
    for feature_set_name, feature_columns in runtime_safe_feature_set_definitions().items():
        cv_result, cv_predictions = cross_validated_predictor_baseline(
            train_dataset,
            feature_columns=feature_columns,
            n_splits=int(args.cv_folds),
            seed=int(args.seed),
        )
        cv_result = cv_result.__class__(feature_set_name=feature_set_name, **{k: getattr(cv_result, k) for k in cv_result.__dataclass_fields__ if k != "feature_set_name"})
        metrics_rows.append(_result_row(cv_result, eval_split="train_cv", num_features=len(feature_columns)))
        cv_predictions["feature_set_name"] = feature_set_name
        cv_predictions["eval_split"] = "train_cv"
        prediction_frames.append(cv_predictions)

        model = fit_predictor_model(train_dataset, feature_columns=feature_columns)
        gen_predictions = predict_with_model(model, generalization_dataset)
        gen_predictions.attrs["feature_columns"] = list(model.feature_columns)
        gen_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=gen_predictions)
        metrics_rows.append(_result_row(gen_result, eval_split="generalization", num_features=len(feature_columns)))
        gen_predictions["feature_set_name"] = feature_set_name
        gen_predictions["eval_split"] = "generalization"
        prediction_frames.append(gen_predictions)

        coef_df = fit_full_model_coefficients(train_dataset, feature_columns=feature_columns)
        coef_df["feature_set_name"] = feature_set_name
        coef_df["eval_split"] = "train_fit"
        coefficient_frames.append(coef_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["eval_split", "roc_auc", "average_precision", "f1"], ascending=[True, False, False, False])
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True)
    metrics_df.to_csv(output_dir / "generalization_metrics.csv", index=False)
    metrics_df.to_csv(output_dir / "feature_ablation_metrics.csv", index=False)
    coefficients_df.to_csv(output_dir / "feature_coefficients.csv", index=False)
    predictions_df.to_csv(output_dir / "generalization_predictions.csv", index=False)

    generalization_table = metrics_df[metrics_df["eval_split"] == "generalization"].to_csv(index=False).strip()
    train_positive_ratio = float(train_dataset[TARGET_LABEL].mean()) if len(train_dataset) else None
    generalization_positive_ratio = float(generalization_dataset[TARGET_LABEL].mean()) if len(generalization_dataset) else None
    report_lines = [
        "# Runtime-Safe Predictor Summary",
        "",
        f"- Train raw probe samples: {int(train_probe_run_payload.get('num_samples', len(train_probe_rows)))}",
        f"- Train baseline-correct samples: {len(train_dataset)}",
        f"- Train positives / negatives: {int(train_dataset[TARGET_LABEL].sum())} / {int(len(train_dataset) - train_dataset[TARGET_LABEL].sum())}",
        f"- Train positive ratio: {train_positive_ratio:.4f}" if train_positive_ratio is not None else "- Train positive ratio: n/a",
        f"- Held-out raw probe samples: {int(generalization_probe_run_payload.get('num_samples', len(gen_probe_rows)))}",
        f"- Held-out baseline-correct samples: {len(generalization_dataset)}",
        f"- Held-out positives / negatives: {int(generalization_dataset[TARGET_LABEL].sum())} / {int(len(generalization_dataset) - generalization_dataset[TARGET_LABEL].sum())}",
        f"- Held-out positive ratio: {generalization_positive_ratio:.4f}" if generalization_positive_ratio is not None else "- Held-out positive ratio: n/a",
        "",
        "## Generalization Metrics",
        "",
        "```csv",
        generalization_table,
        "```",
        "",
        "## Notes",
        "",
        "- This report excludes any feature that directly uses ground-truth, correct-option rank, or correct-option margin.",
        "- Remaining features are restricted to runtime-safe internal signals, runtime-safe output summaries, and optional external metadata.",
        "",
    ]
    (output_dir / "runtime_safe_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    def _lookup(feature_set_name: str, metric: str):
        row = metrics_df[(metrics_df["eval_split"] == "generalization") & (metrics_df["feature_set_name"] == feature_set_name)]
        if row.empty:
            return None
        value = row.iloc[0][metric]
        return None if pd.isna(value) else float(value)

    summary = {
        "train_probe_run_dir": str(Path(args.train_probe_run_dir).resolve()),
        "train_mechanistic_run_dir": str(Path(args.train_mechanistic_run_dir).resolve()),
        "generalization_probe_run_dir": str(Path(args.generalization_probe_run_dir).resolve()),
        "generalization_mechanistic_run_dir": str(Path(args.generalization_mechanistic_run_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "focus_layers": list(focus_layers),
        "train_raw_num_samples": int(train_probe_run_payload.get("num_samples", len(train_probe_rows))),
        "train_num_samples": int(len(train_dataset)),
        "train_num_positive": int(train_dataset[TARGET_LABEL].sum()),
        "train_positive_ratio": train_positive_ratio,
        "generalization_raw_num_samples": int(generalization_probe_run_payload.get("num_samples", len(gen_probe_rows))),
        "generalization_num_samples": int(len(generalization_dataset)),
        "generalization_num_positive": int(generalization_dataset[TARGET_LABEL].sum()),
        "generalization_positive_ratio": generalization_positive_ratio,
        "metrics": metrics_df.to_dict(orient="records"),
        "conclusions": {
            "runtime_internal_safe_vs_output_safe_roc_auc_delta": (
                _lookup("runtime_internal_safe", "roc_auc") - _lookup("runtime_output_only_safe", "roc_auc")
                if _lookup("runtime_internal_safe", "roc_auc") is not None and _lookup("runtime_output_only_safe", "roc_auc") is not None
                else None
            ),
            "runtime_output_plus_internal_vs_output_safe_roc_auc_delta": (
                _lookup("runtime_output_plus_internal_safe", "roc_auc") - _lookup("runtime_output_only_safe", "roc_auc")
                if _lookup("runtime_output_plus_internal_safe", "roc_auc") is not None and _lookup("runtime_output_only_safe", "roc_auc") is not None
                else None
            ),
            "runtime_output_plus_attention_vs_output_safe_roc_auc_delta": (
                _lookup("runtime_output_plus_attention_safe", "roc_auc") - _lookup("runtime_output_only_safe", "roc_auc")
                if _lookup("runtime_output_plus_attention_safe", "roc_auc") is not None and _lookup("runtime_output_only_safe", "roc_auc") is not None
                else None
            ),
            "runtime_output_plus_head_vs_output_safe_roc_auc_delta": (
                _lookup("runtime_output_plus_head_safe", "roc_auc") - _lookup("runtime_output_only_safe", "roc_auc")
                if _lookup("runtime_output_plus_head_safe", "roc_auc") is not None and _lookup("runtime_output_only_safe", "roc_auc") is not None
                else None
            ),
        },
        "files": {
            "generalization_dataset": str((output_dir / "generalization_dataset.csv").resolve()),
            "generalization_metrics": str((output_dir / "generalization_metrics.csv").resolve()),
            "feature_ablation_metrics": str((output_dir / "feature_ablation_metrics.csv").resolve()),
            "feature_coefficients": str((output_dir / "feature_coefficients.csv").resolve()),
            "generalization_predictions": str((output_dir / "generalization_predictions.csv").resolve()),
            "runtime_safe_summary_md": str((output_dir / "runtime_safe_summary.md").resolve()),
        },
    }
    save_json(output_dir / "runtime_safe_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
