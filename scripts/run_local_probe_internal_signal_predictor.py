#!/usr/bin/env python3
"""Run internal-signal predictor baselines on local probe mechanistic outputs."""

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
    build_internal_signal_dataset,
    cross_validated_predictor_baseline,
    feature_set_definitions,
    fit_full_model_coefficients,
    load_jsonl,
)
from src.open_model_probe.io_utils import prepare_output_dir, save_json


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal-signal predictor baselines for local Qwen probe results.")
    parser.add_argument("--probe-run-dir", required=True, help="Directory containing probe_comparisons.jsonl.")
    parser.add_argument("--mechanistic-run-dir", required=True, help="Directory containing mechanistic_sample_cases.jsonl.")
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_internal_predictor")
    parser.add_argument("--focus-layers", default="31,32,33,34,35")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-all-samples", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_internal_signal_predictor", level=args.log_level)

    probe_dir = Path(args.probe_run_dir)
    mech_dir = Path(args.mechanistic_run_dir)
    output_dir = prepare_output_dir(Path(args.output_dir), run_name="qwen_internal_signal_predictor")

    probe_rows = load_jsonl(probe_dir / "probe_comparisons.jsonl")
    sample_rows = load_jsonl(mech_dir / "mechanistic_sample_cases.jsonl")
    focus_layers = _parse_layers(args.focus_layers)
    dataset = build_internal_signal_dataset(
        probe_comparisons=probe_rows,
        sample_cases=sample_rows,
        focus_layers=focus_layers,
        baseline_correct_only=not bool(args.include_all_samples),
    )

    if dataset.empty:
        raise ValueError("Predictor dataset is empty; check inputs and filter settings.")

    feature_sets = feature_set_definitions()
    metric_rows = []
    prediction_frames = []
    coefficient_frames = []
    for feature_set_name, feature_columns in feature_sets.items():
        result, prediction_df = cross_validated_predictor_baseline(
            dataset,
            feature_columns=feature_columns,
            n_splits=int(args.cv_folds),
            seed=int(args.seed),
        )
        metric_rows.append(
            {
                "feature_set_name": feature_set_name,
                "num_samples": result.num_samples,
                "num_positive": result.num_positive,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
                "average_precision": result.average_precision,
                "feature_columns": list(result.feature_columns),
            }
        )
        prediction_df["feature_set_name"] = feature_set_name
        prediction_frames.append(prediction_df)

        coef_df = fit_full_model_coefficients(dataset, feature_columns=feature_columns)
        coef_df["feature_set_name"] = feature_set_name
        coefficient_frames.append(coef_df)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False])
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True)

    dataset.to_csv(output_dir / "predictor_dataset.csv", index=False)
    metrics_df.to_csv(output_dir / "predictor_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictor_cv_predictions.csv", index=False)
    coefficients_df.to_csv(output_dir / "predictor_coefficients.csv", index=False)

    by_feature_set_and_type = (
        predictions_df.groupby(["feature_set_name", "sample_type"], dropna=False)
        .agg(
            num_samples=("sample_id", "count"),
            positive_rate=(TARGET_LABEL, "mean"),
            mean_predicted_risk=("predicted_risk_score", "mean"),
        )
        .reset_index()
    )
    by_feature_set_and_type.to_csv(output_dir / "predictor_by_sample_type.csv", index=False)

    summary = {
        "probe_run_dir": str(probe_dir.resolve()),
        "mechanistic_run_dir": str(mech_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "focus_layers": list(focus_layers),
        "baseline_correct_only": not bool(args.include_all_samples),
        "num_samples": int(len(dataset)),
        "num_positive": int(dataset[TARGET_LABEL].sum()),
        "files": {
            "predictor_dataset": str((output_dir / "predictor_dataset.csv").resolve()),
            "predictor_metrics": str((output_dir / "predictor_metrics.csv").resolve()),
            "predictor_cv_predictions": str((output_dir / "predictor_cv_predictions.csv").resolve()),
            "predictor_coefficients": str((output_dir / "predictor_coefficients.csv").resolve()),
            "predictor_by_sample_type": str((output_dir / "predictor_by_sample_type.csv").resolve()),
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }
    save_json(output_dir / "predictor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
