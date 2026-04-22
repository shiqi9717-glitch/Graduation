#!/usr/bin/env python3
"""Generalization + feature ablation evaluation for local probe internal predictors."""

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
    ablation_feature_set_definitions,
    build_internal_signal_dataset,
    cross_validated_predictor_baseline,
    evaluate_predictions,
    feature_set_definitions,
    fit_full_model_coefficients,
    fit_predictor_model,
    load_jsonl,
    predict_with_model,
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
    parser = argparse.ArgumentParser(description="Generalization evaluation for local probe internal-signal predictors.")
    parser.add_argument("--train-probe-run-dir", required=True)
    parser.add_argument("--train-mechanistic-run-dir", required=True)
    parser.add_argument("--generalization-probe-run-dir", required=True)
    parser.add_argument("--generalization-mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_internal_predictor_generalization")
    parser.add_argument("--focus-layers", default="31,32,33,34,35")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_predictor_generalization", level=args.log_level)

    output_dir = prepare_output_dir(Path(args.output_dir), run_name="qwen_internal_predictor_generalization")
    focus_layers = _parse_layers(args.focus_layers)

    train_dataset = build_internal_signal_dataset(
        probe_comparisons=load_jsonl(Path(args.train_probe_run_dir) / "probe_comparisons.jsonl"),
        sample_cases=load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl"),
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    generalization_dataset = build_internal_signal_dataset(
        probe_comparisons=load_jsonl(Path(args.generalization_probe_run_dir) / "probe_comparisons.jsonl"),
        sample_cases=load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl"),
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

    # Original three-set comparison: CV on train + held-out generalization.
    for feature_set_name, feature_columns in feature_set_definitions().items():
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
        generalization_predictions = predict_with_model(model, generalization_dataset)
        generalization_predictions.attrs["feature_columns"] = list(model.feature_columns)
        gen_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=generalization_predictions)
        metrics_rows.append(_result_row(gen_result, eval_split="generalization", num_features=len(feature_columns)))
        generalization_predictions["feature_set_name"] = feature_set_name
        generalization_predictions["eval_split"] = "generalization"
        prediction_frames.append(generalization_predictions)

        coef_df = fit_full_model_coefficients(train_dataset, feature_columns=feature_columns)
        coef_df["feature_set_name"] = feature_set_name
        coef_df["eval_split"] = "train_fit"
        coefficient_frames.append(coef_df)

    # Feature ablation comparison on train and generalization.
    ablation_rows = []
    ablation_coef_frames = []
    for feature_set_name, feature_columns in ablation_feature_set_definitions().items():
        cv_result, _ = cross_validated_predictor_baseline(
            train_dataset,
            feature_columns=feature_columns,
            n_splits=int(args.cv_folds),
            seed=int(args.seed),
        )
        cv_result = cv_result.__class__(feature_set_name=feature_set_name, **{k: getattr(cv_result, k) for k in cv_result.__dataclass_fields__ if k != "feature_set_name"})
        ablation_rows.append(_result_row(cv_result, eval_split="train_cv", num_features=len(feature_columns)))

        model = fit_predictor_model(train_dataset, feature_columns=feature_columns)
        generalization_predictions = predict_with_model(model, generalization_dataset)
        generalization_predictions.attrs["feature_columns"] = list(model.feature_columns)
        gen_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=generalization_predictions)
        ablation_rows.append(_result_row(gen_result, eval_split="generalization", num_features=len(feature_columns)))

        coef_df = fit_full_model_coefficients(train_dataset, feature_columns=feature_columns)
        coef_df["feature_set_name"] = feature_set_name
        coef_df["eval_split"] = "train_fit"
        ablation_coef_frames.append(coef_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["eval_split", "roc_auc", "average_precision", "f1"], ascending=[True, False, False, False])
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    coefficients_df = pd.concat(coefficient_frames + ablation_coef_frames, ignore_index=True)
    ablation_df = pd.DataFrame(ablation_rows).sort_values(["eval_split", "roc_auc", "average_precision", "f1"], ascending=[True, False, False, False])

    metrics_df.to_csv(output_dir / "generalization_metrics.csv", index=False)
    ablation_df.to_csv(output_dir / "feature_ablation_metrics.csv", index=False)
    coefficients_df.to_csv(output_dir / "feature_coefficients.csv", index=False)
    predictions_df.to_csv(output_dir / "generalization_predictions.csv", index=False)

    original_cv = metrics_df[metrics_df["eval_split"] == "train_cv"].copy()
    held_out = metrics_df[metrics_df["eval_split"] == "generalization"].copy()

    def _lookup(df: pd.DataFrame, feature_set_name: str, metric: str):
        row = df[df["feature_set_name"] == feature_set_name]
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
        "train_num_samples": int(len(train_dataset)),
        "train_num_positive": int(train_dataset[TARGET_LABEL].sum()),
        "generalization_num_samples": int(len(generalization_dataset)),
        "generalization_num_positive": int(generalization_dataset[TARGET_LABEL].sum()),
        "original_cv_metrics": original_cv.to_dict(orient="records"),
        "generalization_metrics": held_out.to_dict(orient="records"),
        "feature_ablation_metrics": ablation_df.to_dict(orient="records"),
        "conclusions": {
            "internal_vs_output_generalization_roc_auc_delta": (
                _lookup(held_out, "internal_signal", "roc_auc") - _lookup(held_out, "output_only", "roc_auc")
                if _lookup(held_out, "internal_signal", "roc_auc") is not None and _lookup(held_out, "output_only", "roc_auc") is not None
                else None
            ),
            "internal_vs_external_generalization_roc_auc_delta": (
                _lookup(held_out, "internal_signal", "roc_auc") - _lookup(held_out, "external_only", "roc_auc")
                if _lookup(held_out, "internal_signal", "roc_auc") is not None and _lookup(held_out, "external_only", "roc_auc") is not None
                else None
            ),
            "best_minimal_feature_set": (
                ablation_df[ablation_df["eval_split"] == "generalization"]
                .sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False])
                .iloc[0]["feature_set_name"]
                if not ablation_df[ablation_df["eval_split"] == "generalization"].empty
                else None
            ),
        },
        "files": {
            "generalization_dataset": str((output_dir / "generalization_dataset.csv").resolve()),
            "generalization_metrics": str((output_dir / "generalization_metrics.csv").resolve()),
            "feature_ablation_metrics": str((output_dir / "feature_ablation_metrics.csv").resolve()),
            "feature_coefficients": str((output_dir / "feature_coefficients.csv").resolve()),
            "generalization_predictions": str((output_dir / "generalization_predictions.csv").resolve()),
        },
    }
    save_json(output_dir / "generalization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
