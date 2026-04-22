#!/usr/bin/env python3
"""Runtime-safe internal signal discovery on existing local probe outputs."""

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
    build_runtime_safe_signal_dataset,
    cross_validated_predictor_baseline,
    evaluate_predictions,
    fit_full_model_coefficients,
    fit_predictor_model,
    load_jsonl,
    predict_with_model,
    runtime_safe_signal_feature_groups,
    runtime_safe_signal_feature_set_definitions,
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
    parser = argparse.ArgumentParser(description="Runtime-safe signal discovery for Qwen local probe outputs.")
    parser.add_argument("--train-probe-run-dir", required=True)
    parser.add_argument("--train-mechanistic-run-dir", required=True)
    parser.add_argument("--generalization-probe-run-dir", required=True)
    parser.add_argument("--generalization-mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_runtime_safe_signal_discovery")
    parser.add_argument("--focus-layers", default="31,32,33,34,35")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_runtime_safe_signal_discovery", level=args.log_level)

    focus_layers = _parse_layers(args.focus_layers)
    output_dir = prepare_output_dir(Path(args.output_dir), run_name="runtime_safe_signal_discovery")

    train_probe_run_payload = json.loads((Path(args.train_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))
    generalization_probe_run_payload = json.loads((Path(args.generalization_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))

    train_probe_rows = load_jsonl(Path(args.train_probe_run_dir) / "probe_comparisons.jsonl")
    train_case_rows = load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl")
    train_scenario_rows = load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl")
    gen_probe_rows = load_jsonl(Path(args.generalization_probe_run_dir) / "probe_comparisons.jsonl")
    gen_case_rows = load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl")
    gen_scenario_rows = load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl")

    train_dataset = build_runtime_safe_signal_dataset(
        probe_comparisons=train_probe_rows,
        sample_cases=train_case_rows,
        scenario_records=train_scenario_rows,
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    generalization_dataset = build_runtime_safe_signal_dataset(
        probe_comparisons=gen_probe_rows,
        sample_cases=gen_case_rows,
        scenario_records=gen_scenario_rows,
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    train_dataset["eval_split"] = "train"
    generalization_dataset["eval_split"] = "generalization"
    combined_dataset = pd.concat([train_dataset, generalization_dataset], ignore_index=True)
    combined_dataset.to_csv(output_dir / "runtime_safe_dataset.csv", index=False)

    feature_groups = runtime_safe_signal_feature_groups()
    save_json(output_dir / "runtime_safe_feature_groups.json", feature_groups)

    metrics_rows = []
    prediction_frames = []
    coefficient_frames = []
    feature_sets = runtime_safe_signal_feature_set_definitions()
    for feature_set_name, feature_columns in feature_sets.items():
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

    metrics_df.to_csv(output_dir / "runtime_safe_metrics.csv", index=False)
    coefficients_df.to_csv(output_dir / "runtime_safe_coefficients.csv", index=False)
    predictions_df.to_csv(output_dir / "runtime_safe_predictions.csv", index=False)

    generalization_rows = metrics_df[metrics_df["eval_split"] == "generalization"].copy()

    def _lookup(feature_set_name: str, metric: str):
        row = generalization_rows[generalization_rows["feature_set_name"] == feature_set_name]
        if row.empty:
            return None
        value = row.iloc[0][metric]
        return None if pd.isna(value) else float(value)

    best_generalization = (
        generalization_rows.sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
        if not generalization_rows.empty
        else None
    )
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
        "train_positive_ratio": float(train_dataset[TARGET_LABEL].mean()) if len(train_dataset) else None,
        "generalization_raw_num_samples": int(generalization_probe_run_payload.get("num_samples", len(gen_probe_rows))),
        "generalization_num_samples": int(len(generalization_dataset)),
        "generalization_num_positive": int(generalization_dataset[TARGET_LABEL].sum()),
        "generalization_positive_ratio": float(generalization_dataset[TARGET_LABEL].mean()) if len(generalization_dataset) else None,
        "feature_groups": feature_groups,
        "metrics": metrics_df.to_dict(orient="records"),
        "best_runtime_safe_predictor": best_generalization,
        "conclusions": {
            "runtime_internal_only_safe_v2_is_still_weak": bool(
                (_lookup("runtime_internal_only_safe_v2", "roc_auc") or 0.0)
                < (_lookup("runtime_output_only_safe", "roc_auc") or 0.0)
            ),
            "runtime_output_plus_internal_safe_v2_vs_output_only_roc_auc_delta": (
                _lookup("runtime_output_plus_internal_safe_v2", "roc_auc") - _lookup("runtime_output_only_safe", "roc_auc")
                if _lookup("runtime_output_plus_internal_safe_v2", "roc_auc") is not None and _lookup("runtime_output_only_safe", "roc_auc") is not None
                else None
            ),
            "runtime_output_plus_internal_safe_v2_vs_output_only_ap_delta": (
                _lookup("runtime_output_plus_internal_safe_v2", "average_precision") - _lookup("runtime_output_only_safe", "average_precision")
                if _lookup("runtime_output_plus_internal_safe_v2", "average_precision") is not None and _lookup("runtime_output_only_safe", "average_precision") is not None
                else None
            ),
            "best_runtime_safe_feature_set_name": None if best_generalization is None else best_generalization.get("feature_set_name"),
        },
        "files": {
            "runtime_safe_dataset": str((output_dir / "runtime_safe_dataset.csv").resolve()),
            "runtime_safe_metrics": str((output_dir / "runtime_safe_metrics.csv").resolve()),
            "runtime_safe_coefficients": str((output_dir / "runtime_safe_coefficients.csv").resolve()),
            "runtime_safe_feature_groups": str((output_dir / "runtime_safe_feature_groups.json").resolve()),
            "runtime_safe_predictions": str((output_dir / "runtime_safe_predictions.csv").resolve()),
        },
    }
    save_json(output_dir / "runtime_safe_summary.json", summary)

    lines = [
        "# Runtime-Safe Signal Discovery",
        "",
        f"- Train raw probe samples: {summary['train_raw_num_samples']}",
        f"- Train baseline-correct samples: {summary['train_num_samples']}",
        f"- Held-out raw probe samples: {summary['generalization_raw_num_samples']}",
        f"- Held-out baseline-correct samples: {summary['generalization_num_samples']}",
        "",
        "## Generalization Metrics",
        "",
        "```csv",
        generalization_rows.to_csv(index=False).strip(),
        "```",
        "",
        "## Boundary",
        "",
        "- `analysis_only_features`: oracle-style explanatory features tied to correct-answer identity or ground truth.",
        "- `runtime_safe_features`: only signals available from outputs, known injected wrong-option cue, internal states, and attention without consulting the answer key.",
        "",
    ]
    (output_dir / "runtime_safe_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
