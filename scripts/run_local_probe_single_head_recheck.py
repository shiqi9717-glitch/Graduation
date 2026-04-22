#!/usr/bin/env python3
"""Focused recheck for output-only vs output-plus-single-head runtime monitors."""

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
    RegularizedSearchResult,
    budget_utility_rows_extended,
    count_nonzero_coefficients,
    evaluate_predictions,
    fit_full_model_coefficients,
    load_probe_sample_metadata,
    predict_with_model,
    runtime_safe_signal_feature_groups,
    select_best_regularized_model,
)
from src.open_model_probe.io_utils import prepare_output_dir, save_json


CANDIDATE_HEADS: tuple[tuple[int, int], ...] = (
    (34, 11),
    (31, 9),
    (31, 1),
    (34, 6),
    (34, 7),
    (34, 8),
    (31, 0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-head runtime monitor validation.")
    parser.add_argument(
        "--dataset-file",
        default="outputs/experiments/local_probe_qwen3b_sparse_runtime_monitor/sparse_runtime_monitor/20260419_164504/runtime_sparse_dataset.csv",
    )
    parser.add_argument(
        "--train-sample-file",
        default="outputs/experiments/local_probe_qwen3b/probe_sample_set_100.json",
    )
    parser.add_argument(
        "--generalization-sample-file",
        default="outputs/experiments/local_probe_qwen3b/generalization_probe_sample_set_200.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/local_probe_qwen3b_single_head_recheck",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _candidate_configs() -> list[tuple[str, float, float]]:
    return [
        ("l1_logistic", 0.01, 1.0),
        ("l1_logistic", 0.05, 1.0),
        ("l1_logistic", 0.10, 1.0),
        ("l1_logistic", 0.50, 1.0),
        ("l1_logistic", 1.00, 1.0),
    ]


def _output_feature_columns() -> list[str]:
    return list(runtime_safe_signal_feature_groups()["runtime_safe_features"]["output_safe"])


def _head_feature_columns(layer_index: int, head_index: int, available_columns: set[str]) -> list[str]:
    candidates = [
        f"head_L{int(layer_index)}H{int(head_index)}_prefix_attention",
        f"head_L{int(layer_index)}H{int(head_index)}_wrong_prefix_attention",
    ]
    return [column for column in candidates if column in available_columns]


def _result_row(
    result,
    *,
    split_type: str,
    eval_split: str,
    model_type: str,
    alpha: float,
    l1_wt: float,
    num_features: int,
    num_nonzero_features: int,
) -> dict:
    return {
        "feature_set_name": result.feature_set_name,
        "split_type": split_type,
        "eval_split": eval_split,
        "model_type": model_type,
        "alpha": float(alpha),
        "l1_wt": float(l1_wt),
        "num_samples": result.num_samples,
        "num_positive": result.num_positive,
        "num_features": int(num_features),
        "num_nonzero_features": int(num_nonzero_features),
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "average_precision": result.average_precision,
    }


def _fit_and_score(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_set_name: str,
    feature_columns: list[str],
    split_type: str,
    cv_folds: int,
    seed: int,
    candidate_configs: list[tuple[str, float, float]],
) -> tuple[RegularizedSearchResult, dict, dict, pd.DataFrame, pd.DataFrame]:
    search, model, train_predictions = select_best_regularized_model(
        train_df,
        feature_set_name=feature_set_name,
        feature_columns=feature_columns,
        candidate_configs=candidate_configs,
        n_splits=int(cv_folds),
        seed=int(seed),
    )
    train_predictions.attrs["feature_columns"] = list(model.feature_columns)
    train_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=train_predictions)

    eval_predictions = predict_with_model(model, eval_df)
    eval_predictions.attrs["feature_columns"] = list(model.feature_columns)
    eval_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=eval_predictions)

    coefficients_df = fit_full_model_coefficients(train_df, feature_columns=feature_columns)
    coefficients_df["feature_set_name"] = feature_set_name
    coefficients_df["split_type"] = split_type
    coefficients_df["model_type"] = search.model_type
    coefficients_df["alpha"] = float(search.alpha)
    coefficients_df["l1_wt"] = float(search.l1_wt)
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df["selected"] = (coefficients_df["feature_name"] != "const") & (coefficients_df["abs_coefficient"] > 1e-8)

    return search, train_result.__dict__, eval_result.__dict__, train_predictions, eval_predictions


def _best_metric_row(metrics_df: pd.DataFrame, *, feature_set_name: str, split_type: str, eval_split: str) -> dict | None:
    subset = metrics_df[
        (metrics_df["feature_set_name"] == feature_set_name)
        & (metrics_df["split_type"] == split_type)
        & (metrics_df["eval_split"] == eval_split)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False])
    return subset.iloc[0].to_dict()


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_single_head_recheck", level=args.log_level)

    output_dir = prepare_output_dir(Path(args.output_dir), run_name="single_head_recheck")

    dataset = pd.read_csv(args.dataset_file)
    metadata = load_probe_sample_metadata([args.train_sample_file, args.generalization_sample_file])
    dataset = dataset.merge(metadata, on="sample_id", how="left")

    train_df = dataset[dataset["eval_split"] == "train"].copy().reset_index(drop=True)
    generalization_df = dataset[dataset["eval_split"] == "generalization"].copy().reset_index(drop=True)
    train_subjects = set(train_df["subject"].dropna().astype(str))
    subject_unseen_df = generalization_df[~generalization_df["subject"].astype(str).isin(train_subjects)].copy().reset_index(drop=True)

    output_features = [column for column in _output_feature_columns() if column in dataset.columns]
    available_columns = set(dataset.columns)

    feature_sets: dict[str, list[str]] = {"runtime_output_only_safe": output_features}
    for layer_index, head_index in CANDIDATE_HEADS:
        head_columns = _head_feature_columns(layer_index, head_index, available_columns)
        feature_sets[f"runtime_output_plus_head_L{layer_index}H{head_index}"] = list(dict.fromkeys(output_features + head_columns))

    candidate_configs = _candidate_configs()
    metrics_rows: list[dict] = []
    budget_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    best_eval_rows: dict[str, dict] = {}
    best_heads_by_split: dict[str, dict] = {}

    split_specs = [
        ("random_generalization", generalization_df),
        ("subject_unseen_holdout", subject_unseen_df),
    ]

    for split_type, eval_df in split_specs:
        if eval_df.empty:
            continue
        for feature_set_name, feature_columns in feature_sets.items():
            search, train_result, eval_result, train_predictions, eval_predictions = _fit_and_score(
                train_df=train_df,
                eval_df=eval_df,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                split_type=split_type,
                cv_folds=args.cv_folds,
                seed=args.seed,
                candidate_configs=candidate_configs,
            )

            metrics_rows.append(
                _result_row(
                    type("Obj", (), train_result),
                    split_type=split_type,
                    eval_split="train_cv",
                    model_type=search.model_type,
                    alpha=search.alpha,
                    l1_wt=search.l1_wt,
                    num_features=len(feature_columns),
                    num_nonzero_features=search.num_nonzero_features,
                )
            )
            metrics_rows.append(
                _result_row(
                    type("Obj", (), eval_result),
                    split_type=split_type,
                    eval_split="heldout",
                    model_type=search.model_type,
                    alpha=search.alpha,
                    l1_wt=search.l1_wt,
                    num_features=len(feature_columns),
                    num_nonzero_features=search.num_nonzero_features,
                )
            )

            budget_rows.extend(
                budget_utility_rows_extended(
                    predictions_df=eval_predictions,
                    feature_set_name=feature_set_name,
                    eval_split="heldout",
                    split_type=split_type,
                    trigger_budgets=(0.01, 0.02, 0.03, 0.05),
                    topk_budgets=(1, 3, 5),
                )
            )

            coef_df = fit_full_model_coefficients(train_df, feature_columns=feature_columns)
            coef_df["feature_set_name"] = feature_set_name
            coef_df["split_type"] = split_type
            coef_df["model_type"] = search.model_type
            coef_df["alpha"] = float(search.alpha)
            coef_df["l1_wt"] = float(search.l1_wt)
            coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
            coef_df["selected"] = (coef_df["feature_name"] != "const") & (coef_df["abs_coefficient"] > 1e-8)
            coefficient_frames.append(coef_df)

            if feature_set_name != "runtime_output_only_safe":
                current = best_heads_by_split.get(split_type)
                row = {
                    "feature_set_name": feature_set_name,
                    "roc_auc": eval_result["roc_auc"],
                    "average_precision": eval_result["average_precision"],
                    "f1": eval_result["f1"],
                    "model_type": search.model_type,
                    "alpha": search.alpha,
                    "l1_wt": search.l1_wt,
                }
                if current is None or (
                    (row["roc_auc"] or -999.0),
                    (row["average_precision"] or -999.0),
                    row["f1"],
                ) > (
                    (current["roc_auc"] or -999.0),
                    (current["average_precision"] or -999.0),
                    current["f1"],
                ):
                    best_heads_by_split[split_type] = row

    metrics_df = pd.DataFrame(metrics_rows)
    budget_df = pd.DataFrame(budget_rows)
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()

    metrics_df.to_csv(output_dir / "single_head_metrics.csv", index=False)
    budget_df.to_csv(output_dir / "single_head_budget_metrics.csv", index=False)
    if not coefficients_df.empty:
        coefficients_df.to_csv(output_dir / "selected_feature_coefficients.csv", index=False)

    random_output = _best_metric_row(metrics_df, feature_set_name="runtime_output_only_safe", split_type="random_generalization", eval_split="heldout")
    subject_output = _best_metric_row(metrics_df, feature_set_name="runtime_output_only_safe", split_type="subject_unseen_holdout", eval_split="heldout")
    random_best_head = best_heads_by_split.get("random_generalization")
    subject_best_head = best_heads_by_split.get("subject_unseen_holdout")

    delta_random = None
    if random_output and random_best_head:
        delta_random = {
            "roc_auc_delta": (random_best_head["roc_auc"] - random_output["roc_auc"]) if random_best_head["roc_auc"] is not None and random_output["roc_auc"] is not None else None,
            "average_precision_delta": (random_best_head["average_precision"] - random_output["average_precision"]) if random_best_head["average_precision"] is not None and random_output["average_precision"] is not None else None,
            "f1_delta": random_best_head["f1"] - random_output["f1"],
        }
    delta_subject = None
    if subject_output and subject_best_head:
        delta_subject = {
            "roc_auc_delta": (subject_best_head["roc_auc"] - subject_output["roc_auc"]) if subject_best_head["roc_auc"] is not None and subject_output["roc_auc"] is not None else None,
            "average_precision_delta": (subject_best_head["average_precision"] - subject_output["average_precision"]) if subject_best_head["average_precision"] is not None and subject_output["average_precision"] is not None else None,
            "f1_delta": subject_best_head["f1"] - subject_output["f1"],
        }

    budget_summary: dict[str, dict] = {}
    for split_type in ("random_generalization", "subject_unseen_holdout"):
        split_budget = budget_df[budget_df["split_type"] == split_type].copy()
        if split_budget.empty:
            continue
        summary_rows: dict[str, dict] = {}
        for budget_label in ("1%", "2%", "3%", "5%", "top-1", "top-3", "top-5"):
            subset = split_budget[split_budget["trigger_budget"] == budget_label].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(
                ["precision_at_budget", "capture_rate", "lift_over_random"],
                ascending=[False, False, False],
            )
            summary_rows[budget_label] = subset.iloc[0].to_dict()
        budget_summary[split_type] = summary_rows

    summary_payload = {
        "dataset_summary": {
            "train_num_samples": int(len(train_df)),
            "train_num_positive": int(train_df[TARGET_LABEL].sum()),
            "random_generalization_num_samples": int(len(generalization_df)),
            "random_generalization_num_positive": int(generalization_df[TARGET_LABEL].sum()),
            "subject_unseen_holdout_num_samples": int(len(subject_unseen_df)),
            "subject_unseen_holdout_num_positive": int(subject_unseen_df[TARGET_LABEL].sum()),
            "train_subject_count": int(train_df["subject"].nunique(dropna=True)),
            "generalization_subject_count": int(generalization_df["subject"].nunique(dropna=True)),
            "subject_unseen_holdout_subject_count": int(subject_unseen_df["subject"].nunique(dropna=True)),
        },
        "candidate_heads": [f"L{layer}H{head}" for layer, head in CANDIDATE_HEADS],
        "output_only_random_generalization": random_output,
        "best_single_head_random_generalization": random_best_head,
        "best_single_head_random_generalization_delta": delta_random,
        "output_only_subject_unseen_holdout": subject_output,
        "best_single_head_subject_unseen_holdout": subject_best_head,
        "best_single_head_subject_unseen_holdout_delta": delta_subject,
        "budget_summary": budget_summary,
        "conclusion_flags": {
            "single_head_improves_random_generalization_auc": bool(delta_random and delta_random.get("roc_auc_delta", 0.0) > 0.0),
            "single_head_improves_subject_unseen_auc": bool(delta_subject and delta_subject.get("roc_auc_delta", 0.0) > 0.0),
            "single_head_improves_any_fine_budget_capture": bool(
                any(
                    (
                        item.get("feature_set_name") != "runtime_output_only_safe"
                        and item.get("capture_rate", 0.0)
                        > budget_df[
                            (budget_df["split_type"] == split_type)
                            & (budget_df["trigger_budget"] == budget_label)
                            & (budget_df["feature_set_name"] == "runtime_output_only_safe")
                        ]["capture_rate"].max()
                    )
                    for split_type, rows in budget_summary.items()
                    for budget_label, item in rows.items()
                )
            ),
        },
    }
    save_json(output_dir / "single_head_validation_summary.json", summary_payload)

    report_lines = [
        "# Single-Head Recheck Summary",
        "",
        f"- Output-only random held-out ROC-AUC: `{random_output['roc_auc']:.4f}`" if random_output and random_output["roc_auc"] is not None else "- Output-only random held-out ROC-AUC: `n/a`",
        f"- Best single head on random held-out: `{random_best_head['feature_set_name']}` with ROC-AUC `{random_best_head['roc_auc']:.4f}`" if random_best_head and random_best_head["roc_auc"] is not None else "- Best single head on random held-out: `n/a`",
        f"- Output-only subject-unseen ROC-AUC: `{subject_output['roc_auc']:.4f}`" if subject_output and subject_output["roc_auc"] is not None else "- Output-only subject-unseen ROC-AUC: `n/a`",
        f"- Best single head on subject-unseen held-out: `{subject_best_head['feature_set_name']}` with ROC-AUC `{subject_best_head['roc_auc']:.4f}`" if subject_best_head and subject_best_head["roc_auc"] is not None else "- Best single head on subject-unseen held-out: `n/a`",
        "",
        "## Budget Check",
        "",
    ]
    for split_type in ("random_generalization", "subject_unseen_holdout"):
        if split_type not in budget_summary:
            continue
        report_lines.append(f"- `{split_type}`")
        for budget_label in ("1%", "2%", "3%", "5%"):
            item = budget_summary[split_type].get(budget_label)
            if not item:
                continue
            report_lines.append(
                f"  - `{budget_label}` best: `{item['feature_set_name']}` | capture `{item['capture_rate']:.4f}` | precision `{item['precision_at_budget']:.4f}`"
            )
    report_lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- This run only rechecks output-only versus a handful of fixed single-head augmentations.",
            "- The question is whether single-head gain survives finer trigger budgets and a subject-unseen harder split.",
        ]
    )
    (output_dir / "single_head_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
