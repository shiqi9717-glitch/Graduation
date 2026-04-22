#!/usr/bin/env python3
"""Sparse runtime-safe monitor experiments: output-first, internal-as-incremental."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.internal_signal_predictor import (
    TARGET_LABEL,
    RegularizedSearchResult,
    budget_utility_rows,
    build_sparse_runtime_monitor_dataset,
    count_nonzero_coefficients,
    evaluate_predictions,
    fit_full_model_coefficients,
    load_jsonl,
    predict_with_model,
    runtime_safe_signal_feature_groups,
    select_best_regularized_model,
)
from src.open_model_probe.io_utils import prepare_output_dir, save_json


HEAD_PATTERN = re.compile(r"^head_L(?P<layer>\d+)H(?P<head>\d+)_(?P<kind>prefix_attention|wrong_prefix_attention)$")


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())


def _result_row(
    result,
    *,
    eval_split: str,
    model_type: str,
    alpha: float,
    l1_wt: float,
    num_features: int,
    num_nonzero_features: int,
) -> dict:
    return {
        "feature_set_name": result.feature_set_name,
        "model_type": model_type,
        "alpha": float(alpha),
        "l1_wt": float(l1_wt),
        "eval_split": eval_split,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse runtime monitor experiments on existing local probe outputs.")
    parser.add_argument("--train-probe-run-dir", required=True)
    parser.add_argument("--train-mechanistic-run-dir", required=True)
    parser.add_argument("--generalization-probe-run-dir", required=True)
    parser.add_argument("--generalization-mechanistic-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b_sparse_runtime_monitor")
    parser.add_argument("--focus-layers", default="31,32,33,34,35")
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
        ("elastic_net_logistic", 0.01, 0.5),
        ("elastic_net_logistic", 0.05, 0.5),
        ("elastic_net_logistic", 0.10, 0.5),
        ("elastic_net_logistic", 0.10, 0.8),
    ]


def _output_feature_columns() -> list[str]:
    return list(runtime_safe_signal_feature_groups()["runtime_safe_features"]["output_safe"])


def _sparse_internal_candidate_columns(focus_layers: tuple[int, ...]) -> list[str]:
    columns: list[str] = []
    for layer in focus_layers:
        columns.extend(
            [
                f"layer_{layer}_final_cosine_shift",
                f"layer_{layer}_pooled_cosine_shift",
                f"layer_{layer}_wrong_prefix_attention",
                f"layer_{layer}_lens_wrong_rank",
                f"layer_{layer}_final_norm_delta",
                f"layer_{layer}_pooled_norm_delta",
            ]
        )
    return columns


def _head_bundle_map(df: pd.DataFrame, focus_layers: tuple[int, ...]) -> dict[tuple[int, int], list[str]]:
    bundles: dict[tuple[int, int], list[str]] = {}
    layer_set = {int(layer) for layer in focus_layers}
    for column in df.columns:
        match = HEAD_PATTERN.match(str(column))
        if not match:
            continue
        layer = int(match.group("layer"))
        head = int(match.group("head"))
        if layer not in layer_set:
            continue
        bundles.setdefault((layer, head), []).append(column)
    return {key: sorted(value) for key, value in bundles.items()}


def _nonzero_coefficients_df(
    *,
    search: RegularizedSearchResult,
    coefficients_df: pd.DataFrame,
    feature_set_name: str,
) -> pd.DataFrame:
    out = coefficients_df.copy()
    out["feature_set_name"] = feature_set_name
    out["model_type"] = search.model_type
    out["alpha"] = float(search.alpha)
    out["l1_wt"] = float(search.l1_wt)
    out["abs_coefficient"] = out["coefficient"].abs()
    out["selected"] = (out["feature_name"] != "const") & (out["abs_coefficient"] > 1e-8)
    return out


def main() -> int:
    args = build_parser().parse_args()
    setup_logger(name="local_probe_sparse_runtime_monitor", level=args.log_level)

    focus_layers = _parse_layers(args.focus_layers)
    output_dir = prepare_output_dir(Path(args.output_dir), run_name="sparse_runtime_monitor")

    train_probe_run_payload = json.loads((Path(args.train_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))
    generalization_probe_run_payload = json.loads((Path(args.generalization_probe_run_dir) / "probe_run.json").read_text(encoding="utf-8"))

    train_dataset = build_sparse_runtime_monitor_dataset(
        probe_comparisons=load_jsonl(Path(args.train_probe_run_dir) / "probe_comparisons.jsonl"),
        sample_cases=load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl"),
        scenario_records=load_jsonl(Path(args.train_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl"),
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    generalization_dataset = build_sparse_runtime_monitor_dataset(
        probe_comparisons=load_jsonl(Path(args.generalization_probe_run_dir) / "probe_comparisons.jsonl"),
        sample_cases=load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_sample_cases.jsonl"),
        scenario_records=load_jsonl(Path(args.generalization_mechanistic_run_dir) / "mechanistic_scenario_records.jsonl"),
        focus_layers=focus_layers,
        baseline_correct_only=True,
    )
    train_dataset["eval_split"] = "train"
    generalization_dataset["eval_split"] = "generalization"
    combined_dataset = pd.concat([train_dataset, generalization_dataset], ignore_index=True)
    combined_dataset.to_csv(output_dir / "runtime_sparse_dataset.csv", index=False)

    output_features = [col for col in _output_feature_columns() if col in train_dataset.columns]
    sparse_internal_features = [col for col in _sparse_internal_candidate_columns(focus_layers) if col in train_dataset.columns]
    head_bundles = _head_bundle_map(train_dataset, focus_layers)

    metrics_rows: list[dict] = []
    budget_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    head_ranking_rows: list[dict] = []
    summary_feature_results: dict[str, dict] = {}

    candidate_configs = _candidate_configs()

    def evaluate_feature_set(feature_set_name: str, feature_columns: list[str]) -> tuple[RegularizedSearchResult, pd.DataFrame, pd.DataFrame]:
        search, model, train_predictions = select_best_regularized_model(
            train_dataset,
            feature_set_name=feature_set_name,
            feature_columns=feature_columns,
            candidate_configs=candidate_configs,
            n_splits=int(args.cv_folds),
            seed=int(args.seed),
        )
        train_predictions.attrs["feature_columns"] = list(model.feature_columns)
        train_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=train_predictions)
        metrics_rows.append(
            _result_row(
                train_result,
                eval_split="train_cv",
                model_type=search.model_type,
                alpha=search.alpha,
                l1_wt=search.l1_wt,
                num_features=len(feature_columns),
                num_nonzero_features=search.num_nonzero_features,
            )
        )
        budget_rows.extend(
            budget_utility_rows(
                predictions_df=train_predictions,
                feature_set_name=feature_set_name,
                eval_split="train_cv",
                trigger_budgets=(0.05, 0.10, 0.15),
            )
        )

        heldout_predictions = predict_with_model(model, generalization_dataset)
        heldout_predictions.attrs["feature_columns"] = list(model.feature_columns)
        heldout_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=heldout_predictions)
        metrics_rows.append(
            _result_row(
                heldout_result,
                eval_split="generalization",
                model_type=search.model_type,
                alpha=search.alpha,
                l1_wt=search.l1_wt,
                num_features=len(feature_columns),
                num_nonzero_features=search.num_nonzero_features,
            )
        )
        budget_rows.extend(
            budget_utility_rows(
                predictions_df=heldout_predictions,
                feature_set_name=feature_set_name,
                eval_split="generalization",
                trigger_budgets=(0.05, 0.10, 0.15),
            )
        )

        coef_df = fit_full_model_coefficients(train_dataset, feature_columns=feature_columns)
        coefficient_frames.append(_nonzero_coefficients_df(search=search, coefficients_df=coef_df, feature_set_name=feature_set_name))
        summary_feature_results[feature_set_name] = {
            "search": search,
            "train_result": train_result,
            "heldout_result": heldout_result,
        }
        return search, train_predictions, heldout_predictions

    # Output-only baseline.
    evaluate_feature_set("runtime_output_only_safe", output_features)

    # Output + sparse internal.
    evaluate_feature_set("runtime_output_plus_sparse_internal", list(dict.fromkeys(output_features + sparse_internal_features)))

    # Output + all sparse heads with regularization.
    all_head_features = sorted({feature for bundle in head_bundles.values() for feature in bundle})
    evaluate_feature_set("runtime_output_plus_sparse_heads", list(dict.fromkeys(output_features + all_head_features)))

    # Single-head ranking.
    for (layer_index, head_index), bundle_features in sorted(head_bundles.items()):
        feature_set_name = f"runtime_output_plus_head_L{layer_index}H{head_index}"
        search, _train_predictions, heldout_predictions = evaluate_feature_set(
            feature_set_name,
            list(dict.fromkeys(output_features + bundle_features)),
        )
        heldout_result = evaluate_predictions(feature_set_name=feature_set_name, predictions_df=heldout_predictions)
        head_ranking_rows.append(
            {
                "layer_index": int(layer_index),
                "head_index": int(head_index),
                "feature_name": "+".join(bundle_features),
                "train_score": search.cv_result.roc_auc,
                "heldout_score": heldout_result.roc_auc,
                "model_type": search.model_type,
                "alpha": float(search.alpha),
                "l1_wt": float(search.l1_wt),
                "selected_rank": None,
            }
        )

    head_ranking_df = pd.DataFrame(head_ranking_rows).sort_values(
        ["train_score", "heldout_score", "layer_index", "head_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    if not head_ranking_df.empty:
        head_ranking_df["selected_rank"] = head_ranking_df.index + 1
    head_ranking_df.to_csv(output_dir / "head_ranking.csv", index=False)

    # Top-k sparse head combinations.
    top_heads = [(int(row.layer_index), int(row.head_index)) for row in head_ranking_df.head(8).itertuples()]
    for k in (1, 2, 4, 8):
        selected = top_heads[: min(k, len(top_heads))]
        selected_features: list[str] = []
        for key in selected:
            selected_features.extend(head_bundles.get(key, []))
        evaluate_feature_set(
            f"runtime_output_plus_top{k}_heads",
            list(dict.fromkeys(output_features + selected_features)),
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["eval_split", "roc_auc", "average_precision", "f1"],
        ascending=[True, False, False, False],
    )
    budget_df = pd.DataFrame(budget_rows).sort_values(
        ["eval_split", "trigger_budget", "capture_rate", "precision_at_budget"],
        ascending=[True, True, False, False],
    )
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True)
    coefficients_df = coefficients_df[coefficients_df["selected"] | (coefficients_df["feature_name"] == "const")].copy()

    metrics_df.to_csv(output_dir / "sparse_runtime_metrics.csv", index=False)
    budget_df.to_csv(output_dir / "budget_utility_metrics.csv", index=False)
    coefficients_df.to_csv(output_dir / "selected_feature_coefficients.csv", index=False)

    heldout_metrics = metrics_df[metrics_df["eval_split"] == "generalization"].copy()
    heldout_budget = budget_df[budget_df["eval_split"] == "generalization"].copy()

    def _metric_lookup(feature_set_name: str, metric: str):
        row = heldout_metrics[heldout_metrics["feature_set_name"] == feature_set_name]
        if row.empty:
            return None
        value = row.iloc[0][metric]
        return None if pd.isna(value) else float(value)

    def _budget_lookup(feature_set_name: str, trigger_budget: float, metric: str):
        row = heldout_budget[
            (heldout_budget["feature_set_name"] == feature_set_name)
            & (heldout_budget["trigger_budget"].round(4) == round(float(trigger_budget), 4))
        ]
        if row.empty:
            return None
        value = row.iloc[0][metric]
        return None if pd.isna(value) else float(value)

    topk_feature_names = [name for name in heldout_metrics["feature_set_name"].tolist() if name.startswith("runtime_output_plus_top")]
    best_topk_row = (
        heldout_metrics[heldout_metrics["feature_set_name"].isin(topk_feature_names)]
        .sort_values(["roc_auc", "average_precision", "f1"], ascending=[False, False, False])
        .iloc[0]
        .to_dict()
        if topk_feature_names
        else None
    )

    summary = {
        "train_probe_run_dir": str(Path(args.train_probe_run_dir).resolve()),
        "train_mechanistic_run_dir": str(Path(args.train_mechanistic_run_dir).resolve()),
        "generalization_probe_run_dir": str(Path(args.generalization_probe_run_dir).resolve()),
        "generalization_mechanistic_run_dir": str(Path(args.generalization_mechanistic_run_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "focus_layers": list(focus_layers),
        "train_raw_num_samples": int(train_probe_run_payload.get("num_samples", len(train_dataset))),
        "train_num_samples": int(len(train_dataset)),
        "train_num_positive": int(train_dataset[TARGET_LABEL].sum()),
        "generalization_raw_num_samples": int(generalization_probe_run_payload.get("num_samples", len(generalization_dataset))),
        "generalization_num_samples": int(len(generalization_dataset)),
        "generalization_num_positive": int(generalization_dataset[TARGET_LABEL].sum()),
        "best_output_only_result": heldout_metrics[heldout_metrics["feature_set_name"] == "runtime_output_only_safe"].iloc[0].to_dict(),
        "best_output_plus_internal_result": heldout_metrics[heldout_metrics["feature_set_name"] == "runtime_output_plus_sparse_internal"].iloc[0].to_dict(),
        "best_output_plus_topk_head_result": best_topk_row,
        "head_ranking_top10": head_ranking_df.head(10).to_dict(orient="records"),
        "conclusions": {
            "output_plus_sparse_internal_vs_output_only_roc_auc_delta": (
                _metric_lookup("runtime_output_plus_sparse_internal", "roc_auc")
                - _metric_lookup("runtime_output_only_safe", "roc_auc")
            ),
            "output_plus_sparse_heads_vs_output_only_roc_auc_delta": (
                _metric_lookup("runtime_output_plus_sparse_heads", "roc_auc")
                - _metric_lookup("runtime_output_only_safe", "roc_auc")
            ),
            "best_topk_head_vs_output_only_roc_auc_delta": (
                None
                if best_topk_row is None
                else float(best_topk_row["roc_auc"]) - _metric_lookup("runtime_output_only_safe", "roc_auc")
            ),
            "budget_10_best_topk_capture_delta_vs_output_only": (
                None
                if best_topk_row is None
                else _budget_lookup(str(best_topk_row["feature_set_name"]), 0.10, "capture_rate")
                - _budget_lookup("runtime_output_only_safe", 0.10, "capture_rate")
            ),
            "budget_10_sparse_internal_capture_delta_vs_output_only": (
                _budget_lookup("runtime_output_plus_sparse_internal", 0.10, "capture_rate")
                - _budget_lookup("runtime_output_only_safe", 0.10, "capture_rate")
            ),
        },
        "files": {
            "sparse_runtime_metrics": str((output_dir / "sparse_runtime_metrics.csv").resolve()),
            "budget_utility_metrics": str((output_dir / "budget_utility_metrics.csv").resolve()),
            "head_ranking": str((output_dir / "head_ranking.csv").resolve()),
            "selected_feature_coefficients": str((output_dir / "selected_feature_coefficients.csv").resolve()),
            "runtime_sparse_dataset": str((output_dir / "runtime_sparse_dataset.csv").resolve()),
        },
    }
    save_json(output_dir / "sparse_runtime_summary.json", summary)

    summary_lines = [
        "# Sparse Runtime Monitor",
        "",
        f"- Train baseline-correct samples: {len(train_dataset)}",
        f"- Held-out baseline-correct samples: {len(generalization_dataset)}",
        "",
        "## Held-out Metrics",
        "",
        "```csv",
        heldout_metrics.to_csv(index=False).strip(),
        "```",
        "",
        "## Held-out Budget Utility",
        "",
        "```csv",
        heldout_budget.to_csv(index=False).strip(),
        "```",
        "",
        "## Head Ranking Top 10",
        "",
        "```csv",
        head_ranking_df.head(10).to_csv(index=False).strip(),
        "```",
        "",
    ]
    (output_dir / "sparse_runtime_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
