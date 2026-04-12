#!/usr/bin/env python3
"""Build, train, and score a lightweight local interference detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.mitigation import (
    DEFAULT_DATASET_SUMMARY_NAME,
    DEFAULT_FULL_DATASET_NAME,
    DESIGN_VERSION_CHOICES,
    build_interference_dataset,
    evaluate_predictions,
    load_detector,
    save_detector,
)
from src.mitigation.interference_models import (
    EmbeddingLogRegDetector,
    HybridSentenceStructuredLogRegDetector,
    SentenceEmbeddingLogRegDetector,
    StructuredLogisticDetector,
    TextNGramNBDetector,
    TextTfidfLogRegDetector,
    select_operating_points,
    threshold_sweep,
)


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"Dataset file is empty: {path}")
        return df
    raise ValueError(f"Unsupported dataset format: {path}")


def _train_detector(
    dataset: pd.DataFrame,
    model_kind: str,
    label_column: str,
    embedding_model: str = SentenceEmbeddingLogRegDetector.DEFAULT_MODEL_NAME,
    embedding_batch_size: int = 32,
):
    if model_kind in {"structured", "structured-safe"}:
        detector = StructuredLogisticDetector(feature_mode=StructuredLogisticDetector.FEATURE_MODE_SAFE)
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "structured-oracle":
        detector = StructuredLogisticDetector(feature_mode=StructuredLogisticDetector.FEATURE_MODE_ORACLE)
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "text":
        detector = TextNGramNBDetector()
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "text-tfidf-logreg":
        detector = TextTfidfLogRegDetector()
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "embedding-logreg":
        detector = EmbeddingLogRegDetector()
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "sentence-embedding-logreg":
        detector = SentenceEmbeddingLogRegDetector(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
        )
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "hybrid-sentence-structured-logreg":
        detector = HybridSentenceStructuredLogRegDetector(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
        )
        detector.fit(dataset, label_column=label_column)
        return detector
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _sample_train_split(
    train_df: pd.DataFrame,
    label_column: str,
    ratio: float,
    random_state: int,
    balance_classes: bool,
    negative_sampling: str,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    positives = train_df[train_df[label_column].astype(int) == 1].copy()
    negatives = train_df[train_df[label_column].astype(int) == 0].copy()
    hard_negative_series = (
        pd.to_numeric(negatives["is_hard_negative"], errors="coerce").fillna(0).astype(int)
        if "is_hard_negative" in negatives.columns
        else pd.Series(0, index=negatives.index, dtype=int)
    )
    hard_negatives = negatives[hard_negative_series == 1].copy()
    easy_negatives = negatives[hard_negative_series != 1].copy()
    summary = {
        "raw_train_rows": int(len(train_df)),
        "raw_positive_rows": int(len(positives)),
        "raw_negative_rows": int(len(negatives)),
        "raw_hard_negative_rows": int(len(hard_negatives)),
        "sampled_positive_rows": int(len(positives)),
        "sampled_negative_rows": int(len(negatives)),
        "sampled_hard_negative_rows": int(len(hard_negatives)),
        "sampled_random_negative_rows": int(len(easy_negatives)),
    }
    if positives.empty or negatives.empty or not balance_classes:
        return train_df.reset_index(drop=True), summary
    max_negatives = int(max(len(positives), round(len(positives) * ratio)))
    selected_hard = hard_negatives.iloc[0:0].copy()
    selected_random = easy_negatives.iloc[0:0].copy()
    if negative_sampling == "hard_only":
        if hard_negatives.empty:
            raise ValueError("negative-sampling=hard_only requested, but no hard negatives are available.")
        take = min(max_negatives, len(hard_negatives))
        selected_hard = hard_negatives.sample(n=take, random_state=random_state) if len(hard_negatives) > take else hard_negatives
    elif negative_sampling == "hard_first":
        take_hard = min(max_negatives, len(hard_negatives))
        selected_hard = hard_negatives.sample(n=take_hard, random_state=random_state) if len(hard_negatives) > take_hard else hard_negatives
        remaining = max_negatives - len(selected_hard)
        if remaining > 0 and not easy_negatives.empty:
            selected_random = easy_negatives.sample(n=min(remaining, len(easy_negatives)), random_state=random_state)
    else:
        if len(negatives) > max_negatives:
            negatives = negatives.sample(n=max_negatives, random_state=random_state)
        sampled_hard_series = (
            pd.to_numeric(negatives["is_hard_negative"], errors="coerce").fillna(0).astype(int)
            if "is_hard_negative" in negatives.columns
            else pd.Series(0, index=negatives.index, dtype=int)
        )
        selected_hard = negatives[sampled_hard_series == 1].copy()
        selected_random = negatives[sampled_hard_series != 1].copy()
    selected_negatives = pd.concat([selected_hard, selected_random], axis=0)
    if selected_negatives.empty:
        selected_negatives = negatives
    balanced = pd.concat([positives, selected_negatives], axis=0).sample(frac=1.0, random_state=random_state)
    summary.update(
        {
            "sampled_positive_rows": int(len(positives)),
            "sampled_negative_rows": int(len(selected_negatives)),
            "sampled_hard_negative_rows": int(len(selected_hard)),
            "sampled_random_negative_rows": int(len(selected_random)),
        }
    )
    return balanced.reset_index(drop=True), summary


def _evaluate_splits(df: pd.DataFrame, detector, label_column: str) -> dict:
    metrics = {}
    for split in ("train", "dev", "test"):
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        y_prob = detector.predict_proba(split_df)
        metrics[split] = evaluate_predictions(split_df[label_column].astype(int).tolist(), y_prob.tolist())
        metrics[split]["n"] = int(len(split_df))
    return metrics


def _select_best_threshold(df: pd.DataFrame, detector, label_column: str) -> float:
    dev_df = df[df["split"] == "dev"].copy()
    if dev_df.empty:
        return 0.5
    y_true = dev_df[label_column].astype(int).tolist()
    y_prob = detector.predict_proba(dev_df).tolist()
    best_threshold = 0.5
    best_f1 = -1.0
    for raw in range(10, 95, 5):
        threshold = raw / 100.0
        metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
    return float(best_threshold)


def _threshold_grid() -> List[float]:
    return [round(raw / 100.0, 2) for raw in range(5, 100, 5)]


def _select_eval_split(df: pd.DataFrame) -> str:
    for split in ("dev", "test", "train"):
        if not df[df["split"] == split].empty:
            return split
    return "train"


def _compute_threshold_artifacts(
    df: pd.DataFrame,
    detector,
    label_column: str,
    target_trigger_rate: float,
    max_trigger_rate: float,
) -> tuple[pd.DataFrame, Dict[str, float], str]:
    eval_split = _select_eval_split(df)
    eval_df = df[df["split"] == eval_split].copy()
    if eval_df.empty:
        raise ValueError("No non-empty split available for threshold sweep.")
    sweep_df = threshold_sweep(
        eval_df[label_column].astype(int).tolist(),
        detector.predict_proba(eval_df).tolist(),
        thresholds=_threshold_grid(),
    )
    operating_points = select_operating_points(
        sweep_df,
        target_trigger_rate=target_trigger_rate,
        max_trigger_rate=max_trigger_rate,
    )
    return sweep_df, operating_points, eval_split


def _comparison_rows_for_thresholds(
    dataset: pd.DataFrame,
    detector,
    label_column: str,
    model_kind: str,
    threshold_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    test_df = dataset[dataset["split"] == "test"].copy()
    if test_df.empty:
        test_df = dataset[dataset["split"] == _select_eval_split(dataset)].copy()
    y_true = test_df[label_column].astype(int).tolist()
    y_prob = detector.predict_proba(test_df).tolist()
    rows: List[Dict[str, Any]] = []
    for threshold_name, threshold in threshold_map.items():
        metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
        rows.append(
            {
                "model_kind": model_kind,
                "label_mode": label_column.replace("_label", ""),
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                "test_precision": float(metrics["precision"]),
                "test_recall": float(metrics["recall"]),
                "test_f1": float(metrics["f1"]),
                "test_roc_auc": float(metrics["roc_auc"]),
                "trigger_rate": float((pd.Series(y_prob) >= float(threshold)).mean()) if y_prob else 0.0,
            }
        )
    return rows


def _guard_metrics_for_threshold(
    dataset: pd.DataFrame,
    threshold: float,
    threshold_name: str,
    recheck_source: str,
    model_kind: str,
) -> Dict[str, Any]:
    working = dataset.copy()
    working["raw_answer"] = working["predicted_answer"].fillna("").astype(str).str.strip().str.upper()
    working["recheck_answer"] = working[recheck_source].fillna("").astype(str).str.strip().str.upper()
    working["trigger_recheck"] = (
        working["interference_score"].astype(float) >= float(threshold)
    ).astype(int)
    working["final_answer"] = working["raw_answer"]
    trigger_mask = working["trigger_recheck"].astype(int) == 1
    valid_recheck_mask = trigger_mask & working["recheck_answer"].astype(bool)
    working.loc[valid_recheck_mask, "final_answer"] = working.loc[valid_recheck_mask, "recheck_answer"]
    working["raw_correct"] = (
        working["raw_answer"].astype(str).str.upper() == working["ground_truth"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["final_correct"] = (
        working["final_answer"].astype(str).str.upper() == working["ground_truth"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["raw_wrong_follow"] = (
        working["raw_answer"].astype(str).str.upper() == working["wrong_option"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["final_wrong_follow"] = (
        working["final_answer"].astype(str).str.upper() == working["wrong_option"].fillna("").astype(str).str.upper()
    ).astype(int)
    w1_rows = working[pd.to_numeric(working.get("explicit_wrong_option", 0), errors="coerce").fillna(0).astype(int) == 1]
    return {
        "model_kind": model_kind,
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "raw_accuracy": float(working["raw_correct"].mean()),
        "guarded_accuracy": float(working["final_correct"].mean()),
        "raw_wrong_option_follow_rate": float(working["raw_wrong_follow"].mean()),
        "guarded_wrong_option_follow_rate": float(working["final_wrong_follow"].mean()),
        "trigger_rate": float(working["trigger_recheck"].mean()),
        "avg_extra_calls": float(working["trigger_recheck"].mean()),
        "strict_positive_recall": float(
            working.loc[working["strict_label"].fillna(-1).astype(int) == 1, "trigger_recheck"].mean()
        )
        if "strict_label" in working.columns and (working["strict_label"].fillna(-1).astype(int) == 1).any()
        else 0.0,
        "strict_negative_false_positive_rate": float(
            working.loc[working["strict_label"].fillna(-1).astype(int) == 0, "trigger_recheck"].mean()
        )
        if "strict_label" in working.columns and (working["strict_label"].fillna(-1).astype(int) == 0).any()
        else 0.0,
        "w1_trigger_rate": float(w1_rows["trigger_recheck"].mean()) if not w1_rows.empty else 0.0,
    }


def build_dataset_command(args: argparse.Namespace) -> int:
    summary = build_interference_dataset(
        output_dir=Path(args.output_dir),
        design_version=args.design_version,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    dataset = _read_dataset(Path(args.dataset))
    label_column = f"{args.label_mode}_label"
    if label_column not in dataset.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset: {args.dataset}")
    dataset = dataset[dataset[label_column].notna()].copy()
    if dataset.empty:
        raise ValueError(f"No labeled rows found for {label_column} in {args.dataset}")

    train_df = dataset[dataset["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Training split is empty.")
    train_df, sampling_summary = _sample_train_split(
        train_df,
        label_column=label_column,
        ratio=float(args.negative_positive_ratio),
        random_state=int(args.random_state),
        balance_classes=bool(args.balance_classes),
        negative_sampling=str(args.negative_sampling),
    )

    detector = _train_detector(
        train_df,
        model_kind=args.model_kind,
        label_column=label_column,
        embedding_model=str(args.embedding_model),
        embedding_batch_size=int(args.embedding_batch_size),
    )
    sweep_df, operating_points, threshold_eval_split = _compute_threshold_artifacts(
        dataset,
        detector,
        label_column=label_column,
        target_trigger_rate=float(args.target_trigger_rate),
        max_trigger_rate=float(args.max_trigger_rate),
    )
    tuned_threshold = float(operating_points["best_f1_threshold"])
    metrics = {}
    for split in ("train", "dev", "test"):
        split_df = dataset[dataset["split"] == split].copy()
        if split_df.empty:
            continue
        y_prob = detector.predict_proba(split_df)
        metrics[split] = evaluate_predictions(
            split_df[label_column].astype(int).tolist(),
            y_prob.tolist(),
            threshold=tuned_threshold,
        )
        metrics[split]["n"] = int(len(split_df))
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_sweep_path = output_path.parent / f"threshold_sweep_{args.model_kind}_{args.label_mode}.csv"
    sweep_df.to_csv(threshold_sweep_path, index=False)
    threshold_map = {
        "artifact_default": tuned_threshold,
        "best_f1": float(operating_points["best_f1_threshold"]),
        "high_precision": float(operating_points["high_precision_threshold"]),
        "high_recall": float(operating_points["high_recall_threshold"]),
    }
    if "aggressive_threshold" in operating_points:
        threshold_map["aggressive"] = float(operating_points["aggressive_threshold"])
    if "matched_trigger_budget_threshold" in operating_points:
        threshold_map["matched_trigger_budget"] = float(operating_points["matched_trigger_budget_threshold"])
    if "recall_constrained_threshold" in operating_points:
        threshold_map["recall_constrained"] = float(operating_points["recall_constrained_threshold"])
    comparison_rows = _comparison_rows_for_thresholds(
        dataset=dataset,
        detector=detector,
        label_column=label_column,
        model_kind=args.model_kind,
        threshold_map=threshold_map,
    )
    comparison_path = output_path.parent / "detector_model_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    metadata = {
        "model_kind": args.model_kind,
        "structured_feature_mode": (
            "oracle_upper_bound"
            if args.model_kind == "structured-oracle"
            else ("safe" if args.model_kind in {"structured", "structured-safe"} else None)
        ),
        "label_mode": args.label_mode,
        "label_column": label_column,
        "dataset_path": str(Path(args.dataset).resolve()),
        "train_rows_used": int(len(train_df)),
        "recommended_threshold": tuned_threshold,
        "negative_sampling": args.negative_sampling,
        "sampling_summary": sampling_summary,
        "threshold_eval_split": threshold_eval_split,
        "operating_points": operating_points,
        "target_trigger_rate": float(args.target_trigger_rate),
        "max_trigger_rate": float(args.max_trigger_rate),
        "threshold_sweep_csv": str(threshold_sweep_path.resolve()),
        "comparison_csv": str(comparison_path.resolve()),
        "metrics": metrics,
    }
    save_detector(detector, output_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def score_command(args: argparse.Namespace) -> int:
    detector, metadata = load_detector(Path(args.model_path))
    dataset = _read_dataset(Path(args.dataset))
    dataset = dataset.copy()
    if dataset.empty:
        raise ValueError(f"Input dataset is empty: {args.dataset}")
    y_prob = detector.predict_proba(dataset)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata.get("recommended_threshold", 0.5))
    )
    dataset["interference_score"] = y_prob
    dataset["predicted_label"] = (dataset["interference_score"] >= threshold).astype(int)
    dataset["trigger_recheck"] = dataset["predicted_label"]

    preferred_columns = [
        "task_id",
        "model_name",
        "arm_id",
        "predicted_answer",
        "baseline_answer",
        "wrong_option",
        "strict_label",
        "relaxed_label",
        "interference_score",
        "predicted_label",
        "trigger_recheck",
    ]
    remaining_columns = [column for column in dataset.columns if column not in preferred_columns]
    dataset = dataset[[column for column in preferred_columns if column in dataset.columns] + remaining_columns]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_path": str(Path(args.dataset).resolve()),
        "output_file": str(output_path.resolve()),
        "threshold": threshold,
        "trigger_rate": float(dataset["trigger_recheck"].mean()) if len(dataset) else 0.0,
        "num_rows": int(len(dataset)),
        "training_metadata": metadata,
    }
    label_column = metadata.get("label_column")
    if label_column and label_column in dataset.columns and dataset[label_column].notna().any():
        labeled = dataset[dataset[label_column].notna()].copy()
        summary["labeled_eval"] = evaluate_predictions(
            labeled[label_column].astype(int).tolist(),
            labeled["interference_score"].astype(float).tolist(),
            threshold=threshold,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def guard_eval_command(args: argparse.Namespace) -> int:
    dataset = _read_dataset(Path(args.dataset))
    if dataset.empty:
        raise ValueError(f"Input dataset is empty: {args.dataset}")
    required_columns = {
        "predicted_answer",
        "ground_truth",
        "wrong_option",
        "trigger_recheck",
        "baseline_answer",
        "strict_label",
    }
    missing = sorted(column for column in required_columns if column not in dataset.columns)
    if missing:
        raise ValueError(
            "guard-eval requires a scored dataset with columns: "
            + ", ".join(missing)
        )

    recheck_source = str(args.recheck_source)
    if recheck_source not in dataset.columns:
        raise ValueError(f"Requested recheck source column not found: {recheck_source}")
    if "interference_score" not in dataset.columns:
        raise ValueError("guard-eval requires a scored dataset that contains interference_score.")

    model_metadata: Dict[str, Any] = {}
    if args.model_path:
        _, model_metadata = load_detector(Path(args.model_path))
    threshold_map: Dict[str, float] = {}
    if args.compare_thresholds:
        if model_metadata:
            threshold_map["artifact_default"] = float(model_metadata.get("recommended_threshold", 0.5))
            operating_points = model_metadata.get("operating_points", {})
            for name in ("best_f1", "high_precision", "high_recall"):
                key = f"{name}_threshold"
                if key in operating_points:
                    threshold_map[name] = float(operating_points[key])
            if "aggressive_threshold" in operating_points:
                threshold_map["aggressive"] = float(operating_points["aggressive_threshold"])
            if "matched_trigger_budget_threshold" in operating_points:
                threshold_map["matched_trigger_budget"] = float(operating_points["matched_trigger_budget_threshold"])
            if "recall_constrained_threshold" in operating_points:
                threshold_map["recall_constrained"] = float(operating_points["recall_constrained_threshold"])
        if not threshold_map:
            threshold_map["artifact_default"] = 0.5
    else:
        threshold_value = (
            float(args.threshold)
            if args.threshold is not None
            else float(model_metadata.get("recommended_threshold", 0.5) if model_metadata else 0.5)
        )
        threshold_map[str(args.threshold_name)] = threshold_value

    comparison_rows = [
        _guard_metrics_for_threshold(
            dataset=dataset,
            threshold=threshold,
            threshold_name=threshold_name,
            recheck_source=recheck_source,
            model_kind=str(model_metadata.get("model_kind", args.model_kind or "unknown")),
        )
        for threshold_name, threshold in threshold_map.items()
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    if args.output_comparison_file:
        comparison_path = Path(args.output_comparison_file)
    elif args.output_file:
        comparison_path = Path(args.output_file).with_name("guard_eval_comparison.csv")
    else:
        comparison_path = Path(args.dataset).with_name("guard_eval_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "mode": "offline_selective_recheck_simulation",
        "dataset_path": str(Path(args.dataset).resolve()),
        "recheck_source": recheck_source,
        "num_rows": int(len(dataset)),
        "comparison_csv": str(comparison_path.resolve()),
        "comparisons": comparison_rows,
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight local interference detector")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-dataset", help="Build detector dataset from existing objective outputs")
    build_cmd.add_argument(
        "--output-dir",
        default="outputs/experiments/interference_detector",
        help="Directory to store detector dataset artifacts",
    )
    build_cmd.add_argument(
        "--design-version",
        default="new15",
        choices=DESIGN_VERSION_CHOICES,
        help="Which objective design family to include in the detector dataset.",
    )
    build_cmd.set_defaults(func=build_dataset_command)

    train_cmd = subparsers.add_parser("train", help="Train a minimal detector model")
    train_cmd.add_argument("--dataset", required=True, help="CSV or JSONL detector dataset")
    train_cmd.add_argument("--label-mode", default="strict", choices=["strict", "relaxed"])
    train_cmd.add_argument(
        "--model-kind",
        default="structured-safe",
        choices=[
            "structured",
            "structured-safe",
            "structured-oracle",
            "text",
            "text-tfidf-logreg",
            "embedding-logreg",
            "sentence-embedding-logreg",
            "hybrid-sentence-structured-logreg",
        ],
        help="Detector family to train. 'structured-safe' is the recommended lightweight baseline.",
    )
    train_cmd.add_argument("--balance-classes", action="store_true", default=True)
    train_cmd.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    train_cmd.add_argument("--negative-positive-ratio", type=float, default=3.0)
    train_cmd.add_argument(
        "--negative-sampling",
        default="hard_first",
        choices=["random", "hard_first", "hard_only"],
        help="How to prioritize negatives under class imbalance.",
    )
    train_cmd.add_argument("--random-state", type=int, default=42)
    train_cmd.add_argument(
        "--embedding-model",
        default=SentenceEmbeddingLogRegDetector.DEFAULT_MODEL_NAME,
        help="Sentence-transformers model name for sentence-embedding-logreg.",
    )
    train_cmd.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Encoding batch size for sentence-embedding-logreg.",
    )
    train_cmd.add_argument(
        "--target-trigger-rate",
        type=float,
        default=0.075,
        help="Dev-set trigger-rate target for matched-trigger-budget threshold calibration.",
    )
    train_cmd.add_argument(
        "--max-trigger-rate",
        type=float,
        default=0.08,
        help="Dev-set maximum trigger rate for recall-constrained threshold calibration.",
    )
    train_cmd.add_argument(
        "--output-model",
        default="outputs/experiments/interference_detector/detector.pkl",
        help="Where to store the serialized detector",
    )
    train_cmd.set_defaults(func=train_command)

    score_cmd = subparsers.add_parser("score", help="Score a detector dataset and emit warnings / triggers")
    score_cmd.add_argument("--model-path", required=True)
    score_cmd.add_argument("--dataset", required=True)
    score_cmd.add_argument(
        "--output-file",
        default="outputs/experiments/interference_detector/scored_samples.csv",
        help="Where to write scored samples",
    )
    score_cmd.add_argument("--threshold", type=float, default=None)
    score_cmd.set_defaults(func=score_command)

    guard_cmd = subparsers.add_parser(
        "guard-eval",
        help="Offline selective re-check simulation over a scored detector dataset.",
    )
    guard_cmd.add_argument("--dataset", required=True, help="Scored CSV from the score command.")
    guard_cmd.add_argument(
        "--recheck-source",
        default="baseline_answer",
        choices=["baseline_answer", "control_reference_answer"],
        help="Which existing answer source to use as the offline re-check proxy.",
    )
    guard_cmd.add_argument(
        "--output-file",
        default="",
        help="Optional JSON path to save the guard evaluation summary.",
    )
    guard_cmd.add_argument("--model-path", default="", help="Optional detector artifact to load threshold metadata from.")
    guard_cmd.add_argument("--threshold", type=float, default=None, help="Explicit threshold for single-threshold guard evaluation.")
    guard_cmd.add_argument("--threshold-name", default="explicit", help="Name for the explicitly provided threshold.")
    guard_cmd.add_argument(
        "--compare-thresholds",
        action="store_true",
        help="Compare artifact_default / best_f1 / high_precision / high_recall thresholds when available.",
    )
    guard_cmd.add_argument(
        "--output-comparison-file",
        default="",
        help="Optional CSV path for per-threshold guard evaluation comparison rows.",
    )
    guard_cmd.add_argument(
        "--model-kind",
        default="unknown",
        help="Fallback model kind label when no detector artifact metadata is provided.",
    )
    guard_cmd.set_defaults(func=guard_eval_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(name="interference_detector", level=args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
