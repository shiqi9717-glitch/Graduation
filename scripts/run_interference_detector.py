#!/usr/bin/env python3
"""Build, train, and score a lightweight local interference detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.mitigation import (
    DEFAULT_DATASET_SUMMARY_NAME,
    DEFAULT_FULL_DATASET_NAME,
    build_interference_dataset,
    evaluate_predictions,
    load_detector,
    save_detector,
)
from src.mitigation.interference_models import StructuredLogisticDetector, TextNGramNBDetector


def _read_dataset(path: Path) -> pd.DataFrame:
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
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported dataset format: {path}")


def _train_detector(dataset: pd.DataFrame, model_kind: str, label_column: str):
    if model_kind == "structured":
        detector = StructuredLogisticDetector()
        detector.fit(dataset, label_column=label_column)
        return detector
    if model_kind == "text":
        detector = TextNGramNBDetector()
        detector.fit(dataset, label_column=label_column)
        return detector
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _balance_train_split(train_df: pd.DataFrame, label_column: str, ratio: float, random_state: int) -> pd.DataFrame:
    positives = train_df[train_df[label_column].astype(int) == 1].copy()
    negatives = train_df[train_df[label_column].astype(int) == 0].copy()
    if positives.empty or negatives.empty:
        return train_df
    max_negatives = int(max(len(positives), round(len(positives) * ratio)))
    if len(negatives) > max_negatives:
        negatives = negatives.sample(n=max_negatives, random_state=random_state)
    balanced = pd.concat([positives, negatives], axis=0).sample(frac=1.0, random_state=random_state)
    return balanced.reset_index(drop=True)


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


def build_dataset_command(args: argparse.Namespace) -> int:
    summary = build_interference_dataset(output_dir=Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    dataset = _read_dataset(Path(args.dataset))
    label_column = f"{args.label_mode}_label"
    dataset = dataset[dataset[label_column].notna()].copy()
    if dataset.empty:
        raise ValueError(f"No labeled rows found for {label_column} in {args.dataset}")

    train_df = dataset[dataset["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Training split is empty.")
    if args.balance_classes:
        train_df = _balance_train_split(
            train_df,
            label_column=label_column,
            ratio=float(args.negative_positive_ratio),
            random_state=int(args.random_state),
        )

    detector = _train_detector(train_df, model_kind=args.model_kind, label_column=label_column)
    tuned_threshold = _select_best_threshold(dataset, detector, label_column=label_column)
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
    metadata = {
        "model_kind": args.model_kind,
        "label_mode": args.label_mode,
        "label_column": label_column,
        "dataset_path": str(Path(args.dataset).resolve()),
        "train_rows_used": int(len(train_df)),
        "recommended_threshold": tuned_threshold,
        "metrics": metrics,
    }

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_detector(detector, output_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def score_command(args: argparse.Namespace) -> int:
    detector, metadata = load_detector(Path(args.model_path))
    dataset = _read_dataset(Path(args.dataset))
    dataset = dataset.copy()
    y_prob = detector.predict_proba(dataset)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata.get("recommended_threshold", 0.5))
    )
    dataset["interference_score"] = y_prob
    dataset["predicted_label"] = (dataset["interference_score"] >= threshold).astype(int)
    dataset["trigger_recheck"] = dataset["predicted_label"]

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight local interference detector")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build-dataset", help="Build detector dataset from existing objective outputs")
    build_cmd.add_argument(
        "--output-dir",
        default="outputs/interference_detector",
        help="Directory to store detector dataset artifacts",
    )
    build_cmd.set_defaults(func=build_dataset_command)

    train_cmd = subparsers.add_parser("train", help="Train a minimal detector model")
    train_cmd.add_argument("--dataset", required=True, help="CSV or JSONL detector dataset")
    train_cmd.add_argument("--label-mode", default="strict", choices=["strict", "relaxed"])
    train_cmd.add_argument("--model-kind", default="structured", choices=["structured", "text"])
    train_cmd.add_argument("--balance-classes", action="store_true", default=True)
    train_cmd.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    train_cmd.add_argument("--negative-positive-ratio", type=float, default=3.0)
    train_cmd.add_argument("--random-state", type=int, default=42)
    train_cmd.add_argument(
        "--output-model",
        default="outputs/interference_detector/detector.pkl",
        help="Where to store the serialized detector",
    )
    train_cmd.set_defaults(func=train_command)

    score_cmd = subparsers.add_parser("score", help="Score a detector dataset and emit warnings / triggers")
    score_cmd.add_argument("--model-path", required=True)
    score_cmd.add_argument("--dataset", required=True)
    score_cmd.add_argument(
        "--output-file",
        default="outputs/interference_detector/scored_samples.csv",
        help="Where to write scored samples",
    )
    score_cmd.add_argument("--threshold", type=float, default=None)
    score_cmd.set_defaults(func=score_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(name="interference_detector", level=args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
