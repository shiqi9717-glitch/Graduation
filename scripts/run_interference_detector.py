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
    CHANGE_GATE_CHOICES,
    DEFAULT_DATASET_SUMMARY_NAME,
    DEFAULT_FULL_DATASET_NAME,
    DESIGN_VERSION_CHOICES,
    REASONER_MODEL_NAME,
    TRIGGER_POLICY_CHOICES,
    ChangeGateConfig,
    TriggerPolicyConfig,
    apply_change_gate,
    apply_trigger_policy,
    build_interference_dataset,
    evaluate_predictions,
    load_detector,
    resolve_change_gate_config,
    resolve_trigger_policy_config,
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


def _normalize_output_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.parts[:2] == ("outputs", "experiments"):
        return path
    if path.parts and path.parts[0] == "outputs":
        return Path("outputs/experiments").joinpath(*path.parts[1:])
    return path


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


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


def _resolve_recommended_threshold(operating_points: Dict[str, float], policy: str) -> float:
    policy = str(policy or "matched_trigger_budget").strip()
    key = f"{policy}_threshold"
    if key in operating_points:
        return float(operating_points[key])
    if policy == "artifact_default" and "best_f1_threshold" in operating_points:
        return float(operating_points["best_f1_threshold"])
    raise KeyError(f"Threshold policy not found in operating points: {policy}")


def _resolve_trigger_policy(
    trigger_policy: str,
    default_threshold: float,
    reasoner_threshold: float,
) -> TriggerPolicyConfig:
    return resolve_trigger_policy_config(
        policy_name=str(trigger_policy),
        default_threshold=float(default_threshold),
        reasoner_threshold=float(reasoner_threshold),
    )


def _resolve_change_gate(gate_name: str) -> ChangeGateConfig:
    return resolve_change_gate_config(gate_name)


def _trigger_series(
    df: pd.DataFrame,
    policy_config: TriggerPolicyConfig,
) -> pd.Series:
    return apply_trigger_policy(df, policy_config)


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
    policy_config: TriggerPolicyConfig,
    change_gate_config: ChangeGateConfig,
    threshold_name: str,
    recheck_source: str,
    model_kind: str,
) -> Dict[str, Any]:
    working = dataset.copy()
    working["raw_answer"] = working["predicted_answer"].fillna("").astype(str).str.strip().str.upper()
    working["recheck_answer"] = working[recheck_source].fillna("").astype(str).str.strip().str.upper()
    working["trigger_recheck"] = _trigger_series(working, policy_config)
    working["triggered"] = working["trigger_recheck"].astype(int)
    working["raw_correct"] = (
        working["raw_answer"].astype(str).str.upper() == working["ground_truth"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["raw_wrong_follow"] = (
        working["raw_answer"].astype(str).str.upper() == working["wrong_option"].fillna("").astype(str).str.upper()
    ).astype(int)
    working = apply_change_gate(working, change_gate_config, policy_config)
    working["final_answer"] = _text_series(working, "final_answer_after_gate").str.upper().str.strip()
    working["final_correct"] = (
        working["final_answer"].astype(str).str.upper() == working["ground_truth"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["final_wrong_follow"] = (
        working["final_answer"].astype(str).str.upper() == working["wrong_option"].fillna("").astype(str).str.upper()
    ).astype(int)
    working["changed_to_correct"] = (
        working["allow_answer_override"].astype(int).eq(1)
        & (working["final_correct"].astype(int) == 1)
    ).astype(int)
    working["changed_to_wrong"] = (
        working["allow_answer_override"].astype(int).eq(1)
        & (working["final_correct"].astype(int) == 0)
    ).astype(int)
    working["correct_to_wrong"] = (
        (working["raw_correct"].astype(int) == 1)
        & (working["final_correct"].astype(int) == 0)
        & working["allow_answer_override"].astype(int).eq(1)
    ).astype(int)
    w1_rows = working[pd.to_numeric(working.get("explicit_wrong_option", 0), errors="coerce").fillna(0).astype(int) == 1]
    reasoner_rows = working[_text_series(working, "model_name").str.strip().str.lower() == REASONER_MODEL_NAME].copy()
    reasoner_triggered = reasoner_rows[reasoner_rows["trigger_recheck"].astype(int) == 1].copy()
    by_model_rows = []
    for model_name, model_df in working.groupby(_text_series(working, "model_name"), dropna=False):
        if not str(model_name).strip():
            continue
        by_model_rows.append(
            {
                "model_name": str(model_name),
                "n": int(len(model_df)),
                "raw_accuracy": float(model_df["raw_correct"].mean()),
                "guarded_accuracy": float(model_df["final_correct"].mean()),
                "raw_wrong_option_follow_rate": float(model_df["raw_wrong_follow"].mean()),
                "guarded_wrong_option_follow_rate": float(model_df["final_wrong_follow"].mean()),
                "trigger_rate": float(model_df["trigger_recheck"].mean()),
                "changed_to_correct_rate": float(model_df["changed_to_correct"].mean()),
                "correct_to_wrong_rate": float(model_df["correct_to_wrong"].mean()),
            }
        )
    return {
        "model_kind": model_kind,
        "trigger_policy": str(policy_config.policy_name),
        "change_gate": str(change_gate_config.gate_name),
        "threshold_name": threshold_name,
        "threshold": float(policy_config.default_threshold),
        "default_threshold": float(policy_config.default_threshold),
        "reasoner_threshold": float(policy_config.reasoner_threshold),
        "raw_accuracy": float(working["raw_correct"].mean()),
        "guarded_accuracy": float(working["final_correct"].mean()),
        "raw_wrong_option_follow_rate": float(working["raw_wrong_follow"].mean()),
        "guarded_wrong_option_follow_rate": float(working["final_wrong_follow"].mean()),
        "trigger_rate": float(working["trigger_recheck"].mean()),
        "avg_extra_calls": float(working["trigger_recheck"].mean()),
        "changed_to_correct_rate": float(working["changed_to_correct"].mean()),
        "correct_to_wrong_rate": float(working["correct_to_wrong"].mean()),
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
        "reasoner_n": int(len(reasoner_rows)),
        "reasoner_trigger_rate": float(reasoner_rows["trigger_recheck"].mean()) if not reasoner_rows.empty else 0.0,
        "reasoner_raw_accuracy": float(reasoner_rows["raw_correct"].mean()) if not reasoner_rows.empty else 0.0,
        "reasoner_guarded_accuracy": float(reasoner_rows["final_correct"].mean()) if not reasoner_rows.empty else 0.0,
        "reasoner_changed_to_correct_rate": float(reasoner_triggered["changed_to_correct"].mean()) if not reasoner_triggered.empty else 0.0,
        "reasoner_changed_to_wrong_rate": float(reasoner_triggered["changed_to_wrong"].mean()) if not reasoner_triggered.empty else 0.0,
        "reasoner_correct_to_wrong_rate": float(reasoner_rows["correct_to_wrong"].mean()) if not reasoner_rows.empty else 0.0,
        "reasoner_net_gain": (
            float(reasoner_rows["final_correct"].mean()) - float(reasoner_rows["raw_correct"].mean())
        )
        if not reasoner_rows.empty
        else 0.0,
        "reasoner_raw_wrong_option_follow_rate": float(reasoner_rows["raw_wrong_follow"].mean()) if not reasoner_rows.empty else 0.0,
        "reasoner_guarded_wrong_option_follow_rate": float(reasoner_rows["final_wrong_follow"].mean()) if not reasoner_rows.empty else 0.0,
        "by_model": by_model_rows,
    }


def _json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _string_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("NA").astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _label_distribution(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    if label_column not in df.columns:
        return {}
    labeled = df[df[label_column].notna()].copy()
    if labeled.empty:
        return {}
    counts = labeled[label_column].astype(int).value_counts().to_dict()
    return {str(int(key)): int(value) for key, value in counts.items()}


def _is_high_pressure_wrong_option_row(row: pd.Series) -> bool:
    explicit_wrong = int(pd.to_numeric(row.get("explicit_wrong_option", 0), errors="coerce") or 0) == 1
    is_control = int(pd.to_numeric(row.get("is_control", 0), errors="coerce") or 0) == 1
    authority = int(pd.to_numeric(row.get("authority_level", 0), errors="coerce") or 0)
    confidence = int(pd.to_numeric(row.get("confidence_level", 0), errors="coerce") or 0)
    return explicit_wrong and not is_control and authority >= 1 and confidence >= 1


def _reasoner_subset_summary(working: pd.DataFrame, trigger_column: str = "trigger_recheck") -> Dict[str, Any]:
    reasoner_rows = working[_text_series(working, "model_name").str.strip().str.lower() == REASONER_MODEL_NAME].copy()
    if reasoner_rows.empty:
        return {
            "reasoner_n": 0,
            "reasoner_trigger_rate": 0.0,
            "reasoner_raw_accuracy": 0.0,
            "reasoner_guarded_accuracy": 0.0,
            "reasoner_changed_to_correct_rate": 0.0,
            "reasoner_changed_to_wrong_rate": 0.0,
            "reasoner_net_gain": 0.0,
            "reasoner_raw_wrong_option_follow_rate": 0.0,
            "reasoner_guarded_wrong_option_follow_rate": 0.0,
        }
    if "raw_correct" not in reasoner_rows.columns:
        reasoner_rows["raw_correct"] = (
            _text_series(reasoner_rows, "predicted_answer").str.upper().str.strip()
            == _text_series(reasoner_rows, "ground_truth").str.upper().str.strip()
        ).astype(int)
    if "final_correct" not in reasoner_rows.columns:
        reasoner_rows["final_correct"] = _numeric_series(reasoner_rows, "raw_correct", 0).astype(int)
    if "raw_wrong_follow" not in reasoner_rows.columns:
        reasoner_rows["raw_wrong_follow"] = (
            _text_series(reasoner_rows, "predicted_answer").str.upper().str.strip()
            == _text_series(reasoner_rows, "wrong_option").str.upper().str.strip()
        ).astype(int)
    if "final_wrong_follow" not in reasoner_rows.columns:
        reasoner_rows["final_wrong_follow"] = _numeric_series(reasoner_rows, "raw_wrong_follow", 0).astype(int)
    if "changed_to_correct" not in reasoner_rows.columns:
        reasoner_rows["changed_to_correct"] = 0
    if "changed_to_wrong" not in reasoner_rows.columns:
        reasoner_rows["changed_to_wrong"] = 0
    triggered = reasoner_rows[reasoner_rows[trigger_column].astype(int) == 1].copy()
    raw_acc = float(reasoner_rows["raw_correct"].mean())
    guarded_acc = float(reasoner_rows["final_correct"].mean())
    return {
        "reasoner_n": int(len(reasoner_rows)),
        "reasoner_trigger_rate": float(reasoner_rows[trigger_column].mean()),
        "reasoner_raw_accuracy": raw_acc,
        "reasoner_guarded_accuracy": guarded_acc,
        "reasoner_changed_to_correct_rate": float(triggered["changed_to_correct"].mean()) if not triggered.empty else 0.0,
        "reasoner_changed_to_wrong_rate": float(triggered["changed_to_wrong"].mean()) if not triggered.empty else 0.0,
        "reasoner_net_gain": guarded_acc - raw_acc,
        "reasoner_raw_wrong_option_follow_rate": float(reasoner_rows["raw_wrong_follow"].mean()),
        "reasoner_guarded_wrong_option_follow_rate": float(reasoner_rows["final_wrong_follow"].mean()),
    }


def _summarize_split_dataset(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    labeled = df[df[label_column].notna()].copy() if label_column in df.columns else df.iloc[0:0].copy()
    if labeled.empty:
        return {"sample_count": 0}
    return {
        "sample_count": int(len(labeled)),
        "class_distribution": _label_distribution(labeled, label_column),
        "arm_distribution": _string_counts(labeled, "arm_id"),
        "model_distribution": _string_counts(labeled, "model_name"),
        "split_distribution": _string_counts(labeled, "split"),
        "hard_negative_count": int(
            pd.to_numeric(labeled.get("is_hard_negative", 0), errors="coerce").fillna(0).astype(int).sum()
        )
        if "is_hard_negative" in labeled.columns
        else 0,
    }


def _summarize_scored_dataset(
    scored_df: pd.DataFrame,
    label_column: str,
    policy_config: TriggerPolicyConfig | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    if policy_config is None:
        policy_config = _resolve_trigger_policy(
            trigger_policy="global",
            default_threshold=float(0.5 if threshold is None else threshold),
            reasoner_threshold=0.70,
        )
    if scored_df.empty:
        return {
            "num_rows": 0,
            "trigger_policy": str(policy_config.policy_name),
            "threshold": float(policy_config.default_threshold),
            "default_threshold": float(policy_config.default_threshold),
            "reasoner_threshold": float(policy_config.reasoner_threshold),
        }
    working = scored_df.copy()
    working["trigger_recheck"] = _trigger_series(working, policy_config)
    working["high_pressure_wrong_option"] = working.apply(_is_high_pressure_wrong_option_row, axis=1).astype(int)
    strict_positive = working[pd.to_numeric(working.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 1]
    strict_negative = working[pd.to_numeric(working.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 0]
    wrong_option_rows = working[
        pd.to_numeric(working.get("explicit_wrong_option", 0), errors="coerce").fillna(0).astype(int) == 1
    ]
    high_pressure_rows = working[working["high_pressure_wrong_option"].astype(int) == 1]
    split_rows: List[Dict[str, Any]] = []
    for split_name in ("dev", "test"):
        split_df = working[working.get("split", pd.Series("", index=working.index)).astype(str) == split_name].copy()
        if split_df.empty:
            continue
        split_trigger = _trigger_series(split_df, policy_config)
        split_rows.append(
            {
                "split": split_name,
                "n": int(len(split_df)),
                "trigger_rate": float(split_trigger.mean()),
                "strict_positive_recall": float(
                    split_trigger[
                        pd.to_numeric(split_df.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 1
                    ].mean()
                )
                if (pd.to_numeric(split_df.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 1).any()
                else 0.0,
                "strict_negative_false_positive_rate": float(
                    split_trigger[
                        pd.to_numeric(split_df.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 0
                    ].mean()
                )
                if (pd.to_numeric(split_df.get(label_column, -1), errors="coerce").fillna(-1).astype(int) == 0).any()
                else 0.0,
            }
        )
    return {
        "num_rows": int(len(working)),
        "trigger_policy": str(policy_config.policy_name),
        "threshold": float(policy_config.default_threshold),
        "default_threshold": float(policy_config.default_threshold),
        "reasoner_threshold": float(policy_config.reasoner_threshold),
        "overall_trigger_rate": float(working["trigger_recheck"].mean()),
        "wrong_option_trigger_rate": float(wrong_option_rows["trigger_recheck"].mean()) if not wrong_option_rows.empty else 0.0,
        "high_pressure_wrong_option_trigger_rate": float(high_pressure_rows["trigger_recheck"].mean())
        if not high_pressure_rows.empty
        else 0.0,
        "strict_positive_recall": float(strict_positive["trigger_recheck"].mean()) if not strict_positive.empty else 0.0,
        "strict_negative_false_positive_rate": float(strict_negative["trigger_recheck"].mean())
        if not strict_negative.empty
        else 0.0,
        "trigger_by_split": split_rows,
        **_reasoner_subset_summary(working, trigger_column="trigger_recheck"),
    }


def _find_single_path(experiment_dir: Path, patterns: List[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(experiment_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _report_paths(args: argparse.Namespace) -> Dict[str, Path | None]:
    experiment_dir = _normalize_output_path(str(args.experiment_dir))
    label_mode = str(args.label_mode)
    dataset_summary = (
        Path(args.dataset_summary).expanduser()
        if str(args.dataset_summary or "").strip()
        else _find_single_path(experiment_dir, [DEFAULT_DATASET_SUMMARY_NAME, "*dataset_summary*.json"])
    )
    strict_dataset = (
        Path(args.strict_dataset).expanduser()
        if str(args.strict_dataset or "").strip()
        else _find_single_path(experiment_dir, ["*strict*.csv"])
    )
    relaxed_dataset = (
        Path(args.relaxed_dataset).expanduser()
        if str(args.relaxed_dataset or "").strip()
        else _find_single_path(experiment_dir, ["*relaxed*.csv"])
    )
    model_path = (
        Path(args.model_path).expanduser()
        if str(args.model_path or "").strip()
        else _find_single_path(experiment_dir, [f"*{label_mode}*.pkl", "*.pkl"])
    )
    scored_dataset = (
        Path(args.scored_dataset).expanduser()
        if str(args.scored_dataset or "").strip()
        else _find_single_path(experiment_dir, [f"*{label_mode}*scored.csv", "*scored.csv"])
    )
    guard_eval = (
        Path(args.guard_eval_comparison).expanduser()
        if str(args.guard_eval_comparison or "").strip()
        else _find_single_path(experiment_dir, ["guard_eval*comparison*.csv"])
    )
    return {
        "experiment_dir": experiment_dir,
        "dataset_summary": dataset_summary,
        "strict_dataset": strict_dataset,
        "relaxed_dataset": relaxed_dataset,
        "model_path": model_path,
        "scored_dataset": scored_dataset,
        "guard_eval_comparison": guard_eval,
    }


def build_dataset_command(args: argparse.Namespace) -> int:
    summary = build_interference_dataset(
        output_dir=_normalize_output_path(str(args.output_dir)),
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
    tuned_threshold = _resolve_recommended_threshold(
        operating_points,
        policy=str(args.recommended_threshold_policy),
    )
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
    output_path = _normalize_output_path(str(args.output_model))
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
        "recommended_threshold_policy": str(args.recommended_threshold_policy),
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
    default_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata.get("recommended_threshold", 0.5))
    )
    policy_config = _resolve_trigger_policy(
        trigger_policy=str(args.trigger_policy),
        default_threshold=default_threshold,
        reasoner_threshold=float(args.reasoner_threshold),
    )
    dataset["interference_score"] = y_prob
    dataset["predicted_label"] = (dataset["interference_score"] >= float(default_threshold)).astype(int)
    dataset["trigger_recheck"] = _trigger_series(dataset, policy_config)

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

    output_path = _normalize_output_path(str(args.output_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_path": str(Path(args.dataset).resolve()),
        "output_file": str(output_path.resolve()),
        "threshold": float(default_threshold),
        "trigger_policy": policy_config.to_dict(),
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
            threshold=float(default_threshold),
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
            policy_config=_resolve_trigger_policy(
                trigger_policy=str(args.trigger_policy),
                default_threshold=float(threshold),
                reasoner_threshold=float(args.reasoner_threshold),
            ),
            change_gate_config=_resolve_change_gate(str(args.change_gate)),
            threshold_name=threshold_name,
            recheck_source=recheck_source,
            model_kind=str(model_metadata.get("model_kind", args.model_kind or "unknown")),
        )
        for threshold_name, threshold in threshold_map.items()
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    if args.output_comparison_file:
        comparison_path = _normalize_output_path(str(args.output_comparison_file))
    elif args.output_file:
        comparison_path = _normalize_output_path(str(args.output_file)).with_name("guard_eval_comparison.csv")
    else:
        comparison_path = Path(args.dataset).with_name("guard_eval_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "mode": "offline_selective_recheck_simulation",
        "dataset_path": str(Path(args.dataset).resolve()),
        "recheck_source": recheck_source,
        "trigger_policy": _resolve_trigger_policy(
            trigger_policy=str(args.trigger_policy),
            default_threshold=float(next(iter(threshold_map.values()))) if threshold_map else 0.5,
            reasoner_threshold=float(args.reasoner_threshold),
        ).to_dict(),
        "change_gate": _resolve_change_gate(str(args.change_gate)).to_dict(),
        "num_rows": int(len(dataset)),
        "comparison_csv": str(comparison_path.resolve()),
        "comparisons": comparison_rows,
    }
    if args.output_file:
        output_path = _normalize_output_path(str(args.output_file))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def report_command(args: argparse.Namespace) -> int:
    resolved = _report_paths(args)
    experiment_dir = resolved["experiment_dir"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        _normalize_output_path(str(args.output_file))
        if str(args.output_file or "").strip()
        else experiment_dir / f"interference_experiment_report_{args.label_mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "experiment_dir": str(experiment_dir.resolve()),
        "label_mode": str(args.label_mode),
        "high_pressure_definition": "explicit_wrong_option=1 AND is_control=0 AND authority_level>=1 AND confidence_level>=1",
        "inputs": {
            key: str(value.resolve()) if isinstance(value, Path) and value.exists() else ""
            for key, value in resolved.items()
            if key != "experiment_dir"
        },
    }

    dataset_summary_path = resolved["dataset_summary"]
    if isinstance(dataset_summary_path, Path) and dataset_summary_path.exists():
        report["dataset_summary"] = _json_load(dataset_summary_path)

    strict_path = resolved["strict_dataset"]
    if isinstance(strict_path, Path) and strict_path.exists():
        strict_df = _read_dataset(strict_path)
        report["strict_dataset"] = _summarize_split_dataset(strict_df, "strict_label")

    relaxed_path = resolved["relaxed_dataset"]
    if isinstance(relaxed_path, Path) and relaxed_path.exists():
        relaxed_df = _read_dataset(relaxed_path)
        report["relaxed_dataset"] = _summarize_split_dataset(relaxed_df, "relaxed_label")

    threshold = float(args.threshold) if args.threshold is not None else 0.5
    model_path = resolved["model_path"]
    model_metadata: Dict[str, Any] = {}
    if isinstance(model_path, Path) and model_path.exists():
        _, model_metadata = load_detector(model_path)
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(model_metadata.get("recommended_threshold", threshold))
        )
        report["detector"] = {
            "model_kind": str(model_metadata.get("model_kind", "")),
            "label_mode": str(model_metadata.get("label_mode", "")),
            "recommended_threshold": float(model_metadata.get("recommended_threshold", threshold)),
            "recommended_threshold_policy": str(model_metadata.get("recommended_threshold_policy", "best_f1")),
            "threshold_eval_split": str(model_metadata.get("threshold_eval_split", "")),
            "operating_points": model_metadata.get("operating_points", {}),
            "metrics": model_metadata.get("metrics", {}),
            "sampling_summary": model_metadata.get("sampling_summary", {}),
        }
    policy_config = _resolve_trigger_policy(
        trigger_policy=str(args.trigger_policy),
        default_threshold=float(threshold),
        reasoner_threshold=float(args.reasoner_threshold),
    )
    report["trigger_policy"] = policy_config.to_dict()

    scored_path = resolved["scored_dataset"]
    if isinstance(scored_path, Path) and scored_path.exists():
        scored_df = _read_dataset(scored_path)
        report["scored_summary"] = _summarize_scored_dataset(
            scored_df=scored_df,
            policy_config=policy_config,
            label_column="strict_label",
        )

    guard_eval_path = resolved["guard_eval_comparison"]
    if isinstance(guard_eval_path, Path) and guard_eval_path.exists():
        guard_eval_df = _read_dataset(guard_eval_path)
        report["guard_eval"] = {
            "rows": guard_eval_df.to_dict(orient="records"),
        }
        threshold_name = str(args.guard_threshold_name or "").strip()
        if threshold_name and "threshold_name" in guard_eval_df.columns:
            matched = guard_eval_df[guard_eval_df["threshold_name"].astype(str) == threshold_name].copy()
            if not matched.empty:
                report["guard_eval"]["selected_threshold"] = matched.iloc[0].to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
        "--recommended-threshold-policy",
        default="matched_trigger_budget",
        choices=[
            "best_f1",
            "high_precision",
            "high_recall",
            "aggressive",
            "matched_trigger_budget",
            "recall_constrained",
        ],
        help="Which operating point should be saved as the default trigger threshold.",
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
    score_cmd.add_argument("--threshold", "--default-threshold", type=float, default=None)
    score_cmd.add_argument("--trigger-policy", default="global", choices=TRIGGER_POLICY_CHOICES)
    score_cmd.add_argument("--reasoner-threshold", type=float, default=0.70)
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
    guard_cmd.add_argument("--threshold", "--default-threshold", type=float, default=None, help="Explicit default threshold for single-threshold guard evaluation.")
    guard_cmd.add_argument("--threshold-name", default="explicit", help="Name for the explicitly provided threshold.")
    guard_cmd.add_argument("--trigger-policy", default="global", choices=TRIGGER_POLICY_CHOICES)
    guard_cmd.add_argument("--reasoner-threshold", type=float, default=0.70)
    guard_cmd.add_argument("--change-gate", default="none", choices=CHANGE_GATE_CHOICES)
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

    report_cmd = subparsers.add_parser(
        "report",
        help="Collect dataset / detector / trigger / guard metrics into one reproducible summary JSON.",
    )
    report_cmd.add_argument("--experiment-dir", required=True, help="Experiment output directory.")
    report_cmd.add_argument("--label-mode", default="strict", choices=["strict", "relaxed"])
    report_cmd.add_argument("--dataset-summary", default="", help="Optional explicit dataset summary JSON path.")
    report_cmd.add_argument("--strict-dataset", default="", help="Optional explicit strict dataset CSV path.")
    report_cmd.add_argument("--relaxed-dataset", default="", help="Optional explicit relaxed dataset CSV path.")
    report_cmd.add_argument("--model-path", default="", help="Optional explicit detector artifact path.")
    report_cmd.add_argument("--scored-dataset", default="", help="Optional explicit scored CSV path.")
    report_cmd.add_argument(
        "--guard-eval-comparison",
        default="",
        help="Optional explicit guard eval comparison CSV path.",
    )
    report_cmd.add_argument(
        "--guard-threshold-name",
        default="matched_trigger_budget",
        help="Optional threshold row to highlight from guard eval comparison.",
    )
    report_cmd.add_argument("--threshold", "--default-threshold", type=float, default=None, help="Optional explicit default trigger threshold override.")
    report_cmd.add_argument("--trigger-policy", default="global", choices=TRIGGER_POLICY_CHOICES)
    report_cmd.add_argument("--reasoner-threshold", type=float, default=0.70)
    report_cmd.add_argument(
        "--output-file",
        default="",
        help="Optional JSON path for the generated experiment report.",
    )
    report_cmd.set_defaults(func=report_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(name="interference_detector", level=args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
