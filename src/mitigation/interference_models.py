"""Minimal local interference detector models."""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(y_true))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return wins / float(len(pos) * len(neg))


def evaluate_predictions(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    metrics = _binary_metrics(y_true_arr, y_prob_arr, threshold=threshold)
    metrics["roc_auc"] = _roc_auc(y_true_arr, y_prob_arr)
    metrics["mean_positive_score"] = float(y_prob_arr[y_true_arr == 1].mean()) if (y_true_arr == 1).any() else 0.0
    metrics["mean_negative_score"] = float(y_prob_arr[y_true_arr == 0].mean()) if (y_true_arr == 0).any() else 0.0
    return metrics


class StructuredLogisticDetector:
    """Statsmodels-based logistic classifier over structured metadata features."""

    MODEL_TYPE = "structured_logistic"

    BASE_NUMERIC_COLUMNS = [
        "authority_level",
        "confidence_level",
        "explicit_wrong_option",
        "is_control",
        "answer_equals_wrong_option",
        "answer_equals_ground_truth",
        "answer_equals_baseline_answer",
        "answer_changed_from_baseline",
        "baseline_accuracy_prob",
        "control_reference_accuracy_prob",
        "prefix_length",
        "question_length",
        "answer_length",
    ]
    BASE_CATEGORICAL_COLUMNS = ["arm_id", "subject", "category", "model_name"]

    def __init__(self) -> None:
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = list(self.BASE_NUMERIC_COLUMNS)
        self.categorical_columns: List[str] = list(self.BASE_CATEGORICAL_COLUMNS)
        self.params: Dict[str, float] = {}

    def _prepare_frame(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        working = df.copy()
        for column in self.numeric_columns:
            if column not in working.columns:
                working[column] = 0.0
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

        categorical_frames = []
        for column in self.categorical_columns:
            if column not in working.columns:
                working[column] = "NA"
            dummies = pd.get_dummies(working[column].fillna("NA").astype(str), prefix=column)
            categorical_frames.append(dummies)

        feature_df = pd.concat([working[self.numeric_columns]] + categorical_frames, axis=1)
        if fit:
            self.feature_columns = list(feature_df.columns)
        for column in self.feature_columns:
            if column not in feature_df.columns:
                feature_df[column] = 0.0
        feature_df = feature_df[self.feature_columns]
        feature_df = sm.add_constant(feature_df, has_constant="add")
        return feature_df.astype(float)

    def fit(self, train_df: pd.DataFrame, label_column: str) -> None:
        y = train_df[label_column].astype(int)
        x = self._prepare_frame(train_df, fit=True)
        model = sm.GLM(y, x, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
        self.params = {column: float(value) for column, value in result.params.items()}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.params:
            raise ValueError("StructuredLogisticDetector has not been fitted.")
        x = self._prepare_frame(df, fit=False)
        ordered_columns = list(self.params.keys())
        for column in ordered_columns:
            if column not in x.columns:
                x[column] = 0.0
        x = x[ordered_columns]
        beta = np.asarray([self.params[column] for column in ordered_columns], dtype=float)
        return _sigmoid(x.to_numpy(dtype=float) @ beta)

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "model_type": self.MODEL_TYPE,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "StructuredLogisticDetector":
        detector = cls()
        detector.feature_columns = list(artifact.get("feature_columns", []))
        detector.numeric_columns = list(artifact.get("numeric_columns", cls.BASE_NUMERIC_COLUMNS))
        detector.categorical_columns = list(artifact.get("categorical_columns", cls.BASE_CATEGORICAL_COLUMNS))
        detector.params = {str(k): float(v) for k, v in artifact.get("params", {}).items()}
        return detector


class TextNGramNBDetector:
    """Tiny text classifier using character n-gram multinomial naive Bayes."""

    MODEL_TYPE = "text_ngram_nb"

    def __init__(self, ngram_range: Tuple[int, int] = (2, 4), max_features: int = 4000) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.class_log_prior: Dict[int, float] = {}
        self.feature_log_prob: Dict[int, np.ndarray] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = str(text or "").lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _ngrams(self, text: str) -> List[str]:
        normalized = self._normalize(text)
        grams: List[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(normalized) < n:
                continue
            for idx in range(len(normalized) - n + 1):
                grams.append(normalized[idx : idx + n])
        return grams

    def fit(self, train_df: pd.DataFrame, label_column: str, text_column: str = "text_input") -> None:
        labels = train_df[label_column].astype(int).tolist()
        texts = train_df[text_column].astype(str).tolist()
        df_counts: Counter[str] = Counter()
        per_doc_counts: List[Counter[str]] = []
        for text in texts:
            grams = self._ngrams(text)
            counter = Counter(grams)
            per_doc_counts.append(counter)
            df_counts.update(counter.keys())
        most_common = df_counts.most_common(self.max_features)
        self.vocab = {gram: idx for idx, (gram, _) in enumerate(most_common)}

        class_counts = Counter(labels)
        total_docs = float(len(labels))
        self.class_log_prior = {
            label: math.log(count / total_docs) for label, count in class_counts.items()
        }

        for label in sorted(class_counts.keys()):
            token_totals = np.ones(len(self.vocab), dtype=float)
            for doc_label, counts in zip(labels, per_doc_counts):
                if doc_label != label:
                    continue
                for gram, count in counts.items():
                    idx = self.vocab.get(gram)
                    if idx is not None:
                        token_totals[idx] += float(count)
            self.feature_log_prob[label] = np.log(token_totals / token_totals.sum())

    def predict_proba(self, df: pd.DataFrame, text_column: str = "text_input") -> np.ndarray:
        if not self.vocab:
            raise ValueError("TextNGramNBDetector has not been fitted.")
        probs: List[float] = []
        for text in df[text_column].astype(str).tolist():
            counts = Counter(self._ngrams(text))
            class_scores: Dict[int, float] = {}
            for label, prior in self.class_log_prior.items():
                score = prior
                log_probs = self.feature_log_prob[label]
                for gram, count in counts.items():
                    idx = self.vocab.get(gram)
                    if idx is not None:
                        score += float(count) * float(log_probs[idx])
                class_scores[label] = score
            score0 = class_scores.get(0, -100.0)
            score1 = class_scores.get(1, -100.0)
            normalized = _sigmoid(np.asarray([score1 - score0], dtype=float))[0]
            probs.append(float(normalized))
        return np.asarray(probs, dtype=float)

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "model_type": self.MODEL_TYPE,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "vocab": self.vocab,
            "class_log_prior": self.class_log_prior,
            "feature_log_prob": {
                str(label): values.tolist() for label, values in self.feature_log_prob.items()
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "TextNGramNBDetector":
        detector = cls(
            ngram_range=tuple(int(x) for x in artifact.get("ngram_range", [2, 4])),
            max_features=int(artifact.get("max_features", 4000)),
        )
        detector.vocab = {str(k): int(v) for k, v in artifact.get("vocab", {}).items()}
        detector.class_log_prior = {int(k): float(v) for k, v in artifact.get("class_log_prior", {}).items()}
        detector.feature_log_prob = {
            int(k): np.asarray(v, dtype=float) for k, v in artifact.get("feature_log_prob", {}).items()
        }
        return detector


def save_detector(detector: Any, path: Path, metadata: Dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "artifact": detector.to_artifact(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_detector(path: Path) -> Tuple[Any, Dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    artifact = payload["artifact"]
    model_type = artifact.get("model_type")
    if model_type == StructuredLogisticDetector.MODEL_TYPE:
        detector = StructuredLogisticDetector.from_artifact(artifact)
    elif model_type == TextNGramNBDetector.MODEL_TYPE:
        detector = TextNGramNBDetector.from_artifact(artifact)
    else:
        raise ValueError(f"Unsupported detector artifact type: {model_type}")
    return detector, dict(payload.get("metadata", {}))
