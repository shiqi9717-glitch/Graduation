"""Minimal local interference detector models."""

from __future__ import annotations

import csv
import hashlib
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


def threshold_sweep(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true_arr.tolist(), y_prob_arr.tolist(), threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
                "trigger_rate": float((y_prob_arr >= float(threshold)).mean()) if len(y_prob_arr) else 0.0,
                "false_positive_rate": _safe_div(float(metrics["fp"]), float(metrics["fp"] + metrics["tn"])),
                "true_positive_rate": float(metrics["recall"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
        )
    return pd.DataFrame(rows)


def select_operating_points(
    sweep_df: pd.DataFrame,
    target_trigger_rate: float = 0.075,
    max_trigger_rate: float = 0.08,
) -> Dict[str, float]:
    if sweep_df.empty:
        return {
            "best_f1_threshold": 0.5,
            "high_precision_threshold": 0.5,
            "high_recall_threshold": 0.5,
            "aggressive_threshold": 0.5,
            "matched_trigger_budget_threshold": 0.5,
            "recall_constrained_threshold": 0.5,
        }
    best_f1_row = sweep_df.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]

    precision_candidates = sweep_df[sweep_df["precision"] >= 0.5]
    if precision_candidates.empty:
        precision_candidates = sweep_df[sweep_df["precision"] >= 0.4]
    if precision_candidates.empty:
        precision_candidates = sweep_df
    high_precision_row = precision_candidates.sort_values(
        ["recall", "precision", "f1", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]

    recall_candidates = sweep_df[sweep_df["recall"] >= 0.5]
    if recall_candidates.empty:
        recall_candidates = sweep_df[sweep_df["recall"] >= 0.4]
    if recall_candidates.empty:
        recall_candidates = sweep_df
    high_recall_row = recall_candidates.sort_values(
        ["precision", "recall", "f1", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]

    matched_trigger_row = sweep_df.assign(
        trigger_distance=(sweep_df["trigger_rate"] - float(target_trigger_rate)).abs()
    ).sort_values(
        ["trigger_distance", "recall", "precision", "threshold"],
        ascending=[True, False, False, False],
    ).iloc[0]

    budget_candidates = sweep_df[sweep_df["trigger_rate"] <= float(max_trigger_rate)]
    if budget_candidates.empty:
        budget_candidates = sweep_df.assign(
            trigger_overrun=(sweep_df["trigger_rate"] - float(max_trigger_rate)).abs()
        ).sort_values(["trigger_overrun", "recall", "precision"], ascending=[True, False, False])
        recall_budget_row = budget_candidates.iloc[0]
    else:
        recall_budget_row = budget_candidates.sort_values(
            ["recall", "precision", "f1", "trigger_rate", "threshold"],
            ascending=[False, False, False, False, True],
        ).iloc[0]

    return {
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "high_precision_threshold": float(high_precision_row["threshold"]),
        "high_recall_threshold": float(high_recall_row["threshold"]),
        "aggressive_threshold": float(sweep_df["threshold"].min()),
        "matched_trigger_budget_threshold": float(matched_trigger_row["threshold"]),
        "recall_constrained_threshold": float(recall_budget_row["threshold"]),
    }


class StructuredLogisticDetector:
    """Statsmodels-based logistic classifier over structured metadata features."""

    MODEL_TYPE = "structured_logistic"
    FEATURE_MODE_SAFE = "safe"
    FEATURE_MODE_ORACLE = "oracle_upper_bound"

    SAFE_NUMERIC_COLUMNS = [
        "authority_level",
        "confidence_level",
        "explicit_wrong_option",
        "is_control",
        "prefix_length",
        "question_length",
        "answer_length",
    ]
    SAFE_CATEGORICAL_COLUMNS = ["arm_id", "subject", "category", "model_name"]

    ORACLE_NUMERIC_COLUMNS = [
        "authority_level",
        "confidence_level",
        "explicit_wrong_option",
        "is_control",
        "answer_equals_wrong_option",
        "answer_equals_ground_truth",
        "answer_equals_baseline_answer",
        "answer_changed_from_baseline",
        "arm_correct_prob",
        "arm_sycophancy_prob",
        "arm_wrong_option_follow_prob",
        "baseline_accuracy_prob",
        "control_reference_accuracy_prob",
        "prefix_length",
        "question_length",
        "answer_length",
    ]
    ORACLE_CATEGORICAL_COLUMNS = ["arm_id", "subject", "category", "model_name"]

    def __init__(self, feature_mode: str = FEATURE_MODE_SAFE) -> None:
        if feature_mode not in {self.FEATURE_MODE_SAFE, self.FEATURE_MODE_ORACLE}:
            raise ValueError(f"Unsupported structured feature mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.feature_columns: List[str] = []
        if feature_mode == self.FEATURE_MODE_ORACLE:
            self.numeric_columns = list(self.ORACLE_NUMERIC_COLUMNS)
            self.categorical_columns = list(self.ORACLE_CATEGORICAL_COLUMNS)
        else:
            self.numeric_columns = list(self.SAFE_NUMERIC_COLUMNS)
            self.categorical_columns = list(self.SAFE_CATEGORICAL_COLUMNS)
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
            "feature_mode": self.feature_mode,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "StructuredLogisticDetector":
        detector = cls(feature_mode=str(artifact.get("feature_mode", cls.FEATURE_MODE_SAFE)))
        detector.feature_columns = list(artifact.get("feature_columns", []))
        detector.numeric_columns = list(artifact.get("numeric_columns", detector.numeric_columns))
        detector.categorical_columns = list(artifact.get("categorical_columns", detector.categorical_columns))
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


class TextTfidfLogRegDetector:
    """Lightweight char n-gram TF-IDF + logistic regression baseline."""

    MODEL_TYPE = "text_tfidf_logreg"

    def __init__(self, ngram_range: Tuple[int, int] = (2, 5), max_features: int = 5000) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.asarray([], dtype=float)
        self.params: Dict[str, float] = {}

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

    def _build_vocab(self, texts: List[str], labels: List[int]) -> None:
        positive_counts: Counter[str] = Counter()
        negative_counts: Counter[str] = Counter()
        corpus_counts: Counter[str] = Counter()
        doc_freq: Counter[str] = Counter()
        for text, label in zip(texts, labels):
            counter = Counter(self._ngrams(text))
            corpus_counts.update(counter)
            doc_freq.update(counter.keys())
            if int(label) == 1:
                positive_counts.update(counter)
            else:
                negative_counts.update(counter)
        ranked = sorted(
            corpus_counts.keys(),
            key=lambda gram: (
                abs(float(positive_counts.get(gram, 0)) - float(negative_counts.get(gram, 0))),
                float(corpus_counts.get(gram, 0)),
                -float(doc_freq.get(gram, 0)),
                gram,
            ),
            reverse=True,
        )
        selected = ranked[: self.max_features]
        self.vocab = {gram: idx for idx, gram in enumerate(selected)}
        num_docs = max(len(texts), 1)
        idf = np.ones(len(self.vocab), dtype=float)
        for gram, idx in self.vocab.items():
            df_count = int(doc_freq.get(gram, 0))
            idf[idx] = math.log((1.0 + num_docs) / (1.0 + df_count)) + 1.0
        self.idf = idf

    def _vectorize(self, texts: List[str], fit: bool = False) -> pd.DataFrame:
        doc_counters: List[Counter[str]] = []
        for text in texts:
            counter = Counter(self._ngrams(text))
            doc_counters.append(counter)
        if fit and not self.vocab:
            raise ValueError("Vocabulary must be initialized before fit=True vectorization.")

        if not self.vocab:
            raise ValueError("TextTfidfLogRegDetector vocabulary is empty.")

        rows: List[np.ndarray] = []
        for counter in doc_counters:
            vector = np.zeros(len(self.vocab), dtype=float)
            total = float(sum(counter.values()))
            if total <= 0:
                rows.append(vector)
                continue
            for gram, count in counter.items():
                idx = self.vocab.get(gram)
                if idx is not None:
                    vector[idx] = float(count) / total
            vector = vector * self.idf
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            rows.append(vector)

        columns = [f"tfidf_{gram}" for gram, _ in sorted(self.vocab.items(), key=lambda item: item[1])]
        return pd.DataFrame(rows, columns=columns)

    def fit(self, train_df: pd.DataFrame, label_column: str, text_column: str = "text_input") -> None:
        texts = train_df[text_column].astype(str).tolist()
        y = train_df[label_column].astype(int)
        self._build_vocab(texts, y.astype(int).tolist())
        x_base = self._vectorize(texts, fit=True)
        x = sm.add_constant(x_base, has_constant="add")
        model = sm.GLM(y, x, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
        params = {column: float(value) for column, value in result.params.items()}
        non_constant = [abs(value) for column, value in params.items() if column != "const"]
        if not non_constant or max(non_constant) < 1e-10:
            positive_mask = y == 1
            negative_mask = y == 0
            positive_mean = x_base.loc[positive_mask].mean(axis=0) if positive_mask.any() else pd.Series(0.0, index=x_base.columns)
            negative_mean = x_base.loc[negative_mask].mean(axis=0) if negative_mask.any() else pd.Series(0.0, index=x_base.columns)
            prior = float(positive_mask.mean()) if len(y) else 0.5
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            params = {"const": float(math.log(prior / (1.0 - prior)))}
            for column in x_base.columns:
                params[column] = float(positive_mean.get(column, 0.0) - negative_mean.get(column, 0.0))
        self.params = params

    def predict_proba(self, df: pd.DataFrame, text_column: str = "text_input") -> np.ndarray:
        if not self.params:
            raise ValueError("TextTfidfLogRegDetector has not been fitted.")
        x = self._vectorize(df[text_column].astype(str).tolist(), fit=False)
        x = sm.add_constant(x, has_constant="add")
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
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "vocab": self.vocab,
            "idf": self.idf.tolist(),
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "TextTfidfLogRegDetector":
        detector = cls(
            ngram_range=tuple(int(x) for x in artifact.get("ngram_range", [2, 5])),
            max_features=int(artifact.get("max_features", 5000)),
        )
        detector.vocab = {str(k): int(v) for k, v in artifact.get("vocab", {}).items()}
        detector.idf = np.asarray(artifact.get("idf", []), dtype=float)
        detector.params = {str(k): float(v) for k, v in artifact.get("params", {}).items()}
        return detector


class EmbeddingLogRegDetector:
    """Local hashed n-gram embedding + logistic regression baseline.

    This intentionally avoids heavy transformer dependencies while preserving the
    embedding-style interface needed for apples-to-apples detector comparisons:
    text_input -> fixed-size dense vector -> lightweight classifier.
    """

    MODEL_TYPE = "embedding_logreg"

    def __init__(self, embedding_dim: int = 384, ngram_range: Tuple[int, int] = (2, 5)) -> None:
        self.embedding_dim = embedding_dim
        self.ngram_range = ngram_range
        self.bucket_idf = np.ones(self.embedding_dim, dtype=float)
        self.params: Dict[str, float] = {}
        self.embedding_backend = "local_hashing_char_ngram"

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

    def _bucket_and_sign(self, gram: str) -> Tuple[int, float]:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="big", signed=False)
        bucket = raw % self.embedding_dim
        sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
        return bucket, sign

    def _fit_idf(self, texts: List[str]) -> None:
        doc_freq = np.zeros(self.embedding_dim, dtype=float)
        for text in texts:
            seen = set()
            for gram in self._ngrams(text):
                bucket, _ = self._bucket_and_sign(gram)
                seen.add(bucket)
            for bucket in seen:
                doc_freq[bucket] += 1.0
        num_docs = max(len(texts), 1)
        self.bucket_idf = np.log((1.0 + num_docs) / (1.0 + doc_freq)) + 1.0

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=float)
        grams = self._ngrams(text)
        if not grams:
            return vector
        for gram in grams:
            bucket, sign = self._bucket_and_sign(gram)
            vector[bucket] += sign
        vector = (vector / float(len(grams))) * self.bucket_idf
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def _vectorize(self, texts: List[str]) -> pd.DataFrame:
        rows = [self._embed_one(text) for text in texts]
        columns = [f"embed_{idx}" for idx in range(self.embedding_dim)]
        return pd.DataFrame(rows, columns=columns)

    def fit(self, train_df: pd.DataFrame, label_column: str, text_column: str = "text_input") -> None:
        texts = train_df[text_column].astype(str).tolist()
        y = train_df[label_column].astype(int)
        self._fit_idf(texts)
        x_base = self._vectorize(texts)
        x = sm.add_constant(x_base, has_constant="add")
        model = sm.GLM(y, x, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
        params = {column: float(value) for column, value in result.params.items()}
        non_constant = [abs(value) for column, value in params.items() if column != "const"]
        if not non_constant or max(non_constant) < 1e-10:
            positive_mask = y == 1
            negative_mask = y == 0
            positive_mean = x_base.loc[positive_mask].mean(axis=0) if positive_mask.any() else pd.Series(0.0, index=x_base.columns)
            negative_mean = x_base.loc[negative_mask].mean(axis=0) if negative_mask.any() else pd.Series(0.0, index=x_base.columns)
            prior = float(positive_mask.mean()) if len(y) else 0.5
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            params = {"const": float(math.log(prior / (1.0 - prior)))}
            for column in x_base.columns:
                params[column] = float(positive_mean.get(column, 0.0) - negative_mean.get(column, 0.0))
        self.params = params

    def predict_proba(self, df: pd.DataFrame, text_column: str = "text_input") -> np.ndarray:
        if not self.params:
            raise ValueError("EmbeddingLogRegDetector has not been fitted.")
        x = self._vectorize(df[text_column].astype(str).tolist())
        x = sm.add_constant(x, has_constant="add")
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
            "embedding_backend": self.embedding_backend,
            "embedding_dim": self.embedding_dim,
            "ngram_range": list(self.ngram_range),
            "bucket_idf": self.bucket_idf.tolist(),
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "EmbeddingLogRegDetector":
        detector = cls(
            embedding_dim=int(artifact.get("embedding_dim", 384)),
            ngram_range=tuple(int(x) for x in artifact.get("ngram_range", [2, 5])),
        )
        detector.embedding_backend = str(artifact.get("embedding_backend", detector.embedding_backend))
        detector.bucket_idf = np.asarray(artifact.get("bucket_idf", np.ones(detector.embedding_dim)), dtype=float)
        if len(detector.bucket_idf) != detector.embedding_dim:
            detector.bucket_idf = np.ones(detector.embedding_dim, dtype=float)
        detector.params = {str(k): float(v) for k, v in artifact.get("params", {}).items()}
        return detector


class SentenceEmbeddingLogRegDetector:
    """Sentence-transformer embedding + logistic regression detector.

    The encoder dependency is optional and loaded lazily so the lightweight
    detector workflow remains usable without transformer packages installed.
    """

    MODEL_TYPE = "sentence_embedding_logreg"
    DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.params: Dict[str, float] = {}
        self.embedding_dim: int = 0
        self.embedding_backend = "sentence_transformers"
        self._encoder: Any | None = None

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-embedding-logreg requires the optional 'sentence-transformers' package. "
                "Install it with: ./.venv/bin/pip install sentence-transformers"
            ) from exc
        self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _encode(self, texts: List[str]) -> pd.DataFrame:
        encoder = self._load_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(embeddings, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.embedding_dim = int(arr.shape[1]) if arr.size else int(self.embedding_dim)
        columns = [f"sent_embed_{idx}" for idx in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)

    def fit(self, train_df: pd.DataFrame, label_column: str, text_column: str = "text_input") -> None:
        texts = train_df[text_column].fillna("").astype(str).tolist()
        y = train_df[label_column].astype(int)
        x_base = self._encode(texts)
        x = sm.add_constant(x_base, has_constant="add")
        model = sm.GLM(y, x, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
        params = {column: float(value) for column, value in result.params.items()}
        non_constant = [abs(value) for column, value in params.items() if column != "const"]
        if not non_constant or max(non_constant) < 1e-10:
            positive_mask = y == 1
            negative_mask = y == 0
            positive_mean = x_base.loc[positive_mask].mean(axis=0) if positive_mask.any() else pd.Series(0.0, index=x_base.columns)
            negative_mean = x_base.loc[negative_mask].mean(axis=0) if negative_mask.any() else pd.Series(0.0, index=x_base.columns)
            prior = float(positive_mask.mean()) if len(y) else 0.5
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            params = {"const": float(math.log(prior / (1.0 - prior)))}
            for column in x_base.columns:
                params[column] = float(positive_mean.get(column, 0.0) - negative_mean.get(column, 0.0))
        self.params = params

    def predict_proba(self, df: pd.DataFrame, text_column: str = "text_input") -> np.ndarray:
        if not self.params:
            raise ValueError("SentenceEmbeddingLogRegDetector has not been fitted.")
        x = self._encode(df[text_column].fillna("").astype(str).tolist())
        x = sm.add_constant(x, has_constant="add")
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
            "embedding_backend": self.embedding_backend,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "SentenceEmbeddingLogRegDetector":
        detector = cls(
            model_name=str(artifact.get("model_name", cls.DEFAULT_MODEL_NAME)),
            batch_size=int(artifact.get("batch_size", 32)),
        )
        detector.embedding_backend = str(artifact.get("embedding_backend", detector.embedding_backend))
        detector.embedding_dim = int(artifact.get("embedding_dim", 0))
        detector.params = {str(k): float(v) for k, v in artifact.get("params", {}).items()}
        return detector


class HybridSentenceStructuredLogRegDetector(SentenceEmbeddingLogRegDetector):
    """Sentence embedding plus non-leaky structured-safe features."""

    MODEL_TYPE = "hybrid_sentence_structured_logreg"
    SAFE_NUMERIC_COLUMNS = list(StructuredLogisticDetector.SAFE_NUMERIC_COLUMNS)
    SAFE_CATEGORICAL_COLUMNS = list(StructuredLogisticDetector.SAFE_CATEGORICAL_COLUMNS)

    def __init__(self, model_name: str = SentenceEmbeddingLogRegDetector.DEFAULT_MODEL_NAME, batch_size: int = 32) -> None:
        super().__init__(model_name=model_name, batch_size=batch_size)
        self.embedding_backend = "sentence_transformers_plus_structured_safe"
        self.numeric_columns = list(self.SAFE_NUMERIC_COLUMNS)
        self.categorical_columns = list(self.SAFE_CATEGORICAL_COLUMNS)
        self.structured_feature_columns: List[str] = []

    def _prepare_structured_frame(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        working = df.copy()
        for column in self.numeric_columns:
            if column not in working.columns:
                working[column] = 0.0
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

        categorical_frames = []
        for column in self.categorical_columns:
            if column not in working.columns:
                working[column] = "NA"
            categorical_frames.append(pd.get_dummies(working[column].fillna("NA").astype(str), prefix=column))

        feature_df = pd.concat([working[self.numeric_columns]] + categorical_frames, axis=1)
        feature_df.columns = [f"structured_{column}" for column in feature_df.columns]
        if fit:
            self.structured_feature_columns = list(feature_df.columns)
        for column in self.structured_feature_columns:
            if column not in feature_df.columns:
                feature_df[column] = 0.0
        return feature_df[self.structured_feature_columns].astype(float)

    def _combined_features(self, df: pd.DataFrame, text_column: str = "text_input", fit: bool = False) -> pd.DataFrame:
        embeddings = self._encode(df[text_column].fillna("").astype(str).tolist())
        structured = self._prepare_structured_frame(df, fit=fit)
        return pd.concat([embeddings.reset_index(drop=True), structured.reset_index(drop=True)], axis=1)

    def fit(self, train_df: pd.DataFrame, label_column: str, text_column: str = "text_input") -> None:
        y = train_df[label_column].astype(int)
        x_base = self._combined_features(train_df, text_column=text_column, fit=True)
        x = sm.add_constant(x_base, has_constant="add")
        model = sm.GLM(y, x, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=200)
        params = {column: float(value) for column, value in result.params.items()}
        non_constant = [abs(value) for column, value in params.items() if column != "const"]
        if not non_constant or max(non_constant) < 1e-10:
            positive_mask = y == 1
            negative_mask = y == 0
            positive_mean = x_base.loc[positive_mask].mean(axis=0) if positive_mask.any() else pd.Series(0.0, index=x_base.columns)
            negative_mean = x_base.loc[negative_mask].mean(axis=0) if negative_mask.any() else pd.Series(0.0, index=x_base.columns)
            prior = float(positive_mask.mean()) if len(y) else 0.5
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            params = {"const": float(math.log(prior / (1.0 - prior)))}
            for column in x_base.columns:
                params[column] = float(positive_mean.get(column, 0.0) - negative_mean.get(column, 0.0))
        self.params = params

    def predict_proba(self, df: pd.DataFrame, text_column: str = "text_input") -> np.ndarray:
        if not self.params:
            raise ValueError("HybridSentenceStructuredLogRegDetector has not been fitted.")
        x = self._combined_features(df, text_column=text_column, fit=False)
        x = sm.add_constant(x, has_constant="add")
        ordered_columns = list(self.params.keys())
        for column in ordered_columns:
            if column not in x.columns:
                x[column] = 0.0
        x = x[ordered_columns]
        beta = np.asarray([self.params[column] for column in ordered_columns], dtype=float)
        return _sigmoid(x.to_numpy(dtype=float) @ beta)

    def to_artifact(self) -> Dict[str, Any]:
        payload = super().to_artifact()
        payload.update(
            {
                "model_type": self.MODEL_TYPE,
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "structured_feature_columns": self.structured_feature_columns,
            }
        )
        return payload

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "HybridSentenceStructuredLogRegDetector":
        detector = cls(
            model_name=str(artifact.get("model_name", SentenceEmbeddingLogRegDetector.DEFAULT_MODEL_NAME)),
            batch_size=int(artifact.get("batch_size", 32)),
        )
        detector.embedding_backend = str(artifact.get("embedding_backend", detector.embedding_backend))
        detector.embedding_dim = int(artifact.get("embedding_dim", 0))
        detector.params = {str(k): float(v) for k, v in artifact.get("params", {}).items()}
        detector.numeric_columns = list(artifact.get("numeric_columns", detector.numeric_columns))
        detector.categorical_columns = list(artifact.get("categorical_columns", detector.categorical_columns))
        detector.structured_feature_columns = list(artifact.get("structured_feature_columns", []))
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
    elif model_type == TextTfidfLogRegDetector.MODEL_TYPE:
        detector = TextTfidfLogRegDetector.from_artifact(artifact)
    elif model_type == EmbeddingLogRegDetector.MODEL_TYPE:
        detector = EmbeddingLogRegDetector.from_artifact(artifact)
    elif model_type == SentenceEmbeddingLogRegDetector.MODEL_TYPE:
        detector = SentenceEmbeddingLogRegDetector.from_artifact(artifact)
    elif model_type == HybridSentenceStructuredLogRegDetector.MODEL_TYPE:
        detector = HybridSentenceStructuredLogRegDetector.from_artifact(artifact)
    else:
        raise ValueError(f"Unsupported detector artifact type: {model_type}")
    return detector, dict(payload.get("metadata", {}))
