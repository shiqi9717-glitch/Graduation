"""Lightweight mitigation utilities for post-evaluation interference detection."""

from .interference_dataset import (
    DEFAULT_DATASET_SUMMARY_NAME,
    DEFAULT_FULL_DATASET_NAME,
    DEFAULT_RELAXED_SPLIT_NAME,
    DEFAULT_STRICT_SPLIT_NAME,
    DESIGN_VERSION_CHOICES,
    build_interference_dataset,
)
from .interference_models import (
    EmbeddingLogRegDetector,
    HybridSentenceStructuredLogRegDetector,
    SentenceEmbeddingLogRegDetector,
    StructuredLogisticDetector,
    TextNGramNBDetector,
    TextTfidfLogRegDetector,
    evaluate_predictions,
    load_detector,
    save_detector,
    select_operating_points,
    threshold_sweep,
)

__all__ = [
    "DEFAULT_DATASET_SUMMARY_NAME",
    "DEFAULT_FULL_DATASET_NAME",
    "DEFAULT_RELAXED_SPLIT_NAME",
    "DEFAULT_STRICT_SPLIT_NAME",
    "DESIGN_VERSION_CHOICES",
    "EmbeddingLogRegDetector",
    "HybridSentenceStructuredLogRegDetector",
    "SentenceEmbeddingLogRegDetector",
    "StructuredLogisticDetector",
    "TextNGramNBDetector",
    "TextTfidfLogRegDetector",
    "build_interference_dataset",
    "evaluate_predictions",
    "load_detector",
    "save_detector",
    "select_operating_points",
    "threshold_sweep",
]
