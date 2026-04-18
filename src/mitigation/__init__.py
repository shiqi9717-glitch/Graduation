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
from .recheck_gate import (
    CHANGE_GATE_CHOICES,
    ChangeGateConfig,
    apply_change_gate,
    resolve_change_gate_config,
)
from .trigger_policy import (
    REASONER_MODEL_NAME,
    TRIGGER_POLICY_CHOICES,
    TriggerPolicyConfig,
    apply_trigger_policy,
    resolve_trigger_policy_config,
    trigger_threshold_series,
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
    "CHANGE_GATE_CHOICES",
    "ChangeGateConfig",
    "apply_change_gate",
    "resolve_change_gate_config",
    "REASONER_MODEL_NAME",
    "TRIGGER_POLICY_CHOICES",
    "TriggerPolicyConfig",
    "apply_trigger_policy",
    "resolve_trigger_policy_config",
    "trigger_threshold_series",
]
