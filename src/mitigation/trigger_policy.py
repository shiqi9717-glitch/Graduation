"""Trigger policy helpers for detector-driven selective recheck."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import pandas as pd

TRIGGER_POLICY_CHOICES = ("global", "reasoner_gated_v1")
REASONER_MODEL_NAME = "deepseek-reasoner"


@dataclass(frozen=True)
class TriggerPolicyConfig:
    policy_name: str = "global"
    default_threshold: float = 0.55
    reasoner_threshold: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_trigger_policy_config(
    policy_name: str = "global",
    default_threshold: float = 0.55,
    reasoner_threshold: float = 0.70,
) -> TriggerPolicyConfig:
    normalized = str(policy_name or "global").strip().lower()
    if normalized not in TRIGGER_POLICY_CHOICES:
        raise ValueError(f"Unsupported trigger policy: {policy_name}. Expected one of {TRIGGER_POLICY_CHOICES}.")
    return TriggerPolicyConfig(
        policy_name=normalized,
        default_threshold=float(default_threshold),
        reasoner_threshold=float(reasoner_threshold),
    )


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def trigger_threshold_series(df: pd.DataFrame, config: TriggerPolicyConfig) -> pd.Series:
    model_name = _text_series(df, "model_name").str.strip().str.lower()
    thresholds = pd.Series([float(config.default_threshold)] * len(df), index=df.index, dtype="float64")
    if config.policy_name == "reasoner_gated_v1":
        thresholds.loc[model_name.eq(REASONER_MODEL_NAME)] = float(config.reasoner_threshold)
    return thresholds


def apply_trigger_policy(df: pd.DataFrame, config: TriggerPolicyConfig) -> pd.Series:
    scores = _numeric_series(df, "interference_score", 0.0)
    base_trigger = (scores >= float(config.default_threshold)).astype(int)
    if config.policy_name != "reasoner_gated_v1":
        return base_trigger

    model_name = _text_series(df, "model_name").str.strip().str.lower()
    reasoner_mask = model_name.eq(REASONER_MODEL_NAME)
    explicit_wrong = _numeric_series(df, "explicit_wrong_option", 0.0).astype(int) == 1
    non_control = _numeric_series(df, "is_control", 0.0).astype(int) == 0
    authority = _numeric_series(df, "authority_level", 0.0)
    confidence = _numeric_series(df, "confidence_level", 0.0)
    pressure_signal = (authority >= 1.0) | (confidence >= 1.0)
    reasoner_trigger = (
        reasoner_mask
        & (scores >= float(config.reasoner_threshold))
        & explicit_wrong
        & non_control
        & pressure_signal
    )
    trigger = base_trigger.copy()
    trigger.loc[reasoner_mask] = reasoner_trigger.loc[reasoner_mask].astype(int)
    return trigger.astype(int)
