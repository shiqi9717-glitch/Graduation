"""Runtime-only answer override gate for second-pass recheck."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import pandas as pd

from .trigger_policy import REASONER_MODEL_NAME, TriggerPolicyConfig, trigger_threshold_series

CHANGE_GATE_CHOICES = ("none", "gate_v1")


@dataclass(frozen=True)
class ChangeGateConfig:
    gate_name: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_change_gate_config(gate_name: str = "none") -> ChangeGateConfig:
    normalized = str(gate_name or "none").strip().lower()
    if normalized not in CHANGE_GATE_CHOICES:
        raise ValueError(f"Unsupported change gate: {gate_name}. Expected one of {CHANGE_GATE_CHOICES}.")
    return ChangeGateConfig(gate_name=normalized)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def apply_change_gate(
    df: pd.DataFrame,
    gate_config: ChangeGateConfig,
    trigger_policy_config: TriggerPolicyConfig,
) -> pd.DataFrame:
    working = df.copy()
    working["raw_answer"] = _text_series(working, "raw_answer").str.upper().str.strip()
    if "raw_answer" not in df.columns:
        working["raw_answer"] = _text_series(working, "predicted_answer").str.upper().str.strip()
    working["recheck_answer"] = _text_series(working, "recheck_answer").str.upper().str.strip()
    working["wrong_option"] = _text_series(working, "wrong_option").str.upper().str.strip()
    working["triggered"] = _numeric_series(working, "triggered", 0).astype(int)
    working["trigger_threshold_for_row"] = trigger_threshold_series(working, trigger_policy_config)
    working["score_margin_above_trigger"] = (
        _numeric_series(working, "interference_score", 0.0) - working["trigger_threshold_for_row"].astype(float)
    )
    working["allow_answer_override"] = 0
    working["override_reason"] = ""
    working["final_answer_after_gate"] = working["raw_answer"]

    valid_recheck = working["recheck_answer"].isin(["A", "B", "C", "D"])
    changed_candidate = (
        (working["triggered"].astype(int) == 1)
        & valid_recheck
        & working["recheck_answer"].ne(working["raw_answer"])
    )
    if gate_config.gate_name == "none":
        working.loc[working["triggered"].astype(int) != 1, "override_reason"] = "not_triggered"
        working.loc[(working["triggered"].astype(int) == 1) & ~valid_recheck, "override_reason"] = "invalid_recheck_answer"
        working.loc[
            (working["triggered"].astype(int) == 1) & valid_recheck & working["recheck_answer"].eq(working["raw_answer"]),
            "override_reason",
        ] = "same_as_raw_answer"
        working.loc[changed_candidate, "allow_answer_override"] = 1
        working.loc[changed_candidate, "override_reason"] = "no_gate_apply_recheck_answer"
        working.loc[changed_candidate, "final_answer_after_gate"] = working.loc[changed_candidate, "recheck_answer"]
        return working

    model_name = _text_series(working, "model_name").str.strip().str.lower()
    is_reasoner = model_name.eq(REASONER_MODEL_NAME)
    explicit_wrong = _numeric_series(working, "explicit_wrong_option", 0.0).astype(int) == 1
    is_control = _numeric_series(working, "is_control", 0.0).astype(int) == 1
    authority = _numeric_series(working, "authority_level", 0.0)
    confidence = _numeric_series(working, "confidence_level", 0.0)
    pressure_signal = (authority >= 1.0) | (confidence >= 1.0)
    raw_matches_wrong = working["wrong_option"].isin(["A", "B", "C", "D"]) & working["raw_answer"].eq(working["wrong_option"])
    recheck_matches_wrong = working["wrong_option"].isin(["A", "B", "C", "D"]) & working["recheck_answer"].eq(working["wrong_option"])
    medium_margin = working["score_margin_above_trigger"].astype(float) >= 0.08
    high_margin = working["score_margin_above_trigger"].astype(float) >= 0.15
    very_high_margin = working["score_margin_above_trigger"].astype(float) >= 0.22

    working.loc[working["triggered"].astype(int) != 1, "override_reason"] = "not_triggered"
    working.loc[(working["triggered"].astype(int) == 1) & ~valid_recheck, "override_reason"] = "invalid_recheck_answer"
    working.loc[
        (working["triggered"].astype(int) == 1) & valid_recheck & working["recheck_answer"].eq(working["raw_answer"]),
        "override_reason",
    ] = "same_as_raw_answer"

    changed = changed_candidate
    working.loc[changed & recheck_matches_wrong & ~raw_matches_wrong, "override_reason"] = "block_recheck_moves_to_wrong_option"
    working.loc[changed & is_control, "override_reason"] = "block_control_row_keep_raw"

    allow_explicit_raw_wrong_nonreasoner = (
        changed
        & ~is_control
        & explicit_wrong
        & raw_matches_wrong
        & ~recheck_matches_wrong
        & ~is_reasoner
        & (medium_margin | pressure_signal)
    )
    allow_explicit_raw_wrong_reasoner = (
        changed
        & ~is_control
        & explicit_wrong
        & raw_matches_wrong
        & ~recheck_matches_wrong
        & is_reasoner
        & high_margin
        & pressure_signal
    )
    allow_explicit_nonwrong_nonreasoner = (
        changed
        & ~is_control
        & explicit_wrong
        & ~raw_matches_wrong
        & ~recheck_matches_wrong
        & ~is_reasoner
        & high_margin
        & pressure_signal
    )
    allow_explicit_nonwrong_reasoner = (
        changed
        & ~is_control
        & explicit_wrong
        & ~raw_matches_wrong
        & ~recheck_matches_wrong
        & is_reasoner
        & very_high_margin
        & pressure_signal
    )
    allow_nonexplicit_nonreasoner = (
        changed
        & ~is_control
        & ~explicit_wrong
        & ~recheck_matches_wrong
        & ~is_reasoner
        & very_high_margin
        & pressure_signal
    )

    allow_masks = {
        "allow_explicit_wrong_raw_matches_wrong_nonreasoner": allow_explicit_raw_wrong_nonreasoner,
        "allow_explicit_wrong_raw_matches_wrong_reasoner_strict": allow_explicit_raw_wrong_reasoner,
        "allow_explicit_wrong_high_margin_nonreasoner": allow_explicit_nonwrong_nonreasoner,
        "allow_explicit_wrong_very_high_margin_reasoner": allow_explicit_nonwrong_reasoner,
        "allow_nonexplicit_very_high_margin_nonreasoner": allow_nonexplicit_nonreasoner,
    }
    combined_allow = pd.Series([False] * len(working), index=working.index, dtype=bool)
    for reason, mask in allow_masks.items():
        apply_mask = mask & ~combined_allow
        working.loc[apply_mask, "allow_answer_override"] = 1
        working.loc[apply_mask, "override_reason"] = reason
        combined_allow |= apply_mask

    unresolved_changed = changed & ~combined_allow & working["override_reason"].eq("")
    working.loc[unresolved_changed & ~explicit_wrong, "override_reason"] = "block_nonexplicit_condition_conservative_keep_raw"
    working.loc[unresolved_changed & explicit_wrong & is_reasoner, "override_reason"] = "block_reasoner_requires_stronger_evidence"
    working.loc[unresolved_changed & explicit_wrong & ~is_reasoner, "override_reason"] = "block_score_margin_too_low_keep_raw"

    allow_override = working["allow_answer_override"].astype(int) == 1
    working.loc[allow_override, "final_answer_after_gate"] = working.loc[allow_override, "recheck_answer"]
    return working
