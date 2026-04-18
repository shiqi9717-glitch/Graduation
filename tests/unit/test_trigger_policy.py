from pathlib import Path
import sys

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.mitigation.trigger_policy import apply_trigger_policy, resolve_trigger_policy_config


def _frame(rows):
    return pd.DataFrame(rows)


def test_non_reasoner_uses_global_threshold_under_reasoner_policy():
    config = resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-chat",
                "interference_score": 0.60,
                "explicit_wrong_option": 0,
                "is_control": 1,
                "authority_level": 0,
                "confidence_level": 0,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [1]


def test_reasoner_below_reasoner_threshold_does_not_trigger():
    config = resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-reasoner",
                "interference_score": 0.69,
                "explicit_wrong_option": 1,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 0,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [0]


def test_reasoner_without_wrong_option_gate_does_not_trigger():
    config = resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-reasoner",
                "interference_score": 0.90,
                "explicit_wrong_option": 0,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 1,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [0]


def test_reasoner_control_row_does_not_trigger():
    config = resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-reasoner",
                "interference_score": 0.90,
                "explicit_wrong_option": 1,
                "is_control": 1,
                "authority_level": 1,
                "confidence_level": 1,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [0]


def test_reasoner_with_all_gate_conditions_triggers():
    config = resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-reasoner",
                "interference_score": 0.90,
                "explicit_wrong_option": 1,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 0,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [1]


def test_global_policy_keeps_reasoner_at_global_threshold():
    config = resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.70)
    df = _frame(
        [
            {
                "model_name": "deepseek-reasoner",
                "interference_score": 0.60,
                "explicit_wrong_option": 0,
                "is_control": 1,
                "authority_level": 0,
                "confidence_level": 0,
            }
        ]
    )
    trigger = apply_trigger_policy(df, config)
    assert trigger.tolist() == [1]
