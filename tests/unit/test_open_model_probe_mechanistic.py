import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.open_model_probe.mechanistic import (
    build_anchor_head_expansion_sets,
    build_drop_wrong_option_case_rows,
    build_sample_case_analysis,
    select_anchor_expansion_heads,
    summarize_drop_wrong_option_cases,
    transition_label,
)


def _record(scenario: str, predicted: str, correct: str, wrong: str, correct_logit: float, wrong_logit: float):
    return {
        "sample_id": "s1",
        "scenario": scenario,
        "model_name": "fake",
        "sample_type": "strict_positive",
        "condition_id": "a1_c1_w1",
        "ground_truth": correct,
        "wrong_option": wrong,
        "predicted_answer": predicted,
        "correct_option_logit": correct_logit,
        "wrong_option_logit": wrong_logit,
        "correct_wrong_margin": correct_logit - wrong_logit,
        "layer_logit_lens": [
            {
                "layer_index": 0,
                "answer_logits": {"A": correct_logit, "B": wrong_logit, "C": -1.0, "D": -2.0},
                "predicted_answer": predicted,
                "ranked_options": [predicted, wrong if predicted != wrong else correct, "C", "D"],
            }
        ],
        "attention_summary": [
            {
                "layer_index": 0,
                "prefix_attention_mean": 0.2,
                "wrong_option_prefix_attention_mean": 0.1,
            }
        ],
        "_hidden_state_arrays": {
            "layer_0_final_token": np.asarray([1.0, 0.0], dtype=float),
            "layer_0_pooled_mean": np.asarray([0.5, 0.5], dtype=float),
        },
    }


def test_transition_label_covers_core_paths():
    assert transition_label(True, False) == "baseline_correct_to_interference_wrong"
    assert transition_label(False, False) == "baseline_wrong_to_interference_wrong"
    assert transition_label(True, True) == "baseline_correct_to_interference_correct"


def test_build_sample_case_analysis_emits_per_layer_drift():
    baseline = _record("baseline", "A", "A", "B", 4.0, 1.0)
    interference = _record("interference", "B", "A", "B", 1.0, 3.5)
    interference["_hidden_state_arrays"]["layer_0_final_token"] = np.asarray([0.0, 1.0], dtype=float)
    interference["_hidden_state_arrays"]["layer_0_pooled_mean"] = np.asarray([0.0, 1.0], dtype=float)
    recheck = _record("recheck", "B", "A", "B", 1.5, 3.0)
    row = build_sample_case_analysis(
        {
            "baseline": baseline,
            "interference": interference,
            "recheck": recheck,
        }
    )
    assert row["transition_label"] == "baseline_correct_to_interference_wrong"
    assert row["interference_margin_delta"] < 0
    assert len(row["per_layer"]) == 1
    assert row["per_layer"][0]["baseline_to_interference_final_cosine"] == 0.0


def test_anchor_expansion_keeps_anchor_and_grows_by_target_sizes():
    head_summary = pd.DataFrame(
        [
            {"patch_layer_index": 31, "head_index": 7, "restore_correct_rate": 0.4, "mean_margin_gain": 2.5},
            {"patch_layer_index": 32, "head_index": 7, "restore_correct_rate": 0.3, "mean_margin_gain": 0.8},
            {"patch_layer_index": 34, "head_index": 12, "restore_correct_rate": 0.14, "mean_margin_gain": 1.5},
            {"patch_layer_index": 34, "head_index": 14, "restore_correct_rate": 0.14, "mean_margin_gain": 1.0},
            {"patch_layer_index": 31, "head_index": 5, "restore_correct_rate": 0.14, "mean_margin_gain": 0.5},
        ]
    )
    ordered = select_anchor_expansion_heads(
        head_summary,
        anchor_head=(31, 7),
        allowed_layers=[31, 32, 34],
        max_heads=8,
    )
    assert ordered[0] == (31, 7)
    expansions = build_anchor_head_expansion_sets(ordered, target_sizes=(1, 2, 4, 8))
    assert [item["expansion_size"] for item in expansions] == [1, 2, 4, 5]
    assert expansions[1]["head_set"] == [(31, 7), (32, 7)]


def test_drop_wrong_option_case_rows_attach_best_head_and_expansion():
    sample_notes = [
        {
            "sample_id": "s1",
            "sample_type": "strict_positive",
            "condition_id": "a2_c1_w1",
            "baseline_answer": "A",
            "interference_answer": "B",
            "ground_truth": "A",
            "wrong_option": "B",
        }
    ]
    ablation_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "ablation": "none",
                "predicted_answer": "B",
                "is_correct": False,
                "wrong_option_follow": True,
                "late_layer_margin_delta": -4.0,
            },
            {
                "sample_id": "s1",
                "ablation": "drop_wrong_option",
                "predicted_answer": "A",
                "is_correct": True,
                "wrong_option_follow": False,
                "late_layer_margin_delta": -1.5,
            },
        ]
    )
    head_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "patch_layer_index": 31,
                "head_index": 7,
                "restored_correct": True,
                "mean_margin_gain": 2.0,
            }
        ]
    )
    expansion_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "expansion_size": 2,
                "head_set_label": "L31H7+L32H7",
                "restored_correct": True,
                "mean_margin_gain": 3.5,
            }
        ]
    )
    case_df = build_drop_wrong_option_case_rows(
        sample_notes,
        ablation_df=ablation_df,
        head_df=head_df,
        expansion_df=expansion_df,
    )
    assert case_df.iloc[0]["explicit_wrong_option_cue_dependence"] == "high"
    assert case_df.iloc[0]["best_single_head_label"] == "L31H7"
    assert case_df.iloc[0]["best_expansion_label"] == "L31H7+L32H7"
    summary = summarize_drop_wrong_option_cases(case_df)
    assert summary.iloc[0]["best_expansion_restore_rate"] == 1.0
