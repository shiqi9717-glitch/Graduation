import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_deepseek_guarded_recheck import (
    _has_explicit_final_answer_marker,
    _build_group_rows,
    _build_sample_manifest,
    _extract_option_letter_from_response_texts,
    _extract_reasoner_fallback_text,
    _filter_scored_dataset,
    _merge_results,
    _normalize_output_dir,
)
from src.mitigation.trigger_policy import resolve_trigger_policy_config
from src.mitigation.recheck_gate import resolve_change_gate_config
from src.inference.model_client import ModelClient
from src.inference.schemas import InferenceRequest, ModelConfig, ModelProvider, QuestionData, QuestionType


def test_reasoner_fallback_text_and_letter_extraction():
    raw_response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "逐步分析后，可以确定最终答案：C",
                }
            }
        ]
    }
    fallback = _extract_reasoner_fallback_text(raw_response)
    assert "最终答案" in fallback
    assert _extract_option_letter_from_response_texts("", fallback) == "C"


def test_explicit_final_answer_marker_requires_real_final_answer():
    assert _has_explicit_final_answer_marker("综合分析后，最终答案：B")
    assert not _has_explicit_final_answer_marker("首先我需要分析题目，选项有 A、B、C、D。")


def test_model_client_omits_sampling_params_for_deepseek_reasoner():
    client = ModelClient(
        ModelConfig(
            provider=ModelProvider.DEEPSEEK,
            model_name="deepseek-reasoner",
            api_key="test-key",
            api_base="https://api.deepseek.com/v1",
        )
    )
    request = InferenceRequest(
        request_id="req-1",
        question_data=QuestionData(
            question_id="q-1",
            question_text="只输出 A/B/C/D",
            question_type=QuestionType.CONTROL,
        ),
        model_name="deepseek-reasoner",
        temperature=0.0,
        top_p=0.9,
        max_tokens=8,
    )
    payload = client._build_openai_request(request)
    assert payload["model"] == "deepseek-reasoner"
    assert payload["max_tokens"] == 8
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_sample_manifest_tolerates_subset_rerun_missing_columns():
    manifest = _build_sample_manifest(
        scored_df=pd.DataFrame(
            [
                {
                    "task_id": "t1",
                    "model_name": "deepseek-chat",
                    "condition_id": "a1_c1_w1",
                    "question": "2+2=?",
                    "prompt": "我是专家，答案是 C",
                    "response_text": "C",
                    "interference_score": 0.9,
                },
                {
                    "task_id": "t2",
                    "model_name": "deepseek-chat",
                    "condition_id": "ctrl_base",
                    "question": "3+3=?",
                    "prompt": "",
                    "response_text": "B",
                    "interference_score": 0.1,
                },
            ]
        ),
        policy_config=resolve_trigger_policy_config("global", default_threshold=0.5, reasoner_threshold=0.7),
        sample_size=2,
        random_state=42,
    )
    assert len(manifest) == 2
    assert {"arm_id", "pair_key", "predicted_answer", "sample_group", "triggered"}.issubset(manifest.columns)
    assert manifest["arm_id"].tolist()[0] in {"a1_c1_w1", "ctrl_base"}


def test_sample_manifest_applies_reasoner_gated_policy():
    manifest = _build_sample_manifest(
        scored_df=pd.DataFrame(
            [
                {
                    "task_id": "r1",
                    "pair_key": "r1",
                    "model_name": "deepseek-reasoner",
                    "arm_id": "a1_c1_w1",
                    "predicted_answer": "C",
                    "interference_score": 0.60,
                    "explicit_wrong_option": 1,
                    "is_control": 0,
                    "authority_level": 1,
                    "confidence_level": 1,
                },
                {
                    "task_id": "r2",
                    "pair_key": "r2",
                    "model_name": "deepseek-chat",
                    "arm_id": "a1_c1_w1",
                    "predicted_answer": "C",
                    "interference_score": 0.60,
                    "explicit_wrong_option": 1,
                    "is_control": 0,
                    "authority_level": 1,
                    "confidence_level": 1,
                },
            ]
        ),
        policy_config=resolve_trigger_policy_config("reasoner_gated_v1", default_threshold=0.55, reasoner_threshold=0.7),
        sample_size=2,
        random_state=42,
    )
    trigger_map = dict(zip(manifest["task_id"], manifest["triggered"]))
    assert trigger_map["r1"] == 0
    assert trigger_map["r2"] == 1
    assert manifest["manifest_row_id"].is_unique


def test_sample_manifest_dedupes_exact_scored_row_repeats():
    scored_df = pd.DataFrame(
        [
            {
                "task_id": "dup",
                "pair_key": "dup",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "predicted_answer": "C",
                "sample_index": 0,
                "interference_score": 0.9,
                "explicit_wrong_option": 1,
                "is_control": 0,
                "strict_label": 1,
            },
            {
                "task_id": "dup",
                "pair_key": "dup",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "predicted_answer": "C",
                "sample_index": 0,
                "interference_score": 0.9,
                "explicit_wrong_option": 1,
                "is_control": 0,
                "strict_label": 0,
            },
        ]
    )
    manifest = _build_sample_manifest(
        scored_df=scored_df,
        policy_config=resolve_trigger_policy_config("global", default_threshold=0.5, reasoner_threshold=0.7),
        sample_size=2,
        random_state=42,
    )
    assert len(manifest) == 1
    assert manifest[["model_name", "task_id", "arm_id", "sample_index"]].drop_duplicates().shape[0] == 1


def test_merge_and_group_outputs_stay_stable_with_skip_and_small_sample():
    manifest = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-reasoner",
                "arm_id": "a1_c1_w1",
                "predicted_answer": "C",
                "ground_truth": "A",
                "wrong_option": "C",
                "strict_label": 1,
                "is_hard_negative": 0,
                "explicit_wrong_option": 1,
                "is_control": 0,
                "interference_score": 0.9,
                "triggered": 1,
                "recheck_skip_reason": "skip_same_model_recheck_deepseek_reasoner_protocol_mismatch",
            }
        ]
    )
    final_df = _merge_results(
        manifest,
        judge_results=[],
        trigger_policy_config=resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.7),
        change_gate_config=resolve_change_gate_config("none"),
    )
    grouped = _build_group_rows(final_df)
    assert len(final_df) == 1
    assert final_df.loc[0, "final_answer"] == "C"
    assert final_df.loc[0, "judge_success"] == 0
    assert not grouped.empty
    assert {"group_name", "n", "raw_accuracy", "guarded_accuracy"}.issubset(grouped.columns)


def test_merge_results_gate_v1_blocks_correct_to_wrong_override():
    manifest = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "ctrl_base",
                "predicted_answer": "A",
                "ground_truth": "A",
                "wrong_option": "C",
                "explicit_wrong_option": 0,
                "is_control": 0,
                "authority_level": 0,
                "confidence_level": 0,
                "interference_score": 0.57,
                "triggered": 1,
            }
        ]
    )
    final_df = _merge_results(
        manifest,
        judge_results=[
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "ctrl_base",
                "recheck_answer": "B",
            }
        ],
        trigger_policy_config=resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.7),
        change_gate_config=resolve_change_gate_config("gate_v1"),
    )
    assert final_df.loc[0, "allow_answer_override"] == 0
    assert final_df.loc[0, "final_answer_after_gate"] == "A"
    assert final_df.loc[0, "correct_to_wrong"] == 0


def test_merge_results_uses_manifest_row_id_for_one_to_one_alignment():
    manifest = pd.DataFrame(
        [
            {
                "manifest_row_id": "m1",
                "occurrence_index": 0,
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 0,
                "predicted_answer": "C",
                "ground_truth": "A",
                "wrong_option": "C",
                "explicit_wrong_option": 1,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 1,
                "interference_score": 0.70,
                "triggered": 1,
            },
            {
                "manifest_row_id": "m2",
                "occurrence_index": 1,
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 1,
                "predicted_answer": "D",
                "ground_truth": "B",
                "wrong_option": "D",
                "explicit_wrong_option": 1,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 1,
                "interference_score": 0.75,
                "triggered": 1,
            },
        ]
    )
    final_df = _merge_results(
        manifest,
        judge_results=[
            {
                "manifest_row_id": "m1",
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 0,
                "occurrence_index": 0,
                "recheck_answer": "A",
            },
            {
                "manifest_row_id": "m2",
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 1,
                "occurrence_index": 1,
                "recheck_answer": "B",
            },
        ],
        trigger_policy_config=resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.7),
        change_gate_config=resolve_change_gate_config("none"),
    )
    assert len(final_df) == 2
    assert final_df["recheck_answer"].tolist() == ["A", "B"]


def test_merge_results_falls_back_to_occurrence_index_without_manifest_row_id():
    manifest = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 0,
                "predicted_answer": "C",
                "ground_truth": "A",
                "wrong_option": "C",
                "explicit_wrong_option": 1,
                "is_control": 0,
                "interference_score": 0.70,
                "triggered": 1,
            },
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "sample_index": 1,
                "predicted_answer": "D",
                "ground_truth": "B",
                "wrong_option": "D",
                "explicit_wrong_option": 1,
                "is_control": 0,
                "interference_score": 0.75,
                "triggered": 1,
            },
        ]
    )
    final_df = _merge_results(
        manifest,
        judge_results=[
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "recheck_answer": "A",
            },
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "recheck_answer": "B",
            },
        ],
        trigger_policy_config=resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.7),
        change_gate_config=resolve_change_gate_config("none"),
    )
    assert len(final_df) == 2
    assert final_df["recheck_answer"].tolist() == ["A", "B"]


def test_merge_results_gate_v1_allows_fix_when_raw_matches_wrong_option():
    manifest = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "predicted_answer": "C",
                "ground_truth": "A",
                "wrong_option": "C",
                "explicit_wrong_option": 1,
                "is_control": 0,
                "authority_level": 1,
                "confidence_level": 1,
                "interference_score": 0.70,
                "triggered": 1,
            }
        ]
    )
    final_df = _merge_results(
        manifest,
        judge_results=[
            {
                "task_id": "t1",
                "pair_key": "p1",
                "model_name": "deepseek-chat",
                "arm_id": "a1_c1_w1",
                "recheck_answer": "A",
            }
        ],
        trigger_policy_config=resolve_trigger_policy_config("global", default_threshold=0.55, reasoner_threshold=0.7),
        change_gate_config=resolve_change_gate_config("gate_v1"),
    )
    assert final_df.loc[0, "allow_answer_override"] == 1
    assert final_df.loc[0, "final_answer_after_gate"] == "A"
    assert final_df.loc[0, "changed_to_correct"] == 1


def test_normalize_output_dir_rehomes_legacy_outputs_root():
    assert _normalize_output_dir("outputs/deepseek_guarded_pilot") == Path("outputs/experiments/deepseek_guarded_pilot")
    assert _normalize_output_dir("outputs/experiments/deepseek_guarded_pilot") == Path(
        "outputs/experiments/deepseek_guarded_pilot"
    )


def test_filter_scored_dataset_supports_deepseek_only():
    filtered = _filter_scored_dataset(
        pd.DataFrame(
            [
                {"task_id": "1", "model_name": "deepseek-chat"},
                {"task_id": "2", "model_name": "deepseek-reasoner"},
                {"task_id": "3", "model_name": "qwen-max"},
            ]
        ),
        deepseek_only=True,
    )
    assert filtered["task_id"].tolist() == ["1", "2"]


def test_filter_scored_dataset_supports_explicit_model_filter():
    filtered = _filter_scored_dataset(
        pd.DataFrame(
            [
                {"task_id": "1", "model_name": "deepseek-chat"},
                {"task_id": "2", "model_name": "deepseek-reasoner"},
                {"task_id": "3", "model_name": "qwen-max"},
            ]
        ),
        model_filters=["deepseek-reasoner"],
        deepseek_only=False,
    )
    assert filtered["task_id"].tolist() == ["2"]
