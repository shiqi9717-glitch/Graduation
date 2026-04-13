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
    _merge_results,
    _normalize_output_dir,
)
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
        threshold=0.5,
        sample_size=2,
        random_state=42,
    )
    assert len(manifest) == 2
    assert {"arm_id", "pair_key", "predicted_answer", "sample_group", "triggered"}.issubset(manifest.columns)
    assert manifest["arm_id"].tolist()[0] in {"a1_c1_w1", "ctrl_base"}


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
    final_df = _merge_results(manifest, judge_results=[])
    grouped = _build_group_rows(final_df)
    assert len(final_df) == 1
    assert final_df.loc[0, "final_answer"] == "C"
    assert final_df.loc[0, "judge_success"] == 0
    assert not grouped.empty
    assert {"group_name", "n", "raw_accuracy", "guarded_accuracy"}.issubset(grouped.columns)


def test_normalize_output_dir_rehomes_legacy_outputs_root():
    assert _normalize_output_dir("outputs/deepseek_guarded_pilot") == Path("outputs/experiments/deepseek_guarded_pilot")
    assert _normalize_output_dir("outputs/experiments/deepseek_guarded_pilot") == Path(
        "outputs/experiments/deepseek_guarded_pilot"
    )
