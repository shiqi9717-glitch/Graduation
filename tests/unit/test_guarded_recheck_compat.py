import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_deepseek_guarded_recheck import (
    _has_explicit_final_answer_marker,
    _extract_option_letter_from_response_texts,
    _extract_reasoner_fallback_text,
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
