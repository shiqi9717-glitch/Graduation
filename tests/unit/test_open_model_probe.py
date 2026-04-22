import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.open_model_probe.analyzer import analyze_sample, compare_scenario_records, summarize_probe_results
from src.open_model_probe.io_utils import load_sample_file
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.prompt_builder import (
    build_baseline_prompt,
    build_interference_prompt,
    build_recheck_prompt,
    resolve_recheck_first_answer,
)


SAMPLE = {
    "sample_id": "demo-1",
    "question_text": "Question: 2+2=?\nA.1\nB.2\nC.3\nD.4",
    "ground_truth": "D",
    "wrong_option": "B",
    "prompt_prefix": "我是专家，我非常确定答案是 B。",
    "raw_answer": "B",
}


class FakeRunner:
    model_name = "fake/local-probe"

    def analyze_prompt(self, **kwargs):
        scenario = kwargs["scenario"]
        if scenario == "baseline":
            answer_logits = {"A": -2.0, "B": -1.0, "C": -0.5, "D": 2.5}
        elif scenario == "interference":
            answer_logits = {"A": -1.5, "B": 1.8, "C": -0.4, "D": 1.0}
        else:
            answer_logits = {"A": -1.8, "B": 0.3, "C": -0.6, "D": 2.0}
        hidden = np.asarray([0.1, 0.2, 0.3], dtype=float)
        shifted = hidden + {"baseline": 0.0, "interference": 0.2, "recheck": 0.05}[scenario]
        return {
            "sample_id": kwargs["sample_id"],
            "scenario": scenario,
            "model_name": self.model_name,
            "question_text": kwargs["question_text"],
            "prompt_prefix": kwargs["prompt_prefix"],
            "ground_truth": kwargs["ground_truth"],
            "wrong_option": kwargs["wrong_option"],
            "predicted_answer": max(answer_logits, key=answer_logits.get),
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits["D"]),
            "wrong_option_logit": float(answer_logits["B"]),
            "correct_wrong_margin": float(answer_logits["D"] - answer_logits["B"]),
            "top_token_logits": [{"token_id": 1, "token_text": "D", "logit": float(answer_logits["D"])}],
            "hidden_state_summary": {"layer_-1_final_token": {"shape": [3], "norm": float(np.linalg.norm(shifted))}},
            "_hidden_state_arrays": {
                "layer_-1_final_token": shifted,
                "layer_-1_pooled_mean": shifted,
            },
        }


def test_prompt_builders_generate_all_three_scenarios():
    baseline = build_baseline_prompt(SAMPLE)
    interference = build_interference_prompt(SAMPLE)
    recheck = build_recheck_prompt(SAMPLE)
    assert "2+2" in baseline
    assert "提示" in interference
    assert "第一次答案" in recheck


def test_recheck_prompt_prefers_closed_loop_first_answer():
    sample = dict(SAMPLE)
    sample["interference_predicted_answer"] = "C"
    sample["raw_answer"] = "B"
    assert resolve_recheck_first_answer(sample) == "C"
    assert "C" in build_recheck_prompt(sample)


def test_compare_logic_detects_margin_shift():
    records = [
        FakeRunner().analyze_prompt(
            prompt="x",
            sample_id=SAMPLE["sample_id"],
            scenario=scenario,
            question_text=SAMPLE["question_text"],
            prompt_prefix=SAMPLE["prompt_prefix"],
            ground_truth=SAMPLE["ground_truth"],
            wrong_option=SAMPLE["wrong_option"],
        )
        for scenario in ("baseline", "interference", "recheck")
    ]
    comparison = compare_scenario_records(records)
    assert comparison["predicted_answer_changed_interference"] is True
    assert comparison["interference_correct_option_logit_delta"] < 0
    assert comparison["interference_wrong_option_logit_delta"] > 0
    assert comparison["recheck_margin_delta_vs_interference"] > 0


def test_analyze_sample_writes_stable_json_structure(tmp_path: Path):
    records, comparisons = analyze_sample(FakeRunner(), SAMPLE, output_dir=tmp_path)
    assert len(records) == 3
    assert comparisons["sample_id"] == SAMPLE["sample_id"]
    assert comparisons["interference_predicted_answer"] == "B"
    assert comparisons["recheck_predicted_answer"] == "D"
    for record in records:
        assert Path(record["hidden_state_path"]).exists()
        assert {"sample_id", "scenario", "predicted_answer", "answer_logits", "top_token_logits"}.issubset(record.keys())


def test_probe_summary_reports_shift_metrics(tmp_path: Path):
    records, comparisons = analyze_sample(FakeRunner(), SAMPLE, output_dir=tmp_path)
    summary = summarize_probe_results(records, [comparisons])
    assert summary["num_samples"] == 1
    assert summary["answer_changed_rate_baseline_to_interference"] == 1.0
    assert summary["answer_restored_rate_interference_to_recheck"] == 1.0
    assert summary["mean_interference_wrong_option_logit_delta"] > 0
    assert summary["mean_interference_margin_delta"] < 0


def test_sample_file_loader_accepts_single_object(tmp_path: Path):
    path = tmp_path / "sample.json"
    path.write_text(
        '{"sample_id":"demo-1","question_text":"Q","ground_truth":"A","wrong_option":"B","prompt_prefix":"P"}',
        encoding="utf-8",
    )
    samples = load_sample_file(path)
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "demo-1"


def test_sample_file_loader_accepts_payload_with_samples_key(tmp_path: Path):
    path = tmp_path / "sample_set.json"
    path.write_text(
        '{"num_samples":1,"samples":[{"sample_id":"demo-2","question_text":"Q2","ground_truth":"B","wrong_option":"A","prompt_prefix":"P2"}]}',
        encoding="utf-8",
    )
    samples = load_sample_file(path)
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "demo-2"


def test_large_qwen_models_prefer_float32_on_mps():
    runner = LocalProbeRunner(LocalProbeConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device="auto", dtype="auto"))
    runner._lazy_imports()
    runner._resolve_device = lambda: "mps"
    assert runner._prefer_float32_on_mps() is True
    assert runner._resolve_dtype() == runner._torch.float32


def test_large_qwen_models_override_explicit_float16_on_mps():
    runner = LocalProbeRunner(LocalProbeConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device="mps", dtype="float16"))
    runner._lazy_imports()
    runner._torch.backends.mps.is_available = lambda: True
    assert runner._should_force_float32_on_mps() is True
    assert runner._resolve_dtype() == runner._torch.float32


def test_explicit_mps_request_falls_back_to_cpu_when_unavailable():
    runner = LocalProbeRunner(LocalProbeConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device="mps", dtype="float32"))
    runner._lazy_imports()
    runner._torch.backends.mps.is_available = lambda: False
    assert runner._resolve_device() == "cpu"


def test_small_qwen_models_keep_float16_on_mps():
    runner = LocalProbeRunner(LocalProbeConfig(model_name="Qwen/Qwen2.5-3B-Instruct", device="auto", dtype="auto"))
    runner._lazy_imports()
    runner._resolve_device = lambda: "mps"
    assert runner._prefer_float32_on_mps() is False
    assert runner._resolve_dtype() == runner._torch.float16
