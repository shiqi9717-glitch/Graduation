"""Scenario analyzer for local open-model white-box probing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .io_utils import save_hidden_state_arrays
from .prompt_builder import SCENARIO_CHOICES, build_prompt

RESEARCH_METADATA_FIELDS = (
    "sample_type",
    "condition_id",
    "authority_level",
    "confidence_level",
    "explicit_wrong_option",
    "is_control",
    "is_hard_negative",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _option(record: Dict[str, Any] | None, key: str) -> str | None:
    if record is None:
        return None
    value = str(record.get(key) or "").strip().upper()
    return value or None


def _is_correct(record: Dict[str, Any] | None) -> bool | None:
    if record is None:
        return None
    predicted = _option(record, "predicted_answer")
    truth = _option(record, "ground_truth")
    if predicted is None or truth is None:
        return None
    return predicted == truth


def compare_scenario_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_scenario = {record["scenario"]: record for record in records}
    baseline = by_scenario.get("baseline")
    interference = by_scenario.get("interference")
    recheck = by_scenario.get("recheck")

    def _delta(first: Dict[str, Any] | None, second: Dict[str, Any] | None, key: str) -> float | None:
        if first is None or second is None:
            return None
        return float(second.get(key, 0.0) - first.get(key, 0.0))

    layer_cosine_shift: Dict[str, float] = {}
    if baseline and interference:
        baseline_arrays = baseline.get("_hidden_state_arrays", {}) or {}
        interference_arrays = interference.get("_hidden_state_arrays", {}) or {}
        for key, baseline_value in baseline_arrays.items():
            if not key.endswith("_final_token"):
                continue
            other = interference_arrays.get(key)
            if other is None:
                continue
            layer_cosine_shift[key] = _cosine(np.asarray(baseline_value), np.asarray(other))

    recheck_layer_cosine_shift: Dict[str, float] = {}
    if interference and recheck:
        interference_arrays = interference.get("_hidden_state_arrays", {}) or {}
        recheck_arrays = recheck.get("_hidden_state_arrays", {}) or {}
        for key, interference_value in interference_arrays.items():
            if not key.endswith("_final_token"):
                continue
            other = recheck_arrays.get(key)
            if other is None:
                continue
            recheck_layer_cosine_shift[key] = _cosine(np.asarray(interference_value), np.asarray(other))

    baseline_answer = _option(baseline, "predicted_answer")
    interference_answer = _option(interference, "predicted_answer")
    recheck_answer = _option(recheck, "predicted_answer")
    baseline_correct = _is_correct(baseline)
    interference_correct = _is_correct(interference)
    recheck_correct = _is_correct(recheck)

    return {
        "sample_id": records[0]["sample_id"] if records else "",
        "model_name": records[0]["model_name"] if records else "",
        **{
            field: records[0].get(field)
            for field in RESEARCH_METADATA_FIELDS
            if records and field in records[0]
        },
        "scenarios": [record["scenario"] for record in records],
        "baseline_predicted_answer": baseline_answer,
        "interference_predicted_answer": interference_answer,
        "recheck_predicted_answer": recheck_answer,
        "baseline_correct": baseline_correct,
        "interference_correct": interference_correct,
        "recheck_correct": recheck_correct,
        "predicted_answer_changed_interference": (
            baseline["predicted_answer"] != interference["predicted_answer"]
            if baseline and interference
            else None
        ),
        "predicted_answer_changed_recheck": (
            interference["predicted_answer"] != recheck["predicted_answer"]
            if interference and recheck
            else None
        ),
        "interference_correct_option_logit_delta": _delta(baseline, interference, "correct_option_logit"),
        "interference_wrong_option_logit_delta": _delta(baseline, interference, "wrong_option_logit"),
        "interference_margin_delta": _delta(baseline, interference, "correct_wrong_margin"),
        "recheck_correct_option_logit_delta_vs_interference": _delta(interference, recheck, "correct_option_logit"),
        "recheck_wrong_option_logit_delta_vs_interference": _delta(interference, recheck, "wrong_option_logit"),
        "recheck_margin_delta_vs_interference": _delta(interference, recheck, "correct_wrong_margin"),
        "recheck_restores_baseline_answer": (
            baseline["predicted_answer"] == recheck["predicted_answer"]
            if baseline and recheck
            else None
        ),
        "harmful_recheck": (
            bool(interference_correct) and recheck_correct is False
            if interference and recheck
            else None
        ),
        "interference_induced_error": (
            bool(baseline_correct) and interference_correct is False
            if baseline and interference
            else None
        ),
        "recheck_recovers_from_interference": (
            interference_correct is False and bool(recheck_correct)
            if interference and recheck
            else None
        ),
        "layer_wise_cosine_shift_baseline_vs_interference": layer_cosine_shift,
        "layer_wise_cosine_shift_interference_vs_recheck": recheck_layer_cosine_shift,
    }


def _finalize_record(
    record: Dict[str, Any],
    hidden_state_dir: Path,
) -> Dict[str, Any]:
    hidden_arrays = record.pop("_hidden_state_arrays", {}) or {}
    hidden_path = save_hidden_state_arrays(
        hidden_state_dir,
        sample_id=str(record["sample_id"]),
        scenario=str(record["scenario"]),
        arrays=hidden_arrays,
    )
    record["hidden_state_path"] = str(hidden_path.resolve())
    return record


def analyze_sample(
    runner,
    sample: Dict[str, Any],
    output_dir: Path,
    scenarios: Iterable[str] = SCENARIO_CHOICES,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_records: List[Dict[str, Any]] = []
    hidden_dir = output_dir / "hidden_states"
    working_sample = dict(sample)
    normalized_scenarios = [str(s) for s in scenarios]
    for scenario in normalized_scenarios:
        prompt = build_prompt(working_sample, scenario)
        record = runner.analyze_prompt(
            prompt=prompt,
            sample_id=str(working_sample.get("sample_id") or "sample"),
            scenario=str(scenario),
            question_text=str(working_sample.get("question_text") or ""),
            prompt_prefix=str(working_sample.get("prompt_prefix") or ""),
            ground_truth=str(working_sample.get("ground_truth") or ""),
            wrong_option=str(working_sample.get("wrong_option") or ""),
        )
        for field in RESEARCH_METADATA_FIELDS:
            if field in working_sample:
                record[field] = working_sample.get(field)
        raw_records.append(record)
        if scenario == "interference":
            working_sample["interference_predicted_answer"] = record.get("predicted_answer")
            working_sample.setdefault("recheck_first_answer", record.get("predicted_answer"))
    comparison = compare_scenario_records(raw_records)
    finalized_records = [_finalize_record(record, hidden_dir) for record in raw_records]
    return finalized_records, comparison


def analyze_samples(
    runner,
    samples: Iterable[Dict[str, Any]],
    output_dir: Path,
    scenarios: Iterable[str] = SCENARIO_CHOICES,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_records: List[Dict[str, Any]] = []
    all_comparisons: List[Dict[str, Any]] = []
    for sample in samples:
        records, comparison = analyze_sample(runner, sample, output_dir=output_dir, scenarios=scenarios)
        all_records.extend(records)
        all_comparisons.append(comparison)
    return all_records, all_comparisons


def summarize_probe_results(
    records: Iterable[Dict[str, Any]],
    comparisons: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    record_list = list(records)
    comparison_list = list(comparisons)

    def _mean(values: List[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(values))

    answer_changed = [
        1.0 for row in comparison_list if row.get("predicted_answer_changed_interference") is True
    ]
    answer_changed_den = [
        row for row in comparison_list if row.get("predicted_answer_changed_interference") is not None
    ]
    answer_restored = [
        1.0 for row in comparison_list if row.get("recheck_restores_baseline_answer") is True
    ]
    answer_restored_den = [
        row for row in comparison_list if row.get("recheck_restores_baseline_answer") is not None
    ]

    summary = {
        "num_records": len(record_list),
        "num_samples": len(comparison_list),
        "answer_changed_rate_baseline_to_interference": (
            float(len(answer_changed) / len(answer_changed_den)) if answer_changed_den else None
        ),
        "answer_restored_rate_interference_to_recheck": (
            float(len(answer_restored) / len(answer_restored_den)) if answer_restored_den else None
        ),
        "mean_interference_correct_option_logit_delta": _mean(
            [float(row["interference_correct_option_logit_delta"]) for row in comparison_list if row.get("interference_correct_option_logit_delta") is not None]
        ),
        "mean_interference_wrong_option_logit_delta": _mean(
            [float(row["interference_wrong_option_logit_delta"]) for row in comparison_list if row.get("interference_wrong_option_logit_delta") is not None]
        ),
        "mean_interference_margin_delta": _mean(
            [float(row["interference_margin_delta"]) for row in comparison_list if row.get("interference_margin_delta") is not None]
        ),
        "mean_recheck_margin_delta_vs_interference": _mean(
            [float(row["recheck_margin_delta_vs_interference"]) for row in comparison_list if row.get("recheck_margin_delta_vs_interference") is not None]
        ),
        "interference_induced_error_rate": (
            float(
                sum(1 for row in comparison_list if row.get("interference_induced_error") is True)
                / sum(1 for row in comparison_list if row.get("interference_induced_error") is not None)
            )
            if any(row.get("interference_induced_error") is not None for row in comparison_list)
            else None
        ),
        "harmful_recheck_rate": (
            float(
                sum(1 for row in comparison_list if row.get("harmful_recheck") is True)
                / sum(1 for row in comparison_list if row.get("harmful_recheck") is not None)
            )
            if any(row.get("harmful_recheck") is not None for row in comparison_list)
            else None
        ),
        "recheck_recovery_rate": (
            float(
                sum(1 for row in comparison_list if row.get("recheck_recovers_from_interference") is True)
                / sum(1 for row in comparison_list if row.get("recheck_recovers_from_interference") is not None)
            )
            if any(row.get("recheck_recovers_from_interference") is not None for row in comparison_list)
            else None
        ),
    }
    by_sample_type: Dict[str, Dict[str, Any]] = {}
    sample_types = sorted({str(row.get("sample_type")) for row in comparison_list if row.get("sample_type")})
    for sample_type in sample_types:
        subset = [row for row in comparison_list if str(row.get("sample_type")) == sample_type]
        by_sample_type[sample_type] = {
            "num_samples": len(subset),
            "answer_changed_rate_baseline_to_interference": (
                float(
                    sum(1 for row in subset if row.get("predicted_answer_changed_interference") is True)
                    / sum(1 for row in subset if row.get("predicted_answer_changed_interference") is not None)
                )
                if any(row.get("predicted_answer_changed_interference") is not None for row in subset)
                else None
            ),
            "mean_interference_margin_delta": _mean(
                [float(row["interference_margin_delta"]) for row in subset if row.get("interference_margin_delta") is not None]
            ),
            "recheck_recovery_rate": (
                float(
                    sum(1 for row in subset if row.get("recheck_recovers_from_interference") is True)
                    / sum(1 for row in subset if row.get("recheck_recovers_from_interference") is not None)
                )
                if any(row.get("recheck_recovers_from_interference") is not None for row in subset)
                else None
            ),
        }
    if by_sample_type:
        summary["by_sample_type"] = by_sample_type
    return summary
