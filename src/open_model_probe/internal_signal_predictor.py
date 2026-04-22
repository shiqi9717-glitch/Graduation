"""Internal-signal predictor baselines for local probe research."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .prompt_builder import build_prompt


TARGET_LABEL = "is_baseline_correct_to_interference_wrong"


@dataclass(frozen=True)
class PredictorResult:
    feature_set_name: str
    feature_columns: tuple[str, ...]
    num_samples: int
    num_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None


@dataclass(frozen=True)
class PredictorModel:
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    params: np.ndarray


@dataclass(frozen=True)
class RegularizedSearchResult:
    feature_set_name: str
    model_type: str
    alpha: float
    l1_wt: float
    num_features: int
    num_nonzero_features: int
    cv_result: PredictorResult


_TOKENIZER_CACHE: Dict[str, Any] = {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_pretrained_source(model_name: str) -> str:
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = os.path.join(cache_root, "models--" + str(model_name).replace("/", "--"))
    refs_main = os.path.join(cache_dir, "refs", "main")
    if os.path.exists(refs_main):
        revision = Path(refs_main).read_text(encoding="utf-8").strip()
        snapshot_dir = os.path.join(cache_dir, "snapshots", revision)
        if os.path.isdir(snapshot_dir):
            return snapshot_dir
    return str(model_name)


def _load_tokenizer(model_name: str):
    key = str(model_name)
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    from transformers import AutoTokenizer

    source = _resolve_pretrained_source(key)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True, local_files_only=True)
    _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int] | None:
    if not subsequence:
        return None
    needle = list(subsequence)
    haystack = list(sequence)
    limit = len(haystack) - len(needle) + 1
    for start in range(max(limit, 0)):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def _extract_span_positions(span: tuple[int, int] | None) -> List[int]:
    if span is None:
        return []
    start, end = span
    return list(range(int(start), int(end)))


def _candidate_option_token_ids(tokenizer, option: str) -> List[int]:
    candidates = []
    for text in (option, f" {option}", f"\n{option}"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            candidates.append(token_ids[0])
    return sorted(set(candidates))


def _extract_token_positions(input_ids: List[int], token_ids: Iterable[int]) -> List[int]:
    token_id_set = {int(token_id) for token_id in token_ids}
    return [idx for idx, token_id in enumerate(input_ids) if int(token_id) in token_id_set]


def _summarize_head_attention(array: np.ndarray, positions: List[int], head_index: int) -> float | None:
    if not positions:
        return None
    if array.ndim != 2 or head_index < 0 or head_index >= array.shape[0]:
        return None
    return float(array[int(head_index), positions].mean())


def _safe_attention_positions(record: Dict[str, Any], *, model_name: str) -> Dict[str, List[int]]:
    tokenizer = _load_tokenizer(model_name)
    sample = {
        "question_text": record.get("question_text"),
        "prompt_prefix": record.get("prompt_prefix"),
        "ground_truth": record.get("ground_truth"),
        "wrong_option": record.get("wrong_option"),
    }
    prompt = build_prompt(sample, "interference")
    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix_ids = tokenizer.encode(str(record.get("prompt_prefix") or ""), add_special_tokens=False)
    prefix_span = _find_subsequence(full_ids, prefix_ids)
    prefix_positions = _extract_span_positions(prefix_span)
    wrong_option = str(record.get("wrong_option") or "").strip().upper()
    wrong_ids = _candidate_option_token_ids(tokenizer, wrong_option) if wrong_option else []
    wrong_positions = _extract_token_positions(full_ids, wrong_ids)
    prefix_set = set(prefix_positions)
    wrong_prefix_positions = [pos for pos in wrong_positions if pos in prefix_set]
    return {
        "prefix_positions": prefix_positions,
        "wrong_prefix_positions": wrong_prefix_positions,
    }


def _safe_mean(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not valid:
        return None
    return float(np.mean(valid))


def _safe_min(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not valid:
        return None
    return float(np.min(valid))


def _safe_max(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not valid:
        return None
    return float(np.max(valid))


def _aggregate_focus_layers(per_layer: Sequence[Dict[str, Any]], focus_layers: Sequence[int]) -> Dict[str, float | None]:
    focus = [row for row in per_layer if int(row.get("layer_index", -999)) in {int(layer) for layer in focus_layers}]
    margin_deltas = [row.get("interference_margin_delta") for row in focus]
    final_cosines = [row.get("baseline_to_interference_final_cosine") for row in focus]
    correct_ranks = [row.get("interference_correct_rank") for row in focus]
    wrong_ranks = [row.get("interference_wrong_rank") for row in focus]
    prefix_attention = [row.get("interference_prefix_attention_mean") for row in focus]
    wrong_prefix_attention = [row.get("interference_wrong_option_prefix_attention_mean") for row in focus]

    return {
        "late_mean_margin_delta": _safe_mean(margin_deltas),
        "late_min_margin_delta": _safe_min(margin_deltas),
        "late_max_margin_delta": _safe_max(margin_deltas),
        "late_mean_final_cosine_shift": (
            float(1.0 - _safe_mean(final_cosines)) if _safe_mean(final_cosines) is not None else None
        ),
        "late_max_final_cosine_shift": (
            float(1.0 - _safe_min(final_cosines)) if _safe_min(final_cosines) is not None else None
        ),
        "late_mean_correct_rank": _safe_mean(correct_ranks),
        "late_mean_wrong_rank": _safe_mean(wrong_ranks),
        "late_mean_prefix_attention": _safe_mean(prefix_attention),
        "late_mean_wrong_prefix_attention": _safe_mean(wrong_prefix_attention),
        "late_max_wrong_prefix_attention": _safe_max(wrong_prefix_attention),
    }


def build_internal_signal_dataset(
    *,
    probe_comparisons: Sequence[Dict[str, Any]],
    sample_cases: Sequence[Dict[str, Any]],
    focus_layers: Sequence[int] = (31, 32, 33, 34, 35),
    baseline_correct_only: bool = True,
) -> pd.DataFrame:
    comparison_map = {str(row["sample_id"]): dict(row) for row in probe_comparisons}
    rows: List[Dict[str, Any]] = []
    for case in sample_cases:
        sample_id = str(case["sample_id"])
        comparison = comparison_map.get(sample_id, {})
        baseline_correct = bool(case.get("baseline_correct"))
        if baseline_correct_only and not baseline_correct:
            continue

        focus_stats = _aggregate_focus_layers(case.get("per_layer", []), focus_layers)
        row = {
            "sample_id": sample_id,
            "model_name": case.get("model_name"),
            "sample_type": case.get("sample_type"),
            "condition_id": case.get("condition_id"),
            "authority_level": float(case.get("authority_level") or 0.0),
            "confidence_level": float(case.get("confidence_level") or 0.0),
            "explicit_wrong_option": float(case.get("explicit_wrong_option") or 0.0),
            "is_control": float(case.get("is_control") or 0.0),
            "is_hard_negative": float(case.get("is_hard_negative") or 0.0),
            "baseline_correct": float(bool(case.get("baseline_correct"))),
            "interference_correct": float(bool(case.get("interference_correct"))),
            "recheck_correct": float(bool(case.get("recheck_correct"))),
            "transition_label": case.get("transition_label"),
            TARGET_LABEL: float(case.get("transition_label") == "baseline_correct_to_interference_wrong"),
            "output_interference_margin_delta": case.get("interference_margin_delta"),
            "output_baseline_correct_option_logit": case.get("baseline_correct_option_logit"),
            "output_baseline_wrong_option_logit": case.get("baseline_wrong_option_logit"),
            "output_interference_correct_option_logit": case.get("interference_correct_option_logit"),
            "output_interference_wrong_option_logit": case.get("interference_wrong_option_logit"),
            "output_recheck_margin_delta_vs_interference": case.get("recheck_margin_delta_vs_interference"),
            "probe_answer_changed_interference": float(bool(comparison.get("predicted_answer_changed_interference"))),
            "probe_answer_changed_recheck": float(bool(comparison.get("predicted_answer_changed_recheck"))),
            "probe_recheck_restores_baseline_answer": float(bool(comparison.get("recheck_restores_baseline_answer"))),
            "probe_harmful_recheck": float(bool(comparison.get("harmful_recheck"))),
            "probe_recheck_recovers_from_interference": float(bool(comparison.get("recheck_recovers_from_interference"))),
            **focus_stats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if col not in {"sample_id", "model_name", "sample_type", "condition_id", "transition_label"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def feature_set_definitions() -> Dict[str, List[str]]:
    return {
        "internal_signal": [
            "late_mean_margin_delta",
            "late_min_margin_delta",
            "late_mean_final_cosine_shift",
            "late_max_final_cosine_shift",
            "late_mean_correct_rank",
            "late_mean_wrong_rank",
            "late_mean_prefix_attention",
            "late_mean_wrong_prefix_attention",
            "late_max_wrong_prefix_attention",
        ],
        "output_only": [
            "output_interference_margin_delta",
            "output_baseline_correct_option_logit",
            "output_baseline_wrong_option_logit",
            "output_interference_correct_option_logit",
            "output_interference_wrong_option_logit",
        ],
        "external_only": [
            "authority_level",
            "confidence_level",
            "explicit_wrong_option",
            "is_control",
            "is_hard_negative",
        ],
    }


def ablation_feature_set_definitions() -> Dict[str, List[str]]:
    return {
        "internal_signal_full": list(feature_set_definitions()["internal_signal"]),
        "margin_only": [
            "late_mean_margin_delta",
            "late_min_margin_delta",
        ],
        "cosine_only": [
            "late_mean_final_cosine_shift",
            "late_max_final_cosine_shift",
        ],
        "attention_only": [
            "late_mean_prefix_attention",
            "late_mean_wrong_prefix_attention",
            "late_max_wrong_prefix_attention",
        ],
        "minimal_mechanistic": [
            "late_min_margin_delta",
            "late_max_final_cosine_shift",
            "late_mean_wrong_prefix_attention",
            "late_mean_correct_rank",
        ],
    }


def runtime_safe_feature_set_definitions() -> Dict[str, List[str]]:
    output_only = [
        "safe_interference_wrong_option_logit",
        "safe_baseline_wrong_option_logit",
        "safe_wrong_option_logit_delta",
        "safe_interference_wrong_rank",
        "safe_baseline_wrong_rank",
        "safe_interference_top1_margin",
        "safe_baseline_top1_margin",
        "safe_interference_top1_is_wrong",
        "safe_baseline_top1_is_wrong",
    ]
    attention_only = [
        "late_mean_prefix_attention",
        "late_mean_wrong_prefix_attention",
        "late_max_wrong_prefix_attention",
    ]
    head_only = [
        "fixed_head_mean_prefix_attention",
        "fixed_head_max_prefix_attention",
        "fixed_head_mean_wrong_prefix_attention",
        "fixed_head_max_wrong_prefix_attention",
        "head_L31H7_prefix_attention",
        "head_L31H7_wrong_prefix_attention",
        "head_L32H7_prefix_attention",
        "head_L32H7_wrong_prefix_attention",
        "head_L34H12_prefix_attention",
        "head_L34H12_wrong_prefix_attention",
        "head_L34H14_prefix_attention",
        "head_L34H14_wrong_prefix_attention",
    ]
    internal_safe = [
        "late_mean_final_cosine_shift",
        "late_max_final_cosine_shift",
        "late_mean_wrong_rank",
        "late_min_wrong_rank",
        *attention_only,
        "fixed_head_mean_wrong_prefix_attention",
        "fixed_head_max_wrong_prefix_attention",
    ]

    return {
        "runtime_output_only_safe": output_only,
        "runtime_internal_safe": internal_safe,
        "runtime_output_plus_internal_safe": list(dict.fromkeys(output_only + internal_safe)),
        "runtime_output_plus_attention_safe": list(dict.fromkeys(output_only + attention_only)),
        "runtime_output_plus_head_safe": list(dict.fromkeys(output_only + head_only)),
    }


def runtime_safe_signal_feature_groups() -> Dict[str, Any]:
    analysis_only_features = [
        "late_mean_correct_rank",
        "late_min_margin_delta",
        "late_mean_margin_delta",
        "late_max_margin_delta",
        "output_baseline_correct_option_logit",
        "output_interference_correct_option_logit",
        "output_interference_margin_delta",
        "baseline_correct_option_logit",
        "interference_correct_option_logit",
        "recheck_correct_option_logit",
        "correct_option_logit",
        "correct_wrong_margin",
        "baseline_correct_rank",
        "interference_correct_rank",
        "recheck_correct_rank",
        "correct_option_attention_mean",
        "baseline_margin",
        "interference_margin",
        "recheck_margin",
    ]
    runtime_safe_features = {
        "output_safe": [
            "safe_baseline_wrong_option_logit",
            "safe_baseline_wrong_rank",
            "safe_interference_top1_margin",
            "safe_interference_wrong_option_logit",
            "safe_interference_wrong_rank",
            "safe_baseline_top1_margin",
            "safe_interference_top1_is_wrong",
            "safe_baseline_top1_is_wrong",
            "safe_wrong_option_logit_delta",
        ],
        "output_exploratory_safe": [
            "safe_baseline_top1_logit",
            "safe_baseline_top2_logit",
            "safe_baseline_option_entropy",
            "safe_interference_top1_logit",
            "safe_interference_top2_logit",
            "safe_interference_option_entropy",
            "safe_wrong_rank_delta",
            "safe_top1_margin_delta",
            "safe_entropy_delta",
        ],
        "internal_shift_safe": [
            "late_mean_final_cosine_shift",
            "late_max_final_cosine_shift",
            "late_mean_pooled_cosine_shift",
            "late_max_pooled_cosine_shift",
            "late_mean_final_norm_delta",
            "late_max_abs_final_norm_delta",
            "late_mean_pooled_norm_delta",
            "late_max_abs_pooled_norm_delta",
            "late_mean_final_std_delta",
            "late_max_abs_final_std_delta",
            "late_mean_pooled_std_delta",
            "late_max_abs_pooled_std_delta",
        ],
        "attention_safe": [
            "late_mean_prefix_attention",
            "late_max_prefix_attention",
            "late_mean_wrong_option_attention",
            "late_max_wrong_option_attention",
            "late_mean_wrong_prefix_attention",
            "late_max_wrong_prefix_attention",
            "late_prefix_concentration",
            "late_wrong_prefix_concentration",
        ],
        "head_safe": [
            "fixed_head_mean_prefix_attention",
            "fixed_head_max_prefix_attention",
            "fixed_head_mean_wrong_prefix_attention",
            "fixed_head_max_wrong_prefix_attention",
            "head_L31H7_prefix_attention",
            "head_L31H7_wrong_prefix_attention",
            "head_L32H7_prefix_attention",
            "head_L32H7_wrong_prefix_attention",
            "head_L34H12_prefix_attention",
            "head_L34H12_wrong_prefix_attention",
            "head_L34H14_prefix_attention",
            "head_L34H14_wrong_prefix_attention",
        ],
        "stability_safe": [
            "safe_top1_stayed_wrong_free",
            "safe_wrong_option_promoted_to_top2",
            "late_wrong_rank_instability",
        ],
    }
    excluded_features = [
        "Any feature derived from ground_truth or correct option identity.",
        "Any margin defined as correct minus wrong or correct minus top competitor.",
        "Any correct-option hidden/logit rank feature.",
        "Any offline patching success proxy used as predictor input.",
        "Near-label runtime proxies such as `probe_answer_changed_interference` and `safe_top1_changed_baseline_to_interference` are excluded from predictor feature sets.",
    ]
    return {
        "analysis_only_features": analysis_only_features,
        "runtime_safe_features": runtime_safe_features,
        "excluded_features": excluded_features,
        "notes": {
            "analysis_only_features": "Useful for mechanistic explanation and oracle-style analysis, but not valid for runtime trigger policies.",
            "runtime_safe_features": "Available at runtime using model outputs, known injected wrong-option cue, and internal states/attention without consulting ground truth.",
            "excluded_features": "Explicitly disallowed from the runtime-safe predictor evaluation.",
        },
    }


def runtime_safe_signal_feature_set_definitions() -> Dict[str, List[str]]:
    groups = runtime_safe_signal_feature_groups()["runtime_safe_features"]
    output_only = list(groups["output_safe"])
    internal_only = list(dict.fromkeys(groups["internal_shift_safe"] + groups["attention_safe"] + groups["head_safe"] + groups["stability_safe"]))
    attention_only = list(dict.fromkeys(groups["attention_safe"] + groups["head_safe"]))
    minimal = [
        "safe_interference_top1_margin",
        "safe_wrong_option_logit_delta",
        "late_mean_final_cosine_shift",
        "late_mean_wrong_prefix_attention",
        "late_lens_mean_wrong_rank",
    ]
    return {
        "runtime_output_only_safe": output_only,
        "runtime_internal_only_safe_v2": internal_only,
        "runtime_output_plus_internal_safe_v2": list(dict.fromkeys(output_only + internal_only)),
        "runtime_attention_only_safe_v2": attention_only,
        "runtime_minimal_safe": minimal,
    }


def _stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    folds = [[] for _ in range(int(n_splits))]
    for idx, value in enumerate(pos_idx):
        folds[idx % n_splits].append(int(value))
    for idx, value in enumerate(neg_idx):
        folds[idx % n_splits].append(int(value))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def _prepare_matrix(df: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray, List[str]]:
    matrix = df.loc[:, list(feature_columns)].astype(float).copy()
    matrix = matrix.fillna(matrix.mean()).fillna(0.0)
    y = df[TARGET_LABEL].astype(int).to_numpy()
    return matrix.to_numpy(dtype=float), y, list(matrix.columns)


def _top1_margin(answer_logits: Dict[str, float]) -> float | None:
    if not answer_logits:
        return None
    sorted_values = sorted((float(v) for v in answer_logits.values()), reverse=True)
    if len(sorted_values) < 2:
        return None
    return float(sorted_values[0] - sorted_values[1])


def _sorted_option_items(answer_logits: Dict[str, float]) -> List[Tuple[str, float]]:
    return [(str(key), float(value)) for key, value in sorted(answer_logits.items(), key=lambda item: item[1], reverse=True)]


def _topk_option_stats(answer_logits: Dict[str, float]) -> Dict[str, float | str | None]:
    ranked = _sorted_option_items(answer_logits)
    if not ranked:
        return {
            "top1_option": None,
            "top1_logit": None,
            "top2_option": None,
            "top2_logit": None,
            "top1_top2_gap": None,
            "option_entropy": None,
        }
    values = np.asarray([value for _key, value in ranked], dtype=float)
    shifted = values - np.max(values)
    probs = np.exp(shifted)
    probs = probs / probs.sum() if probs.sum() else np.zeros_like(probs)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    return {
        "top1_option": ranked[0][0],
        "top1_logit": ranked[0][1],
        "top2_option": ranked[1][0] if len(ranked) > 1 else None,
        "top2_logit": ranked[1][1] if len(ranked) > 1 else None,
        "top1_top2_gap": float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else None,
        "option_entropy": entropy,
    }


def _safe_value_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None or pd.isna(after) or pd.isna(before):
        return None
    return float(float(after) - float(before))


def _hidden_summary_stat(record: Dict[str, Any], layer_index: int, kind: str, field: str) -> float | None:
    key = f"layer_{int(layer_index)}_{kind}"
    payload = (record.get("hidden_state_summary") or {}).get(key, {})
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    return None if value is None or pd.isna(value) else float(value)


def _attention_summary_map(record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = record.get("attention_summary") or []
    return {
        int(row.get("layer_index")): dict(row)
        for row in rows
        if row.get("layer_index") is not None
    }


def _logit_lens_map(record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = record.get("layer_logit_lens") or []
    return {
        int(row.get("layer_index")): dict(row)
        for row in rows
        if row.get("layer_index") is not None
    }


def _rank_of_option(answer_logits: Dict[str, float], option: str) -> float | None:
    option = str(option or "").strip().upper()
    if option not in answer_logits:
        return None
    ranked = [key for key, _value in sorted(answer_logits.items(), key=lambda item: item[1], reverse=True)]
    return float(ranked.index(option) + 1)


def build_runtime_safe_dataset(
    *,
    probe_comparisons: Sequence[Dict[str, Any]],
    sample_cases: Sequence[Dict[str, Any]],
    scenario_records: Sequence[Dict[str, Any]],
    focus_layers: Sequence[int] = (31, 32, 33, 34, 35),
    baseline_correct_only: bool = True,
    fixed_heads: Sequence[Tuple[int, int]] = ((31, 7), (32, 7), (34, 12), (34, 14)),
) -> pd.DataFrame:
    comparison_map = {str(row["sample_id"]): dict(row) for row in probe_comparisons}
    case_map = {str(row["sample_id"]): dict(row) for row in sample_cases}
    scenario_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row["sample_id"]), str(row["scenario"])): dict(row)
        for row in scenario_records
    }

    rows: List[Dict[str, Any]] = []
    for sample_id, case in case_map.items():
        if baseline_correct_only and not bool(case.get("baseline_correct")):
            continue
        comparison = comparison_map.get(sample_id, {})
        baseline_record = scenario_map.get((sample_id, "baseline"), {})
        interference_record = scenario_map.get((sample_id, "interference"), {})
        if not baseline_record or not interference_record:
            continue

        focus = [row for row in case.get("per_layer", []) if int(row.get("layer_index", -999)) in {int(layer) for layer in focus_layers}]
        final_cosines = [row.get("baseline_to_interference_final_cosine") for row in focus]
        wrong_ranks = [row.get("interference_wrong_rank") for row in focus]
        prefix_attention = [row.get("interference_prefix_attention_mean") for row in focus]
        wrong_prefix_attention = [row.get("interference_wrong_option_prefix_attention_mean") for row in focus]

        baseline_logits = dict(baseline_record.get("answer_logits", {}) or {})
        interference_logits = dict(interference_record.get("answer_logits", {}) or {})
        wrong_option = str(case.get("wrong_option") or "").strip().upper()

        row = {
            "sample_id": sample_id,
            "model_name": case.get("model_name"),
            "sample_type": case.get("sample_type"),
            "condition_id": case.get("condition_id"),
            "authority_level": float(case.get("authority_level") or 0.0),
            "confidence_level": float(case.get("confidence_level") or 0.0),
            "explicit_wrong_option": float(case.get("explicit_wrong_option") or 0.0),
            "is_control": float(case.get("is_control") or 0.0),
            "is_hard_negative": float(case.get("is_hard_negative") or 0.0),
            TARGET_LABEL: float(case.get("transition_label") == "baseline_correct_to_interference_wrong"),
            "transition_label": case.get("transition_label"),
            "late_mean_final_cosine_shift": (float(1.0 - _safe_mean(final_cosines)) if _safe_mean(final_cosines) is not None else None),
            "late_max_final_cosine_shift": (float(1.0 - _safe_min(final_cosines)) if _safe_min(final_cosines) is not None else None),
            "late_mean_wrong_rank": _safe_mean(wrong_ranks),
            "late_min_wrong_rank": _safe_min(wrong_ranks),
            "late_mean_prefix_attention": _safe_mean(prefix_attention),
            "late_mean_wrong_prefix_attention": _safe_mean(wrong_prefix_attention),
            "late_max_wrong_prefix_attention": _safe_max(wrong_prefix_attention),
            "fixed_head_mean_prefix_attention": None,
            "fixed_head_max_prefix_attention": None,
            "fixed_head_mean_wrong_prefix_attention": None,
            "fixed_head_max_wrong_prefix_attention": None,
            "safe_baseline_wrong_option_logit": baseline_logits.get(wrong_option),
            "safe_interference_wrong_option_logit": interference_logits.get(wrong_option),
            "safe_wrong_option_logit_delta": (
                float(interference_logits.get(wrong_option, 0.0) - baseline_logits.get(wrong_option, 0.0))
                if wrong_option in baseline_logits and wrong_option in interference_logits
                else None
            ),
            "safe_baseline_wrong_rank": _rank_of_option(baseline_logits, wrong_option),
            "safe_interference_wrong_rank": _rank_of_option(interference_logits, wrong_option),
            "safe_baseline_top1_margin": _top1_margin(baseline_logits),
            "safe_interference_top1_margin": _top1_margin(interference_logits),
            "safe_baseline_top1_is_wrong": float(baseline_record.get("predicted_answer") == wrong_option),
            "safe_interference_top1_is_wrong": float(interference_record.get("predicted_answer") == wrong_option),
            "probe_answer_changed_interference": float(bool(comparison.get("predicted_answer_changed_interference"))),
        }

        positions = _safe_attention_positions(interference_record, model_name=str(case.get("model_name") or baseline_record.get("model_name") or "Qwen/Qwen2.5-3B-Instruct"))
        attention_path = interference_record.get("attention_array_path")
        attention_arrays = np.load(attention_path) if attention_path and Path(attention_path).exists() else None
        head_prefix_values: List[float] = []
        head_wrong_prefix_values: List[float] = []
        for layer_index, head_index in fixed_heads:
            prefix_value = None
            wrong_prefix_value = None
            key = f"layer_{int(layer_index)}_final_token_attention"
            if attention_arrays is not None and key in attention_arrays:
                array = attention_arrays[key]
                prefix_value = _summarize_head_attention(array, positions["prefix_positions"], int(head_index))
                wrong_prefix_value = _summarize_head_attention(array, positions["wrong_prefix_positions"], int(head_index))
            if prefix_value is not None:
                head_prefix_values.append(float(prefix_value))
            if wrong_prefix_value is not None:
                head_wrong_prefix_values.append(float(wrong_prefix_value))
            row[f"head_L{int(layer_index)}H{int(head_index)}_prefix_attention"] = prefix_value
            row[f"head_L{int(layer_index)}H{int(head_index)}_wrong_prefix_attention"] = wrong_prefix_value
        row["fixed_head_mean_prefix_attention"] = _safe_mean(head_prefix_values)
        row["fixed_head_max_prefix_attention"] = _safe_max(head_prefix_values)
        row["fixed_head_mean_wrong_prefix_attention"] = _safe_mean(head_wrong_prefix_values)
        row["fixed_head_max_wrong_prefix_attention"] = _safe_max(head_wrong_prefix_values)
        if attention_arrays is not None:
            attention_arrays.close()

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if col not in {"sample_id", "model_name", "sample_type", "condition_id", "transition_label"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_runtime_safe_signal_dataset(
    *,
    probe_comparisons: Sequence[Dict[str, Any]],
    sample_cases: Sequence[Dict[str, Any]],
    scenario_records: Sequence[Dict[str, Any]],
    focus_layers: Sequence[int] = (31, 32, 33, 34, 35),
    baseline_correct_only: bool = True,
    fixed_heads: Sequence[Tuple[int, int]] = ((31, 7), (32, 7), (34, 12), (34, 14)),
) -> pd.DataFrame:
    base_df = build_runtime_safe_dataset(
        probe_comparisons=probe_comparisons,
        sample_cases=sample_cases,
        scenario_records=scenario_records,
        focus_layers=focus_layers,
        baseline_correct_only=baseline_correct_only,
        fixed_heads=fixed_heads,
    )
    if base_df.empty:
        return base_df

    case_map = {str(row["sample_id"]): dict(row) for row in sample_cases}
    scenario_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row["sample_id"]), str(row["scenario"])): dict(row)
        for row in scenario_records
    }

    extra_rows: List[Dict[str, Any]] = []
    focus_layer_set = {int(layer) for layer in focus_layers}
    for sample_id in base_df["sample_id"].astype(str).tolist():
        case = case_map[sample_id]
        baseline_record = scenario_map[(sample_id, "baseline")]
        interference_record = scenario_map[(sample_id, "interference")]
        baseline_logits = dict(baseline_record.get("answer_logits", {}) or {})
        interference_logits = dict(interference_record.get("answer_logits", {}) or {})
        baseline_top = _topk_option_stats(baseline_logits)
        interference_top = _topk_option_stats(interference_logits)
        wrong_option = str(case.get("wrong_option") or "").strip().upper()

        focus_rows = [row for row in (case.get("per_layer") or []) if int(row.get("layer_index", -999)) in focus_layer_set]
        pooled_cosines = [row.get("baseline_to_interference_pooled_cosine") for row in focus_rows]
        wrong_option_attention = []
        prefix_attention_max = []
        wrong_option_attention_max = []
        wrong_prefix_concentration = []
        prefix_concentration = []
        wrong_rank_instability = []
        for row in focus_rows:
            wrong_option_attention.append(row.get("interference_wrong_logit"))
            baseline_wrong_rank = row.get("baseline_wrong_rank")
            interference_wrong_rank = row.get("interference_wrong_rank")
            if baseline_wrong_rank is not None and interference_wrong_rank is not None:
                wrong_rank_instability.append(float(baseline_wrong_rank) - float(interference_wrong_rank))

        baseline_attn = _attention_summary_map(baseline_record)
        interference_attn = _attention_summary_map(interference_record)
        for layer_index in focus_layer_set:
            attn = interference_attn.get(layer_index, {})
            if attn:
                prefix_mean = attn.get("prefix_attention_mean")
                prefix_max = attn.get("prefix_attention_max_head_mean")
                wrong_mean = attn.get("wrong_option_attention_mean")
                wrong_prefix_mean = attn.get("wrong_option_prefix_attention_mean")
                if prefix_max is not None:
                    prefix_attention_max.append(prefix_max)
                if wrong_mean is not None:
                    wrong_option_attention_max.append(wrong_mean)
                if prefix_mean not in (None, 0) and prefix_max is not None:
                    prefix_concentration.append(float(prefix_max) / max(float(prefix_mean), 1e-6))
                if wrong_mean not in (None, 0) and wrong_prefix_mean is not None:
                    wrong_prefix_concentration.append(float(wrong_prefix_mean) / max(float(wrong_mean), 1e-6))

        final_norm_deltas = []
        pooled_norm_deltas = []
        final_std_deltas = []
        pooled_std_deltas = []
        for layer_index in focus_layer_set:
            final_norm_deltas.append(_safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "final_token", "norm"),
                _hidden_summary_stat(baseline_record, layer_index, "final_token", "norm"),
            ))
            pooled_norm_deltas.append(_safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "pooled_mean", "norm"),
                _hidden_summary_stat(baseline_record, layer_index, "pooled_mean", "norm"),
            ))
            final_std_deltas.append(_safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "final_token", "std"),
                _hidden_summary_stat(baseline_record, layer_index, "final_token", "std"),
            ))
            pooled_std_deltas.append(_safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "pooled_mean", "std"),
                _hidden_summary_stat(baseline_record, layer_index, "pooled_mean", "std"),
            ))

        lens_map = _logit_lens_map(interference_record)
        lens_wrong_ranks = []
        for layer_index in focus_layer_set:
            lens = lens_map.get(layer_index, {})
            ranked = list(lens.get("ranked_options") or [])
            if wrong_option and wrong_option in ranked:
                lens_wrong_ranks.append(float(ranked.index(wrong_option) + 1))

        extra_rows.append(
            {
                "sample_id": sample_id,
                "safe_baseline_top1_logit": baseline_top["top1_logit"],
                "safe_baseline_top2_logit": baseline_top["top2_logit"],
                "safe_baseline_option_entropy": baseline_top["option_entropy"],
                "safe_interference_top1_logit": interference_top["top1_logit"],
                "safe_interference_top2_logit": interference_top["top2_logit"],
                "safe_interference_option_entropy": interference_top["option_entropy"],
                "safe_top1_margin_delta": _safe_value_delta(interference_top["top1_top2_gap"], baseline_top["top1_top2_gap"]),
                "safe_entropy_delta": _safe_value_delta(interference_top["option_entropy"], baseline_top["option_entropy"]),
                "safe_top1_changed_baseline_to_interference": float(baseline_top["top1_option"] != interference_top["top1_option"]),
                "safe_top1_stayed_wrong_free": float(
                    baseline_top["top1_option"] not in (None, wrong_option)
                    and interference_top["top1_option"] not in (None, wrong_option)
                ),
                "safe_wrong_rank_delta": _safe_value_delta(
                    base_df.loc[base_df["sample_id"] == sample_id, "safe_interference_wrong_rank"].iloc[0],
                    base_df.loc[base_df["sample_id"] == sample_id, "safe_baseline_wrong_rank"].iloc[0],
                ),
                "safe_wrong_option_promoted_to_top2": float(
                    (base_df.loc[base_df["sample_id"] == sample_id, "safe_baseline_wrong_rank"].iloc[0] or 99) > 2
                    and (base_df.loc[base_df["sample_id"] == sample_id, "safe_interference_wrong_rank"].iloc[0] or 99) <= 2
                ),
                "late_mean_pooled_cosine_shift": (float(1.0 - _safe_mean(pooled_cosines)) if _safe_mean(pooled_cosines) is not None else None),
                "late_max_pooled_cosine_shift": (float(1.0 - _safe_min(pooled_cosines)) if _safe_min(pooled_cosines) is not None else None),
                "late_mean_final_norm_delta": _safe_mean(final_norm_deltas),
                "late_max_abs_final_norm_delta": _safe_max([abs(v) if v is not None else None for v in final_norm_deltas]),
                "late_mean_pooled_norm_delta": _safe_mean(pooled_norm_deltas),
                "late_max_abs_pooled_norm_delta": _safe_max([abs(v) if v is not None else None for v in pooled_norm_deltas]),
                "late_mean_final_std_delta": _safe_mean(final_std_deltas),
                "late_max_abs_final_std_delta": _safe_max([abs(v) if v is not None else None for v in final_std_deltas]),
                "late_mean_pooled_std_delta": _safe_mean(pooled_std_deltas),
                "late_max_abs_pooled_std_delta": _safe_max([abs(v) if v is not None else None for v in pooled_std_deltas]),
                "late_mean_wrong_option_attention": _safe_mean(wrong_option_attention_max),
                "late_max_wrong_option_attention": _safe_max(wrong_option_attention_max),
                "late_max_prefix_attention": _safe_max(prefix_attention_max),
                "late_prefix_concentration": _safe_mean(prefix_concentration),
                "late_wrong_prefix_concentration": _safe_mean(wrong_prefix_concentration),
                "late_wrong_rank_instability": _safe_mean(wrong_rank_instability),
                "late_lens_mean_wrong_rank": _safe_mean(lens_wrong_ranks),
            }
        )

    extra_df = pd.DataFrame(extra_rows)
    merged = base_df.merge(extra_df, on="sample_id", how="left")
    numeric_cols = [col for col in merged.columns if col not in {"sample_id", "model_name", "sample_type", "condition_id", "transition_label", "eval_split"}]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def build_sparse_runtime_monitor_dataset(
    *,
    probe_comparisons: Sequence[Dict[str, Any]],
    sample_cases: Sequence[Dict[str, Any]],
    scenario_records: Sequence[Dict[str, Any]],
    focus_layers: Sequence[int] = (31, 32, 33, 34, 35),
    baseline_correct_only: bool = True,
) -> pd.DataFrame:
    base_df = build_runtime_safe_signal_dataset(
        probe_comparisons=probe_comparisons,
        sample_cases=sample_cases,
        scenario_records=scenario_records,
        focus_layers=focus_layers,
        baseline_correct_only=baseline_correct_only,
    )
    if base_df.empty:
        return base_df

    case_map = {str(row["sample_id"]): dict(row) for row in sample_cases}
    scenario_map: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row["sample_id"]), str(row["scenario"])): dict(row)
        for row in scenario_records
    }
    extra_rows: List[Dict[str, Any]] = []
    focus_layer_set = {int(layer) for layer in focus_layers}

    for sample_id in base_df["sample_id"].astype(str).tolist():
        case = case_map[sample_id]
        baseline_record = scenario_map[(sample_id, "baseline")]
        interference_record = scenario_map[(sample_id, "interference")]
        attention_path = interference_record.get("attention_array_path")
        attention_arrays = np.load(attention_path) if attention_path and Path(attention_path).exists() else None
        positions = _safe_attention_positions(
            interference_record,
            model_name=str(case.get("model_name") or baseline_record.get("model_name") or "Qwen/Qwen2.5-3B-Instruct"),
        )

        row: Dict[str, Any] = {"sample_id": sample_id}
        for layer_index in sorted(focus_layer_set):
            per_layer_row = next(
                (item for item in (case.get("per_layer") or []) if int(item.get("layer_index", -999)) == int(layer_index)),
                {},
            )
            row[f"layer_{layer_index}_final_cosine_shift"] = (
                float(1.0 - float(per_layer_row["baseline_to_interference_final_cosine"]))
                if per_layer_row.get("baseline_to_interference_final_cosine") is not None
                else None
            )
            row[f"layer_{layer_index}_pooled_cosine_shift"] = (
                float(1.0 - float(per_layer_row["baseline_to_interference_pooled_cosine"]))
                if per_layer_row.get("baseline_to_interference_pooled_cosine") is not None
                else None
            )
            row[f"layer_{layer_index}_wrong_prefix_attention"] = per_layer_row.get("interference_wrong_option_prefix_attention_mean")
            row[f"layer_{layer_index}_prefix_attention"] = per_layer_row.get("interference_prefix_attention_mean")
            row[f"layer_{layer_index}_lens_wrong_rank"] = per_layer_row.get("interference_wrong_rank")
            row[f"layer_{layer_index}_wrong_rank_delta"] = _safe_value_delta(
                per_layer_row.get("interference_wrong_rank"),
                per_layer_row.get("baseline_wrong_rank"),
            )
            row[f"layer_{layer_index}_final_norm_delta"] = _safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "final_token", "norm"),
                _hidden_summary_stat(baseline_record, layer_index, "final_token", "norm"),
            )
            row[f"layer_{layer_index}_final_std_delta"] = _safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "final_token", "std"),
                _hidden_summary_stat(baseline_record, layer_index, "final_token", "std"),
            )
            row[f"layer_{layer_index}_pooled_norm_delta"] = _safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "pooled_mean", "norm"),
                _hidden_summary_stat(baseline_record, layer_index, "pooled_mean", "norm"),
            )
            row[f"layer_{layer_index}_pooled_std_delta"] = _safe_value_delta(
                _hidden_summary_stat(interference_record, layer_index, "pooled_mean", "std"),
                _hidden_summary_stat(baseline_record, layer_index, "pooled_mean", "std"),
            )

            key = f"layer_{layer_index}_final_token_attention"
            if attention_arrays is not None and key in attention_arrays:
                array = attention_arrays[key]
                for head_index in range(int(array.shape[0])):
                    row[f"head_L{layer_index}H{head_index}_prefix_attention"] = _summarize_head_attention(
                        array, positions["prefix_positions"], head_index
                    )
                    row[f"head_L{layer_index}H{head_index}_wrong_prefix_attention"] = _summarize_head_attention(
                        array, positions["wrong_prefix_positions"], head_index
                    )
        if attention_arrays is not None:
            attention_arrays.close()
        extra_rows.append(row)

    extra_df = pd.DataFrame(extra_rows)
    merged = base_df.merge(extra_df, on="sample_id", how="left")
    numeric_cols = [col for col in merged.columns if col not in {"sample_id", "model_name", "sample_type", "condition_id", "transition_label", "eval_split"}]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_predict_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    train_scaled = (X_train - mean) / std
    test_scaled = (X_test - mean) / std
    train_design = sm.add_constant(train_scaled, has_constant="add")
    test_design = sm.add_constant(test_scaled, has_constant="add")
    try:
        result = sm.GLM(y_train, train_design, family=sm.families.Binomial()).fit()
        probs = np.asarray(result.predict(test_design), dtype=float)
    except Exception:
        result = sm.GLM(y_train, train_design, family=sm.families.Binomial()).fit_regularized(alpha=1e-4, L1_wt=0.0)
        linear = np.asarray(test_design @ result.params, dtype=float)
        probs = _sigmoid(linear)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def fit_regularized_predictor_model(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    alpha: float,
    l1_wt: float,
) -> PredictorModel:
    X, y, used_columns = _prepare_matrix(df, feature_columns)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (X - mean) / std
    design = sm.add_constant(scaled, has_constant="add")
    result = sm.GLM(y, design, family=sm.families.Binomial()).fit_regularized(alpha=float(alpha), L1_wt=float(l1_wt))
    params = np.asarray(result.params, dtype=float)
    return PredictorModel(
        feature_columns=tuple(used_columns),
        mean=np.asarray(mean, dtype=float),
        std=np.asarray(std, dtype=float),
        params=np.asarray(params, dtype=float),
    )


def fit_predictor_model(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> PredictorModel:
    X, y, used_columns = _prepare_matrix(df, feature_columns)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = (X - mean) / std
    design = sm.add_constant(scaled, has_constant="add")
    try:
        result = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        params = np.asarray(result.params, dtype=float)
    except Exception:
        result = sm.GLM(y, design, family=sm.families.Binomial()).fit_regularized(alpha=1e-4, L1_wt=0.0)
        params = np.asarray(result.params, dtype=float)
    return PredictorModel(
        feature_columns=tuple(used_columns),
        mean=np.asarray(mean, dtype=float),
        std=np.asarray(std, dtype=float),
        params=np.asarray(params, dtype=float),
    )


def predict_with_model(model: PredictorModel, df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.loc[:, list(model.feature_columns)].astype(float).copy()
    matrix = matrix.fillna(matrix.mean()).fillna(0.0)
    X = matrix.to_numpy(dtype=float)
    scaled = (X - model.mean) / model.std
    design = sm.add_constant(scaled, has_constant="add")
    linear = np.asarray(design @ model.params, dtype=float)
    probs = np.clip(_sigmoid(linear), 1e-6, 1.0 - 1e-6)
    out = df.loc[:, ["sample_id", TARGET_LABEL, "sample_type", "condition_id", "transition_label"]].copy()
    out["predicted_risk_score"] = probs
    out["predicted_label"] = (probs >= 0.5).astype(int)
    return out


def count_nonzero_coefficients(model: PredictorModel, *, tol: float = 1e-8) -> int:
    if model.params.size <= 1:
        return 0
    return int(np.sum(np.abs(model.params[1:]) > float(tol)))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float | None]:
    y_pred = (y_score >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_manual(y_true, y_score),
        "average_precision": average_precision_manual(y_true, y_score),
    }


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    wins = 0.0
    total = float(len(pos_scores) * len(neg_scores))
    for pos in pos_scores:
        wins += float(np.sum(pos > neg_scores))
        wins += 0.5 * float(np.sum(pos == neg_scores))
    return float(wins / total)


def average_precision_manual(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = 0
    fp = 0
    precision_sum = 0.0
    for label in y_sorted:
        if int(label) == 1:
            tp += 1
            precision_sum += float(tp / (tp + fp))
        else:
            fp += 1
    return float(precision_sum / positives)


def cross_validated_predictor_baseline(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[PredictorResult, pd.DataFrame]:
    X, y, used_columns = _prepare_matrix(df, feature_columns)
    folds = _stratified_folds(y, n_splits=min(int(n_splits), int(np.sum(y == 1)), int(np.sum(y == 0))), seed=seed)
    preds = np.zeros(len(df), dtype=float)
    for test_idx in folds:
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        preds[test_idx] = _fit_predict_glm(X[train_mask], y[train_mask], X[test_idx])
    metrics = _classification_metrics(y, preds)
    result = PredictorResult(
        feature_set_name="",
        feature_columns=tuple(used_columns),
        num_samples=int(len(df)),
        num_positive=int(np.sum(y)),
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        average_precision=metrics["average_precision"],
    )
    prediction_df = df.loc[:, ["sample_id", TARGET_LABEL, "sample_type", "condition_id", "transition_label"]].copy()
    prediction_df["predicted_risk_score"] = preds
    prediction_df["predicted_label"] = (preds >= 0.5).astype(int)
    return result, prediction_df


def cross_validated_regularized_predictor(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    alpha: float,
    l1_wt: float,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[PredictorResult, pd.DataFrame]:
    X, y, used_columns = _prepare_matrix(df, feature_columns)
    folds = _stratified_folds(y, n_splits=min(int(n_splits), int(np.sum(y == 1)), int(np.sum(y == 0))), seed=seed)
    preds = np.zeros(len(df), dtype=float)
    for test_idx in folds:
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_df = df.iloc[np.where(train_mask)[0]]
        test_df = df.iloc[test_idx]
        model = fit_regularized_predictor_model(train_df, feature_columns=feature_columns, alpha=float(alpha), l1_wt=float(l1_wt))
        fold_predictions = predict_with_model(model, test_df)
        preds[test_idx] = fold_predictions["predicted_risk_score"].astype(float).to_numpy()
    metrics = _classification_metrics(y, preds)
    result = PredictorResult(
        feature_set_name="",
        feature_columns=tuple(used_columns),
        num_samples=int(len(df)),
        num_positive=int(np.sum(y)),
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        average_precision=metrics["average_precision"],
    )
    prediction_df = df.loc[:, ["sample_id", TARGET_LABEL, "sample_type", "condition_id", "transition_label"]].copy()
    prediction_df["predicted_risk_score"] = preds
    prediction_df["predicted_label"] = (preds >= 0.5).astype(int)
    return result, prediction_df


def select_best_regularized_model(
    df: pd.DataFrame,
    *,
    feature_set_name: str,
    feature_columns: Sequence[str],
    candidate_configs: Sequence[Tuple[str, float, float]],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[RegularizedSearchResult, PredictorModel, pd.DataFrame]:
    best_search: RegularizedSearchResult | None = None
    best_model: PredictorModel | None = None
    best_predictions: pd.DataFrame | None = None
    for model_type, alpha, l1_wt in candidate_configs:
        cv_result, cv_predictions = cross_validated_regularized_predictor(
            df,
            feature_columns=feature_columns,
            alpha=float(alpha),
            l1_wt=float(l1_wt),
            n_splits=n_splits,
            seed=seed,
        )
        model = fit_regularized_predictor_model(df, feature_columns=feature_columns, alpha=float(alpha), l1_wt=float(l1_wt))
        search = RegularizedSearchResult(
            feature_set_name=feature_set_name,
            model_type=str(model_type),
            alpha=float(alpha),
            l1_wt=float(l1_wt),
            num_features=len(feature_columns),
            num_nonzero_features=count_nonzero_coefficients(model),
            cv_result=cv_result.__class__(
                feature_set_name=feature_set_name,
                **{k: getattr(cv_result, k) for k in cv_result.__dataclass_fields__ if k != "feature_set_name"},
            ),
        )
        rank_key = (
            -999.0 if search.cv_result.roc_auc is None else float(search.cv_result.roc_auc),
            -999.0 if search.cv_result.average_precision is None else float(search.cv_result.average_precision),
            float(search.cv_result.f1),
            -float(search.num_nonzero_features),
        )
        if best_search is None:
            best_search, best_model, best_predictions = search, model, cv_predictions
            best_key = rank_key
            continue
        if rank_key > best_key:
            best_search, best_model, best_predictions = search, model, cv_predictions
            best_key = rank_key
    assert best_search is not None and best_model is not None and best_predictions is not None
    return best_search, best_model, best_predictions


def budget_utility_rows(
    *,
    predictions_df: pd.DataFrame,
    feature_set_name: str,
    eval_split: str,
    trigger_budgets: Sequence[float],
) -> List[Dict[str, Any]]:
    df = predictions_df.copy().sort_values("predicted_risk_score", ascending=False).reset_index(drop=True)
    num_samples = int(len(df))
    num_positive = int(df[TARGET_LABEL].sum())
    base_rate = float(num_positive / num_samples) if num_samples else 0.0
    rows: List[Dict[str, Any]] = []
    for budget in trigger_budgets:
        num_triggered = int(max(1, np.ceil(float(num_samples) * float(budget)))) if num_samples else 0
        top = df.head(num_triggered).copy()
        positives_captured = int(top[TARGET_LABEL].sum()) if num_triggered else 0
        precision_at_budget = float(positives_captured / num_triggered) if num_triggered else 0.0
        recall_at_budget = float(positives_captured / num_positive) if num_positive else 0.0
        capture_rate = recall_at_budget
        lift_over_random = float(precision_at_budget / base_rate) if base_rate else None
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "eval_split": eval_split,
                "trigger_budget": float(budget),
                "num_triggered": num_triggered,
                "positives_captured": positives_captured,
                "capture_rate": capture_rate,
                "precision_at_budget": precision_at_budget,
                "recall_at_budget": recall_at_budget,
                "lift_over_random": lift_over_random,
            }
        )
    return rows


def budget_utility_rows_extended(
    *,
    predictions_df: pd.DataFrame,
    feature_set_name: str,
    eval_split: str,
    split_type: str,
    trigger_budgets: Sequence[float] | None = None,
    topk_budgets: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    df = predictions_df.copy().sort_values("predicted_risk_score", ascending=False).reset_index(drop=True)
    num_samples = int(len(df))
    num_positive = int(df[TARGET_LABEL].sum())
    base_rate = float(num_positive / num_samples) if num_samples else 0.0
    rows: List[Dict[str, Any]] = []

    for budget in trigger_budgets or ():
        num_triggered = int(max(1, np.ceil(float(num_samples) * float(budget)))) if num_samples else 0
        top = df.head(num_triggered).copy()
        positives_captured = int(top[TARGET_LABEL].sum()) if num_triggered else 0
        precision_at_budget = float(positives_captured / num_triggered) if num_triggered else 0.0
        recall_at_budget = float(positives_captured / num_positive) if num_positive else 0.0
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "eval_split": eval_split,
                "split_type": split_type,
                "trigger_budget": f"{int(round(float(budget) * 100.0))}%",
                "num_triggered": num_triggered,
                "positives_captured": positives_captured,
                "capture_rate": recall_at_budget,
                "precision_at_budget": precision_at_budget,
                "recall_at_budget": recall_at_budget,
                "lift_over_random": (float(precision_at_budget / base_rate) if base_rate else None),
            }
        )

    for topk in topk_budgets or ():
        num_triggered = int(min(max(int(topk), 0), num_samples))
        top = df.head(num_triggered).copy()
        positives_captured = int(top[TARGET_LABEL].sum()) if num_triggered else 0
        precision_at_budget = float(positives_captured / num_triggered) if num_triggered else 0.0
        recall_at_budget = float(positives_captured / num_positive) if num_positive else 0.0
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "eval_split": eval_split,
                "split_type": split_type,
                "trigger_budget": f"top-{int(topk)}",
                "num_triggered": num_triggered,
                "positives_captured": positives_captured,
                "capture_rate": recall_at_budget,
                "precision_at_budget": precision_at_budget,
                "recall_at_budget": recall_at_budget,
                "lift_over_random": (float(precision_at_budget / base_rate) if base_rate else None),
            }
        )
    return rows


def load_probe_sample_metadata(sample_files: Sequence[str | Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample_file in sample_files:
        payload = json.loads(Path(sample_file).read_text(encoding="utf-8"))
        for sample in payload.get("samples", []):
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "subject": sample.get("subject"),
                    "category": sample.get("category"),
                    "task_id": sample.get("task_id"),
                    "condition_id_meta": sample.get("condition_id"),
                    "sample_type_meta": sample.get("sample_type"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["sample_id", "subject", "category", "task_id", "condition_id_meta", "sample_type_meta"])
    return pd.DataFrame(rows).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)


def fit_full_model_coefficients(df: pd.DataFrame, *, feature_columns: Sequence[str]) -> pd.DataFrame:
    model = fit_predictor_model(df, feature_columns=feature_columns)
    params = model.params
    rows = [{"feature_name": "const", "coefficient": float(params[0])}]
    for name, value in zip(model.feature_columns, params[1:]):
        rows.append({"feature_name": str(name), "coefficient": float(value)})
    return pd.DataFrame(rows)


def evaluate_predictions(
    *,
    feature_set_name: str,
    predictions_df: pd.DataFrame,
) -> PredictorResult:
    y_true = predictions_df[TARGET_LABEL].astype(int).to_numpy()
    y_score = predictions_df["predicted_risk_score"].astype(float).to_numpy()
    metrics = _classification_metrics(y_true, y_score)
    feature_columns: Sequence[str] = []
    if "feature_columns" in predictions_df.attrs:
        feature_columns = list(predictions_df.attrs["feature_columns"])
    return PredictorResult(
        feature_set_name=feature_set_name,
        feature_columns=tuple(feature_columns),
        num_samples=int(len(predictions_df)),
        num_positive=int(np.sum(y_true)),
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        average_precision=metrics["average_precision"],
    )
