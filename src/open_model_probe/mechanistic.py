"""Mechanistic analysis helpers for local open-model probing."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join([header, separator, *body])


def transition_label(
    baseline_correct: bool | None,
    interference_correct: bool | None,
) -> str:
    if baseline_correct is True and interference_correct is False:
        return "baseline_correct_to_interference_wrong"
    if baseline_correct is False and interference_correct is False:
        return "baseline_wrong_to_interference_wrong"
    if baseline_correct is True and interference_correct is True:
        return "baseline_correct_to_interference_correct"
    if baseline_correct is False and interference_correct is True:
        return "baseline_wrong_to_interference_correct"
    return "unknown"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _option_rank(answer_logits: Dict[str, float], option: str) -> int | None:
    ranked = [item[0] for item in sorted(answer_logits.items(), key=lambda item: item[1], reverse=True)]
    try:
        return int(ranked.index(str(option or "").strip().upper()) + 1)
    except ValueError:
        return None


def _layer_rows_map(record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(row["layer_index"]): dict(row) for row in record.get("layer_logit_lens", [])}


def _attention_rows_map(record: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(row["layer_index"]): dict(row) for row in record.get("attention_summary", [])}


def build_sample_case_analysis(sample_bundle: Dict[str, Any]) -> Dict[str, Any]:
    baseline = sample_bundle.get("baseline")
    interference = sample_bundle.get("interference")
    recheck = sample_bundle.get("recheck")
    if baseline is None or interference is None:
        raise ValueError("Sample bundle must contain baseline and interference records.")

    baseline_layer_map = _layer_rows_map(baseline)
    interference_layer_map = _layer_rows_map(interference)
    recheck_layer_map = _layer_rows_map(recheck or {})
    baseline_attention_map = _attention_rows_map(baseline)
    interference_attention_map = _attention_rows_map(interference)
    recheck_attention_map = _attention_rows_map(recheck or {})

    baseline_hidden = baseline.get("_hidden_state_arrays", {}) or {}
    interference_hidden = interference.get("_hidden_state_arrays", {}) or {}
    recheck_hidden = (recheck or {}).get("_hidden_state_arrays", {}) or {}

    layer_indices = sorted(set(baseline_layer_map) | set(interference_layer_map) | set(recheck_layer_map))
    per_layer: List[Dict[str, Any]] = []
    correct_option = str(baseline.get("ground_truth") or "").strip().upper()
    wrong_option = str(baseline.get("wrong_option") or "").strip().upper()

    for layer_index in layer_indices:
        base_row = baseline_layer_map.get(layer_index, {})
        int_row = interference_layer_map.get(layer_index, {})
        re_row = recheck_layer_map.get(layer_index, {})

        base_logits = dict(base_row.get("answer_logits", {}))
        int_logits = dict(int_row.get("answer_logits", {}))
        re_logits = dict(re_row.get("answer_logits", {}))

        base_final = baseline_hidden.get(f"layer_{layer_index}_final_token")
        int_final = interference_hidden.get(f"layer_{layer_index}_final_token")
        re_final = recheck_hidden.get(f"layer_{layer_index}_final_token")
        base_pooled = baseline_hidden.get(f"layer_{layer_index}_pooled_mean")
        int_pooled = interference_hidden.get(f"layer_{layer_index}_pooled_mean")
        re_pooled = recheck_hidden.get(f"layer_{layer_index}_pooled_mean")

        base_attn = baseline_attention_map.get(layer_index, {})
        int_attn = interference_attention_map.get(layer_index, {})
        re_attn = recheck_attention_map.get(layer_index, {})

        per_layer.append(
            {
                "layer_index": int(layer_index),
                "baseline_predicted_answer": base_row.get("predicted_answer"),
                "interference_predicted_answer": int_row.get("predicted_answer"),
                "recheck_predicted_answer": re_row.get("predicted_answer"),
                "baseline_correct_logit": float(base_logits.get(correct_option, np.nan)),
                "interference_correct_logit": float(int_logits.get(correct_option, np.nan)),
                "recheck_correct_logit": float(re_logits.get(correct_option, np.nan)),
                "baseline_wrong_logit": float(base_logits.get(wrong_option, np.nan)),
                "interference_wrong_logit": float(int_logits.get(wrong_option, np.nan)),
                "recheck_wrong_logit": float(re_logits.get(wrong_option, np.nan)),
                "baseline_margin": float(base_logits.get(correct_option, 0.0) - base_logits.get(wrong_option, 0.0)),
                "interference_margin": float(int_logits.get(correct_option, 0.0) - int_logits.get(wrong_option, 0.0)),
                "recheck_margin": float(re_logits.get(correct_option, 0.0) - re_logits.get(wrong_option, 0.0)),
                "interference_margin_delta": float(
                    (int_logits.get(correct_option, 0.0) - int_logits.get(wrong_option, 0.0))
                    - (base_logits.get(correct_option, 0.0) - base_logits.get(wrong_option, 0.0))
                ),
                "recheck_margin_delta_vs_interference": float(
                    (re_logits.get(correct_option, 0.0) - re_logits.get(wrong_option, 0.0))
                    - (int_logits.get(correct_option, 0.0) - int_logits.get(wrong_option, 0.0))
                ),
                "baseline_correct_rank": _option_rank(base_logits, correct_option),
                "interference_correct_rank": _option_rank(int_logits, correct_option),
                "recheck_correct_rank": _option_rank(re_logits, correct_option),
                "baseline_wrong_rank": _option_rank(base_logits, wrong_option),
                "interference_wrong_rank": _option_rank(int_logits, wrong_option),
                "recheck_wrong_rank": _option_rank(re_logits, wrong_option),
                "baseline_ranked_options": base_row.get("ranked_options"),
                "interference_ranked_options": int_row.get("ranked_options"),
                "recheck_ranked_options": re_row.get("ranked_options"),
                "baseline_to_interference_final_cosine": (
                    _cosine(np.asarray(base_final), np.asarray(int_final))
                    if base_final is not None and int_final is not None
                    else None
                ),
                "baseline_to_interference_pooled_cosine": (
                    _cosine(np.asarray(base_pooled), np.asarray(int_pooled))
                    if base_pooled is not None and int_pooled is not None
                    else None
                ),
                "interference_to_recheck_final_cosine": (
                    _cosine(np.asarray(int_final), np.asarray(re_final))
                    if int_final is not None and re_final is not None
                    else None
                ),
                "interference_to_recheck_pooled_cosine": (
                    _cosine(np.asarray(int_pooled), np.asarray(re_pooled))
                    if int_pooled is not None and re_pooled is not None
                    else None
                ),
                "baseline_prefix_attention_mean": base_attn.get("prefix_attention_mean"),
                "interference_prefix_attention_mean": int_attn.get("prefix_attention_mean"),
                "recheck_prefix_attention_mean": re_attn.get("prefix_attention_mean"),
                "baseline_wrong_option_prefix_attention_mean": base_attn.get("wrong_option_prefix_attention_mean"),
                "interference_wrong_option_prefix_attention_mean": int_attn.get("wrong_option_prefix_attention_mean"),
                "recheck_wrong_option_prefix_attention_mean": re_attn.get("wrong_option_prefix_attention_mean"),
            }
        )

    sample_row = {
        "sample_id": baseline.get("sample_id"),
        "model_name": baseline.get("model_name"),
        "sample_type": baseline.get("sample_type"),
        "condition_id": baseline.get("condition_id"),
        "authority_level": baseline.get("authority_level"),
        "confidence_level": baseline.get("confidence_level"),
        "explicit_wrong_option": baseline.get("explicit_wrong_option"),
        "is_control": baseline.get("is_control"),
        "baseline_answer": baseline.get("predicted_answer"),
        "interference_answer": interference.get("predicted_answer"),
        "recheck_answer": (recheck or {}).get("predicted_answer"),
        "ground_truth": correct_option,
        "wrong_option": wrong_option,
        "baseline_correct": baseline.get("predicted_answer") == correct_option,
        "interference_correct": interference.get("predicted_answer") == correct_option,
        "recheck_correct": (recheck or {}).get("predicted_answer") == correct_option if recheck else None,
        "transition_label": transition_label(
            baseline.get("predicted_answer") == correct_option,
            interference.get("predicted_answer") == correct_option,
        ),
        "baseline_correct_option_logit": baseline.get("correct_option_logit"),
        "interference_correct_option_logit": interference.get("correct_option_logit"),
        "recheck_correct_option_logit": (recheck or {}).get("correct_option_logit"),
        "baseline_wrong_option_logit": baseline.get("wrong_option_logit"),
        "interference_wrong_option_logit": interference.get("wrong_option_logit"),
        "recheck_wrong_option_logit": (recheck or {}).get("wrong_option_logit"),
        "interference_margin_delta": float(interference.get("correct_wrong_margin", 0.0) - baseline.get("correct_wrong_margin", 0.0)),
        "recheck_margin_delta_vs_interference": float((recheck or {}).get("correct_wrong_margin", 0.0) - interference.get("correct_wrong_margin", 0.0)),
        "per_layer": per_layer,
    }
    return sample_row


def layer_summary_dataframe(sample_rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    flat_rows: List[Dict[str, Any]] = []
    for sample in sample_rows:
        for layer_row in sample.get("per_layer", []):
            flat_rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "transition_label": sample.get("transition_label"),
                    "sample_type": sample.get("sample_type"),
                    **layer_row,
                }
            )
    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    grouped = (
        df.groupby(["transition_label", "sample_type", "layer_index"], dropna=False)
        .agg(
            num_samples=("sample_id", "nunique"),
            mean_interference_margin_delta=("interference_margin_delta", "mean"),
            mean_recheck_margin_delta_vs_interference=("recheck_margin_delta_vs_interference", "mean"),
            mean_baseline_to_interference_final_cosine=("baseline_to_interference_final_cosine", "mean"),
            mean_baseline_to_interference_pooled_cosine=("baseline_to_interference_pooled_cosine", "mean"),
            mean_interference_to_recheck_final_cosine=("interference_to_recheck_final_cosine", "mean"),
            mean_interference_to_recheck_pooled_cosine=("interference_to_recheck_pooled_cosine", "mean"),
            mean_interference_wrong_option_prefix_attention=("interference_wrong_option_prefix_attention_mean", "mean"),
            mean_interference_prefix_attention=("interference_prefix_attention_mean", "mean"),
            mean_interference_correct_rank=("interference_correct_rank", "mean"),
            mean_interference_wrong_rank=("interference_wrong_rank", "mean"),
        )
        .reset_index()
        .sort_values(["transition_label", "sample_type", "layer_index"])
        .reset_index(drop=True)
    )
    return grouped


def focused_subset(sample_rows: Iterable[Dict[str, Any]], transition_name: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in sample_rows if row.get("transition_label") == transition_name]


def save_markdown_report(
    output_path: Path,
    *,
    sample_rows: Iterable[Dict[str, Any]],
    layer_summary: pd.DataFrame,
    patching_rows: Iterable[Dict[str, Any]],
) -> Path:
    rows = list(sample_rows)
    patch_rows = list(patching_rows)
    focus = [row for row in rows if row.get("transition_label") == "baseline_correct_to_interference_wrong"]
    lines = [
        "# Local Probe Mechanistic Report",
        "",
        f"- Total samples: {len(rows)}",
        f"- Baseline correct -> interference wrong: {sum(1 for row in rows if row.get('transition_label') == 'baseline_correct_to_interference_wrong')}",
        f"- Baseline wrong -> interference wrong: {sum(1 for row in rows if row.get('transition_label') == 'baseline_wrong_to_interference_wrong')}",
        f"- Baseline correct -> interference correct: {sum(1 for row in rows if row.get('transition_label') == 'baseline_correct_to_interference_correct')}",
        "",
    ]

    if not layer_summary.empty:
        focus_summary = layer_summary[layer_summary["transition_label"] == "baseline_correct_to_interference_wrong"]
        if not focus_summary.empty:
            worst_layers = focus_summary.nsmallest(5, "mean_interference_margin_delta")[
                ["layer_index", "mean_interference_margin_delta", "mean_baseline_to_interference_final_cosine"]
            ]
            lines.extend(
                [
                    "## Margin Collapse Layers",
                    "",
                    _markdown_table(worst_layers),
                    "",
                ]
            )

    if patch_rows:
        patch_df = pd.DataFrame(patch_rows)
        patch_summary = (
            patch_df.groupby("patch_layer_index")
            .agg(
                num_trials=("sample_id", "count"),
                restore_correct_rate=("restored_correct", "mean"),
                mean_margin_gain=("patched_margin_gain_vs_interference", "mean"),
            )
            .reset_index()
            .sort_values(["restore_correct_rate", "mean_margin_gain"], ascending=[False, False])
            .head(10)
        )
        lines.extend(
            [
                "## Patching Highlights",
                "",
                _markdown_table(patch_summary),
                "",
            ]
        )

    if focus:
        lines.extend(
            [
                "## Interpretation",
                "",
                "1. 受压后正确答案是否在哪些层开始丢优势：看 `layer_summary.csv` 中 `mean_interference_margin_delta` 最负的层。",
                "2. 错误选项优势是逐层积累还是输出层突然形成：对比 `interference_wrong_rank` 与逐层 margin。",
                "3. recheck 为什么没救回来：对比 `interference_to_recheck_*_cosine` 和 `recheck_margin_delta_vs_interference`。",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def select_anchor_expansion_heads(
    head_summary: pd.DataFrame,
    *,
    anchor_head: Tuple[int, int],
    allowed_layers: Sequence[int] | None = None,
    max_heads: int = 8,
) -> List[Tuple[int, int]]:
    if head_summary.empty:
        return [anchor_head]

    filtered = head_summary.copy()
    if allowed_layers is not None:
        allowed = {int(layer) for layer in allowed_layers}
        filtered = filtered[filtered["patch_layer_index"].astype(int).isin(allowed)]

    filtered = filtered.sort_values(
        ["restore_correct_rate", "mean_margin_gain", "patch_layer_index", "head_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    selected: List[Tuple[int, int]] = [(int(anchor_head[0]), int(anchor_head[1]))]
    seen = {selected[0]}
    for row in filtered.to_dict(orient="records"):
        candidate = (int(row["patch_layer_index"]), int(row["head_index"]))
        if candidate in seen:
            continue
        selected.append(candidate)
        seen.add(candidate)
        if len(selected) >= int(max_heads):
            break
    return selected


def build_anchor_head_expansion_sets(
    ordered_heads: Sequence[Tuple[int, int]],
    *,
    target_sizes: Sequence[int] = (1, 2, 4, 8),
) -> List[Dict[str, Any]]:
    if not ordered_heads:
        return []

    expansions: List[Dict[str, Any]] = []
    seen_sizes = set()
    for raw_size in target_sizes:
        target_size = min(int(raw_size), len(ordered_heads))
        if target_size <= 0 or target_size in seen_sizes:
            continue
        seen_sizes.add(target_size)
        head_set = [(int(layer), int(head)) for layer, head in islice(ordered_heads, target_size)]
        expansions.append(
            {
                "expansion_size": int(target_size),
                "head_set": head_set,
                "head_set_label": "+".join(f"L{layer}H{head}" for layer, head in head_set),
            }
        )
    return expansions


def build_drop_wrong_option_case_rows(
    sample_notes: Sequence[Dict[str, Any]],
    *,
    ablation_df: pd.DataFrame,
    head_df: pd.DataFrame | None = None,
    expansion_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ablation_map = {
        (row["sample_id"], row["ablation"]): row
        for row in ablation_df.to_dict(orient="records")
    }
    head_best_map: Dict[str, Dict[str, Any]] = {}
    if head_df is not None and not head_df.empty:
        best_head_df = (
            head_df.sort_values(["restored_correct", "mean_margin_gain"], ascending=[False, False])
            .groupby("sample_id", dropna=False)
            .head(1)
        )
        head_best_map = {str(row["sample_id"]): row for row in best_head_df.to_dict(orient="records")}

    expansion_best_map: Dict[str, Dict[str, Any]] = {}
    if expansion_df is not None and not expansion_df.empty:
        best_expansion_df = (
            expansion_df.sort_values(["restored_correct", "mean_margin_gain"], ascending=[False, False])
            .groupby("sample_id", dropna=False)
            .head(1)
        )
        expansion_best_map = {str(row["sample_id"]): row for row in best_expansion_df.to_dict(orient="records")}

    case_rows: List[Dict[str, Any]] = []
    for note in sample_notes:
        sample_id = str(note["sample_id"])
        none_row = ablation_map.get((sample_id, "none"), {})
        drop_wrong_row = ablation_map.get((sample_id, "drop_wrong_option"), {})
        best_head = head_best_map.get(sample_id, {})
        best_expansion = expansion_best_map.get(sample_id, {})
        margin_improvement = None
        if none_row.get("late_layer_margin_delta") is not None and drop_wrong_row.get("late_layer_margin_delta") is not None:
            margin_improvement = float(drop_wrong_row["late_layer_margin_delta"]) - float(none_row["late_layer_margin_delta"])
        recovers_correct = bool(drop_wrong_row.get("is_correct"))
        none_wrong_follow = bool(none_row.get("wrong_option_follow"))
        dependence = "limited"
        if recovers_correct and none_wrong_follow:
            dependence = "high"
        elif margin_improvement is not None and margin_improvement >= 1.5:
            dependence = "moderate"
        case_rows.append(
            {
                **note,
                "drop_wrong_option_recovers_correct": recovers_correct,
                "none_predicted_answer": none_row.get("predicted_answer"),
                "drop_wrong_option_answer": drop_wrong_row.get("predicted_answer"),
                "none_wrong_option_follow": none_wrong_follow,
                "drop_wrong_option_wrong_option_follow": bool(drop_wrong_row.get("wrong_option_follow")),
                "none_late_layer_margin_delta": none_row.get("late_layer_margin_delta"),
                "drop_wrong_option_late_layer_margin_delta": drop_wrong_row.get("late_layer_margin_delta"),
                "drop_wrong_option_margin_improvement": margin_improvement,
                "best_single_head_label": (
                    f"L{int(best_head['patch_layer_index'])}H{int(best_head['head_index'])}"
                    if best_head
                    else None
                ),
                "best_single_head_restores_correct": bool(best_head.get("restored_correct")) if best_head else None,
                "best_single_head_margin_gain": float(best_head["mean_margin_gain"]) if best_head else None,
                "best_expansion_label": best_expansion.get("head_set_label"),
                "best_expansion_size": int(best_expansion["expansion_size"]) if best_expansion else None,
                "best_expansion_restores_correct": bool(best_expansion.get("restored_correct")) if best_expansion else None,
                "best_expansion_margin_gain": float(best_expansion["mean_margin_gain"]) if best_expansion else None,
                "explicit_wrong_option_cue_dependence": dependence,
            }
        )
    case_df = pd.DataFrame(case_rows)
    if case_df.empty:
        return case_df
    return case_df.sort_values(
        [
            "drop_wrong_option_recovers_correct",
            "explicit_wrong_option_cue_dependence",
            "drop_wrong_option_margin_improvement",
        ],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def summarize_drop_wrong_option_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame(
            columns=[
                "drop_wrong_option_recovers_correct",
                "num_samples",
                "mean_margin_improvement",
                "wrong_option_follow_rate_none",
                "wrong_option_follow_rate_drop_wrong_option",
                "mean_best_single_head_margin_gain",
                "mean_best_expansion_margin_gain",
                "best_expansion_restore_rate",
            ]
        )
    return (
        case_df.groupby("drop_wrong_option_recovers_correct", dropna=False)
        .agg(
            num_samples=("sample_id", "count"),
            mean_margin_improvement=("drop_wrong_option_margin_improvement", "mean"),
            wrong_option_follow_rate_none=("none_wrong_option_follow", "mean"),
            wrong_option_follow_rate_drop_wrong_option=("drop_wrong_option_wrong_option_follow", "mean"),
            mean_best_single_head_margin_gain=("best_single_head_margin_gain", "mean"),
            mean_best_expansion_margin_gain=("best_expansion_margin_gain", "mean"),
            best_expansion_restore_rate=("best_expansion_restores_correct", "mean"),
        )
        .reset_index()
        .sort_values("drop_wrong_option_recovers_correct", ascending=False)
        .reset_index(drop=True)
    )
