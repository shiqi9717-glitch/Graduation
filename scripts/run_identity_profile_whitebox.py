#!/usr/bin/env python3
"""Run the minimal identity_profile white-box study."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_bridge_dataset
from src.logging_config import setup_logger
from src.open_model_probe.identity_profile import (
    baseline_core_question,
    final_token_position,
    matched_nonprefix_positions_for_prompt,
    mean_bool,
    parse_layers,
    prefix_positions_for_prompt,
    sample_identity_subset,
    summarize_norm_delta,
    write_csv,
    write_jsonl,
)
from src.open_model_probe.io_utils import prepare_output_dir, save_json
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _scenario_prompt(item: Dict[str, Any], scenario: str) -> str:
    mapping = {
        "baseline": "baseline_prompt",
        "identity_profile": "pressured_prompt",
        "belief_argument": "pressured_prompt",
        "recovery": "recovery_prompt",
    }
    return str(item[mapping[scenario]])


def _extract_record_arrays(
    *,
    runner: LocalProbeRunner,
    item: Dict[str, Any],
    scenario: str,
    early_layers: Sequence[int],
    late_layers: Sequence[int],
    max_prefix_positions: int,
) -> Dict[str, Any]:
    prompt = _scenario_prompt(item, scenario)
    core_question = baseline_core_question(item)
    tokenizer = runner._tokenizer
    assert tokenizer is not None
    prefix_positions = prefix_positions_for_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        core_question=core_question,
        max_prefix_positions=max_prefix_positions,
    )
    matched_control_positions = matched_nonprefix_positions_for_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        core_question=core_question,
        match_count=len(prefix_positions),
    )
    final_pos = final_token_position(tokenizer=tokenizer, prompt=prompt)
    selected_layers = tuple(sorted(set(int(layer) for layer in [*early_layers, *late_layers])))
    result = runner.analyze_prompt_selected_layers(
        prompt=prompt,
        sample_id=str(item["item_id"]),
        scenario=scenario,
        question_text=prompt,
        prompt_prefix="",
        ground_truth=str(item["answer_matching_behavior"]),
        wrong_option=str(item["answer_not_matching_behavior"]),
        selected_layers=selected_layers,
    )
    early_capture = runner.capture_layer_token_residuals(
        prompt=prompt,
        target_layers=tuple(int(layer) for layer in early_layers),
        token_positions=sorted(set([*prefix_positions, *matched_control_positions])),
    )
    early_prefix_arrays: Dict[str, np.ndarray] = {}
    early_control_arrays: Dict[str, np.ndarray] = {}
    for layer in early_layers:
        states = [np.asarray(early_capture.get(int(layer), {}).get(int(pos)), dtype=np.float32) for pos in prefix_positions if int(pos) in early_capture.get(int(layer), {})]
        early_prefix_arrays[f"layer_{int(layer)}_prefix_states"] = (
            np.stack(states).astype(np.float32) if states else np.zeros((0, 0), dtype=np.float32)
        )
        control_states = [np.asarray(early_capture.get(int(layer), {}).get(int(pos)), dtype=np.float32) for pos in matched_control_positions if int(pos) in early_capture.get(int(layer), {})]
        early_control_arrays[f"layer_{int(layer)}_control_states"] = (
            np.stack(control_states).astype(np.float32) if control_states else np.zeros((0, 0), dtype=np.float32)
        )
    late_final_arrays = {
        f"layer_{int(layer)}_final_token": np.asarray(result["_hidden_state_arrays"][f"layer_{int(layer)}_final_token"], dtype=np.float32)
        for layer in late_layers
    }
    return {
        "prompt": prompt,
        "prefix_positions": prefix_positions,
        "matched_control_positions": matched_control_positions,
        "final_token_position": int(final_pos),
        "predicted_answer": result.get("predicted_answer", ""),
        "answer_logits": result.get("answer_logits", {}),
        "correct_wrong_margin": result.get("correct_wrong_margin"),
        "early_prefix_arrays": early_prefix_arrays,
        "early_control_arrays": early_control_arrays,
        "late_final_arrays": late_final_arrays,
    }


def _save_arrays(
    path: Path,
    *,
    prefix_positions: Sequence[int],
    matched_control_positions: Sequence[int],
    early_prefix_arrays: Dict[str, np.ndarray],
    early_control_arrays: Dict[str, np.ndarray],
    late_final_arrays: Dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "prefix_positions": np.asarray(list(prefix_positions), dtype=np.int32),
        "matched_control_positions": np.asarray(list(matched_control_positions), dtype=np.int32),
    }
    payload.update(early_prefix_arrays)
    payload.update(early_control_arrays)
    payload.update(late_final_arrays)
    np.savez_compressed(path, **payload)


def _collect_hidden_records(
    *,
    runner: LocalProbeRunner,
    items: Sequence[Dict[str, Any]],
    early_layers: Sequence[int],
    late_layers: Sequence[int],
    output_dir: Path,
    max_prefix_positions: int,
    logger: Any,
    flush_every: int,
) -> List[Dict[str, Any]]:
    arrays_dir = output_dir / "identity_hidden_arrays"
    records: List[Dict[str, Any]] = []
    total = len(items) * 3
    seen = 0
    for item in items:
        for scenario in ("baseline", item["study_group"], "recovery"):
            seen += 1
            extracted = _extract_record_arrays(
                runner=runner,
                item=item,
                scenario=scenario,
                early_layers=early_layers,
                late_layers=late_layers,
                max_prefix_positions=max_prefix_positions,
            )
            array_path = arrays_dir / f"{item['item_id']}__{scenario}.npz"
            _save_arrays(
                array_path,
                prefix_positions=extracted["prefix_positions"],
                matched_control_positions=extracted["matched_control_positions"],
                early_prefix_arrays=extracted["early_prefix_arrays"],
                early_control_arrays=extracted["early_control_arrays"],
                late_final_arrays=extracted["late_final_arrays"],
            )
            records.append(
                {
                    "item_id": item["item_id"],
                    "study_group": item["study_group"],
                    "split": item["split"],
                    "source": item["source"],
                    "pressure_type": item["pressure_type"],
                    "scenario": scenario,
                    "parsed_answer": extracted["predicted_answer"],
                    "answer_logits": extracted["answer_logits"],
                    "correct_wrong_margin": extracted["correct_wrong_margin"],
                    "prefix_positions": list(extracted["prefix_positions"]),
                    "matched_control_positions": list(extracted["matched_control_positions"]),
                    "final_token_position": extracted["final_token_position"],
                    "hidden_array_path": str(array_path.resolve()),
                    "max_prefix_positions": int(max_prefix_positions),
                }
            )
            if seen == 1 or seen % max(int(flush_every), 1) == 0 or seen == total:
                logger.info("Hidden extraction progress %d/%d", seen, total)
                write_jsonl(output_dir / "identity_hidden_records.partial.jsonl", records)
    write_jsonl(output_dir / "identity_hidden_records.jsonl", records)
    return records


def _records_by_key(records: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    return {(str(row["item_id"]), str(row["scenario"])): dict(row) for row in records}


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    return {key: np.asarray(arrays[key]) for key in arrays.files}


def _mean_prefix_templates(records: Sequence[Dict[str, Any]], early_layers: Sequence[int]) -> Dict[int, Dict[int, np.ndarray]]:
    baseline_records = [row for row in records if row["split"] == "train" and row["study_group"] == "identity_profile" and row["scenario"] == "baseline"]
    per_layer_pos: Dict[int, Dict[int, List[np.ndarray]]] = {int(layer): {} for layer in early_layers}
    for record in baseline_records:
        arrays = _load_npz(record["hidden_array_path"])
        for layer in early_layers:
            prefix_states = np.asarray(arrays.get(f"layer_{int(layer)}_prefix_states"), dtype=np.float32)
            for rel_pos in range(prefix_states.shape[0]):
                per_layer_pos[int(layer)].setdefault(int(rel_pos), []).append(prefix_states[rel_pos])
    means: Dict[int, Dict[int, np.ndarray]] = {}
    for layer, position_map in per_layer_pos.items():
        means[int(layer)] = {
            int(pos): np.stack(vectors).mean(axis=0).astype(np.float32)
            for pos, vectors in position_map.items()
            if vectors
        }
    return means


def _mean_control_templates(records: Sequence[Dict[str, Any]], early_layers: Sequence[int]) -> Dict[int, Dict[int, np.ndarray]]:
    baseline_records = [row for row in records if row["split"] == "train" and row["study_group"] == "identity_profile" and row["scenario"] == "baseline"]
    per_layer_pos: Dict[int, Dict[int, List[np.ndarray]]] = {int(layer): {} for layer in early_layers}
    for record in baseline_records:
        arrays = _load_npz(record["hidden_array_path"])
        for layer in early_layers:
            control_states = np.asarray(arrays.get(f"layer_{int(layer)}_control_states"), dtype=np.float32)
            for rel_pos in range(control_states.shape[0]):
                per_layer_pos[int(layer)].setdefault(int(rel_pos), []).append(control_states[rel_pos])
    means: Dict[int, Dict[int, np.ndarray]] = {}
    for layer, position_map in per_layer_pos.items():
        means[int(layer)] = {
            int(pos): np.stack(vectors).mean(axis=0).astype(np.float32)
            for pos, vectors in position_map.items()
            if vectors
        }
    return means


def _mean_late_final_templates(records: Sequence[Dict[str, Any]], late_layers: Sequence[int]) -> Dict[int, np.ndarray]:
    baseline_records = [row for row in records if row["split"] == "train" and row["study_group"] == "identity_profile" and row["scenario"] == "baseline"]
    per_layer: Dict[int, List[np.ndarray]] = {int(layer): [] for layer in late_layers}
    for record in baseline_records:
        arrays = _load_npz(record["hidden_array_path"])
        for layer in late_layers:
            per_layer[int(layer)].append(np.asarray(arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32))
    return {
        int(layer): np.stack(vectors).mean(axis=0).astype(np.float32)
        for layer, vectors in per_layer.items()
        if vectors
    }


def _localization_summary(
    *,
    records: Sequence[Dict[str, Any]],
    early_layers: Sequence[int],
    late_layers: Sequence[int],
    output_path: Path,
) -> List[Dict[str, Any]]:
    by_key = _records_by_key(records)
    items = sorted({(row["item_id"], row["study_group"]) for row in records if row["scenario"] == "baseline"})
    rows: List[Dict[str, Any]] = []
    group_values: Dict[str, Dict[str, List[float]]] = {}
    for item_id, study_group in items:
        baseline = by_key[(item_id, "baseline")]
        pressured = by_key[(item_id, study_group)]
        recovery = by_key[(item_id, "recovery")]
        base_arrays = _load_npz(baseline["hidden_array_path"])
        pressured_arrays = _load_npz(pressured["hidden_array_path"])
        recovery_arrays = _load_npz(recovery["hidden_array_path"])
        early_bp = []
        early_pr = []
        for layer in early_layers:
            base_prefix = np.asarray(base_arrays.get(f"layer_{int(layer)}_prefix_states"), dtype=np.float32)
            press_prefix = np.asarray(pressured_arrays.get(f"layer_{int(layer)}_prefix_states"), dtype=np.float32)
            rec_prefix = np.asarray(recovery_arrays.get(f"layer_{int(layer)}_prefix_states"), dtype=np.float32)
            shared_bp = min(base_prefix.shape[0], press_prefix.shape[0])
            shared_pr = min(press_prefix.shape[0], rec_prefix.shape[0])
            if shared_bp > 0:
                early_bp.append(float(np.linalg.norm(press_prefix[:shared_bp] - base_prefix[:shared_bp])))
            if shared_pr > 0:
                early_pr.append(float(np.linalg.norm(rec_prefix[:shared_pr] - press_prefix[:shared_pr])))
        late_bp = []
        late_pr = []
        for layer in late_layers:
            base_final = np.asarray(base_arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
            press_final = np.asarray(pressured_arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
            rec_final = np.asarray(recovery_arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
            late_bp.append(summarize_norm_delta(base_final, press_final))
            late_pr.append(summarize_norm_delta(press_final, rec_final))
        metrics = group_values.setdefault(study_group, {"early_bp": [], "early_pr": [], "late_bp": [], "late_pr": [], "aff": [], "frame": []})
        metrics["early_bp"].append(float(np.mean(early_bp)) if early_bp else 0.0)
        metrics["early_pr"].append(float(np.mean(early_pr)) if early_pr else 0.0)
        metrics["late_bp"].append(float(np.mean(late_bp)) if late_bp else 0.0)
        metrics["late_pr"].append(float(np.mean(late_pr)) if late_pr else 0.0)
        metrics["aff"].append(1.0 if pressured.get("parsed_answer") == pressured.get("answer_logits", {}).get("answer_matching_behavior") else 0.0)
        metrics["frame"].append(1.0 if baseline.get("parsed_answer") != pressured.get("parsed_answer") else 0.0)

    for study_group, metrics in group_values.items():
        rows.append(
            {
                "study_group": study_group,
                "mean_baseline_to_pressured_prefix_delta": float(np.mean(metrics["early_bp"])),
                "mean_pressured_to_recovery_prefix_delta": float(np.mean(metrics["early_pr"])),
                "mean_baseline_to_pressured_late_final_delta": float(np.mean(metrics["late_bp"])),
                "mean_pressured_to_recovery_late_final_delta": float(np.mean(metrics["late_pr"])),
                "early_mid_vs_late_ratio": float(np.mean(metrics["early_bp"])) / max(float(np.mean(metrics["late_bp"])), 1e-6),
            }
        )
    if "identity_profile" in group_values and "belief_argument" in group_values:
        ident = next(row for row in rows if row["study_group"] == "identity_profile")
        belief = next(row for row in rows if row["study_group"] == "belief_argument")
        rows.append(
            {
                "study_group": "identity_minus_belief",
                "mean_baseline_to_pressured_prefix_delta": ident["mean_baseline_to_pressured_prefix_delta"] - belief["mean_baseline_to_pressured_prefix_delta"],
                "mean_pressured_to_recovery_prefix_delta": ident["mean_pressured_to_recovery_prefix_delta"] - belief["mean_pressured_to_recovery_prefix_delta"],
                "mean_baseline_to_pressured_late_final_delta": ident["mean_baseline_to_pressured_late_final_delta"] - belief["mean_baseline_to_pressured_late_final_delta"],
                "mean_pressured_to_recovery_late_final_delta": ident["mean_pressured_to_recovery_late_final_delta"] - belief["mean_pressured_to_recovery_late_final_delta"],
                "early_mid_vs_late_ratio": ident["early_mid_vs_late_ratio"] - belief["early_mid_vs_late_ratio"],
            }
        )
    write_csv(output_path, rows)
    return rows


def _identity_eval_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(item) for item in items if item["study_group"] == "identity_profile" and item["split"] == "eval"]


def _profile_patch_map(
    *,
    record: Dict[str, Any],
    early_layers: Sequence[int],
    prefix_means: Dict[int, Dict[int, np.ndarray]],
    alpha: float,
) -> Dict[int, Dict[int, np.ndarray]]:
    arrays = _load_npz(record["hidden_array_path"])
    absolute_positions = [int(pos) for pos in record.get("prefix_positions", [])]
    patch_map: Dict[int, Dict[int, np.ndarray]] = {}
    for layer in early_layers:
        prefix_states = np.asarray(arrays.get(f"layer_{int(layer)}_prefix_states"), dtype=np.float32)
        patch_map[int(layer)] = {}
        for rel_pos in range(prefix_states.shape[0]):
            if rel_pos not in prefix_means.get(int(layer), {}):
                continue
            target = prefix_means[int(layer)][int(rel_pos)]
            state = prefix_states[rel_pos]
            if rel_pos >= len(absolute_positions):
                continue
            patch_map[int(layer)][int(absolute_positions[rel_pos])] = (state - float(alpha) * (state - target)).astype(np.float32)
    return {layer: pos_map for layer, pos_map in patch_map.items() if pos_map}


def _matched_control_patch_map(
    *,
    record: Dict[str, Any],
    early_layers: Sequence[int],
    control_means: Dict[int, Dict[int, np.ndarray]],
    alpha: float,
) -> Dict[int, Dict[int, np.ndarray]]:
    arrays = _load_npz(record["hidden_array_path"])
    absolute_positions = [int(pos) for pos in record.get("matched_control_positions", [])]
    patch_map: Dict[int, Dict[int, np.ndarray]] = {}
    for layer in early_layers:
        control_states = np.asarray(arrays.get(f"layer_{int(layer)}_control_states"), dtype=np.float32)
        patch_map[int(layer)] = {}
        for rel_pos in range(control_states.shape[0]):
            if rel_pos not in control_means.get(int(layer), {}):
                continue
            if rel_pos >= len(absolute_positions):
                continue
            target = control_means[int(layer)][int(rel_pos)]
            state = control_states[rel_pos]
            patch_map[int(layer)][int(absolute_positions[rel_pos])] = (state - float(alpha) * (state - target)).astype(np.float32)
    return {layer: pos_map for layer, pos_map in patch_map.items() if pos_map}


def _late_patch_map(
    *,
    record: Dict[str, Any],
    late_layers: Sequence[int],
    late_means: Dict[int, np.ndarray],
    alpha: float,
) -> Dict[int, np.ndarray]:
    arrays = _load_npz(record["hidden_array_path"])
    patch_map: Dict[int, np.ndarray] = {}
    for layer in late_layers:
        state = np.asarray(arrays[f"layer_{int(layer)}_final_token"], dtype=np.float32)
        target = late_means[int(layer)]
        patch_map[int(layer)] = (state - float(alpha) * (state - target)).astype(np.float32)
    return patch_map


def _run_interventions(
    *,
    runner: LocalProbeRunner,
    items: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    early_layers: Sequence[int],
    late_layers: Sequence[int],
    alpha_values: Sequence[float],
    output_dir: Path,
    logger: Any,
    flush_every: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = _records_by_key(records)
    identity_eval = _identity_eval_items(items)
    prefix_means = _mean_prefix_templates(records, early_layers=early_layers)
    control_means = _mean_control_templates(records, early_layers=early_layers)
    late_means = _mean_late_final_templates(records, late_layers=late_layers)
    methods = ("profile_prefix_gating", "matched_early_mid_negative_control", "late_layer_negative_control")

    result_records: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    total = len(identity_eval) * len(alpha_values) * len(methods)
    done = 0
    for method in methods:
        for alpha in alpha_values:
            for item in identity_eval:
                done += 1
                item_id = str(item["item_id"])
                outputs: Dict[str, Dict[str, Any]] = {}
                for scenario in ("baseline", "identity_profile", "recovery"):
                    ref = by_key[(item_id, scenario)]
                    prompt = _scenario_prompt(item, scenario)
                    if method == "profile_prefix_gating":
                        patch_map = _profile_patch_map(
                            record=ref,
                            early_layers=early_layers,
                            prefix_means=prefix_means,
                            alpha=float(alpha),
                        )
                        if patch_map:
                            outputs[scenario] = runner.patch_residual_positions_multi_layer(
                                prompt=prompt,
                                layer_position_patch_map=patch_map,
                                ground_truth=str(item["answer_matching_behavior"]),
                                wrong_option=str(item["answer_not_matching_behavior"]),
                            )
                        else:
                            outputs[scenario] = {
                                "predicted_answer": ref.get("parsed_answer", ""),
                                "answer_logits": ref.get("answer_logits", {}),
                                "correct_wrong_margin": ref.get("correct_wrong_margin"),
                            }
                    elif method == "matched_early_mid_negative_control":
                        patch_map = _matched_control_patch_map(
                            record=ref,
                            early_layers=early_layers,
                            control_means=control_means,
                            alpha=float(alpha),
                        )
                        if patch_map:
                            outputs[scenario] = runner.patch_residual_positions_multi_layer(
                                prompt=prompt,
                                layer_position_patch_map=patch_map,
                                ground_truth=str(item["answer_matching_behavior"]),
                                wrong_option=str(item["answer_not_matching_behavior"]),
                            )
                        else:
                            outputs[scenario] = {
                                "predicted_answer": ref.get("parsed_answer", ""),
                                "answer_logits": ref.get("answer_logits", {}),
                                "correct_wrong_margin": ref.get("correct_wrong_margin"),
                            }
                    else:
                        late_patch = _late_patch_map(
                            record=ref,
                            late_layers=late_layers,
                            late_means=late_means,
                            alpha=float(alpha),
                        )
                        outputs[scenario] = runner.patch_final_token_residuals_multi(
                            prompt=prompt,
                            layer_patch_map=late_patch,
                            ground_truth=str(item["answer_matching_behavior"]),
                            wrong_option=str(item["answer_not_matching_behavior"]),
                        )
                    result_records.append(
                        {
                            "item_id": item_id,
                            "source": item["source"],
                            "method": method,
                            "alpha": float(alpha),
                            "scenario": scenario,
                            "parsed_answer": outputs[scenario].get("predicted_answer", ""),
                            "answer_logits": outputs[scenario].get("answer_logits", {}),
                            "correct_wrong_margin": outputs[scenario].get("correct_wrong_margin"),
                        }
                    )
                baseline_ref = by_key[(item_id, "baseline")]
                pressured_ref = by_key[(item_id, "identity_profile")]
                recovery_ref = by_key[(item_id, "recovery")]
                comparisons.append(
                    {
                        "item_id": item_id,
                        "source": item["source"],
                        "method": method,
                        "alpha": float(alpha),
                        "baseline_choice": baseline_ref.get("parsed_answer", ""),
                        "identity_profile_choice_baseline": pressured_ref.get("parsed_answer", ""),
                        "recovery_choice_baseline": recovery_ref.get("parsed_answer", ""),
                        "baseline_choice_intervention": outputs["baseline"].get("predicted_answer", ""),
                        "identity_profile_choice_intervention": outputs["identity_profile"].get("predicted_answer", ""),
                        "recovery_choice_intervention": outputs["recovery"].get("predicted_answer", ""),
                        "answer_matching_behavior": item["answer_matching_behavior"],
                    }
                )
                if done == 1 or done % max(int(flush_every), 1) == 0 or done == total:
                    logger.info("Intervention progress %d/%d", done, total)
                    write_jsonl(output_dir / "identity_profile_intervention_records.partial.jsonl", result_records)
                    write_jsonl(output_dir / "identity_profile_intervention_comparisons.partial.jsonl", comparisons)

    summary_rows = _summarize_interventions(comparisons)
    write_jsonl(output_dir / "identity_profile_intervention_records.jsonl", result_records)
    write_jsonl(output_dir / "identity_profile_intervention_comparisons.jsonl", comparisons)
    write_csv(output_dir / "identity_profile_intervention_summary.csv", summary_rows)
    return result_records, comparisons, summary_rows


def _summarize_interventions(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in comparisons})
    alphas = sorted({float(row["alpha"]) for row in comparisons})
    per_alpha_effects: Dict[float, Dict[str, float]] = {}
    for alpha in alphas:
        alpha_rows = [row for row in comparisons if float(row["alpha"]) == alpha]
        reductions = {}
        for method in methods:
            subset = [row for row in alpha_rows if row["method"] == method]
            base_aff = mean_bool(row["identity_profile_choice_baseline"] == row["answer_matching_behavior"] for row in subset) or 0.0
            int_aff = mean_bool(row["identity_profile_choice_intervention"] == row["answer_matching_behavior"] for row in subset) or 0.0
            reductions[method] = float(base_aff - int_aff)
        per_alpha_effects[float(alpha)] = reductions
    for method in methods:
        for alpha in alphas:
            subset = [row for row in comparisons if row["method"] == method and float(row["alpha"]) == alpha]
            reductions = per_alpha_effects[float(alpha)]
            prefix_reduction = float(reductions.get("profile_prefix_gating", 0.0))
            early_control_reduction = float(reductions.get("matched_early_mid_negative_control", 0.0))
            late_reduction = float(reductions.get("late_layer_negative_control", 0.0))
            rows.append(
                {
                    "method": method,
                    "alpha": float(alpha),
                    "num_items": len(subset),
                    "affiliation_conformity_rate": mean_bool(row["identity_profile_choice_intervention"] == row["answer_matching_behavior"] for row in subset),
                    "framing_shift_rate": mean_bool(row["baseline_choice_intervention"] != row["identity_profile_choice_intervention"] for row in subset),
                    "recovery_after_deprofiling": mean_bool(row["recovery_choice_intervention"] == row["baseline_choice"] for row in subset),
                    "baseline_damage_rate": mean_bool(row["baseline_choice_intervention"] != row["baseline_choice"] for row in subset),
                    "prefix_localization_gain": prefix_reduction - late_reduction,
                    "early_mid_vs_late_effect_ratio": (prefix_reduction / late_reduction) if late_reduction > 0 else None,
                    "profile_vs_matched_early_mid_gap": prefix_reduction - early_control_reduction,
                    "profile_vs_late_gap": prefix_reduction - late_reduction,
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal identity_profile white-box study.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--early-layers", required=True)
    parser.add_argument("--late-layers", required=True)
    parser.add_argument("--output-root", default="outputs/experiments/identity_profile_whitebox")
    parser.add_argument("--identity-n", type=int, default=36)
    parser.add_argument("--belief-n", type=int, default=36)
    parser.add_argument("--train-n-identity", type=int, default=18)
    parser.add_argument("--train-n-belief-per-source", type=int, default=9)
    parser.add_argument("--alpha-values", default="0.25,0.5,0.75")
    parser.add_argument("--max-prefix-positions", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="identity_profile_whitebox", level=args.log_level)
    early_layers = parse_layers(str(args.early_layers))
    late_layers = parse_layers(str(args.late_layers))
    alpha_values = tuple(float(x.strip()) for x in str(args.alpha_values).split(",") if x.strip())
    output_dir = prepare_output_dir(Path(args.output_root), run_name=str(args.model_name).replace("/", "_"))
    all_items = [item.to_dict() for item in build_bridge_dataset(_source_paths())]
    identity_train, identity_eval, belief_train, belief_eval = sample_identity_subset(
        items=all_items,
        identity_n=int(args.identity_n),
        belief_n=int(args.belief_n),
        train_n_identity=int(args.train_n_identity),
        train_n_belief_per_source=int(args.train_n_belief_per_source),
        seed=int(args.seed),
    )
    identity_pairs = identity_train + identity_eval
    all_study_items = identity_pairs + belief_train + belief_eval
    write_jsonl(output_dir / "identity_profile_pairs.jsonl", identity_pairs)
    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=(-1,),
        )
    )
    runner.load()
    hidden_records = _collect_hidden_records(
        runner=runner,
        items=all_study_items,
        early_layers=early_layers,
        late_layers=late_layers,
        output_dir=output_dir,
        max_prefix_positions=int(args.max_prefix_positions),
        logger=logger,
        flush_every=int(args.flush_every),
    )
    localization_rows = _localization_summary(
        records=hidden_records,
        early_layers=early_layers,
        late_layers=late_layers,
        output_path=output_dir / "identity_vs_belief_localization_summary.csv",
    )
    _, comparisons, summary_rows = _run_interventions(
        runner=runner,
        items=identity_pairs,
        records=hidden_records,
        early_layers=early_layers,
        late_layers=late_layers,
        alpha_values=alpha_values,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )
    manifest = {
        "pipeline": "identity_profile_whitebox_v1",
        "model_name": str(args.model_name),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "early_layers": list(early_layers),
        "late_layers": list(late_layers),
        "identity_pairs_n": len(identity_pairs),
        "identity_train_n": len(identity_train),
        "identity_eval_n": len(identity_eval),
        "belief_train_n": len(belief_train),
        "belief_eval_n": len(belief_eval),
        "alpha_values": list(alpha_values),
        "max_prefix_positions": int(args.max_prefix_positions),
        "identity_source_counts": dict(Counter(row["source"] for row in identity_pairs)),
        "belief_source_counts": dict(Counter(row["source"] for row in belief_train + belief_eval)),
        "num_hidden_records": len(hidden_records),
        "num_intervention_comparisons": len(comparisons),
        "num_localization_rows": len(localization_rows),
        "num_intervention_summary_rows": len(summary_rows),
        "outputs": {
            "identity_profile_pairs": str((output_dir / "identity_profile_pairs.jsonl").resolve()),
            "identity_hidden_records": str((output_dir / "identity_hidden_records.jsonl").resolve()),
            "identity_hidden_arrays": str((output_dir / "identity_hidden_arrays").resolve()),
            "identity_vs_belief_localization_summary": str((output_dir / "identity_vs_belief_localization_summary.csv").resolve()),
            "identity_profile_intervention_records": str((output_dir / "identity_profile_intervention_records.jsonl").resolve()),
            "identity_profile_intervention_comparisons": str((output_dir / "identity_profile_intervention_comparisons.jsonl").resolve()),
            "identity_profile_intervention_summary": str((output_dir / "identity_profile_intervention_summary.csv").resolve()),
        },
    }
    save_json(output_dir / "identity_profile_run.json", manifest)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "identity_pairs_n": len(identity_pairs), "identity_eval_n": len(identity_eval)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
