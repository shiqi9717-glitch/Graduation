#!/usr/bin/env python3
"""Run the minimal pressure-subspace damping pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_bridge_dataset
from src.bridge_benchmark.protocol import FIXED_SCENARIOS
from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_json
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.pressure_subspace import (
    cosine_similarity,
    damp_hidden_state,
    default_subspace_specs,
    default_transfer_tests,
    estimate_pca_components,
    mean_bool,
    parse_layers,
    read_jsonl,
    sanitize_key,
    validate_vector,
    write_csv,
    write_jsonl,
)


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _sample_pairs(*, items: Sequence[Dict[str, Any]], items_per_stratum: int, train_per_stratum: int, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(int(seed))
    train: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    strata = sorted({(str(item["source"]), str(item["pressure_type"])) for item in items})
    for source, pressure_type in strata:
        candidates = [dict(item) for item in items if item["source"] == source and item["pressure_type"] == pressure_type]
        rng.shuffle(candidates)
        selected = candidates[: min(int(items_per_stratum), len(candidates))]
        split_at = min(int(train_per_stratum), len(selected))
        for row in selected[:split_at]:
            row["split"] = "train"
            train.append(row)
        for row in selected[split_at:]:
            row["split"] = "eval"
            eval_rows.append(row)
    return train, eval_rows


def _scenario_prompt(item: Dict[str, Any], scenario: str) -> str:
    return str(item[f"{scenario}_prompt"])


def _arrays_for_layers(arrays: Dict[str, np.ndarray], *, item_id: str, scenario: str, layers: Sequence[int]) -> Dict[str, np.ndarray]:
    selected: Dict[str, np.ndarray] = {}
    for layer in layers:
        key = f"layer_{int(layer)}_final_token"
        if key not in arrays:
            raise ValueError(f"{item_id}/{scenario}: missing hidden-state key {key}.")
        value = np.asarray(arrays[key], dtype=np.float32)
        validate_vector(value, item_id=item_id, scenario=scenario, layer=int(layer))
        selected[key] = value
    first_shape = next(iter(selected.values())).shape if selected else None
    for key, value in selected.items():
        if value.shape != first_shape:
            raise ValueError(f"{item_id}/{scenario}: inconsistent hidden-state shape for {key}: {value.shape} vs {first_shape}.")
    return selected


def _save_hidden_arrays(output_dir: Path, *, item_id: str, scenario: str, arrays: Dict[str, np.ndarray]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sanitize_key(item_id)}__{sanitize_key(scenario)}.npz"
    np.savez_compressed(path, **arrays)
    return path


def _load_vector(record: Dict[str, Any], layer: int) -> np.ndarray:
    arrays = np.load(record["hidden_state_path"])
    key = f"layer_{int(layer)}_final_token"
    if key not in arrays:
        raise ValueError(f"Missing {key} in {record['hidden_state_path']}")
    value = np.asarray(arrays[key], dtype=np.float32)
    validate_vector(value, item_id=str(record["item_id"]), scenario=str(record["scenario"]), layer=int(layer))
    return value


def _extract_hidden_states(
    *,
    runner: LocalProbeRunner,
    pairs: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    output_dir: Path,
    logger: Any,
    flush_every: int,
) -> List[Dict[str, Any]]:
    hidden_dir = output_dir / "hidden_state_arrays"
    records: List[Dict[str, Any]] = []
    total = len(pairs) * len(FIXED_SCENARIOS)
    seen = 0
    for item in pairs:
        for scenario in FIXED_SCENARIOS:
            seen += 1
            prompt = _scenario_prompt(item, scenario)
            result = runner.analyze_prompt_selected_layers(
                prompt=prompt,
                sample_id=str(item["item_id"]),
                scenario=scenario,
                question_text=prompt,
                prompt_prefix="",
                ground_truth=str(item["answer_matching_behavior"]),
                wrong_option=str(item["answer_not_matching_behavior"]),
                selected_layers=layers,
            )
            selected_arrays = _arrays_for_layers(
                result.get("_hidden_state_arrays", {}) or {},
                item_id=str(item["item_id"]),
                scenario=scenario,
                layers=layers,
            )
            hidden_path = _save_hidden_arrays(hidden_dir, item_id=str(item["item_id"]), scenario=scenario, arrays=selected_arrays)
            record = {
                "item_id": item["item_id"],
                "split": item["split"],
                "source": item["source"],
                "pressure_type": item["pressure_type"],
                "scenario": scenario,
                "answer_matching_behavior": item["answer_matching_behavior"],
                "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                "parsed_answer": result.get("predicted_answer", ""),
                "answer_logits": result.get("answer_logits", {}),
                "correct_wrong_margin": result.get("correct_wrong_margin"),
                "hidden_state_path": str(hidden_path.resolve()),
                "layers": list(layers),
                "hidden_state_shape": list(next(iter(selected_arrays.values())).shape),
                "nonfinite_detected": False,
            }
            records.append(record)
            if seen == 1 or seen % max(int(flush_every), 1) == 0 or seen == total:
                logger.info("Hidden-state progress %d/%d", seen, total)
                write_jsonl(output_dir / "hidden_state_records.partial.jsonl", records)
    write_jsonl(output_dir / "hidden_state_records.jsonl", records)
    return records


def _records_by_key(records: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    return {(str(row["item_id"]), str(row["scenario"])): dict(row) for row in records}


def _train_items_from_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline = [row for row in records if row["split"] == "train" and row["scenario"] == "baseline"]
    return [dict(row) for row in baseline]


def _subspace_filter(row: Dict[str, Any], *, pressure_type: str, source: str) -> bool:
    if pressure_type and row.get("pressure_type") != pressure_type:
        return False
    if source and row.get("source") != source:
        return False
    return True


def _estimate_subspaces(
    *,
    records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    k_values: Sequence[int],
    output_dir: Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = _records_by_key(records)
    train_baselines = _train_items_from_records(records)
    max_k = max(int(k) for k in k_values)
    artifact: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for spec in default_subspace_specs():
        spec_items = [row for row in train_baselines if _subspace_filter(row, pressure_type=spec.pressure_type, source=spec.source)]
        if len(spec_items) < 2:
            continue
        artifact[spec.name] = {"layers": {}, "num_train_pairs": len(spec_items), "pressure_type": spec.pressure_type, "source": spec.source}
        for layer in layers:
            baselines = []
            deltas = []
            for row in spec_items:
                item_id = str(row["item_id"])
                baseline_vec = _load_vector(by_key[(item_id, "baseline")], int(layer))
                pressured_vec = _load_vector(by_key[(item_id, "pressured")], int(layer))
                baselines.append(baseline_vec)
                deltas.append(pressured_vec - baseline_vec)
            baseline_matrix = np.stack(baselines).astype(np.float32)
            delta_matrix = np.stack(deltas).astype(np.float32)
            components, explained = estimate_pca_components(delta_matrix, max_k=max_k)
            mean_baseline = baseline_matrix.mean(axis=0).astype(np.float32)
            mean_delta = delta_matrix.mean(axis=0).astype(np.float32)
            artifact[spec.name]["layers"][int(layer)] = {
                "mean_baseline": mean_baseline,
                "mean_delta": mean_delta,
                "components": components,
                "explained_variance": explained,
            }
            for k in k_values:
                kk = min(int(k), components.shape[0])
                top_component = components[0] if components.shape[0] else np.zeros_like(mean_delta)
                direction_cosines = [cosine_similarity(delta, top_component) for delta in delta_matrix]
                summary_rows.append(
                    {
                        "subspace_name": spec.name,
                        "pressure_type": spec.pressure_type or "mixed",
                        "source": spec.source or "mixed",
                        "layer": int(layer),
                        "k": kk,
                        "num_train_pairs": len(spec_items),
                        "direction_cosine_mean": float(np.mean(direction_cosines)),
                        "direction_cosine_abs_mean": float(np.mean(np.abs(direction_cosines))),
                        "explained_variance_sum": float(np.sum(explained[:kk])),
                        "top1_explained_variance": float(explained[0]) if len(explained) else 0.0,
                        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
                        "mean_baseline_norm": float(np.linalg.norm(mean_baseline)),
                    }
                )

    overlap_rows = _subspace_overlap_rows(artifact, layers=layers, k_values=k_values)
    _save_subspace_artifact(output_dir / "pressure_subspace_artifact.npz", artifact)
    write_csv(output_dir / "pressure_subspace_summary.csv", summary_rows)
    write_csv(output_dir / "pressure_subspace_overlap.csv", overlap_rows)
    return artifact, summary_rows, overlap_rows


def _subspace_overlap_rows(artifact: Dict[str, Any], *, layers: Sequence[int], k_values: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    names = sorted(artifact)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            for layer in layers:
                left_layer = artifact[left_name]["layers"].get(int(layer))
                right_layer = artifact[right_name]["layers"].get(int(layer))
                if not left_layer or not right_layer:
                    continue
                left_components = left_layer["components"]
                right_components = right_layer["components"]
                for k in k_values:
                    kk = min(int(k), left_components.shape[0], right_components.shape[0])
                    if kk <= 0:
                        continue
                    overlap = float(np.linalg.norm(left_components[:kk] @ right_components[:kk].T, ord="fro") ** 2 / kk)
                    rows.append(
                        {
                            "left_subspace": left_name,
                            "right_subspace": right_name,
                            "layer": int(layer),
                            "k": kk,
                            "subspace_overlap": overlap,
                        }
                    )
    return rows


def _save_subspace_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}
    for name, spec_payload in artifact.items():
        metadata[name] = {
            key: value for key, value in spec_payload.items()
            if key != "layers"
        }
        for layer, payload in spec_payload["layers"].items():
            prefix = f"{sanitize_key(name)}__layer_{int(layer)}"
            arrays[f"{prefix}__mean_baseline"] = payload["mean_baseline"]
            arrays[f"{prefix}__mean_delta"] = payload["mean_delta"]
            arrays[f"{prefix}__components"] = payload["components"]
            arrays[f"{prefix}__explained_variance"] = payload["explained_variance"]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _target_records_for_transfer(eval_items: Sequence[Dict[str, Any]], transfer: Dict[str, str]) -> List[Dict[str, Any]]:
    target_pressure = str(transfer.get("target_pressure_type") or "")
    target_source = str(transfer.get("target_source") or "")
    rows = []
    for row in eval_items:
        if target_pressure and row.get("pressure_type") != target_pressure:
            continue
        if target_source and row.get("source") != target_source:
            continue
        rows.append(dict(row))
    return rows


def _damped_patch_map(
    *,
    artifact: Dict[str, Any],
    subspace_name: str,
    scenario_record: Dict[str, Any],
    layers: Sequence[int],
    alpha: float,
    k: int,
) -> Dict[int, np.ndarray]:
    patch_map: Dict[int, np.ndarray] = {}
    for layer in layers:
        layer_payload = artifact[subspace_name]["layers"][int(layer)]
        hidden_state = _load_vector(scenario_record, int(layer))
        patch_map[int(layer)] = damp_hidden_state(
            hidden_state=hidden_state,
            mean_baseline=layer_payload["mean_baseline"],
            components=layer_payload["components"],
            alpha=float(alpha),
            k=int(k),
        )
    return patch_map


def _run_interventions(
    *,
    runner: LocalProbeRunner,
    pairs_eval: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    artifact: Dict[str, Any],
    layers: Sequence[int],
    k_values: Sequence[int],
    alpha_values: Sequence[float],
    output_dir: Path,
    logger: Any,
    flush_every: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = _records_by_key(records)
    eval_by_id = {str(item["item_id"]): dict(item) for item in pairs_eval}
    intervention_records: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    settings = []
    for transfer in default_transfer_tests():
        if transfer["subspace_name"] not in artifact:
            continue
        target_items = _target_records_for_transfer(pairs_eval, transfer)
        for k in k_values:
            for alpha in alpha_values:
                settings.append((transfer, int(k), float(alpha), target_items))

    total = sum(len(target_items) for _, _, _, target_items in settings)
    done = 0
    for transfer, k, alpha, target_items in settings:
        subspace_name = str(transfer["subspace_name"])
        for item in target_items:
            done += 1
            item_id = str(item["item_id"])
            outputs: Dict[str, Dict[str, Any]] = {}
            for scenario in FIXED_SCENARIOS:
                ref_record = by_key[(item_id, scenario)]
                patch_map = _damped_patch_map(
                    artifact=artifact,
                    subspace_name=subspace_name,
                    scenario_record=ref_record,
                    layers=layers,
                    alpha=alpha,
                    k=k,
                )
                outputs[scenario] = runner.patch_final_token_residuals_multi(
                    prompt=_scenario_prompt(eval_by_id[item_id], scenario),
                    layer_patch_map=patch_map,
                    ground_truth=str(item["answer_matching_behavior"]),
                    wrong_option=str(item["answer_not_matching_behavior"]),
                )
                intervention_records.append(
                    {
                        "item_id": item_id,
                        "source": item["source"],
                        "pressure_type": item["pressure_type"],
                        "scenario": scenario,
                        "transfer_name": transfer["transfer_name"],
                        "subspace_name": subspace_name,
                        "intervention_method": "pressure_subspace_damping",
                        "layers": list(layers),
                        "k": int(k),
                        "alpha": float(alpha),
                        "baseline_choice_reference": by_key[(item_id, "baseline")].get("parsed_answer", ""),
                        "parsed_answer": outputs[scenario].get("predicted_answer", ""),
                        "answer_logits": outputs[scenario].get("answer_logits", {}),
                        "correct_wrong_margin": outputs[scenario].get("correct_wrong_margin"),
                    }
                )

            baseline_ref = by_key[(item_id, "baseline")]
            pressured_ref = by_key[(item_id, "pressured")]
            recovery_ref = by_key[(item_id, "recovery")]
            comparisons.append(
                {
                    "item_id": item_id,
                    "source": item["source"],
                    "pressure_type": item["pressure_type"],
                    "transfer_name": transfer["transfer_name"],
                    "subspace_name": subspace_name,
                    "intervention_method": "pressure_subspace_damping",
                    "layers": list(layers),
                    "k": int(k),
                    "alpha": float(alpha),
                    "answer_matching_behavior": item["answer_matching_behavior"],
                    "answer_not_matching_behavior": item["answer_not_matching_behavior"],
                    "baseline_choice": baseline_ref.get("parsed_answer", ""),
                    "pressured_choice_baseline": pressured_ref.get("parsed_answer", ""),
                    "recovery_choice_baseline": recovery_ref.get("parsed_answer", ""),
                    "baseline_choice_intervention": outputs["baseline"].get("predicted_answer", ""),
                    "pressured_choice_intervention": outputs["pressured"].get("predicted_answer", ""),
                    "recovery_choice_intervention": outputs["recovery"].get("predicted_answer", ""),
                    "stance_drift_baseline": baseline_ref.get("parsed_answer", "") != pressured_ref.get("parsed_answer", ""),
                    "stance_drift_intervention": outputs["baseline"].get("predicted_answer", "") != outputs["pressured"].get("predicted_answer", ""),
                    "pressured_compliance_baseline": pressured_ref.get("parsed_answer", "") == item["answer_matching_behavior"],
                    "pressured_compliance_intervention": outputs["pressured"].get("predicted_answer", "") == item["answer_matching_behavior"],
                    "recovered_baseline": recovery_ref.get("parsed_answer", "") == baseline_ref.get("parsed_answer", ""),
                    "recovered_intervention": outputs["recovery"].get("predicted_answer", "") == outputs["baseline"].get("predicted_answer", ""),
                    "baseline_damage": outputs["baseline"].get("predicted_answer", "") != baseline_ref.get("parsed_answer", ""),
                }
            )
            if done == 1 or done % max(int(flush_every), 1) == 0 or done == total:
                logger.info("Intervention progress %d/%d", done, total)
                write_jsonl(output_dir / "subspace_intervention_records.partial.jsonl", intervention_records)
                write_jsonl(output_dir / "subspace_intervention_comparisons.partial.jsonl", comparisons)

    summary_rows = _summarize_comparisons(comparisons)
    write_jsonl(output_dir / "subspace_intervention_records.jsonl", intervention_records)
    write_jsonl(output_dir / "subspace_intervention_comparisons.jsonl", comparisons)
    write_csv(output_dir / "subspace_intervention_summary.csv", summary_rows)
    return intervention_records, comparisons, summary_rows


def _summarize_comparisons(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted({(row["transfer_name"], row["subspace_name"], int(row["k"]), float(row["alpha"])) for row in comparisons})
    for transfer_name, subspace_name, k, alpha in keys:
        subset = [
            row for row in comparisons
            if row["transfer_name"] == transfer_name and row["subspace_name"] == subspace_name and int(row["k"]) == k and float(row["alpha"]) == alpha
        ]
        groups = [("all", "", "", subset)]
        for source in sorted({row["source"] for row in subset}):
            groups.append(("source", source, "", [row for row in subset if row["source"] == source]))
        for pressure_type in sorted({row["pressure_type"] for row in subset}):
            groups.append(("pressure_type", "", pressure_type, [row for row in subset if row["pressure_type"] == pressure_type]))
        for aggregation_level, source, pressure_type, group_rows in groups:
            rows.append(
                {
                    "transfer_name": transfer_name,
                    "subspace_name": subspace_name,
                    "intervention_method": "pressure_subspace_damping",
                    "k": int(k),
                    "alpha": float(alpha),
                    "aggregation_level": aggregation_level,
                    "source": source,
                    "pressure_type": pressure_type,
                    "num_items": len(group_rows),
                    "stance_drift_rate_baseline": mean_bool(row["stance_drift_baseline"] for row in group_rows),
                    "stance_drift_rate_intervention": mean_bool(row["stance_drift_intervention"] for row in group_rows),
                    "pressured_compliance_rate_baseline": mean_bool(row["pressured_compliance_baseline"] for row in group_rows),
                    "pressured_compliance_rate_intervention": mean_bool(row["pressured_compliance_intervention"] for row in group_rows),
                    "recovery_rate_baseline": mean_bool(row["recovered_baseline"] for row in group_rows),
                    "recovery_rate_intervention": mean_bool(row["recovered_intervention"] for row in group_rows),
                    "baseline_damage_rate": mean_bool(row["baseline_damage"] for row in group_rows),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal pressure_subspace_damping pipeline.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", required=True, help="Layer window, e.g. 24-26 or 31-35.")
    parser.add_argument("--output-root", default="outputs/experiments/pressure_subspace_damping")
    parser.add_argument("--items-per-stratum", type=int, default=48)
    parser.add_argument("--train-per-stratum", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--k-values", default="1,2,4,8")
    parser.add_argument("--intervention-k-values", default="1,2,4")
    parser.add_argument("--alpha-values", default="0.25,0.5,0.75")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="pressure_subspace_damping", level=args.log_level)
    layers = parse_layers(str(args.layers))
    k_values = tuple(int(value.strip()) for value in str(args.k_values).split(",") if value.strip())
    intervention_k_values = tuple(int(value.strip()) for value in str(args.intervention_k_values).split(",") if value.strip())
    alpha_values = tuple(float(value.strip()) for value in str(args.alpha_values).split(",") if value.strip())
    run_name = str(args.model_name).replace("/", "_")
    output_dir = prepare_output_dir(Path(args.output_root), run_name=run_name)

    all_items = [item.to_dict() for item in build_bridge_dataset(_source_paths())]
    train_pairs, eval_pairs = _sample_pairs(
        items=all_items,
        items_per_stratum=int(args.items_per_stratum),
        train_per_stratum=int(args.train_per_stratum),
        seed=int(args.seed),
    )
    write_jsonl(output_dir / "pressure_pairs_train.jsonl", train_pairs)
    write_jsonl(output_dir / "pressure_pairs_eval.jsonl", eval_pairs)

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

    logger.info(
        "Starting pressure_subspace_damping: model=%s train=%d eval=%d layers=%s output=%s",
        args.model_name,
        len(train_pairs),
        len(eval_pairs),
        layers,
        output_dir,
    )
    hidden_records = _extract_hidden_states(
        runner=runner,
        pairs=[*train_pairs, *eval_pairs],
        layers=layers,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )
    artifact, subspace_summary, overlap_rows = _estimate_subspaces(
        records=hidden_records,
        layers=layers,
        k_values=k_values,
        output_dir=output_dir,
    )
    _, comparisons, intervention_summary = _run_interventions(
        runner=runner,
        pairs_eval=eval_pairs,
        records=hidden_records,
        artifact=artifact,
        layers=layers,
        k_values=intervention_k_values,
        alpha_values=alpha_values,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )

    manifest = {
        "pipeline": "pressure_subspace_damping_v1",
        "model_name": str(args.model_name),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "layers": list(layers),
        "items_per_stratum": int(args.items_per_stratum),
        "train_per_stratum": int(args.train_per_stratum),
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "train_strata": dict(Counter(f"{row['source']}::{row['pressure_type']}" for row in train_pairs)),
        "eval_strata": dict(Counter(f"{row['source']}::{row['pressure_type']}" for row in eval_pairs)),
        "k_values": list(k_values),
        "intervention_k_values": list(intervention_k_values),
        "alpha_values": list(alpha_values),
        "transfer_tests": default_transfer_tests(),
        "outputs": {
            "pressure_pairs_train": str((output_dir / "pressure_pairs_train.jsonl").resolve()),
            "pressure_pairs_eval": str((output_dir / "pressure_pairs_eval.jsonl").resolve()),
            "hidden_state_records": str((output_dir / "hidden_state_records.jsonl").resolve()),
            "hidden_state_arrays": str((output_dir / "hidden_state_arrays").resolve()),
            "pressure_subspace_artifact": str((output_dir / "pressure_subspace_artifact.npz").resolve()),
            "pressure_subspace_summary": str((output_dir / "pressure_subspace_summary.csv").resolve()),
            "pressure_subspace_overlap": str((output_dir / "pressure_subspace_overlap.csv").resolve()),
            "subspace_intervention_records": str((output_dir / "subspace_intervention_records.jsonl").resolve()),
            "subspace_intervention_comparisons": str((output_dir / "subspace_intervention_comparisons.jsonl").resolve()),
            "subspace_intervention_summary": str((output_dir / "subspace_intervention_summary.csv").resolve()),
        },
        "num_hidden_records": len(hidden_records),
        "num_intervention_comparisons": len(comparisons),
        "num_subspace_summary_rows": len(subspace_summary),
        "num_subspace_overlap_rows": len(overlap_rows),
        "num_intervention_summary_rows": len(intervention_summary),
    }
    save_json(output_dir / "pressure_subspace_run.json", manifest)
    print(json.dumps({"output_dir": str(output_dir.resolve()), **{k: manifest[k] for k in ("num_train_pairs", "num_eval_pairs", "num_intervention_comparisons")}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
