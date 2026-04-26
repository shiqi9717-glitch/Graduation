#!/usr/bin/env python3
"""Minimal 14B belief_argument causal transfer validation."""

from __future__ import annotations

import argparse
import json
import random
import sys
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
    damp_hidden_state,
    estimate_pca_components,
    mean_bool,
    parse_layers,
    sanitize_key,
    validate_vector,
    write_csv,
    write_jsonl,
)


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _scenario_prompt(item: Dict[str, Any], scenario: str) -> str:
    return str(item[f"{scenario}_prompt"])


def _sample_source(
    *,
    items: Sequence[Dict[str, Any]],
    source: str,
    n: int,
    split: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    candidates = [
        dict(item)
        for item in items
        if item.get("source") == source and item.get("pressure_type") == "belief_argument"
    ]
    rng.shuffle(candidates)
    selected = candidates[: int(n)]
    for row in selected:
        row["split"] = split
    return selected


def _save_hidden_arrays(output_dir: Path, *, item_id: str, scenario: str, arrays: Dict[str, np.ndarray]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sanitize_key(item_id)}__{sanitize_key(scenario)}.npz"
    np.savez_compressed(path, **arrays)
    return path


def _selected_arrays(
    arrays: Dict[str, np.ndarray],
    *,
    item_id: str,
    scenario: str,
    layers: Sequence[int],
) -> Dict[str, np.ndarray]:
    selected: Dict[str, np.ndarray] = {}
    for layer in layers:
        key = f"layer_{int(layer)}_final_token"
        if key not in arrays:
            raise ValueError(f"{item_id}/{scenario}: missing hidden-state key {key}.")
        value = np.asarray(arrays[key], dtype=np.float32)
        validate_vector(value, item_id=item_id, scenario=scenario, layer=int(layer))
        selected[key] = value
    return selected


def _extract_hidden_records(
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
            arrays = _selected_arrays(
                result.get("_hidden_state_arrays", {}) or {},
                item_id=str(item["item_id"]),
                scenario=scenario,
                layers=layers,
            )
            hidden_path = _save_hidden_arrays(hidden_dir, item_id=str(item["item_id"]), scenario=scenario, arrays=arrays)
            records.append(
                {
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
                    "hidden_state_shape": list(next(iter(arrays.values())).shape),
                    "nonfinite_detected": False,
                }
            )
            if seen == 1 or seen % max(int(flush_every), 1) == 0 or seen == total:
                logger.info("Hidden-state progress %d/%d", seen, total)
                write_jsonl(output_dir / "hidden_state_records.partial.jsonl", records)
    write_jsonl(output_dir / "hidden_state_records.jsonl", records)
    return records


def _records_by_key(records: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    return {(str(row["item_id"]), str(row["scenario"])): dict(row) for row in records}


def _load_vector(record: Dict[str, Any], layer: int) -> np.ndarray:
    arrays = np.load(record["hidden_state_path"])
    key = f"layer_{int(layer)}_final_token"
    value = np.asarray(arrays[key], dtype=np.float32)
    validate_vector(value, item_id=str(record["item_id"]), scenario=str(record["scenario"]), layer=int(layer))
    return value


def _estimate_belief_subspace(
    *,
    train_pairs: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    k: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    by_key = _records_by_key(records)
    rng = np.random.default_rng(int(seed))
    artifact: Dict[str, Any] = {"matched_belief_subspace": {"layers": {}}, "matched_negative_control": {"layers": {}}}
    summary_rows: List[Dict[str, Any]] = []
    for layer in layers:
        baseline_vectors = []
        delta_vectors = []
        for item in train_pairs:
            item_id = str(item["item_id"])
            baseline = _load_vector(by_key[(item_id, "baseline")], int(layer))
            pressured = _load_vector(by_key[(item_id, "pressured")], int(layer))
            baseline_vectors.append(baseline)
            delta_vectors.append(pressured - baseline)
        baseline_matrix = np.stack(baseline_vectors).astype(np.float32)
        delta_matrix = np.stack(delta_vectors).astype(np.float32)
        components, explained = estimate_pca_components(delta_matrix, max_k=int(k))
        mean_baseline = baseline_matrix.mean(axis=0).astype(np.float32)
        random_matrix = rng.normal(size=(mean_baseline.shape[0], int(k))).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        negative_components = q.T.astype(np.float32)
        artifact["matched_belief_subspace"]["layers"][int(layer)] = {
            "mean_baseline": mean_baseline,
            "components": components[: int(k)],
        }
        artifact["matched_negative_control"]["layers"][int(layer)] = {
            "mean_baseline": mean_baseline,
            "components": negative_components[: int(k)],
        }
        summary_rows.append(
            {
                "subspace_name": "matched_belief_subspace",
                "source": "philpapers2020",
                "pressure_type": "belief_argument",
                "layer": int(layer),
                "k": int(k),
                "num_train_pairs": len(train_pairs),
                "explained_variance_sum": float(np.sum(explained[: int(k)])),
                "top1_explained_variance": float(explained[0]) if len(explained) else 0.0,
                "mean_delta_norm": float(np.linalg.norm(delta_matrix.mean(axis=0))),
            }
        )
    _save_artifact(output_dir / "qwen14b_belief_subspace_artifact.npz", artifact)
    write_csv(output_dir / "qwen14b_belief_subspace_summary.csv", summary_rows)
    return artifact


def _save_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for subspace_name, payload in artifact.items():
        for layer, layer_payload in payload["layers"].items():
            prefix = f"{sanitize_key(subspace_name)}__layer_{int(layer)}"
            arrays[f"{prefix}__mean_baseline"] = layer_payload["mean_baseline"]
            arrays[f"{prefix}__components"] = layer_payload["components"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _patch_map(
    *,
    artifact: Dict[str, Any],
    subspace_name: str,
    record: Dict[str, Any],
    layers: Sequence[int],
    alpha: float,
    k: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for layer in layers:
        layer_payload = artifact[subspace_name]["layers"][int(layer)]
        hidden_state = _load_vector(record, int(layer))
        out[int(layer)] = damp_hidden_state(
            hidden_state=hidden_state,
            mean_baseline=layer_payload["mean_baseline"],
            components=layer_payload["components"],
            alpha=float(alpha),
            k=int(k),
        )
    return out


def _run_eval(
    *,
    runner: LocalProbeRunner,
    eval_pairs: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    artifact: Dict[str, Any],
    layers: Sequence[int],
    k: int,
    alpha: float,
    output_dir: Path,
    logger: Any,
    flush_every: int,
    output_prefix: str = "qwen14b_belief_causal",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = _records_by_key(records)
    eval_by_id = {str(item["item_id"]): dict(item) for item in eval_pairs}
    methods = ("no_intervention", "matched_belief_subspace_damping", "matched_negative_control")
    comparisons: List[Dict[str, Any]] = []
    intervention_records: List[Dict[str, Any]] = []
    total = len(eval_pairs) * len(methods)
    done = 0
    for method in methods:
        subspace_name = "matched_belief_subspace" if method == "matched_belief_subspace_damping" else "matched_negative_control"
        for item in eval_pairs:
            done += 1
            item_id = str(item["item_id"])
            outputs: Dict[str, Dict[str, Any]] = {}
            for scenario in FIXED_SCENARIOS:
                ref = by_key[(item_id, scenario)]
                if method == "no_intervention":
                    outputs[scenario] = {
                        "predicted_answer": ref.get("parsed_answer", ""),
                        "answer_logits": ref.get("answer_logits", {}),
                        "correct_wrong_margin": ref.get("correct_wrong_margin"),
                    }
                else:
                    patch = _patch_map(
                        artifact=artifact,
                        subspace_name=subspace_name,
                        record=ref,
                        layers=layers,
                        alpha=float(alpha),
                        k=int(k),
                    )
                    outputs[scenario] = runner.patch_final_token_residuals_multi(
                        prompt=_scenario_prompt(eval_by_id[item_id], scenario),
                        layer_patch_map=patch,
                        ground_truth=str(item["answer_matching_behavior"]),
                        wrong_option=str(item["answer_not_matching_behavior"]),
                    )
                intervention_records.append(
                    {
                        "item_id": item_id,
                        "method": method,
                        "scenario": scenario,
                        "source": item["source"],
                        "pressure_type": item["pressure_type"],
                        "layers": list(layers),
                        "k": int(k),
                        "alpha": float(alpha) if method != "no_intervention" else "",
                        "parsed_answer": outputs[scenario].get("predicted_answer", ""),
                        "answer_logits": outputs[scenario].get("answer_logits", {}),
                        "correct_wrong_margin": outputs[scenario].get("correct_wrong_margin"),
                    }
                )
            comparisons.append(_comparison_row(item=item, method=method, outputs=outputs))
            if done == 1 or done % max(int(flush_every), 1) == 0 or done == total:
                logger.info("Evaluation progress %d/%d", done, total)
                write_jsonl(output_dir / f"{output_prefix}_records.partial.jsonl", intervention_records)
                write_jsonl(output_dir / f"{output_prefix}_comparisons.partial.jsonl", comparisons)
    write_jsonl(output_dir / f"{output_prefix}_records.jsonl", intervention_records)
    write_jsonl(output_dir / f"{output_prefix}_comparisons.jsonl", comparisons)
    return intervention_records, comparisons


def _comparison_row(*, item: Dict[str, Any], method: str, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    baseline = outputs["baseline"].get("predicted_answer", "")
    pressured = outputs["pressured"].get("predicted_answer", "")
    recovery = outputs["recovery"].get("predicted_answer", "")
    matching = item["answer_matching_behavior"]
    return {
        "item_id": item["item_id"],
        "source": item["source"],
        "pressure_type": item["pressure_type"],
        "method": method,
        "answer_matching_behavior": matching,
        "baseline_choice": baseline,
        "pressured_choice": pressured,
        "recovery_choice": recovery,
        "stance_drift": bool(baseline and pressured and baseline != pressured),
        "pressured_compliance": bool(pressured and pressured == matching),
        "recovered": bool(baseline and recovery and recovery == baseline),
        "baseline_damage": False,
    }


def _add_baseline_damage(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    no_by_id = {
        str(row["item_id"]): row
        for row in comparisons
        if row["method"] == "no_intervention"
    }
    rows: List[Dict[str, Any]] = []
    for row in comparisons:
        out = dict(row)
        ref = no_by_id.get(str(row["item_id"]), {})
        out["baseline_damage"] = bool(
            row["method"] != "no_intervention"
            and row.get("baseline_choice")
            and ref.get("baseline_choice")
            and row.get("baseline_choice") != ref.get("baseline_choice")
        )
        rows.append(out)
    return rows


def _summary_rows(comparisons: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in ("no_intervention", "matched_belief_subspace_damping", "matched_negative_control"):
        subset = [row for row in comparisons if row["method"] == method]
        rows.append(
            {
                "method": method,
                "num_items": len(subset),
                "stance_drift_rate": mean_bool(row["stance_drift"] for row in subset),
                "pressured_compliance_rate": mean_bool(row["pressured_compliance"] for row in subset),
                "recovery_rate": mean_bool(row["recovered"] for row in subset),
                "baseline_damage_rate": mean_bool(row["baseline_damage"] for row in subset),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal 14B belief_argument causal transfer validation.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layers", default="40-47")
    parser.add_argument("--train-n", type=int, default=24)
    parser.add_argument("--eval-n", type=int, default=24)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--output-root", default="outputs/experiments/qwen14b_belief_causal_transfer")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="qwen14b_belief_causal_transfer", level=args.log_level)
    layers = parse_layers(str(args.layers))
    model_name = str(args.model_name)
    run_name = Path(model_name).name if Path(model_name).exists() else model_name.replace("/", "_")
    output_dir = prepare_output_dir(Path(args.output_root), run_name=run_name)
    all_items = [item.to_dict() for item in build_bridge_dataset(_source_paths())]
    train_pairs = _sample_source(
        items=all_items,
        source="philpapers2020",
        n=int(args.train_n),
        split="train",
        seed=int(args.seed),
    )
    eval_pairs = _sample_source(
        items=all_items,
        source="nlp_survey",
        n=int(args.eval_n),
        split="eval",
        seed=int(args.seed) + 1,
    )
    write_jsonl(output_dir / "pressure_pairs_train.jsonl", train_pairs)
    write_jsonl(output_dir / "pressure_pairs_eval.jsonl", eval_pairs)

    logger.info(
        "Prepared 14B belief causal transfer run: train=%d eval=%d layers=%s output=%s",
        len(train_pairs),
        len(eval_pairs),
        layers,
        output_dir,
    )
    logger.info("Loading model %s on device=%s dtype=%s; 14B MPS load can be silent for several minutes.", model_name, args.device, args.dtype)
    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=model_name,
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=(-1,),
        )
    )
    logger.info(
        "Starting 14B belief causal transfer: train=%d eval=%d layers=%s k=%s alpha=%s output=%s",
        len(train_pairs),
        len(eval_pairs),
        layers,
        args.k,
        args.alpha,
        output_dir,
    )
    hidden_records = _extract_hidden_records(
        runner=runner,
        pairs=[*train_pairs, *eval_pairs],
        layers=layers,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )
    artifact = _estimate_belief_subspace(
        train_pairs=train_pairs,
        records=hidden_records,
        layers=layers,
        k=int(args.k),
        seed=int(args.seed),
        output_dir=output_dir,
    )
    _, comparisons_raw = _run_eval(
        runner=runner,
        eval_pairs=eval_pairs,
        records=hidden_records,
        artifact=artifact,
        layers=layers,
        k=int(args.k),
        alpha=float(args.alpha),
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )
    comparisons = _add_baseline_damage(comparisons_raw)
    summary = _summary_rows(comparisons)
    write_jsonl(output_dir / "qwen14b_belief_causal_comparisons.jsonl", comparisons)
    write_csv(output_dir / "qwen14b_belief_causal_summary.csv", summary)
    manifest = {
        "pipeline": "qwen14b_belief_causal_transfer_v1",
        "model_name": str(args.model_name),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "layers": list(layers),
        "train_source": "philpapers2020",
        "eval_source": "nlp_survey",
        "pressure_type": "belief_argument",
        "train_n": len(train_pairs),
        "eval_n": len(eval_pairs),
        "k": int(args.k),
        "alpha": float(args.alpha),
        "methods": ["no_intervention", "matched_belief_subspace_damping", "matched_negative_control"],
        "outputs": {
            "summary_csv": str((output_dir / "qwen14b_belief_causal_summary.csv").resolve()),
            "comparisons_jsonl": str((output_dir / "qwen14b_belief_causal_comparisons.jsonl").resolve()),
            "records_jsonl": str((output_dir / "qwen14b_belief_causal_records.jsonl").resolve()),
            "subspace_summary_csv": str((output_dir / "qwen14b_belief_subspace_summary.csv").resolve()),
            "hidden_state_records": str((output_dir / "hidden_state_records.jsonl").resolve()),
        },
    }
    save_json(output_dir / "qwen14b_belief_causal_run.json", manifest)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "train_n": len(train_pairs), "eval_n": len(eval_pairs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
