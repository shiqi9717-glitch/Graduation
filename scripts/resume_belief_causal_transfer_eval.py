#!/usr/bin/env python3
"""Resume/finalize a partially completed belief causal transfer run.

This tool is intentionally minimal: it reuses existing hidden states,
subspace artifacts, and partial evaluation outputs inside an existing run
directory, then only evaluates missing method-item pairs and writes the final
artifacts expected by `scripts/run_belief_causal_transfer.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.io_utils import save_json
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.pressure_subspace import read_jsonl, write_csv, write_jsonl


def _load_belief_module():
    path = project_root / "scripts" / "run_belief_causal_transfer.py"
    spec = importlib.util.spec_from_file_location("_belief_causal_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


belief = _load_belief_module()


def _load_artifact(path: Path) -> Dict[str, Any]:
    arrays = np.load(path)
    artifact: Dict[str, Any] = {}
    for key in arrays.files:
        prefix, kind = key.rsplit("__", 1)
        subspace_name, layer_text = prefix.rsplit("__layer_", 1)
        layer = int(layer_text)
        artifact.setdefault(subspace_name, {"layers": {}})
        artifact[subspace_name]["layers"].setdefault(layer, {})
        artifact[subspace_name]["layers"][layer][kind] = np.asarray(arrays[key], dtype=np.float32)
    return artifact


def _normalize_records(records: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in records:
        item = dict(row)
        item["alpha"] = float(alpha)
        out.append(item)
    return out


def _normalize_comparisons(comparisons: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in comparisons:
        item = dict(row)
        item["alpha"] = float(alpha)
        out.append(item)
    return out


def _scenario_prompt(item: Dict[str, Any], scenario: str) -> str:
    return str(item[f"{scenario}_prompt"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume/finalize a partial belief causal transfer run.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing hidden states and partial files.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="resume_belief_causal_transfer_eval", level=args.log_level)

    run_dir = Path(args.run_dir).resolve()
    output_prefix = f"belief_causal_alpha_{str(float(args.alpha)).replace('.', 'p')}"
    records_partial_path = run_dir / f"{output_prefix}_records.partial.jsonl"
    comparisons_partial_path = run_dir / f"{output_prefix}_comparisons.partial.jsonl"
    hidden_records_path = run_dir / "hidden_state_records.jsonl"
    artifact_path = run_dir / "belief_causal_subspace_artifact.npz"
    eval_pairs_path = run_dir / "pressure_pairs_eval.jsonl"
    train_pairs_path = run_dir / "pressure_pairs_train.jsonl"

    for path in (hidden_records_path, artifact_path, eval_pairs_path, train_pairs_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    hidden_records = read_jsonl(hidden_records_path)
    eval_pairs = read_jsonl(eval_pairs_path)
    train_pairs = read_jsonl(train_pairs_path)
    artifact = _load_artifact(artifact_path)

    layers = tuple(int(x) for x in (hidden_records[0].get("layers") or []))
    if not layers:
        raise ValueError("Could not infer layers from hidden_state_records.jsonl")

    existing_records = read_jsonl(records_partial_path) if records_partial_path.exists() else []
    existing_comparisons = read_jsonl(comparisons_partial_path) if comparisons_partial_path.exists() else []
    completed = {(str(row["item_id"]), str(row["method"])) for row in existing_comparisons}

    logger.info(
        "Resume setup: eval_pairs=%d existing_comparisons=%d existing_records=%d pending=%d run_dir=%s",
        len(eval_pairs),
        len(existing_comparisons),
        len(existing_records),
        len(eval_pairs) * 3 - len(existing_comparisons),
        run_dir,
    )

    if len(existing_records) != len(existing_comparisons) * 3:
        logger.warning(
            "Partial file count mismatch: records=%d comparisons=%d; expected records=comparisons*3",
            len(existing_records),
            len(existing_comparisons),
        )

    runner = None
    if len(completed) < len(eval_pairs) * 3:
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

    by_key = belief.base._records_by_key(hidden_records)
    eval_by_id = {str(item["item_id"]): dict(item) for item in eval_pairs}
    methods = ("no_intervention", "matched_belief_subspace_damping", "matched_negative_control")

    intervention_records: List[Dict[str, Any]] = list(existing_records)
    comparisons: List[Dict[str, Any]] = list(existing_comparisons)

    newly_done = 0
    total = len(eval_pairs) * len(methods)
    for method in methods:
        subspace_name = "matched_belief_subspace" if method == "matched_belief_subspace_damping" else "matched_negative_control"
        for item in eval_pairs:
            item_id = str(item["item_id"])
            key = (item_id, method)
            if key in completed:
                continue

            if runner is None:
                raise RuntimeError("Runner was not initialized but there are pending eval items.")

            outputs: Dict[str, Dict[str, Any]] = {}
            for scenario in ("baseline", "pressured", "recovery"):
                ref = by_key[(item_id, scenario)]
                if method == "no_intervention":
                    outputs[scenario] = {
                        "predicted_answer": ref.get("parsed_answer", ""),
                        "answer_logits": ref.get("answer_logits", {}),
                        "correct_wrong_margin": ref.get("correct_wrong_margin"),
                    }
                else:
                    patch = belief.base._patch_map(
                        artifact=artifact,
                        subspace_name=subspace_name,
                        record=ref,
                        layers=layers,
                        alpha=float(args.alpha),
                        k=int(args.k),
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
                        "k": int(args.k),
                        "alpha": float(args.alpha),
                        "parsed_answer": outputs[scenario].get("predicted_answer", ""),
                        "answer_logits": outputs[scenario].get("answer_logits", {}),
                        "correct_wrong_margin": outputs[scenario].get("correct_wrong_margin"),
                    }
                )

            row = belief.base._comparison_row(item=item, method=method, outputs=outputs)
            row["alpha"] = float(args.alpha)
            comparisons.append(row)
            completed.add(key)
            newly_done += 1

            if newly_done == 1 or newly_done % max(int(args.flush_every), 1) == 0:
                logger.info("Resume evaluation progress +%d pending_done, total_now=%d/%d", newly_done, len(comparisons), total)
                write_jsonl(records_partial_path, intervention_records)
                write_jsonl(comparisons_partial_path, comparisons)

    write_jsonl(records_partial_path, intervention_records)
    write_jsonl(comparisons_partial_path, comparisons)

    normalized_records = _normalize_records(intervention_records, float(args.alpha))
    normalized_comparisons = belief.base._add_baseline_damage(_normalize_comparisons(comparisons, float(args.alpha)))

    summary = belief._summary_rows(normalized_comparisons)
    write_jsonl(run_dir / "belief_causal_records.jsonl", normalized_records)
    write_jsonl(run_dir / "belief_causal_comparisons.jsonl", normalized_comparisons)
    write_csv(run_dir / "belief_causal_summary.csv", summary)

    projection_summary = None
    if eval_pairs:
        projection_summary, _ = belief._projection_alignment_outputs(
            eval_pairs=eval_pairs,
            records=hidden_records,
            artifact=artifact,
            layers=layers,
            k=int(args.k),
            alpha=float(args.alpha),
            comparisons=normalized_comparisons,
            intervention_records=normalized_records,
            output_dir=run_dir,
        )

    manifest = {
        "pipeline": "belief_causal_transfer_v1_resumed",
        "model_name": str(args.model_name),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "layers": list(layers),
        "train_source": str(train_pairs[0].get("source", "")) if train_pairs else "",
        "eval_source": str(eval_pairs[0].get("source", "")) if eval_pairs else "",
        "pressure_type": str(eval_pairs[0].get("pressure_type", "")) if eval_pairs else "",
        "prompt_variant": str(eval_pairs[0].get("prompt_variant", "original")) if eval_pairs else "original",
        "train_n": len(train_pairs),
        "eval_n": len(eval_pairs),
        "k": int(args.k),
        "alpha": float(args.alpha),
        "alpha_values": [float(args.alpha)],
        "methods": ["no_intervention", "matched_belief_subspace_damping", "matched_negative_control"],
        "outputs": {
            "summary_csv": str((run_dir / "belief_causal_summary.csv").resolve()),
            "comparisons_jsonl": str((run_dir / "belief_causal_comparisons.jsonl").resolve()),
            "records_jsonl": str((run_dir / "belief_causal_records.jsonl").resolve()),
            "subspace_summary_csv": str((run_dir / "belief_causal_subspace_summary.csv").resolve()),
            "hidden_state_records": str(hidden_records_path.resolve()),
        },
        "resume_info": {
            "records_partial": str(records_partial_path.resolve()),
            "comparisons_partial": str(comparisons_partial_path.resolve()),
            "existing_comparisons_at_start": len(existing_comparisons),
            "newly_completed_comparisons": newly_done,
        },
    }
    if projection_summary is not None:
        manifest["outputs"]["projection_alignment_summary_json"] = str((run_dir / "projection_alignment_summary.json").resolve())
        manifest["outputs"]["projection_alignment_diagnostic_csv"] = str((run_dir / "projection_alignment_diagnostic.csv").resolve())
        manifest["projection_alignment_summary"] = projection_summary

    save_json(run_dir / "belief_causal_run.json", manifest)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "total_expected_comparisons": total,
                "final_comparisons": len(normalized_comparisons),
                "newly_completed_comparisons": newly_done,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
