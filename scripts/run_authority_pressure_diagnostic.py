#!/usr/bin/env python3
"""Run authority-pressure diagnostic transfer using a frozen belief-argument subspace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run authority-pressure diagnostic-only evaluation with a frozen belief subspace.")
    parser.add_argument("--sample-file", default="data/authority_pressure/authority_pressure_pairs.jsonl")
    parser.add_argument("--artifact-run-dir", default="outputs/experiments/pressure_subspace_damping_qwen7b_n100/Qwen_Qwen2.5-7B-Instruct/20260506_181314")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layers", default="24-26")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--output-root", default="outputs/experiments/authority_pressure_diagnostic/qwen7b")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="authority_pressure_diagnostic", level=args.log_level)
    sample_rows = read_jsonl(Path(args.sample_file))
    layers = belief.parse_layers(str(args.layers))
    run_name = str(args.model_name).replace("/", "_")
    output_dir = prepare_output_dir(Path(args.output_root), run_name=run_name)
    write_jsonl(output_dir / "pressure_pairs_eval.jsonl", sample_rows)

    artifact_run_dir = Path(args.artifact_run_dir)
    artifact = _load_artifact(artifact_run_dir / "belief_causal_subspace_artifact.npz")

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
    hidden_records = belief.base._extract_hidden_records(
        runner=runner,
        pairs=sample_rows,
        layers=layers,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )
    intervention_records, comparisons_raw = belief.base._run_eval(
        runner=runner,
        eval_pairs=sample_rows,
        records=hidden_records,
        artifact=artifact,
        layers=layers,
        k=int(args.k),
        alpha=float(args.alpha),
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
        output_prefix=f"belief_causal_alpha_{str(float(args.alpha)).replace('.', 'p')}",
    )
    comparisons = belief.base._add_baseline_damage(comparisons_raw)
    for row in intervention_records:
        row["alpha"] = float(args.alpha)
    for row in comparisons:
        row["alpha"] = float(args.alpha)
    summary_rows = belief._summary_rows(comparisons)
    write_jsonl(output_dir / "belief_causal_records.jsonl", intervention_records)
    write_jsonl(output_dir / "belief_causal_comparisons.jsonl", comparisons)
    write_csv(output_dir / "belief_causal_summary.csv", summary_rows)
    projection_summary, projection_rows = belief._projection_alignment_outputs(
        eval_pairs=sample_rows,
        records=hidden_records,
        artifact=artifact,
        layers=layers,
        k=int(args.k),
        alpha=float(args.alpha),
        comparisons=comparisons,
        intervention_records=intervention_records,
        output_dir=output_dir,
    )
    save_json(
        output_dir / "authority_pressure_diagnostic_manifest.json",
        {
            "pipeline": "authority_pressure_diagnostic_v1",
            "sample_file": str(Path(args.sample_file).resolve()),
            "artifact_run_dir": str(artifact_run_dir.resolve()),
            "model_name": str(args.model_name),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layers": list(layers),
            "k": int(args.k),
            "alpha": float(args.alpha),
            "num_items": len(sample_rows),
            "outputs": {
                "projection_alignment_summary_json": str((output_dir / "projection_alignment_summary.json").resolve()),
                "projection_alignment_diagnostic_csv": str((output_dir / "projection_alignment_diagnostic.csv").resolve()),
                "belief_causal_summary_csv": str((output_dir / "belief_causal_summary.csv").resolve()),
            },
            "projection_alignment_summary": projection_summary,
            "num_projection_rows": len(projection_rows),
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_items": len(sample_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
