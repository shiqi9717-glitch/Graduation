#!/usr/bin/env python3
"""Run cross-model behavioral layer sweeps for pressure_subspace_damping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_bridge_dataset
from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir, save_json
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.pressure_subspace import parse_layers, write_csv, write_jsonl

import scripts.run_belief_causal_transfer as belief_transfer


def _source_paths() -> List[Path]:
    return [project_root / "data" / "external" / "sycophancy_database" / name for name in FIXED_DATA_SOURCES]


def _parse_alpha_values(value: str) -> List[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one alpha value is required.")
    return values


def _parse_layer_configs(value: str) -> List[Tuple[str, List[int]]]:
    configs: List[Tuple[str, List[int]]] = []
    for chunk in str(value).split(";"):
        spec = chunk.strip()
        if not spec:
            continue
        if "=" in spec:
            name, layer_expr = spec.split("=", 1)
            config_name = str(name).strip()
            layer_text = str(layer_expr).strip()
        else:
            layer_text = spec
            config_name = layer_text.replace(",", "+")
        if not config_name:
            raise ValueError(f"Invalid layer config with empty name: {spec}")
        layers = [int(part.strip()) for part in layer_text.split(",") if part.strip()]
        if not layers:
            layers = list(parse_layers(layer_text))
        if not layers:
            raise ValueError(f"Invalid layer config with no layers: {spec}")
        configs.append((config_name, layers))
    if not configs:
        raise ValueError("At least one layer config is required.")
    return configs


def _summary_by_method(summary_rows: Sequence[Dict[str, Any]], alpha: float) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["method"]): dict(row)
        for row in summary_rows
        if float(row.get("alpha", alpha)) == float(alpha)
    }


def _model_label(model_name: str) -> str:
    normalized = str(model_name)
    if "Llama-3.1-8B" in normalized:
        return "Llama-3.1-8B"
    if "glm-4-9b" in normalized.lower():
        return "GLM-4-9B"
    if "Qwen2.5-7B" in normalized:
        return "Qwen-7B"
    return Path(normalized).name.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run behavioral layer sweeps for cross-model pressure_subspace_damping.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--train-source", default="philpapers2020")
    parser.add_argument("--eval-source", default="nlp_survey")
    parser.add_argument("--pressure-type", default="belief_argument")
    parser.add_argument("--prompt-variant", default="original", choices=["original", "english", "en", "chinese", "zh", "zh_instruction"])
    parser.add_argument("--train-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--method", default="pressure_subspace_damping", choices=["pressure_subspace_damping"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha-values", default="0.5,0.75")
    parser.add_argument("--layer-configs", required=True, help="Semicolon-separated configs like '20-21=20,21;20-27=20,21,...'")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--output-root", default="outputs/experiments/cross_model_behavioral_layer_sweep")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger(name="cross_model_behavioral_layer_sweep", level=args.log_level)
    alpha_values = _parse_alpha_values(str(args.alpha_values))
    layer_configs = _parse_layer_configs(str(args.layer_configs))
    union_layers = sorted({layer for _name, layers in layer_configs for layer in layers})
    model_name = str(args.model_name)
    run_name = Path(model_name).name if Path(model_name).exists() else model_name.replace("/", "_")
    output_dir = prepare_output_dir(Path(args.output_root), run_name=run_name)

    all_items = [item.to_dict() for item in build_bridge_dataset(_source_paths())]
    train_pairs = belief_transfer.base._sample_source(
        items=all_items,
        source=str(args.train_source),
        n=int(args.train_n),
        split="train",
        seed=int(args.seed),
    )
    eval_pairs = belief_transfer.base._sample_source(
        items=all_items,
        source=str(args.eval_source),
        n=int(args.eval_n),
        split="eval",
        seed=int(args.seed) + 1,
    )
    if str(args.pressure_type) != "belief_argument":
        train_pairs = [row for row in train_pairs if row.get("pressure_type") == str(args.pressure_type)]
        eval_pairs = [row for row in eval_pairs if row.get("pressure_type") == str(args.pressure_type)]
    train_pairs = belief_transfer._apply_prompt_variant(train_pairs, str(args.prompt_variant))
    eval_pairs = belief_transfer._apply_prompt_variant(eval_pairs, str(args.prompt_variant))
    write_jsonl(output_dir / "pressure_pairs_train.jsonl", train_pairs)
    write_jsonl(output_dir / "pressure_pairs_eval.jsonl", eval_pairs)

    logger.info(
        "Starting behavioral layer sweep: model=%s train=%d eval=%d union_layers=%s configs=%d output=%s",
        model_name,
        len(train_pairs),
        len(eval_pairs),
        union_layers,
        len(layer_configs),
        output_dir,
    )
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
    hidden_records = belief_transfer.base._extract_hidden_records(
        runner=runner,
        pairs=[*train_pairs, *eval_pairs],
        layers=union_layers,
        output_dir=output_dir,
        logger=logger,
        flush_every=int(args.flush_every),
    )

    long_rows: List[Dict[str, Any]] = []
    config_manifest: List[Dict[str, Any]] = []
    model_label = _model_label(model_name)
    for config_name, layers in layer_configs:
        logger.info("Evaluating layer config %s with layers=%s", config_name, layers)
        config_dir = output_dir / "layer_configs" / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        artifact = belief_transfer._estimate_subspace(
            train_pairs=train_pairs,
            records=hidden_records,
            layers=layers,
            k=int(args.k),
            seed=int(args.seed),
            output_dir=config_dir,
            train_source=str(args.train_source),
            pressure_type=str(args.pressure_type),
        )
        all_intervention_records: List[Dict[str, Any]] = []
        all_comparisons: List[Dict[str, Any]] = []
        for alpha in alpha_values:
            output_prefix = f"{config_name}__alpha_{str(alpha).replace('.', 'p')}"
            intervention_records, comparisons_raw = belief_transfer.base._run_eval(
                runner=runner,
                eval_pairs=eval_pairs,
                records=hidden_records,
                artifact=artifact,
                layers=layers,
                k=int(args.k),
                alpha=float(alpha),
                output_dir=config_dir,
                logger=logger,
                flush_every=int(args.flush_every),
                output_prefix=output_prefix,
            )
            comparisons = belief_transfer.base._add_baseline_damage(comparisons_raw)
            for row in intervention_records:
                row["alpha"] = float(alpha)
                row["layer_config_name"] = config_name
            for row in comparisons:
                row["alpha"] = float(alpha)
                row["layer_config_name"] = config_name
            all_intervention_records.extend(intervention_records)
            all_comparisons.extend(comparisons)

        summary_rows = belief_transfer._summary_rows(all_comparisons)
        write_jsonl(config_dir / "belief_causal_records.jsonl", all_intervention_records)
        write_jsonl(config_dir / "belief_causal_comparisons.jsonl", all_comparisons)
        write_csv(config_dir / "belief_causal_summary.csv", summary_rows)

        for alpha in alpha_values:
            by_method = _summary_by_method(summary_rows, float(alpha))
            no_row = by_method.get("no_intervention", {})
            patched_row = by_method.get("matched_belief_subspace_damping", {})
            long_rows.append(
                {
                    "model": model_label,
                    "layer_config_name": config_name,
                    "beta": float(alpha),
                    "num_samples": int(patched_row.get("num_items") or no_row.get("num_items") or 0),
                    "pressured_compliance_delta": float(patched_row.get("pressured_compliance_rate") or 0.0)
                    - float(no_row.get("pressured_compliance_rate") or 0.0),
                    "baseline_damage_rate": float(patched_row.get("baseline_damage_rate") or 0.0),
                    "recovery_delta": float(patched_row.get("recovery_rate") or 0.0)
                    - float(no_row.get("recovery_rate") or 0.0),
                    "stance_drift_delta": float(patched_row.get("stance_drift_rate") or 0.0)
                    - float(no_row.get("stance_drift_rate") or 0.0),
                    "net_recovery_without_damage": (
                        float(patched_row.get("recovery_rate") or 0.0)
                        - float(no_row.get("recovery_rate") or 0.0)
                        - float(patched_row.get("baseline_damage_rate") or 0.0)
                    ),
                }
            )
        config_manifest.append(
            {
                "layer_config_name": config_name,
                "layers": list(layers),
                "output_dir": str(config_dir.resolve()),
                "summary_csv": str((config_dir / "belief_causal_summary.csv").resolve()),
            }
        )

    long_rows.sort(key=lambda row: (str(row["model"]), str(row["layer_config_name"]), float(row["beta"])))
    write_csv(output_dir / "behavioral_sweep_long.csv", long_rows)
    save_json(
        output_dir / "behavioral_sweep_manifest.json",
        {
            "pipeline": "cross_model_behavioral_layer_sweep_v1",
            "model_name": model_name,
            "model_label": model_label,
            "device": str(args.device),
            "dtype": str(args.dtype),
            "train_source": str(args.train_source),
            "eval_source": str(args.eval_source),
            "pressure_type": str(args.pressure_type),
            "prompt_variant": str(args.prompt_variant),
            "train_n": len(train_pairs),
            "eval_n": len(eval_pairs),
            "seed": int(args.seed),
            "method": str(args.method),
            "k": int(args.k),
            "alpha_values": list(alpha_values),
            "union_layers": list(union_layers),
            "layer_configs": config_manifest,
            "outputs": {
                "long_csv": str((output_dir / "behavioral_sweep_long.csv").resolve()),
                "hidden_state_records": str((output_dir / "hidden_state_records.jsonl").resolve()),
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "num_layer_configs": len(layer_configs),
                "num_long_rows": len(long_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
