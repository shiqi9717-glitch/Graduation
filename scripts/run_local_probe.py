#!/usr/bin/env python3
"""Run a minimal local open-model white-box probe on one or more samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import setup_logger
from src.open_model_probe.analyzer import analyze_sample, summarize_probe_results
from src.open_model_probe.io_utils import load_sample_file, prepare_output_dir, save_json, sanitize_filename
from src.open_model_probe.model_runner import LocalProbeConfig, LocalProbeRunner
from src.open_model_probe.prompt_builder import SCENARIO_CHOICES


def _parse_hidden_state_layers(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return (-1, -2, -3, -4)
    return tuple(int(item) for item in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local white-box probe for open-source causal language models.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sample-file", required=True, help="JSON file containing one sample object or a list of samples.")
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen3b")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="auto", help="Torch dtype name, e.g. auto / float16 / float32.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--hidden-state-layers", default="-1,-2,-3,-4")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default=None)
    parser.add_argument("--run-all-scenarios", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(name="local_probe", level=args.log_level)

    samples = load_sample_file(Path(args.sample_file))
    scenarios = list(SCENARIO_CHOICES) if args.run_all_scenarios or not args.scenario else [str(args.scenario)]
    output_dir = prepare_output_dir(Path(args.output_dir), run_name=sanitize_filename(args.model_name))

    runner = LocalProbeRunner(
        LocalProbeConfig(
            model_name=str(args.model_name),
            device=str(args.device),
            dtype=str(args.dtype),
            max_length=int(args.max_length),
            top_k=int(args.top_k),
            hidden_state_layers=_parse_hidden_state_layers(args.hidden_state_layers),
        )
    )
    records: list[dict] = []
    comparisons: list[dict] = []
    records_path = output_dir / "probe_records.jsonl"
    comparisons_path = output_dir / "probe_comparisons.jsonl"
    partial_summary_path = output_dir / "probe_run.partial.json"

    for sample in samples:
        sample_records, comparison = analyze_sample(
            runner,
            sample,
            output_dir=output_dir,
            scenarios=scenarios,
        )
        records.extend(sample_records)
        comparisons.append(comparison)

        with open(records_path, "a", encoding="utf-8") as f:
            for row in sample_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(comparisons_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(comparison, ensure_ascii=False) + "\n")

        partial_payload = {
            "model_name": str(args.model_name),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "max_length": int(args.max_length),
            "top_k": int(args.top_k),
            "hidden_state_layers": list(_parse_hidden_state_layers(args.hidden_state_layers)),
            "sample_file": str(Path(args.sample_file).resolve()),
            "output_dir": str(output_dir.resolve()),
            "num_samples_target": len(samples),
            "num_samples_completed": len(comparisons),
            "scenarios": scenarios,
            "closed_loop_recheck": bool("interference" in scenarios and "recheck" in scenarios),
            "summary_so_far": summarize_probe_results(records, comparisons),
        }
        save_json(partial_summary_path, partial_payload)

    payload = {
        "model_name": str(args.model_name),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "max_length": int(args.max_length),
        "top_k": int(args.top_k),
        "hidden_state_layers": list(_parse_hidden_state_layers(args.hidden_state_layers)),
        "sample_file": str(Path(args.sample_file).resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_samples": len(samples),
        "scenarios": scenarios,
        "closed_loop_recheck": bool("interference" in scenarios and "recheck" in scenarios),
        "summary": summarize_probe_results(records, comparisons),
        "records": records,
        "comparisons": comparisons,
    }

    save_json(output_dir / "probe_run.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
