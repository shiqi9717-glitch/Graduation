#!/usr/bin/env python3
"""Run multi-seed formal white-box control suites and summarize them."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_model_probe.io_utils import load_sample_file, save_json, save_jsonl


FOCUS_SAMPLE_TYPES = ("strict_positive", "high_pressure_wrong_option")
TARGET_METRICS = (
    "stance_drift_delta",
    "pressured_compliance_delta",
    "recovery_delta",
    "baseline_damage_rate",
)


@dataclass(frozen=True)
class SeedRun:
    method: str
    seed: int
    run_dir: Path


def _parse_layers(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw or "").split(",") if part.strip())


def _parse_methods(raw: str) -> tuple[str, ...]:
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    return tuple(values or ("random_direction_control", "shuffled_label_control"))


def _parse_sample_types(raw: str) -> tuple[str, ...]:
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    return tuple(values or FOCUS_SAMPLE_TYPES)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_child_dir(root: Path) -> Path:
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return dirs[-1]


def _compute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    focus_rows = [row for row in rows if str(row.get("sample_type")) in set(FOCUS_SAMPLE_TYPES)]
    baseline_correct = [row for row in focus_rows if bool(row.get("baseline_reference_correct"))]
    reference_error = [row for row in baseline_correct if bool(row.get("interference_induced_error_reference"))]
    intervention_error = [row for row in baseline_correct if not bool(row.get("interference_intervened_correct"))]
    baseline_damage = [row for row in baseline_correct if bool(row.get("baseline_damage"))]

    def _mean_bool(values: Iterable[bool]) -> float | None:
        values = list(values)
        if not values:
            return None
        return float(sum(bool(value) for value in values) / len(values))

    reference_error_rate = _mean_bool(row.get("interference_induced_error_reference") for row in baseline_correct)
    intervention_error_rate = _mean_bool(not bool(row.get("interference_intervened_correct")) for row in baseline_correct)
    wrong_follow_reference = _mean_bool(row.get("wrong_option_follow_reference") for row in focus_rows)
    wrong_follow_intervened = _mean_bool(row.get("wrong_option_follow_intervened") for row in focus_rows)
    recovery_rate = (
        float(sum(bool(row.get("intervention_recovers_error")) for row in reference_error) / len(reference_error))
        if reference_error
        else None
    )
    baseline_damage_rate = (
        float(sum(bool(row.get("baseline_damage")) for row in baseline_correct) / len(baseline_correct))
        if baseline_correct
        else None
    )

    return {
        "num_focus_rows": float(len(focus_rows)),
        "num_baseline_correct": float(len(baseline_correct)),
        "stance_drift_delta": (
            float(intervention_error_rate - reference_error_rate)
            if intervention_error_rate is not None and reference_error_rate is not None
            else None
        ),
        "pressured_compliance_delta": (
            float(wrong_follow_intervened - wrong_follow_reference)
            if wrong_follow_intervened is not None and wrong_follow_reference is not None
            else None
        ),
        "recovery_delta": recovery_rate,
        "baseline_damage_rate": baseline_damage_rate,
        "reference_stance_drift_rate": reference_error_rate,
        "intervention_stance_drift_rate": intervention_error_rate,
        "reference_wrong_option_follow_rate": wrong_follow_reference,
        "intervention_wrong_option_follow_rate": wrong_follow_intervened,
    }


def _bootstrap_seed_aggregate(seed_rows: Sequence[Sequence[dict[str, Any]]], *, seed: int, iters: int) -> dict[str, dict[str, float | None]]:
    per_seed_maps: list[dict[str, dict[str, Any]]] = []
    shared_ids: set[str] | None = None
    for rows in seed_rows:
        focus_rows = [row for row in rows if str(row.get("sample_type")) in set(FOCUS_SAMPLE_TYPES)]
        row_map = {str(row["sample_id"]): dict(row) for row in focus_rows}
        per_seed_maps.append(row_map)
        sample_ids = set(row_map)
        shared_ids = sample_ids if shared_ids is None else shared_ids & sample_ids
    ordered_ids = sorted(shared_ids or [])
    if not ordered_ids:
        raise ValueError("No shared sample IDs across seed runs.")

    estimates: dict[str, list[float]] = {metric: [] for metric in TARGET_METRICS}
    rng = random.Random(int(seed))
    for _ in range(int(iters)):
        sampled_ids = [rng.choice(ordered_ids) for _ in range(len(ordered_ids))]
        metric_rows: list[dict[str, float | None]] = []
        for row_map in per_seed_maps:
            sampled_rows = [row_map[sample_id] for sample_id in sampled_ids]
            metric_rows.append(_compute_metrics(sampled_rows))
        for metric in TARGET_METRICS:
            values = [float(row[metric]) for row in metric_rows if row.get(metric) is not None]
            if values:
                estimates[metric].append(float(sum(values) / len(values)))

    summary: dict[str, dict[str, float | None]] = {}
    for metric, values in estimates.items():
        if not values:
            summary[metric] = {"ci_low": None, "ci_high": None}
            continue
        values.sort()
        low_index = max(0, int(math.floor(0.025 * (len(values) - 1))))
        high_index = min(len(values) - 1, int(math.ceil(0.975 * (len(values) - 1))))
        summary[metric] = {
            "ci_low": float(values[low_index]),
            "ci_high": float(values[high_index]),
        }
    return summary


def _aggregate_seed_metrics(seed_runs: Sequence[SeedRun], *, bootstrap_seed: int, bootstrap_iters: int) -> list[dict[str, Any]]:
    by_method: dict[str, list[SeedRun]] = {}
    for run in seed_runs:
        by_method.setdefault(run.method, []).append(run)

    summary_rows: list[dict[str, Any]] = []
    for method, runs in sorted(by_method.items()):
        seed_rows = [_read_jsonl(run.run_dir / "intervention_comparisons.jsonl") for run in runs]
        seed_metrics = [_compute_metrics(rows) for rows in seed_rows]
        ci_lookup = _bootstrap_seed_aggregate(seed_rows, seed=bootstrap_seed, iters=bootstrap_iters)
        for metric in TARGET_METRICS:
            values = [float(row[metric]) for row in seed_metrics if row.get(metric) is not None]
            mean_value = float(sum(values) / len(values)) if values else None
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": mean_value,
                    "ci_low": ci_lookup.get(metric, {}).get("ci_low"),
                    "ci_high": ci_lookup.get(metric, {}).get("ci_high"),
                    "num_seed_runs": len(runs),
                    "seed_values": values,
                    "seed_mean": mean_value,
                    "seed_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                    "seed_min": min(values) if values else None,
                    "seed_max": max(values) if values else None,
                }
            )
    return summary_rows


def _seed_level_rows(seed_runs: Sequence[SeedRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in seed_runs:
        metrics = _compute_metrics(_read_jsonl(run.run_dir / "intervention_comparisons.jsonl"))
        row = {"method": run.method, "seed": int(run.seed), "run_dir": str(run.run_dir.resolve())}
        row.update(metrics)
        rows.append(row)
    return rows


def _render_markdown(
    *,
    model_name: str,
    sample_file: Path,
    focus_layers: Sequence[int],
    beta: float,
    seed_runs: Sequence[SeedRun],
    aggregate_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        f"# Formal Controls Summary: {model_name}",
        "",
        f"- Sample file: `{sample_file}`",
        f"- Focus layers: `{list(focus_layers)}`",
        f"- Strength / beta: `{beta}`",
        f"- Seeded control runs: `{len(seed_runs)}`",
        "",
        "## Aggregate Results",
        "",
        "| method | metric | estimate | 95% CI |",
        "| --- | --- | ---: | --- |",
    ]
    for row in aggregate_rows:
        estimate = "NA" if row.get("estimate") is None else f"{float(row['estimate']):.4f}"
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        ci_text = "NA" if ci_low is None or ci_high is None else f"[{float(ci_low):.4f}, {float(ci_high):.4f}]"
        lines.append(f"| {row['method']} | {row['metric']} | {estimate} | {ci_text} |")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- `random_direction_control` should not exhibit the stable Qwen-mainline pattern of lower drift, lower pressured compliance, positive recovery, and near-zero damage.",
            "- `shuffled_label_control` should not reproduce the mainline gains under the same layer range and strength.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed formal white-box controls.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--reference-mechanistic-run-dir", required=True)
    parser.add_argument("--focus-layers", required=True, help="Comma-separated layer list.")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--methods", default="random_direction_control,shuffled_label_control")
    parser.add_argument("--seeds", default="20260427,20260428,20260429,20260430,20260431")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "mps"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-types", default="strict_positive,high_pressure_wrong_option,control")
    parser.add_argument("--direction-sample-types", default="strict_positive,high_pressure_wrong_option")
    parser.add_argument("--output-root", default="outputs/experiments/whitebox_formal_controls")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260427)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    focus_layers = _parse_layers(args.focus_layers)
    methods = _parse_methods(args.methods)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    sample_file = Path(args.sample_file).resolve()
    mech_dir = Path(args.reference_mechanistic_run_dir).resolve()

    suite_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = (PROJECT_ROOT / args.output_root / suite_stamp).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)

    seed_runs: list[SeedRun] = []
    command_log: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            root_dir = suite_dir / "per_seed" / f"{method}_seed_{seed}"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_local_probe_intervention.py"),
                "--model-name",
                str(args.model_name),
                "--sample-file",
                str(sample_file),
                "--reference-mechanistic-run-dir",
                str(mech_dir),
                "--output-dir",
                str(root_dir),
                "--device",
                str(args.device),
                "--dtype",
                str(args.dtype),
                "--max-length",
                str(int(args.max_length)),
                "--top-k",
                str(int(args.top_k)),
                "--methods",
                method,
                "--sample-types",
                str(args.sample_types),
                "--direction-sample-types",
                str(args.direction_sample_types),
                "--layer-configs",
                f"mainline={','.join(str(layer) for layer in focus_layers)}",
                "--interpolation-scales",
                str(float(args.beta)),
                "--control-random-seed",
                str(int(seed)),
                "--log-level",
                str(args.log_level),
            ]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            model_root = root_dir / str(args.model_name).replace("/", "_")
            run_dir = _latest_child_dir(model_root)
            seed_runs.append(SeedRun(method=method, seed=int(seed), run_dir=run_dir))
            command_log.append({"method": method, "seed": int(seed), "cmd": cmd, "run_dir": str(run_dir.resolve())})

    seed_level = _seed_level_rows(seed_runs)
    aggregate_rows = _aggregate_seed_metrics(
        seed_runs,
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_iters=int(args.bootstrap_iters),
    )
    manifest = {
        "pipeline": "whitebox_formal_controls_v1",
        "model_name": str(args.model_name),
        "sample_file": str(sample_file),
        "reference_mechanistic_run_dir": str(mech_dir),
        "focus_layers": list(focus_layers),
        "beta": float(args.beta),
        "methods": list(methods),
        "seeds": list(seeds),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "sample_types": list(_parse_sample_types(args.sample_types)),
        "direction_sample_types": list(_parse_sample_types(args.direction_sample_types)),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "suite_dir": str(suite_dir),
        "seed_runs": [
            {"method": run.method, "seed": int(run.seed), "run_dir": str(run.run_dir.resolve())}
            for run in seed_runs
        ],
        "commands": command_log,
    }

    save_json(suite_dir / "formal_controls_manifest.json", manifest)
    save_jsonl(suite_dir / "formal_controls_seed_level.jsonl", seed_level)
    save_jsonl(suite_dir / "formal_controls_aggregate_metrics.jsonl", aggregate_rows)
    (suite_dir / "formal_controls_seed_level.csv").write_text(
        _csv_text(seed_level),
        encoding="utf-8",
    )
    (suite_dir / "formal_controls_aggregate_metrics.csv").write_text(
        _csv_text(aggregate_rows),
        encoding="utf-8",
    )
    (suite_dir / "paper_summary.md").write_text(
        _render_markdown(
            model_name=str(args.model_name),
            sample_file=sample_file,
            focus_layers=focus_layers,
            beta=float(args.beta),
            seed_runs=seed_runs,
            aggregate_rows=aggregate_rows,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"suite_dir": str(suite_dir), "num_seed_runs": len(seed_runs)}, ensure_ascii=False, indent=2))
    return 0


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    lines = [",".join(fieldnames)]
    for row in rows:
        rendered: list[str] = []
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, list):
                rendered.append(json.dumps(value, ensure_ascii=False))
            elif value is None:
                rendered.append("")
            else:
                text = str(value)
                if "," in text or "\n" in text or "\"" in text:
                    text = "\"" + text.replace("\"", "\"\"") + "\""
                rendered.append(text)
        lines.append(",".join(rendered))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
