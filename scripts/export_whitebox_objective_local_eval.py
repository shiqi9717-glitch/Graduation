#!/usr/bin/env python3
"""Export objective-local proxy metrics with bootstrap CIs from an intervention run."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_model_probe.io_utils import prepare_output_dir, save_json, save_jsonl


FOCUS_SAMPLE_TYPES = ("strict_positive", "high_pressure_wrong_option")
TARGET_METRICS = (
    "stance_drift_delta",
    "pressured_compliance_delta",
    "recovery_delta",
    "baseline_damage_rate",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean_bool(values: Iterable[bool]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(bool(value) for value in values) / len(values))


def _compute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    focus_rows = [row for row in rows if str(row.get("sample_type")) in set(FOCUS_SAMPLE_TYPES)]
    baseline_correct = [row for row in focus_rows if bool(row.get("baseline_reference_correct"))]
    reference_error = [row for row in baseline_correct if bool(row.get("interference_induced_error_reference"))]

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
        "n_focus": float(len(focus_rows)),
        "n_baseline_correct": float(len(baseline_correct)),
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


def _bootstrap(rows: Sequence[dict[str, Any]], *, seed: int, iters: int) -> dict[str, dict[str, float | None]]:
    focus_rows = [row for row in rows if str(row.get("sample_type")) in set(FOCUS_SAMPLE_TYPES)]
    rng = random.Random(int(seed))
    values = {metric: [] for metric in TARGET_METRICS}
    for _ in range(int(iters)):
        sampled = [rng.choice(focus_rows) for _ in range(len(focus_rows))]
        metrics = _compute_metrics(sampled)
        for metric in TARGET_METRICS:
            if metrics.get(metric) is not None:
                values[metric].append(float(metrics[metric]))
    out: dict[str, dict[str, float | None]] = {}
    for metric, seq in values.items():
        if not seq:
            out[metric] = {"ci_low": None, "ci_high": None}
            continue
        seq.sort()
        low = seq[max(0, int(0.025 * (len(seq) - 1)))]
        high = seq[min(len(seq) - 1, int(0.975 * (len(seq) - 1)))]
        out[metric] = {"ci_low": float(low), "ci_high": float(high)}
    return out


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        rendered = []
        for field in fields:
            value = row.get(field)
            text = "" if value is None else str(value)
            if "," in text or "\n" in text or "\"" in text:
                text = "\"" + text.replace("\"", "\"\"") + "\""
            rendered.append(text)
        lines.append(",".join(rendered))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export objective-local proxy metrics from a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="outputs/experiments/whitebox_objective_local_eval")
    parser.add_argument("--label", default="")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260427)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_payload = json.loads((run_dir / "intervention_run.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(run_dir / "intervention_comparisons.jsonl")
    metrics = _compute_metrics(rows)
    ci = _bootstrap(rows, seed=int(args.bootstrap_seed), iters=int(args.bootstrap_iters))

    label = str(args.label or run_dir.parent.name or run_dir.name)
    output_dir = prepare_output_dir(Path(args.output_root), run_name=label)
    table_rows = []
    for metric in TARGET_METRICS:
        table_rows.append(
            {
                "metric": metric,
                "estimate": metrics.get(metric),
                "ci_low": ci.get(metric, {}).get("ci_low"),
                "ci_high": ci.get(metric, {}).get("ci_high"),
                "run_dir": str(run_dir),
            }
        )

    manifest = {
        "pipeline": "whitebox_objective_local_eval_export_v1",
        "label": label,
        "source_run_dir": str(run_dir),
        "model_name": run_payload.get("model_name"),
        "sample_file": run_payload.get("sample_file"),
        "tested_settings": run_payload.get("tested_settings", []),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metrics": metrics,
        "ci": ci,
    }
    save_json(output_dir / "objective_local_eval_manifest.json", manifest)
    save_jsonl(output_dir / "objective_local_eval_metrics.jsonl", table_rows)
    (output_dir / "objective_local_eval_metrics.csv").write_text(_csv_text(table_rows), encoding="utf-8")
    (output_dir / "paper_summary.md").write_text(
        "\n".join(
            [
                f"# Objective-local Evaluation Summary: {label}",
                "",
                f"- Source run: `{run_dir}`",
                f"- Sample file: `{run_payload.get('sample_file')}`",
                "",
                "| metric | estimate | 95% CI |",
                "| --- | ---: | --- |",
                *[
                    f"| {row['metric']} | {float(row['estimate']):.4f} | [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
                    if row["estimate"] is not None and row["ci_low"] is not None and row["ci_high"] is not None
                    else f"| {row['metric']} | NA | NA |"
                    for row in table_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
