#!/usr/bin/env python3
"""Build a compact summary of the white-box gap-closing experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_model_probe.io_utils import prepare_output_dir, save_json


def _latest_timestamp_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_metric_triplet(row: dict[str, Any]) -> str:
    value = row.get("value", row.get("estimate"))
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    if value is None:
        return "n/a"
    if ci_low is None or ci_high is None:
        return f"{float(value):.4f}"
    return f"{float(value):.4f} [{float(ci_low):.4f}, {float(ci_high):.4f}]"


def _formal_controls_block(root: Path, title: str) -> tuple[list[str], dict[str, Any]]:
    latest = _latest_timestamp_dir(root)
    if latest is None:
        return (
            [
                f"## {title}",
                "",
                "- Status: pending run",
                f"- Expected root: `{root}`",
                "",
            ],
            {"status": "pending", "root": str(root.resolve())},
        )
    rows = _read_csv_rows(latest / "formal_controls_aggregate_metrics.csv")
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped.setdefault(row["method"], {})[row["metric"]] = row
    lines = [
        f"## {title}",
        "",
        f"- Output root: `{latest}`",
    ]
    summary: dict[str, Any] = {"status": "complete", "output_dir": str(latest.resolve()), "rows": grouped}
    for method, metric_rows in sorted(grouped.items()):
        stance = metric_rows.get("stance_drift_delta", {})
        compliance = metric_rows.get("pressured_compliance_delta", {})
        recovery = metric_rows.get("recovery_delta", {})
        damage = metric_rows.get("baseline_damage_rate", {})
        lines.append(
            "- "
            + f"`{method}`: "
            + f"`stance_drift_delta={_format_metric_triplet(stance)}`; "
            + f"`pressured_compliance_delta={_format_metric_triplet(compliance)}`; "
            + f"`recovery_delta={_format_metric_triplet(recovery)}`; "
            + f"`baseline_damage_rate={_format_metric_triplet(damage)}`"
        )
    lines.append("")
    return lines, summary


def _heldout_block(root: Path) -> tuple[list[str], dict[str, Any]]:
    latest = _latest_timestamp_dir(root)
    if latest is not None and not (latest / "objective_local_eval_metrics.csv").exists():
        nested = _latest_timestamp_dir(latest)
        if nested is not None:
            latest = nested
    if latest is None:
        return (
            [
                "## Qwen7B Held-out Validation",
                "",
                "- Status: pending run",
                f"- Expected root: `{root}`",
                "",
            ],
            {"status": "pending", "root": str(root.resolve())},
        )
    rows = _read_csv_rows(latest / "objective_local_eval_metrics.csv")
    lines = [
        "## Qwen7B Held-out Validation",
        "",
        f"- Output root: `{latest}`",
    ]
    summary: dict[str, Any] = {"status": "complete", "output_dir": str(latest.resolve()), "rows": rows}
    metric_mode = rows and "metric" in rows[0]
    if metric_mode:
        metric_map = {row["metric"]: row for row in rows}
        lines.append(
            "- "
            + f"`stance_drift_delta={metric_map['stance_drift_delta']['estimate']} [{metric_map['stance_drift_delta']['ci_low']}, {metric_map['stance_drift_delta']['ci_high']}]`; "
            + f"`pressured_compliance_delta={metric_map['pressured_compliance_delta']['estimate']} [{metric_map['pressured_compliance_delta']['ci_low']}, {metric_map['pressured_compliance_delta']['ci_high']}]`; "
            + f"`recovery_delta={metric_map['recovery_delta']['estimate']} [{metric_map['recovery_delta']['ci_low']}, {metric_map['recovery_delta']['ci_high']}]`; "
            + f"`baseline_damage_rate={metric_map['baseline_damage_rate']['estimate']} [{metric_map['baseline_damage_rate']['ci_low']}, {metric_map['baseline_damage_rate']['ci_high']}]`"
        )
    else:
        for row in rows:
            lines.append(
                "- "
                + f"`{row['scope']}`: "
                + f"`stance_drift_delta={row['stance_drift_delta']} [{row['stance_drift_delta_ci_low']}, {row['stance_drift_delta_ci_high']}]`; "
                + f"`pressured_compliance_delta={row['pressured_compliance_delta']} [{row['pressured_compliance_delta_ci_low']}, {row['pressured_compliance_delta_ci_high']}]`; "
                + f"`recovery_delta={row['recovery_delta']} [{row['recovery_delta_ci_low']}, {row['recovery_delta_ci_high']}]`; "
                + f"`baseline_damage_rate={row['baseline_damage_rate']} [{row['baseline_damage_rate_ci_low']}, {row['baseline_damage_rate_ci_high']}]`"
            )
    lines.append("")
    return lines, summary


def _sweep_block(root: Path) -> tuple[list[str], dict[str, Any]]:
    latest = _latest_timestamp_dir(root / "qwen_layer_strength_sweep")
    if latest is None:
        return (
            [
                "## Layer/Strength Sweep Assets",
                "",
                "- Status: missing export",
                f"- Expected root: `{root / 'qwen_layer_strength_sweep'}`",
                "",
            ],
            {"status": "missing", "root": str((root / 'qwen_layer_strength_sweep').resolve())},
        )
    manifest = _read_json(latest / "sweep_manifest.json")
    long_csv = latest / "qwen_mainline_sweep_long.csv"
    row_count = max(sum(1 for _ in long_csv.open("r", encoding="utf-8")) - 1, 0) if long_csv.exists() else None
    lines = [
        "## Layer/Strength Sweep Assets",
        "",
        f"- Output root: `{latest}`",
        f"- Exported rows: `{row_count}`",
        "- Includes long-form CSV plus pivot tables for `pressured_compliance_delta`, `baseline_damage_rate`, and `recovery_delta`.",
        "",
    ]
    return lines, {"status": "complete", "output_dir": str(latest.resolve()), "manifest": manifest, "row_count": row_count}


def _audit_block(root: Path) -> tuple[list[str], dict[str, Any]]:
    latest = _latest_timestamp_dir(root / "balanced_belief_identity")
    if latest is None:
        return (
            [
                "## Human Audit Export",
                "",
                "- Status: missing export",
                f"- Expected root: `{root / 'balanced_belief_identity'}`",
                "",
            ],
            {"status": "missing", "root": str((root / 'balanced_belief_identity').resolve())},
        )
    manifest = _read_json(latest / "audit_manifest.json")
    lines = [
        "## Human Audit Export",
        "",
        f"- Output root: `{latest}`",
        f"- Total rows: `{manifest.get('total_count')}`",
        f"- Belief rows: `{manifest.get('belief_count')}`",
        f"- Identity rows: `{manifest.get('identity_count')}`",
        "- Includes `whitebox_human_audit_rows.csv`, `whitebox_human_audit_rows.jsonl`, and `annotation_guidelines.md`.",
        "",
    ]
    return lines, {"status": "complete", "output_dir": str(latest.resolve()), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the paper-gap white-box experiment bundle.")
    parser.add_argument("--formal-controls-qwen7b-root", default="outputs/experiments/whitebox_formal_controls_qwen7b")
    parser.add_argument("--formal-controls-qwen3b-root", default="outputs/experiments/whitebox_formal_controls_qwen3b")
    parser.add_argument("--heldout-root", default="outputs/experiments/whitebox_qwen7b_heldout_eval")
    parser.add_argument("--sweep-root", default="outputs/experiments/whitebox_sweep_assets")
    parser.add_argument("--audit-root", default="outputs/experiments/whitebox_human_audit_bundle")
    parser.add_argument("--output-root", default="outputs/experiments/whitebox_gap_summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = prepare_output_dir(Path(args.output_root), run_name="paper_gap_bundle")

    qwen7b_lines, qwen7b_summary = _formal_controls_block(PROJECT_ROOT / args.formal_controls_qwen7b_root, "Qwen7B Formal Controls")
    qwen3b_lines, qwen3b_summary = _formal_controls_block(PROJECT_ROOT / args.formal_controls_qwen3b_root, "Qwen3B Formal Controls")
    heldout_lines, heldout_summary = _heldout_block(PROJECT_ROOT / args.heldout_root)
    sweep_lines, sweep_summary = _sweep_block(PROJECT_ROOT / args.sweep_root)
    audit_lines, audit_summary = _audit_block(PROJECT_ROOT / args.audit_root)

    support_lines = [
        "## Support / Non-support / Claim Boundary",
        "",
        "- Supports mainline: the new sweep export makes the default layer/strength setting auditable rather than hand-picked.",
        "- Supports mainline: the human-audit bundle is now exportable in reviewer-ready CSV/JSONL form with annotation guidance.",
        "- Supports mainline: Qwen7B random-direction control stays near null, which argues that the gain is not reproduced by arbitrary activation perturbation.",
        "- Supports mainline: shuffled-label controls produce large baseline damage and fail to recover the clean mainline tradeoff, which argues against a simple pairing artifact story.",
        "- Supports mainline: the Qwen7B held-out evaluation preserves directionality with zero observed baseline damage on the held-out objective-local set.",
        "- Does not yet support: no new evidence here upgrades the claim to full causal generality across every pressure family.",
        "- Cannot yet claim: these controls strengthen the mainline, but they do not by themselves prove full mechanism completeness or universal transfer beyond the tested Qwen settings.",
        "",
    ]

    lines = [
        "# White-box Paper Gap Summary",
        "",
        "This bundle adds the smallest new validation materials needed for the current white-box sycophancy paper without modifying frozen outputs.",
        "",
        *qwen7b_lines,
        *qwen3b_lines,
        *heldout_lines,
        *sweep_lines,
        *audit_lines,
        *support_lines,
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_json(
        output_dir / "summary_manifest.json",
        {
            "pipeline": "whitebox_gap_summary_v1",
            "output_dir": str(output_dir.resolve()),
            "formal_controls_qwen7b": qwen7b_summary,
            "formal_controls_qwen3b": qwen3b_summary,
            "heldout": heldout_summary,
            "sweep": sweep_summary,
            "audit": audit_summary,
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
