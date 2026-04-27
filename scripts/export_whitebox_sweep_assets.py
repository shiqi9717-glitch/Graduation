#!/usr/bin/env python3
"""Export table-ready layer/strength sweep assets from frozen intervention stability runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.open_model_probe.io_utils import prepare_output_dir, save_json


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload.get("overall_by_setting", [])]


def _transform_rows(model_label: str, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("method")) != "baseline_state_interpolation":
            continue
        compliance_ref = float(row.get("wrong_option_follow_rate_reference") or 0.0)
        compliance_int = float(row.get("wrong_option_follow_rate") or 0.0)
        out.append(
            {
                "model": model_label,
                "layer_config_name": str(row.get("layer_config_name") or ""),
                "beta": float(row.get("active_scale") or 0.0),
                "num_samples": int(row.get("num_samples") or 0),
                "pressured_compliance_delta": compliance_int - compliance_ref,
                "baseline_damage_rate": float(row.get("baseline_damage_rate") or 0.0),
                "recovery_delta": float(row.get("intervention_recovery_rate") or 0.0),
                "stance_drift_delta": float(row.get("intervention_error_rate") or 0.0)
                - float(row.get("interference_induced_error_rate") or 0.0),
                "net_recovery_without_damage": float(row.get("net_recovery_without_damage") or 0.0),
            }
        )
    out.sort(key=lambda row: (row["model"], row["layer_config_name"], row["beta"]))
    return out


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [",".join(fields)]
    for row in rows:
        rendered = []
        for field in fields:
            text = str(row.get(field, ""))
            if "," in text or "\n" in text or "\"" in text:
                text = "\"" + text.replace("\"", "\"\"") + "\""
            rendered.append(text)
        lines.append(",".join(rendered))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pivot(path: Path, rows: Sequence[dict[str, Any]], *, metric: str) -> None:
    layer_names = sorted({str(row["layer_config_name"]) for row in rows})
    betas = sorted({float(row["beta"]) for row in rows})
    header = ["layer_config_name"] + [str(beta) for beta in betas]
    lines = [",".join(header)]
    for layer_name in layer_names:
        line = [layer_name]
        for beta in betas:
            matched = next(
                (
                    row
                    for row in rows
                    if str(row["layer_config_name"]) == layer_name and float(row["beta"]) == beta
                ),
                None,
            )
            line.append("" if matched is None else str(matched.get(metric, "")))
        lines.append(",".join(line))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper-ready sweep CSV assets from frozen stability runs.")
    parser.add_argument(
        "--qwen7b-summary",
        default="outputs/experiments/local_probe_qwen7b_intervention_stability/Qwen_Qwen2.5-7B-Instruct/20260423_095800/intervention_summary.json",
    )
    parser.add_argument(
        "--qwen3b-summary",
        default="outputs/experiments/local_probe_qwen3b_intervention_stability/Qwen_Qwen2.5-3B-Instruct/20260423_084014/intervention_summary.json",
    )
    parser.add_argument("--output-root", default="outputs/experiments/whitebox_sweep_assets")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = prepare_output_dir(Path(args.output_root), run_name="qwen_layer_strength_sweep")

    qwen7b_rows = _transform_rows("Qwen-7B", _load_summary_rows(Path(args.qwen7b_summary)))
    qwen3b_rows = _transform_rows("Qwen-3B", _load_summary_rows(Path(args.qwen3b_summary)))
    all_rows = [*qwen7b_rows, *qwen3b_rows]

    save_json(
        output_dir / "sweep_manifest.json",
        {
            "pipeline": "whitebox_sweep_asset_export_v1",
            "sources": {
                "qwen7b_summary": str(Path(args.qwen7b_summary).resolve()),
                "qwen3b_summary": str(Path(args.qwen3b_summary).resolve()),
            },
            "output_dir": str(output_dir.resolve()),
        },
    )
    _write_csv(output_dir / "qwen_mainline_sweep_long.csv", all_rows)
    for model_label, rows in (("qwen7b", qwen7b_rows), ("qwen3b", qwen3b_rows)):
        _write_csv(output_dir / f"{model_label}_sweep_long.csv", rows)
        for metric in ("pressured_compliance_delta", "baseline_damage_rate", "recovery_delta"):
            _write_pivot(output_dir / f"{model_label}_{metric}_pivot.csv", rows, metric=metric)

    (output_dir / "paper_summary.md").write_text(
        "\n".join(
            [
                "# Qwen Layer/Strength Sweep Assets",
                "",
                "- Export type: table-ready CSVs from frozen intervention stability summaries",
                f"- Qwen-7B source: `{Path(args.qwen7b_summary).resolve()}`",
                f"- Qwen-3B source: `{Path(args.qwen3b_summary).resolve()}`",
                "",
                "Generated files:",
                "- `qwen_mainline_sweep_long.csv`",
                "- `qwen7b_pressured_compliance_delta_pivot.csv`",
                "- `qwen7b_baseline_damage_rate_pivot.csv`",
                "- `qwen7b_recovery_delta_pivot.csv`",
                "- `qwen3b_pressured_compliance_delta_pivot.csv`",
                "- `qwen3b_baseline_damage_rate_pivot.csv`",
                "- `qwen3b_recovery_delta_pivot.csv`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_rows": len(all_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
