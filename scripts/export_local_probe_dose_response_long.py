#!/usr/bin/env python3
"""Export a qwen-style long CSV from local probe intervention summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload.get("overall_by_setting", [])]


def _transform_rows(model_label: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("method")) != "baseline_state_interpolation":
            continue
        out.append(
            {
                "model": model_label,
                "layer_config_name": str(row.get("layer_config_name") or ""),
                "beta": float(row.get("active_scale") or 0.0),
                "num_samples": int(row.get("num_samples") or 0),
                "pressured_compliance_delta": float(row.get("wrong_option_follow_rate") or 0.0)
                - float(row.get("wrong_option_follow_rate_reference") or 0.0),
                "baseline_damage_rate": float(row.get("baseline_damage_rate") or 0.0),
                "recovery_delta": float(row.get("intervention_recovery_rate") or 0.0),
                "stance_drift_delta": float(row.get("intervention_error_rate") or 0.0)
                - float(row.get("interference_induced_error_rate") or 0.0),
                "net_recovery_without_damage": float(row.get("net_recovery_without_damage") or 0.0),
            }
        )
    out.sort(key=lambda row: (str(row["layer_config_name"]), float(row["beta"])))
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export qwen-style long CSV from local probe intervention summary.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model-label", default="Qwen-7B")
    parser.add_argument("--output-csv", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _transform_rows(str(args.model_label), _load_rows(Path(args.summary_json)))
    _write_csv(Path(args.output_csv), rows)
    print(json.dumps({"output_csv": str(Path(args.output_csv).resolve()), "num_rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
