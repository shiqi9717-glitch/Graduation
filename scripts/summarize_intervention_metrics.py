#!/usr/bin/env python3
"""Print compact intervention metrics from an intervention summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize key intervention metrics from intervention_summary.json.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--sample-types", default="strict_positive,high_pressure_wrong_option,control")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sample_types = {item.strip() for item in str(args.sample_types).split(",") if item.strip()}
    payload = json.loads(Path(args.summary_file).read_text(encoding="utf-8"))
    rows = [
        row for row in payload.get("group_summaries", [])
        if row.get("sample_type") in sample_types
    ]
    rows.sort(key=lambda row: (
        str(row.get("method") or ""),
        str(row.get("layer_config_name") or ""),
        float(row.get("active_scale") or 0.0),
        str(row.get("sample_type") or ""),
    ))
    compact = [
        {
            "method": row.get("method"),
            "layer_config_name": row.get("layer_config_name"),
            "active_scale": row.get("active_scale"),
            "sample_type": row.get("sample_type"),
            "intervention_recovery_rate": row.get("intervention_recovery_rate"),
            "wrong_option_follow_rate": row.get("wrong_option_follow_rate"),
            "baseline_damage_rate": row.get("baseline_damage_rate"),
            "net_recovery_without_damage": row.get("net_recovery_without_damage"),
        }
        for row in rows
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
