#!/usr/bin/env python3
"""Build a larger intervention stability subset from an existing sample set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a larger-sample intervention subset.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--mechanistic-run-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--strict-positive-limit", type=int, default=50)
    parser.add_argument("--high-pressure-limit", type=int, default=50)
    parser.add_argument("--control-limit", type=int, default=8)
    parser.add_argument("--control-transition", default="baseline_correct_to_interference_correct")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    samples = [dict(row) for row in payload.get("samples", [])]
    mech_dir = Path(args.mechanistic_run_dir)
    sample_cases = _load_jsonl(mech_dir / "mechanistic_sample_cases.jsonl")
    transition_by_id = {str(row["sample_id"]): str(row.get("transition_label") or "") for row in sample_cases}

    rng = random.Random(int(args.seed))

    strict_positive = [row for row in samples if str(row.get("sample_type")) == "strict_positive"]
    high_pressure = [row for row in samples if str(row.get("sample_type")) == "high_pressure_wrong_option"]
    controls = [
        row for row in samples
        if str(row.get("sample_type")) == "control"
        and transition_by_id.get(str(row.get("sample_id"))) == str(args.control_transition)
    ]
    rng.shuffle(controls)

    selected = (
        strict_positive[: int(args.strict_positive_limit)]
        + high_pressure[: int(args.high_pressure_limit)]
        + controls[: int(args.control_limit)]
    )
    selected_ids = {str(row.get("sample_id")) for row in selected}

    group_specs = []
    for sample_type in ("strict_positive", "high_pressure_wrong_option", "control"):
        rows = [row for row in selected if str(row.get("sample_type")) == sample_type]
        if not rows:
            continue
        group_specs.append(
            {
                "sample_type": sample_type,
                "count": len(rows),
                "condition_ids": sorted({str(row.get("condition_id") or "") for row in rows}),
            }
        )

    output = {
        "source_file": str(Path(args.input_file).resolve()),
        "mechanistic_run_dir": str(mech_dir.resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_samples": len(selected),
        "strict_positive_limit": int(args.strict_positive_limit),
        "high_pressure_limit": int(args.high_pressure_limit),
        "control_limit": int(args.control_limit),
        "control_transition": str(args.control_transition),
        "seed": int(args.seed),
        "selected_sample_ids": sorted(selected_ids),
        "group_specs": group_specs,
        "samples": selected,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
