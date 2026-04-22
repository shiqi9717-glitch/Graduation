#!/usr/bin/env python3
"""Build a smaller local-probe subset from an existing sample set."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a smaller subset from an existing local probe sample set.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--per-group", type=int, default=6, help="Number of samples per sample_type group.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))
    rng = random.Random(int(args.seed))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("sample_type") or "unknown")].append(dict(sample))

    subset: list[dict] = []
    group_specs: list[dict] = []
    for sample_type in sorted(grouped):
        rows = list(grouped[sample_type])
        rng.shuffle(rows)
        selected = rows[: int(args.per_group)]
        subset.extend(selected)
        if selected:
            group_specs.append(
                {
                    "sample_type": sample_type,
                    "count": len(selected),
                    "condition_ids": sorted({str(row.get("condition_id") or "") for row in selected}),
                }
            )

    out = {
        "source_file": str(Path(args.input_file).resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_samples": len(subset),
        "per_group": int(args.per_group),
        "seed": int(args.seed),
        "group_specs": group_specs,
        "samples": subset,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
