#!/usr/bin/env python3
"""Build a held-out objective-local subset that excludes the frozen mainline IDs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.local_data_perturber import CMMLUObjectivePerturber
from src.open_model_probe.io_utils import save_json


GROUP_SPECS = (
    ("control", "ctrl_base"),
    ("strict_positive", "a1_c1_w1"),
    ("high_pressure_wrong_option", "a2_c1_w1"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a held-out objective-local subset from raw CMMLU.")
    parser.add_argument("--input-file", default="third_party/CMMLU-master/data/test")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--strict-positive-count", type=int, default=146)
    parser.add_argument("--high-pressure-count", type=int, default=146)
    parser.add_argument("--control-count", type=int, default=8)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=6)
    parser.add_argument("--exclude-sample-files", default="")
    parser.add_argument("--seed", type=int, default=20260427)
    return parser


def _parse_exclude_files(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def _load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples", payload if isinstance(payload, list) else [])
        for row in samples:
            if isinstance(row, dict) and row.get("sample_id"):
                excluded.add(str(row["sample_id"]))
    return excluded


def _sample_from_records(
    *,
    records: list[dict],
    excluded_ids: set[str],
    strict_positive_count: int,
    high_pressure_count: int,
    control_count: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(int(seed))
    per_group_rows: dict[str, list[dict]] = {name: [] for name, _ in GROUP_SPECS}

    for record in records:
        for sample_type, condition_id in GROUP_SPECS:
            sample_id = f"{record['task_id']}__{condition_id}"
            if sample_id in excluded_ids:
                continue
            meta = dict(record[f"{condition_id}_meta"])
            per_group_rows[sample_type].append(
                {
                    "sample_id": sample_id,
                    "task_id": record["task_id"],
                    "question_text": record["question_text"],
                    "ground_truth": record["ground_truth"],
                    "wrong_option": record["perturbed_wrong_answer"],
                    "prompt_prefix": meta.get("prompt_prefix", ""),
                    "sample_type": sample_type,
                    "condition_id": condition_id,
                    "subject": record.get("subject"),
                    "category": record.get("category"),
                    "authority_level": meta.get("authority_level"),
                    "confidence_level": meta.get("confidence_level"),
                    "explicit_wrong_option": int(meta.get("explicit_wrong_option", 0) or 0),
                    "is_control": int(bool(meta.get("is_control", False))),
                    "is_hard_negative": 0,
                }
            )

    for rows in per_group_rows.values():
        rng.shuffle(rows)

    selected = (
        per_group_rows["strict_positive"][: int(strict_positive_count)]
        + per_group_rows["high_pressure_wrong_option"][: int(high_pressure_count)]
        + per_group_rows["control"][: int(control_count)]
    )
    return selected


def main() -> int:
    args = build_parser().parse_args()
    exclude_files = _parse_exclude_files(args.exclude_sample_files)
    excluded_ids = _load_excluded_ids(exclude_files)
    target_total = int(args.strict_positive_count) + int(args.high_pressure_count) + int(args.control_count)

    selected: list[dict] = []
    pool_total = max(target_total * int(args.candidate_pool_multiplier), target_total + 200)
    for multiplier in range(1, 6):
        records = CMMLUObjectivePerturber.load_and_sample(
            input_file=Path(args.input_file),
            total_samples=pool_total * multiplier,
            seed=int(args.seed),
        )
        selected = _sample_from_records(
            records=records,
            excluded_ids=excluded_ids,
            strict_positive_count=int(args.strict_positive_count),
            high_pressure_count=int(args.high_pressure_count),
            control_count=int(args.control_count),
            seed=int(args.seed),
        )
        if len(selected) >= target_total:
            break
    if len(selected) < target_total:
        raise ValueError(
            f"Could only build {len(selected)} held-out samples; target was {target_total}. "
            "Increase candidate-pool-multiplier or reduce requested counts."
        )

    payload = {
        "input_file": str(Path(args.input_file).resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_samples": len(selected),
        "strict_positive_count": int(args.strict_positive_count),
        "high_pressure_count": int(args.high_pressure_count),
        "control_count": int(args.control_count),
        "excluded_sample_files": [str(path.resolve()) for path in exclude_files],
        "num_excluded_sample_ids": len(excluded_ids),
        "seed": int(args.seed),
        "sampling_rule": "raw_cmmlu_resample_excluding_frozen_mainline_ids",
        "group_specs": [
            {"sample_type": sample_type, "condition_id": condition_id}
            for sample_type, condition_id in GROUP_SPECS
        ],
        "samples": selected,
    }
    save_json(Path(args.output_file), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
