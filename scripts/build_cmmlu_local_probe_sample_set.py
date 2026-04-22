#!/usr/bin/env python3
"""Build a small CMMLU-based local probe sample set with fixed condition groups."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.local_data_perturber import CMMLUObjectivePerturber
from src.open_model_probe.io_utils import save_json


GROUP_SPECS = (
    ("control", "ctrl_base"),
    ("strict_positive", "a1_c1_w1"),
    ("high_pressure_wrong_option", "a2_c1_w1"),
    ("hard_negative_proxy", "a0_c0_w1"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CMMLU-backed local white-box probe sample set.")
    parser.add_argument("--input-file", default="third_party/CMMLU-master/data/test")
    parser.add_argument("--output-file", default="outputs/experiments/local_probe_qwen3b/probe_sample_set.json")
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    total_samples = int(args.per_group) * len(GROUP_SPECS)
    records = CMMLUObjectivePerturber.load_and_sample(
        input_file=Path(args.input_file),
        total_samples=total_samples,
        seed=int(args.seed),
    )

    rng = random.Random(int(args.seed))
    shuffled = list(records)
    rng.shuffle(shuffled)

    samples = []
    for idx, (sample_type, condition_id) in enumerate(
        [spec for spec in GROUP_SPECS for _ in range(int(args.per_group))]
    ):
        record = shuffled[idx]
        meta = dict(record[f"{condition_id}_meta"])
        sample = {
            "sample_id": f"{record['task_id']}__{condition_id}",
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
            "is_hard_negative": int(sample_type == "hard_negative_proxy"),
        }
        samples.append(sample)

    payload = {
        "input_file": str(Path(args.input_file).resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_samples": len(samples),
        "per_group": int(args.per_group),
        "group_specs": [
            {"sample_type": sample_type, "condition_id": condition_id}
            for sample_type, condition_id in GROUP_SPECS
        ],
        "samples": samples,
    }
    save_json(Path(args.output_file), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
