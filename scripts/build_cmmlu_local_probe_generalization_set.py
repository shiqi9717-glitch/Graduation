#!/usr/bin/env python3
"""Build a non-overlapping CMMLU local probe sample set for predictor generalization."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

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
    parser = argparse.ArgumentParser(description="Build a held-out CMMLU local probe sample set.")
    parser.add_argument("--input-file", default="third_party/CMMLU-master/data/test")
    parser.add_argument("--train-dataset-file", required=True, help="Existing predictor training dataset CSV.")
    parser.add_argument("--reference-sample-file", default="outputs/experiments/local_probe_qwen3b/probe_sample_set_100.json")
    parser.add_argument("--output-file", default="outputs/experiments/local_probe_qwen3b/generalization_probe_sample_set.json")
    parser.add_argument("--split-output-file", default="outputs/experiments/local_probe_qwen3b/predictor_generalization_split.csv")
    parser.add_argument("--per-group", type=int, default=24)
    parser.add_argument("--seed", type=int, default=43)
    return parser


def _task_id_from_sample_id(sample_id: str) -> str:
    sample_id = str(sample_id)
    if "__" in sample_id:
        return sample_id.split("__", 1)[0]
    return sample_id


def main() -> int:
    args = build_parser().parse_args()

    train_df = pd.read_csv(Path(args.train_dataset_file))
    train_task_ids = {_task_id_from_sample_id(sample_id) for sample_id in train_df["sample_id"].astype(str)}

    reference_payload = json.loads(Path(args.reference_sample_file).read_text(encoding="utf-8"))
    reference_task_ids = {_task_id_from_sample_id(sample["sample_id"]) for sample in reference_payload.get("samples", [])}

    total_samples = int(args.per_group) * len(GROUP_SPECS)
    records = CMMLUObjectivePerturber.load_and_sample(
        input_file=Path(args.input_file),
        total_samples=max(total_samples * 3, 400),
        seed=int(args.seed),
    )
    candidate_records = [
        record for record in records
        if str(record.get("task_id")) not in train_task_ids
        and str(record.get("task_id")) not in reference_task_ids
    ]

    rng = random.Random(int(args.seed))
    rng.shuffle(candidate_records)
    if len(candidate_records) < total_samples:
        raise ValueError(f"Not enough held-out CMMLU candidates after exclusion: need {total_samples}, got {len(candidate_records)}.")

    samples = []
    for idx, (sample_type, condition_id) in enumerate(
        [spec for spec in GROUP_SPECS for _ in range(int(args.per_group))]
    ):
        record = candidate_records[idx]
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
        "train_dataset_file": str(Path(args.train_dataset_file).resolve()),
        "reference_sample_file": str(Path(args.reference_sample_file).resolve()),
        "output_file": str(Path(args.output_file).resolve()),
        "num_samples": len(samples),
        "per_group": int(args.per_group),
        "excluded_train_task_ids": len(train_task_ids),
        "excluded_reference_task_ids": len(reference_task_ids),
        "group_specs": [
            {"sample_type": sample_type, "condition_id": condition_id}
            for sample_type, condition_id in GROUP_SPECS
        ],
        "samples": samples,
    }
    save_json(Path(args.output_file), payload)

    split_rows = []
    for sample_id in sorted(train_df["sample_id"].astype(str).tolist()):
        split_rows.append({"sample_id": sample_id, "task_id": _task_id_from_sample_id(sample_id), "split": "train"})
    for sample in samples:
        split_rows.append({"sample_id": sample["sample_id"], "task_id": sample["task_id"], "split": "generalization"})
    pd.DataFrame(split_rows).to_csv(Path(args.split_output_file), index=False)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
