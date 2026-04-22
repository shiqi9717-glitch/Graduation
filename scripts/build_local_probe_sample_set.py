#!/usr/bin/env python3
"""Build a small stratified sample set for local white-box probing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.open_model_probe.io_utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small local probe sample set from a CSV file.")
    parser.add_argument("--input-file", required=True, help="CSV file with objective/interference rows.")
    parser.add_argument("--output-file", default="outputs/experiments/local_probe_qwen3b/probe_sample_set.json")
    parser.add_argument("--per-group", type=int, default=8, help="Target samples per group.")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("authority_level", "confidence_level", "explicit_wrong_option", "is_control", "is_hard_negative"):
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    if "sample_id" not in out.columns:
        if "question_id" in out.columns:
            out["sample_id"] = out["question_id"].astype(str)
        else:
            out["sample_id"] = [f"probe-{idx}" for idx in range(len(out))]
    if "question_text" not in out.columns:
        for candidate in ("question", "question_stem", "prompt", "item_text"):
            if candidate in out.columns:
                out["question_text"] = out[candidate]
                break
    if "prompt_prefix" not in out.columns:
        out["prompt_prefix"] = ""
    return out


def _sample_group(df: pd.DataFrame, mask, per_group: int, seed: int, sample_type: str) -> list[dict]:
    subset = df[mask].copy()
    if subset.empty:
        return []
    take = min(per_group, len(subset))
    sampled = subset.sample(n=take, random_state=seed).copy()
    sampled["sample_type"] = sample_type
    columns = [
        "sample_id",
        "question_text",
        "ground_truth",
        "wrong_option",
        "prompt_prefix",
        "sample_type",
        "condition_id",
        "authority_level",
        "confidence_level",
        "explicit_wrong_option",
        "is_control",
        "is_hard_negative",
    ]
    for col in columns:
        if col not in sampled.columns:
            sampled[col] = ""
    return sampled[columns].to_dict(orient="records")


def main() -> int:
    args = build_parser().parse_args()
    df = _normalize(pd.read_csv(args.input_file))

    groups = []
    groups.extend(
        _sample_group(
            df,
            (df["explicit_wrong_option"] == 1) & (df["is_control"] == 0) & (df["is_hard_negative"] == 0),
            args.per_group,
            args.random_seed,
            "strict_positive",
        )
    )
    groups.extend(
        _sample_group(
            df,
            df["is_hard_negative"] == 1,
            args.per_group,
            args.random_seed + 1,
            "hard_negative",
        )
    )
    groups.extend(
        _sample_group(
            df,
            (df["explicit_wrong_option"] == 1)
            & (df["is_control"] == 0)
            & ((df["authority_level"] >= 1) | (df["confidence_level"] >= 1)),
            args.per_group,
            args.random_seed + 2,
            "high_pressure_wrong_option",
        )
    )
    groups.extend(
        _sample_group(
            df,
            df["is_control"] == 1,
            args.per_group,
            args.random_seed + 3,
            "control",
        )
    )

    output_path = Path(args.output_file)
    payload = {
        "input_file": str(Path(args.input_file).resolve()),
        "output_file": str(output_path.resolve()),
        "num_samples": len(groups),
        "per_group": int(args.per_group),
        "samples": groups,
    }
    save_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
