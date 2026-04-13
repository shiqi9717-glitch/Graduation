#!/usr/bin/env python3
"""Build a stratified objective subset from an existing perturbation JSONL."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _save_jsonl(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _sample_stratified(rows: List[Dict[str, Any]], total_samples: int, seed: int) -> List[Dict[str, Any]]:
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")
    if len(rows) < total_samples:
        raise ValueError(f"Not enough rows in source set: {len(rows)} < {total_samples}")

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category", "Unknown")).strip() or "Unknown"
        by_category.setdefault(category, []).append(row)

    rng = random.Random(seed)
    categories = sorted(by_category.keys())
    if not categories:
        raise ValueError("No categories found in source rows.")

    target_per_category: Dict[str, int] = {}
    remaining = total_samples
    for idx, category in enumerate(categories):
        if idx == len(categories) - 1:
            target = remaining
        else:
            target = total_samples // len(categories)
            remaining -= target
        if len(by_category[category]) < target:
            raise ValueError(
                f"Category {category} does not contain enough rows: {len(by_category[category])} < {target}"
            )
        target_per_category[category] = target

    sampled: List[Dict[str, Any]] = []
    for offset, category in enumerate(categories):
        pool = list(by_category[category])
        local_rng = random.Random(seed + offset)
        local_rng.shuffle(pool)
        sampled.extend(pool[: target_per_category[category]])

    rng.shuffle(sampled)
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified subset from an objective perturbation JSONL.")
    parser.add_argument("--input-file", type=str, required=True, help="Source perturbation JSONL.")
    parser.add_argument("--output-file", type=str, required=True, help="Subset JSONL output path.")
    parser.add_argument("--total-samples", type=int, required=True, help="Subset size to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_jsonl(Path(args.input_file))
    sampled = _sample_stratified(rows, total_samples=args.total_samples, seed=args.seed)
    output = _save_jsonl(sampled, Path(args.output_file))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
