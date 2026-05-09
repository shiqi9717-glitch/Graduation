#!/usr/bin/env python3
"""Select a contiguous 4-8 layer screening window from a subspace summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _score_row(row: Dict[str, Any]) -> float:
    explained = float(row.get("explained_variance_sum") or 0.0)
    coherence = float(row.get("direction_abs_coherence") or 0.0)
    return explained + 0.1 * coherence


def _best_window(rows: List[Dict[str, Any]], min_width: int, max_width: int) -> Tuple[List[int], Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            layer = int(row["layer"])
        except Exception:
            continue
        by_layer[layer] = dict(row)
    layers = sorted(by_layer)
    if not layers:
        raise ValueError("No layers found in subspace summary.")

    best_layers: List[int] = []
    best_meta: Dict[str, Any] = {}
    best_score = float("-inf")
    for width in range(int(min_width), int(max_width) + 1):
        for idx in range(0, len(layers) - width + 1):
            window = layers[idx : idx + width]
            if window != list(range(window[0], window[-1] + 1)):
                continue
            score = sum(_score_row(by_layer[layer]) for layer in window)
            if score > best_score:
                best_score = score
                best_layers = list(window)
                best_meta = {
                    "width": int(width),
                    "score": float(score),
                    "peak_layer": int(max(window, key=lambda layer: _score_row(by_layer[layer]))),
                }
    if not best_layers:
        raise ValueError("Could not find a contiguous window in the requested width range.")
    return best_layers, best_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a contiguous 4-8 layer screening window.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--min-width", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _read_rows(Path(args.summary_csv))
    best_layers, meta = _best_window(rows, int(args.min_width), int(args.max_width))
    payload = {
        "selected_layers": best_layers,
        "selected_window": f"{best_layers[0]}-{best_layers[-1]}",
        **meta,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
