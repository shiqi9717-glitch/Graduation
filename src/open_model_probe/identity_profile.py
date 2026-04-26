"""Helpers for the identity_profile white-box study."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_bool(values: Iterable[bool]) -> float | None:
    vals = [1.0 if value else 0.0 for value in values]
    return (sum(vals) / len(vals)) if vals else None


def parse_layers(raw: str) -> tuple[int, ...]:
    layers: List[int] = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        if "-" in text:
            start, end = text.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(text))
    return tuple(dict.fromkeys(layers))


def sample_identity_subset(
    *,
    items: Sequence[Dict[str, Any]],
    identity_n: int,
    belief_n: int,
    train_n_identity: int,
    train_n_belief_per_source: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(int(seed))

    identity_all = [dict(item) for item in items if item.get("pressure_type") == "identity_profile"]
    rng.shuffle(identity_all)
    identity_selected = identity_all[: int(identity_n)]
    identity_train = [dict(row, split="train", study_group="identity_profile") for row in identity_selected[: int(train_n_identity)]]
    identity_eval = [dict(row, split="eval", study_group="identity_profile") for row in identity_selected[int(train_n_identity) :]]

    belief_selected: List[Dict[str, Any]] = []
    for source in ("nlp_survey", "philpapers2020"):
        belief_source = [dict(item) for item in items if item.get("pressure_type") == "belief_argument" and item.get("source") == source]
        rng.shuffle(belief_source)
        belief_selected.extend(belief_source[: int(belief_n) // 2])
    rng.shuffle(belief_selected)
    train_total = min(int(train_n_belief_per_source) * 2, len(belief_selected))
    belief_train = [dict(row, split="train", study_group="belief_argument") for row in belief_selected[:train_total]]
    belief_eval = [dict(row, split="eval", study_group="belief_argument") for row in belief_selected[train_total:]]
    return identity_train, identity_eval, belief_train, belief_eval


def baseline_core_question(item: Dict[str, Any]) -> str:
    prompt = str(item.get("baseline_prompt") or "")
    if "\n\n" in prompt:
        return prompt.split("\n\n", 1)[1]
    return prompt


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int] | None:
    needle = list(subsequence)
    haystack = list(sequence)
    if not needle:
        return None
    limit = len(haystack) - len(needle) + 1
    for start in range(max(limit, 0)):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def prefix_positions_for_prompt(*, tokenizer: Any, prompt: str, core_question: str, max_prefix_positions: int) -> List[int]:
    prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
    core_ids = tokenizer.encode(str(core_question), add_special_tokens=False)
    span = find_subsequence(prompt_ids, core_ids)
    if span is None:
        return list(range(min(int(max_prefix_positions), len(prompt_ids))))
    return list(range(min(int(span[0]), int(max_prefix_positions))))


def matched_nonprefix_positions_for_prompt(*, tokenizer: Any, prompt: str, core_question: str, match_count: int) -> List[int]:
    prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
    core_ids = tokenizer.encode(str(core_question), add_special_tokens=False)
    span = find_subsequence(prompt_ids, core_ids)
    if span is None:
        start = min(int(match_count), len(prompt_ids))
        tail = [idx for idx in range(start, len(prompt_ids) - 1)]
    else:
        _, end = span
        tail = [idx for idx in range(int(end), len(prompt_ids) - 1)]
    if len(tail) >= int(match_count):
        return tail[: int(match_count)]
    fallback = [idx for idx in range(len(prompt_ids) - 1) if idx not in tail]
    combined = tail + fallback
    return combined[: int(match_count)]


def final_token_position(*, tokenizer: Any, prompt: str) -> int:
    prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
    return max(len(prompt_ids) - 1, 0)


def summarize_norm_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))
