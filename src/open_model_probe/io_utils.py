"""I/O helpers for local open-model probing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("._") or "item"


def load_sample_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return [dict(item) for item in payload["samples"]]
        return [payload]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"Unsupported sample file payload: {path}")


def prepare_output_dir(base_dir: Path, run_name: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        output_dir = base_dir / sanitize_filename(run_name) / timestamp
    else:
        output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_hidden_state_arrays(
    output_dir: Path,
    sample_id: str,
    scenario: str,
    arrays: Dict[str, np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sanitize_filename(sample_id)}__{sanitize_filename(scenario)}_hidden_states.npz"
    np.savez_compressed(path, **arrays)
    return path
