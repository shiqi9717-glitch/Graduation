"""Pressure-subspace estimation and damping helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


SUBSPACE_NAMES = (
    "belief_argument_subspace",
    "identity_profile_subspace",
    "mixed_subspace",
    "philpapers_belief_argument_subspace",
)


def sanitize_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_") or "value"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def validate_vector(value: np.ndarray, *, item_id: str, scenario: str, layer: int) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{item_id}/{scenario}/layer_{layer}: hidden state is not an ndarray.")
    if value.ndim != 1:
        raise ValueError(f"{item_id}/{scenario}/layer_{layer}: expected 1D final-token vector, got shape {value.shape}.")
    if value.size == 0:
        raise ValueError(f"{item_id}/{scenario}/layer_{layer}: empty hidden-state vector.")
    if not np.isfinite(value).all():
        raise ValueError(f"{item_id}/{scenario}/layer_{layer}: hidden-state vector contains non-finite values.")


def estimate_pca_components(delta_matrix: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    if delta_matrix.ndim != 2 or delta_matrix.shape[0] == 0:
        raise ValueError("PCA requires a non-empty 2D delta matrix.")
    centered = delta_matrix - delta_matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(int(max_k), vt.shape[0])].astype(np.float32)
    variance = singular_values.astype(np.float64) ** 2
    total = float(variance.sum())
    explained = (variance / total) if total > 0 else np.zeros_like(variance)
    return components, explained[: components.shape[0]].astype(np.float32)


def project_onto_subspace(vector: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    basis = components[: int(k)]
    if basis.size == 0:
        return np.zeros_like(vector, dtype=np.float32)
    return (basis.T @ (basis @ vector.astype(np.float32))).astype(np.float32)


def damp_hidden_state(
    *,
    hidden_state: np.ndarray,
    mean_baseline: np.ndarray,
    components: np.ndarray,
    alpha: float,
    k: int,
) -> np.ndarray:
    centered = hidden_state.astype(np.float32) - mean_baseline.astype(np.float32)
    projection = project_onto_subspace(centered, components.astype(np.float32), int(k))
    return (hidden_state.astype(np.float32) - float(alpha) * projection).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class SubspaceSpec:
    name: str
    pressure_type: str
    source: str


def default_subspace_specs() -> tuple[SubspaceSpec, ...]:
    return (
        SubspaceSpec("belief_argument_subspace", "belief_argument", ""),
        SubspaceSpec("identity_profile_subspace", "identity_profile", ""),
        SubspaceSpec("mixed_subspace", "", ""),
        SubspaceSpec("philpapers_belief_argument_subspace", "belief_argument", "philpapers2020"),
    )


def default_transfer_tests() -> tuple[Dict[str, str], ...]:
    return (
        {
            "transfer_name": "philpapers_belief_argument_to_nlp_survey_belief_argument",
            "subspace_name": "philpapers_belief_argument_subspace",
            "target_pressure_type": "belief_argument",
            "target_source": "nlp_survey",
        },
        {
            "transfer_name": "belief_argument_to_identity_profile",
            "subspace_name": "belief_argument_subspace",
            "target_pressure_type": "identity_profile",
            "target_source": "",
        },
        {
            "transfer_name": "identity_profile_to_belief_argument",
            "subspace_name": "identity_profile_subspace",
            "target_pressure_type": "belief_argument",
            "target_source": "",
        },
        {
            "transfer_name": "mixed_subspace_to_all",
            "subspace_name": "mixed_subspace",
            "target_pressure_type": "",
            "target_source": "",
        },
    )

