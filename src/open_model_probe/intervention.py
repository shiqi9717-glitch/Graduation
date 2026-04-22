"""Mechanism-informed inference-time intervention helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


TARGET_TRANSITION = "baseline_correct_to_interference_wrong"
SUPPORTED_METHODS = ("late_layer_residual_subtraction", "baseline_state_interpolation")


@dataclass(frozen=True)
class InterventionSpec:
    method: str
    target_layers: tuple[int, ...]
    layer_config_name: str = ""
    subtraction_scale: float = 0.5
    interpolation_scale: float = 0.5
    source_transition: str = TARGET_TRANSITION
    target_sample_types: tuple[str, ...] = ("strict_positive", "high_pressure_wrong_option")

    @property
    def active_scale(self) -> float:
        if self.method == "baseline_state_interpolation":
            return float(self.interpolation_scale)
        return float(self.subtraction_scale)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_mechanistic_scenario_records(mechanistic_run_dir: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    rows = _load_jsonl(Path(mechanistic_run_dir) / "mechanistic_scenario_records.jsonl")
    return {(str(row["sample_id"]), str(row["scenario"])): dict(row) for row in rows}


def load_mechanistic_sample_cases(mechanistic_run_dir: Path) -> List[Dict[str, Any]]:
    return _load_jsonl(Path(mechanistic_run_dir) / "mechanistic_sample_cases.jsonl")


def format_layer_config_name(layers: Sequence[int]) -> str:
    ordered = [int(layer) for layer in layers]
    if not ordered:
        return "none"
    if len(ordered) >= 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return "+".join(str(layer) for layer in ordered)


def select_target_layers(
    *,
    layer_summary_csv: Path,
    sample_types: Sequence[str],
    transition_label: str = TARGET_TRANSITION,
    top_k: int = 3,
    explicit_layers: Sequence[int] | None = None,
) -> List[int]:
    if explicit_layers:
        return [int(layer) for layer in explicit_layers]

    with open(layer_summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    wanted_types = {str(value) for value in sample_types}
    filtered = [
        row for row in rows
        if str(row.get("transition_label")) == str(transition_label)
        and str(row.get("sample_type")) in wanted_types
    ]
    filtered.sort(key=lambda row: float(row.get("mean_interference_margin_delta") or 0.0))

    selected: List[int] = []
    for row in filtered:
        layer = int(row["layer_index"])
        if layer not in selected:
            selected.append(layer)
        if len(selected) >= int(top_k):
            break
    return selected


def load_hidden_array(record: Dict[str, Any], key: str) -> np.ndarray:
    path = Path(str(record["layer_hidden_state_path"]))
    with np.load(path) as data:
        return np.asarray(data[key], dtype=np.float32)


def build_mean_residual_subtraction_directions(
    *,
    sample_cases: Sequence[Dict[str, Any]],
    scenario_record_map: Dict[tuple[str, str], Dict[str, Any]],
    target_layers: Sequence[int],
    sample_types: Sequence[str],
    transition_label: str = TARGET_TRANSITION,
) -> Dict[int, np.ndarray]:
    selected_cases = [
        row for row in sample_cases
        if str(row.get("transition_label")) == str(transition_label)
        and str(row.get("sample_type")) in {str(value) for value in sample_types}
    ]
    directions: Dict[int, np.ndarray] = {}
    for layer_index in target_layers:
        deltas: List[np.ndarray] = []
        array_key = f"layer_{int(layer_index)}_final_token"
        for case in selected_cases:
            sample_id = str(case["sample_id"])
            baseline_record = scenario_record_map.get((sample_id, "baseline"))
            interference_record = scenario_record_map.get((sample_id, "interference"))
            if baseline_record is None or interference_record is None:
                continue
            baseline_tensor = load_hidden_array(baseline_record, array_key)
            interference_tensor = load_hidden_array(interference_record, array_key)
            deltas.append(interference_tensor - baseline_tensor)
        if deltas:
            directions[int(layer_index)] = np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32)
    return directions


def build_layer_patch_map(
    *,
    method: str,
    sample_id: str,
    scenario: str,
    target_layers: Sequence[int],
    scenario_record_map: Dict[tuple[str, str], Dict[str, Any]],
    subtraction_directions: Dict[int, np.ndarray] | None = None,
    subtraction_scale: float = 0.5,
    interpolation_scale: float = 0.5,
) -> Dict[int, np.ndarray]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported intervention method: {method}")

    baseline_record = scenario_record_map.get((str(sample_id), "baseline"))
    interference_record = scenario_record_map.get((str(sample_id), "interference"))
    if baseline_record is None or interference_record is None:
        raise KeyError(f"Missing baseline/interference records for sample_id={sample_id}")

    patch_map: Dict[int, np.ndarray] = {}
    for layer_index in target_layers:
        array_key = f"layer_{int(layer_index)}_final_token"
        baseline_tensor = load_hidden_array(baseline_record, array_key)
        interference_tensor = load_hidden_array(interference_record, array_key)

        if method == "late_layer_residual_subtraction":
            if subtraction_directions is None or int(layer_index) not in subtraction_directions:
                raise KeyError(f"Missing subtraction direction for layer {layer_index}")
            base_tensor = baseline_tensor if str(scenario) == "baseline" else interference_tensor
            patch_map[int(layer_index)] = (
                base_tensor - float(subtraction_scale) * subtraction_directions[int(layer_index)]
            ).astype(np.float32)
        elif method == "baseline_state_interpolation":
            if str(scenario) == "baseline":
                patch_map[int(layer_index)] = baseline_tensor.astype(np.float32)
            else:
                patch_map[int(layer_index)] = (
                    (1.0 - float(interpolation_scale)) * interference_tensor
                    + float(interpolation_scale) * baseline_tensor
                ).astype(np.float32)
    return patch_map


def summarize_intervention_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in records]

    def _mean(values: Iterable[float]) -> float | None:
        clean = [float(value) for value in values]
        if not clean:
            return None
        return float(np.mean(clean))

    def _group_summary(
        subset: List[Dict[str, Any]],
        *,
        method: str,
        sample_type: str | None,
        layer_config_name: str | None,
        active_scale: float | None,
    ) -> Dict[str, Any]:
        baseline_correct_den = [row for row in subset if bool(row.get("baseline_reference_correct"))]
        reference_error_rows = [row for row in baseline_correct_den if row.get("interference_reference_correct") is False]
        baseline_damage_count = sum(
            1 for row in baseline_correct_den if row.get("baseline_intervened_correct") is False
        )
        recovered_count = sum(
            1
            for row in reference_error_rows
            if row.get("interference_intervened_correct") is True
        )
        return {
            "method": method,
            "sample_type": sample_type,
            "layer_config_name": layer_config_name,
            "active_scale": active_scale,
            "num_samples": len(subset),
            "baseline_accuracy": _mean(1.0 if row.get("baseline_reference_correct") else 0.0 for row in subset),
            "interference_accuracy": _mean(1.0 if row.get("interference_reference_correct") else 0.0 for row in subset),
            "intervention_accuracy": _mean(1.0 if row.get("interference_intervened_correct") else 0.0 for row in subset),
            "baseline_accuracy_intervened": _mean(1.0 if row.get("baseline_intervened_correct") else 0.0 for row in subset),
            "interference_induced_error_rate": (
                float(sum(1 for row in reference_error_rows) / len(baseline_correct_den))
                if baseline_correct_den else None
            ),
            "intervention_error_rate": (
                float(sum(1 for row in baseline_correct_den if row.get("interference_intervened_correct") is False) / len(baseline_correct_den))
                if baseline_correct_den else None
            ),
            "intervention_recovery_rate": (
                float(recovered_count / len(reference_error_rows))
                if reference_error_rows else None
            ),
            "wrong_option_follow_rate": _mean(
                1.0 if row.get("interference_intervened_wrong_option_follow") else 0.0 for row in subset
            ),
            "wrong_option_follow_rate_reference": _mean(
                1.0 if row.get("interference_reference_wrong_option_follow") else 0.0 for row in subset
            ),
            "baseline_damage_rate": (
                float(baseline_damage_count / len(baseline_correct_den))
                if baseline_correct_den else None
            ),
            "net_recovery_without_damage": (
                float((recovered_count - baseline_damage_count) / len(baseline_correct_den))
                if baseline_correct_den else None
            ),
            "mean_interference_margin_delta_reference": _mean(
                float(row.get("interference_reference_margin", 0.0) - row.get("baseline_reference_margin", 0.0))
                for row in subset
            ),
            "mean_interference_margin_delta_intervened": _mean(
                float(row.get("interference_intervened_margin", 0.0) - row.get("baseline_reference_margin", 0.0))
                for row in subset
            ),
        }

    summary_rows: List[Dict[str, Any]] = []
    overall_by_setting: List[Dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_subset = [row for row in rows if str(row["method"]) == method]
        layer_config_names = sorted({str(row.get("layer_config_name") or "") for row in method_subset})
        active_scales = sorted({float(row.get("active_scale") or 0.0) for row in method_subset})
        for layer_config_name in layer_config_names:
            for active_scale in active_scales:
                setting_subset = [
                    row for row in method_subset
                    if str(row.get("layer_config_name") or "") == layer_config_name
                    and float(row.get("active_scale") or 0.0) == active_scale
                ]
                if not setting_subset:
                    continue
                overall_by_setting.append(
                    _group_summary(
                        setting_subset,
                        method=method,
                        sample_type=None,
                        layer_config_name=layer_config_name,
                        active_scale=active_scale,
                    )
                )
                for sample_type in sorted({str(row["sample_type"]) for row in setting_subset}):
                    subset = [row for row in setting_subset if str(row["sample_type"]) == sample_type]
                    summary_rows.append(
                        _group_summary(
                            subset,
                            method=method,
                            sample_type=sample_type,
                            layer_config_name=layer_config_name,
                            active_scale=active_scale,
                        )
                    )

    return {
        "num_records": len(rows),
        "overall_by_setting": overall_by_setting,
        "group_summaries": summary_rows,
    }
