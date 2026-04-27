import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.open_model_probe.intervention import (
    build_layer_patch_map,
    build_mean_residual_subtraction_directions,
    format_layer_config_name,
    load_mechanistic_sample_cases,
    load_mechanistic_scenario_records,
    select_target_layers,
    summarize_intervention_records,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_target_layers_prefers_most_negative_rows(tmp_path: Path):
    path = tmp_path / "layer_summary.csv"
    path.write_text(
        "\n".join(
            [
                "transition_label,sample_type,layer_index,num_samples,mean_interference_margin_delta",
                "baseline_correct_to_interference_wrong,strict_positive,24,7,-8.0",
                "baseline_correct_to_interference_wrong,high_pressure_wrong_option,25,2,-9.0",
                "baseline_correct_to_interference_wrong,strict_positive,26,7,-7.5",
                "baseline_correct_to_interference_correct,strict_positive,27,3,-20.0",
            ]
        ),
        encoding="utf-8",
    )
    assert select_target_layers(
        layer_summary_csv=path,
        sample_types=("strict_positive", "high_pressure_wrong_option"),
        top_k=2,
    ) == [25, 24]


def test_direction_building_and_patch_maps(tmp_path: Path):
    baseline_npz = tmp_path / "baseline.npz"
    interference_npz = tmp_path / "interference.npz"
    shuffled_npz = tmp_path / "shuffled_baseline.npz"
    np.savez_compressed(baseline_npz, layer_24_final_token=np.asarray([1.0, 2.0], dtype=np.float32))
    np.savez_compressed(interference_npz, layer_24_final_token=np.asarray([3.0, 6.0], dtype=np.float32))
    np.savez_compressed(shuffled_npz, layer_24_final_token=np.asarray([9.0, 10.0], dtype=np.float32))

    rows = [
        {
            "sample_id": "s1",
            "scenario": "baseline",
            "layer_hidden_state_path": str(baseline_npz),
        },
        {
            "sample_id": "s1",
            "scenario": "interference",
            "layer_hidden_state_path": str(interference_npz),
        },
    ]
    cases = [
        {
            "sample_id": "s1",
            "sample_type": "strict_positive",
            "transition_label": "baseline_correct_to_interference_wrong",
        }
    ]
    jsonl_dir = tmp_path / "mech"
    _write_jsonl(jsonl_dir / "mechanistic_scenario_records.jsonl", rows)
    _write_jsonl(jsonl_dir / "mechanistic_sample_cases.jsonl", cases)

    record_map = load_mechanistic_scenario_records(jsonl_dir)
    sample_cases = load_mechanistic_sample_cases(jsonl_dir)
    directions = build_mean_residual_subtraction_directions(
        sample_cases=sample_cases,
        scenario_record_map=record_map,
        target_layers=(24,),
        sample_types=("strict_positive",),
    )
    assert np.allclose(directions[24], np.asarray([2.0, 4.0], dtype=np.float32))

    subtraction_patch = build_layer_patch_map(
        method="late_layer_residual_subtraction",
        sample_id="s1",
        scenario="interference",
        target_layers=(24,),
        scenario_record_map=record_map,
        subtraction_directions=directions,
        subtraction_scale=0.5,
        interpolation_scale=0.5,
    )
    assert np.allclose(subtraction_patch[24], np.asarray([2.0, 4.0], dtype=np.float32))

    interpolation_patch = build_layer_patch_map(
        method="baseline_state_interpolation",
        sample_id="s1",
        scenario="interference",
        target_layers=(24,),
        scenario_record_map=record_map,
        subtraction_directions=directions,
        subtraction_scale=0.5,
        interpolation_scale=0.25,
    )
    assert np.allclose(interpolation_patch[24], np.asarray([2.5, 5.0], dtype=np.float32))

    shuffled_patch = build_layer_patch_map(
        method="shuffled_label_control",
        sample_id="s1",
        scenario="interference",
        target_layers=(24,),
        scenario_record_map=record_map,
        interpolation_scale=0.25,
        shuffled_baseline_record_map={
            "s1": {
                "sample_id": "s2",
                "scenario": "baseline",
                "layer_hidden_state_path": str(shuffled_npz),
            }
        },
    )
    assert np.allclose(shuffled_patch[24], np.asarray([4.5, 7.0], dtype=np.float32))

    random_patch_a = build_layer_patch_map(
        method="random_direction_control",
        sample_id="s1",
        scenario="interference",
        target_layers=(24,),
        scenario_record_map=record_map,
        interpolation_scale=0.25,
        random_seed=7,
    )
    random_patch_b = build_layer_patch_map(
        method="random_direction_control",
        sample_id="s1",
        scenario="interference",
        target_layers=(24,),
        scenario_record_map=record_map,
        interpolation_scale=0.25,
        random_seed=7,
    )
    assert np.allclose(random_patch_a[24], random_patch_b[24])
    assert not np.allclose(random_patch_a[24], interpolation_patch[24])


def test_format_layer_config_name_supports_ranges_and_sparse_sets():
    assert format_layer_config_name((31, 32, 33, 34, 35)) == "31-35"
    assert format_layer_config_name((31, 33, 35)) == "31+33+35"


def test_intervention_summary_reports_recovery_and_baseline_damage():
    summary = summarize_intervention_records(
        [
            {
                "method": "late_layer_residual_subtraction",
                "layer_config_name": "24-27",
                "active_scale": 0.3,
                "sample_type": "strict_positive",
                "baseline_reference_correct": True,
                "interference_reference_correct": False,
                "baseline_intervened_correct": True,
                "interference_intervened_correct": True,
                "interference_reference_wrong_option_follow": True,
                "interference_intervened_wrong_option_follow": False,
                "baseline_reference_margin": 1.0,
                "interference_reference_margin": -2.0,
                "interference_intervened_margin": 0.5,
            },
            {
                "method": "late_layer_residual_subtraction",
                "layer_config_name": "24-27",
                "active_scale": 0.3,
                "sample_type": "strict_positive",
                "baseline_reference_correct": True,
                "interference_reference_correct": True,
                "baseline_intervened_correct": False,
                "interference_intervened_correct": True,
                "interference_reference_wrong_option_follow": False,
                "interference_intervened_wrong_option_follow": False,
                "baseline_reference_margin": 2.0,
                "interference_reference_margin": 1.0,
                "interference_intervened_margin": 0.8,
            },
        ]
    )
    row = summary["group_summaries"][0]
    overall = summary["overall_by_setting"][0]
    assert row["interference_induced_error_rate"] == 0.5
    assert row["intervention_error_rate"] == 0.0
    assert row["intervention_recovery_rate"] == 1.0
    assert row["baseline_damage_rate"] == 0.5
    assert row["net_recovery_without_damage"] == 0.0
    assert overall["layer_config_name"] == "24-27"
    assert overall["active_scale"] == 0.3
