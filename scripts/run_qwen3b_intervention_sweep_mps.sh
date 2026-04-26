#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

./.venv/bin/python scripts/build_local_probe_intervention_subset.py \
  --input-file outputs/experiments/local_probe_qwen3b/generalization_probe_sample_set_200.json \
  --mechanistic-run-dir outputs/experiments/local_probe_qwen3b_mechanistic/Qwen_Qwen2.5-3B-Instruct/20260419_131943 \
  --output-file outputs/experiments/local_probe_qwen3b_intervention_stability_inputs/qwen3b_intervention_stability_sample_set.json \
  --strict-positive-limit 50 \
  --high-pressure-limit 50 \
  --control-limit 8

echo "Starting 3B intervention sweep: 108 samples x 24 settings = 2592 patched evaluations"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_intervention.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --sample-file outputs/experiments/local_probe_qwen3b_intervention_stability_inputs/qwen3b_intervention_stability_sample_set.json \
  --reference-mechanistic-run-dir outputs/experiments/local_probe_qwen3b_mechanistic/Qwen_Qwen2.5-3B-Instruct/20260419_131943 \
  --output-dir outputs/experiments/local_probe_qwen3b_intervention_stability \
  --device mps \
  --dtype float16 \
  --methods baseline_state_interpolation,late_layer_residual_subtraction \
  --sample-types strict_positive,high_pressure_wrong_option,control \
  --direction-sample-types strict_positive,high_pressure_wrong_option \
  --layer-configs '31-35=31,32,33,34,35;33-35=33,34,35;31+33+35=31,33,35' \
  --interpolation-scales 0.15,0.3,0.45,0.6 \
  --subtraction-scales 0.15,0.3,0.45,0.6 \
  --flush-every 10 \
  --log-level INFO
