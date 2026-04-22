#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

./.venv/bin/python scripts/build_local_probe_intervention_subset.py \
  --input-file outputs/experiments/local_probe_qwen3b/generalization_probe_sample_set_200.json \
  --mechanistic-run-dir outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717 \
  --output-file outputs/experiments/local_probe_qwen7b_intervention_stability_inputs/qwen7b_intervention_stability_sample_set.json \
  --strict-positive-limit 50 \
  --high-pressure-limit 50 \
  --control-limit 8

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_intervention.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file outputs/experiments/local_probe_qwen7b_intervention_stability_inputs/qwen7b_intervention_stability_sample_set.json \
  --reference-mechanistic-run-dir outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717 \
  --output-dir outputs/experiments/local_probe_qwen7b_intervention_stability \
  --device mps \
  --dtype float32 \
  --methods baseline_state_interpolation,late_layer_residual_subtraction \
  --sample-types strict_positive,high_pressure_wrong_option,control \
  --direction-sample-types strict_positive,high_pressure_wrong_option \
  --layer-configs '24-27=24,25,26,27;24-26=24,25,26;24+25+26=24,25,26' \
  --interpolation-scales 0.15,0.3,0.45,0.6 \
  --subtraction-scales 0.15,0.3,0.45,0.6 \
  --log-level INFO
