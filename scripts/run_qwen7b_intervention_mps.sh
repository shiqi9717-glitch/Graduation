#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_intervention.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file outputs/experiments/local_probe_qwen3b/probe_sample_set_100.json \
  --reference-mechanistic-run-dir outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32/Qwen_Qwen2.5-7B-Instruct/20260422_131858 \
  --output-dir outputs/experiments/local_probe_qwen7b_intervention \
  --device mps \
  --dtype float32 \
  --methods baseline_state_interpolation,late_layer_residual_subtraction \
  --sample-types strict_positive,high_pressure_wrong_option \
  --explicit-layers 24,25,26,27 \
  --subtraction-scale 0.5 \
  --interpolation-scale 0.5 \
  --log-level INFO
