#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting Table 4 exploratory bridge intervention pilot: Qwen2.5-7B, n=36, baseline_state_interpolation, layers=24-26, scale=0.6"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_bridge_intervention_pilot_qwen7b.py \
  --full-run-dir outputs/experiments/bridge_benchmark_qwen7b_full/20260423_205056 \
  --output-root outputs/experiments/bridge_benchmark_qwen7b_intervention_pilot \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --pilot-n 36 \
  --layers 24,25,26 \
  --layer-range 24-26 \
  --scale 0.6 \
  --max-length 1024 \
  --flush-every 6 \
  --log-level INFO
