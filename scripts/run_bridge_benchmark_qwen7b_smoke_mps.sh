#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting bridge benchmark Qwen2.5-7B smoke: 9 items x 3 scenarios = 27 responses"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_bridge_benchmark_qwen7b.py \
  --mode smoke \
  --output-root outputs/experiments/bridge_benchmark_qwen7b_smoke \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --num-items 9 \
  --max-length 1024 \
  --flush-every 3 \
  --log-level INFO
