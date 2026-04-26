#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting bridge benchmark Qwen2.5-7B full sampled run: 144 items x 3 scenarios = 432 responses"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_bridge_benchmark_qwen7b.py \
  --mode full \
  --output-root outputs/experiments/bridge_benchmark_qwen7b_full \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --num-items 144 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
