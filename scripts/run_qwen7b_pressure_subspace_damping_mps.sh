#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting exploratory mixed-strata pressure_subspace_damping sweep: Qwen2.5-7B, layers=24-26"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_pressure_subspace_damping.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --layers 24-26 \
  --output-root outputs/experiments/pressure_subspace_damping_qwen7b \
  --items-per-stratum 48 \
  --train-per-stratum 24 \
  --seed 20260423 \
  --k-values 1,2,4,8 \
  --intervention-k-values 1,2,4 \
  --alpha-values 0.25,0.5,0.75 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
