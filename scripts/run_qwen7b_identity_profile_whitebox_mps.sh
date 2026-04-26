#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting identity_profile white-box study: Qwen2.5-7B, early/mid=12-16, late control=24-26"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_identity_profile_whitebox.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --early-layers 12-16 \
  --late-layers 24-26 \
  --output-root outputs/experiments/identity_profile_whitebox_qwen7b \
  --identity-n 36 \
  --belief-n 36 \
  --train-n-identity 18 \
  --train-n-belief-per-source 9 \
  --alpha-values 0.25,0.5,0.75 \
  --max-prefix-positions 32 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
