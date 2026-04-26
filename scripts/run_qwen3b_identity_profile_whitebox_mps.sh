#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting identity_profile white-box study: Qwen2.5-3B, early/mid=16-20, late control=31-35"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_identity_profile_whitebox.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --device mps \
  --dtype float32 \
  --early-layers 16-20 \
  --late-layers 31-35 \
  --output-root outputs/experiments/identity_profile_whitebox_qwen3b \
  --identity-n 36 \
  --belief-n 36 \
  --train-n-identity 18 \
  --train-n-belief-per-source 9 \
  --alpha-values 0.25,0.5,0.75 \
  --max-prefix-positions 32 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
