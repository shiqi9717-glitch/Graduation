#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig

echo "Starting identity_profile follow-up: Qwen2.5-7B, eval=36, methods=prefix/matched-early-mid/late"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_identity_profile_whitebox.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype float32 \
  --early-layers 12-16 \
  --late-layers 24-26 \
  --output-root outputs/experiments/identity_profile_whitebox_followup_qwen7b \
  --identity-n 54 \
  --belief-n 0 \
  --train-n-identity 18 \
  --train-n-belief-per-source 0 \
  --alpha-values 0.5,0.75 \
  --max-prefix-positions 32 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO
