#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig
mkdir -p outputs/logs

echo "Starting minimal 14B belief causal transfer: philpapers -> nlp_survey, k=2, alpha=0.75"
echo "Log file: outputs/logs/qwen14b_belief_causal_transfer_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_qwen14b_belief_causal_transfer.py \
  --model-name /Users/shiqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --device mps \
  --dtype auto \
  --layers 40-47 \
  --train-n 24 \
  --eval-n 24 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260424 \
  --output-root outputs/experiments/qwen14b_belief_causal_transfer \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/qwen14b_belief_causal_transfer_mps.log
