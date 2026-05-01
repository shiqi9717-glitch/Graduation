#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting clean belief causal transfer: Qwen2.5-7B, philpapers2020 -> nlp_survey, belief_argument, layers=24-26, k=2, alpha=0.75"
echo "Log file: outputs/logs/qwen7b_belief_causal_transfer_clean_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype auto \
  --layers 24-26 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 48 \
  --eval-n 48 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260430 \
  --output-root outputs/experiments/pressure_subspace_damping_qwen7b_clean \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/qwen7b_belief_causal_transfer_clean_mps.log
