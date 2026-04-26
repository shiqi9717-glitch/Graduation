#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting Llama-3.1-8B belief causal transfer: philpapers -> nlp_survey, k=2, alpha=0.75"
echo "Log file: outputs/logs/llama31_8b_belief_causal_transfer_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 24 \
  --eval-n 24 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260425 \
  --output-root outputs/experiments/non_qwen_belief_causal_transfer \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/llama31_8b_belief_causal_transfer_mps.log
