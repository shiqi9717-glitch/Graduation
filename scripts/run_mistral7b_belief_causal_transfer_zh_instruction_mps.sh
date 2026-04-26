#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting Mistral-7B belief causal transfer: philpapers -> nlp_survey, total n=48, Chinese instruction variant"
echo "Log file: outputs/logs/mistral7b_belief_causal_transfer_zh_instruction_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant zh_instruction \
  --train-n 24 \
  --eval-n 24 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260426 \
  --output-root outputs/experiments/non_chinese_belief_causal_transfer_mistral7b \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/mistral7b_belief_causal_transfer_zh_instruction_mps.log
