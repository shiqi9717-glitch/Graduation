#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting GLM-4-9B belief causal transfer: philpapers -> nlp_survey, k=2, alpha=0.75"
echo "Log file: outputs/logs/glm4_9b_belief_causal_transfer_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name /Users/shiqi/.cache/huggingface/hub/models--zai-org--glm-4-9b-chat-hf/snapshots/8599336fc6c125203efb2360bfaf4c80eef1d1bf \
  --device mps \
  --dtype auto \
  --layers 32-39 \
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
  --log-level INFO 2>&1 | tee -a outputs/logs/glm4_9b_belief_causal_transfer_mps.log
