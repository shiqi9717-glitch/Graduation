#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting Qwen2.5-7B layer-wise clean subspace diagnostic: train-only philpapers2020 belief_argument, layers=20-27, k=2"
echo "Log file: outputs/logs/qwen7b_layer_wise_subspace_clean_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --device mps \
  --dtype auto \
  --layers 20-27 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 48 \
  --eval-n 0 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260501 \
  --output-root outputs/experiments/qwen7b_layer_wise_subspace \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/qwen7b_layer_wise_subspace_clean_mps.log
