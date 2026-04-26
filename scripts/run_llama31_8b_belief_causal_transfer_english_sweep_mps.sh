#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

MODEL_PATH="/Users/shiqi/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

echo "Starting Llama-3.1-8B English belief causal transfer sweep: philpapers -> nlp_survey, k=2, alpha=0.15,0.25,0.35,0.5"
echo "Model path: ${MODEL_PATH}"
echo "Log file: outputs/logs/llama31_8b_belief_causal_transfer_english_sweep_mps.log"

export PYTHONUNBUFFERED=1
MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_belief_causal_transfer.py \
  --model-name "${MODEL_PATH}" \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --prompt-variant english \
  --train-n 24 \
  --eval-n 24 \
  --k 2 \
  --alpha-values 0.15,0.25,0.35,0.5 \
  --seed 20260426 \
  --output-root outputs/experiments/llama31_8b_belief_causal_transfer_english_sweep \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO 2>&1 | tee -a outputs/logs/llama31_8b_belief_causal_transfer_english_sweep_mps.log
