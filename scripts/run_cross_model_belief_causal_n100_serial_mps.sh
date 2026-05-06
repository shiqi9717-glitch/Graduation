#!/bin/zsh
set -euo pipefail

ROOT="/Users/shiqi/code/graduation-project"
PY="$ROOT/.venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/cross_model_belief_causal_n100/$STAMP"

mkdir -p "$LOG_DIR"

echo "Serial cross-model belief causal transfer n=100 run"
echo "Root: $ROOT"
echo "Log dir: $LOG_DIR"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

cd "$ROOT"

echo "[1/3] Llama-3.1-8B-Instruct"
"$PY" scripts/run_belief_causal_transfer.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device mps \
  --dtype auto \
  --layers 24-31 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 100 \
  --eval-n 100 \
  --k 2 \
  --alpha 0.5 \
  --seed 20260504 \
  --output-root outputs/experiments/llama31_8b_belief_causal_transfer_n100 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO | tee "$LOG_DIR/llama31_8b_n100.log"

echo "[2/3] GLM-4-9B"
"$PY" scripts/run_belief_causal_transfer.py \
  --model-name THUDM/glm-4-9b-chat-hf \
  --device mps \
  --dtype auto \
  --layers 30-33 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 100 \
  --eval-n 100 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260504 \
  --output-root outputs/experiments/glm4_9b_belief_causal_transfer_n100 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO | tee "$LOG_DIR/glm4_9b_n100.log"

echo "[3/3] Qwen2.5-14B-Instruct"
"$PY" scripts/run_belief_causal_transfer.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --device mps \
  --dtype auto \
  --layers 40-47 \
  --train-source philpapers2020 \
  --eval-source nlp_survey \
  --pressure-type belief_argument \
  --train-n 100 \
  --eval-n 100 \
  --k 2 \
  --alpha 0.75 \
  --seed 20260504 \
  --output-root outputs/experiments/qwen14b_belief_causal_transfer_n100 \
  --max-length 1024 \
  --flush-every 12 \
  --log-level INFO | tee "$LOG_DIR/qwen14b_n100.log"

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs saved to: $LOG_DIR"
