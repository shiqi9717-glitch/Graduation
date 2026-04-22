#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project

SAMPLE_FILE="outputs/experiments/local_probe_qwen3b/generalization_probe_sample_set_200.json"
mkdir -p .mplconfig

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${SAMPLE_FILE}" \
  --output-dir outputs/experiments/local_probe_qwen7b_mps_fp32_generalization \
  --run-all-scenarios \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --log-level INFO
