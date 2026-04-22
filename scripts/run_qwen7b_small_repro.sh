#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project

SAMPLE_FILE="outputs/experiments/local_probe_qwen7b/probe_sample_set_24.json"

./.venv/bin/python scripts/build_local_probe_subset.py \
  --input-file outputs/experiments/local_probe_qwen3b/probe_sample_set_100.json \
  --output-file "${SAMPLE_FILE}" \
  --per-group 6 \
  --seed 142

./.venv/bin/python scripts/run_local_probe.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${SAMPLE_FILE}" \
  --output-dir outputs/experiments/local_probe_qwen7b \
  --run-all-scenarios \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --log-level INFO

./.venv/bin/python scripts/run_local_probe_mechanistic_analysis.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${SAMPLE_FILE}" \
  --output-dir outputs/experiments/local_probe_qwen7b_mechanistic \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --patching-max-samples 0 \
  --log-level INFO
