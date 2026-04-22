#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project

./.venv/bin/python scripts/run_local_probe.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file data/samples/local_probe_smoke_sample.json \
  --output-dir outputs/experiments/local_probe_qwen7b_smoke \
  --run-all-scenarios \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --log-level INFO
