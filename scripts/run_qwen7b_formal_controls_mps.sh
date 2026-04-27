#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting Qwen7B formal white-box controls: random-direction + shuffled-label"
echo "Log file: outputs/logs/qwen7b_formal_controls_mps.log"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_whitebox_formal_controls.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file outputs/experiments/local_probe_qwen7b_intervention_main_inputs/qwen7b_intervention_main_sample_set.json \
  --reference-mechanistic-run-dir outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717 \
  --focus-layers 24,25,26 \
  --beta 0.6 \
  --methods random_direction_control,shuffled_label_control \
  --seeds 20260427,20260428,20260429,20260430,20260431 \
  --device mps \
  --dtype float32 \
  --output-root outputs/experiments/whitebox_formal_controls_qwen7b \
  --bootstrap-iters 2000 \
  --bootstrap-seed 20260427 \
  --log-level INFO 2>&1 | tee -a outputs/logs/qwen7b_formal_controls_mps.log
