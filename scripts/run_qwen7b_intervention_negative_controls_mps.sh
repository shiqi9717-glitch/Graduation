#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs

echo "Starting Qwen7B white-box intervention negative controls: random direction, shuffled label, and minimal layer controls"
echo "Mainline fixed: baseline_state_interpolation, target_late=24-26, scale=0.6"
echo "Log file: outputs/logs/qwen7b_intervention_negative_controls_mps.log"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_intervention.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file outputs/experiments/local_probe_qwen7b_intervention_main_inputs/qwen7b_intervention_main_sample_set.json \
  --reference-mechanistic-run-dir outputs/experiments/local_probe_qwen7b_mechanistic_mps_fp32_generalization/Qwen_Qwen2.5-7B-Instruct/20260422_134717 \
  --output-dir outputs/experiments/local_probe_qwen7b_intervention_negative_controls \
  --device mps \
  --dtype float32 \
  --methods baseline_state_interpolation,random_direction_control,shuffled_label_control \
  --sample-types strict_positive,high_pressure_wrong_option,control \
  --direction-sample-types strict_positive,high_pressure_wrong_option \
  --layer-configs 'early_mid=12,13,14,15,16;target_late=24,25,26;later_control=27' \
  --interpolation-scales 0.6 \
  --control-random-seed 20260426 \
  --flush-every 10 \
  --log-level INFO 2>&1 | tee -a outputs/logs/qwen7b_intervention_negative_controls_mps.log
