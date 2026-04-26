#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p outputs/logs

echo "Starting DeepSeek V4 Flash THINKING pressure smoke: 20 items x 4 conditions"
echo "Log file: outputs/logs/deepseek_v4_flash_thinking_pressure_smoke.log"

export PYTHONUNBUFFERED=1
./.venv/bin/python scripts/run_deepseek_v4_pressure_benchmark.py \
  --model deepseek-v4-flash \
  --raw-input-file third_party/CMMLU-master/data/test \
  --num-items 20 \
  --conditions baseline,strict_positive,high_pressure_wrong_option,recovery \
  --seed 20260426 \
  --output-root outputs/experiments/deepseek_v4_flash_thinking_pressure_smoke \
  --concurrency 2 \
  --max-tokens 1024 \
  --thinking enabled \
  --reasoning-effort medium \
  --log-level INFO 2>&1 | tee -a outputs/logs/deepseek_v4_flash_thinking_pressure_smoke.log
