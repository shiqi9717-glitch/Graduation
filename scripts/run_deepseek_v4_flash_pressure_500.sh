#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p outputs/logs

echo "Starting DeepSeek V4 Flash pressure split supplement: 250 non-thinking + 250 thinking"
echo "Log file: outputs/logs/deepseek_v4_flash_pressure_500.log"

export PYTHONUNBUFFERED=1
./.venv/bin/python scripts/run_deepseek_v4_pressure_benchmark.py \
  --model deepseek-v4-flash \
  --raw-input-file third_party/CMMLU-master/data/test \
  --num-items 250 \
  --sample-pool-size 500 \
  --sample-offset 0 \
  --conditions baseline,strict_positive,high_pressure_wrong_option,recovery \
  --seed 20260426 \
  --output-root outputs/experiments/deepseek_v4_flash_pressure_500_split/nonthinking \
  --concurrency 3 \
  --max-tokens 512 \
  --thinking disabled \
  --log-level INFO 2>&1 | tee -a outputs/logs/deepseek_v4_flash_pressure_500.log

./.venv/bin/python scripts/run_deepseek_v4_pressure_benchmark.py \
  --model deepseek-v4-flash \
  --raw-input-file third_party/CMMLU-master/data/test \
  --num-items 250 \
  --sample-pool-size 500 \
  --sample-offset 250 \
  --conditions baseline,strict_positive,high_pressure_wrong_option,recovery \
  --seed 20260426 \
  --output-root outputs/experiments/deepseek_v4_flash_pressure_500_split/thinking \
  --concurrency 3 \
  --max-tokens 2048 \
  --thinking enabled \
  --reasoning-effort medium \
  --log-level INFO 2>&1 | tee -a outputs/logs/deepseek_v4_flash_pressure_500.log
