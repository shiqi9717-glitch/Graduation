#!/bin/zsh
set -euo pipefail

ROOT="/Users/shiqi/code/代码/毕业代码"
PROJECT="$ROOT"

export MPLCONFIGDIR="$ROOT/.mplconfig"
export PYTHONPATH="$PROJECT"

cd "$ROOT"

"$ROOT/.venv/bin/python" "$PROJECT/scripts/run_full_pipeline.py" \
  --task-type objective \
  --raw-input-file "$ROOT/third_party/CMMLU-master" \
  --models deepseek-chat \
  --models-config config/models_config.json \
  --total-samples 500 \
  --num-samples 5 \
  --batch-size 4 \
  --concurrency 2 \
  --output-root outputs/experiments/main_study_deepseek_500 \
  --output-format all \
  --log-level INFO
