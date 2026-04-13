#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT"

export MPLCONFIGDIR="$ROOT/.mplconfig"
export PYTHONPATH="$PROJECT"

cd "$ROOT"

QWEN_SUBSET_SIZE="${QWEN_SUBSET_SIZE:-100}"
QWEN_NUM_SAMPLES="${QWEN_NUM_SAMPLES:-3}"
QWEN_SEED="${QWEN_SEED:-42}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-outputs/experiments/generalization_qwen_100}"

if [[ $# -ge 1 ]]; then
  MAIN_PERTURBATION_FILE="$1"
else
  LATEST_MAIN_RUN=$(ls -td "$ROOT"/outputs/experiments/main_study_deepseek_500/* 2>/dev/null | head -n 1 || true)
  if [[ -z "${LATEST_MAIN_RUN}" ]]; then
    echo "No DeepSeek main-study run found under outputs/experiments/main_study_deepseek_500" >&2
    exit 1
  fi
  MAIN_PERTURBATION_FILE="$LATEST_MAIN_RUN/perturbation/objective_cmmlu_prompts.jsonl"
fi

if [[ ! -f "$MAIN_PERTURBATION_FILE" ]]; then
  echo "Main perturbation file not found: $MAIN_PERTURBATION_FILE" >&2
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBSET_DIR="$ROOT/$QWEN_OUTPUT_ROOT/$TIMESTAMP/subset"
SUBSET_FILE="$SUBSET_DIR/qwen_generalization_subset.jsonl"

"$ROOT/.venv/bin/python" "$PROJECT/scripts/build_objective_subset.py" \
  --input-file "$MAIN_PERTURBATION_FILE" \
  --output-file "$SUBSET_FILE" \
  --total-samples "$QWEN_SUBSET_SIZE" \
  --seed "$QWEN_SEED"

"$ROOT/.venv/bin/python" "$PROJECT/scripts/run_full_pipeline.py" \
  --task-type objective \
  --skip-perturbation \
  --inference-input-file "$SUBSET_FILE" \
  --models qwen1.5-110b-chat \
  --models-config config/models_config.json \
  --num-samples "$QWEN_NUM_SAMPLES" \
  --batch-size 4 \
  --concurrency 2 \
  --output-root "$QWEN_OUTPUT_ROOT" \
  --output-format all \
  --log-level INFO
