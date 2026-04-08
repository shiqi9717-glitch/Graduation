#!/bin/zsh
set -euo pipefail

ROOT="/Users/shiqi/code/代码/毕业代码"
PROJECT="$ROOT"

export MPLCONFIGDIR="$ROOT/.mplconfig"
export PYTHONPATH="$PROJECT"

cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [subset_jsonl] [num_samples]" >&2
  exit 1
fi

MODEL_NAME="$1"
SUBSET_FILE="${2:-}"
NUM_SAMPLES="${3:-3}"

if [[ -z "$SUBSET_FILE" ]]; then
  LATEST_SUBSET=$(ls -td "$ROOT"/outputs/experiments/generalization_qwen_100/*/subset 2>/dev/null | head -n 1 || true)
  if [[ -z "$LATEST_SUBSET" ]]; then
    echo "No existing subset directory found under outputs/experiments/generalization_qwen_100" >&2
    exit 1
  fi
  SUBSET_FILE="$LATEST_SUBSET/qwen_generalization_subset.jsonl"
fi

if [[ ! -f "$SUBSET_FILE" ]]; then
  echo "Subset file not found: $SUBSET_FILE" >&2
  exit 1
fi

SANITIZED_MODEL=$(echo "$MODEL_NAME" | tr ' /:' '___')
OUTPUT_ROOT="outputs/experiments/model_subset_runs/${SANITIZED_MODEL}"

"$ROOT/.venv/bin/python" "$PROJECT/scripts/run_full_pipeline.py" \
  --task-type objective \
  --skip-perturbation \
  --inference-input-file "$SUBSET_FILE" \
  --models "$MODEL_NAME" \
  --models-config config/models_config.json \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 4 \
  --concurrency 2 \
  --output-root "$OUTPUT_ROOT" \
  --output-format all \
  --log-level INFO
