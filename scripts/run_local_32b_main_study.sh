#!/bin/zsh
set -euo pipefail

# Generic runner for lab-local 32B models using an OpenAI-compatible endpoint.
# Usage example:
#   ROOT_DIR=/path/to/workspace \
#   LOCAL_MODEL_NAME=qwen2.5-32b-instruct \
#   LOCAL_API_BASE=http://127.0.0.1:8000/v1 \
#   LOCAL_API_KEY=EMPTY \
#   EXISTING_PROMPTS=/path/to/objective_cmmlu_prompts.jsonl \
#   ./scripts/run_local_32b_main_study.sh

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PROJECT_DIR="$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-local-32b-chat}"
LOCAL_API_BASE="${LOCAL_API_BASE:-http://127.0.0.1:8000/v1}"
LOCAL_API_KEY="${LOCAL_API_KEY:-EMPTY}"

RAW_INPUT_DIR="${RAW_INPUT_DIR:-$ROOT_DIR/third_party/CMMLU-master}"
EXISTING_PROMPTS="${EXISTING_PROMPTS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/experiments/local_32b_main_study_500}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONCURRENCY="${CONCURRENCY:-2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export PYTHONPATH="$PROJECT_DIR"

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN"
  "$PROJECT_DIR/scripts/run_full_pipeline.py"
  --task-type objective
  --models "$LOCAL_MODEL_NAME"
  --models-config config/models_config.json
  --output-root "$OUTPUT_ROOT"
  --output-format all
  --log-level "$LOG_LEVEL"
  --batch-size "$BATCH_SIZE"
  --concurrency "$CONCURRENCY"
  --num-samples "$NUM_SAMPLES"
)

if [[ -n "$EXISTING_PROMPTS" ]]; then
  CMD+=(
    --skip-perturbation
    --inference-input-file "$EXISTING_PROMPTS"
  )
else
  CMD+=(
    --raw-input-file "$RAW_INPUT_DIR"
    --total-samples "$TOTAL_SAMPLES"
  )
fi

export LOCAL_32B_API_KEY="$LOCAL_API_KEY"

CMD+=(
  --provider custom
  --api-base "$LOCAL_API_BASE"
  --api-key "$LOCAL_API_KEY"
)

echo "Running local 32B pipeline with model=$LOCAL_MODEL_NAME api_base=$LOCAL_API_BASE"
echo "Existing prompts: ${EXISTING_PROMPTS:-<none; will regenerate 500 questions from third_party/CMMLU-master>}"
echo "Output root: $OUTPUT_ROOT"

exec "${CMD[@]}"
