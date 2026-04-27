#!/bin/zsh
set -euo pipefail

cd /Users/shiqi/code/graduation-project
mkdir -p .mplconfig outputs/logs outputs/experiments/local_probe_qwen7b_heldout_inputs

STAMP=$(date +%Y%m%d_%H%M%S)
HELDOUT_FILE="outputs/experiments/local_probe_qwen7b_heldout_inputs/qwen7b_heldout_n300_${STAMP}.json"

echo "Building Qwen7B held-out objective-local subset"
echo "Held-out file: ${HELDOUT_FILE}"
echo "Log file: outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"

./.venv/bin/python scripts/build_whitebox_objective_local_heldout.py \
  --input-file third_party/CMMLU-master/data/test \
  --output-file "${HELDOUT_FILE}" \
  --strict-positive-count 146 \
  --high-pressure-count 146 \
  --control-count 8 \
  --exclude-sample-files outputs/experiments/local_probe_qwen7b_intervention_main_inputs/qwen7b_intervention_main_sample_set.json \
  --seed 20260427 | tee -a "outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${HELDOUT_FILE}" \
  --output-dir outputs/experiments/local_probe_qwen7b_heldout_probe \
  --run-all-scenarios \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --log-level INFO 2>&1 | tee -a "outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_mechanistic_analysis.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${HELDOUT_FILE}" \
  --output-dir outputs/experiments/local_probe_qwen7b_heldout_mechanistic \
  --device mps \
  --dtype float32 \
  --max-length 512 \
  --top-k 8 \
  --hidden-state-layers=-1,-2,-3,-4 \
  --patching-max-samples 0 \
  --log-level INFO 2>&1 | tee -a "outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"

MECH_ROOT="outputs/experiments/local_probe_qwen7b_heldout_mechanistic/Qwen_Qwen2.5-7B-Instruct"
MECH_RUN=$(find "${MECH_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

MPLCONFIGDIR=/Users/shiqi/code/graduation-project/.mplconfig \
./.venv/bin/python scripts/run_local_probe_intervention.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --sample-file "${HELDOUT_FILE}" \
  --reference-mechanistic-run-dir "${MECH_RUN}" \
  --output-dir outputs/experiments/local_probe_qwen7b_heldout_intervention \
  --device mps \
  --dtype float32 \
  --methods baseline_state_interpolation \
  --sample-types strict_positive,high_pressure_wrong_option,control \
  --direction-sample-types strict_positive,high_pressure_wrong_option \
  --layer-configs '24-26=24,25,26' \
  --interpolation-scales 0.6 \
  --flush-every 20 \
  --log-level INFO 2>&1 | tee -a "outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"

INT_ROOT="outputs/experiments/local_probe_qwen7b_heldout_intervention/Qwen_Qwen2.5-7B-Instruct"
INT_RUN=$(find "${INT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

PYTHONPATH=/Users/shiqi/code/graduation-project \
./.venv/bin/python scripts/export_whitebox_objective_local_eval.py \
  --run-dir "${INT_RUN}" \
  --output-root outputs/experiments/whitebox_qwen7b_heldout_eval \
  --label qwen7b_heldout_mainline \
  --bootstrap-iters 2000 \
  --bootstrap-seed 20260427 | tee -a "outputs/logs/qwen7b_heldout_mainline_mps_${STAMP}.log"
