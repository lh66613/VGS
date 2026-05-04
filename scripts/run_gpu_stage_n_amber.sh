#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage_n_external}"
PLAN="${PLAN:-${OUTPUT_DIR}/amber_discriminative_plan.jsonl}"
PREDICTIONS="${PREDICTIONS:-${OUTPUT_DIR}/amber_predictions.jsonl}"
HIDDEN_DIR="${HIDDEN_DIR:-${OUTPUT_DIR}/amber_hidden}"

eval_args=(
  scripts/run_stage_n_amber_eval.py
  --plan "${PLAN}"
  --output-dir "${OUTPUT_DIR}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
hidden_args=(
  scripts/dump_hidden_states.py
  --predictions "${PREDICTIONS}"
  --layers ${LAYERS}
  --output-dir "${HIDDEN_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  eval_args+=(--max-samples "${MAX_SAMPLES}")
  hidden_args+=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" "${eval_args[@]}"
"${PYTHON_BIN}" "${hidden_args[@]}"
