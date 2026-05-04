#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage_n_external}"
PREDICTIONS="${PREDICTIONS:-${OUTPUT_DIR}/amber_predictions.jsonl}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-${OUTPUT_DIR}/amber_hidden}"
SVD_DIR="${SVD_DIR:-outputs/svd}"
POPE_PREDICTIONS="${POPE_PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
POPE_HIDDEN_STATES_DIR="${POPE_HIDDEN_STATES_DIR:-outputs/hidden_states}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
CONDITION_HIDDEN_DIR="${CONDITION_HIDDEN_DIR:-outputs/stage_b_hidden}"
CONDITION_PLAN="${CONDITION_PLAN:-outputs/stage_b/stage_b_condition_plan.jsonl}"
PLOT_DIR="${PLOT_DIR:-outputs/plots}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vgs}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/analyze_stage_n_external_transfer.py \
  --predictions "${PREDICTIONS}" \
  --hidden-states-dir "${HIDDEN_STATES_DIR}" \
  --svd-dir "${SVD_DIR}" \
  --pope-predictions "${POPE_PREDICTIONS}" \
  --pope-hidden-states-dir "${POPE_HIDDEN_STATES_DIR}" \
  --split-dir "${SPLIT_DIR}" \
  --condition-hidden-dir "${CONDITION_HIDDEN_DIR}" \
  --condition-plan "${CONDITION_PLAN}" \
  --output-dir "${OUTPUT_DIR}" \
  --plot-dir "${PLOT_DIR}" \
  --layers ${LAYERS}
