#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-16 20 24 32}"
POSITIONS="${POSITIONS:-last_prompt_token first_answer_prefill last_4_prompt_mean last_8_prompt_mean}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
HIDDEN_ROOT="${HIDDEN_ROOT:-outputs/stage_k_hidden}"
SVD_ROOT="${SVD_ROOT:-outputs/stage_k_svd}"
PLOT_ROOT="${PLOT_ROOT:-outputs/plots_stage_k_svd}"

for POSITION in ${POSITIONS}; do
  "${PYTHON_BIN}" scripts/build_difference_matrix.py \
    --layers ${LAYERS} \
    --hidden-states-dir "${HIDDEN_ROOT}/${POSITION}" \
    --output-dir "${SVD_ROOT}/${POSITION}"
  "${PYTHON_BIN}" scripts/analyze_spectrum.py \
    --layers ${LAYERS} \
    --matrix-dir "${SVD_ROOT}/${POSITION}" \
    --output-dir "${SVD_ROOT}/${POSITION}" \
    --plot-dir "${PLOT_ROOT}/${POSITION}"
done
