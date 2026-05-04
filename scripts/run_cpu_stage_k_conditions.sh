#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-16 20 24 32}"
POSITIONS="${POSITIONS:-last_prompt_token first_answer_prefill last_4_prompt_mean last_8_prompt_mean}"
TOP_K_GRID="${TOP_K_GRID:-4 64 256}"
TAIL_BANDS="${TAIL_BANDS:-257-1024 65-128 129-256}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
CONDITION_PLAN="${CONDITION_PLAN:-outputs/stage_b/stage_b_condition_plan.jsonl}"
CONDITION_HIDDEN_ROOT="${CONDITION_HIDDEN_ROOT:-outputs/stage_k_condition_hidden}"
HIDDEN_ROOT="${HIDDEN_ROOT:-outputs/stage_k_hidden}"
SVD_ROOT="${SVD_ROOT:-outputs/stage_k_svd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage_k_condition_geometry}"
PLOT_ROOT="${PLOT_ROOT:-outputs/plots_stage_k_conditions}"

for POSITION in ${POSITIONS}; do
  "${PYTHON_BIN}" scripts/analyze_stage_b_geometry.py \
    --layers ${LAYERS} \
    --top-k-grid ${TOP_K_GRID} \
    --tail-bands ${TAIL_BANDS} \
    --condition-plan "${CONDITION_PLAN}" \
    --condition-hidden-dir "${CONDITION_HIDDEN_ROOT}/${POSITION}" \
    --svd-dir "${SVD_ROOT}/${POSITION}" \
    --reference-hidden-states-dir "${HIDDEN_ROOT}/${POSITION}" \
    --output-dir "${OUTPUT_ROOT}/${POSITION}" \
    --plot-dir "${PLOT_ROOT}/${POSITION}"
done
