#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MODEL_ALIAS="${MODEL_ALIAS:-llava_cross_model}"
OUT_ROOT="${OUT_ROOT:-outputs/stage_o_cross_model/${MODEL_ALIAS}}"
LAYERS="${LAYERS:-20 24 32}"
K_GRID="${K_GRID:-4 64 128 256}"
TOP_K_GRID="${TOP_K_GRID:-4 64 256}"
TAIL_BANDS="${TAIL_BANDS:-257-1024 65-128 129-256}"
CONDITION_PLAN="${CONDITION_PLAN:-outputs/stage_b/stage_b_condition_plan.jsonl}"

"${PYTHON_BIN}" scripts/build_difference_matrix.py \
  --layers ${LAYERS} \
  --hidden-states-dir "${OUT_ROOT}/hidden_states" \
  --output-dir "${OUT_ROOT}/svd"

"${PYTHON_BIN}" scripts/analyze_spectrum.py \
  --layers ${LAYERS} \
  --matrix-dir "${OUT_ROOT}/svd" \
  --plot-dir "${OUT_ROOT}/plots" \
  --output-dir "${OUT_ROOT}/svd"

"${PYTHON_BIN}" scripts/train_probe.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --feature-family raw_img raw_blind difference projected_difference random_difference pca_img \
  --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
  --hidden-states-dir "${OUT_ROOT}/hidden_states" \
  --svd-dir "${OUT_ROOT}/svd" \
  --output-dir "${OUT_ROOT}/probes"

"${PYTHON_BIN}" scripts/compare_features.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --probe-dir "${OUT_ROOT}/probes" \
  --output-dir "${OUT_ROOT}/probes"

"${PYTHON_BIN}" scripts/analyze_stage_b_geometry.py \
  --layers ${LAYERS} \
  --top-k-grid ${TOP_K_GRID} \
  --tail-bands ${TAIL_BANDS} \
  --condition-plan "${CONDITION_PLAN}" \
  --condition-hidden-dir "${OUT_ROOT}/condition_hidden" \
  --svd-dir "${OUT_ROOT}/svd" \
  --reference-predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
  --reference-hidden-states-dir "${OUT_ROOT}/hidden_states" \
  --plot-dir "${OUT_ROOT}/plots" \
  --output-dir "${OUT_ROOT}/stage_b"

"${PYTHON_BIN}" scripts/build_stage_o_cross_model_summary.py \
  --model-alias "${MODEL_ALIAS}" \
  --stage-o-dir "${OUT_ROOT}" \
  --output-dir "${OUT_ROOT}"
