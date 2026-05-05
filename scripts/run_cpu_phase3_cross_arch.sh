#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
MODEL_ALIAS="${MODEL_ALIAS:?Set MODEL_ALIAS to the artifact alias used by the GPU Phase 3 run.}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
OUT_ROOT="${OUT_ROOT:-outputs/stage_o_cross_model/${MODEL_ALIAS}}"
K_GRID="${K_GRID:-4 128 256}"
CURVE_K_GRID="${CURVE_K_GRID:-4 8 16 32 64 128 256}"
TOP_K_GRID="${TOP_K_GRID:-4 128 256}"
TAIL_BANDS="${TAIL_BANDS:-257-1024 65-128 129-256}"
CONDITION_PLAN="${CONDITION_PLAN:-${OUT_ROOT}/stage_b_plan/stage_b_condition_plan.jsonl}"
NOTE_PATH="${NOTE_PATH:-${OUT_ROOT}/cross_model_replication.md}"

if [[ -z "${LAYERS:-}" ]]; then
  LAYER_FAMILY="${MODEL_FAMILY}"
  if [[ "${LAYER_FAMILY}" == "auto" ]]; then
    MODEL_HINT="${MODEL_ALIAS,,}"
    if [[ "${MODEL_HINT}" == *qwen2*vl* ]]; then
      LAYER_FAMILY="qwen2_vl"
    fi
  fi
  case "${LAYER_FAMILY}" in
    qwen2_vl|qwen2_5_vl)
      LAYERS="20 24 28"
      ;;
    *)
      LAYERS="20 24 32"
      ;;
  esac
fi

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
  --feature-family raw_img raw_blind difference projected_difference \
  --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
  --hidden-states-dir "${OUT_ROOT}/hidden_states" \
  --svd-dir "${OUT_ROOT}/svd" \
  --output-dir "${OUT_ROOT}/probes"

"${PYTHON_BIN}" scripts/compare_features.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --probe-dir "${OUT_ROOT}/probes" \
  --output-dir "${OUT_ROOT}/probes"

"${PYTHON_BIN}" scripts/analyze_stage_c_deep.py \
  --layers ${LAYERS} \
  --focus-layers ${LAYERS} \
  --k-grid ${CURVE_K_GRID} \
  --bands 1-4 5-8 9-16 17-32 33-64 65-128 129-256 \
  --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
  --hidden-states-dir "${OUT_ROOT}/hidden_states" \
  --svd-dir "${OUT_ROOT}/svd" \
  --plot-dir "${OUT_ROOT}/plots" \
  --output-dir "${OUT_ROOT}/stage_c_deep"

"${PYTHON_BIN}" scripts/build_stage_o_margin_baseline.py \
  --model-alias "${MODEL_ALIAS}" \
  --margins "${OUT_ROOT}/margins/pope_margin_scores.csv" \
  --output-dir "${OUT_ROOT}/margins"

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
  --output-dir "${OUT_ROOT}" \
  --note-path "${NOTE_PATH}"
