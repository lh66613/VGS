#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/lh/ModelandDataset/llava-1.5-7b-hf}"
MODEL_FAMILY="${MODEL_FAMILY:-llava}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
MARGIN_DIR="${MARGIN_DIR:-outputs/margins}"
MARGIN_SCORES="${MARGIN_SCORES:-${MARGIN_DIR}/pope_margin_scores.csv}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
TEST_SUBSET="${TEST_SUBSET:-test}"
TARGET_RATES="${TARGET_RATES:-0.1 0.2 0.3}"
VCD_OPERATORS="${VCD_OPERATORS:-vcd_diffusion vcd_gray icd_blind vcd_blur}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ "${FORCE_RERUN}" == "1" || ! -s "${MARGIN_SCORES}" ]]; then
  "${PYTHON_BIN}" scripts/dump_pope_margins.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --predictions "${PREDICTIONS}" \
    --output-dir "${MARGIN_DIR}"
else
  echo "Skip margins; found ${MARGIN_SCORES}"
fi

"${PYTHON_BIN}" scripts/build_stage_o_margin_baseline.py \
  --margins "${MARGIN_SCORES}" \
  --model-alias "${MODEL_FAMILY}" \
  --output-dir "${MARGIN_DIR}"

MARGIN_SCORES="${MARGIN_SCORES}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_cpu_stage_t_selective_correction_fixed_ids.sh

"${PYTHON_BIN}" scripts/build_stage_t_selective_warning.py \
  --stage-t-dir "${STAGE_T_DIR}" \
  --target-rates ${TARGET_RATES}

for VCD_OPERATOR in ${VCD_OPERATORS}; do
  PRED_PATH="${STAGE_T_DIR}/stage_t_vcd_predictions_${VCD_OPERATOR}.jsonl"
  if [[ ! -s "${PRED_PATH}" ]]; then
    echo "Skip ${VCD_OPERATOR}; missing ${PRED_PATH}"
    continue
  fi
  "${PYTHON_BIN}" scripts/analyze_stage_t_vcd_results.py \
    --gate-assignments "${STAGE_T_DIR}/stage_t_verification_gate_assignments.csv" \
    --vcd-predictions "${PRED_PATH}" \
    --operator "${VCD_OPERATOR}" \
    --test-subset "${TEST_SUBSET}" \
    --split-dir "${SPLIT_DIR}" \
    --target-rates ${TARGET_RATES} \
    --output-dir "${STAGE_T_DIR}"
done

"${PYTHON_BIN}" scripts/build_stage_t_vcd_operator_comparison.py \
  --stage-t-dir "${STAGE_T_DIR}" \
  --target-rates ${TARGET_RATES}

"${PYTHON_BIN}" scripts/build_stage_t_margin_geometry_ablation.py \
  --stage-t-dir "${STAGE_T_DIR}" \
  --target-rates ${TARGET_RATES} \
  --operators all
