#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
TEST_SUBSET="${TEST_SUBSET:-test}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
VCD_OPERATORS="${VCD_OPERATORS:-vcd_diffusion vcd_gray icd_blind}"
BUILD_COMPARISON="${BUILD_COMPARISON:-1}"

for VCD_OPERATOR in ${VCD_OPERATORS}; do
  VCD_OPERATOR="${VCD_OPERATOR}" \
  STAGE_T_DIR="${STAGE_T_DIR}" \
  TEST_SUBSET="${TEST_SUBSET}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  TARGET_RATES="${TARGET_RATES}" \
  bash scripts/run_gpu_stage_t_vcd.sh
done

if [[ "${BUILD_COMPARISON}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/build_stage_t_vcd_operator_comparison.py \
    --stage-t-dir "${STAGE_T_DIR}" \
    --target-rates ${TARGET_RATES}
fi
