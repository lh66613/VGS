#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
SOURCE_STAGE_T_DIR="${SOURCE_STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
EXTERNAL_OUTPUT_DIR="${EXTERNAL_OUTPUT_DIR:-outputs/stage_t_external_amber_fixed_ids}"
EXTERNAL_PREDICTIONS="${EXTERNAL_PREDICTIONS:-outputs/stage_n_external_full/amber_predictions.jsonl}"
TEST_SUBSET="${TEST_SUBSET:-discriminative}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
VCD_OPERATORS="${VCD_OPERATORS:-vcd_diffusion vcd_gray icd_blind}"
GATE_POLICY="${GATE_POLICY:-external_top_rate}"
BUILD_EXTERNAL_WARNING="${BUILD_EXTERNAL_WARNING:-1}"

if [[ "${BUILD_EXTERNAL_WARNING}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/build_stage_t_external_warning.py \
    --stage-t-dir "${SOURCE_STAGE_T_DIR}" \
    --output-dir "${EXTERNAL_OUTPUT_DIR}" \
    --target-rates ${TARGET_RATES}
fi

STAGE_T_DIR="${EXTERNAL_OUTPUT_DIR}" \
VCD_SAMPLES="${EXTERNAL_OUTPUT_DIR}/stage_t_external_vcd_pool.jsonl" \
GATE_ASSIGNMENTS="${EXTERNAL_OUTPUT_DIR}/stage_t_external_gate_assignments_${GATE_POLICY}.csv" \
PREDICTIONS="${EXTERNAL_PREDICTIONS}" \
TEST_SUBSET="${TEST_SUBSET}" \
SPLIT_DIR="" \
TARGET_RATES="${TARGET_RATES}" \
VCD_OPERATORS="${VCD_OPERATORS}" \
bash scripts/run_gpu_stage_t_vcd_sweep.sh
