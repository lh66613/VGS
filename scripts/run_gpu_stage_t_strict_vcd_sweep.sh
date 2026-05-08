#!/usr/bin/env bash
set -euo pipefail

STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction}"
TEST_SUBSET="${TEST_SUBSET:-adversarial}"
SPLIT_DIR="${SPLIT_DIR:-}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
VCD_OPERATORS="${VCD_OPERATORS:-vcd_diffusion vcd_gray icd_blind}"

STAGE_T_DIR="${STAGE_T_DIR}" \
TEST_SUBSET="${TEST_SUBSET}" \
SPLIT_DIR="${SPLIT_DIR}" \
TARGET_RATES="${TARGET_RATES}" \
VCD_OPERATORS="${VCD_OPERATORS}" \
bash scripts/run_gpu_stage_t_vcd_sweep.sh
