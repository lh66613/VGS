#!/usr/bin/env bash
set -euo pipefail

STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
TEST_SUBSET="${TEST_SUBSET:-test}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-forced_evidence conservative internal_rationale}"

for PROMPT_VARIANT in ${PROMPT_VARIANTS}; do
  PROMPT_VARIANT="${PROMPT_VARIANT}" \
  STAGE_T_DIR="${STAGE_T_DIR}" \
  TEST_SUBSET="${TEST_SUBSET}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  bash scripts/run_gpu_stage_t_verification.sh
done

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
"${PYTHON_BIN}" scripts/build_stage_t_operator_upper_bound.py --stage-t-dir "${STAGE_T_DIR}"
