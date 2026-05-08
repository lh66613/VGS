#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
ROOT="${ROOT:-outputs/stage_o_cross_model_user_readout}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/audit}"
AUDIT_NOTE="${AUDIT_NOTE:-notes/stage_o_user_readout_audit.md}"
SANITY_NOTE="${SANITY_NOTE:-notes/stage_o_user_readout_probe_sanity.md}"

"${PYTHON_BIN}" scripts/audit_stage_o_cross_arch_results.py \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --notes-path "${AUDIT_NOTE}"

"${PYTHON_BIN}" scripts/audit_stage_o_probe_sanity.py \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --notes-path "${SANITY_NOTE}"
