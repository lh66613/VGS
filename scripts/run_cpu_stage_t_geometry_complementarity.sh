#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
LAYER="${LAYER:-24}"
SPLIT="${SPLIT:-test}"
CALIBRATION_SPLIT="${CALIBRATION_SPLIT:-calibration}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"

"${PYTHON_BIN}" scripts/analyze_stage_t_geometry_complementarity.py \
  --stage-t-dir "${STAGE_T_DIR}" \
  --layer "${LAYER}" \
  --split "${SPLIT}" \
  --calibration-split "${CALIBRATION_SPLIT}" \
  --target-rates ${TARGET_RATES}
