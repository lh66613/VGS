#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
TAIL_BAND="${TAIL_BAND:-257-1024}"
MAX_SVD_COORDS="${MAX_SVD_COORDS:-1024}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/build_stage_m_memory_bank.py \
  --layers ${LAYERS} \
  --tail-band "${TAIL_BAND}" \
  --max-svd-coords "${MAX_SVD_COORDS}"
