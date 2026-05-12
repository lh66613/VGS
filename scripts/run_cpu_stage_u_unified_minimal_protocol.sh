#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage_u_unified_minimal_protocol}"
NOTES_PATH="${NOTES_PATH:-notes/stage_u_unified_minimal_protocol.md}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
TARGET_RATES="${TARGET_RATES:-0.1 0.2 0.3}"
TAIL_START="${TAIL_START:-257}"
TAIL_END="${TAIL_END:-1024}"
RANDOM_REPEATS="${RANDOM_REPEATS:-200}"

"${PYTHON_BIN}" scripts/build_stage_u_unified_minimal_protocol.py \
  --output-dir "${OUTPUT_DIR}" \
  --notes-path "${NOTES_PATH}" \
  --split-dir "${SPLIT_DIR}" \
  --target-rates ${TARGET_RATES} \
  --tail-start "${TAIL_START}" \
  --tail-end "${TAIL_END}" \
  --random-repeats "${RANDOM_REPEATS}"
