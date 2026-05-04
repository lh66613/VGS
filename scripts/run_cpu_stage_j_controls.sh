#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
K_GRID="${K_GRID:-4 64 128 256}"
REPEATS="${REPEATS:-5}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
STABILITY_SAMPLE_SIZE="${STABILITY_SAMPLE_SIZE:-1024}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/analyze_stage_j_controls.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --repeats "${REPEATS}" \
  --random-repeats "${RANDOM_REPEATS}" \
  --stability-sample-size "${STABILITY_SAMPLE_SIZE}"
