#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-8 12 16 20 24 28 32}"
K_GRID="${K_GRID:-4 8 16 32 48 64}"
STABILITY_SAMPLE_SIZE="${STABILITY_SAMPLE_SIZE:-1024}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/build_difference_matrix.py --layers ${LAYERS}
"${PYTHON_BIN}" scripts/analyze_spectrum.py --layers ${LAYERS}
"${PYTHON_BIN}" scripts/analyze_k_sensitivity.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --stability-method randomized \
  --stability-sample-size "${STABILITY_SAMPLE_SIZE}"
