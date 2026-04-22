#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-8 12 16 20 24 28 32}"
K_GRID="${K_GRID:-4 8 16 32 48 64}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/train_probe.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --feature-family raw_img raw_blind difference projected_difference random_difference pca_img
"${PYTHON_BIN}" scripts/compare_features.py
"${PYTHON_BIN}" scripts/layerwise_analysis.py --layers ${LAYERS} --k-grid ${K_GRID}
