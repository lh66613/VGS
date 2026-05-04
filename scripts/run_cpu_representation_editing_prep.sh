#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
LAYERS="${LAYERS:-20 24 32}"
K_GRID="${K_GRID:-8 16 32 64}"
METHODS="${METHODS:-plain_svd fisher_fp_tn pls_fp_tn matched_vs_adversarial_logistic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/representation_editing_prep}"

"${PYTHON_BIN}" scripts/build_representation_editing_prep.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --methods ${METHODS} \
  --output-dir "${OUTPUT_DIR}"
