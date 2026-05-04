#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
K_GRID="${K_GRID:-4 8 16 32 64}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/analyze_stage_l_evidence_subspace.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID}
