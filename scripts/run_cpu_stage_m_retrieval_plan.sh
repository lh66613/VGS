#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
OUTCOMES="${OUTCOMES:-FP TN TP}"
MAX_TARGETS_PER_OUTCOME="${MAX_TARGETS_PER_OUTCOME:-64}"
K_NEIGHBORS="${K_NEIGHBORS:-8}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/prepare_stage_m_retrieval_plan.py \
  --layers ${LAYERS} \
  --outcomes ${OUTCOMES} \
  --max-targets-per-outcome "${MAX_TARGETS_PER_OUTCOME}" \
  --k-neighbors "${K_NEIGHBORS}"
