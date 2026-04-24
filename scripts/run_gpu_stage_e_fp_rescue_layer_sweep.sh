#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
MAX_SAMPLES_PER_OUTCOME="${MAX_SAMPLES_PER_OUTCOME:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
ALPHA_GRID="${ALPHA_GRID:-1 2 4 6 8}"
GRANULARITIES="${GRANULARITIES:-last_token full_sequence}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/intervention_pilot.py \
  --layers ${LAYERS} \
  --max-samples-per-outcome "${MAX_SAMPLES_PER_OUTCOME}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --alpha-grid ${ALPHA_GRID} \
  --outcomes FP \
  --families rescue \
  --granularities ${GRANULARITIES}
