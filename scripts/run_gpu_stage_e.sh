#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-24}"
MAX_SAMPLES_PER_OUTCOME="${MAX_SAMPLES_PER_OUTCOME:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
ALPHA_GRID="${ALPHA_GRID:-4.0 5.0 6.0 7.0 8.0}"
RANDOM_SCALE="${RANDOM_SCALE:-5.0}"
TAIL_BAND="${TAIL_BAND:-257-1024}"
OUTCOMES="${OUTCOMES:-TN FP}"
FAMILIES="${FAMILIES:-tail rescue}"
GRANULARITIES="${GRANULARITIES:-last_token}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/intervention_precheck.py \
  --layers ${LAYERS} \
  --max-samples 1 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --random-scale "${RANDOM_SCALE}"

"${PYTHON_BIN}" scripts/intervention_pilot.py \
  --layers ${LAYERS} \
  --max-samples-per-outcome "${MAX_SAMPLES_PER_OUTCOME}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --alpha-grid ${ALPHA_GRID} \
  --tail-band "${TAIL_BAND}" \
  --outcomes ${OUTCOMES} \
  --families ${FAMILIES} \
  --granularities ${GRANULARITIES}
