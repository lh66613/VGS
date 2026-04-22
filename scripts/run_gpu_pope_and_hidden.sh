#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-8 12 16 20 24 28 32}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MAX_SAMPLES_ARG=()

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" scripts/run_pope_eval.py "${MAX_SAMPLES_ARG[@]}"
"${PYTHON_BIN}" scripts/dump_hidden_states.py \
  --layers ${LAYERS} \
  --predictions outputs/predictions/pope_predictions.jsonl \
  "${MAX_SAMPLES_ARG[@]}"
