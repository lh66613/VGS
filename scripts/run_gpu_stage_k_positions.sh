#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-16 20 24 32}"
POSITIONS="${POSITIONS:-last_prompt_token first_answer_prefill last_4_prompt_mean last_8_prompt_mean}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage_k_hidden}"

for POSITION in ${POSITIONS}; do
  "${PYTHON_BIN}" scripts/dump_hidden_states.py \
    --layers ${LAYERS} \
    --predictions "${PREDICTIONS}" \
    --readout-position "${POSITION}" \
    --output-dir "${OUTPUT_ROOT}/${POSITION}"
done
