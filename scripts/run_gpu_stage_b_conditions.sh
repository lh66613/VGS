#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
MAX_SAMPLES_PER_OUTCOME="${MAX_SAMPLES_PER_OUTCOME:-256}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/prepare_stage_b_conditions.py \
  --max-samples-per-outcome "${MAX_SAMPLES_PER_OUTCOME}"

"${PYTHON_BIN}" scripts/dump_stage_b_condition_hidden_states.py \
  --layers ${LAYERS} \
  --condition-plan outputs/stage_b/stage_b_condition_plan.jsonl
