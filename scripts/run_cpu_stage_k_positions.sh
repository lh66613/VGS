#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-16 20 24 32}"
K_GRID="${K_GRID:-4 64 128 256}"
POSITIONS="${POSITIONS:-last_prompt_token first_answer_prefill last_4_prompt_mean last_8_prompt_mean}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/analyze_stage_k_positions.py \
  --layers ${LAYERS} \
  --k-grid ${K_GRID} \
  --positions ${POSITIONS}
