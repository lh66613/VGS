#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-20 24 32}"
TOP_K_GRID="${TOP_K_GRID:-4 64 256}"
TAIL_BANDS="${TAIL_BANDS:-257-1024 65-128 129-256}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/analyze_stage_b_geometry.py \
  --layers ${LAYERS} \
  --top-k-grid ${TOP_K_GRID} \
  --tail-bands ${TAIL_BANDS} \
  --condition-plan outputs/stage_b/stage_b_condition_plan.jsonl \
  --condition-hidden-dir outputs/stage_b_hidden
