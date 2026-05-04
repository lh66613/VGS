#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
PER_CATEGORY="${PER_CATEGORY:-4}"

"${PYTHON_BIN}" scripts/build_stage_r_case_panels.py \
  --per-category "${PER_CATEGORY}"
