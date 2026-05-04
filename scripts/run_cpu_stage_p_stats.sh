#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
LAYERS="${LAYERS:-16 20 24 32}"
SEEDS="${SEEDS:-13 17 23 29 31}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage_p_stats}"

"${PYTHON_BIN}" scripts/analyze_stage_p_stats.py \
  --layers ${LAYERS} \
  --seeds ${SEEDS} \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --output-dir "${OUTPUT_DIR}"
