#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
QUERY="${QUERY:-data/amber/data/query/query_discriminative.json}"
ANNOTATION="${ANNOTATION:-data/amber/data/annotations.json}"
IMAGES_DIR="${IMAGES_DIR:-data/amber/image}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage_n_external}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_PER_DIMENSION_LABEL="${MAX_PER_DIMENSION_LABEL:-300}"
DIMENSIONS="${DIMENSIONS:-}"
FULL="${FULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

args=(
  scripts/prepare_stage_n_amber.py
  --query "${QUERY}"
  --annotation "${ANNOTATION}"
  --images-dir "${IMAGES_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${FULL}" != "1" && -n "${MAX_PER_DIMENSION_LABEL}" ]]; then
  args+=(--max-per-dimension-label "${MAX_PER_DIMENSION_LABEL}")
fi

if [[ -n "${DIMENSIONS}" ]]; then
  args+=(--dimensions ${DIMENSIONS})
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

"${PYTHON_BIN}" "${args[@]}"
