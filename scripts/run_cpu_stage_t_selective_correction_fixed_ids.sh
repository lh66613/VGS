#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-24}"
TAIL_BAND="${TAIL_BAND:-257-1024}"
TOP_K_GRID="${TOP_K_GRID:-4 64}"
TRIGGER_RATES="${TRIGGER_RATES:-0.1 0.2 0.3}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MARGIN_SCORES="${MARGIN_SCORES:-}"

ARGS=(
  --split-policy fixed_ids \
  --output-dir outputs/stage_t_selective_correction_fixed_ids \
  --layers ${LAYERS} \
  --tail-band "${TAIL_BAND}" \
  --top-k-grid ${TOP_K_GRID} \
  --trigger-rates ${TRIGGER_RATES}
)
if [[ -n "${MARGIN_SCORES}" ]]; then
  ARGS+=(--margin-scores "${MARGIN_SCORES}")
fi

"${PYTHON_BIN}" scripts/analyze_stage_t_selective_correction.py "${ARGS[@]}"
