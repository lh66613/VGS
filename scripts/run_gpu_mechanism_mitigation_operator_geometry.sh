#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mechanism_mitigation/operator_geometry}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
SVD_DIR="${SVD_DIR:-outputs/svd}"
LAYERS="${LAYERS:-24}"
OPERATORS="${OPERATORS:-icd_blind vcd_diffusion}"
RANDOM_REPEATS="${RANDOM_REPEATS:-0}"
SUBSPACES="${SUBSPACES:-full top4 top16 band5_16 band17_64 band65_256 tail257_1024 top4_complement random12 random4_complement random_tail_dim}"
READOUT_POSITION="${READOUT_POSITION:-last_prompt_token}"
BLUR_RADIUS="${BLUR_RADIUS:-5.0}"
NOISE_STEP="${NOISE_STEP:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
DEVICE="${DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-}"

SUBSPACE_ARGS=(${SUBSPACES})
if [[ "${RANDOM_REPEATS}" -gt 0 ]]; then
  for IDX in $(seq 0 $((RANDOM_REPEATS - 1))); do
    PADDED="$(printf '%02d' "${IDX}")"
    SUBSPACE_ARGS+=("random12_s${PADDED}" "random4_complement_s${PADDED}" "random_tail_dim_s${PADDED}")
  done
fi

ARGS=(
  --predictions "${PREDICTIONS}"
  --svd-dir "${SVD_DIR}"
  --layers ${LAYERS}
  --operators ${OPERATORS}
  --subspaces ${SUBSPACE_ARGS[@]}
  --readout-position "${READOUT_POSITION}"
  --blur-radius "${BLUR_RADIUS}"
  --noise-step "${NOISE_STEP}"
  --model-family "${MODEL_FAMILY}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${TORCH_DTYPE}" ]]; then
  ARGS+=(--torch-dtype "${TORCH_DTYPE}")
fi

"${PYTHON_BIN}" scripts/dump_mechanism_mitigation_operator_geometry.py "${ARGS[@]}"
