#!/usr/bin/env bash
set -euo pipefail

for ARG in "$@"; do
  if [[ "${ARG}" == *=* ]]; then
    export "${ARG}"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
SVD_DIR="${SVD_DIR:-outputs/svd}"
LAYERS="${LAYERS:-24}"
OPERATORS="${OPERATORS:-icd_blind}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_FAMILY="${MODEL_FAMILY:-llava}"
DEVICE="${DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
READOUT_POSITION="${READOUT_POSITION:-last_prompt_token}"
RANDOM_REPEATS="${RANDOM_REPEATS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NOISE_STEP="${NOISE_STEP:-500}"
BLUR_RADIUS="${BLUR_RADIUS:-5.0}"
MARGIN_SUMMARY="${MARGIN_SUMMARY:-outputs/margins/dump_pope_margins_summary.json}"
YES_TOKEN_IDS="${YES_TOKEN_IDS:-}"
NO_TOKEN_IDS="${NO_TOKEN_IDS:-}"

BASE_SUBSPACES=(
  full top4 top16
  band5_5 band5_6 band5_8 band5_12 band5_20
  tail257_1024 top4_complement random12
)

for START in $(seq 1 4 53); do
  END=$((START + 11))
  BASE_SUBSPACES+=("band${START}_${END}")
done

for IDX in $(seq 5 16); do
  BASE_SUBSPACES+=("v${IDX}" "band5_16_minus_v${IDX}")
done

for IDX in $(seq 0 $((RANDOM_REPEATS - 1))); do
  PADDED="$(printf '%02d' "${IDX}")"
  BASE_SUBSPACES+=("random12_s${PADDED}" "randcontig12_s${PADDED}")
done

SUBSPACES="${SUBSPACES:-${BASE_SUBSPACES[*]}}"
SUBSPACE_ARGS=(${SUBSPACES})

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
  --torch-dtype "${TORCH_DTYPE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -z "${YES_TOKEN_IDS}" && -s "${MARGIN_SUMMARY}" ]]; then
  YES_TOKEN_IDS="$("${PYTHON_BIN}" -c "import json; print(' '.join(map(str, json.load(open('${MARGIN_SUMMARY}')).get('yes_token_ids', []))))")"
fi
if [[ -z "${NO_TOKEN_IDS}" && -s "${MARGIN_SUMMARY}" ]]; then
  NO_TOKEN_IDS="$("${PYTHON_BIN}" -c "import json; print(' '.join(map(str, json.load(open('${MARGIN_SUMMARY}')).get('no_token_ids', []))))")"
fi
if [[ -n "${YES_TOKEN_IDS}" ]]; then
  ARGS+=(--yes-token-ids ${YES_TOKEN_IDS})
fi
if [[ -n "${NO_TOKEN_IDS}" ]]; then
  ARGS+=(--no-token-ids ${NO_TOKEN_IDS})
fi

"${PYTHON_BIN}" scripts/dump_mechanism_mitigation_operator_geometry.py "${ARGS[@]}"
