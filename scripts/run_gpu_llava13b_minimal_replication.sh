#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

MODEL_PATH="${MODEL_PATH:-/data/lh/ModelandDataset/llava-1.5-13b-hf}"
MODEL_FAMILY="${MODEL_FAMILY:-llava}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-auto}"
READOUT_POSITION="${READOUT_POSITION:-last_prompt_token}"
LAYERS="${LAYERS:-20}"
LLAVA13B_ROOT="${LLAVA13B_ROOT:-outputs/stage_o_cross_model/llava_13b}"
MITIGATION_DIR="${MITIGATION_DIR:-outputs/mechanism_mitigation/llava13b_minimal}"
PREDICTIONS="${PREDICTIONS:-${LLAVA13B_ROOT}/predictions/pope_predictions.jsonl}"
SVD_DIR="${SVD_DIR:-${LLAVA13B_ROOT}/svd}"
MARGIN_DIR="${MARGIN_DIR:-${LLAVA13B_ROOT}/margins}"
MARGIN_SCORES="${MARGIN_SCORES:-${MARGIN_DIR}/pope_margin_scores.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${MITIGATION_DIR}/operator_geometry}"
RANDOM_REPEATS="${RANDOM_REPEATS:-10}"
SUBSPACES="${SUBSPACES:-full band5_16 tail257_1024 random12}"
STEPS="${STEPS:-margins geometry}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NOISE_STEP="${NOISE_STEP:-500}"
BLUR_RADIUS="${BLUR_RADIUS:-5.0}"

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

SUBSPACE_ARGS=(${SUBSPACES})
if [[ "${RANDOM_REPEATS}" -gt 0 ]]; then
  for IDX in $(seq 0 $((RANDOM_REPEATS - 1))); do
    PADDED="$(printf '%02d' "${IDX}")"
    SUBSPACE_ARGS+=("random12_s${PADDED}")
  done
fi

for STEP in ${STEPS}; do
  case "${STEP}" in
    margins)
      if [[ "${FORCE_RERUN}" == "1" || ! -s "${MARGIN_SCORES}" ]]; then
        "${PYTHON_BIN}" scripts/dump_pope_margins.py \
          --model-path "${MODEL_PATH}" \
          --model-family "${MODEL_FAMILY}" \
          --torch-dtype "${TORCH_DTYPE}" \
          --predictions "${PREDICTIONS}" \
          --output-dir "${MARGIN_DIR}" \
          "${MAX_SAMPLES_ARGS[@]}"
      else
        echo "Skip margins; found ${MARGIN_SCORES}"
      fi
      ;;
    geometry)
      if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUTPUT_DIR}/operator_geometry.csv" ]]; then
        "${PYTHON_BIN}" scripts/dump_mechanism_mitigation_operator_geometry.py \
          --model-path "${MODEL_PATH}" \
          --model-family "${MODEL_FAMILY}" \
          --torch-dtype "${TORCH_DTYPE}" \
          --device "${DEVICE}" \
          --predictions "${PREDICTIONS}" \
          --svd-dir "${SVD_DIR}" \
          --layers ${LAYERS} \
          --operators icd_blind \
          --subspaces ${SUBSPACE_ARGS[@]} \
          --readout-position "${READOUT_POSITION}" \
          --noise-step "${NOISE_STEP}" \
          --blur-radius "${BLUR_RADIUS}" \
          --output-dir "${OUTPUT_DIR}" \
          "${MAX_SAMPLES_ARGS[@]}"
      else
        echo "Skip operator geometry; found ${OUTPUT_DIR}/operator_geometry.csv"
      fi
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
