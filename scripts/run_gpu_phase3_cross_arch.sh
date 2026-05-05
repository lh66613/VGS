#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared Qwen/InternVL/LLaVA checkpoint path.}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL_PATH##*/}}"
OUT_ROOT="${OUT_ROOT:-outputs/stage_o_cross_model/${MODEL_ALIAS}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
READOUT_POSITION="${READOUT_POSITION:-last_prompt_token}"
CONDITION_MAX_SAMPLES_PER_OUTCOME="${CONDITION_MAX_SAMPLES_PER_OUTCOME:-256}"
INTERNVL_MAX_TILES="${INTERNVL_MAX_TILES:-12}"
MAX_SAMPLES_ARG=()
QWEN_PIXEL_ARGS=()
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ -z "${LAYERS:-}" ]]; then
  LAYER_FAMILY="${MODEL_FAMILY}"
  if [[ "${LAYER_FAMILY}" == "auto" ]]; then
    MODEL_HINT="${MODEL_PATH,,} ${MODEL_ALIAS,,}"
    if [[ "${MODEL_HINT}" == *qwen2*vl* ]]; then
      LAYER_FAMILY="qwen2_vl"
    fi
  fi
  case "${LAYER_FAMILY}" in
    qwen2_vl|qwen2_5_vl)
      LAYERS="20 24 28"
      ;;
    *)
      LAYERS="20 24 32"
      ;;
  esac
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${QWEN_MIN_PIXELS:-}" ]]; then
  QWEN_PIXEL_ARGS+=(--qwen-min-pixels "${QWEN_MIN_PIXELS}")
fi
if [[ -n "${QWEN_MAX_PIXELS:-}" ]]; then
  QWEN_PIXEL_ARGS+=(--qwen-max-pixels "${QWEN_MAX_PIXELS}")
fi

if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUT_ROOT}/predictions/pope_predictions.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/run_pope_eval.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --internvl-max-tiles "${INTERNVL_MAX_TILES}" \
    --output-dir "${OUT_ROOT}/predictions" \
    "${QWEN_PIXEL_ARGS[@]}" \
    "${MAX_SAMPLES_ARG[@]}"
else
  echo "Skip POPE predictions; found ${OUT_ROOT}/predictions/pope_predictions.jsonl"
fi

if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUT_ROOT}/margins/pope_margin_scores.csv" ]]; then
  "${PYTHON_BIN}" scripts/dump_pope_margins.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
    --internvl-max-tiles "${INTERNVL_MAX_TILES}" \
    --output-dir "${OUT_ROOT}/margins" \
    "${QWEN_PIXEL_ARGS[@]}" \
    "${MAX_SAMPLES_ARG[@]}"
else
  echo "Skip margins; found ${OUT_ROOT}/margins/pope_margin_scores.csv"
fi

if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUT_ROOT}/stage_b_plan/stage_b_condition_plan.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/prepare_stage_b_conditions.py \
    --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
    --max-samples-per-outcome "${CONDITION_MAX_SAMPLES_PER_OUTCOME}" \
    --output-dir "${OUT_ROOT}/stage_b_plan"
else
  echo "Skip condition plan; found ${OUT_ROOT}/stage_b_plan/stage_b_condition_plan.jsonl"
fi

if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUT_ROOT}/hidden_states/dump_hidden_states_summary.json" ]]; then
  "${PYTHON_BIN}" scripts/dump_hidden_states.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --layers ${LAYERS} \
    --readout-position "${READOUT_POSITION}" \
    --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
    --internvl-max-tiles "${INTERNVL_MAX_TILES}" \
    --output-dir "${OUT_ROOT}/hidden_states" \
    "${QWEN_PIXEL_ARGS[@]}" \
    "${MAX_SAMPLES_ARG[@]}"
else
  echo "Skip paired hidden states; found ${OUT_ROOT}/hidden_states/dump_hidden_states_summary.json"
fi

if [[ "${FORCE_RERUN}" == "1" || ! -s "${OUT_ROOT}/condition_hidden/dump_stage_b_condition_hidden_states_summary.json" ]]; then
  "${PYTHON_BIN}" scripts/dump_stage_b_condition_hidden_states.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --layers ${LAYERS} \
    --readout-position "${READOUT_POSITION}" \
    --condition-plan "${OUT_ROOT}/stage_b_plan/stage_b_condition_plan.jsonl" \
    --internvl-max-tiles "${INTERNVL_MAX_TILES}" \
    --output-dir "${OUT_ROOT}/condition_hidden" \
    "${QWEN_PIXEL_ARGS[@]}" \
    "${MAX_SAMPLES_ARG[@]}"
else
  echo "Skip condition hidden states; found ${OUT_ROOT}/condition_hidden/dump_stage_b_condition_hidden_states_summary.json"
fi
