#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a LLaVA-HF compatible checkpoint path or repo id.}"
MODEL_ALIAS="${MODEL_ALIAS:-${MODEL_PATH##*/}}"
OUT_ROOT="${OUT_ROOT:-outputs/stage_o_cross_model/${MODEL_ALIAS}}"
LAYERS="${LAYERS:-20 24 32}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
CONDITION_PLAN="${CONDITION_PLAN:-outputs/stage_b/stage_b_condition_plan.jsonl}"
MAX_SAMPLES_ARG=()

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" scripts/run_pope_eval.py \
  --model-path "${MODEL_PATH}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUT_ROOT}/predictions" \
  "${MAX_SAMPLES_ARG[@]}"

"${PYTHON_BIN}" scripts/dump_hidden_states.py \
  --model-path "${MODEL_PATH}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --layers ${LAYERS} \
  --predictions "${OUT_ROOT}/predictions/pope_predictions.jsonl" \
  --output-dir "${OUT_ROOT}/hidden_states" \
  "${MAX_SAMPLES_ARG[@]}"

"${PYTHON_BIN}" scripts/dump_stage_b_condition_hidden_states.py \
  --model-path "${MODEL_PATH}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --layers ${LAYERS} \
  --condition-plan "${CONDITION_PLAN}" \
  --output-dir "${OUT_ROOT}/condition_hidden" \
  "${MAX_SAMPLES_ARG[@]}"
