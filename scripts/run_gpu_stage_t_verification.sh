#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction}"
VERIFICATION_SAMPLES="${VERIFICATION_SAMPLES:-${STAGE_T_DIR}/stage_t_verification_samples.jsonl}"
GATE_ASSIGNMENTS="${GATE_ASSIGNMENTS:-${STAGE_T_DIR}/stage_t_verification_gate_assignments.csv}"
TEST_SUBSET="${TEST_SUBSET:-adversarial}"
SPLIT_DIR="${SPLIT_DIR:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PROMPT_VARIANT="${PROMPT_VARIANT:-legacy}"
if [[ "${PROMPT_VARIANT}" == "legacy" ]]; then
  PREDICTION_STEM="stage_t_verification_predictions"
else
  PREDICTION_STEM="stage_t_verification_predictions_${PROMPT_VARIANT}"
fi

"${PYTHON_BIN}" scripts/run_stage_t_verification_eval.py \
  --verification-samples "${VERIFICATION_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  --output-dir "${STAGE_T_DIR}"

ANALYZE_ARGS=(
  --gate-assignments "${GATE_ASSIGNMENTS}"
  --verification-predictions "${STAGE_T_DIR}/${PREDICTION_STEM}.jsonl"
  --test-subset "${TEST_SUBSET}"
  --prompt-variant "${PROMPT_VARIANT}"
  --output-dir "${STAGE_T_DIR}"
)
if [[ -n "${SPLIT_DIR}" ]]; then
  ANALYZE_ARGS+=(--split-dir "${SPLIT_DIR}")
fi

"${PYTHON_BIN}" scripts/analyze_stage_t_verification_results.py "${ANALYZE_ARGS[@]}"
