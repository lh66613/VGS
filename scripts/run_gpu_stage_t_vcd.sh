#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
VCD_SAMPLES="${VCD_SAMPLES:-${STAGE_T_DIR}/stage_t_verification_pool.jsonl}"
GATE_ASSIGNMENTS="${GATE_ASSIGNMENTS:-${STAGE_T_DIR}/stage_t_verification_gate_assignments.csv}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
TEST_SUBSET="${TEST_SUBSET:-test}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
VCD_OPERATOR="${VCD_OPERATOR:-vcd_diffusion}"
VCD_ALPHA="${VCD_ALPHA:-1.0}"
VCD_BETA="${VCD_BETA:-0.1}"
VCD_BLUR_RADIUS="${VCD_BLUR_RADIUS:-5.0}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_DECODE_STRATEGY="${VCD_DECODE_STRATEGY:-sample}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-1.0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-}"
VCD_SEED="${VCD_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
RANDOM_REPEATS="${RANDOM_REPEATS:-200}"

RUN_ARGS=(
  --vcd-samples "${VCD_SAMPLES}" \
  --operator "${VCD_OPERATOR}" \
  --alpha "${VCD_ALPHA}" \
  --beta "${VCD_BETA}" \
  --blur-radius "${VCD_BLUR_RADIUS}" \
  --noise-step "${VCD_NOISE_STEP}" \
  --decode-strategy "${VCD_DECODE_STRATEGY}" \
  --temperature "${VCD_TEMPERATURE}" \
  --top-p "${VCD_TOP_P}" \
  --seed "${VCD_SEED}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${STAGE_T_DIR}"
)
if [[ -n "${VCD_TOP_K}" ]]; then
  RUN_ARGS+=(--top-k "${VCD_TOP_K}")
fi

"${PYTHON_BIN}" scripts/run_stage_t_vcd_eval.py "${RUN_ARGS[@]}"

ANALYZE_ARGS=(
  --predictions "${PREDICTIONS}"
  --gate-assignments "${GATE_ASSIGNMENTS}"
  --vcd-predictions "${STAGE_T_DIR}/stage_t_vcd_predictions_${VCD_OPERATOR}.jsonl"
  --operator "${VCD_OPERATOR}"
  --test-subset "${TEST_SUBSET}"
  --target-rates ${TARGET_RATES}
  --random-repeats "${RANDOM_REPEATS}"
  --output-dir "${STAGE_T_DIR}"
)
if [[ -n "${SPLIT_DIR}" ]]; then
  ANALYZE_ARGS+=(--split-dir "${SPLIT_DIR}")
fi

"${PYTHON_BIN}" scripts/analyze_stage_t_vcd_results.py "${ANALYZE_ARGS[@]}"
