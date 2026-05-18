#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/lh/ModelandDataset/llava-1.5-7b-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/official_vcd_pope_llava15_7b}"
SUBSETS="${SUBSETS:-random popular adversarial}"
VCD_ALPHA="${VCD_ALPHA:-1.0}"
VCD_BETA="${VCD_BETA:-0.1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_DECODE_STRATEGY="${VCD_DECODE_STRATEGY:-sample}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-1.0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-}"
VCD_SEED="${VCD_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-1}"

RUN_ARGS=(
  --model-path "${MODEL_PATH}"
  --subsets ${SUBSETS}
  --alpha "${VCD_ALPHA}"
  --beta "${VCD_BETA}"
  --noise-step "${VCD_NOISE_STEP}"
  --decode-strategy "${VCD_DECODE_STRATEGY}"
  --temperature "${VCD_TEMPERATURE}"
  --top-p "${VCD_TOP_P}"
  --seed "${VCD_SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${VCD_TOP_K}" ]]; then
  RUN_ARGS+=(--top-k "${VCD_TOP_K}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  RUN_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${RESUME}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi

"${PYTHON_BIN}" scripts/run_official_vcd_pope_baseline.py "${RUN_ARGS[@]}"
