#!/usr/bin/env bash
set -euo pipefail

PHASE3_STEP="${PHASE3_STEP:-all}"

MODEL_SPECS=(
  "qwen2_vl|qwen2_vl_7b|/data/lh/ModelandDataset/Qwen2-VL-7B-Instruct"
  "qwen2_5_vl|qwen2_5_vl_7b|/data/lh/ModelandDataset/Qwen2.5-VL-7B-Instruct"
  "internvl2|internvl2_8b|/data/lh/ModelandDataset/InternVL2-8B"
  "internvl2|internvl2_5_8b|/data/lh/ModelandDataset/InternVL2_5-8B"
)

for spec in "${MODEL_SPECS[@]}"; do
  IFS="|" read -r family alias path <<< "${spec}"
  if [[ "${PHASE3_STEP}" == "gpu" || "${PHASE3_STEP}" == "all" ]]; then
    MODEL_FAMILY="${family}" MODEL_ALIAS="${alias}" MODEL_PATH="${path}" \
      bash scripts/run_gpu_phase3_cross_arch.sh
  fi
  if [[ "${PHASE3_STEP}" == "cpu" || "${PHASE3_STEP}" == "all" ]]; then
    MODEL_FAMILY="${family}" MODEL_ALIAS="${alias}" \
      bash scripts/run_cpu_phase3_cross_arch.sh
  fi
done
