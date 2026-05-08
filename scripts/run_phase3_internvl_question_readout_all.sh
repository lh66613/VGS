#!/usr/bin/env bash
set -euo pipefail

PHASE3_STEP="${PHASE3_STEP:-all}"
READOUT_POSITION="${READOUT_POSITION:-last_question_token}"
OUT_BASE="${OUT_BASE:-outputs/stage_o_cross_model_question_readout}"

MODEL_SPECS=(
  "internvl2|internvl2_8b|/data/lh/ModelandDataset/InternVL2-8B"
  "internvl2|internvl2_5_8b|/data/lh/ModelandDataset/InternVL2_5-8B"
)

for spec in "${MODEL_SPECS[@]}"; do
  IFS="|" read -r family alias path <<< "${spec}"
  out_root="${OUT_BASE}/${alias}"
  if [[ "${PHASE3_STEP}" == "gpu" || "${PHASE3_STEP}" == "all" ]]; then
    MODEL_FAMILY="${family}" MODEL_ALIAS="${alias}" MODEL_PATH="${path}" \
      READOUT_POSITION="${READOUT_POSITION}" OUT_ROOT="${out_root}" \
      bash scripts/run_gpu_phase3_cross_arch_user_readout.sh
  fi
  if [[ "${PHASE3_STEP}" == "cpu" || "${PHASE3_STEP}" == "all" ]]; then
    MODEL_FAMILY="${family}" MODEL_ALIAS="${alias}" OUT_ROOT="${out_root}" \
      bash scripts/run_cpu_phase3_cross_arch_user_readout.sh
  fi
done
