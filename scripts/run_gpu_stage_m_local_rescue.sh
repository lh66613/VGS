#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-32}"
TARGET_OUTCOMES="${TARGET_OUTCOMES:-FP TN TP}"
MAX_TARGETS_PER_OUTCOME="${MAX_TARGETS_PER_OUTCOME:-32}"
ALPHA_GRID="${ALPHA_GRID:-2 4 8}"
GATES="${GATES:-always low_abs_margin high_fp_risk margin_and_fp_risk}"
RETRIEVAL_MODES="${RETRIEVAL_MODES:-same_object_tn svd_knn_tn tail_knn_tn random_tn same_object_fp}"
GRANULARITIES="${GRANULARITIES:-last_token}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
LOGITS_ONLY="${LOGITS_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

args=(
  scripts/run_stage_m_local_rescue.py
  --layers ${LAYERS}
  --target-outcomes ${TARGET_OUTCOMES}
  --max-targets-per-outcome "${MAX_TARGETS_PER_OUTCOME}"
  --alpha-grid ${ALPHA_GRID}
  --gates ${GATES}
  --retrieval-modes ${RETRIEVAL_MODES}
  --granularities ${GRANULARITIES}
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${LOGITS_ONLY}" == "1" ]]; then
  args+=(--logits-only)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

"${PYTHON_BIN}" "${args[@]}"
