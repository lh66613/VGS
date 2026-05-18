#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
LLAVA13B_ROOT="${LLAVA13B_ROOT:-outputs/stage_o_cross_model/llava_13b}"
MITIGATION_DIR="${MITIGATION_DIR:-outputs/mechanism_mitigation/llava13b_minimal}"
OPERATOR_GEOMETRY="${OPERATOR_GEOMETRY:-${MITIGATION_DIR}/operator_geometry/operator_geometry.csv}"
PREDICTIONS="${PREDICTIONS:-${LLAVA13B_ROOT}/predictions/pope_predictions.jsonl}"
MARGIN_SCORES="${MARGIN_SCORES:-${LLAVA13B_ROOT}/margins/pope_margin_scores.csv}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
LAYER="${LAYER:-20}"
RANDOM_REPEATS="${RANDOM_REPEATS:-10}"
SUBSPACES="${SUBSPACES:-full band5_16 random12}"
ALPHAS="${ALPHAS:-0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25 0.5 0.75 1 1.5 2 3 4}"
MIN_TP_PRESERVED="${MIN_TP_PRESERVED:-0.95}"
ALWAYS_ALPHA="${ALWAYS_ALPHA:-1.0}"
GATE_SCORE="${GATE_SCORE:-margin+tail}"
TARGET_TRIGGER_RATE="${TARGET_TRIGGER_RATE:-0.3}"
STEPS="${STEPS:-stage2 report}"

SUBSPACE_ARGS=(${SUBSPACES})
if [[ "${RANDOM_REPEATS}" -gt 0 ]]; then
  for IDX in $(seq 0 $((RANDOM_REPEATS - 1))); do
    PADDED="$(printf '%02d' "${IDX}")"
    SUBSPACE_ARGS+=("random12_s${PADDED}")
  done
fi

for STEP in ${STEPS}; do
  case "${STEP}" in
    stage2)
      "${PYTHON_BIN}" scripts/analyze_mechanism_mitigation_stage2.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --subspaces ${SUBSPACE_ARGS[@]} \
        --alphas ${ALPHAS} \
        --split-policy fixed_ids \
        --split-dir "${SPLIT_DIR}" \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${MITIGATION_DIR}/stage2_subspace_icd"
      ;;
    report)
      "${PYTHON_BIN}" scripts/build_llava13b_minimal_replication_report.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --stage2-dir "${MITIGATION_DIR}/stage2_subspace_icd" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --layer "${LAYER}" \
        --always-alpha "${ALWAYS_ALPHA}" \
        --gate-score "${GATE_SCORE}" \
        --target-trigger-rate "${TARGET_TRIGGER_RATE}" \
        --output-dir "${MITIGATION_DIR}/report"
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
