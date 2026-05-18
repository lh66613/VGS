#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MECH_DIR="${MECH_DIR:-outputs/mechanism_mitigation}"
OPERATOR_GEOMETRY="${OPERATOR_GEOMETRY:-${MECH_DIR}/operator_geometry/operator_geometry.csv}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
MARGIN_SCORES="${MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"
SPLIT_POLICY="${SPLIT_POLICY:-fixed_ids}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
BANDS="${BANDS:-full top4 top16 band5_16 band17_64 band65_256 tail257_1024 top4_complement random12 random4_complement random_tail_dim}"
SUBSPACES="${SUBSPACES:-full top4 top16 band5_16 tail257_1024 top4_complement random12 random4_complement random_tail_dim}"
ALPHAS="${ALPHAS:-0.25 0.5 1 2 4}"
MIN_TP_PRESERVED="${MIN_TP_PRESERVED:-0.95}"
STEPS="${STEPS:-stage1 stage2 report}"

for STEP in ${STEPS}; do
  case "${STEP}" in
    stage1)
      "${PYTHON_BIN}" scripts/analyze_mechanism_mitigation_stage1.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --bands ${BANDS} \
        --output-dir "${MECH_DIR}/stage1_vcd_decomposition"
      ;;
    stage2)
      "${PYTHON_BIN}" scripts/analyze_mechanism_mitigation_stage2.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --subspaces ${SUBSPACES} \
        --alphas ${ALPHAS} \
        --split-policy "${SPLIT_POLICY}" \
        --split-dir "${SPLIT_DIR}" \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${MECH_DIR}/stage2_subspace_vcd"
      ;;
    report)
      "${PYTHON_BIN}" scripts/build_mechanism_mitigation_mvp_report.py \
        --stage1-dir "${MECH_DIR}/stage1_vcd_decomposition" \
        --stage2-dir "${MECH_DIR}/stage2_subspace_vcd" \
        --stage3-dir "${STAGE3_DIR:-outputs/stage_t_selective_correction_fixed_ids}" \
        --output-dir "${MECH_DIR}/mvp"
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
