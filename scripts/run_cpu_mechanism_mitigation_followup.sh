#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MECH_DIR="${MECH_DIR:-outputs/mechanism_mitigation}"
OPERATOR_GEOMETRY="${OPERATOR_GEOMETRY:-${MECH_DIR}/operator_geometry/operator_geometry.csv}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
MARGIN_SCORES="${MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
RANDOM_REPEATS="${RANDOM_REPEATS:-0}"
SUBSPACES="${SUBSPACES:-full band5_16 top4_complement random12 random4_complement random_tail_dim tail257_1024}"
ALPHAS="${ALPHAS:-0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25 0.5 1 2 4}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
STEPS="${STEPS:-stage2 reverse followup paper_tables}"

SUBSPACE_ARGS=(${SUBSPACES})
if [[ "${RANDOM_REPEATS}" -gt 0 ]]; then
  for IDX in $(seq 0 $((RANDOM_REPEATS - 1))); do
    PADDED="$(printf '%02d' "${IDX}")"
    SUBSPACE_ARGS+=("random12_s${PADDED}" "random4_complement_s${PADDED}" "random_tail_dim_s${PADDED}")
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
        --output-dir "${MECH_DIR}/stage2_subspace_vcd"
      ;;
    reverse)
      "${PYTHON_BIN}" scripts/analyze_mechanism_mitigation_stage2.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --subspaces ${SUBSPACE_ARGS[@]} \
        --alphas ${ALPHAS} \
        --split-policy subset_transfer \
        --calibration-subset adversarial \
        --test-subset random \
        --output-dir "${MECH_DIR}/stage2_reverse_subspace_vcd"
      ;;
    followup)
      "${PYTHON_BIN}" scripts/build_mechanism_mitigation_followup.py \
        --stage2-dir "${MECH_DIR}/stage2_subspace_vcd" \
        --reverse-stage2-dir "${MECH_DIR}/stage2_reverse_subspace_vcd" \
        --stage3-dir "${STAGE3_DIR:-outputs/stage_t_selective_correction_fixed_ids}" \
        --predictions "${PREDICTIONS}" \
        --n-bootstrap "${N_BOOTSTRAP}" \
        --output-dir "${MECH_DIR}/followup"
      ;;
    paper_tables)
      "${PYTHON_BIN}" scripts/build_mechanism_mitigation_paper_tables.py \
        --followup-dir "${MECH_DIR}/followup" \
        --output-dir "${MECH_DIR}/paper_tables"
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
