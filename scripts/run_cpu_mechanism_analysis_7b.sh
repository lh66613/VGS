#!/usr/bin/env bash
set -euo pipefail

for ARG in "$@"; do
  if [[ "${ARG}" == *=* ]]; then
    export "${ARG}"
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/vlm-exp/bin/python}"
ROOT_DIR="${ROOT_DIR:-outputs/mechanism_mitigation/mechanism_analysis}"
OPERATOR_GEOMETRY="${OPERATOR_GEOMETRY:-${ROOT_DIR}/operator_geometry_7b_icd/operator_geometry.csv}"
FROZEN_OPERATOR_GEOMETRY="${FROZEN_OPERATOR_GEOMETRY:-${ROOT_DIR}/frozen_operator_geometry/operator_geometry.csv}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
MARGIN_SCORES="${MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"
SPLIT_DIR="${SPLIT_DIR:-outputs/splits}"
PAPER_TABLES_DIR="${PAPER_TABLES_DIR:-outputs/mechanism_mitigation/paper_tables}"
ALPHAS="${ALPHAS:-0.005 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25 0.5 1 2 4}"
MIN_TP_PRESERVED="${MIN_TP_PRESERVED:-0.95}"
MIN_CALIBRATION_ACCURACY_DELTA="${MIN_CALIBRATION_ACCURACY_DELTA:-0.0}"
STEPS="${STEPS:-freeze drift exact band_scan logit_shift internal spectrum flipped split report}"

for STEP in ${STEPS}; do
  case "${STEP}" in
    canonical)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_canonical_protocol.py \
        --output-path "${ROOT_DIR}/canonical_protocol.md"
      ;;
    frozen_geometry)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_frozen_geometry_from_cache.py \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --output-dir "${ROOT_DIR}/frozen_operator_geometry"
      ;;
    frozen_spectrum)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_frozen_spectrum_curve.py \
        --operator-geometry "${FROZEN_OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --alphas ${ALPHAS} \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${ROOT_DIR}/frozen_spectrum_curve_7b"
      ;;
    frozen_flipped)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_frozen_flipped_subset.py \
        --operator-geometry "${FROZEN_OPERATOR_GEOMETRY}" \
        --selected-table "${ROOT_DIR}/frozen_spectrum_curve_7b/frozen_spectrum_selected.csv" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --output-dir "${ROOT_DIR}/frozen_flipped_subset_7b"
      ;;
    frozen_split)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_split_robustness.py \
        --operator-geometry "${FROZEN_OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --alphas ${ALPHAS} \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --min-calibration-accuracy-delta "${MIN_CALIBRATION_ACCURACY_DELTA}" \
        --output-dir "${ROOT_DIR}/frozen_split_spectral_selection_7b"
      ;;
    freeze)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_freeze_table.py \
        --paper-tables-dir "${PAPER_TABLES_DIR}" \
        --output-dir "${ROOT_DIR}/frozen_baseline"
      ;;
    drift)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_drift_audit.py \
        --current-geometry "${OPERATOR_GEOMETRY}" \
        --current-summary "${ROOT_DIR}/operator_geometry_7b_icd/operator_geometry_summary.json" \
        --output-dir "${ROOT_DIR}/drift_audit"
      ;;
    exact)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_exact_reproduction.py \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --output-dir "${ROOT_DIR}/exact_reproduction"
      ;;
    band_scan)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_band_scan.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --alphas ${ALPHAS} \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${ROOT_DIR}/band_scan_7b"
      ;;
    logit_shift)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_logit_shift.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --split-dir "${SPLIT_DIR}" \
        --band-scan-table "${ROOT_DIR}/band_scan_7b/band_scan_table.csv" \
        --output-dir "${ROOT_DIR}/logit_shift_7b"
      ;;
    internal)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_internal_contribution.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --alphas ${ALPHAS} \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${ROOT_DIR}/internal_contribution_7b"
      ;;
    spectrum)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_spectrum_curve.py \
        --operator-geometry "${OPERATOR_GEOMETRY}" \
        --predictions "${PREDICTIONS}" \
        --margin-scores "${MARGIN_SCORES}" \
        --split-dir "${SPLIT_DIR}" \
        --alphas ${ALPHAS} \
        --min-tp-preserved "${MIN_TP_PRESERVED}" \
        --output-dir "${ROOT_DIR}/spectrum_curve_7b"
      ;;
    flipped)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_flipped_subset.py \
        --sample-predictions "${ROOT_DIR}/band_scan_7b/stage2/sample_predictions.csv" \
        --selected-table "${ROOT_DIR}/band_scan_7b/band_scan_table.csv" \
        --output-dir "${ROOT_DIR}/flipped_subset_7b"
      ;;
    split)
      bash scripts/run_cpu_mechanism_analysis_split_robustness.sh \
        PYTHON_BIN="${PYTHON_BIN}" \
        ROOT_DIR="${ROOT_DIR}" \
        OPERATOR_GEOMETRY="${OPERATOR_GEOMETRY}" \
        PREDICTIONS="${PREDICTIONS}" \
        MARGIN_SCORES="${MARGIN_SCORES}" \
        ALPHAS="${ALPHAS}" \
        MIN_TP_PRESERVED="${MIN_TP_PRESERVED}"
      ;;
    amber)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_amber_minimal.py \
        --pope-band-scan "${ROOT_DIR}/band_scan_7b/band_scan_table.csv" \
        --output-dir "${ROOT_DIR}/amber_minimal"
      ;;
    report)
      "${PYTHON_BIN}" scripts/build_mechanism_analysis_final_report.py \
        --root-dir "${ROOT_DIR}"
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
