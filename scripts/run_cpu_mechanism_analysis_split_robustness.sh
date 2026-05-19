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
PREDICTIONS="${PREDICTIONS:-outputs/predictions/pope_predictions.jsonl}"
MARGIN_SCORES="${MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"
PAIRS="${PAIRS:-random:popular random:adversarial popular:adversarial adversarial:random}"
FIXED_SUBSPACE="${FIXED_SUBSPACE:-band5_16}"
CANDIDATE_SUBSPACES="${CANDIDATE_SUBSPACES:-band1_12 band5_16 band9_20 band13_24 band17_28 band21_32 band25_36 band29_40 band33_44 band37_48 band41_52 band45_56 band49_60 band53_64}"
ALPHAS="${ALPHAS:-0.005 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25 0.5 1 2 4}"
MIN_TP_PRESERVED="${MIN_TP_PRESERVED:-0.95}"
MIN_CALIBRATION_ACCURACY_DELTA="${MIN_CALIBRATION_ACCURACY_DELTA:-0.0}"

"${PYTHON_BIN}" scripts/build_mechanism_analysis_split_robustness.py \
  --operator-geometry "${OPERATOR_GEOMETRY}" \
  --predictions "${PREDICTIONS}" \
  --margin-scores "${MARGIN_SCORES}" \
  --pairs ${PAIRS} \
  --fixed-subspace "${FIXED_SUBSPACE}" \
  --candidate-subspaces ${CANDIDATE_SUBSPACES} \
  --alphas ${ALPHAS} \
  --min-tp-preserved "${MIN_TP_PRESERVED}" \
  --min-calibration-accuracy-delta "${MIN_CALIBRATION_ACCURACY_DELTA}" \
  --output-dir "${ROOT_DIR}/split_robustness_7b"
