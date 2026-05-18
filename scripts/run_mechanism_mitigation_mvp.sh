#!/usr/bin/env bash
set -euo pipefail

# MVP from mitigation_plan.md:
#   Stage 1: operator correction decomposition
#   Stage 2: logit-level subspace-filtered VCD/ICD
#   Stage 3: geometry-guided selective VCD/ICD routing
#
# Typical full run:
#   bash scripts/run_mechanism_mitigation_mvp.sh
#
# Fast smoke on a few samples:
#   MAX_SAMPLES=32 STEPS="geometry cpu" bash scripts/run_mechanism_mitigation_mvp.sh
#
# CPU-only reanalysis after geometry/VCD artifacts already exist:
#   STEPS="cpu report" bash scripts/run_mechanism_mitigation_mvp.sh

STEPS="${STEPS:-geometry cpu selective report}"
MECH_DIR="${MECH_DIR:-outputs/mechanism_mitigation}"

for STEP in ${STEPS}; do
  case "${STEP}" in
    geometry)
      OUTPUT_DIR="${MECH_DIR}/operator_geometry" \
      bash scripts/run_gpu_mechanism_mitigation_operator_geometry.sh
      ;;
    cpu)
      MECH_DIR="${MECH_DIR}" \
      STEPS="stage1 stage2" \
      bash scripts/run_cpu_mechanism_mitigation_stage1_stage2.sh
      ;;
    selective)
      bash scripts/run_gpu_mechanism_mitigation_stage3_selective.sh
      ;;
    report)
      MECH_DIR="${MECH_DIR}" \
      STEPS="report" \
      bash scripts/run_cpu_mechanism_mitigation_stage1_stage2.sh
      ;;
    *)
      echo "Unknown STEP: ${STEP}" >&2
      exit 2
      ;;
  esac
done
