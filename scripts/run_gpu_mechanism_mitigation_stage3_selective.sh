#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_selective_correction_fixed_ids}"
LAYERS="${LAYERS:-24}"
TAIL_BAND="${TAIL_BAND:-257-1024}"
TOP_K_GRID="${TOP_K_GRID:-4 64}"
TRIGGER_RATES="${TRIGGER_RATES:-0.1 0.2 0.3}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
VCD_OPERATORS="${VCD_OPERATORS:-icd_blind vcd_diffusion}"
RUN_GATE_ANALYSIS="${RUN_GATE_ANALYSIS:-1}"
RUN_VCD_SWEEP="${RUN_VCD_SWEEP:-1}"
BUILD_REPORT="${BUILD_REPORT:-1}"
MARGIN_SCORES="${MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"

if [[ "${RUN_GATE_ANALYSIS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/analyze_stage_t_selective_correction.py \
    --split-policy fixed_ids \
    --output-dir "${STAGE_T_DIR}" \
    --layers ${LAYERS} \
    --tail-band "${TAIL_BAND}" \
    --top-k-grid ${TOP_K_GRID} \
    --trigger-rates ${TRIGGER_RATES} \
    --margin-scores "${MARGIN_SCORES}"
fi

if [[ "${RUN_VCD_SWEEP}" == "1" ]]; then
  VCD_OPERATORS="${VCD_OPERATORS}" \
  STAGE_T_DIR="${STAGE_T_DIR}" \
  TARGET_RATES="${TARGET_RATES}" \
  bash scripts/run_gpu_stage_t_vcd_sweep.sh
fi

if [[ "${BUILD_REPORT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/build_mechanism_mitigation_mvp_report.py \
    --stage1-dir outputs/mechanism_mitigation/stage1_vcd_decomposition \
    --stage2-dir outputs/mechanism_mitigation/stage2_subspace_vcd \
    --stage3-dir "${STAGE_T_DIR}" \
    --output-dir outputs/mechanism_mitigation/mvp
fi
