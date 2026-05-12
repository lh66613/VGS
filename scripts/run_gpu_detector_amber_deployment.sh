#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/lh/ModelandDataset/llava-1.5-7b-hf}"
MODEL_FAMILY="${MODEL_FAMILY:-llava}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
AMBER_PREDICTIONS="${AMBER_PREDICTIONS:-outputs/stage_n_external_full/amber_predictions.jsonl}"
POPE_MARGIN_SCORES="${POPE_MARGIN_SCORES:-outputs/margins/pope_margin_scores.csv}"
AMBER_MARGIN_DIR="${AMBER_MARGIN_DIR:-outputs/margins_amber}"
AMBER_MARGIN_SCORES="${AMBER_MARGIN_SCORES:-${AMBER_MARGIN_DIR}/pope_margin_scores.csv}"
MERGED_MARGIN_SCORES="${MERGED_MARGIN_SCORES:-outputs/margins/pope_plus_amber_margin_scores.csv}"
STAGE_T_DIR="${STAGE_T_DIR:-outputs/stage_t_detector_amber_margin}"
EXTERNAL_OUTPUT_DIR="${EXTERNAL_OUTPUT_DIR:-outputs/stage_t_external_amber_margin_detector}"
TOP_K_GRID="${TOP_K_GRID:-4 16 64 256}"
TARGET_RATES="${TARGET_RATES:-0.2 0.3}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ "${FORCE_RERUN}" == "1" || ! -s "${AMBER_MARGIN_SCORES}" ]]; then
  "${PYTHON_BIN}" scripts/dump_pope_margins.py \
    --model-path "${MODEL_PATH}" \
    --model-family "${MODEL_FAMILY}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --predictions "${AMBER_PREDICTIONS}" \
    --output-dir "${AMBER_MARGIN_DIR}"
else
  echo "Skip AMBER margins; found ${AMBER_MARGIN_SCORES}"
fi

"${PYTHON_BIN}" scripts/merge_margin_scores.py \
  "${POPE_MARGIN_SCORES}" \
  "${AMBER_MARGIN_SCORES}" \
  --output "${MERGED_MARGIN_SCORES}"

"${PYTHON_BIN}" scripts/analyze_stage_t_selective_correction.py \
  --layers 24 \
  --output-dir "${STAGE_T_DIR}" \
  --top-k-grid ${TOP_K_GRID} \
  --margin-scores "${MERGED_MARGIN_SCORES}"

"${PYTHON_BIN}" scripts/build_stage_t_external_warning.py \
  --stage-t-dir "${STAGE_T_DIR}" \
  --output-dir "${EXTERNAL_OUTPUT_DIR}" \
  --target-rates ${TARGET_RATES} \
  --scores \
    low_margin_probe \
    low_margin_plus_top_16_probe \
    low_margin_plus_tail_257_1024_probe \
    low_margin_plus_full_probe \
    top_16_probe \
    tail_257_1024_probe \
    full_probe
