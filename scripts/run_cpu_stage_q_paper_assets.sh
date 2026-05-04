#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vgs}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/build_stage_q_paper_assets.py
