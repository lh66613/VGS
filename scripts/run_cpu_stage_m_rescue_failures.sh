#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/analyze_stage_m_rescue_failures.py
