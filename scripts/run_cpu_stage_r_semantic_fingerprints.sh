#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/build_stage_r_semantic_fingerprints.py
