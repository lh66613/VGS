#!/usr/bin/env bash
set -euo pipefail

LAYERS="${LAYERS:-8 12 16 20 24 28 32}"
PYTHON_BIN="${PYTHON_BIN:-/data/lh/.conda/envs/after/bin/python}"

"${PYTHON_BIN}" scripts/prepare_stage_i_protocol.py --layers ${LAYERS}
