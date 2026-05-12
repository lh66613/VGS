#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_detector_experiments.py \
  --layers 24 \
  --train-subset random \
  --calibration-subset popular \
  --test-subset adversarial \
  --top-k-grid 4 16 64 256 \
  --dim-k-grid 4 16 64 256 \
  --tail-start 257 \
  --tail-end 1024 \
  --pls-k 32 \
  --random-dim 64 \
  --trigger-rates 0.1 0.2 0.3 \
  --bootstrap-repeats 1000 \
  --bootstrap-trigger-rate 0.2 \
  --output-dir outputs/detector_minimal_package \
  --notes-path notes/detector_experiment_prep.md
