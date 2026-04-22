# VGS

Engineering scaffold for validating blind-reference visual grounding subspaces on
`LLaVA-1.5-7B` with `POPE` as the main benchmark.

The roadmap lives in [TODO_list.md](TODO_list.md). This repository turns that
roadmap into a staged, reproducible pipeline:

1. Run POPE evaluation and save raw/parsed predictions.
2. Dump paired image-conditioned and blind hidden states.
3. Build layerwise difference matrices.
4. Analyze SVD spectra, stability, K sensitivity, probes, layerwise geometry,
   interventions, and semantic interpretation.

## Layout

```text
configs/                 experiment defaults
data/                    local datasets and cache
outputs/                 generated predictions, tensors, tables, and plots
scripts/                 explicit CLI entry points for each stage
src/vgs/                 shared implementation used by scripts
notes/                   experiment logs and evidence tracking
tests/                   lightweight unit tests
```

## Quick Start

```bash
python -m pip install -e ".[dev]"
python scripts/run_pope_eval.py --help
python scripts/analyze_spectrum.py --layers 8 12 --output-dir outputs/svd
```

Every script accepts explicit CLI arguments and writes a JSON summary. Major runs
append one line to [notes/experiment_log.md](notes/experiment_log.md).
