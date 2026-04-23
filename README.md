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
/data/lh/.conda/envs/after/bin/python scripts/validate_pope_data.py
/data/lh/.conda/envs/after/bin/python scripts/run_pope_eval.py --help
/data/lh/.conda/envs/after/bin/python scripts/analyze_spectrum.py --layers 8 12 --output-dir outputs/svd
```

Every script accepts explicit CLI arguments and writes a JSON summary. Major runs
append one line to [notes/experiment_log.md](notes/experiment_log.md).

## CPU Pipeline Smoke Test

These commands do not load LLaVA. They create tiny fake hidden states and verify
the downstream analysis stages:

```bash
/data/lh/.conda/envs/after/bin/python scripts/create_smoke_artifacts.py --layers 8 12
/data/lh/.conda/envs/after/bin/python scripts/build_difference_matrix.py --layers 8 12 --hidden-states-dir outputs/hidden_states_smoke --output-dir outputs/svd_smoke
/data/lh/.conda/envs/after/bin/python scripts/analyze_spectrum.py --layers 8 12 --matrix-dir outputs/svd_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke
/data/lh/.conda/envs/after/bin/python scripts/analyze_k_sensitivity.py --layers 8 12 --k-grid 4 8 16 --svd-dir outputs/svd_smoke --matrix-dir outputs/svd_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke --stability-sample-size 1024
/data/lh/.conda/envs/after/bin/python scripts/train_probe.py --layers 8 12 --k-grid 4 8 16 --feature-family projected_difference random_difference difference --predictions outputs/predictions/smoke_predictions.jsonl --hidden-states-dir outputs/hidden_states_smoke --svd-dir outputs/svd_smoke --output-dir outputs/probes_smoke
/data/lh/.conda/envs/after/bin/python scripts/compare_features.py --probe-dir outputs/probes_smoke --output-dir outputs/probes_smoke
/data/lh/.conda/envs/after/bin/python scripts/layerwise_analysis.py --layers 8 12 --k-grid 4 8 16 --svd-dir outputs/svd_smoke --probe-dir outputs/probes_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke
```

When you run the real GPU stages, keep the same output contract:

```text
outputs/predictions/pope_predictions.jsonl
outputs/hidden_states/layer_{l}.pt
```

## GPU Stages For Real Runs

Run these only in a CUDA-visible shell. They load `LLaVA-1.5-7B`.

```bash
/data/lh/.conda/envs/after/bin/python scripts/run_pope_eval.py --max-samples 10
/data/lh/.conda/envs/after/bin/python scripts/dump_hidden_states.py --layers 8 12 16 20 24 28 32 --predictions outputs/predictions/pope_predictions.jsonl
```

The same two commands are wrapped by:

```bash
MAX_SAMPLES=10 bash scripts/run_gpu_pope_and_hidden.sh
```

After those artifacts exist, the CPU-side stages can be run without loading the
model:

```bash
/data/lh/.conda/envs/after/bin/python scripts/build_difference_matrix.py --layers 8 12 16 20 24 28 32
/data/lh/.conda/envs/after/bin/python scripts/analyze_spectrum.py --layers 8 12 16 20 24 28 32
/data/lh/.conda/envs/after/bin/python scripts/analyze_k_sensitivity.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64 --stability-method randomized --stability-sample-size 1024
/data/lh/.conda/envs/after/bin/python scripts/train_probe.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64 --feature-family raw_img raw_blind difference projected_difference random_difference pca_img
/data/lh/.conda/envs/after/bin/python scripts/compare_features.py
/data/lh/.conda/envs/after/bin/python scripts/layerwise_analysis.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64
```

For deeper Stage C follow-up:

```bash
/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_c_deep.py \
  --layers 8 12 16 20 24 28 32 \
  --k-grid 4 8 16 32 64 128 256 \
  --focus-layers 20 24 28 32

/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_c_supervised.py \
  --layers 8 12 16 20 24 28 32 \
  --k-grid 4 8 16 32 64 128 256 512 1024 \
  --focus-layers 20 24 28 32

/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_c_coordinate_control.py \
  --layers 20 24 32
```

The CPU stages are also wrapped by:

```bash
bash scripts/run_cpu_stage_a.sh
bash scripts/run_cpu_stage_c_d.sh
```

Stage B is split into a GPU hidden-state step and a CPU analysis step. The
default pilot uses FP/TN samples with 256 examples per outcome on L20/L24/L32:

```bash
bash scripts/run_gpu_stage_b_conditions.sh
bash scripts/run_cpu_stage_b.sh
```

To prepare only the condition plan without loading the model:

```bash
/data/lh/.conda/envs/after/bin/python scripts/prepare_stage_b_conditions.py \
  --max-samples-per-outcome 256
```

Stage E causal intervention pilot is GPU-only. It first runs a hook precheck,
then tests tail ablation on TN samples and supervised/TN-correction steering on
FP samples:

```bash
bash scripts/run_gpu_stage_e.sh
```

For a smaller first pass:

```bash
LAYERS="24" MAX_SAMPLES_PER_OUTCOME=4 ALPHA_GRID="4 5 6 7 8" bash scripts/run_gpu_stage_e.sh
```

If the generated text does not change, inspect
`outputs/interventions/intervention_precheck.csv`: the precheck records
next-token logit deltas as well as final decoded text.

For a clean TN tail-ablation dose curve:

```bash
LAYERS="20 24" OUTCOMES="TN" FAMILIES="tail" GRANULARITIES="last_token" \
  ALPHA_GRID="4 5 6 7 8" bash scripts/run_gpu_stage_e.sh
```

For a small granularity comparison:

```bash
LAYERS="24" OUTCOMES="TN" FAMILIES="tail" GRANULARITIES="last_token full_sequence" \
  MAX_SAMPLES_PER_OUTCOME=4 ALPHA_GRID="4 6 8" bash scripts/run_gpu_stage_e.sh
```

## Current Local Setup

- Conda environment: `after`
- Model path: `/data/lh/ModelandDataset/llava-1.5-7b-hf`
- Model implementation: Hugging Face `LlavaForConditionalGeneration`
- Main POPE family: `coco`
- POPE question files:
  - `data/pope/questions/coco_pope_random.json`
  - `data/pope/questions/coco_pope_popular.json`
  - `data/pope/questions/coco_pope_adversarial.json`

The current scaffold can validate local files without GPU. Full LLaVA inference
needs a CUDA-visible environment because the checkpoint is too large for a
practical CPU run.
