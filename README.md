# VGS

**Blind-Reference Differencing Reveals Layered Correction Geometry in
Vision-Language Hallucination**

This repository contains the experiment pipeline and paper artifacts for a
mechanistic study of hallucination in vision-language models. The project starts
from a paired hidden-state construction: for the same POPE question, compare the
image-conditioned representation with a blind/text-only representation, then ask
where hallucination-sensitive evidence correction lives.

```text
z_img   = hidden state from image + question
z_blind = hidden state from blind question-only input
D_l     = z_blind_l - z_img_l
```

The current evidence supports a **layered visual-evidence correction geometry**
rather than a single universal "visual grounding subspace": dominant variance
directions form a stable image-induced backbone, while FP/TN signal appears in
full difference vectors, mid/high-dimensional SVD coordinates, residual/tail
bands, and supervised evidence-specific subspaces.

## Highlights

- Main model and benchmark: `LLaVA-1.5-7B` on POPE COCO random, popular, and
  adversarial subsets.
- Main protocol: paired `image + question` vs `blind question` hidden states at
  layers `8 12 16 20 24 28 32`, primarily read at `last_prompt_token`.
- Best mechanism framing: **evidence-sensitive correction geometry**.
- Strongest detection-style result: full-difference FP/TN probing reaches
  AUROC `0.721` in the 5-seed Stage P analysis.
- Key negative result: top variance directions are not the hallucination
  decision geometry. L24 top-4 coordinates have AUROC `0.471` despite explaining
  a large fraction of variance.
- Key causal probe: ablating the L24 residual/tail slice `257-1024` can flip
  correct TN first-token `No` decisions to `Yes` under a dose curve, while a
  norm-matched random-tail last-token control stays at `0.00` yes rate.
- External validity is modest: POPE-trained risk geometry transfers above
  chance to AMBER, with top rows around AUROC `0.63-0.665`.
- Checkpoint-level replication is available on `LLaVA-1.5-13B`; broader
  cross-architecture experiments are in Phase 3 and should be interpreted
  cautiously.

## Repository Layout

```text
configs/                 default experiment configuration
data/                    local POPE/AMBER data and caches
notes/                   experiment log, claims, design decisions, paper framing
outputs/                 generated tensors, predictions, tables, plots, figures
outputs/paper_figures/   paper-ready PDF figures from Stage Q
outputs/paper_tables/    paper-ready CSV tables from Stage Q
scripts/                 CLI entry points and stage wrappers
src/vgs/                 shared implementation for datasets, VLMs, geometry, stages
tests/                   lightweight unit tests for local utilities
```

Most scripts write a JSON summary and append an entry to
[`notes/experiment_log.md`](notes/experiment_log.md). The condensed current
paper-level summary lives in
[`notes/current_experiment_summary.md`](notes/current_experiment_summary.md).

## Paper Artifacts

Stage Q builds the tables and figures used by the current paper draft.

| Artifact | Path | Purpose |
| --- | --- | --- |
| Table 1 | `outputs/paper_tables/table1_pope_summary.csv` | POPE subset accuracy and outcomes |
| Table 2 | `outputs/paper_tables/table2_feature_comparison.csv` | Feature/probe comparison |
| Table 3 | `outputs/paper_tables/table3_controls.csv` | Shuffle, Gaussian, and label controls |
| Table 4 | `outputs/paper_tables/table4_intervention.csv` | Tail ablation and rescue interventions |
| Figure 1 | `outputs/paper_figures/fig1_method_overview.pdf` | Method overview |
| Figure 2 | `outputs/paper_figures/fig2_variance_vs_auroc.pdf` | Variance vs AUROC dissociation |
| Figure 3 | `outputs/paper_figures/fig3_condition_geometry.pdf` | Matched vs mismatch geometry |
| Figure 4 | `outputs/paper_figures/fig4_intervention_dose.pdf` | Intervention dose curves |
| Figure 5 | `outputs/paper_figures/fig5_layered_geometry.pdf` | Layered geometry summary |

Regenerate them with:

```bash
bash scripts/run_cpu_stage_q_paper_assets.sh
```

## Result Snapshot

### POPE Prediction Quality

| Subset | N | Accuracy | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 3000 | 0.884 | 1202 | 1450 | 50 | 298 |
| popular | 3000 | 0.864 | 1202 | 1390 | 110 | 298 |
| adversarial | 3000 | 0.838 | 1202 | 1311 | 189 | 298 |
| overall | 9000 | 0.862 | 3606 | 4151 | 349 | 894 |

### FP/TN Geometry

| Setting | Layer | Result |
| --- | ---: | --- |
| Full difference, single run | 24 | AUROC `0.694` |
| Full SVD coordinates, coordinate control | 20 | AUROC `0.734` |
| Evidence-specific PLS FP/TN | 24 | AUROC `0.720` at K=32 |
| Full difference, Stage P 5 seeds | 24 | AUROC `0.721`, 95% CI `0.699-0.741` |
| Top-256 SVD, Stage P 5 seeds | 20 | AUROC `0.677` |
| Tail `257-1024`, Stage P 5 seeds | 32 | AUROC `0.667` |
| Top-4 SVD, Stage P 5 seeds | 24 | AUROC `0.471` |

### Matched vs Mismatched Evidence

Residual/tail bands react more clearly to correct visual evidence than the
top-variance backbone. Stage B reports positive matched-vs-mismatch gaps in the
tail band `257-1024`:

| Layer | Tail matched-random | Tail matched-adversarial |
| ---: | ---: | ---: |
| 20 | +5.7 | +11.8 |
| 24 | +14.2 | +25.7 |
| 32 | +17.0 | +39.1 |

The supervised FP-minus-TN decision gap is also largest under matched evidence:

| Layer | Matched | Random mismatch | Adversarial mismatch |
| ---: | ---: | ---: | ---: |
| 20 | +0.623 | +0.102 | +0.133 |
| 24 | +0.925 | +0.155 | +0.231 |
| 32 | +0.834 | +0.101 | +0.196 |

### Intervention

The strongest intervention evidence is TN tail ablation at L24:

| Intervention | Granularity | Alpha | Yes rate |
| --- | --- | ---: | ---: |
| L24 tail ablation | full sequence | 5 | 0.50 |
| L24 tail ablation | full sequence | 6 | 1.00 |
| L24 tail ablation | last token | 5 | 0.25 |
| L24 tail ablation | last token | 6 | 0.625 |
| Norm-matched random tail | last token | 6 | 0.00 |

FP rescue remains boundary-local: Stage M rescues only a small number of
borderline FP samples, and random/global/local TN controls are close on the
gated subset. The paper framing should treat rescue as a causal probe, not as a
reliable mitigation method.

### External and Cross-Model Checks

| Check | Current status |
| --- | --- |
| AMBER full discriminative set | Overall accuracy `0.816`; top transfer rows around AUROC `0.63-0.665` |
| `LLaVA-1.5-13B` | Qualitative LLaVA-family replication: full difference AUROC `0.736/0.726/0.723` at L20/L24/L32; top-4 remains weak |
| Qwen2-VL / Qwen2.5-VL Phase 3 | Artifacts exist under `outputs/stage_o_cross_model/`; margin and raw hidden-state baselines saturate, so these are architecture-sensitivity checks rather than confirmed generality |
| InternVL Phase 3 | Wrapper support exists; artifacts should be checked before claiming a result |

## Installation

The project is packaged with `pyproject.toml` and expects Python `>=3.10`.

```bash
python -m pip install -e ".[dev]"
```

The completed LLaVA-1.5 experiments used the local `after` conda environment and
`transformers==4.37.2`. Phase 3 Qwen/InternVL wrappers use the newer local
`vlm-exp` environment because Qwen2-VL support requires newer Transformers.

Default local paths are recorded in [`configs/default.yaml`](configs/default.yaml).
For a fresh machine, either edit that file or pass `--model-path` explicitly.

## Quick CPU Smoke Test

These commands do not load a VLM. They build tiny fake hidden-state artifacts and
exercise the downstream geometry pipeline:

```bash
python scripts/create_smoke_artifacts.py --layers 8 12
python scripts/build_difference_matrix.py \
  --layers 8 12 \
  --hidden-states-dir outputs/hidden_states_smoke \
  --output-dir outputs/svd_smoke
python scripts/analyze_spectrum.py \
  --layers 8 12 \
  --matrix-dir outputs/svd_smoke \
  --output-dir outputs/svd_smoke \
  --plot-dir outputs/plots_smoke
python scripts/train_probe.py \
  --layers 8 12 \
  --k-grid 4 8 16 \
  --feature-family projected_difference random_difference difference \
  --predictions outputs/predictions/smoke_predictions.jsonl \
  --hidden-states-dir outputs/hidden_states_smoke \
  --svd-dir outputs/svd_smoke \
  --output-dir outputs/probes_smoke
python scripts/compare_features.py \
  --probe-dir outputs/probes_smoke \
  --output-dir outputs/probes_smoke
```

Run the unit tests with:

```bash
pytest
```

## Reproducing The Main Pipeline

### 1. Validate POPE data

```bash
python scripts/validate_pope_data.py
```

Expected POPE question files:

```text
data/pope/questions/coco_pope_random.json
data/pope/questions/coco_pope_popular.json
data/pope/questions/coco_pope_adversarial.json
```

### 2. Run LLaVA POPE prediction and paired hidden-state extraction

Run this in a CUDA-visible shell:

```bash
bash scripts/run_gpu_pope_and_hidden.sh
```

The core output contract is:

```text
outputs/predictions/pope_predictions.jsonl
outputs/hidden_states/layer_{l}.pt
```

For a smaller pilot:

```bash
MAX_SAMPLES=10 bash scripts/run_gpu_pope_and_hidden.sh
```

Stage B condition geometry needs matched, random-mismatch, and
adversarial-mismatch hidden states. Dump those on GPU before running the Stage B
CPU analysis:

```bash
bash scripts/run_gpu_stage_b_conditions.sh
```

### 3. Build the CPU-side geometry analyses

```bash
bash scripts/run_cpu_stage_a.sh
bash scripts/run_cpu_stage_c_d.sh
bash scripts/run_cpu_stage_b.sh
bash scripts/run_cpu_stage_j_controls.sh
bash scripts/run_cpu_stage_k_positions.sh
bash scripts/run_cpu_stage_l_evidence_subspace.sh
bash scripts/run_cpu_stage_p_stats.sh
bash scripts/run_cpu_stage_q_paper_assets.sh
```

### 4. Run intervention and rescue studies

Stage E and the Stage M rescue run load the VLM and should be executed on GPU.
Stage M also needs CPU preparation and post-hoc analysis:

```bash
bash scripts/run_gpu_stage_e.sh
bash scripts/run_cpu_stage_m_memory_bank.sh
bash scripts/run_cpu_stage_m_retrieval_plan.sh
bash scripts/run_gpu_stage_m_local_rescue.sh
bash scripts/run_cpu_stage_m_local_rescue.sh
bash scripts/run_cpu_stage_m_rescue_failures.sh
```

### 5. External validity and replication

AMBER:

```bash
bash scripts/run_cpu_stage_n_amber_prepare.sh
bash scripts/run_gpu_stage_n_amber.sh
bash scripts/run_cpu_stage_n_transfer.sh
```

LLaVA-family checkpoint replication:

```bash
MODEL_ALIAS=llava_13b \
MODEL_PATH=/data/lh/ModelandDataset/llava-1.5-13b-hf \
bash scripts/run_gpu_stage_o_cross_model.sh

MODEL_ALIAS=llava_13b bash scripts/run_cpu_stage_o_cross_model.sh
```

Phase 3 cross-architecture minimal replication:

```bash
MODEL_FAMILY=qwen2_5_vl \
MODEL_ALIAS=qwen2_5_vl_7b \
MODEL_PATH=/data/lh/ModelandDataset/Qwen2.5-VL-7B-Instruct \
bash scripts/run_gpu_phase3_cross_arch.sh

MODEL_FAMILY=qwen2_5_vl \
MODEL_ALIAS=qwen2_5_vl_7b \
bash scripts/run_cpu_phase3_cross_arch.sh
```

To run the configured Phase 3 model list sequentially:

```bash
PHASE3_STEP=all bash scripts/run_phase3_cross_arch_all.sh
```

## Key Outputs

| Stage | Directory | Description |
| --- | --- | --- |
| A | `outputs/svd/` | Difference matrices, SVD spectra, effective ranks |
| B | `outputs/stage_b/` | Matched/random/adversarial condition geometry |
| C/P | `outputs/stage_c_*`, `outputs/stage_p_stats/` | FP/TN probes, K sensitivity, multi-seed stats |
| J | `outputs/stage_j_controls/` | Destructive and random controls |
| K | `outputs/stage_k_positions/` | Token-position robustness |
| L | `outputs/stage_l_evidence_subspace/` | Evidence-specific subspaces |
| E/M | `outputs/interventions/`, `outputs/stage_m_local_rescue/` | Causal ablation and rescue probes |
| N | `outputs/stage_n_external_full/` | AMBER external transfer |
| O/Phase 3 | `outputs/stage_o_cross_model/` | LLaVA-13B and cross-architecture replication artifacts |
| R/S | `outputs/stage_r_semantics/`, `outputs/stage_s_baselines/` | Semantic fingerprints and baseline positioning |

## Claims and Non-Claims

Supported by current artifacts:

- Blind-reference differencing reveals hallucination-relevant correction geometry
  in LLaVA-family LVLMs.
- Dominant variance directions are not the hallucination decision geometry.
- Residual/tail and supervised evidence-specific views better expose matched
  visual evidence and FP/TN differences.
- L24 residual/tail coordinates are causally relevant for correct negative
  first-token decisions.
- AMBER and `LLaVA-1.5-13B` provide modest external/checkpoint-level support.

Not supported by current artifacts:

- A universal visual grounding subspace.
- Top singular directions as hallucination directions.
- State-of-the-art hallucination detection.
- A reliable hallucination mitigation method.
- Architecture-universal claims across LVLM families.

## Citation

This is an active paper artifact. Add the final citation once the manuscript is
public:

```bibtex
@misc{vgs2026correctiongeometry,
  title  = {Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination},
  author = {TBD},
  year   = {2026},
  note   = {Code and artifacts for mechanistic LVLM hallucination analysis}
}
```

## License

A repository-level license has not been added yet. Add one before public
release, and keep third-party dataset/checkpoint licenses separate.
