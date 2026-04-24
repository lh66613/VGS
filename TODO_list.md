# TODO List — Validation Plan for Blind-Reference Subspace on LLaVA-1.5-7B + POPE

## 0. Goal

Build a **step-by-step validation pipeline** for the hypothesis that the blind-reference difference subspace extracted from

\[
D^{(l)} = \{z^{(l)}_{\text{blind},i} - z^{(l)}_{\text{img},i}\}_{i=1}^N
\]

is not merely useful for hallucination detection, but is a **stable, grounding-related geometric structure** inside the model, may carry **interpretable visual-semantic directions**, and may further have **causal influence** on faithfulness.

This plan deliberately fixes the core scope to:

- **Model**: `LLaVA-1.5-7B`
- **Main benchmark / main evaluation**: `POPE` only
- **Target**: first finish a clean validation chain on one model and one benchmark before expanding

A very small **optional sanity check** on a caption benchmark is added near the end only to reduce the risk that all conclusions are yes/no-specific.

---

## 1. Fixed Experimental Settings

### Current completed run snapshot

- [x] Main implementation fixed to HuggingFace `LlavaForConditionalGeneration`
- [x] Conda environment used for completed runs: `after`
- [x] Model checkpoint path: `/data/lh/ModelandDataset/llava-1.5-7b-hf`
- [x] Transformers version observed: `4.37.2`
- [x] Torch version observed: `2.2.1+cu121`
- [x] Main POPE family fixed to `coco`
- [x] Full POPE run completed on 9000 samples
- [x] Tested layers: `8 / 12 / 16 / 20 / 24 / 28 / 32`
- [x] Current readout position: `last_prompt_token`
- [x] Current hidden artifacts: `outputs/hidden_states/layer_{8,12,16,20,24,28,32}.pt`

### 1.1 Fixed model

- [x] Use only `LLaVA-1.5-7B`
- [x] Freeze all weights; no fine-tuning
- [x] Use one inference implementation and keep it unchanged across all completed experiments
- [x] Record exact model source / checkpoint path / loader type (`official` vs `HF`) / transformers version / torch version
- [x] Record whether hidden states are read from the official LLaVA implementation or the HuggingFace port
- [x] Do **not** mix implementations in the same study

### 1.2 Fixed dataset

- [x] Use only `POPE` for the full validation chain completed so far
- [x] Evaluate all three subsets if available:
  - [x] `random`
  - [x] `popular`
  - [x] `adversarial`
- [x] Keep the same image path resolution logic and question formatting across all completed runs
- [x] Save the exact sample IDs used in every experiment
- [x] Define train / val / test split policy once and keep it fixed for all probe experiments
  - Current probe policy: deterministic stratified held-out split inside `train_test_split(random_state=42)`
  - Future improvement: persist explicit train / val / test sample-id files

### 1.3 Fixed labels and outputs

- [x] Convert model outputs into binary yes/no predictions with one unified post-processing rule
- [x] Save raw generation text for each sample
- [x] Save parsed yes/no label for each sample
- [x] Save ground-truth label for each sample
- [x] Save whether the prediction is correct / FP hallucination / FN miss / TP / TN

Completed POPE prediction summary:

- [x] `outputs/predictions/pope_predictions.jsonl`
- [x] Accuracy: `0.8619`
- [x] TP / TN / FP / FN: `3606 / 4151 / 349 / 894`

### 1.4 Fixed hidden-state extraction protocol

- [x] Decide once and keep fixed for the completed main run:
  - [x] which hidden stream to read (recommended: transformer block output / post-block hidden state)
  - [x] which layers to test first (recommended sparse set: 8 / 12 / 16 / 20 / 24 / 28 / 32)
  - [x] whether the representation is read from one token position or a pooled set of candidate positions
- [x] For each sample, extract both:
  - [x] `z_img^(l)` from image+question input
  - [x] `z_blind^(l)` from text-only / blind-reference input
- [x] Ensure the prompt template differs only in the image condition, not in wording
- [ ] Revisit prompt/template exactness during Stage B, especially blind vs image chat template alignment

### 1.5 Token-position sensitivity protocol

Because the whole study depends on where the hidden state is read, do **not** choose the token position by convenience only.

- [ ] Predefine 3–5 candidate readout positions, e.g.:
  - [ ] last prompt token
  - [ ] first answer token prefill position
  - [ ] average of the last 4 prompt tokens
  - [ ] one image-adjacent text token if accessible in the chosen implementation
- [ ] Run a small pilot on one POPE subset to compare the candidates
- [ ] Choose one primary position based on the pilot and freeze it for the main study
- [ ] Keep one secondary position only for robustness checks

Current status:

- [x] First main run uses `last_prompt_token`
- [ ] Token-position pilot is still pending
- [ ] Do not claim token-position robustness yet

### 1.6 Unified K-sensitivity protocol

K must be treated as a first-class experimental variable rather than a scattered detail.

- [x] Predefine a global candidate set, e.g. `K in {4, 8, 16, 32, 48, 64}`
- [x] Extend Stage C deep grid to `K in {4, 8, 16, 32, 64, 128, 256}`
- [x] Extend Stage C supervised/exclusion grid to `K in {4, 8, 16, 32, 64, 128, 256, 512, 1024}`
- [x] Use exactly the same K grid in Stages A / C / D whenever feasible
- [x] Report all main curves as functions of K, not just a best case
- [ ] Choose a default `K*` only after Stage A + Stage C
- [ ] Selection rule for `K*`:
  - [ ] prefer the smallest K that reaches near-peak predictive performance
  - [ ] prefer a stable K region over a single sharp optimum
- [x] If performance is flat across a wide range of K, explicitly note that this supports a concentrated low-rank structure
- [x] If performance only appears at large K, explicitly note that the low-rank hypothesis is weakened

Current K finding:

- [x] Variance concentration is strong at low K, but FP-vs-TN AUROC rises mainly at K=128/256
- [x] K=512/1024 does not uniformly improve performance; L20 and L16 peak at K=256, while L24/L32 rise more slowly and remain weaker
- [x] Current interpretation: most variance-explaining directions are not the most hallucination-discriminative directions
- [ ] Do not finalize a single `K*` yet; keep `K=128/256` as candidate predictive settings and `K=4` as a stable-structure setting

---

## 2. Global Deliverables

By the end of this TODO list, the project should produce:

- [x] one reproducible hidden-state dump pipeline
- [x] one reproducible POPE evaluation pipeline
- [x] one SVD analysis pipeline for all tested layers
- [x] one standardized K-sensitivity report
- [x] one probe baseline comparison table
- [x] one supervised-vs-SVD subspace alignment report
- [x] one full-space coordinate-control sanity check
- [x] one layerwise geometry report
- [x] one causal intervention pilot result
- [ ] one semantic interpretation report for singular directions
- [ ] one optional caption sanity-check report
- [x] one running summary document answering: “What has actually been validated?”
  - Current file: `notes/findings.md`

---

## 3. Project Structure

- [x] Create / confirm folders:

```text
project/
├── data/
│   ├── pope/
│   ├── chair_optional/
│   └── cache/
├── outputs/
│   ├── predictions/
│   ├── hidden_states/
│   ├── svd/
│   ├── probes/
│   ├── plots/
│   ├── interventions/
│   ├── semantics/
│   ├── stage_c_deep/
│   ├── stage_c_supervised/
│   ├── stage_c_coordinate_control/
│   ├── stage_b/
│   ├── stage_b_hidden/
│   └── sanity_checks/
├── scripts/
│   ├── run_pope_eval.py
│   ├── dump_hidden_states.py
│   ├── build_difference_matrix.py
│   ├── analyze_spectrum.py
│   ├── analyze_k_sensitivity.py
│   ├── train_probe.py
│   ├── compare_features.py
│   ├── layerwise_analysis.py
│   ├── intervention_precheck.py
│   ├── intervention_pilot.py
│   ├── analyze_stage_e_results.py
│   ├── run_gpu_stage_e_fp_rescue.sh
│   ├── semantic_interpretation.py
│   ├── chair_sanity_check.py
│   ├── validate_pope_data.py
│   ├── create_smoke_artifacts.py
│   ├── analyze_stage_c_deep.py
│   ├── analyze_stage_c_supervised.py
│   ├── analyze_stage_c_coordinate_control.py
│   ├── prepare_stage_b_conditions.py
│   ├── dump_stage_b_condition_hidden_states.py
│   ├── analyze_stage_b_geometry.py
│   ├── run_gpu_pope_and_hidden.sh
│   ├── run_cpu_stage_a.sh
│   ├── run_cpu_stage_c_d.sh
│   ├── run_gpu_stage_b_conditions.sh
│   ├── run_cpu_stage_b.sh
│   └── run_gpu_stage_e.sh
└── notes/
    ├── experiment_log.md
    ├── findings.md
    ├── implementation_notes.md
    └── claim_evidence_table.md
```

- [x] Every script must accept explicit CLI args
- [x] Every script must save a JSON / CSV summary
- [x] Every major run must write one line into `notes/experiment_log.md`
- [x] Every figure must be reproducible from a saved command
- [x] Add progress bars to long-running scripts

---

## 4. Validation Roadmap Overview

Validation is divided into eight stages, from weak evidence to strong evidence:

1. **Stage A**: Show the subspace is real and stable, not noise
2. **Stage B**: Show the subspace is related to image-conditioned correction, not generic modality difference
3. **Stage C**: Show the subspace is useful for hallucination prediction beyond raw hidden-state probing
4. **Stage D**: Show the geometry evolves layer by layer
5. **Stage E**: Test causal influence via ablation / rescue
6. **Stage F**: Decide whether it is better described as grounding-related or arbitration-related
7. **Stage G**: Interpret the singular directions semantically
8. **Stage H (optional)**: Run a caption sanity check to test whether the signal is not purely yes/no-specific

---

## 5. Stage A — Stability / Low-Rank Validation

## A1. Build the difference matrix

### Task
- [x] For each tested layer `l`, compute
  - [x] `d_i^(l) = z_blind^(l) - z_img^(l)`
- [x] Stack them into `D^(l) in R^(N x d)`
- [x] Save one matrix per layer

### Output
- [x] `outputs/svd/D_layer_{l}.pt`
- [x] metadata file with sample IDs and split name

---

## A2. Singular spectrum analysis

### Task
- [x] Run SVD on each `D^(l)`
- [x] Plot:
  - [x] singular values
  - [x] normalized singular values
  - [x] cumulative explained variance
  - [x] effective rank

### What to check
- [x] Does the spectrum show clear decay?
- [x] Is there evidence of low-rank structure?
- [x] Which layers are most concentrated?

### Success criterion
This stage supports the hypothesis if:
- [x] the top singular directions capture substantially more variance than the tail
- [x] effective rank is meaningfully smaller than `d`
- [x] some layers show sharper concentration than others

### Output
- [x] `spectrum_layer_{l}.png`
- [x] `effective_rank_summary.csv`

Completed observation:

- [x] K=4 explained variance is high for L8/L12/L16/L20/L24/L28, lower for L32
- [x] L32 appears less concentrated than middle layers

---

## A3. Split-half stability

### Task
- [x] Randomly split samples into two halves multiple times
- [x] Compute `V_K^(l)` on each half
- [x] Compare the two subspaces via:
  - [ ] principal angles
  - [x] overlap / projection similarity

### What to check
- [x] Does the same layer produce similar subspaces across splits?
- [x] Is stability robust for multiple values of `K`?

### Success criterion
This stage supports the hypothesis if:
- [x] top-K subspaces are stable across random splits
- [x] stability is significantly better than random-subspace baselines

### Output
- [x] `k_sensitivity_summary.csv`
- [x] `k_sensitivity_plot.png`

Implementation note:

- [x] Stability currently uses randomized/subsampled split-half estimation for tractability
- [ ] Exact full-data split-half stability remains optional

---

## A4. Shuffle controls

### Task
- [ ] Rebuild `D^(l)` under destructive controls:
  - [ ] shuffle image-question pairing
  - [ ] shuffle blind/image pairing
  - [ ] compare against random Gaussian matrices of matched shape
- [ ] Repeat SVD and stability analysis

### Success criterion
This stage supports the hypothesis if:
- [ ] the real data shows sharper spectra and stronger stability than shuffled controls

Current status:

- [ ] Not yet run. This is the main missing piece before making the strongest Stage A claim.

---

## A5. Dedicated K-sensitivity analysis

### Task
- [x] For each layer, summarize how stability and explained variance change with K
- [x] Plot:
  - [x] stability vs K
  - [x] explained variance vs K
  - [x] predictive performance placeholder vs K once Stage C is done
- [x] Mark the smallest K that reaches a stable region

### Success criterion
This stage supports the low-rank story if:
- [x] a small or moderate K already captures most stable structure
- [ ] there is a plateau rather than only late improvement at very large K
  - Current predictive performance rises substantially at larger K; this weakens a simple low-K predictive story

### Decision after Stage A
- [ ] If Stage A fails, stop all geometric interpretation claims
- [x] If Stage A passes, continue to Stage B
- [x] Propose 1–2 candidate values for `K*`, but do not finalize until Stage C
  - Current candidates depend on purpose: `K=4` for stable dominant geometry, `K=128/256` for prediction

---

## 6. Stage B — Is It Really Image-Conditioned Correction?

Stage B is upgraded after Stage C because top-K directions are not the whole story. Every condition comparison should report three views:

- **Top-backbone score**: `||P_{V_1:K}(z_blind - z_cond)||^2`, with K in `4 / 64 / 256`
- **Residual / tail score**: bands such as `257-1024`, plus informative mid-rank bands such as `65-128` and `129-256`
- **Supervised decision score**: fixed logistic / LDA-Fisher directions learned from the matched FP-vs-TN reference data, used only as scalar scores in Stage B

Prepared implementation:

- [x] Build condition-plan script: `scripts/prepare_stage_b_conditions.py`
- [x] Build GPU hidden dump script: `scripts/dump_stage_b_condition_hidden_states.py`
- [x] Build CPU geometry analysis script: `scripts/analyze_stage_b_geometry.py`
- [x] Add wrappers:
  - [x] `scripts/run_gpu_stage_b_conditions.sh`
  - [x] `scripts/run_cpu_stage_b.sh`

Current default pilot:

- [x] Focus on `FP/TN` because the current hallucination target is FP-vs-TN
- [x] Use L20 / L24 / L32 first
- [x] Use `max_samples_per_outcome=256` for the first GPU pilot to avoid an unnecessarily large first run
- [x] Generate pilot condition plan: `outputs/stage_b/stage_b_condition_plan.jsonl`
  - [x] 512 rows total
  - [x] 256 FP / 256 TN
  - [x] 512 / 512 adversarial mismatches available
- [x] Run GPU condition hidden dump for L20 / L24 / L32
- [x] Run CPU Stage B geometry analysis

## B1. Four-condition comparison with multiple geometry views

### Conditions
For the same question, prepare four inference settings when possible:
- [x] matched real image
- [x] mismatched image — random unrelated image
- [x] adversarial mismatched image — same question, opposite POPE label / evidence-conflicting image
- [x] blind / no image
- [ ] weakly related image (optional if easy to construct)

### Important construction note
Do **not** use only one type of mismatch.
The main contrast should include:
- [x] easy mismatch: random unrelated image
- [x] hard mismatch: same question with opposite label

### Task
- [x] Extract hidden states for each condition
- [x] For each layer, compute top-backbone scores:

\[
g_{i,\text{top}}^{(l,K)} =
\|P_{V_{1:K}^{(l)}}(z_{\text{blind},i}^{(l)} - z_{\text{cond},i}^{(l)})\|_2^2
\]

- [x] Compute residual / tail scores:

\[
g_{i,\text{tail}}^{(l,a:b)} =
\|P_{V_{a:b}^{(l)}}(z_{\text{blind},i}^{(l)} - z_{\text{cond},i}^{(l)})\|_2^2
\]

- [x] Compute supervised decision scores:

\[
s_i^{(l)} = w^{(l)\top}(z_{\text{blind},i}^{(l)} - z_{\text{cond},i}^{(l)})
\]

### What to check
- [x] Is the matched image condition systematically different from mismatched / blind?
- [x] Does the top-backbone score distinguish visual correction from no image?
  - Current result: yes for image vs blind, but not cleanly for correct-vs-wrong evidence
- [x] Does the residual / tail score carry condition-specific movement?
  - Current result: yes, `257-1024` is consistently higher for matched than random/adversarial mismatch
- [x] Does the supervised decision score move under real image evidence rather than behaving like a dataset-only artifact?
  - Current result: yes, FP/TN gaps open mainly under matched evidence
- [x] Does adversarial mismatch fall between matched and blind, or form a distinct failure pattern?
  - Current result: adversarial mismatch is not simply between matched and blind; it forms a wrong-evidence condition with lower tail correction and weaker decision separation

### Success criterion
This stage supports the hypothesis if:
- [x] matched-image scores are clearly separated from mismatched / blind controls in at least one stable geometry view
- [x] adversarial mismatch differs from random mismatch, showing the signal is evidence-related rather than only image-presence-related
- [x] supervised decision scores shift with true visual evidence, not only with POPE labels

### Output
- [x] `outputs/stage_b/stage_b_condition_plan.jsonl`
- [x] `outputs/stage_b_hidden/layer_{l}.pt`
- [x] `outputs/stage_b/stage_b_sample_scores.csv`
- [x] `outputs/stage_b/stage_b_condition_score_summary.csv`
- [x] condition-wise score plots under `outputs/plots/`

---

## B2. Matched-correction vs mismatched-correction geometry

### Task
Directly compare:

\[
d_i^{\text{match}} = z_{\text{blind},i} - z_{\text{match},i}
\]

\[
d_i^{\text{mismatch}} = z_{\text{blind},i} - z_{\text{mismatch},i}
\]

- [x] Compare `||d_match||^2` vs `||d_mismatch||^2`
- [x] Compare top-backbone / residual-tail / supervised-score projections
- [x] Compare matched vs random-mismatch vs adversarial-mismatch SVD bases
- [x] Compare each condition basis against the original matched-reference SVD basis

### What to check
- [x] Does a real matched image induce qualitatively different correction geometry than a wrong image?
- [x] Is random mismatch mostly an image-presence perturbation?
- [x] Is adversarial mismatch closer to matched, blind, or its own geometry?
  - Current result: adversarial and random mismatch both diverge from matched at K=64/256; top K=4 remains highly shared

### Success criterion
This stage supports the hypothesis if:
- [x] matched and mismatched corrections are separated in score space
- [x] condition-specific SVD bases are not all interchangeable
- [x] the matched-vs-adversarial contrast remains visible after controlling for image presence

### Output
- [x] `outputs/stage_b/stage_b_pairwise_condition_deltas.csv`
- [x] `outputs/stage_b/stage_b_condition_subspace_similarity.csv`

---

## B3. Link Stage B scores to POPE FP/TN labels without training a new classifier

### Task
- [x] Compare FP/TN score distributions under matched / random-mismatch / adversarial-mismatch / blind conditions
- [x] Check whether TN samples show stronger matched-image correction than mismatch / blind
- [x] Check whether FP samples retain abnormal supervised decision scores even under matched visual evidence

### Success criterion
This stage supports the mechanism story if:
- [x] TN samples show matched evidence inducing the expected correction geometry
- [x] FP samples show weak, distorted, or less condition-sensitive correction geometry
  - Current result: TN has stronger matched-specific residual/tail correction; FP shows a strong shift along FP-like supervised decision directions

Target wording if supported:

- [x] hallucination is associated not merely with the presence of image-conditioned change, but with a failure to induce the appropriate correction geometry under matched visual evidence

### Output
- [x] `outputs/stage_b/stage_b_outcome_condition_summary.csv`
- [x] FP/TN condition plots under `outputs/plots/`

---

## 7. Stage C — Beyond a Generic Probe: Feature Comparison

## C1. Define feature families

### Task
Compare at least the following features for hallucination prediction:
- [x] raw `z_img^(l)`
- [x] raw `z_blind^(l)`
- [x] full difference vector `z_blind^(l) - z_img^(l)`
- [x] projected feature `V_K^(l)^T (z_blind^(l) - z_img^(l))`
- [x] random-K projection of the difference vector
- [x] PCA-K of `z_img^(l)` as a control

Use the same lightweight classifier for all, e.g. logistic regression.

### Task details
- [x] Train on one split of POPE
- [x] Evaluate on held-out split
- [x] Repeat for the global K grid
- [x] Repeat for multiple layers

### Metrics
- [x] AUROC
- [x] AUPRC
- [x] accuracy
- [x] F1
- [x] especially report FP-detection quality

### Success criterion
This stage supports the hypothesis if:
- [ ] the projected `V_K` feature remains competitive or superior at low dimensionality
  - Current result: low-K projected features are not competitive; K=128/256 improves substantially
- [ ] performance is more stable across POPE subsets
- [ ] random projections do noticeably worse
  - Current result: random controls can be nontrivial; more careful controls are needed

### Output
- [x] feature comparison table
- [x] AUROC-vs-K plot
- [x] layer-vs-feature diagnostic plots

Completed outputs:

- [x] `outputs/probes/probe_results.csv`
- [x] `outputs/probes/feature_comparison.csv`
- [x] `outputs/stage_c_deep/stage_c_topk_curve.csv`
- [x] `outputs/plots/stage_c_topk_auroc_explained_variance.png`

---

## C2. Compression advantage

### Task
Specifically test whether the geometric feature is **more compact**:
- [x] compare performance at the full K grid
- [x] compare against full hidden states and full difference vectors

### Success criterion
This stage supports the hypothesis if:
- [ ] a very low-dimensional geometric feature preserves a large fraction of predictive power
  - Current result: not supported for FP-vs-TN; best projected results appear at K=128/256

---

## C3. Finalize K*

### Task
- [x] Combine Stage A and Stage C results
- [ ] Select one default `K*` for later interpretation and interventions
- [x] Save a short note justifying why `K*` was not finalized yet

### Success criterion
- [ ] `K*` is chosen by an explicit rule, not by cherry-picking the single best number

### Interpretation allowed after Stage C
If Stage C passes, it is reasonable to say:
- [x] the subspace is not just mathematically present
- [x] it carries structured hallucination-relevant information
- [ ] it is more compact than generic raw-state probing
  - Current result: compactness is not validated

But do **not** yet claim causality.

## C4. Deep Stage C follow-up — K curves, layer mismatch, and non-top-K bands

### Task
- [x] Plot AUROC-vs-K for every tested layer
- [x] Plot explained-variance-vs-K beside AUROC-vs-K
- [x] Compare L20 / L24 / L28 / L32 in detail
- [x] Probe non-top-K SVD bands:
  - [x] `1-4`
  - [x] `5-8`
  - [x] `9-16`
  - [x] `17-32`
  - [x] `33-64`
  - [x] `65-128`
  - [x] `129-256`
- [x] Compare SVD bands against random-width controls

### Outputs
- [x] `outputs/stage_c_deep/stage_c_topk_curve.csv`
- [x] `outputs/stage_c_deep/stage_c_band_probe.csv`
- [x] `outputs/stage_c_deep/stage_c_layer_diagnostics.csv`
- [x] `outputs/plots/stage_c_band_probe_auroc.png`
- [x] `outputs/plots/stage_c_layer_diagnostics.png`

### Current conclusion
- [x] Variance growth and AUROC growth are not synchronized
- [x] Top-4 directions explain most variance but are weakly discriminative
- [x] L20 is strongest for top-K projection at K=256
- [x] L24 is strongest for full difference
- [x] L32 is weaker both in concentration and predictive behavior
- [x] Best SVD band found so far: L20 directions `65-128`, AUROC `0.6317`
- [x] Current wording: the main geometric structure and hallucination-discriminative structure are related but not identical

---

## C5. Supervised subspace and exclusion follow-up

### Task
- [x] Compare supervised discriminative directions against SVD top-K backbone:
  - [x] logistic regression weight direction
  - [x] LDA / Fisher direction
  - [x] PLS-style supervised directions
- [x] Measure directed projection similarity / principal angle between supervised directions and SVD top-K subspaces
- [x] Extend top-K projected probes to `K=512/1024`
- [x] Run cumulative and exclusion probes:
  - [x] only top `1:K`
  - [x] remove top `1:K`
  - [x] only band `a:b`
  - [x] remove band `a:b`

### Outputs
- [x] `outputs/stage_c_supervised/stage_c_supervised_alignment.csv`
- [x] `outputs/stage_c_supervised/stage_c_extended_k_curve.csv`
- [x] `outputs/stage_c_supervised/stage_c_cumulative_exclusion.csv`
- [x] `outputs/stage_c_supervised/stage_c_band_exclusion.csv`
- [x] `outputs/plots/stage_c_supervised_alignment_logistic_weight.png`
- [x] `outputs/plots/stage_c_supervised_alignment_lda_fisher.png`
- [x] `outputs/plots/stage_c_supervised_alignment_pls_8.png`
- [x] `outputs/plots/stage_c_extended_topk_auroc.png`
- [x] `outputs/plots/stage_c_cumulative_exclusion_auroc.png`
- [x] `outputs/plots/stage_c_band_exclusion_auroc.png`

### Current conclusion
- [x] Logistic and LDA/Fisher directions are nearly orthogonal to very-low-K SVD directions
- [x] L20 logistic direction projection similarity rises from `0.0004` at K=4 to `0.0734` at K=256 and `0.4838` at K=1024
- [x] L20 remains the strongest top-K predictive layer, peaking at K=256 with AUROC `0.6948`
- [x] Full SVD-coordinate probes are stronger than top-K-only probes on focused layers; L20 reaches AUROC `0.7343`
- [x] Removing directions `1-1024` still leaves high AUROC on L20 (`0.7232`), so the top variance directions are not functionally necessary for this probe
- [x] Most damaging single-band removals are modest, suggesting distributed residual / mid-to-tail signal rather than one uniquely necessary band

---

## C6. Full-space coordinate-control sanity check

### Motivation
Full difference vectors and all SVD coordinates span the same linear space, so a large performance gap needs a pipeline/control explanation before moving to Stage B.

### Task
- [x] Use one fixed train/test split
- [x] Use the same standardization, solver, class weighting, seed, and regularization for every representation
- [x] Compare:
  - [x] raw full difference
  - [x] train-split PCA-whitened full difference
  - [x] full SVD coordinates
  - [x] dense random orthogonal rotation of full difference

### Outputs
- [x] `outputs/stage_c_coordinate_control/stage_c_coordinate_control.csv`
- [x] `outputs/plots/stage_c_coordinate_control_auroc.png`

### Current conclusion
- [x] L24 raw full difference reproduces the earlier AUROC `0.6936`, so the previous discrepancy is not mainly split drift
- [x] Full SVD coordinates remain stronger under the same split and logistic pipeline:
  - [x] L20: raw `0.6869` vs SVD coords `0.7343`
  - [x] L24: raw `0.6936` vs SVD coords `0.7096`
  - [x] L32: raw `0.6694` vs SVD coords `0.7139`
- [x] Random dense orthogonal rotation stays close to raw full difference, so the gain is not just any rotation
- [x] Current wording: classifier performance depends on coordinate parameterization under the current standardization / regularization pipeline; full difference and all SVD coordinates should not be described as empirically interchangeable for this probe

---

## 8. Stage D — Layerwise Geometry and Information Flow

## D1. Layerwise summary

### Task
For each tested layer, summarize:
- [x] singular spectrum concentration
- [x] effective rank
- [x] split-half stability
- [x] best probe performance using projected features
- [x] preferred K region

### Output
- [x] one layerwise summary table
- [x] one combined figure

---

## D2. Subspace angle across layers

### Task
- [ ] Compute principal angles between `V_K^(l)` and `V_K^(l+1)`
- [x] Also compare non-adjacent layers via projection similarity

### What to check
- [x] Does the subspace gradually stabilize across layers?
- [x] Are there transition layers where the geometry changes sharply?

### Success criterion
This stage supports the hypothesis if:
- [x] the subspace shows interpretable layer-to-layer evolution instead of random fluctuations
- [ ] the strongest predictive layers align with the most stable / concentrated geometry
  - Current result: concentration, stability, and predictive behavior are partly misaligned

### Output
- [x] layerwise angle heatmap
- [x] line plot of stability / rank / AUROC across layers

### Interpretation allowed after Stage D
If Stage D passes, it is reasonable to say:
- [x] the relevant geometric structure is organized across layers
- [x] there may exist a preferred “grounding-sensitive” stage in the network
  - Candidate predictive stage: L16-L24, especially L20/L24 depending on feature family

Still avoid causal wording until Stage E.

---

## 9. Stage E — Causal Intervention Pilot

> Note: this is the most important stage for “mechanism” claims, but it is also the riskiest technically. Do not enter the formal intervention stage before an engineering pre-check succeeds.

Stage E should reflect the Stage B/C result: do **not** prioritize deleting top-4 directions. The first causal tests should target:

- residual / tail correction slices, especially `257-1024`
- supervised decision-aligned directions, especially logistic and LDA/Fisher FP-vs-TN directions
- TN-like correction vs FP-like decision shift directions separately

Prepared implementation:

- [x] Implement hook-based Stage E helpers: `src/vgs/stage_e.py`
- [x] Replace placeholder `scripts/intervention_precheck.py` with real E0 precheck path
- [x] Replace placeholder `scripts/intervention_pilot.py` with real pilot path
- [x] Add wrapper: `scripts/run_gpu_stage_e.sh`
- [x] Dry-run E0 and pilot CLIs without loading the model
- [x] Add post-hoc Stage E analysis script: `scripts/analyze_stage_e_results.py`
- [x] Expose `MAX_NEW_TOKENS` in `scripts/run_gpu_stage_e.sh` so first-token intervention pilots can be run cleanly
- [x] Add FP rescue wrapper: `scripts/run_gpu_stage_e_fp_rescue.sh`
- [x] Add explicit rescue control: `random_rescue_control`
- [x] Add sign-reversed supervised rescue diagnostics:
  - [x] `reverse_logistic_fp_direction`
  - [x] `reverse_lda_fp_direction`

Default pilot:

- [x] Start with L24 as the first target layer
- [x] Use `tail_band=257-1024`
- [x] First run used `alpha in {0.25, 0.5, 1.0}`
- [x] Next default uses stronger `alpha in {1.0, 2.0, 4.0, 8.0}`
- [x] Clean tail dose curve uses `alpha in {4.0, 5.0, 6.0, 7.0, 8.0}`
- [x] Use `max_samples_per_outcome=16` by default
- [x] TN samples: tail-slice ablation plus random-tail control
- [x] FP samples: supervised-direction reduction plus TN-correction / FP-shift steering
- [x] Add pilot yes/no logit recording for future runs
- [x] Add cleaner controls:
  - [x] random same-width tail control
  - [x] orthogonal random same-width control
  - [x] norm-matched orthogonal random control
- [x] Add intervention granularity options:
  - [x] `last_token`
  - [x] `full_sequence`
  - [x] `generated_token`
- [x] Add outcome/family filters so clean TN tail dose curves can run without FP rescue

## E0. Implementation pre-check

### Goal
Verify that activation intervention is technically feasible and reproducible in the chosen `LLaVA-1.5-7B` codebase.

### Task
- [x] Freeze one codebase only (`official` or `HF`) and do not switch midway
  - Current Stage E path uses HuggingFace `LlavaForConditionalGeneration`
- [x] Inspect the forward path and document:
  - [x] where image embeddings are inserted / fused with text
  - [x] the exact module names for transformer blocks
    - HF hook path: `model.language_model.model.layers[layer-1]`
  - [x] whether `output_hidden_states=True` gives the needed tensors directly
  - [ ] whether full-sequence hidden states can be intercepted and modified cleanly
- [x] Decide one main intervention mechanism:
  - [x] preferred: `register_forward_hook` / forward hook on target transformer block output
  - [ ] backup: manual forward split / wrapper module if hook semantics are unreliable
- [ ] Verify that the intervention can modify:
  - [x] last-token hidden state only
  - [ ] full-sequence hidden states
- [x] Run a no-op hook to confirm output equality before and after adding the hook
  - First run: baseline `Yes`, no-op `Yes`
- [x] Run a tiny random-direction intervention to confirm the logits change in the expected place
  - First run checked decoded text only; random perturbation did not change text
  - Second run: random perturbation changed logits with max delta `0.1641`, but did not change decoded text

### Implementation notes to document
- [ ] whether image tokens are prepended / inserted before the LM blocks in this implementation
- [ ] whether the chosen hidden-state readout position is available during intervention
- [ ] whether sequence-length differences between blind and image-conditioned inputs create alignment problems

### Exit condition
Proceed to E1 only if:
- [x] the intervention path is stable
- [x] no-op hook preserves outputs exactly or near-exactly
- [x] random-direction perturbation creates controlled, measurable output changes
  - Caveat: movement is visible in logits but small relative to yes/no margin

If E0 fails:
- [ ] downgrade Stage E to a smaller pilot or skip rescue first
- [ ] explicitly state that causal claims remain unvalidated due to engineering constraints

---

## E1. Choose one or two target layers

### Task
- [x] Select layers based on earlier evidence:
  - [x] strongest stability
  - [x] strongest spectral concentration
  - [x] strongest predictive power
  - [ ] engineering feasibility from E0

### Output
- [x] one short note explaining why the chosen layer(s) were selected
  - Start at L24 for the first causal pilot because Stage C full-difference/SVD-coordinate probes are strong there and Stage B supervised decision gaps are largest or near-largest at L24; keep L20/L32 as follow-up layers

---

## E2. Decide intervention granularity

### Task
- [x] Compare two intervention granularities on a very small pilot:
  - [x] last-token-only intervention
    - First pilot produced no decoded-answer changes
    - Clean dose run later showed a stable L24 TN tail-ablation effect
  - [x] full-sequence intervention
    - Pilot completed on L24 TN samples with alpha 4/6/8
    - Full-sequence tail ablation is stronger than last-token at the yes/no margin level, but corrupts continuations and controls more often
    - First-token follow-up with `max_new_tokens=1` confirms the full-sequence effect is on the first yes/no decision, not only on post-answer continuation
- [x] Keep the stronger / more stable one as the main Stage E setting
  - Current choice: last-token remains the main clean causal-ablation protocol; full-sequence is reserved for first-token / logit-only follow-up

### Success criterion
- [x] main intervention mode is chosen by measured effect and reproducibility, not by convenience

---

## E3. Ablation of residual / tail correction slices

### Task
For TN samples that are originally correct, modify the matched-image hidden state at layer `l*`:

\[
z' = z - \alpha P_{V_{257:1024}} z
\]

Do not make top-4 deletion the primary intervention. Treat top-direction ablation as a later control if needed.

Next clean dose curve:

- [x] Run alpha `4 / 5 / 6 / 7 / 8`
- [x] Record decoded answer plus yes/no logits and margin
- [x] Run on L24 and replicate on L20
- [ ] Optionally run L32 after L20/L24

### What to measure
- [x] change in yes/no prediction
  - First pilot: no prediction flips; stronger pilot: L24 tail ablation at alpha 8 flips 16/16 TN to FP
  - Clean dose run: L24 flips 0/16, 2/16, 9/16, 15/16, 16/16 at alpha 4/5/6/7/8; L20 flips 1/16, 6/16, 12/16, 14/16, 9/16 with 5 unknowns at alpha 8
- [x] drop in accuracy / F1
  - First pilot: TN accuracy stayed 1.0; L24 clean dose accuracy over valid samples drops to 0.0 at alpha 8
- [x] rise in hallucination rate / FP rate
  - L24 clean dose Yes rate rises monotonically from 0.0 to 1.0 as alpha increases from 4 to 8
- [x] margin dose curve
  - L24 median yes-minus-no margin moves from -0.7500 at alpha 4 to 0.9336 at alpha 8
  - L20 median yes-minus-no margin moves from -0.5703 at alpha 4 to 1.6484 at alpha 8, with format collapse at alpha 8
- [x] granularity comparison
  - L24 full-sequence tail ablation flips 4/4 TN samples at alpha 6, versus 3/4 for last-token
  - Full-sequence alpha 8 median yes-minus-no margin is 4.7402, versus 0.9414 for last-token
- [x] first-token granularity comparison
  - With `max_new_tokens=1`, L24 full-sequence tail ablation flips 8/8 TN samples by alpha 6, versus 5/8 for last-token
  - Full-sequence alpha 6 median yes-minus-no margin is 2.3203, versus 0.0391 for last-token

### Controls
- [x] ablate random same-dimensional subspace
  - Caveat: random control becomes invalid/unknown at high alpha 4/8, so the alpha-8 tail effect is not yet a clean directional-control win
- [x] add orthogonal random same-width control
- [x] add norm-matched orthogonal random same-width control
  - L24 norm-matched control stays `No` for 16/16 samples through alpha 6 while true tail ablation flips 9/16 at alpha 6
  - Caveat: norm-matched control still develops many unknowns at L24 alpha 7/8 and at L20 alpha 6+
  - Granularity caveat: full-sequence controls mostly collapse to unknown, so full-sequence should be evaluated next with `MAX_NEW_TOKENS=1`
  - First-token follow-up: norm-matched last-token control stays `No` for 8/8 samples through alpha 6, while last-token true tail ablation flips 5/8 and full-sequence true tail ablation flips 8/8
- [ ] ablate orthogonal complement slice of matched energy
- [ ] ablate at a weaker layer

### Success criterion
This stage supports the hypothesis if:
- [x] ablating residual / tail correction slices harms grounded answering significantly more than control ablations
  - Supported most cleanly at L24 alpha 6: true tail ablation flips 9/16 TN to FP, while norm-matched control remains 16/16 `No`
  - Still caveated at high alpha because most random/orthogonal controls collapse to unknown output format

---

## E4. Rescue / steering along supervised and correction directions

### Task
For FP samples that are originally hallucinated, try three targeted steering families:

- reduce FP decision score by moving opposite the matched FP-vs-TN decision direction in `d = z_blind - z_img`
- add TN-like correction direction
- subtract / counteract FP-like average shift

\[
z' = z + \alpha w_{\text{decision}}
\]

Next rescue criterion should be logit-level before decoded-answer-level:

- [x] Prepare analysis to check whether `logit(No) - logit(Yes)` increases under rescue directions
- [x] Add random steering control for margin comparisons
- [ ] Only treat decoded flips as a later, stronger success condition

### Primary definition of `r`
Use a concrete default instead of a vague class average:
- [x] define `r` as the mean correction direction from a truthful reference group
- [x] primary choice: mean of `d_i = z_blind - z_img` over **TN samples**
- [ ] secondary choice: mean over all correct samples in a matched subset
- [ ] always project `r` into `V_K` before injection
  - Current implementation uses unit mean direction first; add projected variant if pilot results are promising

### What to measure
- [x] recovery in yes/no correctness
  - First and second pilots: no FP recovery
- [x] reduction in FP rate
  - First and second pilots: FP rate stayed 1.0
- [ ] whether rescue helps more than random-direction injection
  - Updated result: `reverse_logistic_fp_direction` is better than random at the margin level and matches `add_tn_correction` on the single decoded flip, but the effect is still small

### Next FP rescue pilot

- [ ] Run FP-only first-token rescue:
  - [x] First global rescue pilot completed at L24 with `max_new_tokens=1`

```bash
bash scripts/run_gpu_stage_e_fp_rescue.sh
```

- [ ] Analyze with:

```bash
/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_e_results.py \
  --artifact-prefix stage_e_fp_rescue \
  --target-prediction no
```

- [ ] Next diagnostic run: include sign-reversed supervised directions and compare against the current nominal rescue directions
  - [x] Reverse-direction diagnostic completed
  - `reverse_logistic_fp_direction` is the strongest supervised rescue direction so far
  - `reverse_lda_fp_direction` helps at the margin level but does not flip any sample
- [ ] Next local rescue run: compare newly added sample-conditioned directions against reverse logistic / TN correction / random control
  - [x] Local matched-template pilot completed
  - `local_matched_minus_random` and `local_matched_minus_adversarial` stay near random-control strength and flip 0/16 samples
  - [x] Rerun completed after bug fix for `local_knn_tn_correction` / `question_tn_correction` / `object_tn_correction`
  - `question_tn_correction` and `object_tn_correction` flip the same borderline FP sample at alpha 6, earlier than `reverse_logistic_fp_direction` / `add_tn_correction` / `local_knn_tn_correction`
  - Aggregate margin result still favors `reverse_logistic_fp_direction`
  - [x] Multi-layer sweep completed on `L20 / L24 / L32`
  - `L20` flips earlier but is not clean because `random_rescue_control` also flips `1/16`
  - `L24` remains a clean reference layer with strong reverse-logistic rescue
  - `L32` is now the strongest local-rescue layer: `question_tn_correction` / `object_tn_correction` beat `reverse_logistic_fp_direction` on aggregate margin while random control stays clean
  - [x] Expanded-sample check completed on `L32` with `L24` as the reference layer (`32` FP samples each)
  - `L32` local TN-conditioned rescue remains strongest after expansion
  - `L24` still favors `reverse_logistic_fp_direction`
  - both layers now rescue `2/32` borderline FP samples
  - [x] Larger-sample check completed on `L32` with `L24` as the reference layer (`64` FP samples each)
  - `L32` still favors `question_tn_correction` / `object_tn_correction`
  - `L24` still favors `reverse_logistic_fp_direction`
  - rescue grows from `2/32` to `3/64`, but all rescued cases remain borderline `popular` FP samples
  - [ ] Next expansion target: decide whether to keep scaling on POPE FP samples or switch to a different benchmark / model for external validity

Direction upgrade:

- [ ] Replace global TN mean with more local directions if global steering remains weak:
  - [x] same-question / same-object cluster TN-like direction
  - [x] matched-minus-random local correction template
  - [x] matched-minus-adversarial local correction template
  - [x] sample-conditioned TN-neighborhood correction template
  - [ ] layer-specific and outcome-specific correction templates
  - Current rationale: sign reversal fixes part of the supervised-direction problem, but a single global vector still rescues only 1/16 samples

Prepared local rescue directions now implemented:

- [x] `local_knn_tn_correction`
- [x] `question_tn_correction`
- [x] `object_tn_correction`
- [x] `local_matched_minus_random`
- [x] `local_matched_minus_adversarial`

Coverage note for the current 16-sample FP pilot:

- [x] all 16 selected FP samples overlap with Stage B condition hidden artifacts
- [x] all 16 selected FP samples have exact-question TN support
- [x] all 16 selected FP samples have same-object TN support
- [x] indexing bug fixed so the 16 selected FP samples now correctly resolve `local_knn_tn_correction` / `question_tn_correction` / `object_tn_correction`

### Hyperparameters
- [x] test a small grid of `alpha`
- [x] keep all other decoding settings fixed

### Success criterion
This stage supports the hypothesis if:
- [ ] targeted injection improves faithfulness more than matched random controls
  - Current last-token steering through alpha 8 does not support this

---

## E5. Causal interpretation threshold

You may claim **causal relevance** only if at least one of the following is true:
- [ ] removing the subspace hurts grounded answering significantly more than controls
- [ ] injecting the subspace improves hallucinated cases significantly more than controls
- [ ] ideally both happen on the same layer family

If neither happens, then the subspace should be described as:
- [x] correlationally informative
- [x] useful for detection
- [x] not yet causally validated

---

## 10. Stage F — Grounding or Arbitration?

## F1. Error-type interpretation on POPE

### Task
Use POPE errors to study whether the subspace behaves more like:
- [ ] a visual grounding channel
- [ ] a prior-overriding / arbitration channel

### Minimal operational interpretation within POPE
Because POPE is a yes/no hallucination benchmark, use the following practical heuristic:
- [ ] if the subspace primarily distinguishes FP hallucinations from TN, it is more directly tied to hallucination suppression
- [ ] if it also strongly separates FN from TP, it may encode broader evidence integration
- [ ] if intervention mostly changes final answers without changing earlier evidence-sensitive patterns, it may be closer to arbitration than pure perception

### Output
- [ ] short interpretation memo: `grounding_vs_arbitration.md`

### Success criterion
This stage is **not** about proving a theorem.
It is successful if:
- [ ] the evidence lets you defend one naming choice more honestly than the alternatives

Recommended naming policy:
- [ ] If evidence is mixed, use **grounding-related correction subspace**
- [ ] Only use **visual grounding subspace** if Stage B + Stage E strongly support that interpretation
- [ ] Use **arbitration-related subspace** only if intervention evidence clearly points that way

---

## 11. Stage G — Semantic Interpretation of `V_K`

> This stage is the main attempt to turn the subspace from a useful black-box feature into an interpretable visual-semantic basis.

## G1. Vocabulary projection via LM head / logit lens style analysis

### Task
For each singular direction `v_k` in the chosen layer(s):
- [ ] project `v_k` through the LM head or an equivalent token-space map
- [ ] collect top positive / top negative vocabulary items
- [ ] save token lists for several K values and candidate layers

### What to check
- [ ] Do some directions show coherent semantic clusters?
- [ ] Are there recognizable object / attribute / spatial / counting tokens?

### Success criterion
This stage supports semantic interpretability if:
- [ ] top tokens for a direction are not random-looking
- [ ] multiple directions show distinct, reusable semantic themes

### Output
- [ ] `semantic_tokens_layer_{l}_K_{K}.json`
- [ ] one summary table of top directions and token themes

---

## G2. Per-direction semantic clustering

### Task
- [ ] Group projected tokens by rough semantic class:
  - [ ] object category
  - [ ] attribute / color
  - [ ] spatial relation
  - [ ] counting / quantity
  - [ ] negation / uncertainty
- [ ] Visualize per-direction token clusters

### What to check
- [ ] Are leading directions specialized?
- [ ] Do early singular directions look more semantically coherent than later ones?

### Success criterion
This stage supports the “semantic basis” story if:
- [ ] the first few directions appear more structured and interpretable than the tail

### Output
- [ ] semantic cluster figure
- [ ] short notes for each top direction

---

## G3. Sample-level interpretation

### Task
- [ ] For each direction `v_k`, retrieve samples with strongest positive / negative projection
- [ ] Inspect whether those samples share consistent visual semantics
- [ ] Compare grounded correct samples and hallucinated samples on the same direction

### What to check
- [ ] Does a direction activate on a coherent family of images / questions?
- [ ] Are hallucinated cases missing or misusing that direction?

### Success criterion
This stage supports semantic interpretability if:
- [ ] sample-level behavior aligns with the vocabulary projection story

### Output
- [ ] nearest-neighbor sample panels for top directions
- [ ] one qualitative appendix note

---

## G4. Grounded vs hallucinated vocabulary projection comparison

### Task
- [ ] Compare the token-space projections or directional coefficients between:
  - [ ] grounded correct samples
  - [ ] hallucinated samples
- [ ] Check whether certain directions weaken, flip, or become noisy in hallucinated cases

### Success criterion
This stage supports the interpretability story if:
- [ ] grounded and hallucinated cases differ in a direction-specific way rather than only by total norm

### Interpretation allowed after Stage G
If Stage G passes, it is reasonable to say:
- [ ] the subspace is not only predictive but partially interpretable
- [ ] some singular directions behave like reusable visual-semantic coordinates

Do **not** claim a complete semantic basis unless the results are extremely clean.

---

## 12. Stage H (Optional) — Caption Sanity Check Beyond POPE

> POPE is yes/no-only. This optional stage is only meant to test whether the geometric signal shows a weak form of generality beyond binary decisions.

## H1. Minimal caption benchmark sanity check

### Task
- [ ] Choose one lightweight caption hallucination benchmark, preferably `CHAIR` if available in your environment
- [ ] Do **not** rerun the whole pipeline
- [ ] Only compute one or two simple projection-based scores using the already chosen layer and `K*`

### What to check
- [ ] Do hallucinated caption samples show systematically weaker or noisier `g_i`-style projection scores than non-hallucinated caption samples?

### Output
- [ ] one scatter plot / box plot
- [ ] one short memo: `caption_sanity_check.md`

### Success criterion
This stage is successful if:
- [ ] the trend is directionally consistent with POPE, even if effect sizes are modest

### Interpretation rule
- [ ] If H passes, say the signal has a preliminary sign of transfer beyond yes/no POPE
- [ ] If H is not run, explicitly state that all validated claims remain restricted to POPE-style binary settings

---

## 13. Figures and Tables To Prepare

### Essential figures
- [x] singular value spectra by layer
- [x] cumulative explained variance by layer
- [x] effective rank by layer
- [x] split-half stability plot
- [x] stability-vs-K plot
- [x] matched vs mismatched vs blind condition score plots
- [x] AUROC-vs-K comparison plot
- [x] layerwise heatmap / diagnostic plot of performance
- [x] subspace-angle heatmap across layers
- [x] intervention pre-check figure / sanity output
- [x] intervention result bar chart
- [ ] semantic token cluster figure
- [ ] sample-level semantic direction panels
- [ ] optional caption sanity-check plot

### Essential tables
- [x] POPE baseline prediction table
- [x] feature family comparison table
- [x] best-layer summary table
- [ ] K-selection table
- [x] intervention ablation/rescue table
- [x] Stage E dose-curve and flip-threshold tables
- [ ] semantic direction summary table
- [ ] final claim-evidence table
  - Running notes exist in `notes/findings.md`; `notes/claim_evidence_table.md` still needs final pass

---

## 14. Final Claim-Evidence Checklist

Before writing any paper claim, check the boxes honestly.

### Claim 1: “A stable low-rank difference structure exists.”
- [x] supported by spectrum concentration
- [x] supported by effective rank
- [x] supported by split-half stability
- [ ] supported by shuffle controls
- [x] supported by coherent K-sensitivity behavior

Current status: mostly supported, but still missing shuffle controls.

### Claim 2: “The structure is related to image-conditioned correction.”
- [x] supported by matched vs mismatched vs blind comparisons
- [ ] stronger than random-subspace controls
  - Current Stage B uses SVD top/residual bands and supervised directions; add explicit random-subspace score controls if this claim needs to be maximally strong
- [x] not reducible to the mere presence of any image input
  - Current result: residual/tail and supervised-decision views distinguish matched evidence from random/adversarial mismatches, while top-backbone energy is not sufficient

### Claim 3: “The structure carries hallucination-relevant information.”
- [x] projected features predict POPE hallucination risk
- [ ] works across subsets
- [ ] not reducible to trivial random low-dimensional projections

Current status: partially supported. Stronger at K=128/256 than low K; random controls are nontrivial.

### Claim 4: “The structure is more compact than generic raw-state probing.”
- [ ] low-K features preserve strong predictive performance
- [ ] competitive with or better than raw hidden-state baselines
- [ ] K selection is not cherry-picked

Current status: not supported yet for FP-vs-TN. Do not claim compression advantage.

### Claim 5: “The structure evolves meaningfully across layers.”
- [x] supported by layerwise rank / stability / performance analysis
- [x] supported by inter-layer subspace angle analysis

Current status: supported with an important caveat: rank/stability/performance are partly misaligned.

### Claim 6: “The structure has causal relevance.”
- [x] E0 engineering pre-check passed
- [x] ablation harms grounded answering more than controls
  - Current status: L24 tail ablation has a clean dose curve and beats the norm-matched control at alpha 6; first-token full-sequence ablation is stronger than last-token, while full-sequence control stability remains a caveat
- [ ] rescue improves hallucinated answers more than controls
  - Current status: the strongest clean local rescue remains at `L32`, where question/object-conditioned TN directions still beat reverse logistic after expansion to `64` FP samples and random control stays clean; however decoded-answer rescue is still only `3/64` and concentrated on borderline `popular` samples

### Claim 7: “The structure is partially interpretable.”
- [ ] some singular directions map to coherent vocabulary themes
- [ ] sample-level behavior supports those themes
- [ ] grounded vs hallucinated cases differ direction-specifically

### Claim 8: “It should be called a visual grounding subspace.”
- [ ] only check if Stage B and Stage E provide strong support
- [ ] Stage G gives compatible semantic evidence
- [ ] otherwise use a weaker name

### Claim 9: “The signal likely extends beyond POPE yes/no settings.”
- [ ] only check if Stage H is run
- [ ] otherwise explicitly restrict all claims to POPE

---

## 15. Recommended Execution Order

### Week / Phase 1 — Infrastructure
- [x] make POPE evaluation pipeline fully reproducible
- [x] make hidden-state dump pipeline fully reproducible
- [ ] run token-position pilot and freeze the primary readout position
- [x] verify sample-level alignment between predictions and activations

### Week / Phase 2 — Geometry existence
- [ ] Stage A entirely
  - Done: A1/A2/A3/A5
  - Missing: A4 shuffle controls
- [x] produce the dedicated K-sensitivity report
- [x] decide candidate layers and provisional K values
  - Candidate layers: L16/L20/L24/L28; L32 is weaker
  - Candidate K depends on purpose: K=4 for dominant stable geometry; K=128/256 for prediction

### Week / Phase 3 — Relation to hallucination / correction
- [x] Stage B pilot completed for FP/TN on L20/L24/L32
- [x] Stage C feature/probe/deep analysis completed
- [ ] finalize `K*`
  - Deferred because Stage C shows variance/prediction mismatch

### Week / Phase 4 — Layer story
- [x] Stage D entirely except exact principal-angle variant
- [x] produce one clean layerwise summary figure

### Week / Phase 5 — Causal pre-check and pilot (`1–2 weeks depending on E0 outcome`)
- [ ] Stage E0 first
- [ ] if E0 passes, continue to E1–E5
- [ ] if E0 fails, switch to minimal-version causal note and do not overclaim

### Week / Phase 6 — Interpretation and optional generalization
- [ ] Stage F
- [ ] Stage G
- [ ] Stage H optional sanity check if time allows
- [ ] fill final claim-evidence checklist
- [ ] write one-page conclusion on what is validated vs still speculative

---

## 16. Minimal “Must Finish” Version

If time or compute becomes tight, prioritize this reduced plan:

- [x] Stage A2 spectrum analysis
- [x] Stage A3 split-half stability
- [x] Stage A5 dedicated K-sensitivity analysis
- [x] Stage B1 matched vs mismatched vs blind comparison
- [x] Stage C1 feature comparison vs raw hidden-state probes
- [x] Stage D1 layerwise summary
- [ ] Stage E0 implementation pre-check
- [ ] Stage E3 one ablation pilot at the best layer
- [ ] Stage G1 one lightweight semantic token projection figure

If only this minimal version is completed well, you can still make a strong case that the method is more than a detector prototype.

---

## 17. Final Note To Self

Do **not** overclaim.

At every stage, prefer the weakest honest statement:

- “difference subspace” is safer than “grounding subspace”
- “grounding-related” is safer than “grounding-causing”
- “causally relevant” is safer than “the unique mechanism of hallucination”
- “partially interpretable” is safer than “complete semantic basis”
- “validated on POPE” is safer than “general VLM hallucination mechanism”

The research goal is not to force a grand theory too early.
The goal is to accumulate enough evidence that the geometric story becomes the most convincing explanation available.
