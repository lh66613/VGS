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
- [x] one layerwise geometry report
- [ ] one causal intervention pilot result
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
│   ├── semantic_interpretation.py
│   ├── chair_sanity_check.py
│   ├── validate_pope_data.py
│   ├── create_smoke_artifacts.py
│   ├── analyze_stage_c_deep.py
│   ├── run_gpu_pope_and_hidden.sh
│   ├── run_cpu_stage_a.sh
│   └── run_cpu_stage_c_d.sh
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

## B1. Four-condition comparison

### Conditions
For the same question, prepare four inference settings when possible:
- [ ] matched real image
- [ ] mismatched image — random unrelated image
- [ ] mismatched image — same-category / semantically similar distractor if feasible
- [ ] blind / no image
- [ ] weakly related image (optional if easy to construct)

### Important construction note
Do **not** use only one type of mismatch.
The main contrast should include:
- [ ] easy mismatch: random image
- [ ] hard mismatch: semantically similar but evidence-conflicting image

### Task
- [ ] Extract hidden states for each condition
- [ ] For each layer, compute projection-based scores such as:

\[
g_i^{(l)} = \|P_{V_K^{(l)}}(z_{\text{blind},i}^{(l)} - z_{\text{cond},i}^{(l)})\|_2
\]

### What to check
- [ ] Is the matched image condition systematically different from mismatched / blind?
- [ ] Does `V_K` distinguish “true visual correction” from generic input perturbation?
- [ ] Is the hard mismatch closer to matched than the easy mismatch, or does it produce a distinct failure pattern?

### Success criterion
This stage supports the hypothesis if:
- [ ] matched-image scores are clearly separated from mismatched / blind controls
- [ ] random same-dimensional subspaces do not show equally strong separation
- [ ] the matched-vs-hard-mismatch contrast remains visible, showing the signal is not merely “some image exists”

### Output
- [ ] condition-wise boxplots / violin plots
- [ ] `condition_score_summary.csv`

---

## B2. FP vs TN / FN vs TP analysis on POPE

### Task
Use POPE labels and prediction outcomes to compare:
- [ ] FP vs TN among ground-truth “no” questions
- [ ] FN vs TP among ground-truth “yes” questions

### What to check
- [ ] Are false positives associated with weaker / distorted projection patterns?
- [ ] Are false negatives showing the same geometry or a different one?

### Success criterion
This stage supports the hypothesis if:
- [ ] projection statistics separate error types from correct cases in a consistent way
- [ ] false positives are especially identifiable if the method is genuinely tied to hallucination risk

### Output
- [ ] per-error-type plots
- [ ] summary table by subset (`random/popular/adversarial`)

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

## E0. Implementation pre-check

### Goal
Verify that activation intervention is technically feasible and reproducible in the chosen `LLaVA-1.5-7B` codebase.

### Task
- [ ] Freeze one codebase only (`official` or `HF`) and do not switch midway
- [ ] Inspect the forward path and document:
  - [ ] where image embeddings are inserted / fused with text
  - [ ] the exact module names for transformer blocks
  - [ ] whether `output_hidden_states=True` gives the needed tensors directly
  - [ ] whether full-sequence hidden states can be intercepted and modified cleanly
- [ ] Decide one main intervention mechanism:
  - [ ] preferred: `register_forward_hook` / forward hook on target transformer block output
  - [ ] backup: manual forward split / wrapper module if hook semantics are unreliable
- [ ] Verify that the intervention can modify:
  - [ ] last-token hidden state only
  - [ ] full-sequence hidden states
- [ ] Run a no-op hook to confirm output equality before and after adding the hook
- [ ] Run a tiny random-direction intervention to confirm the logits change in the expected place

### Implementation notes to document
- [ ] whether image tokens are prepended / inserted before the LM blocks in this implementation
- [ ] whether the chosen hidden-state readout position is available during intervention
- [ ] whether sequence-length differences between blind and image-conditioned inputs create alignment problems

### Exit condition
Proceed to E1 only if:
- [ ] the intervention path is stable
- [ ] no-op hook preserves outputs exactly or near-exactly
- [ ] random-direction perturbation creates controlled, measurable output changes

If E0 fails:
- [ ] downgrade Stage E to a smaller pilot or skip rescue first
- [ ] explicitly state that causal claims remain unvalidated due to engineering constraints

---

## E1. Choose one or two target layers

### Task
- [ ] Select layers based on earlier evidence:
  - [ ] strongest stability
  - [ ] strongest spectral concentration
  - [ ] strongest predictive power
  - [ ] engineering feasibility from E0

### Output
- [ ] one short note explaining why the chosen layer(s) were selected

---

## E2. Decide intervention granularity

### Task
- [ ] Compare two intervention granularities on a very small pilot:
  - [ ] last-token-only intervention
  - [ ] full-sequence intervention
- [ ] Keep the stronger / more stable one as the main Stage E setting

### Success criterion
- [ ] main intervention mode is chosen by measured effect and reproducibility, not by convenience

---

## E3. Ablation along `V_K`

### Task
For samples that are originally correct, modify the hidden state at layer `l*`:

\[
z' = z - P_{V_K} z
\]

or the full-sequence analogue if full-sequence intervention is adopted.

### What to measure
- [ ] change in yes/no prediction
- [ ] drop in accuracy / F1
- [ ] rise in hallucination rate / FP rate

### Controls
- [ ] ablate random same-dimensional subspace
- [ ] ablate orthogonal complement slice of matched energy
- [ ] ablate at a weaker layer

### Success criterion
This stage supports the hypothesis if:
- [ ] ablating `V_K` harms grounded answering significantly more than control ablations

---

## E4. Rescue / steering along `V_K`

### Task
For samples that are originally hallucinated, try adding a direction aligned with truthful / grounded correction:

\[
z' = z + \alpha \cdot P_{V_K}(r)
\]

### Primary definition of `r`
Use a concrete default instead of a vague class average:
- [ ] define `r` as the mean correction direction from a truthful reference group
- [ ] primary choice: mean of `d_i = z_blind - z_img` over **TN samples**
- [ ] secondary choice: mean over all correct samples in a matched subset
- [ ] always project `r` into `V_K` before injection

### What to measure
- [ ] recovery in yes/no correctness
- [ ] reduction in FP rate
- [ ] whether rescue helps more than random-direction injection

### Hyperparameters
- [ ] test a small grid of `alpha`
- [ ] keep all other decoding settings fixed

### Success criterion
This stage supports the hypothesis if:
- [ ] targeted injection improves faithfulness more than matched random controls

---

## E5. Causal interpretation threshold

You may claim **causal relevance** only if at least one of the following is true:
- [ ] removing the subspace hurts grounded answering significantly more than controls
- [ ] injecting the subspace improves hallucinated cases significantly more than controls
- [ ] ideally both happen on the same layer family

If neither happens, then the subspace should be described as:
- [ ] correlationally informative
- [ ] useful for detection
- [ ] not yet causally validated

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
- [ ] matched vs mismatched vs blind condition score plots
- [x] AUROC-vs-K comparison plot
- [x] layerwise heatmap / diagnostic plot of performance
- [x] subspace-angle heatmap across layers
- [ ] intervention pre-check figure / sanity output
- [ ] intervention result bar chart
- [ ] semantic token cluster figure
- [ ] sample-level semantic direction panels
- [ ] optional caption sanity-check plot

### Essential tables
- [x] POPE baseline prediction table
- [x] feature family comparison table
- [x] best-layer summary table
- [ ] K-selection table
- [ ] intervention ablation/rescue table
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
- [ ] supported by matched vs mismatched vs blind comparisons
- [ ] stronger than random-subspace controls
- [ ] not reducible to the mere presence of any image input

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
- [ ] E0 engineering pre-check passed
- [ ] ablation harms grounded answering more than controls
- [ ] rescue improves hallucinated answers more than controls

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
- [ ] Stage B entirely
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
- [ ] Stage B1 matched vs mismatched vs blind comparison
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
