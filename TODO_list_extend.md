# TODO List — Paper-Upgrade Validation Plan for Blind-Reference Correction Geometry

## 0. Goal

This TODO list is designed as a **second-layer validation plan** built on top of the current `TODO_list.md` and `findings.md`.

The current project has already shown that blind-reference differences

\[
D^{(l)} = \{z^{(l)}_{\text{blind},i} - z^{(l)}_{\text{img},i}\}_{i=1}^N
\]

contain stable, condition-sensitive, hallucination-relevant geometry in `LLaVA-1.5-7B` on `POPE`.

This upgraded plan focuses on raising the paper level from:

> “There exists a useful hallucination-related subspace.”

into:

> “Paired blind-image differencing reveals a layered correction geometry in LVLMs: dominant image-induced movement, hallucination-sensitive residual/tail geometry, and late local arbitration/rescue geometry are related but not identical.”

The goal is not to add many scattered experiments, but to make the final paper more defensible, deeper, and more publishable.

---

## 1. Paper-Level Claim Ladder

Before adding experiments, freeze the claim ladder.

### 1.1 Strong claims already partly supported

- [x] Claim 1: `z_blind - z_img` forms a stable, concentrated image-conditioned difference structure.
  - Supported against Gaussian/noise controls, but not uniquely by pairing shuffle spectra.
- [x] Claim 2: the most variance-explaining directions are not the most hallucination-discriminative directions.
- [x] Claim 3: hallucination-relevant information is distributed in mid-rank / residual / tail coordinates, especially around middle layers.
- [x] Claim 4: matched visual evidence induces different correction geometry from random or adversarial mismatched evidence.
- [x] Claim 5: L24 residual/tail coordinates have causal relevance for correct TN decisions under ablation.

### 1.2 Claims that should remain cautious

- [x] Do **not** claim a universal “visual grounding subspace” yet.
- [x] Do **not** claim low-dimensional compactness unless a small `K` reaches near-peak performance.
- [x] Do **not** claim strong mitigation unless FP rescue improves substantially beyond current local/borderline effects.
- [x] Do **not** claim cross-model universality before at least one additional LVLM sanity check.

### 1.3 Recommended final wording

- [x] Preferred term: **grounding-related correction geometry**.
- [x] Stronger possible term after more evidence: **layered visual-evidence correction geometry**.
- [x] Avoid for now: **visual grounding subspace**, **universal hallucination subspace**, **causal grounding direction**.

---

## 2. Priority Overview

### P0 — Must finish before serious paper writing

- [x] Shuffle / destructive controls.
- [x] Token-position robustness pilot.
- [x] Multi-seed probe and confidence intervals.
- [x] Persist explicit train / validation / test sample-id splits.
- [x] Clean Stage E first-token intervention protocol.
  - Completed in Stage E first-token / margin-sweep runs; see `outputs/interventions/` and Stage E entries in `notes/findings.md`.
- [x] Paper-ready main tables and figures.
  - Initial Stage Q assets generated under `outputs/paper_tables/` and `outputs/paper_figures/`.

### P1 — Strongly recommended for ACL/EMNLP main-level ambition

- [x] Evidence-specific subspace extraction beyond plain SVD.
- [ ] Adaptive local rescue with memory-bank controls.
- [x] One external benchmark sanity check: `AMBER`, `HallusionBench`, or `FaithScore`.
  - Completed as an AMBER discriminative 1500-row pilot; full AMBER remains optional.
- [ ] One external mitigation / detection baseline comparison.
- [ ] Human-readable failure-case panels.

### P2 — High-impact extension, but not required for a first paper

- [ ] Cross-model replication on one additional open LVLM.
- [ ] Lightweight adapter / LoRA based on matched-vs-mismatch correction geometry.
- [ ] Free-form caption hallucination validation.
- [ ] Full representation-editing method paper.

---

## 3. Stage I — Reproducibility and Protocol Locking

This stage prevents reviewers from saying the results come from hidden choices.

## I1. Persist explicit data splits

### Task

- [x] Create fixed split files:
  - [x] `outputs/splits/pope_train_ids.json`
  - [x] `outputs/splits/pope_val_ids.json`
  - [x] `outputs/splits/pope_test_ids.json`
- [x] Stratify by:
  - [x] `subset`: random / popular / adversarial
  - [x] `label`: yes / no
  - [x] `outcome`: TP / TN / FP / FN when applicable
- [ ] Use the same split for all probes, subspace extraction, and intervention memory banks.
  - Current status: new Stage J controls use the fixed split; earlier completed Stage C/E runs have not been rerun under split-locked protocol.

### Success criterion

- [x] Every new Stage I/J reported number can be traced to a fixed split file.
- [x] Stage J subspace/classifier estimates use train IDs and evaluate on test IDs.

### Output

- [x] `outputs/splits/split_summary.csv`
- [x] `notes/protocol_lock.md`

---

## I2. Freeze exact prompt templates

### Task

Save all prompt templates verbatim:

- [x] image + question prompt
- [x] blind / text-only prompt
- [x] random mismatch prompt
- [x] adversarial mismatch prompt
- [ ] optional caption / free-form prompt

### What to check

- [x] Does blind vs image differ only by the presence of image tokens?
- [x] Does the textual instruction remain identical?
- [x] Does the answer format instruction bias Yes/No too strongly?

### Output

- [x] `notes/prompt_templates.md`
- [x] `outputs/sanity_checks/prompt_template_diff.txt`

---

## I3. Record hidden-state readout details

### Task

- [x] Define whether hidden states are:
  - [x] block output
  - [ ] post-attention
  - [ ] post-MLP
  - [ ] final normalized output
- [x] Record exact token index for:
  - [x] `last_prompt_token`
  - [ ] `first_answer_prefill`
  - [ ] `last_4_prompt_mean`
  - [ ] any image-adjacent text token

### Success criterion

- [x] A reader can reproduce exactly which tensor was extracted for the current `last_prompt_token` run.

### Output

- [x] `notes/hidden_readout_protocol.md`

---

## 4. Stage J — Destructive Controls for Geometry Reality

This is the most important missing control for the “real geometry” claim.

## J1. Shuffle image-question pairing

### Task

Construct destructive controls:

- [x] Real matched difference:

\[
d_i = z_{\text{blind},i} - z_{\text{img},i}
\]

- [x] Image-shuffled difference:

\[
d_i^{\text{img-shuffle}} = z_{\text{blind},i} - z_{\text{img},\pi(i)}
\]

- [x] Blind-shuffled difference:

\[
d_i^{\text{blind-shuffle}} = z_{\text{blind},\pi(i)} - z_{\text{img},i}
\]

- [x] Label-shuffled probe control.
- [x] Gaussian matrix matched by mean / variance.

### What to compare

- [x] singular spectrum decay
- [x] cumulative explained variance
- [x] effective rank
- [x] split-half stability
- [x] FP-vs-TN AUROC
- [x] matched-vs-mismatch separation
  - Covered by prior Stage B condition geometry; Stage J adds shuffle destructors.

### Success criterion

This supports the paper if:

- [ ] real matched differences have sharper spectrum than destructive controls;
  - Not supported against image/blind shuffle; supported only against Gaussian.
- [ ] real matched differences have higher split-half stability;
  - Not supported against image/blind shuffle; supported only against Gaussian.
- [x] real matched geometry has better FP/TN or condition separation;
  - Partly supported: better than image-shuffle/Gaussian at L24/L32, but blind-shuffle remains nontrivial.
- [x] label-shuffled probe collapses near chance.

### Output

- [x] `outputs/stage_j_controls/shuffle_spectrum_summary.csv`
- [x] `outputs/stage_j_controls/shuffle_probe_summary.csv`
- [x] `outputs/stage_j_controls/shuffle_stability_summary.csv`
- [x] `outputs/plots/stage_j_real_vs_shuffle_spectrum.png`
- [x] `outputs/plots/stage_j_real_vs_shuffle_auroc.png`

---

## J2. Random subspace controls with matched dimension

### Task

For each layer and K:

- [x] Compare SVD top-K subspace.
- [x] Compare random orthogonal K-dimensional subspace.
- [x] Compare random mid/tail bands with the same width.
- [x] Compare PCA on `z_img` and PCA on `z_blind`.

### Success criterion

- [ ] SVD or evidence-specific coordinates should beat random subspaces consistently.
  - Not supported for plain SVD top-K.
- [x] If random subspaces are close, report honestly that the signal is distributed rather than uniquely captured by SVD directions.

### Output

- [x] `outputs/stage_j_controls/random_subspace_control.csv`
- [x] `outputs/plots/stage_j_random_subspace_boxplot.png`

---

## 5. Stage K — Token-Position Robustness

This stage addresses a likely reviewer question: “Why last prompt token?”

## K1. Define candidate readout positions

### Candidate positions

- [x] `last_prompt_token`
- [x] `first_answer_prefill`
- [x] `last_4_prompt_mean`
- [x] `last_8_prompt_mean`
- [ ] `question_object_token_mean`, if object words can be located
- [ ] `image_adjacent_text_token`, if accessible

### Task

For L16 / L20 / L24 / L32:

- [x] Extract `z_img` and `z_blind` under each readout.
  - GPU script prepared: `scripts/run_gpu_stage_k_positions.sh`
- [x] Recompute `D`.
  - CPU analysis prepared: `scripts/analyze_stage_k_positions.py`
- [x] Run spectrum analysis.
  - CPU analysis prepared: `scripts/analyze_stage_k_positions.py`
- [x] Run FP-vs-TN probe.
  - CPU analysis prepared: `scripts/analyze_stage_k_positions.py`
- [x] Run matched-vs-mismatch condition scores.
  - GPU condition dump prepared: `scripts/run_gpu_stage_k_conditions.sh`
  - CPU SVD prep completed: `outputs/stage_k_svd/{position}/`
  - CPU condition analysis prepared: `scripts/run_cpu_stage_k_conditions.sh`
  - Outputs completed under `outputs/stage_k_condition_geometry/{position}/`

### Success criterion

The claim becomes stronger if:

- [x] the main conclusion survives at least two readout positions;
- [x] L20/L24 remain strong under multiple positions;
- [x] the “variance geometry ≠ discriminative geometry” finding persists.

### Output

- [x] `outputs/stage_k_positions/position_probe_summary.csv`
- [x] `outputs/stage_k_positions/position_spectrum_summary.csv`
- [x] `outputs/plots/stage_k_position_layer_heatmap.png`

---

## K2. Primary and secondary readout decision

### Task

- [x] Choose one primary readout for the main paper.
- [x] Choose one secondary readout for robustness appendix.
- [x] Write a short justification.

### Recommended decision rule

- [x] Primary: best combination of interpretability, reproducibility, and performance.
- [x] Secondary: most different but still stable readout.

### Output

- [x] `notes/readout_position_decision.md`

---

## 6. Stage L — Evidence-Specific Subspace Extraction

Plain SVD captures largest variance, but the current findings suggest hallucination signal is not in the top-variance backbone. This stage upgrades the method.

## L1. Contrastive / generalized eigenspace extraction

### Motivation

Instead of asking:

> Which directions explain most blind-image variance?

ask:

> Which directions explain visual-evidence-correct correction more than wrong-evidence correction?

### Candidate methods

- [x] Plain SVD on matched `D`.
- [x] Contrastive PCA: matched covariance minus mismatch covariance.
- [x] Generalized eigenspace:

\[
\Sigma_{\text{matched}} v = \lambda (\Sigma_{\text{mismatch}} + \epsilon I)v
\]

- [x] Fisher subspace: TN correction vs FP correction.
- [x] PLS directions using FP/TN label.
- [x] Matched-vs-adversarial discriminative subspace.

### Task

Run on L20 / L24 / L32 first.

For each method:

- [x] Extract K-dimensional subspace.
- [x] Evaluate FP-vs-TN AUROC.
- [x] Evaluate matched-vs-random and matched-vs-adversarial separation.
- [x] Evaluate stability across split halves.
- [ ] Test whether directions have better Stage E intervention behavior.

### Success criterion

This becomes a method contribution if:

- [x] evidence-specific subspace reaches similar or better AUROC with smaller K;
  - PLS FP/TN reaches AUROC 0.7196 at L24 K=32.
- [x] it separates matched vs adversarial evidence better than plain SVD;
  - Contrastive PCA gives the strongest condition gaps, especially L32.
- [ ] it improves FP rescue or TN ablation specificity.

### Output

- [x] `outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv`
- [x] `outputs/stage_l_evidence_subspace/evidence_subspace_condition_gap.csv`
- [x] `outputs/stage_l_evidence_subspace/evidence_subspace_stability.csv`
- [x] `outputs/plots/stage_l_plain_svd_vs_evidence_specific.png`

---

## L2. Subspace naming decision

### Task

Based on L1, decide whether the extracted geometry should be called:

- [x] blind-image difference geometry
- [x] correction geometry
- [x] evidence-specific correction geometry
- [x] hallucination-sensitive residual geometry
- [ ] visual grounding subspace

### Decision rule

- [x] If only matched vs blind is separated: use **blind-image difference geometry**.
- [x] If matched vs mismatch is separated: use **evidence-sensitive correction geometry**.
- [ ] If interventions work cleanly: use **causally relevant correction geometry**.
- [x] Use **visual grounding subspace** only if semantic and external benchmark evidence is strong.

### Output

- [x] `notes/naming_decision.md`

---

## 7. Stage M — Adaptive Local Rescue

Current rescue is weak but meaningful: global steering is not enough, while local TN-conditioned directions show small boundary effects. This stage turns that into a more principled experiment.

## M1. Build a train-only correction memory bank

### Memory bank entries

For each train sample, store:

- [x] sample id
- [x] question
- [x] queried object
- [x] POPE subset
- [x] ground-truth label
- [x] outcome type
- [x] layer
- [x] correction vector `d = z_blind - z_img`
- [x] SVD coordinates
- [x] tail coordinates
- [ ] yes/no margin
  - Not available in the current saved POPE prediction artifact; requires a logits/margin dump.

### Retrieval keys

- [x] same object
- [ ] semantically similar object
- [ ] same question template
- [x] nearest neighbor in SVD-coordinate space
- [x] nearest neighbor in tail-coordinate space
- [x] nearest TN sample only
  - Retrieval policy script completed: `scripts/prepare_stage_m_retrieval_plan.py`.

### Critical leakage control

- [x] Build the memory bank using train split only.
- [x] Never retrieve from validation/test samples.
- [x] Report whether retrieved examples share the same image id.
- [x] Add random retrieval control.
  - Retrieval plan includes `random_tn` and excludes same-image candidates by default.

### Output

- [x] `outputs/stage_m_local_rescue/memory_bank_train.pt`
- [x] `outputs/stage_m_local_rescue/retrieval_audit.csv`

---

## M2. Gated local steering

Preparation status:

- [x] GPU runner prepared: `scripts/run_gpu_stage_m_local_rescue.sh`
- [x] Local rescue implementation prepared: `scripts/run_stage_m_local_rescue.py`
- [x] CPU analysis prepared: `scripts/run_cpu_stage_m_local_rescue.sh`
- [x] Analysis implementation prepared: `scripts/analyze_stage_m_local_rescue.py`
- [x] Dry-run passed and wrote `outputs/stage_m_local_rescue/run_stage_m_local_rescue_summary.json`.

### Task

Only apply rescue when the model is uncertain or hallucination-prone.

Candidate gates:

- [x] low absolute yes/no margin
  - Implemented and evaluated in the current L32 run.
- [x] high entropy over yes/no logits
  - Implemented as a Stage M gate; not enabled in the default wrapper.
- [x] high FP-risk probe score
  - Implemented using a train-bank FP/TN tail-coordinate logistic score and evaluated in the current L32 run.
- [ ] weak matched-vs-mismatch tail correction score
- [x] combination of margin + tail score
  - Implemented as `margin_and_fp_risk`; this uses the tail-coordinate FP-risk score rather than a matched-vs-mismatch score.

### Steering directions

- [x] global TN mean correction
- [x] same-object TN mean correction
- [x] kNN TN correction
- [x] local correction minus local FP mean
  - Implemented, but the completed L32 run did not include `same_object_fp` in `--retrieval-modes`, so this direction was not evaluated in the current result. The default wrapper has been updated for the next run.
- [ ] evidence-specific subspace direction from Stage L
- [x] random retrieved direction control

### Metrics

- [x] FP rescue rate
  - Computed in the current L32 run.
- [x] yes/no margin shift
  - Computed in the current L32 run.
- [x] clean TN damage rate
  - Computed in the current L32 run.
- [x] TP damage rate
  - Computed in the current L32 run.
- [x] unknown / malformed output rate
  - Computed in the current L32 run.
- [x] net accuracy change
  - Analysis script prepared through paired baseline/intervention correctness.
- [x] McNemar significance test
  - Computed in the current L32 run.

### Success criterion

This becomes paper-worthy if:

- [ ] FP margin improves significantly without damaging TN/TP too much;
  - Current L32 pilot: FP margins move toward `No`, especially low-margin FP samples, but McNemar p-values are not significant and one borderline TP can be damaged at alpha 8.
- [ ] local retrieval beats global mean direction;
  - Not supported in the current L32 pilot; global and random TN directions are as strong as or stronger than local kNN directions.
- [ ] random retrieval control is much weaker;
  - Not supported in the current L32 pilot; `random_tn` is competitive and sometimes strongest.
- [x] effect is strongest for low-margin/borderline FP samples.
  - Supported: only two FP samples flip, with baseline yes-minus-no margins `0.015625` and `0.03125`.

### Output

- [x] `outputs/stage_m_local_rescue/local_rescue_results.csv`
- [x] `outputs/stage_m_local_rescue/local_rescue_summary.csv`
- [x] `outputs/plots/stage_m_fp_margin_shift.png`
- [x] `outputs/plots/stage_m_gate_tradeoff_curve.png`

---

## M3. Rescue failure analysis

### Task

For every FP sample, label rescue outcome:

- [x] rescued to correct `No`
  - Current result: 2 / 32 FP samples.
- [x] margin improved but answer unchanged
  - Current result: 30 / 32 FP samples.
- [x] no effect
  - Category implemented; no FP samples fell into this bucket in the current L32 run.
- [x] damaged / malformed
  - Category implemented; no FP samples fell into this bucket in the current L32 run.
- [x] moved in wrong direction
  - Category implemented; no FP samples fell into this bucket in the current L32 run.

Then analyze by:

- [x] POPE subset
- [x] object category
- [x] baseline yes/no margin
- [x] retrieval similarity
- [x] tail energy
- [x] supervised FP score

### Success criterion

Even if rescue remains weak, this can support a strong mechanism claim if:

- [x] rescue works mainly on low-margin cases;
  - Supported: rescued FP samples have baseline margins `0.015625` and `0.031250`.
- [x] failures are high-confidence language-prior hallucinations;
  - Partly supported: unrescued but margin-improved FP samples have median baseline margin `0.656250`; high-margin FP cases remain unrescued.
- [ ] local directions are necessary but not sufficient for correction.
  - Not supported by the current L32 run because global/random TN controls remain competitive.

### Output

- [x] `outputs/stage_m_local_rescue/rescue_failure_taxonomy.csv`
- [x] `notes/rescue_failure_analysis.md`

---

## 8. Stage N — External Validity Beyond POPE

This is the main difference between a “promising internal study” and a stronger paper.

## N1. Choose one external benchmark first

### Candidate benchmarks

- [x] `AMBER` discriminative part: best first choice if you want existence / attribute / relation extension.
- [ ] `HallusionBench`: best first choice if you want image-context reasoning and visual illusion.
- [ ] `FaithScore`: best first choice if you want free-form response faithfulness.
- [ ] `MMHal-Bench`: useful but may depend on LLM-as-judge evaluation.
- [ ] caption sanity check: useful but less directly connected to the current yes/no setup.

### Recommended first external benchmark

- [x] Start with **AMBER discriminative** if implementation time is limited.
- [ ] Use **HallusionBench** if the goal is stronger conceptual framing.

### Output

- [x] `notes/external_benchmark_choice.md`

---

## N2. Transfer subspace without refitting

Preparation status:

- [x] AMBER plan preparation script: `scripts/prepare_stage_n_amber.py`
- [x] AMBER GPU yes/no evaluation script: `scripts/run_stage_n_amber_eval.py`
- [x] AMBER GPU wrapper: `scripts/run_gpu_stage_n_amber.sh`
- [x] Zero-shot POPE-SVD transfer analysis script: `scripts/analyze_stage_n_external_transfer.py`
- [x] CPU transfer wrapper: `scripts/run_cpu_stage_n_transfer.sh`
- [x] Dry-run completed for AMBER plan and eval scripts.
- [x] AMBER data/images available locally under `data/amber/`.
  - Actual layout: query/annotations under `data/amber/data/`, images under `data/amber/image/`.
- [x] AMBER discriminative pilot plan prepared.
  - Current plan: 1500 rows using `MAX_PER_DIMENSION_LABEL=300`; use `FULL=1 scripts/run_cpu_stage_n_amber_prepare.sh` to regenerate all 14216 rows.
- [x] Full AMBER discriminative plan prepared separately.
  - Full plan: `outputs/stage_n_external_full/amber_discriminative_plan.jsonl`, 14216 rows, missing images 0, invalid labels 0.

### Task

Use POPE-trained / POPE-estimated geometry on the external benchmark:

- [x] Extract hidden states on external benchmark.
  - Completed for AMBER pilot: `outputs/stage_n_external/amber_hidden/layer_{20,24,32}.pt`.
- [x] Apply POPE SVD basis directly.
  - Completed via SVD energy features in `scripts/analyze_stage_n_external_transfer.py`.
- [x] Apply POPE evidence-specific basis directly.
  - Completed for PLS FP/TN and Fisher FP/TN bases; reusable basis artifacts are saved under `outputs/stage_n_external/evidence_transfer_bases/`.
- [x] Do not refit subspace on external test data.
  - Current transfer uses POPE SVD and POPE-train FP/TN probes only.
- [x] Evaluate whether scores still separate correct vs hallucinated cases.
  - Completed on the AMBER 1500-row pilot.

### Metrics

- [x] AUROC / AUPRC if binary labels exist.
- [ ] correlation with hallucination severity if scalar labels exist.
- [x] per-category score gap: existence / attribute / relation.
  - Implemented as per-dimension AUROC/AUPRC summaries for `attribute`, `existence`, and `relation`.
- [ ] calibration curve if using risk scores.

### Success criterion

Strong if:

- [x] POPE geometry transfers above chance to AMBER/HallusionBench;
  - AMBER pilot supports above-chance transfer for POPE-trained FP-risk probes, especially `L20 existence top-4` FP AUROC `0.771`, `L24 relation top-256` FP AUROC `0.700`, and `L24 attribute top-256` FP AUROC `0.632`.
- [ ] tail/residual geometry transfers better than top backbone;
  - Not supported in the current pilot; top-4/top-64/top-256 POPE probes are stronger than the tail probe for the best categories.
- [ ] evidence-specific subspace transfers better than plain SVD.
  - Partly supported only under the energy-only comparison. Evidence-specific PLS/Fisher probes outperform raw SVD energy, but they do not beat the strongest POPE top-SVD coordinate probe in this pilot.

Weak but still useful if:

- [x] only existence transfers, but attribute/relation does not.
  - Current result is stronger than this weak criterion: existence is strongest, but relation and attribute also show above-chance transfer with POPE-trained probes.

### Output

- [x] `outputs/stage_n_external/external_transfer_scores.csv`
- [x] `outputs/stage_n_external/external_category_summary.csv`
- [x] `outputs/plots/stage_n_external_transfer.png`
- [x] `outputs/stage_n_external/evidence_transfer_bases/layer_{20,24,32}.pt`

---

## N3. Small refit sanity check

### Task

If zero-shot transfer is weak:

- [ ] Fit a tiny logistic probe on external train split.
- [ ] Compare raw hidden state vs blind-reference difference vs SVD coordinates.
- [ ] Check whether the same L20/L24 pattern reappears.

### Success criterion

- [ ] If the same layer/band pattern reappears after small refit, the mechanism is likely not POPE-only.
- [ ] If not, report that the current geometry is mainly object-existence / POPE-specific.

### Output

- [ ] `outputs/stage_n_external/external_refit_probe.csv`
- [ ] `notes/external_validity_conclusion.md`

---

## 9. Stage O — Optional Cross-Model Replication

This is not mandatory for the first version, but it determines whether the paper can claim generality.

## O1. Select one additional open LVLM

### Candidate models

- [ ] `LLaVA-1.6` / `LLaVA-NeXT`
- [ ] `Qwen2-VL-7B-Instruct`
- [ ] `InternVL2` or similar open LVLM
- [x] a smaller/LLaVA-HF-compatible model if compute or wrapper support is limited
  - Current environment supports `transformers.LlavaForConditionalGeneration`, but not LLaVA-NeXT/Qwen2-VL wrapper classes.
  - Completed with local `/data/lh/ModelandDataset/llava-1.5-13b-hf` as `MODEL_ALIAS=llava_13b`.

### Selection rule

- [x] Prefer a model you can run locally and hook reliably.
- [ ] Prefer one with different architecture from LLaVA-1.5.
  - Deferred because the current installed transformers version would require wrapper/dependency work first.
- [x] Do not choose a model that requires rewriting the whole pipeline first.

### Output

- [x] `notes/cross_model_choice.md`

---

## O2. Minimal replication only

### Task

Run only the minimum chain:

- [x] POPE prediction summary.
  - GPU script prepared: `scripts/run_gpu_stage_o_cross_model.sh`
- [x] hidden states for L20-like / L24-like / late layer.
  - GPU script prepared: `scripts/run_gpu_stage_o_cross_model.sh`
- [x] blind-image difference SVD.
  - CPU script prepared: `scripts/run_cpu_stage_o_cross_model.sh`
- [x] Stage C FP-vs-TN probe.
  - CPU script prepared: `scripts/run_cpu_stage_o_cross_model.sh`
- [x] Stage B matched vs mismatch tail score.
  - GPU condition dump and CPU analysis scripts prepared.
- [x] no Stage E unless early results are promising.

### Success criterion

Strong if:

- [x] variance/discrimination mismatch appears again;
- [x] middle-layer residual/tail geometry is again stronger than top backbone;
- [x] matched-vs-mismatch evidence gap appears again.
  - Partly supported: adversarial mismatch tail gap reproduces strongly; random mismatch tail gap is mixed.

Acceptable if:

- [x] only some qualitative patterns reproduce.

### Output

- [x] `outputs/stage_o_cross_model/{MODEL_ALIAS}/minimal_replication_summary.csv`
- [x] `notes/cross_model_replication.md`
  - Summary builder prepared: `scripts/build_stage_o_cross_model_summary.py`

---

## 10. Stage P — Statistical Significance and Robustness

## P1. Multi-seed reruns

### Task

For main probe and subspace results:

- [x] Run at least 5 seeds.
- [x] Vary train/test split seed.
- [x] Vary logistic regression seed if relevant.
- [ ] Vary split-half subspace seed.
  - Not included in the first Stage P run; this run targets probe split robustness.

### Metrics

- [x] mean ± std
- [x] 95% bootstrap CI
- [x] min/max across seeds
- [x] rank stability of best layers

### Output

- [x] `outputs/stage_p_stats/multiseed_probe_summary.csv`
- [x] `outputs/stage_p_stats/multiseed_layer_rank.csv`

---

## P2. Significance tests

### Recommended tests

- [x] AUROC difference: DeLong or stratified bootstrap.
  - Completed with stratified paired bootstrap over FP/TN test predictions.
- [ ] Accuracy difference: McNemar test.
- [ ] AUPRC / F1: paired bootstrap.
- [ ] Margin shift: paired Wilcoxon or permutation test.
- [ ] Subspace similarity: permutation test.
- [ ] Flip-rate comparison: McNemar or bootstrap.

### Output

- [x] `outputs/stage_p_stats/significance_tests.csv`
- [x] `notes/statistical_testing_protocol.md`

---

## 11. Stage Q — Paper-Ready Figures and Tables

## Q1. Main figures

### Figure 1 — Method overview

- [x] Show image+question and blind question paths.
- [x] Show `z_img`, `z_blind`, and `D = z_blind - z_img`.
- [x] Show SVD / residual bands / probes / interventions.

### Figure 2 — Variance vs hallucination discrimination

- [x] cumulative explained variance vs K
- [x] AUROC vs K
- [x] same layers side by side
- [x] highlight that top variance directions are weakly discriminative

### Figure 3 — Matched vs mismatched evidence geometry

- [x] matched / random mismatch / adversarial mismatch / blind score distributions
  - Initial figure uses matched / random mismatch / adversarial mismatch; blind is zero by construction and omitted from the boxplot.
- [x] separate top-backbone and residual/tail views

### Figure 4 — Causal ablation dose curve

- [x] L24 tail ablation alpha curve
- [x] yes/no margin shift
- [x] norm-matched random control
- [x] first-token setting separated from full-sequence setting

### Figure 5 — Layered geometry summary

- [x] layer vs spectrum concentration
- [x] layer vs FP/TN AUROC
- [x] layer vs intervention sensitivity
  - Initial figure uses Stage E layer-sweep FP rescue gain / rescue rate for the intervention-sensitivity panel.
- [ ] emphasize “middle-layer correction, late-layer arbitration” if supported
  - Keep cautious for now; current Figure 5 is a summary aid, not final proof of the full layered-arbitration claim.

### Output

- [x] `outputs/paper_figures/fig1_method_overview.pdf`
- [x] `outputs/paper_figures/fig2_variance_vs_auroc.pdf`
- [x] `outputs/paper_figures/fig3_condition_geometry.pdf`
- [x] `outputs/paper_figures/fig4_intervention_dose.pdf`
- [x] `outputs/paper_figures/fig5_layered_geometry.pdf`

---

## Q2. Main tables

### Table 1 — Main POPE performance and outcome distribution

- [x] subset
- [x] accuracy
- [x] TP / TN / FP / FN
- [x] yes rate
- [x] no rate

### Table 2 — Feature comparison

- [x] raw `z_img`
- [x] raw `z_blind`
- [x] full difference
- [x] top-K SVD coordinates
- [x] full SVD coordinates
- [x] random projection
- [x] evidence-specific subspace, if available

### Table 3 — Geometry controls

- [x] real matched
- [x] image-shuffled
- [x] blind-shuffled
- [x] label-shuffled
- [x] random Gaussian

### Table 4 — Intervention summary

- [x] layer
- [x] direction family
- [x] alpha
- [x] margin shift
- [x] flip count
- [x] clean-control damage
- [x] unknown rate

### Output

- [x] `outputs/paper_tables/table1_pope_summary.csv`
- [x] `outputs/paper_tables/table2_feature_comparison.csv`
- [x] `outputs/paper_tables/table3_controls.csv`
- [x] `outputs/paper_tables/table4_intervention.csv`

---

## 12. Stage R — Semantic Interpretation Upgrade

Stage G exists already, but this upgraded version should avoid overclaiming single directions.

## R1. Semantic fingerprint rather than single-direction semantics

### Task

For each geometric object:

- [x] top SVD backbone directions
- [x] mid-rank discriminative bands
  - Covered indirectly through top-SVD/backbone and Stage C/R summaries; individual mid-rank vocabulary panels remain optional.
- [x] tail correction bands
- [x] local rescue directions
- [x] evidence-specific directions from Stage L
  - Included as quantitative-only Stage L rows; not vocabulary-projected yet.

Compute:

- [x] nearest vocabulary tokens
  - Completed for projected Stage G objects.
- [x] top positive / negative samples
- [x] object-category enrichment
- [ ] image-region / object-label correlation, if possible
  - Not available from current artifacts; would require object detections/regions.
- [x] FP/TN score difference
- [x] matched/mismatch condition difference
  - Included for Stage L evidence-specific quantitative rows and supported by Stage B/Stage L condition summaries.

### Success criterion

Use semantic interpretation only if:

- [x] token-level interpretation and sample-level extremes agree;
  - Partly supported for representative objects; summarized cautiously in `notes/semantic_interpretation_conclusion.md`.
- [ ] semantic clusters are stable across seeds or split halves;
  - Not tested yet.
- [x] results do not imply one single direction is a strong classifier unless metrics support it.

### Output

- [x] `outputs/stage_r_semantics/semantic_fingerprint_summary.csv`
- [x] `outputs/stage_r_semantics/semantic_sample_panels/`
- [x] `notes/semantic_interpretation_conclusion.md`

---

## R2. Human-readable case panels

### Task

Prepare 20–30 case studies:

- [x] successful TN with strong tail correction
- [x] FP with weak matched correction
- [x] FP rescued by local steering
  - Current Stage M run contains 2 unique rescued FP samples.
- [x] FP not rescued despite high score
- [x] adversarial mismatch example
- [x] semantic direction extreme samples

Each case should include:

- [x] image id
- [x] question
- [x] ground truth
- [x] model answer
- [x] geometry scores
- [x] intervention result, if any
- [x] short human explanation

### Output

- [x] `outputs/case_studies/case_panel_metadata.csv`
- [x] `notes/case_studies.md`

---

## 13. Stage S — Baseline Positioning

## S1. Detection baselines

### Task

Compare your risk score against:

- [x] raw hidden-state probe
- [x] yes/no margin baseline
  - Available only on the Stage M first-token subset, not the full POPE set.
- [x] entropy baseline
  - Available only on the Stage M first-token subset.
- [ ] image-text similarity baseline, if available
  - Not available in current artifacts; no CLIP/image-text similarity run.
- [x] HALP-style single-forward hidden-state probe, if reproducible
  - Included as a raw image-state hidden probe proxy; no reproduced HALP implementation is available.

### Success criterion

- [x] Your method should not be framed as merely “higher AUROC” unless it wins clearly.
- [x] If it does not win, emphasize mechanism: paired differencing explains where the signal lives and why top variance is misleading.

### Output

- [x] `outputs/stage_s_baselines/detection_baseline_comparison.csv`

---

## S2. Mitigation baselines

### Task

Compare intervention or rescue against:

- [x] no intervention
- [ ] VCD or ICD baseline if available in your existing code
  - Not available in current repository artifacts.
- [x] random steering control
- [x] global mean correction
- [x] local TN correction
- [ ] evidence-specific correction
  - Stage L evidence-specific subspaces are not wired into steering yet.

### Metrics

- [x] FP reduction
- [x] TN preservation
  - Undefined for some gated Stage M settings where no TN/TP controls passed the gate; recorded explicitly in the notes.
- [x] overall accuracy
- [x] malformed output rate
- [ ] compute overhead
  - Not measured yet.

### Output

- [x] `outputs/stage_s_baselines/mitigation_baseline_comparison.csv`

---

## 14. Stage T — Final Paper Decision

After the upgraded TODO is mostly complete, decide the paper type.

## T1. If controls are strong but rescue remains weak

### Best paper framing

- [x] Mechanistic analysis paper.
- [x] Main contribution: layered correction geometry.
- [x] Intervention is causal evidence for necessity, not a full mitigation method.

### Suggested title style

- [x] “Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination”
- [x] “Dominant Difference Is Not Decision Geometry: A Blind-Reference Analysis of LVLM Hallucination”

### Output

- [x] `notes/final_paper_framing_decision.md`

---

## T2. If local rescue improves substantially

### Best paper framing

- [ ] Mechanism + method paper.
  - Not supported by current Stage M/S rescue results.
- [ ] Main contribution: evidence-specific correction geometry plus adaptive local steering.
- [ ] Intervention becomes a real mitigation method.

### Required extra evidence

- [ ] external benchmark improvement
- [ ] clean-control preservation
- [ ] compute overhead comparison
- [ ] failure analysis

---

## T3. If external transfer fails

### Best paper framing

- [ ] Narrow but honest POPE/object-existence mechanism paper.
  - Full AMBER transfer is modest but nonzero, so this is too narrow as the primary framing.
- [ ] Emphasize limitations.
- [ ] Avoid universal claims.
- [ ] Use external failure as insight: object-existence correction differs from attribute/relation faithfulness.

---

## T4. If cross-model replication succeeds

### Best paper framing

- [ ] General LVLM mechanism paper.
- [ ] Stronger venue target.
- [ ] Claim becomes about a recurring pattern across LVLMs, not a LLaVA artifact.

---

## 15. Minimum Completion Checklist

A practical minimum version of this upgraded TODO should complete:

- [x] J1 shuffle controls
- [x] J2 random subspace controls
- [x] K1 token-position pilot
- [x] P1 five-seed probe rerun
- [x] P2 significance tests
- [x] Q1 main figures 1–4
- [x] Q2 main tables 1–4
- [x] R2 case panels
- [x] T1/T2/T3 final paper framing decision

A stronger version should additionally complete:

- [x] L1 evidence-specific subspace extraction
  - Extraction/evaluation completed; intervention use remains unwired.
- [x] M1/M2 adaptive local rescue
  - Memory bank, gated steering, and analysis completed; rescue remains weak and boundary-local.
- [x] N1/N2 one external benchmark transfer
  - Completed for AMBER discriminative pilot, including PLS/Fisher evidence-specific basis transfer.
- [x] S1/S2 baseline positioning

A high-impact version should additionally complete:

- [x] O1/O2 cross-model minimal replication
  - Completed as a LLaVA-family checkpoint replication (`llava_13b`), not a different-architecture LVLM replication.
- [ ] lightweight adapter or representation-editing extension
  - CPU-side representation-editing direction bank prepared: `outputs/representation_editing_prep/editing_direction_bank.pt`.
  - This is not a trained LoRA adapter and has not yet been GPU-evaluated.

---

## 16. Running Summary Template

Each time a stage is completed, add a short entry to `notes/findings_upgrade.md`:

```markdown
## YYYY-MM-DD Stage X Result

Artifacts:

- `outputs/...`

Setup:

- model:
- dataset:
- layer(s):
- readout:
- split:
- seed(s):

Main result:

- ...

Interpretation:

- ...

Claim status:

- Supported:
- Weakened:
- Still untested:

Next step:

- ...
```

---

## 17. Final Rule

Every upgraded experiment should answer one of four questions:

1. **Is the geometry real rather than an artifact?**
2. **Is it evidence-specific rather than merely image-presence-specific?**
3. **Is it hallucination-relevant rather than just high-variance?**
4. **Is it causally useful or only diagnostically useful?**

If an experiment does not answer one of these questions, it should not be prioritized.
