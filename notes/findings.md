# Findings

Record validated findings here. Keep claims tied to concrete artifacts in
`outputs/` and to rows in `claim_evidence_table.md`.

## 2026-04-22 POPE / LLaVA-1.5-7B Initial Chain

Artifacts:

- Predictions: `outputs/predictions/pope_predictions.jsonl`
- Hidden states: `outputs/hidden_states/layer_{8,12,16,20,24,28,32}.pt`
- Difference matrices and SVD: `outputs/svd/`
- Probe results: `outputs/probes/`
- Plots: `outputs/plots/`

Prediction summary:

- Samples: 9000
- Accuracy: 0.8619
- TP/TN/FP/FN: 3606 / 4151 / 349 / 894

Stage A observations:

- Top-4 directions explain a large fraction of difference-matrix variance for most layers.
- Explained variance at K=4:
  - L8: 0.8859
  - L12: 0.8541
  - L16: 0.8364
  - L20: 0.8421
  - L24: 0.8773
  - L28: 0.8754
  - L32: 0.7270
- L32 is visibly less concentrated than middle layers.
- Split-half stability is strongest at K=4 in the current randomized/subsampled estimate.

Stage C observations:

- Best FP-vs-TN AUROC so far is full difference at L24: 0.6936.
- Full difference features outperform projected low-K features in this first probe setting.
- Best projected difference result in the top rows is L20 K=64 AUROC 0.6338.

Interpretation status:

- Stage A supports a real, concentrated difference structure.
- Stage C currently supports hallucination relevance for the full difference vector, but does not yet show a compression advantage for low-K projected features.
- Causal claims remain untested.

## 2026-04-22 Stage C Deep Follow-Up

Artifacts:

- Top-K AUROC curve: `outputs/stage_c_deep/stage_c_topk_curve.csv`
- SVD band probe: `outputs/stage_c_deep/stage_c_band_probe.csv`
- Layer diagnostics: `outputs/stage_c_deep/stage_c_layer_diagnostics.csv`
- Plots:
  - `outputs/plots/stage_c_topk_auroc_explained_variance.png`
  - `outputs/plots/stage_c_band_probe_auroc.png`
  - `outputs/plots/stage_c_layer_diagnostics.png`
  - `outputs/plots/stage_c_layer_{layer}_auroc_vs_variance.png`

Main result:

- The cumulative explained variance and FP-vs-TN AUROC are not synchronized.
- Top-4 directions already explain a very large amount of variance, but usually have weak AUROC.
- Discriminative performance grows mainly when K reaches 64, 128, or 256.

Top-K projected difference AUROC:

| Layer | K=4 | K=64 | K=128 | K=256 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.4689 | 0.5528 | 0.6091 | 0.6374 |
| 12 | 0.4650 | 0.5899 | 0.6163 | 0.6526 |
| 16 | 0.4653 | 0.6242 | 0.6761 | 0.6862 |
| 20 | 0.5570 | 0.6338 | 0.6846 | 0.6948 |
| 24 | 0.4637 | 0.6192 | 0.6539 | 0.6496 |
| 28 | 0.4807 | 0.6185 | 0.6352 | 0.6515 |
| 32 | 0.5005 | 0.5652 | 0.5900 | 0.6185 |

Corresponding explained variance:

| Layer | K=4 | K=64 | K=128 | K=256 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.8859 | 0.9595 | 0.9759 | 0.9883 |
| 12 | 0.8541 | 0.9424 | 0.9633 | 0.9805 |
| 16 | 0.8364 | 0.9336 | 0.9563 | 0.9759 |
| 20 | 0.8421 | 0.9345 | 0.9563 | 0.9754 |
| 24 | 0.8773 | 0.9474 | 0.9651 | 0.9806 |
| 28 | 0.8754 | 0.9430 | 0.9615 | 0.9784 |
| 32 | 0.7270 | 0.8842 | 0.9231 | 0.9584 |

Layer observations:

- L20 is currently the strongest top-K projected layer, with K=256 AUROC 0.6948.
- L24 remains strongest for full difference, with AUROC 0.6936, but its best top-K projected AUROC is lower at 0.6539.
- L16 has weaker full difference AUROC than L20/L24, but top-K projected AUROC rises strongly to 0.6862 at K=256.
- L32 remains weaker: it is less variance-concentrated and also weaker for top-K projected probing.

Band probe observations:

- Best SVD band result is L20 directions 65-128, AUROC 0.6317.
- L20 directions 33-64 also carry signal, AUROC 0.6181.
- Top 1-4 directions explain the largest variance block, but are not the most discriminative.
- Some random-width controls are nontrivial, so the band result should be treated as evidence of distributed signal, not yet as a uniquely semantic singular-direction claim.

Current interpretation:

- The main low-rank geometric structure and the hallucination-discriminative structure are related but not identical.
- The most variance-explaining directions are not the most hallucination-discriminative directions.
- A more accurate description may be: hallucination information lives in a broader subspace that includes mid-rank directions, especially around layers 16-24.

## 2026-04-23 Stage C Supervised Subspace Follow-Up

Artifacts:

- Supervised/SVD alignment: `outputs/stage_c_supervised/stage_c_supervised_alignment.csv`
- Extended K curve: `outputs/stage_c_supervised/stage_c_extended_k_curve.csv`
- Cumulative / removal probes: `outputs/stage_c_supervised/stage_c_cumulative_exclusion.csv`
- Band exclusion probes: `outputs/stage_c_supervised/stage_c_band_exclusion.csv`
- Plots:
  - `outputs/plots/stage_c_supervised_alignment_logistic_weight.png`
  - `outputs/plots/stage_c_supervised_alignment_lda_fisher.png`
  - `outputs/plots/stage_c_supervised_alignment_pls_8.png`
  - `outputs/plots/stage_c_extended_topk_auroc.png`
  - `outputs/plots/stage_c_cumulative_exclusion_auroc.png`
  - `outputs/plots/stage_c_band_exclusion_auroc.png`

Supervised-vs-SVD alignment:

- Logistic and LDA/Fisher 1D discriminative directions are almost orthogonal to the very top SVD directions.
- For L20, logistic-weight projection similarity into top-SVD subspaces is:
  - K=4: 0.0004
  - K=64: 0.0077
  - K=256: 0.0734
  - K=1024: 0.4838
- For L20, LDA/Fisher projection similarity is even smaller at moderate K:
  - K=4: 0.0000
  - K=64: 0.0010
  - K=256: 0.0123
  - K=1024: 0.2188
- PLS-8 aligns more strongly with the SVD basis as K grows. For L20:
  - K=4: 0.2102
  - K=64: 0.6626
  - K=256: 0.7753
  - K=1024: 0.8884

Extended K observations:

| Layer | Best K | Best AUROC |
| --- | ---: | ---: |
| 8 | 512 | 0.6919 |
| 12 | 512 | 0.6695 |
| 16 | 256 | 0.6862 |
| 20 | 256 | 0.6948 |
| 24 | 1024 | 0.6603 |
| 28 | 512 | 0.6519 |
| 32 | 1024 | 0.6532 |

- K=256 is still the best top-K setting for L16/L20.
- L8/L12 improve to K=512, while L24/L32 continue rising to K=1024 but remain below L20's peak.
- Therefore, saturation differs by layer. L20 looks closer to a predictive plateau around K=256, while L24/L32 are slower and less efficient.

Cumulative / exclusion observations:

- Full SVD-coordinate probes on focused layers show stronger AUROC than top-K-only probes:
  - L20: 0.7343
  - L24: 0.7096
  - L28: 0.6957
  - L32: 0.7139
- Removing top variance directions does not destroy discriminative performance. Even after removing directions 1-1024:
  - L20 remains at AUROC 0.7232
  - L24 remains at AUROC 0.6809
  - L28 remains at AUROC 0.6578
  - L32 remains at AUROC 0.6852
- The most damaging single band removals are modest:
  - L20 remove 129-256: delta -0.0149
  - L24 remove 257-512: delta -0.0156
  - L28 remove 513-1024: delta -0.0124
  - L32 remove 33-64: delta -0.0028

Current interpretation:

- The discriminative subspace is not simply the top-variance SVD backbone.
- Supervised 1D discriminative directions are substantially off-backbone at low and moderate K.
- Hallucination-discriminative signal appears distributed across many residual / mid-to-tail SVD coordinates rather than being functionally necessary in the top few variance directions.
- This strengthens the claim that the dominant blind-reference geometry and the hallucination decision geometry are distinct but partially coupled objects.

## 2026-04-23 Stage C Coordinate-Control Sanity Check

Artifacts:

- Same-split coordinate control: `outputs/stage_c_coordinate_control/stage_c_coordinate_control.csv`
- Plot: `outputs/plots/stage_c_coordinate_control_auroc.png`

Control setup:

- Layers: L20 / L24 / L32
- Target: FP vs TN
- Same train/test split for all feature representations
- Same `StandardScaler`, logistic solver, class weighting, random seed, and regularization strength
- Compared:
  - raw full difference
  - train-split PCA-whitened difference
  - full SVD coordinates
  - dense random orthogonal rotation of the full difference

AUROC:

| Layer | Raw full diff | PCA-whitened | Full SVD coords | Random orthogonal |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.6869 | 0.6692 | 0.7343 | 0.6776 |
| 24 | 0.6936 | 0.6331 | 0.7096 | 0.6896 |
| 32 | 0.6694 | 0.6612 | 0.7139 | 0.6711 |

Interpretation:

- The L24 raw full-difference result reproduces the earlier full-difference AUROC (`0.6936`) inside the same control script, so the discrepancy is not mainly from split drift or a different evaluation path.
- Full SVD coordinates remain better than raw full difference under the same split and classifier settings, especially at L20 and L32.
- A dense random orthogonal rotation is close to raw full difference, not to SVD coordinates.
- Therefore, the full-SVD-coordinate gain is not an arbitrary rotation effect. It is more consistent with SVD-axis parameterization plus per-coordinate standardization / regularized optimization changing the effective inductive bias.
- In writeups, full difference and all SVD coordinates should not be described as empirically interchangeable under this probe pipeline, even though they span the same linear space.

## 2026-04-23 Stage B Experiment Preparation

Prepared scripts:

- Condition planner: `scripts/prepare_stage_b_conditions.py`
- GPU condition hidden dump: `scripts/dump_stage_b_condition_hidden_states.py`
- CPU geometry analysis: `scripts/analyze_stage_b_geometry.py`
- Convenience wrappers:
  - `scripts/run_gpu_stage_b_conditions.sh`
  - `scripts/run_cpu_stage_b.sh`

Condition plan artifact:

- `outputs/stage_b/stage_b_condition_plan.jsonl`
- Summary: `outputs/stage_b/prepare_stage_b_conditions_summary.json`

Default pilot setup:

- Layers: L20 / L24 / L32
- Outcomes: FP / TN
- Samples: 512 total, balanced as 256 FP and 256 TN
- Conditions:
  - matched image
  - random unrelated mismatched image
  - adversarial mismatched image
  - blind / no image

Condition-plan validation:

- All 512 planned rows have an adversarial mismatch.
- For this FP/TN pilot, all source samples have label `no`; adversarial mismatches are same-question, opposite-label images with label `yes`.
- This is a clean setup for separating “some image exists” from “the image provides correct evidence for this question.”

Planned Stage B analysis views:

- Top-backbone scores at K=4 / 64 / 256
- Residual / tail scores for bands including 257-1024, 65-128, and 129-256
- Fixed supervised decision scores using logistic and LDA/Fisher directions from the matched FP-vs-TN reference data

Status:

- Completed. See the Stage B analysis section below.

## 2026-04-24 Extended TODO Stage I/J Preparation

Prepared scripts:

- Protocol lock: `scripts/prepare_stage_i_protocol.py`
- Destructive controls: `scripts/analyze_stage_j_controls.py`
- Convenience wrappers:
  - `scripts/run_cpu_stage_i.sh`
  - `scripts/run_cpu_stage_j_controls.sh`

Stage I artifacts:

- Fixed split IDs:
  - `outputs/splits/pope_train_ids.json`
  - `outputs/splits/pope_val_ids.json`
  - `outputs/splits/pope_test_ids.json`
- Split summary: `outputs/splits/split_summary.csv`
- Protocol notes: `notes/protocol_lock.md`
- Prompt notes: `notes/prompt_templates.md`
- Prompt diff: `outputs/sanity_checks/prompt_template_diff.txt`
- Hidden readout notes: `notes/hidden_readout_protocol.md`

Default split:

- Seed: 42
- Stratification: subset / label / outcome
- Train / val / test: 6300 / 1350 / 1350

Stage J prepared analysis:

- Destructive matrix controls: real matched, image-shuffled, blind-shuffled, Gaussian mean/variance matched.
- Probe controls: FP-vs-TN probes for full difference, full SVD coordinates, and top-K SVD coordinates.
- Label control: shuffled FP/TN labels on the real matched difference.
- Random subspace controls: random orthogonal K-subspaces, random SVD bands, PCA on `z_img`, and PCA on `z_blind`.
- Output targets:
  - `outputs/stage_j_controls/shuffle_spectrum_summary.csv`
  - `outputs/stage_j_controls/shuffle_probe_summary.csv`
  - `outputs/stage_j_controls/shuffle_stability_summary.csv`
  - `outputs/stage_j_controls/random_subspace_control.csv`

Smoke validation:

- Smoke inputs: `outputs/predictions/stage_j_smoke_predictions.jsonl`, `outputs/hidden_states_stage_j_smoke/`
- Smoke Stage J outputs: `outputs/stage_j_smoke/`
- Smoke plots: `outputs/plots_stage_j_smoke/`

## 2026-04-25 Stage J Destructive Controls

Artifacts:

- Spectrum controls: `outputs/stage_j_controls/shuffle_spectrum_summary.csv`
- Probe controls: `outputs/stage_j_controls/shuffle_probe_summary.csv`
- Stability controls: `outputs/stage_j_controls/shuffle_stability_summary.csv`
- Random subspace controls: `outputs/stage_j_controls/random_subspace_control.csv`
- Plots:
  - `outputs/plots/stage_j_real_vs_shuffle_spectrum.png`
  - `outputs/plots/stage_j_real_vs_shuffle_auroc.png`
  - `outputs/plots/stage_j_random_subspace_boxplot.png`

Protocol:

- Layers: L20 / L24 / L32
- Split-locked setup: SVD/spectrum/stability estimated on `outputs/splits/pope_train_ids.json`; FP-vs-TN probes trained on train IDs and evaluated on `outputs/splits/pope_test_ids.json`.
- Train/test FP-vs-TN probe sizes: 3150 / 675.

Spectrum controls:

- Real matched differences are sharply low-rank compared with Gaussian mean/variance-matched controls.
- However, image-shuffled and blind-shuffled differences have almost the same singular spectrum as real matched differences.
- Effective rank for real vs image-shuffled / blind-shuffled:
  - L20: 639.6 vs 646.0 / 642.5
  - L24: 616.4 vs 615.9 / 619.0
  - L32: 703.4 vs 704.2 / 705.7
- Explained variance at K=4 is also nearly unchanged by image/blind shuffle:
  - L20: real 0.8406, image-shuffle 0.8427, blind-shuffle 0.8402
  - L24: real 0.8762, image-shuffle 0.8771, blind-shuffle 0.8757
  - L32: real 0.7254, image-shuffle 0.7282, blind-shuffle 0.7249
- Interpretation: the dominant low-rank backbone is not sufficient evidence of sample-paired visual grounding. It likely includes broad modality/prompt geometry that survives destructive pairing.

Probe controls:

| Layer | Real full SVD | Image shuffle | Blind shuffle | Gaussian | Label shuffle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.6064 | 0.5256 | 0.6524 | 0.5702 | 0.5079 |
| 24 | 0.6837 | 0.5150 | 0.6168 | 0.4995 | 0.4977 |
| 32 | 0.6766 | 0.5015 | 0.6425 | 0.4764 | 0.4487 |

- Label-shuffled probes collapse close to chance, supporting that the FP/TN signal is label-linked rather than a pure classifier artifact.
- Image-shuffled and Gaussian controls are much weaker than real matched at L24/L32.
- Blind-shuffled controls remain nontrivial, especially L20 and L32, so the current destructive control does not support a clean “only correctly paired image evidence matters” claim.

Random subspace controls:

- Plain SVD top-K does not consistently dominate matched random orthogonal K-subspaces.
- For K=256, AUROC is:
  - L20: SVD 0.6094 vs random orthogonal mean 0.6160
  - L24: SVD 0.6290 vs random orthogonal mean 0.6038
  - L32: SVD 0.5891 vs random orthogonal mean 0.5920
- PCA on `z_blind` is surprisingly strong across layers and K, often stronger than plain SVD top-K.
- Interpretation: FP/TN information is distributed and partly present in blind/text-side representations. Plain SVD top-K should not be framed as uniquely capturing the discriminative geometry.

Current Stage J conclusion:

- Strongly supported: real blind-image differences are non-Gaussian, stable, and contain FP/TN signal.
- Not supported: real matched differences have a uniquely sharper spectrum than image/blind-shuffled pairings.
- Partly supported: real matched geometry improves over image-shuffled/Gaussian controls for L24/L32 FP-vs-TN probing.
- Required wording adjustment: use “paired correction geometry contains hallucination-relevant residual information” rather than “the low-rank backbone itself proves paired visual grounding.”

## 2026-04-25 Stage K Preparation

Prepared code:

- Added readout support in `src/vgs/llava_hf.py`:
  - `last_prompt_token`
  - `first_answer_prefill` (implemented as the causal prefill state at the last prompt token)
  - `last_4_prompt_mean`
  - `last_8_prompt_mean`
- GPU dump wrapper: `scripts/run_gpu_stage_k_positions.sh`
- CPU analysis script: `scripts/analyze_stage_k_positions.py`
- CPU wrapper after GPU dump: `scripts/run_cpu_stage_k_positions.sh`

Planned GPU command:

```bash
PYTHON_BIN=/data/lh/.conda/envs/after/bin/python scripts/run_gpu_stage_k_positions.sh
```

After GPU artifacts are available under `outputs/stage_k_hidden/{position}/`, run:

```bash
PYTHON_BIN=/data/lh/.conda/envs/after/bin/python scripts/run_cpu_stage_k_positions.sh
```

Expected Stage K outputs:

- `outputs/stage_k_positions/position_probe_summary.csv`
- `outputs/stage_k_positions/position_spectrum_summary.csv`
- `outputs/plots/stage_k_position_layer_heatmap.png`

Additional Stage K condition-geometry scripts:

- GPU condition dump: `scripts/run_gpu_stage_k_conditions.sh`
- CPU SVD preparation for position-specific bases: `scripts/run_cpu_stage_k_svd.sh`
- CPU matched/mismatch geometry analysis: `scripts/run_cpu_stage_k_conditions.sh`
- Position-specific SVD bases have been prepared under `outputs/stage_k_svd/{position}/`.

Status:

- Program and scripts are ready.
- GPU hidden-state extraction completed by user.

## 2026-04-25 Stage K Token-Position Robustness

Artifacts:

- Hidden states: `outputs/stage_k_hidden/{last_prompt_token,first_answer_prefill,last_4_prompt_mean,last_8_prompt_mean}/layer_{16,20,24,32}.pt`
- Spectrum summary: `outputs/stage_k_positions/position_spectrum_summary.csv`
- Probe summary: `outputs/stage_k_positions/position_probe_summary.csv`
- Plot: `outputs/plots/stage_k_position_layer_heatmap.png`
- Readout decision: `notes/readout_position_decision.md`

Setup:

- Layers: L16 / L20 / L24 / L32
- K grid: 4 / 64 / 128 / 256
- Split-locked protocol: SVD estimated on train IDs; FP-vs-TN probes trained on train IDs and evaluated on test IDs.
- `first_answer_prefill` is identical to `last_prompt_token` in the current implementation because the prefill hidden state is read at the last prompt token before generation.

Full-difference AUROC:

| Position | L16 | L20 | L24 | L32 |
| --- | ---: | ---: | ---: | ---: |
| last_prompt_token | 0.6595 | 0.6588 | 0.6518 | 0.6197 |
| first_answer_prefill | 0.6595 | 0.6588 | 0.6518 | 0.6197 |
| last_4_prompt_mean | 0.6848 | 0.7075 | 0.7108 | 0.6798 |
| last_8_prompt_mean | 0.7094 | 0.7377 | 0.7319 | 0.7023 |

Best top-K projected AUROC:

| Position | Best layer | Best K | AUROC |
| --- | ---: | ---: | ---: |
| last_prompt_token | 24 | 128 | 0.6343 |
| first_answer_prefill | 24 | 128 | 0.6343 |
| last_4_prompt_mean | 24 | 128 | 0.6947 |
| last_8_prompt_mean | 32 | 256 | 0.6791 |

Spectrum observations:

- `last_prompt_token` keeps the previous concentration pattern: top-4 explained variance is high in L16/L20/L24 and lower in L32.
- `last_4_prompt_mean` remains concentrated but shifts the best discriminative layer toward L24.
- `last_8_prompt_mean` is extremely concentrated in early/mid layers:
  - L16 K=4 explained variance: 0.9992
  - L20 K=4 explained variance: 0.9986
  - L24 K=4 explained variance: 0.9969
- Despite this near-total variance concentration, top-K AUROC at K=4 remains weak or unstable, while AUROC improves at K=128/256. This strongly preserves the “variance geometry is not the same as hallucination-discriminative geometry” conclusion.

Current interpretation:

- Token-position robustness is supported for the core mechanism claim: hallucination signal persists across multiple readouts.
- Pooled prompt readouts are stronger for FP-vs-TN detection than the single last-token readout.
- L20/L24 remain important under pooled readouts; L24 is best for `last_4_prompt_mean`, while L20/L24 full-difference are strongest under `last_8_prompt_mean`.
- For continuity with completed Stage A/B/C/E, keep `last_prompt_token` as the primary paper readout for now and use `last_4_prompt_mean` as a robustness appendix readout.
- Matched-vs-mismatch condition scoring under the new readouts is complete; see the next section.

## 2026-04-25 Stage K Condition Geometry Across Readouts

Artifacts:

- Condition hidden states: `outputs/stage_k_condition_hidden/{position}/layer_{16,20,24,32}.pt`
- Condition geometry: `outputs/stage_k_condition_geometry/{position}/`
- Compact delta summary: `outputs/stage_k_condition_geometry/position_condition_delta_summary.csv`
- Position-specific SVD bases: `outputs/stage_k_svd/{position}/`
- Condition plots: `outputs/plots_stage_k_conditions/{position}/`

Setup:

- Positions: `last_prompt_token`, `first_answer_prefill`, `last_4_prompt_mean`, `last_8_prompt_mean`
- Layers: L16 / L20 / L24 / L32
- Conditions: matched / random mismatch / adversarial mismatch / blind
- Stage B condition plan: 512 balanced FP/TN rows.

Selected mean matched-minus-mismatch deltas:

| Position | Top-256 matched-random | Top-256 matched-adversarial | Tail 257-1024 matched-random | Tail 257-1024 matched-adversarial | Logistic matched-random | Logistic matched-adversarial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| last_prompt_token | -356.8 | -120.8 | 9.5 | 20.3 | 0.226 | 0.252 |
| first_answer_prefill | -356.8 | -120.8 | 9.5 | 20.3 | 0.226 | 0.252 |
| last_4_prompt_mean | 111.6 | 128.4 | 0.3 | 4.2 | 0.134 | 0.143 |
| last_8_prompt_mean | 405.2 | 202.4 | -1.6 | 1.7 | 0.125 | 0.142 |

Observations:

- `first_answer_prefill` again exactly matches `last_prompt_token`.
- The supervised logistic condition score separates matched from both random and adversarial mismatches for every readout.
- The separation is largest for `last_prompt_token` / `first_answer_prefill`, smaller but still positive for `last_4_prompt_mean` and `last_8_prompt_mean`.
- Tail band 257-1024 condition separation is strongest under `last_prompt_token`, especially L24/L32:
  - L24 matched-adversarial mean delta: 25.69
  - L32 matched-adversarial mean delta: 39.09
- Pooled readouts change the top-backbone condition geometry substantially. Under `last_4_prompt_mean` and `last_8_prompt_mean`, top-256 matched-minus-mismatch deltas become mostly positive, while under `last_prompt_token` they are often negative in early/mid layers.

Interpretation:

- Token-position robustness is supported for evidence-sensitive condition geometry: matched-vs-mismatch separation persists in supervised decision scores across readouts.
- The residual/tail condition geometry appears most clearly in the single-token readout, while pooled readouts improve FP/TN detection but dampen or reshape tail condition gaps.
- This suggests two related but distinct readout regimes:
  - pooled prompt readouts are stronger for detection;
  - last-token residual/tail coordinates are cleaner for condition-geometry interpretation.

## 2026-04-25 Stage L Evidence-Specific Subspaces

Artifacts:

- Probe results: `outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv`
- Condition gaps: `outputs/stage_l_evidence_subspace/evidence_subspace_condition_gap.csv`
- Split-half stability: `outputs/stage_l_evidence_subspace/evidence_subspace_stability.csv`
- Plot: `outputs/plots/stage_l_plain_svd_vs_evidence_specific.png`
- Naming decision: `notes/naming_decision.md`

Methods tested:

- Plain SVD on matched `D`
- Contrastive PCA: matched covariance minus mismatch covariance
- Generalized eigenspace: matched covariance vs regularized mismatch covariance
- Fisher FP/TN direction plus PCA completion
- PLS FP/TN subspace
- Matched-vs-adversarial logistic direction plus PCA completion

Probe result:

| Method | Best layer | Best K | Best AUROC | Notes |
| --- | ---: | ---: | ---: | --- |
| PLS FP/TN | 24 | 32 | 0.7196 | Best detection method |
| Fisher FP/TN | 20 | 64 | 0.6654 | Moderate, more stable than PLS |
| Plain SVD | 20 | 32 | 0.6103 | Stable but weak |
| Contrastive PCA | 20 | 64 | 0.6029 | Better for condition gaps than FP/TN detection |
| Generalized matched-vs-mismatch | 20 | 8 | 0.5798 | Weak detection in this first setup |
| Matched-vs-adversarial logistic | 24 | 64 | 0.5757 | Weak detection despite condition objective |

Mean AUROC over L20/L24/L32:

| K | Plain SVD | PLS FP/TN | Fisher FP/TN | Contrastive PCA |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5160 | 0.6224 | 0.6048 | 0.5099 |
| 8 | 0.5600 | 0.6488 | 0.6059 | 0.5308 |
| 16 | 0.5308 | 0.6875 | 0.6239 | 0.5604 |
| 32 | 0.5647 | 0.6948 | 0.6256 | 0.5482 |
| 64 | 0.5687 | 0.6949 | 0.6416 | 0.5541 |

Condition-gap result:

- Contrastive PCA produces the largest matched-vs-mismatch score gaps, especially at L32.
- Example L32 contrastive PCA:
  - K=64 matched-minus-random mean delta: 1030.3
  - K=64 matched-minus-adversarial mean delta: 1014.4
- This is much stronger as a condition-separation view than as an FP/TN probe.

Stability:

- Plain SVD remains the most stable split-half subspace:
  - L20 K=4: 0.9947
  - L24 K=4: 0.9945
  - L32 K=4: 0.9955
- Fisher and matched-vs-adversarial logistic are moderately stable.
- PLS is best for detection but has weaker split-half subspace stability, often around 0.4-0.5.

Interpretation:

- Stage L supports an important split:
  - **supervised PLS** is the strongest compact hallucination-detection subspace;
  - **contrastive PCA** is the strongest evidence-condition separation subspace;
  - **plain SVD** is the most stable dominant correction backbone but not the best discriminative object.
- This strengthens the paper's layered-geometry framing: there is not a single best subspace for every role.
- Recommended naming after Stage L: **evidence-sensitive correction geometry**, not **visual grounding subspace**.

## 2026-04-25 Stage M Memory Bank Preparation

Artifacts:

- Memory bank: `outputs/stage_m_local_rescue/memory_bank_train.pt`
- Retrieval audit: `outputs/stage_m_local_rescue/retrieval_audit.csv`
- Object counts: `outputs/stage_m_local_rescue/memory_bank_object_counts.csv`
- Build summary: `outputs/stage_m_local_rescue/build_stage_m_memory_bank_summary.json`

Setup:

- Layers: L20 / L24 / L32
- Split: train only
- Entries per layer: 6300
- Correction vector dim: 4096
- SVD coordinate dim: 1024
- Tail coordinate band: 257-1024, dim 768

Audit:

| Layer | Entries | Unique objects | Unique images | FP | TN | TP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 6300 | 79 | 500 | 244 | 2906 | 2523 | 627 |
| 24 | 6300 | 79 | 500 | 244 | 2906 | 2523 | 627 |
| 32 | 6300 | 79 | 500 | 244 | 2906 | 2523 | 627 |

Notes:

- The memory bank contains no validation/test samples.
- It stores fields needed for same-object retrieval, SVD-coordinate kNN, tail-coordinate kNN, and nearest-TN retrieval.
- Yes/no margins are not available in the current POPE prediction artifact; margin-based gating requires a separate logits/margin dump.

Retrieval plan:

- Plan CSV: `outputs/stage_m_local_rescue/retrieval_plan.csv`
- Plan JSONL: `outputs/stage_m_local_rescue/retrieval_plan.jsonl`
- Plan audit: `outputs/stage_m_local_rescue/retrieval_plan_audit.csv`
- Target split: test
- Targets per layer: 181 total = 53 FP + 64 TN + 64 TP
- Retrieval modes:
  - `same_object_tn`
  - `svd_knn_tn`
  - `tail_knn_tn`
  - `random_tn`
  - `same_object_fp`
- Same-image candidates are excluded by default. The final retrieval plan has zero same-image retrieved examples for all layers and modes.

## 2026-04-25 Stage M Gated Local Rescue Preparation

Prepared scripts:

- GPU runner: `scripts/run_gpu_stage_m_local_rescue.sh`
- Stage M local rescue entry point: `scripts/run_stage_m_local_rescue.py`
- CPU analysis runner: `scripts/run_cpu_stage_m_local_rescue.sh`
- Stage M analysis entry point: `scripts/analyze_stage_m_local_rescue.py`

Default GPU pilot:

- Layer: L32
- Target outcomes: FP / TN / TP
- Max targets per outcome: 32
- Alpha grid: 2 / 4 / 8
- Granularity: `last_token`
- Gates:
  - `always`
  - `low_abs_margin`
  - `high_fp_risk`
  - `margin_and_fp_risk`
- Retrieval modes:
  - `same_object_tn`
  - `svd_knn_tn`
  - `tail_knn_tn`
  - `random_tn`

Implemented steering directions:

- global train-TN mean correction
- same-object train-TN mean correction
- SVD-kNN train-TN mean correction
- tail-kNN train-TN mean correction
- random train-TN retrieval control
- local TN-minus-same-object-FP correction where same-object FP neighbors are available

Implemented analysis metrics:

- FP rescue rate
- yes/no margin shift and `logit(No)-logit(Yes)` gain
- TN and TP damage rate
- unknown / malformed output rate
- paired correctness changes with McNemar exact test
- plots:
  - `outputs/plots/stage_m_fp_margin_shift.png`
  - `outputs/plots/stage_m_gate_tradeoff_curve.png`

Dry-run:

- `scripts/run_stage_m_local_rescue.py --dry-run`
- Summary: `outputs/stage_m_local_rescue/run_stage_m_local_rescue_summary.json`

Notes:

- The default wrapper runs full first-token generation. Set `LOGITS_ONLY=1` for a cheaper margin-only GPU run.
- The evidence-specific Stage L subspace direction is not yet wired into M2 steering because Stage L currently saves evaluation summaries, not reusable intervention bases.

## 2026-04-25 Stage M Gated Local Rescue Result

Artifacts:

- Results: `outputs/stage_m_local_rescue/local_rescue_results.csv`
- Summary: `outputs/stage_m_local_rescue/local_rescue_summary.csv`
- Run summary: `outputs/stage_m_local_rescue/run_stage_m_local_rescue_summary.json`
- Analysis summary: `outputs/stage_m_local_rescue/analyze_stage_m_local_rescue_summary.json`
- Plots:
  - `outputs/plots/stage_m_fp_margin_shift.png`
  - `outputs/plots/stage_m_fp_margin_shift_layer_32.png`
  - `outputs/plots/stage_m_gate_tradeoff_curve.png`

Setup:

- Layer: L32
- Target samples: 32 FP / 32 TN / 32 TP
- Alpha grid: 2 / 4 / 8
- Granularity: `last_token`
- Gates: `always`, `low_abs_margin`, `high_fp_risk`, `margin_and_fp_risk`
- Retrieval modes in this run: `same_object_tn`, `svd_knn_tn`, `tail_knn_tn`, `random_tn`

Baseline first-token margins:

| Outcome | n | Mean yes-no margin | Median yes-no margin | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP | 32 | 0.7324 | 0.6328 | 0.0156 | 2.7344 |
| TN | 32 | -2.0103 | -2.2422 | -3.1406 | -0.2812 |
| TP | 32 | 2.3003 | 2.6484 | 0.0781 | 4.5625 |

At alpha 8:

| Gate | Direction | FP n | FP rescue rate | Median gain in logit(No)-logit(Yes) |
| --- | --- | ---: | ---: | ---: |
| always | global TN mean | 32 | 0.0625 | 0.0781 |
| always | same-object TN mean | 32 | 0.0625 | 0.0781 |
| always | SVD-kNN TN mean | 32 | 0.0625 | 0.0625 |
| always | tail-kNN TN mean | 32 | 0.0625 | 0.0469 |
| always | random TN mean | 32 | 0.0625 | 0.0781 |
| low_abs_margin | global TN mean | 9 | 0.2222 | 0.0781 |
| low_abs_margin | same-object TN mean | 9 | 0.2222 | 0.0781 |
| low_abs_margin | SVD-kNN TN mean | 9 | 0.2222 | 0.0625 |
| low_abs_margin | tail-kNN TN mean | 9 | 0.2222 | 0.0469 |
| low_abs_margin | random TN mean | 9 | 0.2222 | 0.0781 |
| margin_and_fp_risk | global TN mean | 4 | 0.2500 | 0.0781 |
| margin_and_fp_risk | same-object TN mean | 4 | 0.2500 | 0.0703 |
| margin_and_fp_risk | SVD-kNN TN mean | 4 | 0.2500 | 0.0625 |
| margin_and_fp_risk | tail-kNN TN mean | 4 | 0.2500 | 0.0547 |
| margin_and_fp_risk | random TN mean | 4 | 0.2500 | 0.0859 |

Damage / control observations:

- TN damage is zero for all evaluated settings.
- TP damage appears only for one very borderline TP sample at alpha 8:
  - baseline yes-minus-no margin: `0.078125`
  - damaged sample: `coco:popular:121`, object `bowl`
- Under the `always` gate, global / same-object / random TN steering damage TP at `1/32 = 0.03125`; SVD-kNN and tail-kNN cause no TP damage in this run.
- Under the `low_abs_margin` gate, global / same-object / random TN steering damage `1/3` selected TP samples; SVD-kNN and tail-kNN cause no TP damage.

Rescue behavior:

- There are 80 rescued intervention rows, but only 2 unique rescued FP samples:
  - `coco:popular:2714`, object `person`, baseline yes-minus-no margin `0.015625`
  - `coco:popular:966`, object `chair`, baseline yes-minus-no margin `0.03125`
- This confirms the earlier Stage E pattern: steering can move the first-token boundary and sometimes flip answers, but mainly when the baseline decision is already extremely close to the yes/no boundary.

Interpretation:

- Stage M2 provides positive evidence for **boundary-local first-token steerability**.
- It does not support the stronger claim that local memory-bank retrieval beats global or random TN controls.
- The random TN retrieval control is competitive and sometimes strongest, so the current effect should not be described as retrieval-specific rescue.
- The right writeup framing is cautious: local / global TN-like correction directions can nudge borderline hallucinated decisions, but the current implementation has not isolated a semantically specific rescue mechanism.

Follow-up:

- The completed run did not include `same_object_fp` in `--retrieval-modes`, so the implemented local `TN - FP` direction was not evaluated. The default Stage M GPU wrapper has been updated to include `same_object_fp` for the next run.
- A useful next run is L32 only, same sample budget, with `same_object_fp` enabled to test whether local TN-minus-FP improves over global/random TN.

## 2026-04-25 Stage M Rescue Failure Analysis

Artifacts:

- Taxonomy: `outputs/stage_m_local_rescue/rescue_failure_taxonomy.csv`
- Group summary: `outputs/stage_m_local_rescue/rescue_failure_group_summary.csv`
- Notes: `notes/rescue_failure_analysis.md`
- Run summary: `outputs/stage_m_local_rescue/analyze_stage_m_rescue_failures_summary.json`

Taxonomy over 32 FP samples:

| Label | Count |
| --- | ---: |
| margin improved but answer unchanged | 30 |
| rescued to correct `No` | 2 |
| no effect | 0 |
| damaged / malformed | 0 |
| moved in wrong direction | 0 |

Margin bins:

| Baseline margin bin | Count |
| --- | ---: |
| `borderline_abs_le_0.25` | 9 |
| `medium_abs_0.25_1.0` | 16 |
| `high_abs_gt_1.0` | 7 |

Rescued samples:

- `coco:popular:2714`, object `person`, baseline yes-minus-no margin `0.015625`, best gain `0.109375`.
- `coco:popular:966`, object `chair`, baseline yes-minus-no margin `0.031250`, best gain `0.093750`.

Subset pattern:

- Both rescued samples are from `popular`.
- No sampled `adversarial` or `random` FP case was rescued in this L32 run.

Failure profile:

- Unrescued but margin-improved FP samples have median baseline margin `0.656250`.
- Rescued FP samples have median baseline margin `0.023438`.
- This strongly supports the boundary-local interpretation: the intervention often nudges the first-token margin in the right direction, but only extremely low-margin hallucinations cross the decision boundary.

Interpretation:

- Stage M3 supports the claim that first-token rescue is mainly a borderline-decision phenomenon.
- It partly supports the claim that failures are high-confidence language-prior hallucinations: high-margin FP cases remain unrescued, but many medium-margin cases also receive small positive nudges.
- It does not support the stronger claim that local memory directions are necessary for rescue, because the current L32 run still shows competitive global/random TN controls.

## 2026-04-25 Stage N External Benchmark Preparation

Decision:

- Use **AMBER discriminative** as the first external benchmark.
- Rationale recorded in `notes/external_benchmark_choice.md`.

Prepared scripts:

- AMBER plan preparation: `scripts/prepare_stage_n_amber.py`
- CPU wrapper for AMBER preparation: `scripts/run_cpu_stage_n_amber_prepare.sh`
- AMBER LLaVA yes/no evaluation: `scripts/run_stage_n_amber_eval.py`
- GPU wrapper for AMBER evaluation + hidden dump: `scripts/run_gpu_stage_n_amber.sh`
- Zero-shot POPE-SVD transfer analysis: `scripts/analyze_stage_n_external_transfer.py`
- CPU wrapper for transfer analysis: `scripts/run_cpu_stage_n_transfer.sh`

Dry-runs:

- `DRY_RUN=1 scripts/run_cpu_stage_n_amber_prepare.sh`
- `scripts/run_stage_n_amber_eval.py --dry-run`

Planned local data layout:

- Queries: `data/amber/data/query/query_discriminative.json`
- Annotations: `data/amber/data/annotations.json`
- Images: `data/amber/image/AMBER_*.jpg`

Protocol:

- Prepare AMBER rows with `id`, image path, query, truth label, and AMBER dimension.
- Run LLaVA-1.5-7B yes/no prediction on AMBER.
- Dump paired image/blind hidden states using the same `last_prompt_token` protocol as POPE.
- Apply POPE SVD bases directly to AMBER hidden differences; do not refit subspaces on AMBER.
- Report transfer scores by AMBER dimension and by feature family: full diff norm, top-SVD energy, and tail energy.

Current blocker:

- AMBER data/images are now present locally.
- The default preparation wrapper now creates a balanced pilot plan with `MAX_PER_DIMENSION_LABEL=300`.
- Current pilot plan:
  - Rows: 1500
  - Attribute: 300 yes / 300 no
  - Existence: 300 no
  - Relation: 300 yes / 300 no
  - Missing images: 0
  - Invalid labels: 0
- Use `FULL=1 scripts/run_cpu_stage_n_amber_prepare.sh` to regenerate the full 14216-row discriminative plan.

Next commands:
  - `scripts/run_cpu_stage_n_amber_prepare.sh`
  - `scripts/run_gpu_stage_n_amber.sh`
  - `scripts/run_cpu_stage_n_transfer.sh`

## 2026-04-23 Stage B Condition Geometry Analysis

Artifacts:

- Condition hidden states: `outputs/stage_b_hidden/layer_{20,24,32}.pt`
- Sample-level scores: `outputs/stage_b/stage_b_sample_scores.csv`
- Condition summary: `outputs/stage_b/stage_b_condition_score_summary.csv`
- Pairwise condition deltas: `outputs/stage_b/stage_b_pairwise_condition_deltas.csv`
- FP/TN condition summary: `outputs/stage_b/stage_b_outcome_condition_summary.csv`
- Condition subspace similarity: `outputs/stage_b/stage_b_condition_subspace_similarity.csv`
- Plots:
  - `outputs/plots/stage_b_condition_scores_layer_{20,24,32}.png`
  - `outputs/plots/stage_b_outcome_scores_layer_{20,24,32}.png`

Setup:

- Layers: L20 / L24 / L32
- Samples: 512 balanced FP/TN rows from POPE `label=no`
- Conditions: matched / random mismatch / adversarial mismatch / blind
- Blind condition is the zero anchor for `z_blind - z_cond`, so its correction scores are zero by construction.

Top-backbone observations:

- Top-backbone energy clearly separates image-conditioned states from blind, but does not cleanly separate correct evidence from mismatched evidence.
- At K=256, matched-minus-random mean deltas are:
  - L20: -587.1
  - L24: -520.8
  - L32: +268.1
- At K=256, matched-minus-adversarial mean deltas are:
  - L20: -45.5
  - L24: -200.0
  - L32: -227.9
- This suggests top-backbone energy is dominated by large image-conditioned movement rather than by evidence correctness.

Residual / tail observations:

- The `257-1024` residual/tail score is consistently higher for matched than for adversarial mismatch:
  - L20: +11.8
  - L24: +25.7
  - L32: +39.1
- It is also higher for matched than for random mismatch:
  - L20: +5.7
  - L24: +14.2
  - L32: +17.0
- TN samples show a stronger matched-specific residual/tail increase than FP samples:
  - matched-minus-random, L20: FP +0.4 vs TN +11.0
  - matched-minus-random, L24: FP +2.8 vs TN +25.6
  - matched-minus-random, L32: FP -10.2 vs TN +44.2

Supervised decision observations:

- The fixed FP-vs-TN supervised decision scores separate FP/TN mainly under matched visual evidence, not under random or adversarial mismatches.
- Logistic FP-minus-TN mean gaps:
  - L20 matched +0.623, random +0.102, adversarial +0.133
  - L24 matched +0.925, random +0.155, adversarial +0.231
  - L32 matched +0.834, random +0.101, adversarial +0.196
- LDA/Fisher shows the same pattern:
  - L20 matched +0.451, random +0.060, adversarial +0.082
  - L24 matched +0.711, random +0.092, adversarial +0.125
  - L32 matched +0.769, random +0.090, adversarial +0.143
- For FP samples, matched-minus-mismatch supervised decision deltas are positive for most samples. For example, logistic matched-minus-random is positive for 89.8% / 91.4% / 91.8% of FP samples at L20 / L24 / L32.
- TN samples do not show the same positive decision shift; their matched-minus-random logistic means are near zero or negative.

Condition subspace observations:

- K=4 condition SVD bases are all highly similar, so the very top backbone is stable across conditions.
- At K=64 and K=256, condition-specific bases diverge.
- Matched vs adversarial projection similarity at K=256:
  - L20: 0.4836
  - L24: 0.4792
  - L32: 0.4784
- Matched vs random projection similarity at K=256:
  - L20: 0.4937
  - L24: 0.4849
  - L32: 0.4768
- This supports a real condition-specific correction geometry beyond the very top singular directions.

Current interpretation:

- Stage B supports the claim that the geometry is image-condition-sensitive and not merely a static FP/TN probe artifact.
- The evidence-relevant signal is stronger in residual/tail and supervised decision views than in raw top-backbone energy.
- TN samples show stronger matched-evidence residual/tail correction, while FP samples show weaker matched-specific residual/tail correction and a strong abnormal shift along the FP decision direction.
- A careful wording is now justified: hallucination is associated not merely with the presence of image-conditioned change, but with distorted condition-specific correction geometry under matched visual evidence.

## 2026-04-23 Stage E Causal Intervention Preparation

Prepared scripts:

- E0 hook precheck: `scripts/intervention_precheck.py`
- Intervention pilot: `scripts/intervention_pilot.py`
- Convenience wrapper: `scripts/run_gpu_stage_e.sh`
- Core implementation: `src/vgs/stage_e.py`

Rationale:

- Stage A/C/B now support structure existence, discriminative mismatch, and condition sensitivity.
- The key missing evidence is causal: whether removing or steering the relevant coordinates changes model answers.
- Because top-K directions are not the main discriminative object, Stage E should not start with deleting top-4 directions.

Initial intervention families:

- **TN ablation**: on originally correct TN samples, ablate the residual/tail SVD slice `257-1024`; compare to a random same-width tail control.
- **FP decision steering**: on originally hallucinated FP samples, move the matched-image hidden state to reduce the logistic / LDA FP decision score.
- **Correction steering**: on FP samples, add a TN-like mean correction direction or counteract the FP-like mean shift.

Default pilot:

- Layer: L24 first
- Granularity: last prompt token during the initial full-prompt forward
- Alpha grid: `0.25 / 0.5 / 1.0`
- Samples: 16 FP and 16 TN by default
- Output files after GPU run:
  - `outputs/interventions/intervention_precheck.csv`
  - `outputs/interventions/intervention_pilot_results.csv`
  - `outputs/interventions/intervention_pilot_summary.csv`

Status:

- CLI dry-runs pass.
- Actual E0/E-pilot still require a CUDA run.
- Causal claims remain unvalidated until the hook precheck and pilot results succeed.

## 2026-04-23 Stage E First Pilot Result

Artifacts:

- Precheck: `outputs/interventions/intervention_precheck.csv`
- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`

Setup:

- Layer: L24
- Samples: 16 TN and 16 FP
- Alpha grid: `0.25 / 0.5 / 1.0`
- Tail band: `257-1024`
- Intervention granularity: last prompt token only

Precheck:

- No-op hook preserved generated text: baseline `Yes`, no-op `Yes`.
- Random perturbation with the first implementation did not change generated text: random perturbation also `Yes`.
- This does not prove the hook has no effect, because the check only measured final decoded text; the next precheck should inspect next-token logits directly.

Pilot outcome:

- TN baseline accuracy: 1.0; tail ablation at all tested alphas kept accuracy at 1.0 and FP rate at 0.0.
- Random tail control also kept TN accuracy at 1.0.
- FP baseline accuracy: 0.0 and FP rate: 1.0.
- Logistic/LDA decision steering, TN-correction steering, and FP-shift counter-steering all kept FP accuracy at 0.0 and FP rate at 1.0 for alpha up to 1.0.

Interpretation:

- First Stage E pilot is a null result at the decoded-answer level.
- It should not yet be interpreted as evidence against causality, because the intervention may be too weak, last-token-only may be insufficient, or the generation argmax may be insensitive.
- Stage E tooling has been updated so the precheck records next-token logit deltas, not only decoded text changes.
- Next pilot should use stronger alpha values, e.g. `1 / 2 / 4 / 8`, and decide whether to add full-sequence or answer-step hooks if logits remain insensitive.

## 2026-04-23 Stage E Stronger-Alpha Pilot Result

Artifacts:

- Precheck: `outputs/interventions/intervention_precheck.csv`
- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`

Setup:

- Layer: L24
- Samples: 16 TN and 16 FP
- Alpha grid: `1 / 2 / 4 / 8`
- Tail band: `257-1024`
- Intervention granularity: last prompt token only

Precheck:

- No-op hook preserved generated text and logits exactly: max logit delta `0.0`.
- Random perturbation changed logits but not decoded text: max logit delta `0.1641`.
- The yes-vs-no next-token margin barely changed: baseline margin `1.3594`, random margin `1.3750`.
- Interpretation: the hook is active, but final yes/no output can be insensitive to small last-token perturbations.

TN ablation:

- Baseline TN accuracy: `1.0`.
- Tail ablation `257-1024`:
  - alpha 1 / 2 / 4: TN accuracy remains `1.0`
  - alpha 8: TN accuracy drops to `0.0`, with 16 / 16 samples flipping to FP (`Yes`)
- Random same-width tail control:
  - alpha 1 / 2: TN accuracy remains `1.0`
  - alpha 4 / 8: outputs become `unknown` for 16 / 16 samples

FP steering:

- Baseline FP accuracy: `0.0`, FP rate `1.0`.
- Logistic FP-score reduction, LDA FP-score reduction, TN-correction steering, and FP-shift counter-steering all leave FP rate at `1.0` through alpha 8.
- The generated answer remains `Yes` for every FP sample and intervention in this pilot.

Interpretation:

- This is the first causal-looking signal: sufficiently strong ablation of the residual/tail slice can destroy correct TN behavior and flip it to hallucinated `Yes`.
- However, it is not yet a clean causal claim because the random control becomes invalid/unknown at high alpha rather than preserving normal answer format.
- FP rescue is not supported by the current last-token steering setup.
- Next Stage E should record yes/no logits for every pilot intervention and run a finer tail-ablation dose curve around alpha 4-8, ideally with a better norm-matched control and possibly L20/L32 replication.

Updated Stage E plan:

- The intervention pilot now records yes/no logits and yes-minus-no margin for every intervention row.
- The summary now records mean and median yes-minus-no margin, unknown rate, and mean margin shift relative to each sample's baseline.
- Added cleaner tail controls:
  - random same-width projection
  - random same-width projection orthogonalized against the `257-1024` tail slice
  - norm-matched orthogonal random projection, scaled per sample to match the target tail projection norm
- Added intervention granularities:
  - `last_token`
  - `full_sequence`
  - `generated_token`
- Recommended next run:

```bash
LAYERS="20 24" OUTCOMES="TN" FAMILIES="tail" GRANULARITIES="last_token" \
  ALPHA_GRID="4 5 6 7 8" bash scripts/run_gpu_stage_e.sh
```

- Recommended granularity pilot:

```bash
LAYERS="24" OUTCOMES="TN" FAMILIES="tail" GRANULARITIES="last_token full_sequence" \
  MAX_SAMPLES_PER_OUTCOME=4 ALPHA_GRID="4 6 8" bash scripts/run_gpu_stage_e.sh
```

## 2026-04-23 Stage E Clean Tail Dose Analysis

Artifacts:

- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`
- Post-hoc dose analysis: `outputs/interventions/stage_e_dose_curve.csv`
- Flip thresholds: `outputs/interventions/stage_e_flip_thresholds.csv`
- Flip threshold summary: `outputs/interventions/stage_e_flip_threshold_summary.csv`
- Plots:
  - `outputs/plots/stage_e_tail_ablation_dose.png`
  - `outputs/plots/stage_e_control_unknown_rate.png`

Setup:

- Layers: L20 and L24
- Outcome: TN only
- Family: tail ablation and controls only
- Tail band: `257-1024`
- Granularity: last prompt token
- Alpha grid: `4 / 5 / 6 / 7 / 8`
- Samples: 16 TN per layer

Main result:

- Tail ablation produces a clear dose-dependent movement from correct `No` toward hallucinated `Yes`.
- L24 is cleaner than L20 in this run: it flips all 16 / 16 TN samples to `Yes` at alpha 8 with no unknown outputs.
- L20 also shows a strong effect, but alpha 8 starts to induce format collapse: 9 / 16 `Yes`, 2 / 16 `No`, and 5 / 16 `unknown`.

Tail ablation dose curve:

| Layer | alpha 4 | alpha 5 | alpha 6 | alpha 7 | alpha 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L20 Yes rate | 0.0625 | 0.3750 | 0.7500 | 0.8750 | 0.5625 |
| L20 median yes-no margin | -0.5703 | -0.0938 | 0.2812 | 0.6289 | 1.6484 |
| L24 Yes rate | 0.0000 | 0.1250 | 0.5625 | 0.9375 | 1.0000 |
| L24 median yes-no margin | -0.7500 | -0.3281 | 0.0156 | 0.3906 | 0.9336 |

Flip thresholds:

- L20: 15 / 16 samples eventually flip to `Yes`; median first-Yes alpha is 6.0. First-Yes counts are alpha 4: 1, alpha 5: 5, alpha 6: 6, alpha 7: 3.
- L24: 16 / 16 samples eventually flip to `Yes`; median first-Yes alpha is 6.0. First-Yes counts are alpha 5: 2, alpha 6: 7, alpha 7: 6, alpha 8: 1.

Controls:

- Random same-width and orthogonal same-width controls mostly collapse answer format to `unknown`, so they are poor high-alpha controls.
- Norm-matched orthogonal random control is more informative:
  - L24 stays `No` for 16 / 16 samples through alpha 6, while true tail ablation already flips 9 / 16 at alpha 6.
  - L24 norm-matched control starts producing many unknowns at alpha 7/8 but still produces 0 `Yes` among all samples.
  - L20 norm-matched control is less clean: it produces 2 / 16 `Yes` at alpha 5 and many unknowns at alpha 6+.

Interpretation:

- This is stronger causal evidence than the previous alpha-8-only observation, because the yes/no margin changes continuously and the answer flips occur around alpha 5-7 rather than only at an extreme endpoint.
- The cleanest current causal-looking claim is layer-specific: removing the L24 `257-1024` residual/tail slice at the matched-image last-token state degrades originally correct TN answers in a dose-dependent way.
- The result still should be called "causal relevance" rather than a fully isolated mechanism, because high-alpha control stability remains imperfect and the intervention is last-token-only.

Next Stage E step:

- Run a small granularity pilot on L24 comparing `last_token` vs `full_sequence`, using TN tail ablation first.
- Then run FP rescue as a logit-margin experiment, not as an answer-flip experiment, with `full_sequence` and/or `generated_token` included.

## 2026-04-23 Stage E Granularity Pilot

Artifacts:

- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`
- Post-hoc dose analysis: `outputs/interventions/stage_e_dose_curve.csv`
- Flip threshold summary: `outputs/interventions/stage_e_flip_threshold_summary.csv`

Setup:

- Layer: L24
- Outcome: TN only
- Family: tail ablation and controls only
- Tail band: `257-1024`
- Granularities: `last_token` and `full_sequence`
- Alpha grid: `4 / 6 / 8`
- Samples: 4 TN

Tail ablation result:

| Granularity | alpha 4 Yes rate | alpha 6 Yes rate | alpha 8 Yes rate | alpha 8 median yes-no margin |
| --- | ---: | ---: | ---: | ---: |
| last_token | 0.00 | 0.75 | 1.00 | 0.9414 |
| full_sequence | 0.00 | 1.00 | 1.00 | 4.7402 |

Flip thresholds:

- `full_sequence`: 4 / 4 samples first flip to `Yes` at alpha 6.
- `last_token`: 3 / 4 samples first flip at alpha 6 and 1 / 4 first flips at alpha 8.

Controls and caveats:

- `full_sequence` true tail ablation is stronger than last-token ablation at the yes/no margin level.
- However, full-sequence interventions often corrupt the continuation after the first answer token; e.g. outputs start with `Yes` but then continue with nonsensical token strings.
- Full-sequence random / orthogonal controls mostly become `unknown`, so this pilot does not yet provide a clean full-sequence directional-control result.
- Last-token remains the cleaner protocol for the current causal-ablation claim, while full-sequence is promising for a first-token / logit-only follow-up.

Next recommended run:

```bash
LAYERS="24" OUTCOMES="TN" FAMILIES="tail" GRANULARITIES="last_token full_sequence" \
  MAX_SAMPLES_PER_OUTCOME=8 MAX_NEW_TOKENS=1 ALPHA_GRID="4 5 6" bash scripts/run_gpu_stage_e.sh
```

Rationale:

- `MAX_NEW_TOKENS=1` should isolate the first yes/no decision and avoid interpreting post-answer format collapse as part of the causal effect.
- Alpha `4 / 5 / 6` is the most informative range because alpha 8 is already saturated for both granularities.

## 2026-04-23 Stage E First-Token Granularity Pilot

Artifacts:

- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`
- Tagged post-hoc dose analysis: `outputs/interventions/stage_e_first_token_dose_curve.csv`
- Tagged flip thresholds: `outputs/interventions/stage_e_first_token_flip_thresholds.csv`
- Tagged flip threshold summary: `outputs/interventions/stage_e_first_token_flip_threshold_summary.csv`
- Plots:
  - `outputs/plots/stage_e_first_token_tail_ablation_dose.png`
  - `outputs/plots/stage_e_first_token_control_unknown_rate.png`

Setup:

- Layer: L24
- Outcome: TN only
- Family: tail ablation and controls only
- Tail band: `257-1024`
- Granularities: `last_token` and `full_sequence`
- Alpha grid: `4 / 5 / 6`
- Samples: 8 TN
- Generation: `max_new_tokens=1`

Tail ablation result:

| Granularity | alpha 4 Yes rate | alpha 5 Yes rate | alpha 6 Yes rate | alpha 6 median yes-no margin |
| --- | ---: | ---: | ---: | ---: |
| last_token | 0.00 | 0.25 | 0.625 | 0.0391 |
| full_sequence | 0.00 | 0.50 | 1.00 | 2.3203 |

Flip thresholds:

- `full_sequence`: 8 / 8 samples eventually flip to `Yes`; first-Yes alpha is between 5 and 6, with median 5.5.
- `last_token`: 5 / 8 samples flip to `Yes` by alpha 6; 3 / 8 remain `No` through alpha 6.

Controls:

- The best control remains norm-matched random tail at `last_token`: it stays `No` for 8 / 8 samples at alpha 4 / 5 / 6, with no unknowns.
- Norm-matched random tail at `full_sequence` never produces `Yes`, but its unknown rate rises from 0.375 to 0.875 as alpha grows from 4 to 6.
- Random same-width and orthogonal controls mostly produce `unknown` for both granularities, so they remain format-stability controls rather than good directional controls.

Interpretation:

- The stronger full-sequence effect survives the `max_new_tokens=1` test, so it is not merely a post-answer continuation artifact.
- Full-sequence tail ablation appears to affect the actual first yes/no decision more strongly than last-token-only ablation.
- However, full-sequence controls are still unstable, so the cleanest directional-control evidence remains the last-token comparison against norm-matched random tail.
- A conservative claim is now: L24 residual/tail correction coordinates have causal relevance for the first yes/no decision, and the effect is stronger when the intervention is distributed across the full prompt sequence.

Next Stage E step:

- Use `max_new_tokens=1` for FP rescue / steering and judge success by yes-no margin movement first.
- Run a small FP-only rescue pilot with `last_token full_sequence generated_token`, because answer flips are likely too strict but first-token margins may reveal partial steering.

## 2026-04-23 Stage E FP Rescue Preparation

Prepared updates:

- Added explicit rescue control: `random_rescue_control`.
- Added FP rescue post-hoc metrics to `scripts/analyze_stage_e_results.py`:
  - `outputs/interventions/{artifact_prefix}_rescue_margin_summary.csv`
  - `outputs/plots/{artifact_prefix}_rescue_margin.png`
- Added convenience wrapper: `scripts/run_gpu_stage_e_fp_rescue.sh`

Default FP rescue pilot:

```bash
bash scripts/run_gpu_stage_e_fp_rescue.sh
```

Equivalent explicit command:

```bash
LAYERS="24" OUTCOMES="FP" FAMILIES="rescue" GRANULARITIES="last_token full_sequence" \
  MAX_SAMPLES_PER_OUTCOME=16 MAX_NEW_TOKENS=1 ALPHA_GRID="1 2 4 6 8" \
  bash scripts/run_gpu_stage_e.sh
```

Important protocol note:

- Do not include `generated_token` in the default first-token rescue run. For `max_new_tokens=1`, the first decision token comes from the prompt forward pass, so `last_token` and `full_sequence` are the relevant granularities. `generated_token` is more appropriate for a later multi-token continuation intervention.

Primary success criterion:

- Treat decoded FP-to-TN flips as a strong bonus, not the required first signal.
- The first signal to look for is positive gain in `logit(No) - logit(Yes)`, equivalently negative movement in `yes_minus_no_logit`, compared with `random_rescue_control`.

Recommended post-hoc analysis after the GPU run:

```bash
/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_e_results.py \
  --artifact-prefix stage_e_fp_rescue \
  --target-prediction no
```

Interpretation rule:

- If logistic / LDA / TN-correction directions move the first-token margin toward `No` more than `random_rescue_control`, that is evidence for partial rescue even without answer flips.
- If all rescue directions are indistinguishable from random control at the margin level, the current global rescue directions should be considered too coarse, and the next method should use more local / sample-conditioned correction templates.

## 2026-04-23 Stage E FP Rescue Result

Artifacts:

- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Pilot summary: `outputs/interventions/intervention_pilot_summary.csv`
- Rescue margin summary: `outputs/interventions/stage_e_fp_rescue_rescue_margin_summary.csv`
- Rescue plot: `outputs/plots/stage_e_fp_rescue_rescue_margin.png`
- Rescue flip thresholds:
  - `outputs/interventions/stage_e_fp_rescue_flip_threshold_summary.csv`
  - `outputs/interventions/stage_e_fp_rescue_flip_thresholds.csv`

Setup:

- Layer: L24
- Outcome: FP only
- Family: rescue only
- Granularities: `last_token` and `full_sequence`
- Alpha grid: `1 / 2 / 4 / 6 / 8`
- Samples: 16 FP
- Generation: `max_new_tokens=1`

Main result:

- No rescue direction produces a robust improvement at the decoded-answer level.
- Logistic and LDA supervised directions are consistently counterproductive: they increase `yes_minus_no_logit`, meaning they push the first-token decision further toward `Yes`.
- `add_tn_correction` is the only direction with a weak beneficial effect, but it is very small and only flips 1 / 16 FP samples to `No` at alpha 8.

Representative margin movement:

| Direction | Granularity | alpha 8 median gain in `logit(No)-logit(Yes)` |
| --- | --- | ---: |
| `reduce_logistic_fp_score` | last_token | -0.0625 |
| `reduce_logistic_fp_score` | full_sequence | -0.0625 |
| `reduce_lda_fp_score` | last_token | -0.0312 |
| `reduce_lda_fp_score` | full_sequence | -0.0469 |
| `subtract_fp_shift` | last_token | -0.0391 |
| `subtract_fp_shift` | full_sequence | -0.0469 |
| `random_rescue_control` | last_token | 0.0156 |
| `random_rescue_control` | full_sequence | 0.0156 |
| `add_tn_correction` | last_token | 0.0312 |
| `add_tn_correction` | full_sequence | 0.0234 |

Decoded-answer effect:

- `reduce_logistic_fp_score`, `reduce_lda_fp_score`, and `subtract_fp_shift` keep all 16 samples as `Yes` for all tested alphas.
- `random_rescue_control` also keeps all 16 samples as `Yes`.
- `add_tn_correction` flips exactly one sample (`coco:popular:336`) to `No` at alpha 8 for both `last_token` and `full_sequence`.

Interpretation:

- The global supervised FP-vs-TN probe directions are not usable as direct rescue directions under the current intervention protocol.
- More strongly: they are not merely ineffective; they move the generation margin in the wrong direction relative to the intended rescue objective.
- This suggests a mismatch between the discriminative probe geometry and the manipulable generation-control geometry.
- `add_tn_correction` is mildly promising, but its effect is only slightly stronger than random control, so it is not yet persuasive rescue evidence.

Next step:

- Add sign-reversed supervised directions as a diagnostic control, because the current nominal “reduce FP score” directions appear anti-aligned with actual rescue.
- After that, move to more local rescue templates:
  - same-question or same-cluster TN correction
  - matched-minus-random local correction directions
  - outcome-conditioned or sample-conditioned templates rather than one global mean/probe vector

## 2026-04-23 Stage E Reverse-Direction Rescue Result

Artifacts:

- Pilot results: `outputs/interventions/intervention_pilot_results.csv`
- Rescue margin summary: `outputs/interventions/stage_e_fp_rescue_reverse_rescue_margin_summary.csv`
- Rescue plot: `outputs/plots/stage_e_fp_rescue_reverse_rescue_margin.png`
- Rescue flip thresholds:
  - `outputs/interventions/stage_e_fp_rescue_reverse_flip_threshold_summary.csv`
  - `outputs/interventions/stage_e_fp_rescue_reverse_flip_thresholds.csv`

Setup:

- Layer: L24
- Outcome: FP only
- Granularities: `last_token` and `full_sequence`
- Alpha grid: `1 / 2 / 4 / 6 / 8`
- Generation: `max_new_tokens=1`
- Rescue families include both nominal and sign-reversed supervised directions

Main result:

- `reverse_logistic_fp_direction` is the strongest supervised rescue direction so far.
- It consistently reduces `yes_minus_no_logit`, meaning it moves the first-token decision toward `No`.
- At alpha 8, it flips 1 / 16 FP samples to `No` for both `last_token` and `full_sequence`.
- `reverse_lda_fp_direction` also improves the margin, but weaker than reverse logistic and without any decoded flips.

Representative margin movement at alpha 8:

| Direction | Granularity | median gain in `logit(No)-logit(Yes)` |
| --- | --- | ---: |
| `reverse_logistic_fp_direction` | last_token | 0.0625 |
| `reverse_logistic_fp_direction` | full_sequence | 0.0625 |
| `reverse_lda_fp_direction` | last_token | 0.0312 |
| `reverse_lda_fp_direction` | full_sequence | 0.0469 |
| `add_tn_correction` | last_token | 0.0312 |
| `add_tn_correction` | full_sequence | 0.0234 |
| `random_rescue_control` | last_token | 0.0156 |
| `random_rescue_control` | full_sequence | 0.0156 |

Decoded-answer effect:

- The only sample that flips to `No` is still `coco:popular:336`.
- It flips under:
  - `add_tn_correction` at alpha 8
  - `reverse_logistic_fp_direction` at alpha 8
- No other rescue direction produces decoded `No`.

Interpretation:

- The nominal logistic / LDA rescue directions were indeed sign-misaligned with the intervention objective.
- However, sign reversal only partially fixes the problem: reverse logistic now moves the margin in the right direction, but the effect remains small and highly localized.
- This suggests the supervised discriminative subspace contains usable directional information, but a single global vector is still too coarse to robustly rescue hallucinated samples.
- The next rescue step should therefore move from global vectors to more local templates rather than only tuning alpha further.

Recommended next rescue direction:

- same-question or same-cluster TN correction vectors
- matched-minus-random local correction templates
- layer-specific and sample-conditioned steering rather than one global probe vector

## 2026-04-23 Stage E Local Rescue Preparation

Prepared local rescue directions:

- `local_knn_tn_correction`
  - sample-conditioned TN-neighborhood correction direction in layer-specific blind-image difference space
- `question_tn_correction`
  - exact-question TN mean correction direction
- `object_tn_correction`
  - same-object TN mean correction direction extracted from POPE questions
- `local_matched_minus_random`
  - per-sample matched-minus-random template from Stage B condition hidden states
- `local_matched_minus_adversarial`
  - per-sample matched-minus-adversarial template from Stage B condition hidden states

Implementation notes:

- These local directions are integrated directly into `src/vgs/stage_e.py`, so the existing FP rescue wrapper now evaluates them automatically.
- `scripts/intervention_pilot.py` now also records the Stage B condition artifact paths used for local templates.

Coverage check for the current default FP rescue pilot (`layer=24`, 16 FP samples, seed 42):

- All 16 selected FP samples overlap with the existing Stage B condition hidden dump, so both local matched-minus-random and matched-minus-adversarial templates are available.
- All 16 selected FP samples also have exact-question TN support.
- All 16 selected FP samples have same-object TN support.

This means the next local rescue pilot is a fully populated test rather than a sparse fallback experiment.

Recommended next run:

```bash
bash scripts/run_gpu_stage_e_fp_rescue.sh
```

Recommended post-hoc analysis:

```bash
/data/lh/.conda/envs/after/bin/python scripts/analyze_stage_e_results.py \
  --artifact-prefix stage_e_fp_rescue_local \
  --target-prediction no
```

Primary interpretation target:

- Check whether any local direction improves `logit(No)-logit(Yes)` more than:
  - `reverse_logistic_fp_direction`
  - `add_tn_correction`
  - `random_rescue_control`

If a local direction consistently beats those baselines, that would support the hypothesis that rescue needs sample-conditioned geometry rather than one global steering vector.

## 2026-04-23 Stage E Local Rescue Result

Two local matched-template directions were successfully evaluated in the latest FP rescue pilot:

- `local_matched_minus_random`
- `local_matched_minus_adversarial`

Result:

- Neither direction beats `reverse_logistic_fp_direction`.
- Neither direction beats `add_tn_correction`.
- At alpha 8, both local matched-template directions are roughly at the random-control level:
  - `local_matched_minus_random`
    - `full_sequence`: median gain in `logit(No)-logit(Yes)` = `+0.0156`
    - `last_token`: median gain = `0.0000`
  - `local_matched_minus_adversarial`
    - `full_sequence`: median gain in `logit(No)-logit(Yes)` = `+0.0156`
    - `last_token`: median gain = `0.0000`
- Neither local matched-template direction flips any FP sample to decoded `No`.

Current strongest rescue direction therefore remains:

- `reverse_logistic_fp_direction`
  - alpha 8 median gain:
    - `last_token`: `+0.0625`
    - `full_sequence`: `+0.0625`
  - decoded `No` flips: `1/16`

Important methods note:

- The same local-rescue run did **not** actually test `local_knn_tn_correction`, `question_tn_correction`, or `object_tn_correction`.
- Root cause: those sample-conditioned vectors were keyed with the wrong hidden-state sample-id index, so they were silently skipped when the pilot tried to resolve payloads for the selected FP samples.
- This indexing bug has now been fixed in `src/vgs/stage_e.py`.
- Post-fix verification shows all 16 selected FP pilot samples now correctly overlap with:
  - `local_knn_tn_correction`
  - `question_tn_correction`
  - `object_tn_correction`

Interpretation:

- The local matched correction templates alone do not rescue hallucinations better than the best global reverse-logistic direction.
- We therefore do **not** yet have evidence that sample-conditioned rescue is stronger.
- But we also cannot conclude that all local rescue directions fail, because the TN-neighbor / question / object-conditioned directions need one clean rerun after the indexing fix.

## 2026-04-23 Stage E Local Rescue Rerun

After fixing the sample-id indexing bug, the previously skipped sample-conditioned TN-based rescue directions were rerun successfully:

- `local_knn_tn_correction`
- `question_tn_correction`
- `object_tn_correction`

Main result:

- All three now show real, non-random rescue signal.
- However, the strongest overall rescue direction is **still** `reverse_logistic_fp_direction`.

At alpha 8, median gain in `logit(No)-logit(Yes)`:

| Direction | last_token | full_sequence |
| --- | ---: | ---: |
| `reverse_logistic_fp_direction` | `+0.0625` | `+0.0625` |
| `local_knn_tn_correction` | `+0.0391` | `+0.0391` |
| `question_tn_correction` | `+0.0312` | `+0.0312` |
| `object_tn_correction` | `+0.0312` | `+0.0312` |
| `add_tn_correction` | `+0.0312` | `+0.0234` |
| `random_rescue_control` | `+0.0156` | `+0.0156` |

Decoded-answer effect:

- `question_tn_correction` flips one FP sample to `No` at alpha `6.0`
- `object_tn_correction` flips one FP sample to `No` at alpha `6.0`
- `local_knn_tn_correction` flips one FP sample to `No` at alpha `8.0`
- the flipped sample is still the same one: `coco:popular:336`

This means:

- `question_tn_correction` and `object_tn_correction` are the first local directions that beat the previous decoded flip threshold:
  - they succeed at alpha `6`
  - whereas `reverse_logistic_fp_direction`, `add_tn_correction`, and `local_knn_tn_correction` need alpha `8`
- but they do **not** beat reverse logistic on aggregate margin gain

Interpretation:

- The rerun upgrades the local-rescue story from “not yet tested cleanly” to “tested, weakly positive”.
- There is now some evidence that **sample-conditioned TN-like directions contain useful causal signal**.
- But the effect remains narrow:
  - still only `1/16` decoded flips
  - still only one unusually borderline sample is rescued
- So the best honest statement is:
  - local conditioning helps a little
  - but it does not yet outperform the best global reverse-logistic direction at the distribution level

## 2026-04-23 Stage E Multi-Layer Rescue Sweep

FP rescue was replicated at `L20`, `L24`, and `L32` with the same pilot protocol:

- `16` FP samples
- `max_new_tokens=1`
- alpha grid `1, 2, 4, 6, 8`
- `last_token` and `full_sequence`

### Main comparison

The layers now separate into three distinct regimes:

1. `L20`: early rescue, but less clean control behavior
2. `L24`: most consistent with the earlier rescue story
3. `L32`: strongest local-rescue margin gains with clean controls

At alpha `8`, median gain in `logit(No)-logit(Yes)`:

| Layer | Best direction(s) | last_token | full_sequence | Control note |
| --- | --- | ---: | ---: | --- |
| `L20` | `reverse_logistic_fp_direction`, `local_knn_tn_correction`, `random_rescue_control` | `+0.0469` | `+0.0469` (`reverse_logistic`) | `random_rescue_control` also flips `1/16` at `last_token`, so this layer is not clean |
| `L24` | `reverse_logistic_fp_direction` | `+0.0625` | `+0.0625` | clean random control |
| `L32` | `question_tn_correction`, `object_tn_correction` | `+0.0703` | `+0.0703` | clean random control |

### Decoded-answer thresholds

- `L20`
  - `question_tn_correction` / `object_tn_correction` flip at alpha `4`
  - but `random_rescue_control` also flips at alpha `6` for `last_token`
- `L24`
  - `question_tn_correction` / `object_tn_correction` flip at alpha `6`
  - `reverse_logistic_fp_direction` flips at alpha `8`
  - random control never flips
- `L32`
  - `question_tn_correction` / `object_tn_correction` flip at alpha `6`
  - `add_tn_correction` also flips at alpha `6`
  - `reverse_logistic_fp_direction` flips at alpha `8`
  - random control never flips

As before, the flipped sample is still the same borderline case:

- `coco:popular:336`

### Interpretation

- `L20` is no longer the best next expansion target, because its rescue effect is partially undermined by a control flip.
- `L24` remains a solid baseline layer:
  - rescue is real
  - controls are clean
  - but the strongest aggregate direction is still the global reverse-logistic one
- `L32` is the most interesting new result:
  - `question_tn_correction` and `object_tn_correction` now outperform `reverse_logistic_fp_direction` on aggregate margin gain
  - the random control stays clean
  - local TN-conditioned rescue therefore looks strongest at `L32`

Current best next-step choice:

- expand sample count first on `L32`
- keep `L24` as a stability reference layer

This is a stronger result than the single-layer pilots because it suggests:

- the best rescue layer is **not** the same as the best discriminative layer from earlier Stage C summaries
- and the most effective rescue directions at that layer are **local TN-conditioned** rather than the global supervised probe direction

## 2026-04-23 Stage E Expanded-Sample Rescue Check

Rescue was then expanded to:

- `L32` and `L24`
- `32` FP samples per layer
- same first-token protocol and alpha grid

### Main result

The earlier multi-layer pattern survives the sample expansion.

At alpha `8`, median gain in `logit(No)-logit(Yes)`:

| Layer | Best direction(s) | last_token | full_sequence | Random control |
| --- | --- | ---: | ---: | ---: |
| `L24` | `reverse_logistic_fp_direction` | `+0.0625` | `+0.0625` | `+0.0156` |
| `L32` | `question_tn_correction`, `object_tn_correction` | `+0.0781` | `+0.0781` | `-0.0156` |

This means:

- `L24` still prefers the global reverse-logistic rescue direction.
- `L32` still prefers local TN-conditioned rescue directions.
- The `L32` advantage is actually clearer after expansion because the random control remains clean and even slightly moves in the wrong direction.

### Decoded-answer rescue count

Both layers now rescue `2/32` FP samples rather than `1/16`:

- `coco:popular:336`
- `coco:popular:1348`

Both rescued samples are still very borderline at baseline:

- baseline `logit(Yes)-logit(No)` = `0.0625`

Threshold comparison:

- `L24`
  - `question_tn_correction` / `object_tn_correction` rescue the two samples at alpha `6` and `8`
  - `reverse_logistic_fp_direction` also rescues two samples, but its aggregate margin remains strongest
- `L32`
  - `question_tn_correction` / `object_tn_correction` rescue both samples at alpha `6`
  - `add_tn_correction` rescues them at alpha `4` and `6`
  - `reverse_logistic_fp_direction` needs alpha `8`

### Interpretation

- The layer split is now much more credible:
  - `L24`: best aggregate rescue remains a global supervised direction
  - `L32`: best aggregate rescue remains local TN-conditioned correction geometry
- The result is still modest in absolute decoded-answer terms:
  - only `2/32` rescues
  - both are borderline FP samples from the `popular` subset
- But this is now more than a one-off anecdote:
  - the same qualitative layer pattern survived doubling the sample count
  - and `L32` local rescue remains stronger than `reverse_logistic_fp_direction` while control stays clean

Current strongest honest claim:

- rescue remains weak overall
- but there is now repeatable evidence that **later-layer (`L32`) local TN-conditioned directions are more causally useful than the global supervised rescue direction**

## 2026-04-24 Stage E Larger-Sample Rescue Check

Rescue was then expanded again to:

- `L32` and `L24`
- `64` FP samples per layer
- same first-token intervention protocol

### Main result

The pattern from the `32`-sample run remains intact.

At alpha `8`, median gain in `logit(No)-logit(Yes)`:

| Layer | Best direction(s) | last_token | full_sequence | Random control |
| --- | --- | ---: | ---: | ---: |
| `L24` | `reverse_logistic_fp_direction` | `+0.0625` | `+0.0625` | `+0.0156` |
| `L32` | `question_tn_correction`, `object_tn_correction` | `+0.0781` | `+0.0781` | `-0.0156` |

So the layer split is now replicated across three sample scales:

- `L24`: best aggregate rescue is still the global reverse-logistic direction
- `L32`: best aggregate rescue is still local TN-conditioned rescue

### Decoded-answer rescue count

The decoded rescue count grows from `2/32` to `3/64`.

Rescued samples:

- `coco:popular:336`
- `coco:popular:1348`
- `coco:popular:162`

Shared properties:

- all three are from the `popular` subset
- all three are borderline at baseline with `logit(Yes)-logit(No) = 0.0625`

This strongly suggests the current steering directions are moving the decision boundary in the right direction, but mostly affecting samples already very close to that boundary.

### Threshold pattern

For `L32`:

- `question_tn_correction` / `object_tn_correction`
  - rescue all three samples by alpha `6`
- `add_tn_correction`
  - rescues one sample already at alpha `4`
  - rescues all three by alpha `6`
- `reverse_logistic_fp_direction`
  - needs alpha `8` for all rescued samples
- `random_rescue_control`
  - rescues `0/64`

For `L24`:

- rescue is still present, but weaker and less decisive than `L32`
- `reverse_logistic_fp_direction` remains the strongest aggregate direction
- `question/object` rescue exists but does not surpass the global direction on margin

### Interpretation

- The causal story is now more stable than before:
  - the `L32` local-rescue advantage survived expansion from `16` to `32` to `64`
  - the random control stayed clean throughout
- However, rescue remains narrow in semantic scope:
  - only `3/64` decoded flips
  - all rescued cases are borderline `popular` FP samples

Current best honest reading:

- there is repeatable causal evidence that **later-layer local TN-conditioned correction geometry is more useful than the global supervised rescue direction**
- but the effect is currently strongest only for borderline hallucinations, not for the bulk of FP samples

## 2026-04-24 Stage G Semantic Projection

After the Stage E returns began to diminish, we shifted to semantic interpretation rather than further alpha / rescue-vector search.

The first Stage G pass projects three geometry objects into vocabulary space using the LLaVA language-model head:

- top-SVD backbone directions at `L20 / L24 / L32`
- the `L24` residual/tail slice `257-1024`
- `L32` local TN-conditioned rescue directions:
  - `question_tn_correction`
  - `object_tn_correction`
  - `local_knn_tn_correction`

Implementation notes:

- only LM-head / final-norm tensors are loaded, not the full model
- token vectors are row-normalized before projection to reduce rare-token norm bias
- a natural-token filter is applied to remove most invalid, punctuation-only, and noisy subword tokens
- outputs are saved under `outputs/semantics/`

Main artifacts:

- `outputs/semantics/semantic_projection_summary.md`
- `outputs/semantics/semantic_projection_tokens.jsonl`
- `outputs/semantics/semantic_cluster_summary.csv`
- `outputs/semantics/semantic_object_summary.csv`
- `outputs/semantics/semantic_interpretation_summary.json`

### Main result

The three geometry objects have visibly different semantic fingerprints.

#### 1. Top-SVD backbone directions look like broad visual / scene axes

The cleanest examples appear around `L24` and `L32`.

At `L24`, several top directions map to interpretable visual clusters:

- `L24_svd_4`: sky / clouds / sun / view-like tokens
- `L24_svd_5`: trees / leaves / branches versus indoor wall / door-like tokens
- `L24_svd_6`: sky / blue / cloud versus tree / leaf / woods-like tokens
- `L24_svd_7` and `L24_svd_8`: window / glass / wall / floor / roof-like tokens

At `L32`, the backbone directions are still interpretable but begin to mix visual content with decision / relation structure:

- `L32_svd_5`: door / room / kitchen / cabinet versus tree / sky / cloud-like tokens
- `L32_svd_7`: person-action words such as holding / sitting / watching versus scene-background words
- `L32_svd_3`: counting / digit tokens versus visual scene/object tokens

Interpretation:

- the top-SVD backbone is not random-looking
- it appears to contain broad reusable visual-semantic axes
- however, the axes are not purely object-presence detectors; many are scene, texture, spatial, count, or action contrasts

#### 2. The `L24` tail slice is more object-heavy than the top backbone

The `L24_tail_257_1024` subspace-energy projection surfaces concrete POPE-relevant object tokens:

- `horse`
- `cat`
- `bus`
- `cow`
- `horses`
- `motor`
- stop / stopped / stopping-like tokens

The automatic category summary marks the slice as object-heavy relative to most individual top-SVD directions.

Interpretation:

- this supports the earlier Stage B / E story that the useful residual/tail slice is not just generic variance
- semantically, it is closer to concrete object evidence than the broad top-SVD backbone
- this is exactly the kind of distinction needed for the claim that top variance geometry and hallucination-relevant correction geometry are partly misaligned

#### 3. The `L32` local rescue directions look decision-arbitration-like, not object-noun-like

The `L32` local TN-conditioned directions share a very stable token signature:

- positive side: `with`, `over`, `being`, `like`, `out`, `near`, `high`, `back`, `large`
- negative side: `yes`, `no`, `there`, `yeah`, `despite`, `usually`, `unfortunately`, `whether`

This is different from the `L24` tail slice.

Interpretation:

- `L32` local rescue directions do not look like simple object axes
- they look more like relational / contextual / decision-arbitration axes
- this fits the Stage E result: `L32` rescue is useful only near the yes/no boundary and mostly shifts margins rather than broadly rewriting visual evidence

### Current interpretation

Stage G gives a useful semantic decomposition:

| Geometry object | Current semantic reading |
| --- | --- |
| top-SVD backbone | broad visual-scene / attribute / count / action axes |
| `L24` tail `257-1024` | more concrete object-evidence-heavy correction slice |
| `L32` local TN rescue | later decision / relation / yes-no arbitration direction |

This makes the mechanism chain more narratable:

1. The dominant SVD backbone captures large visual-semantic changes caused by image conditioning.
2. The hallucination-sensitive correction signal is partly displaced into lower-variance residual/tail directions.
3. Later-layer local TN-conditioned directions appear to act less like object detectors and more like boundary-level arbitration signals.

### Caveat

This is a vocabulary-space semantic projection, not a full mechanistic logit-lens proof.

The result should be described as a semantic fingerprint of the directions, not as proof that these exact tokens mediate the behavior.

Current strongest honest claim:

- the geometry is **partially interpretable**
- top backbone, residual/tail correction, and local rescue directions have different semantic fingerprints
- this supports a weaker but useful name such as **grounding-related correction geometry**
- it is still too early to call it a complete or universal visual grounding subspace

## 2026-04-24 Stage G Sample-Level Semantic Check

We then extended Stage G from token-space projection to sample-level checks.

For each interpreted object, the script now computes sample coefficients over the full POPE hidden-state set:

- signed direction score for top-SVD and local-rescue directions
- subspace-energy score for the `L24` tail slice
- top positive / negative / energy samples
- outcome-level score summaries
- pairwise contrasts such as `FP_vs_TN`

New artifacts:

- `outputs/semantics/semantic_sample_extremes.csv`
- `outputs/semantics/semantic_outcome_contrasts.csv`
- updated `outputs/semantics/semantic_projection_summary.md`

### Main result

The sample-level check supports the semantic-fingerprint interpretation, but **does not** support treating any single semantic direction as a strong FP/TN classifier.

The strongest `FP_vs_TN` single-object contrasts remain small:

| Object | Reading | Mean diff FP-TN | Cohen d | AUC |
| --- | --- | ---: | ---: | ---: |
| `L20_svd_8` | window / wall-ish backbone axis | `+0.8904` | `+0.212` | `0.562` |
| `L24_svd_7` | window / wall-ish backbone axis | `-0.9872` | `-0.154` | `0.456` |
| `L32_svd_5` | door / room vs tree / sky axis | `-1.6153` | `-0.130` | `0.454` |
| `L24_tail_257_1024` | object-heavy tail slice | `-0.3978` | `-0.077` | `0.485` |
| `L32_local_knn_tn_correction` | local rescue arbitration direction | `-0.7211` | `-0.096` | `0.476` |
| `L32_question_tn_correction` | local rescue arbitration direction | `-0.5124` | `-0.067` | `0.482` |

This is useful because it prevents overclaiming:

- the vocabulary projection gives interpretable fingerprints
- the sample coefficients show some coherent extreme examples
- but individual directions are not enough to separate hallucinated from grounded cases by themselves

### Object-specific observations

#### `L24_tail_257_1024`

The highest-energy samples are mostly correct negative cases:

- top-20 energy outcomes: `18 TN`, `2 FP`
- examples include absent-object questions such as refrigerator / dining table / toothbrush

This is compatible with the Stage E ablation result:

- the tail slice seems especially important for maintaining correct negative evidence
- removing it damages TN behavior
- but its raw energy alone is not a strong global FP/TN detector

#### `L32` local TN rescue directions

For `question_tn_correction` and `object_tn_correction`, positive extremes include a mix of TN and TP, while negative extremes are heavily TP/FN.

For `local_knn_tn_correction`:

- positive top-20: `10 TN`, `9 TP`, `1 FP`
- negative top-20: `10 TP`, `6 FN`, `4 TN`

This fits the previous interpretation:

- the L32 rescue directions are not simple object-presence axes
- they look more like late-stage decision / relation / yes-no arbitration directions
- their intervention effect is therefore expected to be boundary-local, not broadly separative

#### Top-SVD visual axes

Several top-SVD axes show coherent sample extremes:

- `L24_svd_5`: potted-plant / tree-like positives and person-heavy negatives
- `L32_svd_5`: indoor dining-table / chair-like positives and outdoor / object-mismatch negatives
- `L32_svd_7`: person/action-like positive token signature, but sample extremes are mixed

This supports partial interpretability, but not a clean semantic basis.

### Updated interpretation

Stage G now supports three increasingly careful claims:

1. Vocabulary-space projections reveal non-random semantic fingerprints.
2. Sample extremes often contain recognizable object / scene / decision themes.
3. Outcome separation from any single interpreted direction is weak.

So the right framing is:

- **not** “we found a semantic coordinate that detects hallucination”
- but rather “the geometry contains partially interpretable visual and decision-related components, whose causal relevance appears only when combined with the Stage B/E evidence”

This makes the overall mechanism story stronger because it is more honest:

- Stage C/B/E provide the behavioral and causal evidence
- Stage G explains what the geometry seems to be about
- Stage G alone should not be used as a detector claim

## 2026-04-25 Stage N AMBER External Pilot

We ran the first external-validity check on the AMBER discriminative subset, using the local AMBER layout under `data/amber/`.

Artifacts:

- AMBER pilot plan: `outputs/stage_n_external/amber_discriminative_plan.jsonl`
- AMBER predictions: `outputs/stage_n_external/amber_predictions.jsonl`
- Official-eval formatted responses: `outputs/stage_n_external/amber_responses_for_official_eval.json`
- Hidden states: `outputs/stage_n_external/amber_hidden/layer_{20,24,32}.pt`
- Transfer scores: `outputs/stage_n_external/external_transfer_scores.csv`
- Per-category summary: `outputs/stage_n_external/external_category_summary.csv`
- Plot: `outputs/plots/stage_n_external_transfer.png`

Pilot setup:

- Samples: 1500 AMBER discriminative questions.
- Dimensions: `attribute`, `existence`, and `relation`.
- Sampling cap: `MAX_PER_DIMENSION_LABEL=300`.
- Hidden-state layers: L20 / L24 / L32.
- External analysis uses POPE-estimated geometry without refitting on AMBER.

AMBER prediction quality:

| Dimension | N | Accuracy | Notes |
| --- | ---: | ---: | --- |
| attribute | 600 | 0.775 | 75 FP, 60 FN |
| existence | 300 | 0.883 | 35 FP, no positive-label rows in this pilot |
| relation | 600 | 0.743 | 29 FP, 125 FN |
| overall | 1500 | 0.784 | pilot aggregate |

Transfer result:

The first energy-only SVD features transferred weakly, but the updated analysis added POPE-train FP/TN logistic probes over POPE SVD coordinates and applied those probes zero-shot to AMBER. These POPE-trained risk scores show above-chance transfer.

Strongest AMBER FP-risk rows:

| Layer | Dimension | Feature | FP AUROC | FP AUPRC |
| ---: | --- | --- | ---: | ---: |
| 20 | existence | `pope_probe_top_4_fp_risk` | 0.771 | 0.323 |
| 20 | existence | `evidence_fisher_fp_tn_k4_fp_risk` | 0.724 | 0.248 |
| 24 | existence | `evidence_fisher_fp_tn_k4_fp_risk` | 0.716 | 0.229 |
| 24 | relation | `pope_probe_top_256_fp_risk` | 0.700 | 0.179 |
| 24 | existence | `pope_probe_top_4_fp_risk` | 0.680 | 0.202 |
| 24 | attribute | `evidence_fisher_fp_tn_k64_fp_risk` | 0.644 | 0.189 |
| 24 | relation | `evidence_fisher_fp_tn_k64_fp_risk` | 0.644 | 0.103 |
| 20 | attribute | `evidence_pls_fp_tn_k64_fp_risk` | 0.641 | 0.183 |
| 24 | attribute | `pope_probe_top_256_fp_risk` | 0.632 | 0.205 |
| 20 | relation | `pope_probe_top_256_fp_risk` | 0.632 | 0.106 |

Evidence-specific transfer:

- PLS FP/TN and Fisher FP/TN bases were exported to `outputs/stage_n_external/evidence_transfer_bases/layer_{20,24,32}.pt`.
- These bases were trained/estimated from POPE only and applied zero-shot to AMBER.
- Evidence-specific FP-risk scores transfer above chance:
  - L20 existence Fisher K=4: FP AUROC 0.724
  - L24 existence Fisher K=4: FP AUROC 0.716
  - L24 attribute Fisher K=64: FP AUROC 0.644
  - L24 relation Fisher K=64: FP AUROC 0.644
  - L20 attribute PLS K=64: FP AUROC 0.641
- This is stronger than raw SVD-energy transfer, but it does not beat the best POPE top-SVD coordinate probe in the current pilot.

Current interpretation:

- POPE-trained FP/TN decision geometry does transfer beyond POPE on this AMBER pilot.
- The transfer is strongest for existence, but relation and attribute are not zero-signal.
- This supports an external-validity claim for POPE-trained correction geometry, not merely for POPE-specific labels.
- The current pilot does **not** support the stronger claim that tail/residual transfer is better than top-SVD/top-probe transfer on AMBER.
- POPE evidence-specific Stage L bases also transfer above chance, but their advantage over plain SVD depends on the comparison: they beat raw SVD energy, not the strongest SVD-coordinate FP-risk probe.

Recommended next step:

- Run the full AMBER discriminative set using the separately prepared full plan under `outputs/stage_n_external_full/`. The plan has 14216 rows, no missing images, and no invalid labels.
- Suggested GPU command: `OUTPUT_DIR=outputs/stage_n_external_full scripts/run_gpu_stage_n_amber.sh`.
- After GPU finishes, run: `OUTPUT_DIR=outputs/stage_n_external_full scripts/run_cpu_stage_n_transfer.sh`.

## 2026-04-27 Stage N Full AMBER External Result

The full AMBER discriminative run completed under `outputs/stage_n_external_full/`.

Artifacts:

- Full plan: `outputs/stage_n_external_full/amber_discriminative_plan.jsonl`
- Predictions: `outputs/stage_n_external_full/amber_predictions.jsonl`
- Hidden states: `outputs/stage_n_external_full/amber_hidden/layer_{20,24,32}.pt`
- Transfer scores: `outputs/stage_n_external_full/external_transfer_scores.csv`
- Summary: `outputs/stage_n_external_full/external_category_summary.csv`
- Evidence-specific bases: `outputs/stage_n_external_full/evidence_transfer_bases/layer_{20,24,32}.pt`

Full AMBER prediction quality:

| Dimension | N | Accuracy | Outcomes |
| --- | ---: | ---: | --- |
| attribute | 7628 | 0.798 | 3044 TP / 3044 TN / 770 FP / 770 FN |
| existence | 4924 | 0.878 | 4325 TN / 599 FP |
| relation | 1664 | 0.712 | 552 TP / 632 TN / 57 FP / 423 FN |
| overall | 14216 | 0.816 | 3596 TP / 8001 TN / 1426 FP / 1193 FN |

Strongest full-AMBER FP-risk rows:

| Layer | Dimension | Feature | FP AUROC | FP AUPRC |
| ---: | --- | --- | ---: | ---: |
| 24 | relation | `evidence_fisher_fp_tn_k64_fp_risk` | 0.665 | 0.075 |
| 24 | relation | `pope_probe_top_256_fp_risk` | 0.664 | 0.098 |
| 24 | existence | `pope_probe_top_64_fp_risk` | 0.663 | 0.207 |
| 20 | existence | `pope_probe_top_4_fp_risk` | 0.661 | 0.257 |
| 24 | existence | `evidence_fisher_fp_tn_k16_fp_risk` | 0.657 | 0.188 |
| 24 | attribute | `evidence_pls_fp_tn_k8_fp_risk` | 0.633 | 0.150 |
| 24 | attribute | `evidence_fisher_fp_tn_k64_fp_risk` | 0.629 | 0.149 |

Interpretation:

- Full AMBER confirms above-chance transfer, but the pilot overestimated the peak AUROC.
- The strongest layer is now consistently L24 across relation, existence, and attribute summaries.
- Evidence-specific Fisher/PLS transfer remains competitive and sometimes best, especially for relation and attribute.
- Raw energy-style scores remain weak: the best tail-energy FP AUROC is L20 existence at 0.561.
- This makes the external claim more conservative but stronger statistically: POPE-trained risk geometry transfers modestly beyond POPE, while raw SVD/tail energy alone is not a strong external detector.

## 2026-04-27 Stage P Multi-Seed Robustness

We ran Stage P probe robustness on POPE FP-vs-TN detection.

Artifacts:

- Per-seed rows: `outputs/stage_p_stats/multiseed_probe_rows.csv`
- Summary: `outputs/stage_p_stats/multiseed_probe_summary.csv`
- Layer ranks: `outputs/stage_p_stats/multiseed_layer_rank.csv`
- Paired bootstrap tests: `outputs/stage_p_stats/significance_tests.csv`
- Protocol note: `notes/statistical_testing_protocol.md`

Setup:

- Layers: L16 / L20 / L24 / L32
- Seeds: 13 / 17 / 23 / 29 / 31
- Split: stratified FP/TN, test fraction 0.3
- Features: full difference, top-4/top-64/top-256 SVD coordinates, tail 257-1024

Main multi-seed AUROC:

| Layer | Feature | Mean AUROC | Std | 95% CI |
| ---: | --- | ---: | ---: | --- |
| 24 | full_diff | 0.721 | 0.027 | 0.699-0.741 |
| 20 | full_diff | 0.720 | 0.028 | 0.696-0.741 |
| 16 | full_diff | 0.714 | 0.019 | 0.699-0.727 |
| 32 | full_diff | 0.703 | 0.021 | 0.685-0.717 |
| 20 | top_256 | 0.677 | 0.035 | 0.653-0.706 |
| 32 | tail_257_1024 | 0.667 | 0.028 | 0.646-0.686 |
| 24 | tail_257_1024 | 0.656 | 0.046 | 0.622-0.691 |
| 24 | top_4 | 0.471 | 0.007 | 0.466-0.476 |

Rank stability:

- `full_diff` is the top-ranked feature in all 5 seeds for L16, L20, L24, and L32.

Paired bootstrap conclusions:

- `top_256` is significantly stronger than `top_4` at all tested layers.
- `tail_257_1024` is significantly stronger than `top_4` at all tested layers.
- `full_diff` is significantly stronger than both `top_256` and `tail_257_1024` at all tested layers.
- Example deltas:
  - L24 `top_256 - top_4`: +0.193 AUROC, 95% CI 0.155-0.227
  - L24 `full_diff - top_256`: +0.053 AUROC, 95% CI 0.032-0.075
  - L32 `full_diff - tail`: +0.036 AUROC, 95% CI 0.011-0.060

Interpretation:

- The variance/discrimination mismatch is robust: top-4 directions remain weak despite dominating variance.
- Mid/high-dimensional SVD coordinates and tail coordinates carry stable FP/TN signal.
- However, the full difference vector remains the strongest detector under this logistic setup, so tail/residual geometry should be framed as mechanistically informative and causally relevant, not as the best standalone detector.

## 2026-04-27 Stage Q Paper Asset Draft

We generated the first paper-ready table/figure bundle from the accumulated outputs.

Artifacts:

- Asset index: `outputs/paper_tables/stage_q_asset_index.md`
- Tables:
  - `outputs/paper_tables/table1_pope_summary.csv`
  - `outputs/paper_tables/table2_feature_comparison.csv`
  - `outputs/paper_tables/table3_controls.csv`
  - `outputs/paper_tables/table4_intervention.csv`
- Figures:
  - `outputs/paper_figures/fig1_method_overview.pdf`
  - `outputs/paper_figures/fig2_variance_vs_auroc.pdf`
  - `outputs/paper_figures/fig3_condition_geometry.pdf`
  - `outputs/paper_figures/fig4_intervention_dose.pdf`
  - `outputs/paper_figures/fig5_layered_geometry.pdf`

Figure contents:

- Figure 1: paired image/question and blind-question pipeline, `z_img`, `z_blind`, difference geometry, SVD bands/probes/interventions.
- Figure 2: cumulative explained variance vs FP/TN AUROC over top-K SVD coordinates.
- Figure 3: matched vs random/adversarial mismatch geometry, separated into top-4 backbone and tail 257-1024 views.
- Figure 4: L24 tail-ablation dose curve, margin shift, TN damage, and norm-matched random-tail control.
- Figure 5: layered summary of top-4 variance concentration vs full-diff/tail FP/TN AUROC, plus Stage E layer-sweep intervention sensitivity.

Table contents:

- Table 1: POPE subset performance and outcome distribution.
- Table 2: feature comparison across raw image/blind states, full difference, top-K SVD, full SVD coordinates, random controls, and evidence-specific subspaces.
- Table 3: destructive geometry controls from Stage J.
- Table 4: intervention summary covering Stage E tail ablation and Stage M local rescue rows.

Current caveat:

- Figure 5 is still a first-pass layered summary. It now includes a cross-layer intervention-sensitivity panel, but the “middle-layer correction / late-layer arbitration” wording should remain cautious until the intervention evidence is cleaned up further.

## 2026-05-04 Stage R2 Case Panels

We generated human-readable case panels for qualitative analysis.

Artifacts:

- Metadata: `outputs/case_studies/case_panel_metadata.csv`
- Notes: `notes/case_studies.md`
- Build summary: `outputs/case_studies/build_stage_r_case_panels_summary.json`

Case coverage:

| Category | Count |
| --- | ---: |
| successful TN with strong tail correction | 4 |
| FP with weak matched correction | 4 |
| FP rescued by local steering | 2 |
| FP not rescued despite high score | 4 |
| adversarial mismatch example | 4 |
| semantic direction extreme samples | 4 |
| total | 22 |

Notes:

- The current Stage M run has only 2 unique rescued FP samples, so that category cannot reach the default 4 examples without duplicating cases.
- Each case includes image path, question, ground truth, model answer, outcome, target object, geometry scores, intervention summary when available, and a short human-readable explanation.
- These panels are intended as qualitative support and failure analysis, not as new quantitative evidence.

## 2026-05-04 Stage R1 Semantic Fingerprints

We consolidated the Stage G semantic analysis into a paper-facing semantic fingerprint bundle.

Artifacts:

- Summary: `outputs/stage_r_semantics/semantic_fingerprint_summary.csv`
- Per-object panels: `outputs/stage_r_semantics/semantic_sample_panels/`
- Conclusion note: `notes/semantic_interpretation_conclusion.md`
- Build summary: `outputs/stage_r_semantics/build_stage_r_semantic_fingerprints_summary.json`

Coverage:

- 28 projected geometry objects:
  - top-SVD backbone directions
  - L24 tail 257-1024 slice
  - L32 local TN/rescue directions
- 4 Stage L evidence-specific quantitative-only rows:
  - PLS FP/TN
  - Fisher FP/TN
  - contrastive PCA
  - matched-vs-adversarial logistic

Main conclusion:

- Top-SVD backbone directions often have broad visual-scene, attribute, count, action, or spatial vocabulary fingerprints.
- The L24 tail 257-1024 slice is object-heavy and aligns with the tail-ablation story.
- L32 local TN rescue directions look more like relational/contextual or yes-no arbitration directions than simple object-presence detectors.
- Stage L evidence-specific subspaces are quantitatively useful, but they have not yet been vocabulary-projected.

Important limitation:

- No single projected semantic direction is a strong hallucination detector. The strongest sample-level FP/TN contrasts remain modest, e.g. L20 SVD-8 FP/TN AUC 0.562.
- The recommended wording is **partially interpretable grounding-related correction geometry**, not **semantic hallucination coordinate** or **universal grounding subspace**.

## 2026-05-04 Stage S Baseline Positioning

We generated baseline comparison tables for detection and mitigation/rescue positioning.

Artifacts:

- Detection baselines: `outputs/stage_s_baselines/detection_baseline_comparison.csv`
- Mitigation baselines: `outputs/stage_s_baselines/mitigation_baseline_comparison.csv`
- Notes: `notes/baseline_positioning.md`
- Build summary: `outputs/stage_s_baselines/build_stage_s_baselines_summary.json`

Detection result:

- On the Stage M first-token subset, yes/no margin is a very strong FP-vs-TN baseline:
  - yes/no margin AUROC: 1.000
  - binary entropy AUROC: 0.884
  - Stage M FP-risk score AUROC: 0.827
- On full POPE hidden-state probes / multi-seed summaries:
  - paired full difference, 5-seed mean AUROC: 0.721
  - Stage L PLS FP/TN AUROC: 0.720
  - single-run paired full difference AUROC: 0.694
  - raw blind-state hidden probe AUROC: 0.672
  - raw image-state hidden probe AUROC: 0.651

Interpretation:

- The paired-difference method should not be framed as a pure detection leaderboard win, especially because margin/entropy can be very strong when logits are available.
- The stronger claim is mechanistic: paired differencing localizes hallucination-related signal and shows why top-variance directions are not the decision geometry.

Mitigation/rescue result:

- Current Stage M rescue remains weak and boundary-local.
- Random TN, global TN, and local TN correction all reach 0.25 rescue rate on a gated 4-FP subset at alpha 8, so local retrieval does not clearly beat controls.
- TN/TP preservation is undefined for those gated best rows because no TN/TP controls passed the same `margin_and_fp_risk` gate.
- VCD/ICD and evidence-specific steering were not run in the current artifact set.

## 2026-05-04 Stage T Final Paper Framing

We made the final paper-positioning decision based on the completed Stage P/Q/R/S evidence.

Artifact:

- Decision note: `notes/final_paper_framing_decision.md`

Decision:

- The paper should be framed as a **mechanistic analysis paper**, not a mitigation-method paper.
- Main contribution: blind-reference differencing reveals layered correction geometry in LVLM hallucination.
- Intervention/rescue should be presented as causal evidence for the relevance of the geometry, not as a reliable mitigation method.

Rationale:

- The paired-difference family remains mechanistically strong and competitive across hidden-state probes.
- Top-variance SVD directions are repeatedly shown to be insufficient as decision geometry.
- Semantic fingerprints support partially interpretable grounding-related correction geometry, but not a single semantic hallucination coordinate.
- Detection baselines using first-token margin/entropy can be very strong when logits are available.
- Current rescue is weak, boundary-local, and not clearly better than random/global/local TN controls on the small gated subset.

Recommended title direction:

- "Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination"

## 2026-05-04 TODO Checklist Synchronization

After Stage T, we reconciled the bottom completion checklist with the already completed stage artifacts.

Updated checklist status:

- J1 shuffle controls: completed; results are in `outputs/stage_j_controls/`.
- J2 random subspace controls: completed; random subspaces are close enough that the paper should report distributed signal rather than uniquely SVD-captured signal.
- K1 token-position pilot: completed; the core signal survives multiple readout positions.
- L1 evidence-specific subspace extraction: completed for extraction/evaluation, but not wired into intervention.
- M1/M2 adaptive local rescue: completed for memory bank, gated steering, and analysis; rescue remains weak and boundary-local.

Remaining high-impact open items:

- O1/O2 cross-model minimal replication.
- Lightweight adapter or representation-editing extension.
- Evidence-specific steering and compute-overhead measurement if the intervention story is revisited.

## 2026-05-04 Stage O Cross-Model Preparation

We prepared the next high-impact experiment: a minimal cross-model replication.

Choice note:

- `notes/cross_model_choice.md`

Prepared scripts:

- GPU: `scripts/run_gpu_stage_o_cross_model.sh`
- CPU: `scripts/run_cpu_stage_o_cross_model.sh`
- Summary builder: `scripts/build_stage_o_cross_model_summary.py`

Current model choice:

- Use a LLaVA-HF-compatible additional checkpoint first, preferably `llava-hf/llava-1.5-13b-hf` if locally available and GPU memory allows.
- This is a conservative wrapper-compatible choice because the current environment has `transformers==4.37.2` and does not expose the LLaVA-NeXT or Qwen2-VL generation classes.

Prepared minimal chain:

- POPE prediction summary.
- L20/L24/L32 hidden states.
- Blind-image difference SVD and spectrum summary.
- Stage C-style FP/TN probes.
- Stage B matched-vs-random/adversarial mismatch condition geometry.
- No Stage E intervention unless early cross-model results look promising.

## 2026-05-04 Stage O Cross-Model Replication Result

The Stage O minimal replication completed for `llava_13b`, using local checkpoint `/data/lh/ModelandDataset/llava-1.5-13b-hf`.

Artifacts:

- Summary: `outputs/stage_o_cross_model/llava_13b/minimal_replication_summary.csv`
- Note: `notes/cross_model_replication.md`
- Prediction summary: `outputs/stage_o_cross_model/llava_13b/predictions/run_pope_eval_summary.json`
- Probe table: `outputs/stage_o_cross_model/llava_13b/probes/probe_results.csv`
- Condition geometry: `outputs/stage_o_cross_model/llava_13b/stage_b/stage_b_pairwise_condition_deltas.csv`

POPE outcome:

- Accuracy: `0.871`
- FP: `468`
- TN: `4032`
- TP: `3804`
- FN: `696`
- unknown: `0`

Main replication result:

- The full blind-image difference is again the strongest available hidden-state detector:
  - L20 full difference AUROC: `0.736`
  - L24 full difference AUROC: `0.726`
  - L32 full difference AUROC: `0.723`
- Variance/discrimination mismatch clearly reappears:
  - top-4 explained variance is large (`0.752` to `0.808`)
  - but top-4 projected AUROC remains weak (`0.535` to `0.552`)
- Larger projected coordinates become useful:
  - best projected row: L32 K=128 AUROC `0.699`
- Tail condition geometry partly reproduces:
  - band 257-1024 matched-minus-adversarial deltas are positive at L20 `18.126`, L24 `28.860`, L32 `75.504`
  - matched-minus-random tail deltas are mixed: L20 `3.622`, L24 `-0.711`, L32 `-14.696`

Conclusion:

- This is an acceptable-to-strong checkpoint-level replication across LLaVA-1.5 7B/13B.
- It supports saying the mechanism is not a single-checkpoint artifact.
- It still does not support a broad cross-architecture LVLM generality claim.

## 2026-05-04 Representation Editing Prep

We prepared a CPU-side direction bank for a future lightweight representation-editing pilot.

Artifacts:

- Direction bank: `outputs/representation_editing_prep/editing_direction_bank.pt`
- Direction summary: `outputs/representation_editing_prep/direction_summary.csv`
- Candidate evaluation plan: `outputs/representation_editing_prep/candidate_eval_plan.csv`
- Note: `notes/representation_editing_prep.md`
- Build summary: `outputs/representation_editing_prep/build_representation_editing_prep_summary.json`

What was built:

- 102 normalized candidate edit directions across L20/L24/L32.
- 384 candidate GPU evaluation rows over direction, alpha, and gate choices.
- Directions include:
  - global TN mean correction
  - global TN-minus-FP correction
  - low-rank projected TN mean directions
  - low-rank projected TN-minus-FP directions
  - bases from plain SVD, Fisher FP/TN, PLS FP/TN, and matched-vs-adversarial logistic subspaces

Top recommendation:

- First GPU test: L24 `pls_fp_tn` k=32 `projected_tn_minus_fp`, with alpha 2/4/8 and `margin_and_fp_risk` gating.

Important limitation:

- This is a prepared activation-editing direction bank, not a trained LoRA adapter and not an evaluated mitigation method.
