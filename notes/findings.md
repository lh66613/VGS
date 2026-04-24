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
