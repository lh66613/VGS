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
