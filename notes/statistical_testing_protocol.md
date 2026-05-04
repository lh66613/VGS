# Statistical Testing Protocol

## Stage P Multi-Seed Probe Protocol

- Layers: `[16, 20, 24, 32]`
- Seeds: `[13, 17, 23, 29, 31]`
- Split: stratified FP/TN split with test fraction `0.3`
- Features: full blind-image difference, top-4/top-64/top-256 SVD coordinates, and tail SVD coordinates
- Tail band: `257-1024`
- Model: class-balanced logistic regression with per-feature standardization
- Reported metrics: AUROC, AUPRC, accuracy, F1, seed mean/std/min/max, and bootstrap CI over seed means

## Pairwise Significance Tests

- Bootstrap samples: `1000`
- AUROC deltas use stratified paired bootstrap over FP/TN test predictions.
- The p-value is a two-sided bootstrap sign probability around zero.
- Because the same POPE samples can appear across different random splits, these tests should be treated as robustness diagnostics rather than final inferential statistics.
