# Stage T Geometry Complementarity Results

Date: 2026-05-08

Command:

```bash
bash scripts/run_cpu_stage_t_geometry_complementarity.sh
```

Primary outputs:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_geometry_margin_bin_analysis.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_geometry_residual_prediction.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_geometry_margin_correlations.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_geometry_same_margin_pairs.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_geometry_complementarity_summary.md`

## Setup

- Layer: L24.
- Split: fixed-id `test`.
- Population: model-predicted `Yes` samples only, i.e. FP/TP.
- Calibration split for margin-residual gates: fixed-id `calibration`.
- Primary geometry score: `pls32_probe`.
- Margin baseline: `low_margin_probe`, where higher score means lower yes/no margin and higher FP risk among predicted-Yes samples.

## 1. Margin-bin analysis

Margin bins use fixed yes-minus-no logit edges:

| Bin | Margin range | N | FP | TP |
| --- | --- | ---: | ---: | ---: |
| very_low | 0.016 to 0.500 | 61 | 21 | 40 |
| low | 0.516 to 1.500 | 135 | 24 | 111 |
| medium | 1.516 to 3.000 | 215 | 8 | 207 |
| high | 3.016 to 5.062 | 185 | 0 | 185 |

At top-20% warning rate within each margin bin:

| Score | Bin | AUROC | FP recall | Warning precision |
| --- | --- | ---: | ---: | ---: |
| `pls32_probe` | very_low | 0.599 | 0.286 | 0.462 |
| `pls32_probe` | low | 0.737 | 0.417 | 0.370 |
| `pls32_probe` | medium | 0.646 | 0.250 | 0.047 |
| `full_probe` | low | 0.673 | 0.542 | 0.481 |
| `tail_257_1024_probe` | low | 0.691 | 0.500 | 0.444 |

Takeaway: margin explains a lot of the global FP/TP separation, but geometry still has local ranking power in the low-margin region. The medium bin is directionally positive for PLS but underpowered, with only 8 FP. The high-margin bin has no FP in this split, so it cannot test complementarity.

## 2. Residual prediction

With the margin-only gate calibrated at top 20% on the calibration split:

- Margin-only triggers 133/596 predicted-Yes test samples.
- It captures 39/53 FP, so FP recall is 0.736.
- It misses 14 FP.
- Warning precision is 0.293.

Geometry applied after margin-only:

| Geometry score | Residual AUROC | Extra FP caught | Extra precision | Missed-FP captured | Union FP recall | Union trigger rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pls32_probe` | 0.633 | 3 | 0.060 | 0.214 | 0.792 | 0.307 |
| `full_probe` | 0.542 | 4 | 0.053 | 0.286 | 0.811 | 0.351 |
| `tail_257_1024_probe` | 0.494 | 4 | 0.055 | 0.286 | 0.811 | 0.346 |

Takeaway: PLS geometry predicts the residual margin-missed pool above chance and recovers 3/14 missed FP. The cost is substantial additional triggers, so this is evidence for complementarity, not yet a clean selective-correction operating point.

## 3. Correlation analysis

On predicted-Yes FP/TP test samples:

| Score | Reference | Pearson | Spearman |
| --- | --- | ---: | ---: |
| `pls32_probe` | yes/no margin | -0.231 | -0.197 |
| `pls32_probe` | binary entropy | 0.248 | 0.197 |
| `full_probe` | yes/no margin | -0.179 | -0.104 |
| `full_probe` | binary entropy | 0.196 | 0.104 |
| `tail_257_1024_probe` | yes/no margin | -0.201 | -0.104 |
| `tail_257_1024_probe` | binary entropy | 0.213 | 0.104 |

Takeaway: geometry is related to confidence, but far from redundant. The correlations are low-to-moderate, which supports the non-duplication claim.

## 4. Same-margin pair case study

Examples where FP and TP have nearly identical yes/no margins, but PLS geometry is much higher on the FP:

| FP sample | TP sample | Margin delta | FP PLS | TP PLS | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `coco:popular:438` | `coco:popular:1869` | 0.062 | 0.999 | 0.013 | 0.986 |
| `coco:adversarial:750` | `coco:adversarial:2871` | 0.062 | 0.996 | 0.020 | 0.975 |
| `coco:adversarial:526` | `coco:popular:1253` | 0.062 | 0.990 | 0.024 | 0.967 |
| `coco:popular:2598` | `coco:adversarial:1653` | 0.062 | 0.960 | 0.021 | 0.939 |
| `coco:popular:982` | `coco:random:1141` | 0.016 | 0.940 | 0.012 | 0.928 |

Takeaway: these are useful appendix cases. They make the complementarity story more readable because the output margin is nearly tied while the geometry risk changes sharply.

## Recommended wording

Do not claim that geometry beats margin globally. The cleaner claim is:

> Output margin is the strongest single global confidence signal, but correction geometry contributes non-redundant local information. In low-margin predicted-Yes regions and among margin-missed FP residuals, geometry scores still rank FP above TP, and their correlations with margin/entropy remain modest rather than near-identical.

