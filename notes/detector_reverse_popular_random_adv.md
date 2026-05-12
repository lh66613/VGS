# Detector Experiment Prep

## Protocol

- Strict subset-transfer: train `popular`, calibrate `random`, test `adversarial`.
- Task A: FP vs TN on ground-truth No samples.
- Task B: predicted-Yes FP vs TP deployment risk.
- Threshold policy for classification metrics: F1-optimal threshold selected on calibration only.

## Files

- `baseline_table`: `outputs/detector_reverse_popular_random_adv/detector_baseline_table.csv`
- `deployment_warning`: `outputs/detector_reverse_popular_random_adv/deployment_warning.csv`
- `threshold_audit`: `outputs/detector_reverse_popular_random_adv/threshold_audit.csv`
- `dimension_curve`: `outputs/detector_reverse_popular_random_adv/dimension_curve.csv`
- `spectral_band_curve`: `outputs/detector_reverse_popular_random_adv/spectral_band_curve.csv`
- `pls_diagnostics`: `outputs/detector_reverse_popular_random_adv/pls_diagnostics.csv`
- `bootstrap_comparisons`: `outputs/detector_reverse_popular_random_adv/bootstrap_comparisons.csv`
- `bootstrap_main_table`: `outputs/detector_reverse_popular_random_adv/bootstrap_main_table.csv`
- `trigger_curve_table`: `outputs/detector_reverse_popular_random_adv/trigger_curve_table.csv`
- `speed_cost_table`: `outputs/detector_reverse_popular_random_adv/speed_cost_table.csv`
- `feature_audit`: `outputs/detector_reverse_popular_random_adv/feature_audit.csv`

## Best Test Rows

| task                     | layer | method                      | feature_dim | test_auroc | test_auprc | f1    | mcc   | detector_score_ms_per_sample |
| ------------------------ | ----- | --------------------------- | ----------- | ---------- | ---------- | ----- | ----- | ---------------------------- |
| task_a_fp_vs_tn          | 24    | yes_no_margin               | 1           | 1.000      | 1.000      | 0.989 | 0.988 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top16_svd_diff  | 17          | 1.000      | 1.000      | 0.997 | 0.997 | 0.000                        |
| task_a_fp_vs_tn          | 24    | output_logistic             | 8           | 1.000      | 1.000      | 0.979 | 0.976 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top4_svd_diff   | 5           | 1.000      | 0.999      | 0.995 | 0.994 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top64_svd_diff  | 65          | 1.000      | 0.997      | 0.974 | 0.970 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top256_svd_diff | 257         | 1.000      | 0.996      | 0.964 | 0.959 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_pls32_diff      | 33          | 0.997      | 0.979      | 0.953 | 0.946 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_tail_diff       | 769         | 0.996      | 0.970      | 0.934 | 0.924 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top256_svd_diff | 257         | 0.882      | 0.662      | 0.557 | 0.486 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_tail_diff       | 769         | 0.867      | 0.613      | 0.523 | 0.481 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top16_svd_diff  | 17          | 0.860      | 0.474      | 0.467 | 0.387 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_full_diff       | 4097        | 0.856      | 0.623      | 0.506 | 0.532 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_pls32_diff      | 33          | 0.854      | 0.516      | 0.455 | 0.373 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | output_logistic             | 8           | 0.851      | 0.439      | 0.485 | 0.397 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top4_svd_diff   | 5           | 0.848      | 0.395      | 0.485 | 0.399 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | yes_no_margin               | 1           | 0.848      | 0.401      | 0.489 | 0.402 | 0.000                        |

## Deployment Warning Snapshot

| layer | method                      | trigger_rate | warning_precision | relative_precision_gain | fp_recall | tp_damage |
| ----- | --------------------------- | ------------ | ----------------- | ----------------------- | --------- | --------- |
| 24    | margin_plus_top256_svd_diff | 0.253        | 0.401             | 2.948                   | 0.746     | 0.176     |
| 24    | margin_plus_tail_diff       | 0.253        | 0.389             | 2.864                   | 0.725     | 0.179     |
| 24    | margin_plus_top16_svd_diff  | 0.247        | 0.384             | 2.824                   | 0.698     | 0.176     |
| 24    | output_logistic             | 0.246        | 0.383             | 2.819                   | 0.693     | 0.176     |
| 24    | margin_plus_top4_svd_diff   | 0.246        | 0.383             | 2.819                   | 0.693     | 0.176     |
| 24    | binary_entropy              | 0.249        | 0.378             | 2.778                   | 0.693     | 0.180     |
| 24    | yes_no_margin               | 0.249        | 0.378             | 2.778                   | 0.693     | 0.180     |
| 24    | margin_plus_pls32_diff      | 0.249        | 0.373             | 2.744                   | 0.683     | 0.181     |
| 24    | margin_plus_full_diff       | 0.259        | 0.367             | 2.699                   | 0.698     | 0.190     |
| 24    | margin_plus_top64_svd_diff  | 0.250        | 0.356             | 2.622                   | 0.656     | 0.186     |
| 24    | raw_concat                  | 0.254        | 0.329             | 2.419                   | 0.614     | 0.197     |
| 24    | tail_257_1024_diff          | 0.240        | 0.308             | 2.270                   | 0.545     | 0.192     |

## Subspace Dimension Curve

| task                     | layer | method                  | feature_dim | test_auroc | test_auprc |
| ------------------------ | ----- | ----------------------- | ----------- | ---------- | ---------- |
| task_a_fp_vs_tn          | 24    | raw_full_diff_reference | 4096        | 0.745      | 0.409      |
| task_a_fp_vs_tn          | 24    | pls                     | 64          | 0.721      | 0.383      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 256         | 0.718      | 0.337      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 256         | 0.717      | 0.341      |
| task_a_fp_vs_tn          | 24    | pls                     | 16          | 0.703      | 0.271      |
| task_a_fp_vs_tn          | 24    | pls                     | 4           | 0.699      | 0.261      |
| task_a_fp_vs_tn          | 24    | pls                     | 256         | 0.693      | 0.374      |
| task_a_fp_vs_tn          | 24    | random                  | 256         | 0.675      | 0.268      |
| task_a_fp_vs_tn          | 24    | tail                    | 768         | 0.669      | 0.345      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 64          | 0.655      | 0.204      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 64          | 0.653      | 0.202      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 16          | 0.631      | 0.189      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 16          | 0.606      | 0.179      |
| task_a_fp_vs_tn          | 24    | random                  | 64          | 0.592      | 0.189      |
| task_a_fp_vs_tn          | 24    | random                  | 4           | 0.548      | 0.137      |
| task_a_fp_vs_tn          | 24    | random                  | 16          | 0.541      | 0.146      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 4           | 0.528      | 0.133      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 4           | 0.487      | 0.125      |
| task_b_pred_yes_fp_vs_tp | 24    | pls                     | 256         | 0.746      | 0.327      |
| task_b_pred_yes_fp_vs_tp | 24    | pls                     | 64          | 0.733      | 0.359      |

## Spectral Band Curve

| task                     | layer | mode                         | spectral_feature | feature_dim | test_auroc | test_auprc | warning_precision | fp_recall | tp_damage |
| ------------------------ | ----- | ---------------------------- | ---------------- | ----------- | ---------- | ---------- | ----------------- | --------- | --------- |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | tail_257_1024    | 768         | 0.677      | 0.436      | 0.308             | 0.545     | 0.192     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_5_16        | 12          | 0.676      | 0.236      | 0.236             | 0.386     | 0.196     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_65_256      | 192         | 0.644      | 0.300      | 0.274             | 0.476     | 0.198     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | top_1_4          | 4           | 0.570      | 0.147      | 0.114             | 0.164     | 0.200     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_17_64       | 48          | 0.542      | 0.246      | 0.189             | 0.302     | 0.203     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_16         | 16          | 0.697      | 0.270      | 0.238             | 0.386     | 0.195     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_1024       | 1024        | 0.692      | 0.559      | 0.321             | 0.582     | 0.194     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_256        | 256         | 0.675      | 0.517      | 0.275             | 0.476     | 0.197     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_64         | 64          | 0.662      | 0.301      | 0.252             | 0.429     | 0.200     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_4          | 4           | 0.570      | 0.147      | 0.114             | 0.164     | 0.200     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | tail_257_1024    | 769         | 0.867      | 0.613      | 0.389             | 0.725     | 0.179     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_5_16        | 13          | 0.858      | 0.473      | 0.379             | 0.688     | 0.177     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | top_1_4          | 5           | 0.848      | 0.395      | 0.383             | 0.693     | 0.176     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_65_256      | 193         | 0.842      | 0.521      | 0.351             | 0.646     | 0.188     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_17_64       | 49          | 0.828      | 0.459      | 0.366             | 0.667     | 0.181     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_256        | 257         | 0.882      | 0.662      | 0.401             | 0.746     | 0.176     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_16         | 17          | 0.860      | 0.474      | 0.384             | 0.698     | 0.176     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_1024       | 1025        | 0.858      | 0.658      | 0.368             | 0.683     | 0.185     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_4          | 5           | 0.848      | 0.395      | 0.383             | 0.693     | 0.176     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_64         | 65          | 0.829      | 0.455      | 0.356             | 0.656     | 0.186     |

## PLS Diagnostics

| task                     | layer | k   | train_auroc | calibration_auroc | test_auroc | split_half_overlap | overlap_top16 | overlap_tail_257_1024 |
| ------------------------ | ----- | --- | ----------- | ----------------- | ---------- | ------------------ | ------------- | --------------------- |
| task_a_fp_vs_tn          | 24    | 4   | 0.831       | 0.585             | 0.699      | 0.388              | 0.582         | 0.076                 |
| task_a_fp_vs_tn          | 24    | 16  | 0.839       | 0.581             | 0.703      | 0.417              | 0.483         | 0.100                 |
| task_a_fp_vs_tn          | 24    | 32  | 0.985       | 0.584             | 0.735      | 0.405              | 0.358         | 0.115                 |
| task_a_fp_vs_tn          | 24    | 64  | 0.983       | 0.576             | 0.721      | 0.439              | 0.190         | 0.134                 |
| task_a_fp_vs_tn          | 24    | 256 | 0.986       | 0.590             | 0.693      | 0.460              | 0.048         | 0.247                 |
| task_b_pred_yes_fp_vs_tp | 24    | 4   | 0.900       | 0.649             | 0.732      | 0.481              | 0.610         | 0.054                 |
| task_b_pred_yes_fp_vs_tp | 24    | 16  | 0.906       | 0.646             | 0.733      | 0.477              | 0.482         | 0.104                 |
| task_b_pred_yes_fp_vs_tp | 24    | 32  | 0.909       | 0.643             | 0.732      | 0.414              | 0.346         | 0.117                 |
| task_b_pred_yes_fp_vs_tp | 24    | 64  | 0.902       | 0.640             | 0.733      | 0.415              | 0.188         | 0.137                 |
| task_b_pred_yes_fp_vs_tp | 24    | 256 | 0.933       | 0.638             | 0.746      | 0.429              | 0.049         | 0.256                 |

## Bootstrap Comparisons

| comparison                | metric            | delta  | ci95            | significant |
| ------------------------- | ----------------- | ------ | --------------- | ----------- |
| margin+tail - margin      | AUROC             | 0.019  | [-0.008, 0.046] | no          |
| margin+tail - margin      | AUPRC             | 0.208  | [0.138, 0.278]  | yes         |
| margin+tail - margin      | Warning Precision | 0.011  | [-0.022, 0.044] | no          |
| margin+tail - margin      | FP Recall         | 0.032  | [-0.045, 0.101] | no          |
| margin+tail - margin      | TP Damage         | -0.001 | [-0.019, 0.017] | no          |
| margin+full - margin      | AUROC             | 0.008  | [-0.017, 0.036] | no          |
| margin+full - margin      | AUPRC             | 0.218  | [0.145, 0.286]  | yes         |
| margin+full - margin      | Warning Precision | -0.011 | [-0.045, 0.025] | no          |
| margin+full - margin      | FP Recall         | 0.006  | [-0.066, 0.079] | no          |
| margin+full - margin      | TP Damage         | 0.010  | [-0.009, 0.030] | no          |
| margin+tail - margin+full | AUROC             | 0.011  | [-0.007, 0.029] | no          |
| margin+tail - margin+full | AUPRC             | -0.010 | [-0.054, 0.034] | no          |
| margin+tail - margin+full | Warning Precision | 0.022  | [-0.011, 0.055] | no          |
| margin+tail - margin+full | FP Recall         | 0.026  | [-0.026, 0.076] | no          |
| margin+tail - margin+full | TP Damage         | -0.011 | [-0.031, 0.011] | no          |
| margin+top16 - margin     | AUROC             | 0.012  | [0.003, 0.021]  | yes         |
| margin+top16 - margin     | AUPRC             | 0.071  | [0.029, 0.110]  | yes         |
| margin+top16 - margin     | Warning Precision | 0.006  | [-0.014, 0.027] | no          |
| margin+top16 - margin     | FP Recall         | 0.006  | [-0.035, 0.044] | no          |
| margin+top16 - margin     | TP Damage         | -0.003 | [-0.016, 0.009] | no          |
| margin+tail - raw diff    | AUROC             | 0.181  | [0.139, 0.223]  | yes         |
| margin+tail - raw diff    | AUPRC             | 0.089  | [0.039, 0.141]  | yes         |
| margin+tail - raw diff    | Warning Precision | 0.097  | [0.055, 0.141]  | yes         |
| margin+tail - raw diff    | FP Recall         | 0.207  | [0.143, 0.272]  | yes         |
| margin+tail - raw diff    | TP Damage         | -0.018 | [-0.048, 0.010] | no          |

## Trigger Curve Table

| method       | target_trigger_rate | actual_trigger_rate | warning_precision | fp_recall | tp_damage |
| ------------ | ------------------- | ------------------- | ----------------- | --------- | --------- |
| random       | 0.100               | 0.156               | 0.138             | 0.159     | 0.156     |
| margin-only  | 0.100               | 0.127               | 0.424             | 0.397     | 0.085     |
| margin+top16 | 0.100               | 0.137               | 0.482             | 0.487     | 0.082     |
| margin+tail  | 0.100               | 0.156               | 0.521             | 0.598     | 0.087     |
| margin+full  | 0.100               | 0.152               | 0.469             | 0.524     | 0.093     |
| tail-only    | 0.100               | 0.141               | 0.423             | 0.439     | 0.094     |
| random       | 0.200               | 0.253               | 0.137             | 0.255     | 0.253     |
| margin-only  | 0.200               | 0.249               | 0.378             | 0.693     | 0.180     |
| margin+top16 | 0.200               | 0.247               | 0.384             | 0.698     | 0.176     |
| margin+tail  | 0.200               | 0.253               | 0.389             | 0.725     | 0.179     |
| margin+full  | 0.200               | 0.259               | 0.367             | 0.698     | 0.190     |
| tail-only    | 0.200               | 0.240               | 0.308             | 0.545     | 0.192     |
| random       | 0.300               | 0.352               | 0.136             | 0.351     | 0.352     |
| margin-only  | 0.300               | 0.352               | 0.321             | 0.831     | 0.276     |
| margin+top16 | 0.300               | 0.354               | 0.331             | 0.862     | 0.274     |
| margin+tail  | 0.300               | 0.352               | 0.313             | 0.810     | 0.280     |
| margin+full  | 0.300               | 0.352               | 0.311             | 0.804     | 0.280     |
| tail-only    | 0.300               | 0.338               | 0.247             | 0.614     | 0.295     |

## Speed And Cost

| method       | extra_forward                  | feature_dim | projection_probe_ms_per_sample | probe_fit_seconds    | total_cost_class | notes                                                                                                                        |
| ------------ | ------------------------------ | ----------- | ------------------------------ | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| margin-only  | 0                              | 1           | 0.0                            | 0.0                  | lowest           | No extra model pass; uses first-token logits.                                                                                |
| margin+top16 | 1                              | 17          | 4.197253535191218e-05          | 0.02948534581810236  | medium           | Requires cached or online blind-reference hidden state plus tiny projection/probe.                                           |
| margin+tail  | 1                              | 769         | 4.1588106089168126e-05         | 0.030178559012711048 | medium           | Requires blind-reference hidden state and 768D tail projection/probe.                                                        |
| margin+full  | 1                              | 4097        | 4.094808051983515e-05          | 0.030130027793347836 | medium           | Requires blind-reference hidden state and full 4096D diff probe.                                                             |
| tail-only    | 1                              | 768         | 0.011961261327895854           | 2.7909205006435513   | medium           | Geometry-only tail detector; no output margin.                                                                               |
| VCD/ICD      | extra distorted/blind decoding |             |                                |                      | high             | Downstream correction operator; substantially more expensive than a linear detector, so selective routing can be worthwhile. |

## Sanity Checks And Interpretation

- `task_a_fp_vs_tn` raw check: baseline `raw_diff` AUROC/AUPRC = 0.745/0.409; dimension `raw_full_diff_reference` = 0.745/0.409. They use the same StandardScaler + logistic grid protocol.
- `task_b_pred_yes_fp_vs_tp` raw check: baseline `raw_diff` AUROC/AUPRC = 0.685/0.524; dimension `raw_full_diff_reference` = 0.685/0.524. They use the same StandardScaler + logistic grid protocol.
- PLS transfer check: Task A `pls32_diff` train/calibration/test AUROC = 0.985/0.584/0.735; this indicates substantial strict-split domain shift.
- PLS deployment check: Task B `pls32_diff` train/calibration/test AUROC = 0.909/0.643/0.732; it transfers modestly but is not a stable strongest detector.
- `task_a_fp_vs_tn` top-SVD check: top-4 AUROC = 0.487, top-16 AUROC = 0.631. The precise claim should be that the dominant top-4 directions are weak, while useful signal can appear in slightly deeper early spectral coordinates.
- `task_b_pred_yes_fp_vs_tp` top-SVD check: top-4 AUROC = 0.570, top-16 AUROC = 0.697. The precise claim should be that the dominant top-4 directions are weak, while useful signal can appear in slightly deeper early spectral coordinates.
- Warning-vs-AUROC check: at the 20% predicted-Yes trigger target, `raw_diff` precision/FP recall = 0.292/0.519, while tail-only = 0.308/0.545. Fixed-trigger warning can look better than global AUROC because it evaluates only the top-risk slice.
- Recommended wording: margin/output confidence remains the strongest simple baseline; geometry-only is strict-transfer fragile, but selected spectral coordinates and margin+geometry provide complementary predicted-Yes warning signal.

## Artifact Audit

| layer | artifact             | n_samples | feature_dim | available | notes                                                                   |
| ----- | -------------------- | --------- | ----------- | --------- | ----------------------------------------------------------------------- |
| 24    | hidden_states        | 9000      | 4096        | True      | z_img, z_blind, and diff are available from cached hidden-state tensors |
| 24    | train_svd_basis      | 9000      | 1024        | True      | basis fitted on train subset only                                       |
| 24    | train_pca_diff_basis | 9000      | 256         | True      | centered PCA basis fitted on train subset only                          |
| 24    | tail_257_1024        | 9000      | 768         | True      | tail is unavailable when train SVD rank is smaller than tail_start      |
