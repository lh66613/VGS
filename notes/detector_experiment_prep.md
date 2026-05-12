# Detector Experiment Prep

## Protocol

- Strict subset-transfer: train `random`, calibrate `popular`, test `adversarial`.
- Task A: FP vs TN on ground-truth No samples.
- Task B: predicted-Yes FP vs TP deployment risk.
- Threshold policy for classification metrics: F1-optimal threshold selected on calibration only.

## Files

- `baseline_table`: `outputs/detector_minimal_package/detector_baseline_table.csv`
- `deployment_warning`: `outputs/detector_minimal_package/deployment_warning.csv`
- `threshold_audit`: `outputs/detector_minimal_package/threshold_audit.csv`
- `dimension_curve`: `outputs/detector_minimal_package/dimension_curve.csv`
- `spectral_band_curve`: `outputs/detector_minimal_package/spectral_band_curve.csv`
- `pls_diagnostics`: `outputs/detector_minimal_package/pls_diagnostics.csv`
- `bootstrap_comparisons`: `outputs/detector_minimal_package/bootstrap_comparisons.csv`
- `bootstrap_main_table`: `outputs/detector_minimal_package/bootstrap_main_table.csv`
- `trigger_curve_table`: `outputs/detector_minimal_package/trigger_curve_table.csv`
- `speed_cost_table`: `outputs/detector_minimal_package/speed_cost_table.csv`
- `feature_audit`: `outputs/detector_minimal_package/feature_audit.csv`

## Best Test Rows

| task                     | layer | method                      | feature_dim | test_auroc | test_auprc | f1    | mcc   | detector_score_ms_per_sample |
| ------------------------ | ----- | --------------------------- | ----------- | ---------- | ---------- | ----- | ----- | ---------------------------- |
| task_a_fp_vs_tn          | 24    | output_logistic             | 8           | 1.000      | 1.000      | 0.997 | 0.997 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top16_svd_diff  | 17          | 1.000      | 1.000      | 0.997 | 0.997 | 0.000                        |
| task_a_fp_vs_tn          | 24    | yes_no_margin               | 1           | 1.000      | 1.000      | 1.000 | 1.000 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top4_svd_diff   | 5           | 1.000      | 1.000      | 0.989 | 0.988 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top256_svd_diff | 257         | 0.999      | 0.995      | 0.946 | 0.939 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_top64_svd_diff  | 65          | 0.999      | 0.994      | 0.952 | 0.946 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_pls32_diff      | 33          | 0.999      | 0.990      | 0.951 | 0.944 | 0.000                        |
| task_a_fp_vs_tn          | 24    | margin_plus_full_diff       | 4097        | 0.993      | 0.937      | 0.909 | 0.896 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_full_diff       | 4097        | 0.884      | 0.612      | 0.472 | 0.478 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_tail_diff       | 769         | 0.881      | 0.609      | 0.460 | 0.468 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top16_svd_diff  | 17          | 0.853      | 0.447      | 0.452 | 0.367 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | output_logistic             | 8           | 0.850      | 0.428      | 0.495 | 0.409 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top4_svd_diff   | 5           | 0.849      | 0.392      | 0.498 | 0.412 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | yes_no_margin               | 1           | 0.848      | 0.401      | 0.495 | 0.409 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | binary_entropy              | 1           | 0.848      | 0.401      | 0.495 | 0.409 | 0.000                        |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_top256_svd_diff | 257         | 0.831      | 0.427      | 0.463 | 0.370 | 0.000                        |

## Deployment Warning Snapshot

| layer | method                      | trigger_rate | warning_precision | relative_precision_gain | fp_recall | tp_damage |
| ----- | --------------------------- | ------------ | ----------------- | ----------------------- | --------- | --------- |
| 24    | margin_plus_tail_diff       | 0.217        | 0.437             | 3.217                   | 0.698     | 0.141     |
| 24    | margin_plus_full_diff       | 0.218        | 0.436             | 3.206                   | 0.698     | 0.142     |
| 24    | output_logistic             | 0.223        | 0.406             | 2.991                   | 0.667     | 0.153     |
| 24    | binary_entropy              | 0.224        | 0.404             | 2.972                   | 0.667     | 0.155     |
| 24    | yes_no_margin               | 0.224        | 0.404             | 2.972                   | 0.667     | 0.155     |
| 24    | margin_plus_top4_svd_diff   | 0.220        | 0.402             | 2.958                   | 0.651     | 0.152     |
| 24    | margin_plus_top16_svd_diff  | 0.218        | 0.389             | 2.866                   | 0.624     | 0.154     |
| 24    | margin_plus_top64_svd_diff  | 0.217        | 0.377             | 2.778                   | 0.603     | 0.156     |
| 24    | margin_plus_top256_svd_diff | 0.212        | 0.369             | 2.719                   | 0.577     | 0.155     |
| 24    | margin_plus_pls32_diff      | 0.217        | 0.361             | 2.656                   | 0.577     | 0.161     |
| 24    | raw_concat                  | 0.199        | 0.296             | 2.179                   | 0.434     | 0.162     |
| 24    | raw_diff                    | 0.198        | 0.293             | 2.160                   | 0.429     | 0.162     |

## Subspace Dimension Curve

| task                     | layer | method                  | feature_dim | test_auroc | test_auprc |
| ------------------------ | ----- | ----------------------- | ----------- | ---------- | ---------- |
| task_a_fp_vs_tn          | 24    | top_svd                 | 16          | 0.610      | 0.195      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 16          | 0.593      | 0.174      |
| task_a_fp_vs_tn          | 24    | pls                     | 256         | 0.573      | 0.174      |
| task_a_fp_vs_tn          | 24    | pls                     | 16          | 0.569      | 0.172      |
| task_a_fp_vs_tn          | 24    | pls                     | 64          | 0.567      | 0.171      |
| task_a_fp_vs_tn          | 24    | pls                     | 4           | 0.564      | 0.171      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 256         | 0.559      | 0.165      |
| task_a_fp_vs_tn          | 24    | raw_full_diff_reference | 4096        | 0.546      | 0.167      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 256         | 0.546      | 0.167      |
| task_a_fp_vs_tn          | 24    | random                  | 256         | 0.544      | 0.147      |
| task_a_fp_vs_tn          | 24    | random                  | 64          | 0.532      | 0.141      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 64          | 0.531      | 0.144      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 64          | 0.527      | 0.143      |
| task_a_fp_vs_tn          | 24    | tail                    | 768         | 0.519      | 0.136      |
| task_a_fp_vs_tn          | 24    | pca_diff                | 4           | 0.501      | 0.128      |
| task_a_fp_vs_tn          | 24    | random                  | 16          | 0.489      | 0.123      |
| task_a_fp_vs_tn          | 24    | top_svd                 | 4           | 0.489      | 0.120      |
| task_a_fp_vs_tn          | 24    | random                  | 4           | 0.481      | 0.123      |
| task_b_pred_yes_fp_vs_tp | 24    | top_svd                 | 16          | 0.677      | 0.229      |
| task_b_pred_yes_fp_vs_tp | 24    | pls                     | 16          | 0.666      | 0.247      |

## Spectral Band Curve

| task                     | layer | mode                         | spectral_feature | feature_dim | test_auroc | test_auprc | warning_precision | fp_recall | tp_damage |
| ------------------------ | ----- | ---------------------------- | ---------------- | ----------- | ---------- | ---------- | ----------------- | --------- | --------- |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_5_16        | 12          | 0.648      | 0.214      | 0.206             | 0.312     | 0.190     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | top_1_4          | 4           | 0.580      | 0.158      | 0.124             | 0.180     | 0.200     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | tail_257_1024    | 768         | 0.554      | 0.409      | 0.268             | 0.376     | 0.161     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_65_256      | 192         | 0.516      | 0.157      | 0.152             | 0.228     | 0.199     |
| task_b_pred_yes_fp_vs_tp | 24    | band_only                    | band_17_64       | 48          | 0.509      | 0.142      | 0.122             | 0.180     | 0.204     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_16         | 16          | 0.677      | 0.229      | 0.235             | 0.360     | 0.184     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_64         | 64          | 0.599      | 0.175      | 0.179             | 0.275     | 0.198     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_1024       | 1024        | 0.588      | 0.478      | 0.318             | 0.466     | 0.157     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_4          | 4           | 0.580      | 0.158      | 0.124             | 0.180     | 0.200     |
| task_b_pred_yes_fp_vs_tp | 24    | cumulative_top_k             | top_1_256        | 256         | 0.519      | 0.222      | 0.149             | 0.217     | 0.195     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | tail_257_1024    | 769         | 0.881      | 0.609      | 0.437             | 0.698     | 0.141     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_5_16        | 13          | 0.852      | 0.462      | 0.402             | 0.651     | 0.152     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | top_1_4          | 5           | 0.849      | 0.392      | 0.402             | 0.651     | 0.152     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_17_64       | 49          | 0.822      | 0.370      | 0.363             | 0.582     | 0.161     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_band_only        | band_65_256      | 193         | 0.820      | 0.385      | 0.383             | 0.624     | 0.158     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_1024       | 1025        | 0.894      | 0.650      | 0.446             | 0.725     | 0.141     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_16         | 17          | 0.853      | 0.447      | 0.389             | 0.624     | 0.154     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_4          | 5           | 0.849      | 0.392      | 0.402             | 0.651     | 0.152     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_256        | 257         | 0.831      | 0.427      | 0.369             | 0.577     | 0.155     |
| task_b_pred_yes_fp_vs_tp | 24    | margin_plus_cumulative_top_k | top_1_64         | 65          | 0.830      | 0.382      | 0.377             | 0.603     | 0.156     |

## PLS Diagnostics

| task                     | layer | k   | train_auroc | calibration_auroc | test_auroc | split_half_overlap | overlap_top16 | overlap_tail_257_1024 |
| ------------------------ | ----- | --- | ----------- | ----------------- | ---------- | ------------------ | ------------- | --------------------- |
| task_a_fp_vs_tn          | 24    | 4   | 0.960       | 0.603             | 0.564      | 0.354              | 0.477         | 0.129                 |
| task_a_fp_vs_tn          | 24    | 16  | 0.969       | 0.595             | 0.569      | 0.429              | 0.470         | 0.121                 |
| task_a_fp_vs_tn          | 24    | 32  | 0.969       | 0.598             | 0.569      | 0.429              | 0.343         | 0.126                 |
| task_a_fp_vs_tn          | 24    | 64  | 0.969       | 0.597             | 0.567      | 0.420              | 0.183         | 0.145                 |
| task_a_fp_vs_tn          | 24    | 256 | 0.972       | 0.584             | 0.573      | 0.429              | 0.048         | 0.252                 |
| task_b_pred_yes_fp_vs_tp | 24    | 4   | 0.905       | 0.647             | 0.655      | 0.326              | 0.595         | 0.102                 |
| task_b_pred_yes_fp_vs_tp | 24    | 16  | 0.982       | 0.658             | 0.666      | 0.408              | 0.486         | 0.107                 |
| task_b_pred_yes_fp_vs_tp | 24    | 32  | 0.909       | 0.648             | 0.657      | 0.415              | 0.360         | 0.123                 |
| task_b_pred_yes_fp_vs_tp | 24    | 64  | 0.909       | 0.647             | 0.657      | 0.406              | 0.188         | 0.146                 |
| task_b_pred_yes_fp_vs_tp | 24    | 256 | 0.911       | 0.635             | 0.650      | 0.418              | 0.049         | 0.266                 |

## Bootstrap Comparisons

| comparison                | metric            | delta  | ci95             | significant |
| ------------------------- | ----------------- | ------ | ---------------- | ----------- |
| margin+tail - margin      | AUROC             | 0.033  | [0.017, 0.052]   | yes         |
| margin+tail - margin      | AUPRC             | 0.204  | [0.148, 0.259]   | yes         |
| margin+tail - margin      | Warning Precision | 0.033  | [0.006, 0.061]   | yes         |
| margin+tail - margin      | FP Recall         | 0.032  | [-0.021, 0.089]  | no          |
| margin+tail - margin      | TP Damage         | -0.013 | [-0.025, -0.002] | better      |
| margin+full - margin      | AUROC             | 0.037  | [0.019, 0.057]   | yes         |
| margin+full - margin      | AUPRC             | 0.208  | [0.149, 0.265]   | yes         |
| margin+full - margin      | Warning Precision | 0.031  | [0.003, 0.061]   | yes         |
| margin+full - margin      | FP Recall         | 0.032  | [-0.023, 0.091]  | no          |
| margin+full - margin      | TP Damage         | -0.012 | [-0.026, -0.002] | better      |
| margin+tail - margin+full | AUROC             | -0.004 | [-0.014, 0.007]  | no          |
| margin+tail - margin+full | AUPRC             | -0.003 | [-0.037, 0.034]  | no          |
| margin+tail - margin+full | Warning Precision | 0.002  | [-0.017, 0.021]  | no          |
| margin+tail - margin+full | FP Recall         | 0.000  | [-0.037, 0.039]  | no          |
| margin+tail - margin+full | TP Damage         | -0.001 | [-0.008, 0.006]  | no          |
| margin+top16 - margin     | AUROC             | 0.005  | [-0.001, 0.011]  | no          |
| margin+top16 - margin     | AUPRC             | 0.043  | [0.006, 0.082]   | yes         |
| margin+top16 - margin     | Warning Precision | -0.015 | [-0.036, 0.005]  | no          |
| margin+top16 - margin     | FP Recall         | -0.043 | [-0.084, -0.005] | worse       |
| margin+top16 - margin     | TP Damage         | -0.001 | [-0.011, 0.008]  | no          |
| margin+tail - raw diff    | AUROC             | 0.318  | [0.262, 0.368]   | yes         |
| margin+tail - raw diff    | AUPRC             | 0.175  | [0.121, 0.231]   | yes         |
| margin+tail - raw diff    | Warning Precision | 0.143  | [0.091, 0.195]   | yes         |
| margin+tail - raw diff    | FP Recall         | 0.270  | [0.192, 0.347]   | yes         |
| margin+tail - raw diff    | TP Damage         | -0.021 | [-0.047, 0.005]  | no          |

## Trigger Curve Table

| method       | target_trigger_rate | actual_trigger_rate | warning_precision | fp_recall | tp_damage |
| ------------ | ------------------- | ------------------- | ----------------- | --------- | --------- |
| random       | 0.100               | 0.105               | 0.136             | 0.105     | 0.105     |
| margin-only  | 0.100               | 0.111               | 0.442             | 0.360     | 0.072     |
| margin+top16 | 0.100               | 0.110               | 0.458             | 0.370     | 0.069     |
| margin+tail  | 0.100               | 0.105               | 0.575             | 0.444     | 0.052     |
| margin+full  | 0.100               | 0.105               | 0.582             | 0.450     | 0.051     |
| tail-only    | 0.100               | 0.094               | 0.481             | 0.333     | 0.057     |
| random       | 0.200               | 0.217               | 0.137             | 0.218     | 0.217     |
| margin-only  | 0.200               | 0.224               | 0.404             | 0.667     | 0.155     |
| margin+top16 | 0.200               | 0.218               | 0.389             | 0.624     | 0.154     |
| margin+tail  | 0.200               | 0.217               | 0.437             | 0.698     | 0.141     |
| margin+full  | 0.200               | 0.218               | 0.436             | 0.698     | 0.142     |
| tail-only    | 0.200               | 0.191               | 0.268             | 0.376     | 0.161     |
| random       | 0.300               | 0.326               | 0.137             | 0.328     | 0.326     |
| margin-only  | 0.300               | 0.329               | 0.333             | 0.804     | 0.254     |
| margin+top16 | 0.300               | 0.329               | 0.343             | 0.831     | 0.250     |
| margin+tail  | 0.300               | 0.326               | 0.350             | 0.841     | 0.245     |
| margin+full  | 0.300               | 0.329               | 0.350             | 0.847     | 0.247     |
| tail-only    | 0.300               | 0.288               | 0.198             | 0.418     | 0.267     |

## Speed And Cost

| method       | extra_forward                  | feature_dim | projection_probe_ms_per_sample | probe_fit_seconds    | total_cost_class | notes                                                                                                                        |
| ------------ | ------------------------------ | ----------- | ------------------------------ | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| margin-only  | 0                              | 1           | 0.0                            | 0.0                  | lowest           | No extra model pass; uses first-token logits.                                                                                |
| margin+top16 | 1                              | 17          | 4.154598961273829e-05          | 0.03029770404100418  | medium           | Requires cached or online blind-reference hidden state plus tiny projection/probe.                                           |
| margin+tail  | 1                              | 769         | 4.063608745733897e-05          | 0.030475680716335773 | medium           | Requires blind-reference hidden state and 768D tail projection/probe.                                                        |
| margin+full  | 1                              | 4097        | 4.092283133003447e-05          | 0.03063095733523369  | medium           | Requires blind-reference hidden state and full 4096D diff probe.                                                             |
| tail-only    | 1                              | 768         | 0.011865365836355422           | 1.9271575575694442   | medium           | Geometry-only tail detector; no output margin.                                                                               |
| VCD/ICD      | extra distorted/blind decoding |             |                                |                      | high             | Downstream correction operator; substantially more expensive than a linear detector, so selective routing can be worthwhile. |

## Sanity Checks And Interpretation

- `task_a_fp_vs_tn` raw check: baseline `raw_diff` AUROC/AUPRC = 0.546/0.167; dimension `raw_full_diff_reference` = 0.546/0.167. They use the same StandardScaler + logistic grid protocol.
- `task_b_pred_yes_fp_vs_tp` raw check: baseline `raw_diff` AUROC/AUPRC = 0.563/0.434; dimension `raw_full_diff_reference` = 0.563/0.434. They use the same StandardScaler + logistic grid protocol.
- PLS transfer check: Task A `pls32_diff` train/calibration/test AUROC = 0.969/0.598/0.569; this indicates substantial strict-split domain shift.
- PLS deployment check: Task B `pls32_diff` train/calibration/test AUROC = 0.909/0.648/0.657; it transfers modestly but is not a stable strongest detector.
- `task_a_fp_vs_tn` top-SVD check: top-4 AUROC = 0.489, top-16 AUROC = 0.610. The precise claim should be that the dominant top-4 directions are weak, while useful signal can appear in slightly deeper early spectral coordinates.
- `task_b_pred_yes_fp_vs_tp` top-SVD check: top-4 AUROC = 0.580, top-16 AUROC = 0.677. The precise claim should be that the dominant top-4 directions are weak, while useful signal can appear in slightly deeper early spectral coordinates.
- Warning-vs-AUROC check: at the 20% predicted-Yes trigger target, `raw_diff` precision/FP recall = 0.293/0.429, while tail-only = 0.268/0.376. Fixed-trigger warning can look better than global AUROC because it evaluates only the top-risk slice.
- Recommended wording: margin/output confidence remains the strongest simple baseline; geometry-only is strict-transfer fragile, but selected spectral coordinates and margin+geometry provide complementary predicted-Yes warning signal.

## Artifact Audit

| layer | artifact             | n_samples | feature_dim | available | notes                                                                   |
| ----- | -------------------- | --------- | ----------- | --------- | ----------------------------------------------------------------------- |
| 24    | hidden_states        | 9000      | 4096        | True      | z_img, z_blind, and diff are available from cached hidden-state tensors |
| 24    | train_svd_basis      | 9000      | 1024        | True      | basis fitted on train subset only                                       |
| 24    | train_pca_diff_basis | 9000      | 256         | True      | centered PCA basis fitted on train subset only                          |
| 24    | tail_257_1024        | 9000      | 768         | True      | tail is unavailable when train SVD rank is smaller than tail_start      |
