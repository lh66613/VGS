# Detector Follow-up Summary

## Files

- `protocol_summary`: `outputs/detector_followup/protocol_replication_summary.csv`
- `bootstrap_main_table`: `outputs/detector_followup/bootstrap_main_table.csv`
- `trigger_curve_table`: `outputs/detector_followup/trigger_curve_table.csv`
- `speed_cost_table`: `outputs/detector_followup/speed_cost_table.csv`
- `amber_geometry_external_summary`: `outputs/detector_followup/amber_geometry_external_summary.csv`
- `amber_margin_external_summary`: `outputs/detector_followup/amber_margin_external_summary.csv`

## Protocol Replication

| protocol                     | method                     | feature_dim | test_auroc | test_auprc | f1    | mcc   | warning_precision_20pct | fp_recall_20pct | tp_damage_20pct |
| ---------------------------- | -------------------------- | ----------- | ---------- | ---------- | ----- | ----- | ----------------------- | --------------- | --------------- |
| random->popular->adversarial | yes_no_margin              | 1           | 0.848      | 0.401      | 0.495 | 0.409 | 0.404                   | 0.667           | 0.155           |
| random->popular->adversarial | margin_plus_top16_svd_diff | 17          | 0.853      | 0.447      | 0.452 | 0.367 | 0.389                   | 0.624           | 0.154           |
| random->popular->adversarial | margin_plus_tail_diff      | 769         | 0.881      | 0.609      | 0.460 | 0.468 | 0.437                   | 0.698           | 0.141           |
| random->popular->adversarial | margin_plus_full_diff      | 4097        | 0.884      | 0.612      | 0.472 | 0.478 | 0.436                   | 0.698           | 0.142           |
| popular->random->adversarial | yes_no_margin              | 1           | 0.848      | 0.401      | 0.489 | 0.402 | 0.378                   | 0.693           | 0.180           |
| popular->random->adversarial | margin_plus_top16_svd_diff | 17          | 0.860      | 0.474      | 0.467 | 0.387 | 0.384                   | 0.698           | 0.176           |
| popular->random->adversarial | margin_plus_tail_diff      | 769         | 0.867      | 0.613      | 0.523 | 0.481 | 0.389                   | 0.725           | 0.179           |
| popular->random->adversarial | margin_plus_full_diff      | 4097        | 0.856      | 0.623      | 0.506 | 0.532 | 0.367                   | 0.698           | 0.190           |

## Bootstrap Main Table

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

## Trigger Curve

| method       | target_trigger_rate | actual_trigger_rate | warning_precision | fp_recall | tp_damage | source_method                    |
| ------------ | ------------------- | ------------------- | ----------------- | --------- | --------- | -------------------------------- |
| random       | 0.100               | 0.105               | 0.136             | 0.105     | 0.105     | same_trigger_as_margin_plus_tail |
| margin-only  | 0.100               | 0.111               | 0.442             | 0.360     | 0.072     | yes_no_margin                    |
| margin+top16 | 0.100               | 0.110               | 0.458             | 0.370     | 0.069     | margin_plus_top16_svd_diff       |
| margin+tail  | 0.100               | 0.105               | 0.575             | 0.444     | 0.052     | margin_plus_tail_diff            |
| margin+full  | 0.100               | 0.105               | 0.582             | 0.450     | 0.051     | margin_plus_full_diff            |
| tail-only    | 0.100               | 0.094               | 0.481             | 0.333     | 0.057     | tail_257_1024_diff               |
| random       | 0.200               | 0.217               | 0.137             | 0.218     | 0.217     | same_trigger_as_margin_plus_tail |
| margin-only  | 0.200               | 0.224               | 0.404             | 0.667     | 0.155     | yes_no_margin                    |
| margin+top16 | 0.200               | 0.218               | 0.389             | 0.624     | 0.154     | margin_plus_top16_svd_diff       |
| margin+tail  | 0.200               | 0.217               | 0.437             | 0.698     | 0.141     | margin_plus_tail_diff            |
| margin+full  | 0.200               | 0.218               | 0.436             | 0.698     | 0.142     | margin_plus_full_diff            |
| tail-only    | 0.200               | 0.191               | 0.268             | 0.376     | 0.161     | tail_257_1024_diff               |
| random       | 0.300               | 0.326               | 0.137             | 0.328     | 0.326     | same_trigger_as_margin_plus_tail |
| margin-only  | 0.300               | 0.329               | 0.333             | 0.804     | 0.254     | yes_no_margin                    |
| margin+top16 | 0.300               | 0.329               | 0.343             | 0.831     | 0.250     | margin_plus_top16_svd_diff       |
| margin+tail  | 0.300               | 0.326               | 0.350             | 0.841     | 0.245     | margin_plus_tail_diff            |
| margin+full  | 0.300               | 0.329               | 0.350             | 0.847     | 0.247     | margin_plus_full_diff            |
| tail-only    | 0.300               | 0.288               | 0.198             | 0.418     | 0.267     | tail_257_1024_diff               |

## Speed And Cost

| method       | extra_forward                  | feature_dim | projection_probe_ms_per_sample | probe_fit_seconds | total_cost_class | notes                                                                                                                        |
| ------------ | ------------------------------ | ----------- | ------------------------------ | ----------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| margin-only  | 0                              | 1.000       | 0.000                          | 0.000             | lowest           | No extra model pass; uses first-token logits.                                                                                |
| margin+top16 | 1                              | 17.000      | 0.000                          | 0.030             | medium           | Requires cached or online blind-reference hidden state plus tiny projection/probe.                                           |
| margin+tail  | 1                              | 769.000     | 0.000                          | 0.030             | medium           | Requires blind-reference hidden state and 768D tail projection/probe.                                                        |
| margin+full  | 1                              | 4097.000    | 0.000                          | 0.031             | medium           | Requires blind-reference hidden state and full 4096D diff probe.                                                             |
| tail-only    | 1                              | 768.000     | 0.012                          | 1.927             | medium           | Geometry-only tail detector; no output margin.                                                                               |
| VCD/ICD      | extra distorted/blind decoding |             |                                |                   | high             | Downstream correction operator; substantially more expensive than a linear detector, so selective routing can be worthwhile. |

## AMBER External Transfer

| dataset | comparison_scope                         | method         | score               | target_trigger_rate | actual_trigger_rate | warning_precision | fp_recall | tp_damage | notes                                                                                                                            |
| ------- | ---------------------------------------- | -------------- | ------------------- | ------------------- | ------------------- | ----------------- | --------- | --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| AMBER   | geometry-only; margin logits unavailable | FullD geometry | full_probe          | 0.200               | 0.200               | 0.288             | 0.203     | 0.199     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | FullD geometry | full_probe          | 0.300               | 0.300               | 0.294             | 0.311     | 0.296     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | Top-4 geometry | top_4_probe         | 0.200               | 0.200               | 0.147             | 0.104     | 0.238     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | Top-4 geometry | top_4_probe         | 0.300               | 0.300               | 0.161             | 0.170     | 0.352     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | Tail geometry  | tail_257_1024_probe | 0.200               | 0.200               | 0.211             | 0.149     | 0.221     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | Tail geometry  | tail_257_1024_probe | 0.300               | 0.300               | 0.224             | 0.237     | 0.325     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | PLS geometry   | pls32_probe         | 0.200               | 0.200               | 0.300             | 0.211     | 0.196     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |
| AMBER   | geometry-only; margin logits unavailable | PLS geometry   | pls32_probe         | 0.300               | 0.300               | 0.297             | 0.313     | 0.295     | AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv. |

## AMBER Margin Deployment

| dataset | policy            | method       | score                               | target_trigger_rate | actual_trigger_rate | warning_precision | relative_precision_gain | fp_recall | tp_damage | base_pred_yes_fp_rate |
| ------- | ----------------- | ------------ | ----------------------------------- | ------------------- | ------------------- | ----------------- | ----------------------- | --------- | --------- | --------------------- |
| AMBER   | external_top_rate | margin-only  | low_margin_probe                    | 0.200               | 0.200               | 0.413             | 1.454                   | 0.291     | 0.164     | 0.284                 |
| AMBER   | external_top_rate | margin+tail  | low_margin_plus_tail_257_1024_probe | 0.200               | 0.200               | 0.319             | 1.125                   | 0.225     | 0.190     | 0.284                 |
| AMBER   | external_top_rate | margin+full  | low_margin_plus_full_probe          | 0.200               | 0.200               | 0.262             | 0.922                   | 0.184     | 0.206     | 0.284                 |
| AMBER   | external_top_rate | margin+top16 | low_margin_plus_top_16_probe        | 0.200               | 0.200               | 0.237             | 0.834                   | 0.167     | 0.213     | 0.284                 |
| AMBER   | external_top_rate | tail-only    | tail_257_1024_probe                 | 0.200               | 0.200               | 0.217             | 0.764                   | 0.153     | 0.219     | 0.284                 |
| AMBER   | external_top_rate | full-only    | full_probe                          | 0.200               | 0.200               | 0.172             | 0.606                   | 0.121     | 0.231     | 0.284                 |
| AMBER   | external_top_rate | top16-only   | top_16_probe                        | 0.200               | 0.200               | 0.154             | 0.543                   | 0.109     | 0.236     | 0.284                 |
| AMBER   | external_top_rate | margin-only  | low_margin_probe                    | 0.300               | 0.300               | 0.404             | 1.423                   | 0.427     | 0.250     | 0.284                 |
| AMBER   | external_top_rate | margin+tail  | low_margin_plus_tail_257_1024_probe | 0.300               | 0.300               | 0.338             | 1.189                   | 0.357     | 0.278     | 0.284                 |
| AMBER   | external_top_rate | margin+full  | low_margin_plus_full_probe          | 0.300               | 0.300               | 0.313             | 1.103                   | 0.331     | 0.288     | 0.284                 |
| AMBER   | external_top_rate | margin+top16 | low_margin_plus_top_16_probe        | 0.300               | 0.300               | 0.271             | 0.953                   | 0.286     | 0.306     | 0.284                 |
| AMBER   | external_top_rate | tail-only    | tail_257_1024_probe                 | 0.300               | 0.300               | 0.241             | 0.848                   | 0.255     | 0.318     | 0.284                 |
| AMBER   | external_top_rate | top16-only   | top_16_probe                        | 0.300               | 0.300               | 0.212             | 0.748                   | 0.224     | 0.330     | 0.284                 |
| AMBER   | external_top_rate | full-only    | full_probe                          | 0.300               | 0.300               | 0.207             | 0.729                   | 0.219     | 0.332     | 0.284                 |

## Remaining AMBER Step

AMBER margin-based deployment has been run if `amber_margin_external_summary` is populated. Re-run `bash scripts/run_gpu_detector_amber_deployment.sh` only when regenerating AMBER first-token margins or changing score sets.

## Reading

- Strict split supports margin+tail/full over margin-only with positive bootstrap CIs for AUROC, AUPRC, warning precision, and lower TP damage.
- Reverse split reproduces the broad benefit in AUPRC and ranking, but warning precision gains are weaker and not uniformly significant.
- AMBER margin transfer is now available. Low-margin is the strongest external warning signal; adding POPE-trained geometry reduces AMBER warning precision under fixed external trigger budgets, so report geometry external transfer as modest and not robust.
