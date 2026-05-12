# Stage U: Unified Cross-Model Minimal Protocol

This note consolidates the required third experiment into one reusable protocol.

## Files

- `mechanism_layer_metrics`: `outputs/stage_u_unified_minimal_protocol/mechanism_layer_metrics.csv`
- `mechanism_model_summary`: `outputs/stage_u_unified_minimal_protocol/mechanism_model_summary.csv`
- `deployment_model_summary`: `outputs/stage_u_unified_minimal_protocol/deployment_model_summary.csv`
- `deployment_gate_metrics`: `outputs/stage_u_unified_minimal_protocol/deployment_gate_metrics.csv`
- `layer_deployment_sweep`: `outputs/stage_u_unified_minimal_protocol/layer_deployment_sweep.csv`
- `failure_diagnosis`: `outputs/stage_u_unified_minimal_protocol/failure_diagnosis.csv`
- `predicted_yes_score_distributions`: `outputs/stage_u_unified_minimal_protocol/predicted_yes_score_distributions.csv`
- `shuffle_controls`: `outputs/stage_u_unified_minimal_protocol/shuffle_controls.csv`
- `missing_artifacts`: `outputs/stage_u_unified_minimal_protocol/missing_artifacts.csv`

## Mechanism Task: Variance vs Discrimination

| Model | Readout | Layer | Top-4 Var | Top-4 AUROC | Top-64 AUROC | Top-256 AUROC | Full Diff AUROC | Tail AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | `last_prompt_token` | 20 | 0.842 | 0.531 | 0.585 | 0.613 | 0.659 | 0.649 |
| LLaVA-1.5-13B | `last_prompt_token` | 20 | 0.752 | 0.600 | 0.671 | 0.698 | 0.744 | 0.766 |
| Qwen2-VL-7B | `last_user_content_token` | 20 | 0.645 | 0.528 | 0.526 | 0.583 | 0.612 | 0.529 |
| Qwen2.5-VL-7B | `last_user_content_token` | 24 | 0.685 | 0.597 | 0.643 | 0.612 | 0.749 | 0.742 |
| InternVL2-8B | `last_user_content_token` | 20 | 0.928 | 0.977 | 0.985 | 0.980 | 0.997 | 0.663 |
| InternVL2.5-8B | `last_user_content_token` | 32 | 0.878 | 0.993 | 0.995 | 0.994 | 0.998 | 0.734 |

Reading: LLaVA/Qwen retain the variance-discrimination decoupling pattern: top-4 directions explain large variance but are not the best discriminators. InternVL is the exception: FP/TN separability is already near-perfect in top coordinates, which is exactly why it needs the deployment diagnosis below.

## Deployment Task: Predicted-Yes FP vs TP

| Model | Layer | FP Base | Geometry AUROC | Low-Margin AUROC | Entropy AUROC | Low-Margin+Geometry AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | 20 | 0.089 | 0.656 | 0.867 | 0.867 | 0.873 |
| LLaVA-1.5-13B | 20 | 0.095 | 0.738 |  |  |  |
| Qwen2-VL-7B | 20 | 0.038 | 0.659 | 0.869 | 0.869 | 0.872 |
| Qwen2.5-VL-7B | 24 | 0.045 | 0.739 | 0.883 | 0.883 | 0.898 |
| InternVL2-8B | 20 | 0.034 | 0.249 | 0.883 | 0.883 | 0.771 |
| InternVL2.5-8B | 32 | 0.074 | 0.126 | 0.903 | 0.903 | 0.845 |

At the calibrated 20% predicted-Yes target rate:

| Model | Gate | Trigger | Precision FP | FP Recall | TP Damage |
| --- | --- | ---: | ---: | ---: | ---: |
| InternVL2-8B | `low_margin_plus_geometry` | 0.196 | 0.125 | 0.722 | 0.177 |
| InternVL2-8B | `same_trigger_random` | 0.196 | 0.033 | 0.192 | 0.196 |
| InternVL2.5-8B | `low_margin_plus_geometry` | 0.192 | 0.262 | 0.681 | 0.153 |
| InternVL2.5-8B | `same_trigger_random` | 0.192 | 0.072 | 0.186 | 0.192 |
| LLaVA-1.5-7B | `low_margin_plus_geometry` | 0.190 | 0.319 | 0.679 | 0.142 |
| LLaVA-1.5-7B | `same_trigger_random` | 0.190 | 0.087 | 0.186 | 0.190 |
| Qwen2-VL-7B | `low_margin_plus_geometry` | 0.206 | 0.142 | 0.762 | 0.184 |
| Qwen2-VL-7B | `same_trigger_random` | 0.206 | 0.036 | 0.195 | 0.207 |
| Qwen2.5-VL-7B | `low_margin_plus_geometry` | 0.181 | 0.170 | 0.680 | 0.157 |
| Qwen2.5-VL-7B | `same_trigger_random` | 0.181 | 0.046 | 0.182 | 0.181 |

## Failure-Mode Diagnosis

| Model | Readout | Layer | FP/TN Full | Pred-Yes Full | Pred-Yes Low-Margin | Corr(score, margin) | Flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| InternVL2-8B | `last_user_content_token` | 20 | 0.997 | 0.249 | 0.883 | 0.242 | `near_perfect_fp_tn_but_non_deployable` |
| InternVL2-8B question readout | `last_question_token` | 20 | 0.999 | 0.187 | 0.883 | -0.055 | `near_perfect_fp_tn_but_non_deployable` |
| InternVL2.5-8B | `last_user_content_token` | 32 | 0.998 | 0.126 | 0.903 | 0.238 | `near_perfect_fp_tn_but_non_deployable` |
| InternVL2.5-8B question readout | `last_question_token` | 24 | 0.999 | 0.100 | 0.903 | 0.111 | `near_perfect_fp_tn_but_non_deployable` |

Key boundary finding: Some architectures can exhibit near-perfect FP/TN internal separability that does not translate into deployable FP/TP risk detection.

InternVL predicted-Yes score distribution snapshot:

| Model | Readout | Outcome | N | Mean | Median | Q25 | Q75 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| InternVL2-8B | `last_user_content_token` | FP | 18 | 0.871 | 0.996 | 0.951 | 1.000 |
| InternVL2-8B | `last_user_content_token` | TP | 513 | 0.945 | 1.000 | 0.999 | 1.000 |
| InternVL2-8B question readout | `last_question_token` | FP | 18 | 0.967 | 1.000 | 1.000 | 1.000 |
| InternVL2-8B question readout | `last_question_token` | TP | 513 | 0.985 | 1.000 | 1.000 | 1.000 |
| InternVL2.5-8B | `last_user_content_token` | FP | 47 | 0.937 | 1.000 | 0.998 | 1.000 |
| InternVL2.5-8B | `last_user_content_token` | TP | 589 | 0.987 | 1.000 | 1.000 | 1.000 |
| InternVL2.5-8B question readout | `last_question_token` | FP | 47 | 0.959 | 1.000 | 0.999 | 1.000 |
| InternVL2.5-8B question readout | `last_question_token` | TP | 589 | 0.991 | 1.000 | 1.000 | 1.000 |

Shuffle controls show whether the signal requires matched blind/image pairing:

| Model | Readout | Layer | Feature | FP/TN AUROC | Pred-Yes AUROC |
| --- | --- | ---: | --- | ---: | ---: |
| InternVL2-8B | `last_user_content_token` | 20 | `blind_shuffle_difference` | 0.992 | 0.203 |
| InternVL2-8B | `last_user_content_token` | 20 | `image_shuffle_difference` | 0.470 | 0.546 |
| InternVL2-8B | `last_user_content_token` | 20 | `paired_difference` | 0.997 | 0.249 |
| InternVL2-8B | `last_user_content_token` | 24 | `blind_shuffle_difference` | 0.996 | 0.235 |
| InternVL2-8B | `last_user_content_token` | 24 | `image_shuffle_difference` | 0.548 | 0.600 |
| InternVL2-8B | `last_user_content_token` | 24 | `paired_difference` | 0.997 | 0.218 |
| InternVL2-8B | `last_user_content_token` | 32 | `blind_shuffle_difference` | 0.996 | 0.223 |
| InternVL2-8B | `last_user_content_token` | 32 | `image_shuffle_difference` | 0.454 | 0.483 |
| InternVL2-8B | `last_user_content_token` | 32 | `paired_difference` | 0.997 | 0.210 |
| InternVL2-8B question readout | `last_question_token` | 20 | `blind_shuffle_difference` | 0.999 | 0.182 |
| InternVL2-8B question readout | `last_question_token` | 20 | `image_shuffle_difference` | 0.446 | 0.528 |
| InternVL2-8B question readout | `last_question_token` | 20 | `paired_difference` | 0.999 | 0.187 |
| InternVL2-8B question readout | `last_question_token` | 24 | `blind_shuffle_difference` | 0.999 | 0.218 |
| InternVL2-8B question readout | `last_question_token` | 24 | `image_shuffle_difference` | 0.572 | 0.604 |
| InternVL2-8B question readout | `last_question_token` | 24 | `paired_difference` | 0.999 | 0.160 |
| InternVL2-8B question readout | `last_question_token` | 32 | `blind_shuffle_difference` | 0.997 | 0.183 |
| InternVL2-8B question readout | `last_question_token` | 32 | `image_shuffle_difference` | 0.470 | 0.549 |
| InternVL2-8B question readout | `last_question_token` | 32 | `paired_difference` | 0.998 | 0.165 |
| InternVL2.5-8B | `last_user_content_token` | 20 | `blind_shuffle_difference` | 0.998 | 0.112 |
| InternVL2.5-8B | `last_user_content_token` | 20 | `image_shuffle_difference` | 0.638 | 0.633 |
| InternVL2.5-8B | `last_user_content_token` | 20 | `paired_difference` | 0.998 | 0.110 |
| InternVL2.5-8B | `last_user_content_token` | 24 | `blind_shuffle_difference` | 0.998 | 0.118 |
| InternVL2.5-8B | `last_user_content_token` | 24 | `image_shuffle_difference` | 0.516 | 0.560 |
| InternVL2.5-8B | `last_user_content_token` | 24 | `paired_difference` | 0.997 | 0.117 |
| InternVL2.5-8B | `last_user_content_token` | 32 | `blind_shuffle_difference` | 0.998 | 0.134 |
| InternVL2.5-8B | `last_user_content_token` | 32 | `image_shuffle_difference` | 0.568 | 0.560 |
| InternVL2.5-8B | `last_user_content_token` | 32 | `paired_difference` | 0.998 | 0.126 |
| InternVL2.5-8B question readout | `last_question_token` | 20 | `blind_shuffle_difference` | 1.000 | 0.101 |
| InternVL2.5-8B question readout | `last_question_token` | 20 | `image_shuffle_difference` | 0.655 | 0.654 |
| InternVL2.5-8B question readout | `last_question_token` | 20 | `paired_difference` | 0.999 | 0.102 |
| InternVL2.5-8B question readout | `last_question_token` | 24 | `blind_shuffle_difference` | 0.999 | 0.101 |
| InternVL2.5-8B question readout | `last_question_token` | 24 | `image_shuffle_difference` | 0.516 | 0.552 |
| InternVL2.5-8B question readout | `last_question_token` | 24 | `paired_difference` | 0.999 | 0.100 |
| InternVL2.5-8B question readout | `last_question_token` | 32 | `blind_shuffle_difference` | 0.999 | 0.127 |
| InternVL2.5-8B question readout | `last_question_token` | 32 | `image_shuffle_difference` | 0.646 | 0.653 |
| InternVL2.5-8B question readout | `last_question_token` | 32 | `paired_difference` | 0.999 | 0.121 |

## Missing Artifacts

- `llava_13b`: margin logits unavailable: outputs/stage_o_cross_model/llava_13b/margins/pope_margin_scores.csv

## Recommended Paper Framing

- Use the mechanism table to claim cross-model recurrence of variance/discrimination decoupling where it actually holds.
- Use the deployment table to keep the practical claim honest: geometry is a complementary risk signal, not a universal replacement for output confidence.
- Use InternVL as a boundary discovery rather than a failed replication: near-perfect FP/TN separability can be real internally yet non-deployable for predicted-Yes FP/TP routing.
