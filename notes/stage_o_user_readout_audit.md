# Stage O Cross-Architecture Audit

## Files

- `model_summary`: `outputs/stage_o_cross_model_user_readout/audit/model_summary.csv`
- `probe_summary`: `outputs/stage_o_cross_model_user_readout/audit/probe_summary.csv`
- `margin_summary`: `outputs/stage_o_cross_model_user_readout/audit/margin_summary.csv`
- `condition_summary`: `outputs/stage_o_cross_model_user_readout/audit/condition_summary.csv`
- `spectrum_summary`: `outputs/stage_o_cross_model_user_readout/audit/spectrum_summary.csv`
- `diagnostics`: `outputs/stage_o_cross_model_user_readout/audit/diagnostics.csv`

## Model Summary

- `internvl2_5_8b`: family `internvl2`, accuracy `0.903`, FP/TN/TP/FN/unk `290/4210/3921/579/0`, layers `20 24 32`, readout `last_user_content_token`
- `internvl2_8b`: family `internvl2`, accuracy `0.867`, FP/TN/TP/FN/unk `121/4379/3420/1080/0`, layers `20 24 32`, readout `last_user_content_token`
- `qwen2_5_vl_7b`: family `qwen2_5_vl`, accuracy `0.878`, FP/TN/TP/FN/unk `146/4354/3549/951/0`, layers `20 24 28`, readout `last_user_content_token`
- `qwen2_vl_7b`: family `qwen2_vl`, accuracy `0.873`, FP/TN/TP/FN/unk `160/4340/3516/984/0`, layers `20 24 28`, readout `last_user_content_token`

## Probe Snapshot

- `internvl2_5_8b`: difference `L32 k=full AUROC 0.998`, raw_blind `L24 k=full AUROC 0.693`, projected `L20 k=128 AUROC 0.992`
- `internvl2_8b`: difference `L32 k=full AUROC 0.999`, raw_blind `L24 k=full AUROC 0.725`, projected `L32 k=128 AUROC 0.998`
- `qwen2_5_vl_7b`: difference `L20 k=full AUROC 0.771`, raw_blind `L20 k=full AUROC 0.749`, projected `L20 k=256 AUROC 0.707`
- `qwen2_vl_7b`: difference `L24 k=full AUROC 0.772`, raw_blind `L20 k=full AUROC 0.675`, projected `L24 k=128 AUROC 0.672`

## Margin Baseline Snapshot

- `internvl2_5_8b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `internvl2_8b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `qwen2_5_vl_7b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `qwen2_vl_7b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)

## Condition Geometry Snapshot

- `internvl2_5_8b` adversarial tail deltas: L20 -0.15, L24 -0.82, L32 -16.64
- `internvl2_8b` adversarial tail deltas: L20 -0.27, L24 -2.39, L32 -17.72
- `qwen2_5_vl_7b` adversarial tail deltas: L20 7.80, L24 -6.59, L28 146.88
- `qwen2_vl_7b` adversarial tail deltas: L20 98.70, L24 236.12, L28 988.95

## Diagnostics

- `internvl2_5_8b` `medium` `nonstandard_readout`: readout_position=last_user_content_token
- `internvl2_8b` `medium` `nonstandard_readout`: readout_position=last_user_content_token
- `qwen2_5_vl_7b` `medium` `nonstandard_readout`: readout_position=last_user_content_token
- `qwen2_vl_7b` `medium` `nonstandard_readout`: readout_position=last_user_content_token

## Initial Reading

- If the Qwen/InternVL rows have strong margin baselines but weak hidden probes, the model output is probably fine while the hidden readout is not architecture-equivalent.
- If both margin baselines and hidden probes collapse, first inspect yes/no tokenization and prompt formatting.
- If condition deltas flip sign only in full space but not in tail bands, treat the condition result as geometry-specific rather than a pipeline failure.
