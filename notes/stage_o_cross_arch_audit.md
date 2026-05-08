# Stage O Cross-Architecture Audit

## Files

- `model_summary`: `outputs/stage_o_cross_model/audit/model_summary.csv`
- `probe_summary`: `outputs/stage_o_cross_model/audit/probe_summary.csv`
- `margin_summary`: `outputs/stage_o_cross_model/audit/margin_summary.csv`
- `condition_summary`: `outputs/stage_o_cross_model/audit/condition_summary.csv`
- `spectrum_summary`: `outputs/stage_o_cross_model/audit/spectrum_summary.csv`
- `diagnostics`: `outputs/stage_o_cross_model/audit/diagnostics.csv`

## Model Summary

- `internvl2_5_8b`: family `internvl2`, accuracy `0.903`, FP/TN/TP/FN/unk `290/4210/3921/579/0`, layers `20 24 32`, readout `last_prompt_token`
- `internvl2_8b`: family `internvl2`, accuracy `0.867`, FP/TN/TP/FN/unk `121/4379/3420/1080/0`, layers `20 24 32`, readout `last_prompt_token`
- `llava_13b`: family `None`, accuracy `0.871`, FP/TN/TP/FN/unk `468/4032/3804/696/0`, layers `20 24 32`, readout `last_prompt_token`
- `qwen2_5_vl_7b`: family `qwen2_5_vl`, accuracy `0.877`, FP/TN/TP/FN/unk `147/4353/3543/957/0`, layers `20 24 28`, readout `last_prompt_token`
- `qwen2_vl_7b`: family `qwen2_vl`, accuracy `0.875`, FP/TN/TP/FN/unk `158/4342/3534/966/0`, layers `20 24 28`, readout `last_prompt_token`

## Probe Snapshot

- `internvl2_5_8b`: difference `L32 k=full AUROC 1.000`, raw_blind `L32 k=full AUROC 0.694`, projected `L20 k=128 AUROC 1.000`
- `internvl2_8b`: difference `L24 k=full AUROC 1.000`, raw_blind `L20 k=full AUROC 0.661`, projected `L24 k=4 AUROC 0.999`
- `llava_13b`: difference `L20 k=full AUROC 0.736`, raw_blind `L32 k=full AUROC 0.693`, projected `L32 k=128 AUROC 0.699`
- `qwen2_5_vl_7b`: difference `L28 k=full AUROC 1.000`, raw_blind `L24 k=full AUROC 0.736`, projected `L28 k=256 AUROC 0.999`
- `qwen2_vl_7b`: difference `L24 k=full AUROC 1.000`, raw_blind `L20 k=full AUROC 0.688`, projected `L28 k=128 AUROC 0.999`

## Margin Baseline Snapshot

- `internvl2_5_8b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `internvl2_8b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `qwen2_5_vl_7b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)
- `qwen2_vl_7b`: best `yes_minus_no_logit` AUROC `1.000` (higher_means_fp)

## Condition Geometry Snapshot

- `internvl2_5_8b` adversarial tail deltas: L20 -0.04, L24 0.27, L32 -12.61
- `internvl2_8b` adversarial tail deltas: L20 -7.03, L24 -8.31, L32 -4.29
- `llava_13b` adversarial tail deltas: L20 18.13, L24 28.86, L32 75.50
- `qwen2_5_vl_7b` adversarial tail deltas: L20 2.18, L24 -14.52, L28 -2.17
- `qwen2_vl_7b` adversarial tail deltas: L20 0.63, L24 -2.92, L28 -1.39

## Diagnostics

- `internvl2_5_8b` `info` `basic_metadata_checks`: no basic metadata mismatch detected
- `internvl2_8b` `info` `basic_metadata_checks`: no basic metadata mismatch detected
- `llava_13b` `info` `basic_metadata_checks`: no basic metadata mismatch detected
- `qwen2_5_vl_7b` `info` `basic_metadata_checks`: no basic metadata mismatch detected
- `qwen2_vl_7b` `info` `basic_metadata_checks`: no basic metadata mismatch detected

## Initial Reading

- If the Qwen/InternVL rows have strong margin baselines but weak hidden probes, the model output is probably fine while the hidden readout is not architecture-equivalent.
- If both margin baselines and hidden probes collapse, first inspect yes/no tokenization and prompt formatting.
- If condition deltas flip sign only in full space but not in tail bands, treat the condition result as geometry-specific rather than a pipeline failure.

## Diagnosis

The cross-architecture files are mostly internally consistent: prediction counts, hidden-state sample counts, condition hidden summaries, layers, and readout metadata line up. I do not see a simple file-alignment failure in the artifacts.

The strange result is instead a **readout/label-definition confound**.

For Qwen-style prompts, the current `last_prompt_token` is the final assistant-generation token position:

```text
... Answer with yes or no only.<|im_end|>
<|im_start|>assistant\n
```

So the hidden state is taken at the position used to predict the next answer token. Since FP/TN is defined by whether the model answers yes/no on ground-truth-no samples, this readout is very close to the model's own output decision. That explains why:

- first-token margin AUROC is approximately 1.0;
- `raw_img` hidden probes reach approximately 1.0 AUROC;
- `difference = z_blind - z_img` also reaches approximately 1.0 AUROC;
- top-4/top-32 SVD coordinates become nearly perfect for Qwen/InternVL, unlike the original LLaVA-7B/13B story.

This is not necessarily a coding crash, but it means the current cross-architecture run is not measuring the same mechanism as the original blind-reference correction geometry experiment.

## Probe Sanity Check

Additional split-locked sanity checks are in:

- `outputs/stage_o_cross_model/audit/probe_sanity.csv`
- `notes/stage_o_probe_sanity.md`

They show that near-perfect separability persists under the fixed protocol split for Qwen/InternVL, so the issue is not just random train/test split leakage. However, the separability is already present in `raw_img`, which strongly supports the answer-position confound interpretation.

## What To Trust

Trust cautiously:

- model accuracies and FP/TN/TP/FN counts;
- the fact that Qwen/InternVL expose the next yes/no decision very strongly at the assistant-generation position;
- condition geometry as a model-specific diagnostic, not as a clean replication.

Do not use as cross-architecture mechanism evidence:

- near-1.0 FP/TN AUROC from `raw_img`, `difference`, or top-K SVD at the current readout;
- first-token margin AUROC as a baseline win;
- the claim that top-variance SVD directions replicate the LLaVA correction geometry story.

## Recommended Fix

Rerun cross-architecture hidden extraction with a readout that excludes assistant-generation prompt tokens, for example:

- `last_user_content_token`: the token before Qwen `<|im_end|>` / before the model-specific assistant marker;
- `last_user_content_4_mean`: mean over the last 4 user-content tokens only;
- optionally keep the current `last_prompt_token` as an answer-position diagnostic, not the main mechanism readout.

After that rerun:

- repeat the split-locked probe sanity check;
- require `raw_img` not to be trivially perfect before interpreting `difference`;
- treat margin baselines as output-decision diagnostics rather than independent mechanism baselines.
