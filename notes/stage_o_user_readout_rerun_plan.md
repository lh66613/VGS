# Stage O User-Content Readout Rerun Plan

## Reason

The previous Qwen/InternVL cross-architecture run used `last_prompt_token`, which lands at the assistant-generation prompt. That position can directly expose the next yes/no answer decision, producing near-perfect FP/TN probes from `raw_img`, `difference`, and top-K SVD coordinates.

The corrected run uses `last_user_content_token`, i.e. the final token of the user question/instruction before the model-specific assistant marker.

For Qwen-style chat templates this changes the readout from:

```text
<|im_start|>assistant\n
```

to the final user-content token:

```text
... Answer with yes or no only.
```

## Code Changes

- Added `last_user_content_token`.
- Added `last_user_content_4_mean`.
- Added `last_user_content_8_mean`.
- Qwen readout index is computed from the rendered chat template before `<|im_end|><|im_start|>assistant`.
- InternVL readout index is computed from the rendered conversation template before the assistant response slot.

## New Scripts

GPU:

- `scripts/run_gpu_phase3_cross_arch_user_readout.sh`

CPU:

- `scripts/run_cpu_phase3_cross_arch_user_readout.sh`

Audit after CPU:

- `scripts/run_cpu_phase3_cross_arch_user_readout_audit.sh`

All four cross-architecture models:

- `scripts/run_phase3_cross_arch_user_readout_all.sh`

Default output root:

- `outputs/stage_o_cross_model_user_readout/{MODEL_ALIAS}/`

## GPU Invocation

Run all four models:

```bash
PHASE3_STEP=gpu scripts/run_phase3_cross_arch_user_readout_all.sh
```

Run one model:

```bash
MODEL_FAMILY=qwen2_vl \
MODEL_ALIAS=qwen2_vl_7b \
MODEL_PATH=/data/lh/ModelandDataset/Qwen2-VL-7B-Instruct \
scripts/run_gpu_phase3_cross_arch_user_readout.sh
```

## CPU Invocation After GPU Completes

Run all four models:

```bash
PHASE3_STEP=cpu scripts/run_phase3_cross_arch_user_readout_all.sh
```

Run one model:

```bash
MODEL_FAMILY=qwen2_vl \
MODEL_ALIAS=qwen2_vl_7b \
scripts/run_cpu_phase3_cross_arch_user_readout.sh
```

## Audit Invocation After CPU Completes

```bash
scripts/run_cpu_phase3_cross_arch_user_readout_audit.sh
```

## Expected Sanity Criteria

The corrected run is more credible if:

- `raw_img` FP/TN AUROC is no longer approximately 1.0.
- top-4 projected-difference AUROC is no longer approximately 1.0.
- first-token margin remains strong, but is treated as an output-decision diagnostic rather than independent mechanism evidence.
- `difference` improves over raw states without trivially matching the margin baseline.

If `raw_img` remains near-perfect, the next suspect is model-specific prompt/content indexing or a dataset/output shortcut rather than blind-reference correction geometry.
