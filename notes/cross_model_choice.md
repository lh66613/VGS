# Cross-Model Choice

## Current Decision

Use a **LLaVA-HF compatible additional checkpoint** for the first Stage O minimal replication.

Recommended default:

- `llava-hf/llava-1.5-13b-hf`, if available locally and GPU memory allows it.

Fallback:

- Any local LLaVA-HF compatible checkpoint that can be loaded by `transformers.LlavaForConditionalGeneration`.

## Reason

The current environment has `transformers==4.37.2`, which exposes the LLaVA-1.5 HF path used by the existing pipeline but does not expose `LlavaNextForConditionalGeneration` or `Qwen2VLForConditionalGeneration`. Running Qwen2-VL, InternVL, or LLaVA-NeXT would require dependency and model-wrapper changes before the actual mechanism replication can begin.

This choice prioritizes hook reliability and a minimal end-to-end cross-model check over architecture diversity.

## Limitation

This does **not** support a strong general-LVLM claim. If the LLaVA-compatible replication succeeds, the paper can say the pattern is not limited to one checkpoint, but it should not claim recurrence across different LVLM architectures.

## Prepared Scripts

GPU side:

- `scripts/run_gpu_stage_o_cross_model.sh`

CPU side:

- `scripts/run_cpu_stage_o_cross_model.sh`
- `scripts/build_stage_o_cross_model_summary.py`

Expected invocation:

```bash
MODEL_PATH=/path/to/llava-hf-compatible-checkpoint MODEL_ALIAS=llava_13b scripts/run_gpu_stage_o_cross_model.sh
MODEL_ALIAS=llava_13b scripts/run_cpu_stage_o_cross_model.sh
```

Optional smoke run:

```bash
MODEL_PATH=/path/to/checkpoint MODEL_ALIAS=stage_o_smoke MAX_SAMPLES=32 scripts/run_gpu_stage_o_cross_model.sh
MODEL_ALIAS=stage_o_smoke scripts/run_cpu_stage_o_cross_model.sh
```
