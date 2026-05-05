# Phase 3 Cross-Architecture Minimal Replication

目标：在非 LLaVA-family 模型上复现 Finding 1/2 的最小证据链，先不做跨架构 activation intervention。

## Models

默认脚本覆盖四个本地 checkpoint：

- `qwen2_vl_7b`: `/data/lh/ModelandDataset/Qwen2-VL-7B-Instruct`
- `qwen2_5_vl_7b`: `/data/lh/ModelandDataset/Qwen2.5-VL-7B-Instruct`
- `internvl2_8b`: `/data/lh/ModelandDataset/InternVL2-8B`
- `internvl2_5_8b`: `/data/lh/ModelandDataset/InternVL2_5-8B`

Qwen2-VL/Qwen2.5-VL 的语言层数是 28，默认层为 `20 24 28`。InternVL2/2.5-8B 的语言层数是 32，默认层为 `20 24 32`。

## Minimal Runs

单模型 GPU artifacts：

```bash
MODEL_FAMILY=qwen2_5_vl \
MODEL_ALIAS=qwen2_5_vl_7b \
MODEL_PATH=/data/lh/ModelandDataset/Qwen2.5-VL-7B-Instruct \
bash scripts/run_gpu_phase3_cross_arch.sh
```

单模型 CPU analysis：

```bash
MODEL_FAMILY=qwen2_5_vl \
MODEL_ALIAS=qwen2_5_vl_7b \
bash scripts/run_cpu_phase3_cross_arch.sh
```

四模型顺序运行：

```bash
PHASE3_STEP=all bash scripts/run_phase3_cross_arch_all.sh
```

只跑 GPU 或只跑 CPU：

```bash
PHASE3_STEP=gpu bash scripts/run_phase3_cross_arch_all.sh
PHASE3_STEP=cpu bash scripts/run_phase3_cross_arch_all.sh
```

## Artifacts

每个模型写入：

```text
outputs/stage_o_cross_model/<model_alias>/
```

核心产物：

- `probes/probe_results.csv`: `difference` full and `projected_difference` top-4/top-128/top-256 AUROC, plus raw `z_img`/`z_blind`;
- `stage_c_deep/stage_c_topk_curve.csv` and `plots/stage_c_topk_auroc_explained_variance.png`: explained variance vs AUROC curve;
- `stage_b/stage_b_pairwise_condition_deltas.csv`: matched/random/adversarial residual-tail gap;
- `margins/margin_baseline_metrics.csv`: raw first-token yes/no margin baseline;
- `minimal_replication_summary.csv`: compact summary consumed by paper notes.

## Interpretation

优先报告 qualitative pattern，而不是追求每个数值和 LLaVA-1.5 完全一致：

- Strong: full difference 明显强于 top-4，top-128/256 才恢复；variance/AUROC dissociation 清楚；residual-tail matched-vs-mismatch gap 出现；margin/raw hidden 不能单独解释全部效果。
- Acceptable: 上述四项中至少两到三项清楚复现。
- Failed: 如 Qwen/InternVL 不复现，作为 architecture limitation 写入，而不是扩张 generality claim。

## Environment Note

`after` 环境的 transformers 版本低于 Qwen2-VL 官方要求；Phase 3 GPU/CPU wrapper 默认使用 `/data/lh/.conda/envs/vlm-exp/bin/python`。

已检查 `vlm-exp` 环境：`transformers 4.52.4`，可 import `Qwen2VLForConditionalGeneration` 与 `Qwen2_5_VLForConditionalGeneration`，并包含 `pandas/sklearn/matplotlib/scipy/safetensors/accelerate/flash_attn/qwen_vl_utils`。当前非 CUDA shell 里 `torch.cuda.is_available()` 为 false；真实 GPU 阶段仍需在 CUDA-visible shell 中运行。该环境暂缺 `pytest` 和 `ruff`，不影响实验运行。
