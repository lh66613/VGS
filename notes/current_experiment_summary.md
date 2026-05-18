# VGS 最新实验系统性技术总结

> 生成日期：2026-05-18  
> 数据截止：本仓库已落盘结果，最新关键产物为 2026-05-14 的 LLaVA-13B minimal replication，以及 2026-05-13 的 mechanism mitigation paper tables。  
> 主模型：LLaVA-1.5-7B；补充模型：LLaVA-1.5-13B、Qwen2-VL-7B、Qwen2.5-VL-7B、InternVL2-8B、InternVL2.5-8B。  
> 主基准：POPE COCO；外部基准：AMBER discriminative。

## 一、汇报级结论

本阶段实验已经从“发现 correction geometry”推进到“用该几何解释并改进 VCD/ICD 缓解”的闭环。当前最稳妥的主结论是：

> Blind-reference correction geometry 不是一个可以替代输出置信度的万能检测器，而是一个可解释、可选择、可过滤的内部风险信号。它揭示了 VCD/ICD correction 中哪些子空间更接近 hallucination-relevant 成分，并能在 POPE 固定划分上改善 FP reduction 与 TP preservation 的权衡。

当前最强正结果是 LLaVA-1.5-7B / POPE fixed split 上的 subspace-filtered ICD：

| 方法 | FP reduction | TP preserved | Accuracy delta | 说明 |
| --- | ---: | ---: | ---: | --- |
| Full VCD-diffusion | 0.151 | 0.971 | -0.008 | TP-safe，但 FP reduction 弱 |
| Tail VCD-diffusion | 0.264 | 0.961 | -0.005 | 过滤后优于 full VCD |
| Full ICD TP-safe | 0.283 | 0.972 | 0.000 | 安全 full-space ICD baseline |
| Always ICD | 0.340 | 0.912 | -0.022 | FP reduction 强，但 TP damage 明显 |
| Gated ICD | 0.340 | 0.937 | -0.012 | 约 30% routed，低于 always-on 的 TP damage |
| **Band5-16 ICD** | **0.396** | **0.959** | **+0.001** | 当前最佳 TP-safe 结果 |

因此，最适合汇报的技术路线是：

1. **机制发现**：blind-reference difference 的 dominant variance directions 与 hallucination discrimination 解耦；有效判别信息更多出现在 residual/tail、PLS/Fisher、band 5-16 / top-complement 等非 dominant backbone 成分中。
2. **检测与路由**：输出 low-margin 是最强全局风险信号；geometry 与 margin 低度到中度相关，能在低触发预算、margin-missed residual pool、same-margin cases 中提供互补排序信息。
3. **缓解闭环**：prompt verification 较弱；always-on VCD/ICD 能减少 FP 但伤 TP；geometry-gated VCD/ICD 与 subspace-filtered ICD 能把 correction 用在更合适的样本或子空间上。
4. **边界条件**：跨模型上机制现象不完全一致；LLaVA/Qwen 支持 variance-discrimination decoupling，InternVL 出现“FP/TN 内部分离近乎完美但 predicted-Yes FP/TP 不可部署”的边界案例；LLaVA-13B minimal 复现为方向性支持，但 mitigation 尚不稳定。

## 二、核心定义与指标

每个样本提取 image-conditioned 和 blind/text-only 两个 hidden state：

```text
z_img   = hidden_state(image + question)
z_blind = hidden_state(question only)
d       = z_blind - z_img
```

主要分析层为 L24，辅助层为 L20 / L32。主要任务分为两类：

| 任务 | 部署含义 | 评价重点 |
| --- | --- | --- |
| FP vs TN | ground-truth = No 中区分错误 Yes 与正确 No | 机制分析、probe sanity、subspace discovery |
| predicted-Yes FP vs TP | 部署时只知道模型预测 Yes，需区分 hallucinated Yes 与 correct Yes | warning precision、FP recall、TP damage、TP preserved |

关键指标：

| 指标 | 含义 |
| --- | --- |
| FP reduction | 原始 FP 中被修正为 No 的比例 |
| TP preserved | 原始 TP 中仍保持 Yes 的比例 |
| TP damage | `1 - TP preserved` |
| Warning precision | 被 gate 触发样本中 FP 占比 |
| Compute saved | 相对 always-on VCD/ICD 少跑的 predicted-Yes 二次处理比例 |
| Accuracy delta | 相对原始模型 accuracy 的变化 |

## 三、实验产物地图

| 证据模块 | 关键文件 |
| --- | --- |
| 完整历史总结 | `notes/complete_experiment_summary.md` |
| Stage T 选择性修正 | `notes/stage_t_selective_correction_results.md` |
| Stage T margin-geometry 互补性 | `notes/stage_t_geometry_complementarity_results.md` |
| Stage U 跨模型最小协议 | `notes/stage_u_unified_minimal_protocol.md` |
| detector follow-up | `notes/detector_followup_summary.md` |
| mechanism mitigation 任务书 | `mitigation_plan.md` |
| mechanism mitigation MVP | `outputs/mechanism_mitigation/mvp/mvp_summary.md` |
| mechanism mitigation follow-up | `outputs/mechanism_mitigation/followup/followup_summary.md` |
| paper-ready mitigation tables | `outputs/mechanism_mitigation/paper_tables/mechanism_mitigation_paper_tables.md` |
| LLaVA-13B minimal replication | `outputs/mechanism_mitigation/llava13b_minimal/report/llava13b_minimal_replication_summary.md` |

## 四、结论 1：Top variance 不是 hallucination decision geometry

早期谱分析显示 blind-reference difference 具有强低秩结构，但高方差方向并不等价于 hallucination 判别方向。跨模型最小协议进一步验证了这一点：LLaVA/Qwen 系列中 top-4 能解释大量方差，但 FP/TN AUROC 通常低于 full/tail difference。

| Model | Readout | Layer | Top-4 Var | Top-4 AUROC | Full Diff AUROC | Tail AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | `last_prompt_token` | 20 | 0.842 | 0.531 | 0.659 | 0.649 |
| LLaVA-1.5-13B | `last_prompt_token` | 20 | 0.752 | 0.600 | 0.744 | 0.766 |
| Qwen2-VL-7B | `last_user_content_token` | 20 | 0.645 | 0.528 | 0.612 | 0.529 |
| Qwen2.5-VL-7B | `last_user_content_token` | 24 | 0.685 | 0.597 | 0.749 | 0.742 |
| InternVL2-8B | `last_user_content_token` | 20 | 0.928 | 0.977 | 0.997 | 0.663 |
| InternVL2.5-8B | `last_user_content_token` | 32 | 0.878 | 0.993 | 0.998 | 0.734 |

解释口径：

- LLaVA/Qwen 支持“方差主轴主要承载 image-conditioning backbone，而非 hallucination decision”的结论。
- InternVL 是重要边界案例：它的 FP/TN separability 在 top coordinates 中已经接近完美，但后续 predicted-Yes 部署任务失败，说明 FP/TN 内部可分并不自动等于可部署 hallucination detector。

因果证据也支持“tail/residual 不是无关噪声”。L24 tail ablation 在 TN 样本上有清晰剂量效应：

| Layer / alpha | Yes rate | Median margin shift |
| --- | ---: | ---: |
| L24 alpha 4 | 0.000 | -0.750 |
| L24 alpha 5 | 0.125 | -0.328 |
| L24 alpha 6 | 0.563 | +0.016 |
| L24 alpha 7 | 0.938 | +0.391 |
| L24 alpha 8 | 1.000 | +0.934 |

这说明 L24 tail coordinates 对正确拒绝 TN 具有因果必要性；但 FP rescue 仍弱，64 个 FP 扩展样本中仅 3 个 borderline case 被翻转，因此 direct hidden steering 不能包装为可靠 mitigation。

## 五、结论 2：Geometry 是 margin 的互补风险信号，不是 margin 替代品

### 2.1 POPE strict split 上，margin+tail/full 显著优于 margin-only

Detector follow-up 使用 `random -> popular -> adversarial` 严格协议：random 训练，popular 校准，adversarial 测试。

| Method | AUROC | AUPRC | Warning precision @20% | FP recall @20% | TP damage @20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| yes/no margin | 0.848 | 0.401 | 0.404 | 0.667 | 0.155 |
| margin + top16 | 0.853 | 0.447 | 0.389 | 0.624 | 0.154 |
| margin + tail | 0.881 | 0.609 | 0.437 | 0.698 | 0.141 |
| margin + full | 0.884 | 0.612 | 0.436 | 0.698 | 0.142 |

Bootstrap 支持 margin+tail/full 的增益：相对 margin-only，margin+tail 的 AUROC +0.033、AUPRC +0.204、warning precision +0.033，且 TP damage -0.013；margin+full 类似，AUROC +0.037、AUPRC +0.208。

### 2.2 固定划分 predicted-Yes 池中，low-margin 是强 baseline，geometry 提供低预算增益

固定划分测试池包含 596 个 predicted-Yes 样本，其中 53 个 FP、543 个 TP，base FP prevalence 约 0.089。直接用高 margin 作为风险方向是错误的，因为高 yes/no margin 选择的是非常自信的 Yes；应使用 low-margin 作为 confidence baseline。

在 20% predicted-Yes 触发预算下：

| Gate | Triggered FP ratio | FP recall | TP damage | Warning precision |
| --- | ---: | ---: | ---: | ---: |
| low-margin only | 0.293 | 0.736 | 0.173 | 0.293 |
| low-margin + FullD | 0.327 | 0.660 | 0.133 | 0.327 |
| low-margin + PLS | 0.324 | 0.642 | 0.131 | 0.324 |
| PLS only | 0.226 | 0.396 | 0.133 | 0.226 |
| random gate | 0.088-0.093 | 0.155-0.233 | 0.156-0.222 | 0.088-0.093 |

这组结果的正确解读是：low-margin 决定了最高 FP recall，但加入 geometry 后，在低预算下能提高 warning precision 并降低 TP damage。

固定 trigger ablation 进一步给出 reviewer-facing 证据。在 10% 预算下，`Margin + PLS` 与 `Margin + full/tail` 都把 warning precision 从 margin-only 的 0.350 提升到 0.383，并把 TP damage 从 0.072 降到 0.068。

### 2.3 Geometry 与 margin 相关但不冗余

Stage T complementarity 分析中，geometry score 与 yes/no margin 的相关性较低：

| Geometry score | Pearson vs margin | Spearman vs margin |
| --- | ---: | ---: |
| PLS32 | -0.231 | -0.197 |
| FullD | -0.179 | -0.104 |
| Tail | -0.201 | -0.104 |

在 margin-only gate 已捕获 39/53 FP 后，仍有 14 个 margin-missed FP。PLS geometry 在 residual pool 上 AUROC = 0.633，并额外捕获 3/14 个 missed FP；full/tail 各捕获 4/14，但额外触发成本较高。这支撑“互补信号”而非“全局替代 margin”的写法。

### 2.4 AMBER 外部迁移：tail warning 有信号，但 margin 仍更强

AMBER 上存在两类结果，需要分开汇报：

| 设置 | 最好方法 | Trigger | FP recall | TP damage | Warning precision | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| geometry-only external top-rate | tail energy | 0.20 | 0.285 | 0.166 | 0.405 | 几何排序有外部信号，优于随机约 0.284 |
| AMBER margin deployment | margin-only | 0.20 | 0.291 | 0.164 | 0.413 | 外部部署中 low-margin 仍最强 |
| AMBER margin deployment | margin+tail | 0.20 | 0.225 | 0.190 | 0.319 | POPE-trained geometry 加到 AMBER margin 后反而降低 precision |

因此外部结论应写成：geometry 在 AMBER 上有 modest ranking transfer，尤其 tail energy；但在 AMBER margin logits 可用时，low-margin 是更稳的外部 warning baseline，POPE-trained geometry 的外部组合迁移不稳定。

## 六、结论 3：Prompt verification 弱，VCD/ICD 才是更合适的 correction operator

Stage T 先测试了 gated verification prompt。固定划分上共有 513 个 gated predicted-Yes 样本，legacy prompt 的实际变化很少：

| Original outcome | Verification Yes | Verification No |
| --- | ---: | ---: |
| TP | 460 | 7 |
| FP | 37 | 9 |

更强 prompt 反而更倾向复读 Yes：

| Prompt | TP -> No | FP -> No |
| --- | ---: | ---: |
| legacy | 7 | 9 |
| forced_evidence | 2 | 3 |
| conservative | 1 | 2 |
| internal_rationale | 3 | 6 |

在 30% trigger 下，PLS gate 的 oracle FP reduction 可达 0.604，但 legacy prompt 实际仅 0.132；prompt wording 不是足够强的缓解算子。后续实验转向 VCD/ICD 是正确方向。

## 七、结论 4：Always-on VCD/ICD 有效但伤 TP，selective routing 改善 tradeoff

固定划分 predicted-Yes 池上，always-on VCD/ICD 会减少 FP，但 TP damage 明显：

| Operator | Always FP reduction | TP preserved | Accuracy delta |
| --- | ---: | ---: | ---: |
| ICD-blind | 0.340 | 0.912 | -0.022 |
| VCD-diffusion | 0.302 | 0.912 | -0.024 |
| VCD-gray | 0.264 | 0.890 | -0.034 |
| VCD-blur | 0.151 | 0.971 | -0.006 |

Selective routing 把同一 correction operator 用到更集中的样本上，形成两种部署取向：

| Operator + Gate | Target | Trigger | FP reduction | TP preserved | Acc delta | Compute saved | 适用目标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ICD + low-margin+tail | 0.30 | 0.297 | 0.340 | 0.937 | -0.012 | 0.703 | 最大 FP suppression |
| ICD + low-margin+PLS | 0.20 | 0.176 | 0.321 | 0.965 | -0.001 | 0.824 | 高 FP/trigger |
| ICD + PLS | 0.30 | 0.284 | 0.283 | 0.976 | +0.001 | 0.716 | TP-preserving geometry-only |
| ICD + FullD | 0.30 | 0.262 | 0.245 | 0.985 | +0.004 | 0.738 | 最平衡 accuracy/F1 |
| ICD + margin+FullD | 0.30 | 0.267 | 0.170 | 0.994 | +0.004 | 0.733 | 极低 TP damage |
| VCD-diffusion + low-margin+PLS | 0.30 | 0.273 | 0.283 | 0.948 | -0.010 | 0.727 | official VCD FP reduction |

Bootstrap 结果支持 tradeoff 的稳健性。例如：

| Row | Metric | Point | 95% CI |
| --- | --- | ---: | --- |
| ICD + low-margin+PLS @20% | FP reduction | 0.321 | [0.196, 0.452] |
| ICD + low-margin+PLS @20% | TP preserved | 0.965 | [0.949, 0.980] |
| ICD + full_probe @30% | FP reduction | 0.245 | [0.130, 0.370] |
| ICD + full_probe @30% | TP preserved | 0.985 | [0.974, 0.994] |
| ICD + margin+full @30% | Accuracy delta | +0.004 | [0.000, 0.010] |

关键汇报措辞：

> Always-on VCD/ICD can reduce hallucinated Yes answers, but it over-corrects true positives. Geometry-guided routing turns VCD/ICD into a controllable FP-TP tradeoff: low-margin+geometry maximizes FP capture, while geometry-only or margin+geometry gives safer TP-preserving operating points.

## 八、结论 5：最新 mechanism mitigation 结果支持“过滤 correction spectrum”

最新的 mitigation 任务进一步从 routing 推进到 subspace-filtered correction。核心思想是：不使用完整 ICD/VCD correction，而只保留 correction spectrum 中更相关的子空间：

```text
h' = h_orig + alpha * P_S(h_orig - h_neg)
```

其中 `S` 可以是 band 5-16、top4-complement、tail 等子空间。

### 5.1 Stage 1 decomposition：ICD-blind 的判别性贡献集中在非 full/top4 子空间

对 L24 的 operator decomposition 显示，ICD-blind 的 top4/full 成分虽然贡献大，但对 FP 与 TP 的区分性并不强；band5-16 和 top4-complement 更有 FP-specific correction 倾向。

| Operator | Band | FP positive rate | TP positive rate | FP-TP gap |
| --- | --- | ---: | ---: | ---: |
| ICD-blind | top4-complement | 0.811 | 0.518 | 0.293 |
| ICD-blind | band5-16 | 0.808 | 0.582 | 0.226 |
| ICD-blind | band17-64 | 0.725 | 0.542 | 0.182 |
| ICD-blind | random12 | 0.997 | 0.988 | 0.009 |
| VCD-diffusion | full | 0.599 | 0.582 | 0.016 |
| VCD-diffusion | top4 | 0.361 | 0.349 | 0.012 |

这直接解释了为什么 full ICD/VCD 会同时修 FP 和伤 TP：完整 correction 混入大量非判别成分；过滤到更合适的 band 后，FP-TP tradeoff 更好。

### 5.2 Stage 2 / paper tables：Band5-16 ICD 是当前最佳 TP-safe 方法

在 LLaVA-1.5-7B / fixed split test 上，Band5-16 ICD 取得当前最佳 TP-safe mitigation：

| Method | FP reduction | TP preserved | Accuracy delta | Overall Yes rate | FP Yes rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.000 | 1.000 | 0.000 | 0.441 | 1.000 |
| Full VCD-diffusion | 0.151 | 0.971 | -0.008 | 0.430 | 0.849 |
| Tail VCD-diffusion | 0.264 | 0.961 | -0.005 | 0.430 | 0.736 |
| Full ICD TP-safe | 0.283 | 0.972 | 0.000 | 0.419 | 0.717 |
| Always ICD | 0.340 | 0.912 | -0.022 | 0.393 | 0.660 |
| Gated ICD | 0.340 | 0.937 | -0.012 | 0.403 | 0.660 |
| **Band5-16 ICD** | **0.396** | **0.959** | **+0.001** | **0.416** | **0.604** |

随机控制表明该结果不是任意 12 维子空间都能做到：

| Target | Random family | Target FP | Random FP mean | Random range | Percentile | Outperforms |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Band5-16 ICD | random12 | 0.396 | 0.165 | [0.000, 0.340] | 100 | 20/20 |
| Top4-complement ICD | random4-complement | 0.321 | 0.283 | [0.283, 0.283] | 100 | 20/20 |
| Tail VCD-diffusion | random-tail-dim | 0.264 | 0.148 | [0.075, 0.283] | 95 | 19/20 |

Bootstrap comparison 支持 Band5-16 相对关键 baseline 的优势：

| Comparison | Metric | A | B | Diff | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| Band5-16 ICD vs Always ICD | TP preserved | 0.959 | 0.912 | +0.048 | [0.025, 0.073] |
| Band5-16 ICD vs Always ICD | Accuracy delta | +0.001 | -0.022 | +0.023 | [0.012, 0.036] |
| Band5-16 ICD vs Random12 ICD | FP reduction | 0.396 | 0.170 | +0.226 | [0.117, 0.345] |
| Tail VCD vs Full VCD | FP reduction | 0.264 | 0.151 | +0.113 | [-0.016, 0.246] |

No-bias audit 也很重要：Band5-16 ICD 不是简单地把模型整体推向 No。它保持 TN Yes rate 约 0.005，同时把 FP Yes rate 从 1.000 降到 0.604，TP Yes rate 保持 0.959，accuracy 从 base 0.863 提到 0.864。

### 5.3 Reverse split 与 LLaVA-13B：正结果有边界

Reverse split 中 Band5-16 ICD 仍 TP-safe，但不是最优：

| Method | Calibrated on | Tested on | FP reduction | TP preserved | Accuracy delta |
| --- | --- | --- | ---: | ---: | ---: |
| Full ICD TP-safe | adversarial | random | 0.280 | 0.954 | -0.014 |
| Band5-16 ICD | adversarial | random | 0.220 | 0.953 | -0.013 |
| Random12 ICD | adversarial | random | 0.080 | 0.977 | -0.006 |
| Tail VCD-diffusion | adversarial | random | 0.300 | 0.953 | -0.009 |

LLaVA-13B minimal replication 提供方向性支持，但还不足以支撑 universal mitigation claim：

| Criterion | Status | Value |
| --- | --- | --- |
| Detector margin+tail/full beats margin-only | pass | margin=0.339, tail=0.344, full=0.340 AUPRC |
| Band5-16 TP-safe beats Full ICD TP-safe | pass | band FP reduction=0.133, full=0.033 |
| Gated ICD keeps most Always ICD FP reduction with higher TP preserved | fail | gated 0.033/0.995, always 0.033/0.995 |
| Always ICD shows stronger conservative bias than Base | fail | always yes=0.483, base yes=0.466 |

LLaVA-13B mitigation表：

| Method | FP reduction | TP preserved | Accuracy delta | Overall Yes rate | FP Yes rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.000 | 1.000 | 0.000 | 0.466 | 1.000 |
| Full ICD TP-safe | 0.033 | 1.000 | +0.002 | 0.470 | 0.967 |
| Band5-16 ICD | 0.133 | 0.963 | -0.010 | 0.450 | 0.867 |
| Always ICD | 0.033 | 0.995 | -0.004 | 0.483 | 0.967 |
| Gated ICD | 0.033 | 0.995 | -0.001 | 0.462 | 0.967 |
| Random12 mean | 0.050 | 0.994 | +0.001 | 0.465 | 0.950 |
| Random12 best | 0.083 | 0.988 | -0.002 | 0.458 | 0.917 |

建议表述：LLaVA-13B 上 Band5-16 方向性优于 full ICD，但收益变小且有 accuracy cost；gated ICD 未显示额外优势。这应作为 scale-up boundary，而不是主负结果。

## 九、跨模型边界：InternVL 证明 FP/TN probe 不等于 deployable detector

Stage U 的关键边界发现是 InternVL：内部 FP/TN 分离接近完美，但 predicted-Yes FP/TP 部署 AUROC 很低。

| Model | FP/TN Full AUROC | Pred-Yes Full AUROC | Pred-Yes Low-Margin AUROC | Flag |
| --- | ---: | ---: | ---: | --- |
| InternVL2-8B | 0.997 | 0.249 | 0.883 | near-perfect FP/TN but non-deployable |
| InternVL2.5-8B | 0.998 | 0.126 | 0.903 | near-perfect FP/TN but non-deployable |

原因从 score distribution 可见：InternVL 的 FP 和 TP 在 predicted-Yes 池内都被 geometry score 打到非常高，导致 FP/TP 不可区分。

| Model | Outcome | N | Mean | Median | Q25 | Q75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| InternVL2-8B | FP | 18 | 0.871 | 0.996 | 0.951 | 1.000 |
| InternVL2-8B | TP | 513 | 0.945 | 1.000 | 0.999 | 1.000 |
| InternVL2.5-8B | FP | 47 | 0.937 | 1.000 | 0.998 | 1.000 |
| InternVL2.5-8B | TP | 589 | 0.987 | 1.000 | 1.000 | 1.000 |

汇报中应把 InternVL 写成边界发现：内部 correction geometry 可以真实存在，但如果部署 population 从 FP/TN 变成 predicted-Yes FP/TP，信号可能饱和并失去路由价值。

## 十、最终推荐汇报结构

建议将报告组织为四页主线：

1. **Mechanistic discovery**：Blind-reference differencing reveals layered correction geometry. Top variance carries image-conditioning backbone; hallucination-relevant signal is in residual/tail/supervised or mid-band coordinates.
2. **Risk routing**：Geometry is complementary to low-margin confidence. It improves low-budget warning precision and provides TP-preserving routing choices, but it is not a global replacement for margin.
3. **Mitigation method**：Subspace-filtered ICD, especially Band5-16 ICD, gives the strongest LLaVA-7B POPE TP-safe FP reduction: 0.396 FP reduction, 0.959 TP preserved, +0.001 accuracy delta.
4. **Boundary and honesty**：Prompt verification is weak; AMBER transfer is modest; LLaVA-13B and InternVL show the method is not universal. The contribution should be framed as a mechanistically motivated FP-TP tradeoff improvement, not a universal hallucination solver.

## 十一、可直接使用的核心表述

中文汇报版：

> 我们发现，视觉语言模型的盲参考差分 `d = z_blind - z_img` 形成了层次化 correction geometry。主方差方向主要反映图像条件化 backbone，并不等价于 hallucination 判别方向；真正与错误 Yes / 正确 No 相关的信号更多出现在 residual/tail、PLS/Fisher 以及 ICD 的 band 5-16 / top-complement 成分中。基于这一机制，我们不再使用完整 VCD/ICD correction，而是进行子空间过滤或选择性路由。在 LLaVA-1.5-7B / POPE fixed split 上，Band5-16 ICD 达到 0.396 FP reduction、0.959 TP preserved 和 +0.001 accuracy delta，优于 full VCD-diffusion、full ICD TP-safe 和 random12 控制。该方法的边界也很清楚：low-margin 仍是强 baseline，AMBER 和 LLaVA-13B 迁移更弱，InternVL 出现内部可分但部署不可用的反例。因此本文贡献应定位为 correction geometry 指导的可解释 tradeoff improvement，而不是通用幻觉检测或通用修复器。

英文论文版：

> Blind-reference differencing exposes a layered correction geometry in VLM hallucination. The dominant variance directions largely encode image-conditioning backbone rather than hallucination decision geometry, while hallucination-relevant correction signals concentrate in residual/tail, supervised, and mid-band subspaces. This enables targeted correction: filtering ICD to the Band5-16 subspace achieves 39.6% FP reduction with 95.9% TP preservation and a non-negative accuracy delta on the LLaVA-1.5-7B POPE fixed split, outperforming full VCD, TP-safe full ICD, and random subspace controls. The effect should be framed as a mechanistically grounded FP-TP tradeoff improvement: output low-margin remains a strong confidence baseline, external transfer is modest, and some architectures exhibit internally separable but non-deployable correction signals.

## 十二、后续最小行动项

| 优先级 | 行动 | 目的 |
| --- | --- | --- |
| P0 | 对 Band5-16 ICD 做更多 seed / split bootstrap | 巩固 paper 主表统计显著性 |
| P0 | 扩展 LLaVA-13B full protocol，而不只 minimal replication | 判断 scale-up 是否稳定 |
| P1 | 在 AMBER 上补齐与 Band5-16 ICD 对齐的 correction evaluation | 判断外部 mitigation 是否存在 |
| P1 | 把 low-margin、geometry-only、low-margin+geometry 的 operating points 固定为三类部署策略 | 防止汇报时混淆 warning 与 correction 目标 |
| P2 | 继续探索 stronger correction operator，而不是 prompt wording | prompt verification 已被证明较弱 |

