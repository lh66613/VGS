下面是一份可以直接执行的“小计划书”。我建议你把这个落地模块命名为：

# Correction-Geometry Guided Selective Correction

中文可以叫：**基于校正几何的选择性幻觉修正**

核心目标不是做一个“全局强 mitigation”，而是证明：

> 你发现的 blind-reference correction geometry 不仅能解释幻觉，还能指导什么时候需要额外验证、对比解码或保守处理，从而以较低触发率降低 FP，并减少对正常样本的副作用。

这个定位最稳。因为你现有结果已经说明：full difference / PLS / tail 坐标有 FP/TN 信号，但 logits margin baseline 有时很强，FP rescue 又只对 borderline case 有效，所以不应该硬写成 SOTA detector 或 reliable mitigation。

---

# 一、总体思路

已有机制发现是：

1. `z_blind - z_img` 存在结构化 correction geometry；
2. top variance directions 不是 hallucination decision geometry；
3. FP/TN 信号主要在 full difference、PLS/Fisher、residual/tail coordinates 中；
4. matched evidence 与 mismatch 的差异也主要体现在 residual/tail 或 supervised decision view；
5. L24 residual/tail 坐标对 TN 的正确 `No` 决策有因果相关性，但 FP rescue 不是可靠 mitigation。

所以落地方法应该是：

> 用 residual/tail correction geometry 作为内部风险信号，决定哪些样本需要额外处理。

不是所有样本都修正。只对高风险样本触发 verification 或 VCD。

VCD 本身是一种 training-free 的 decoding-time 方法，通过对比原始视觉输入和扰动视觉输入的输出分布来缓解 object hallucination；你的方法不需要重新发明 VCD，而是用内部 correction geometry 来决定什么时候触发它。([arXiv][1])

---

# 二、方法设计

## 2.1 输入与基本表示

对每个样本，跑两次 forward：

```text
image-conditioned:
z_img = hidden_state(image + question)

blind/text-only:
z_blind = hidden_state(question only)
```

定义差分：

```text
d = z_blind - z_img
```

主用 layer 建议：

```text
L24 为主，L20 / L32 为辅助
```

原因是：你已有结果中 L24 full difference、PLS、tail ablation 都比较关键；L32 可以作为 late arbitration 的补充。

---

## 2.2 构造 geometry risk score

建议先做三个分数，不要一开始太复杂。

### Score A：Full-Difference Risk

用完整 `d` 训练一个轻量 logistic probe：

```text
s_full = Logistic(d)
```

训练任务：

```text
FP vs TN
```

也就是只在 ground-truth no 样本里区分：

```text
FP: 模型错误回答 Yes
TN: 模型正确回答 No
```

已有依据：full difference 5-seed AUROC 约 0.721，是稳定信号。

---

### Score B：Tail / Residual Risk

先对 `D = z_blind - z_img` 做 SVD，然后取 tail band：

```text
P_tail(d) = SVD coordinates 257-1024
```

构造两种版本：

```text
s_tail_energy = ||P_tail(d)||^2
s_tail_probe = Logistic(P_tail(d))
```

已有依据：matched evidence 与 mismatch 的差异更明显体现在 residual/tail view，而不是 top variance backbone。

---

### Score C：PLS / Fisher Risk

用 Stage L 中表现较好的 supervised evidence-specific direction：

```text
s_pls = w_pls^T d
```

主设定可以用：

```text
L24, K=32 PLS
```

已有依据：PLS FP/TN 在 L24 K=32 达到约 0.720，是最强 compact detection subspace；但要注意 PLS 稳定性弱一些，所以最好作为一个可比较分数，而不是唯一主分数。

---

## 2.3 构造 gate

不要直接改模型。先做一个 gate：

```text
G(x) = 1 表示触发额外处理
G(x) = 0 表示直接使用原始回答
```

最小版本：

```text
G_geo(x) = 1 if s_geo(x) > τ_geo
```

更合理版本：

```text
G_geo_yes(x) = 1 if model predicts Yes and s_geo(x) > τ_geo
```

因为 POPE 中你最关心的是 object hallucination，也就是模型把不存在的对象答成 `Yes`。

再做一个组合版本：

```text
G_margin+geo(x) = 1 if model predicts Yes and 
                  (s_geo(x) > τ_geo or margin is uncertain)
```

其中 yes/no margin 可以定义为：

```text
margin = logit(Yes) - logit(No)
```

这里必须设置校准集和测试集：

```text
calibration split: 选 τ_geo / τ_margin
held-out split: 固定阈值评估
```

不要在 test 上调阈值。

---

# 三、触发后的处理方式

我建议分两级做。

---

## 3.1 第一优先级：Geometry-Gated Verification

这是最稳、最容易实现的落地实验。

如果：

```text
G(x) = 0
```

直接保留原始回答。

如果：

```text
G(x) = 1
```

触发二次 verification prompt。

示例 prompt：

```text
Please verify the visual evidence before answering.
Answer "Yes" only if the queried object is clearly visible in the image.
If the visual evidence is insufficient or unclear, answer "No".

Question: {question}
```

中文理解就是：

> 先让模型重新检查图像证据，只有明确看见目标对象才回答 Yes，否则回答 No。

这个方法优点：

* 实现成本低；
* 不需要改模型内部；
* 能直接证明 geometry score 有 routing utility；
* 不会被审稿人质疑“复杂编辑技巧调参”。

---

## 3.2 第二优先级：Geometry-Gated VCD

如果 verification 有效果，再做 VCD gate。

方法：

```text
G(x) = 0: 原始 greedy decoding
G(x) = 1: 使用 VCD / ICD / 你已有的 contrastive decoding baseline
```

对比对象：

```text
Original
Always-on VCD
Margin-gated VCD
Geometry-gated VCD
Margin+Geometry-gated VCD
Random-gated VCD
```

其中 `Random-gated VCD` 非常重要。

它的作用是控制：

> 不是因为少触发一些样本就有效，而是 geometry gate 真的选中了更需要修正的样本。

---

# 四、需要完成的实验

## 实验 1：Geometry 分数是否有互补价值

目的：

> 先确认 geometry score 是否能补充 margin / entropy，而不是直接进入 decoding。

比较：

```text
margin-only
entropy-only
full-difference risk
tail risk
PLS risk
margin + full-difference
margin + tail
margin + PLS
```

指标：

```text
AUROC
AUPRC
Risk-Coverage AUC
FP rate at fixed coverage
Coverage at fixed FP rate
ECE / calibration error
```

重点看：

```text
margin + geometry 是否优于 margin-only
```

预期：

* geometry-only 不一定超过 margin；
* 但 margin+geometry 在中等置信度样本、mismatch 样本、AMBER transfer 上可能更稳；
* 如果完全没有提升，就不要主打 gated correction，改回机制分析 + calibration 负结果。

---

## 实验 2：Margin-bin / Overconfidence 分析

目的：

> 验证 geometry 是否能发现 logits 过自信但视觉证据不足的样本。

按 yes/no margin 分桶：

```text
low margin
medium margin
high margin
```

每个桶里统计：

```text
FP/TN 数量
geometry score 分布
geometry AUROC
margin+geometry AUROC
```

特别关注：

```text
high-margin FP
medium-margin FP
```

预期有三种情况：

### 理想结果

high-margin FP 中 geometry 仍能区分 FP/TN。

可以写：

> correction geometry detects visually unsupported overconfident hallucinations.

### 中等结果

geometry 主要在 medium-margin / boundary-near 样本有效。

也可以写：

> correction geometry is most useful for boundary-near hallucination handling.

这和你已有 FP rescue 主要影响 borderline cases 的结果一致。

### 差结果

geometry 在所有 margin bin 中都没有额外价值。

那就不要做 gated method 主线，只把它作为 limitation。

---

## 实验 3：Geometry-Gated Verification

比较方法：

| 方法                                 | 说明                     |
| ---------------------------------- | ---------------------- |
| Original                           | 原始模型                   |
| Always Verification                | 所有样本都二次验证              |
| Margin-Gated Verification          | margin 触发              |
| Geometry-Gated Verification        | geometry score 触发      |
| Margin+Geometry-Gated Verification | margin 和 geometry 组合触发 |
| Random-Gated Verification          | 同触发率随机触发               |

指标：

| 指标             | 作用          |
| -------------- | ----------- |
| Trigger Rate   | 额外处理比例      |
| Accuracy       | 总体性能        |
| F1             | yes/no 平衡性能 |
| FP Rate ↓      | 幻觉减少程度      |
| TN Preserved ↑ | 正确否定是否保留    |
| TP Damage ↓    | 是否误伤正确 Yes  |
| Extra Compute  | 额外计算成本      |

核心目标不是一定超过 Always Verification，而是证明：

> 在相同触发率下，Geometry-Gated Verification 比 Random-Gated / Margin-Gated 更有效降低 FP，且副作用更小。

---

## 实验 4：Geometry-Gated VCD

如果实验 3 有效果，再做这个。

比较方法：

| 方法                        | 说明          |
| ------------------------- | ----------- |
| Original                  | 原始解码        |
| Always VCD                | 所有样本使用 VCD  |
| Margin-Gated VCD          | margin 触发   |
| Geometry-Gated VCD        | geometry 触发 |
| Margin+Geometry-Gated VCD | 联合触发        |
| Random-Gated VCD          | 同触发率随机触发    |

指标同上，但额外加：

```text
Average decoding cost
VCD trigger ratio
FP reduction per triggered sample
```

核心目标：

> 用更少的触发比例获得接近 Always VCD 的 FP reduction，或者在相同 FP reduction 下减少 TP/TN damage。

如果 VCD 实现成本高，这个实验可以后置。

---

## 实验 5：Top-SVD vs Tail-Guided Gate Ablation

目的：

> 证明你的机制发现真的指导了方法设计。

比较 gate 来源：

```text
Top-4 SVD gate
Top-64 SVD gate
Tail 257-1024 gate
Full difference gate
PLS/Fisher gate
Random subspace gate
```

预期：

* Top-4 gate 效果弱；
* Tail / full difference / PLS gate 更有效；
* 这直接呼应你的核心发现：dominant variance ≠ decision geometry。

已有结果中 top-4 弱、full difference/PLS 较强，这个实验大概率能接上。

---

## 实验 6：外部验证

至少做两个层级：

### 6.1 LLaVA-1.5-13B

你已有 13B 结果，可以复用。当前已有 evidence 支持 full difference 强、top-4 弱、tail gap 存在，但仍属于 LLaVA-family replication。

### 6.2 AMBER

用 POPE calibration 得到的 gate 或 risk score，在 AMBER 上测试：

```text
risk transfer
selective prediction
gated verification
```

预期：

* 不要求强；
* 只要 above-chance 或能在部分子任务上降低 FP，就可以写成 modest transfer。

你已有 AMBER transfer 最高约 0.63-0.665，所以语气要克制。

### 6.3 可选：Qwen2-VL / InternVL

如果想冲更强 venue，补一个不同架构。你的原总结也建议，如果想把方法做硬，需要跑一个不同架构 LVLM，并补 VCD/ICD 或 CLIP/image-text similarity baseline。

---

# 五、预期结果与写法

## 最理想结果

如果结果如下：

```text
Geometry-gated verification/VCD 在 20%-40% trigger rate 下，
达到接近 always-on verification/VCD 的 FP reduction，
并且比 margin-gated / random-gated 更少伤害 TP/TN。
```

那么论文可以写：

> Correction geometry enables selective hallucination handling: it identifies samples where visual evidence correction is abnormal and routes them to additional verification or contrastive decoding.

这时你的文章就从机制分析升级为：

```text
mechanistic analysis + actionable selective correction
```

---

## 中等结果

如果结果是：

```text
Geometry gate 不全面超过 margin gate，
但在 medium-margin / mismatch / adversarial subset 上更有效。
```

也可以写：

> Correction geometry is complementary to output confidence, especially under evidence mismatch and boundary-near hallucination.

这仍然有价值。

---

## 较差结果

如果：

```text
geometry gate 几乎不提升，
verification/VCD 也没有比 random gate 好。
```

那就不要主打方法，改成：

> 机制发现对 direct correction 的落地有限，说明 hallucination decision 不容易通过简单 routing 修复。

这虽然弱一些，但仍然能作为 limitation 和 discussion。

---

# 六、最终产出物

你按计划跑完后，至少应该生成这些表和图。

## 表 1：Geometry score comparison

```text
margin / entropy / fullD / tail / PLS / margin+geo
AUROC, AUPRC, Risk-Coverage AUC, FP@Coverage
```

## 表 2：Gated verification result

```text
Original / Always / Margin-gated / Geometry-gated / Margin+Geometry / Random
Trigger Rate, Acc, F1, FP Rate, TN Preserved, TP Damage, Extra Compute
```

## 表 3：Gated VCD result

如果完成 VCD，就放这张；如果没完成，可以放 appendix 或 future work。

## 图 1：Risk-coverage curve

比较：

```text
margin-only
geometry-only
margin+geometry
```

## 图 2：Trigger-rate vs FP-reduction curve

横轴：

```text
Trigger Rate
```

纵轴：

```text
FP Reduction
```

比较：

```text
random gate
margin gate
geometry gate
margin+geometry gate
```

## 图 3：Margin-bin analysis

展示不同置信度区间中 geometry 的效果。

## 图 4：Top-SVD vs Tail gate ablation

证明 tail/full difference 比 top-4 更适合作为 gate。

---

# 七、执行顺序

你可以严格按下面顺序做。

## Step 1：整理现有缓存

确认每个样本都有：

```text
image_id
question
label
original prediction
yes/no logits or margin
entropy
z_img
z_blind
d = z_blind - z_img
FP/TN/TP/FN outcome
split id
```

---

## Step 2：生成 geometry scores

生成：

```text
s_full
s_tail_energy
s_tail_probe
s_pls
s_fisher
s_top4
s_random
```

输出文件：

```text
outputs/geometry_gated/scores_pope.csv
```

---

## Step 3：做 selective prediction

输出：

```text
outputs/geometry_gated/risk_coverage.csv
outputs/geometry_gated/margin_bin_analysis.csv
outputs/geometry_gated/score_comparison.csv
```

如果 `margin+geometry` 完全没提升，先暂停 gated correction，重新判断。

---

## Step 4：做 gated verification

先用 3 个触发率：

```text
10%
20%
30%
```

每个触发率比较：

```text
random
margin
geometry
margin+geometry
```

输出：

```text
outputs/geometry_gated/gated_verification_results.csv
```

---

## Step 5：做 gated VCD

如果 Step 4 有积极结果，再做 VCD。

输出：

```text
outputs/geometry_gated/gated_vcd_results.csv
```

---

## Step 6：做 ablation

比较：

```text
top4 gate
tail gate
fullD gate
PLS gate
random subspace gate
```

输出：

```text
outputs/geometry_gated/gate_ablation.csv
```

---

## Step 7：外部验证

优先：

```text
AMBER
LLaVA-1.5-13B
```

有余力再做：

```text
Qwen2-VL / InternVL
```

输出：

```text
outputs/geometry_gated/external_transfer.csv
```

---

# 八、论文中对应的新贡献

如果实验跑通，论文贡献可以改成：

1. We introduce blind-reference differencing to analyze visual-evidence correction geometry in LVLMs.
2. We show that dominant variance directions are not hallucination decision directions.
3. We identify residual/tail correction coordinates that are evidence-sensitive and causally relevant to faithful negative decisions.
4. We propose correction-geometry guided selective correction, which uses this internal geometry to route high-risk samples to verification or contrastive decoding, reducing unnecessary intervention.

最后一句是新增落地贡献。

---

# 九、我建议你现在先做的最小版本

不要一上来做所有东西。先做这个最小闭环：

```text
1. 用已有 POPE hidden states 生成 s_full / s_tail / s_pls / margin。
2. 做 margin vs geometry vs margin+geometry 的 risk-coverage。
3. 做 10%、20%、30% trigger rate 的 gated verification。
4. 加 random gate 和 top-4 gate 对照。
```

这个最小闭环如果有效，你的论文就有了非常清楚的落地点：

> 机制发现不仅解释了幻觉，还能指导选择性验证。

如果这个最小闭环都无效，再考虑是否转向更保守的机制论文。

[1]: https://arxiv.org/abs/2311.16922?utm_source=chatgpt.com "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding"
