下面我帮你把这次汇报整理成一个**可以直接照着讲的版本**。核心原则是：**不要按 Stage A/B/C/E/G/J/K/L/M/N/P/O/T 顺序讲**，否则听众一定会迷失。你要把大量实验压缩成一条清晰故事线：

> **我们发现 VLM 中“图像带来的主变化方向”并不是“幻觉判别方向”；真正与幻觉相关的信号藏在 residual/tail correction coordinates 中。它能解释模型为什么正确拒绝，也能辅助选择性风险路由，但目前还不能声称可靠修正幻觉。**

---

# 一、这次汇报的总标题

建议标题：

> **Blind-Reference Differencing Reveals Layered Correction Geometry in VLM Hallucination**

中文汇报标题可以是：

> **基于盲参考差分的视觉语言模型幻觉校正几何分析**

如果想更口语、更适合组会：

> **VLM 幻觉信号到底藏在哪里？从盲参考差分到分层校正几何**

---

# 二、这次汇报只讲 5 个核心结论

你实验很多，但汇报时只需要让听众记住这 5 句话：

1. **我们构造了 blind-reference difference：比较 image+question 与 question-only 的 hidden state 差异。**
   也就是 (d = z_{\text{blind}} - z_{\text{img}})，用它表示视觉证据对模型内部表征的“校正”。你的实验总结里明确将这个作为核心方法论，并以 FP/TN 区分作为主要机制任务。

2. **这个 correction space 有强低秩结构，但主方差方向不是幻觉判别方向。**
   Top-4 SVD 方向解释了很高的方差，但 AUROC 接近随机；真正的判别能力在 K=64/128/256、full difference 或 tail/residual coordinates 中出现。你的 Stage C/P 都支持这个结论，尤其多种子结果里 full difference 约 0.72，而 L24 top-4 只有约 0.47。

3. **Residual/tail 坐标更像 evidence-sensitive geometry。**
   Top-backbone 主要区分“有没有图像”，而不是“图像证据是否支持回答”；tail band 和 supervised decision view 更能区分 matched / random / adversarial evidence。

4. **L24 tail 对正确拒绝有因果必要性，但 FP rescue 很弱。**
   消融 L24 的 tail slice 可以剂量依赖地把 TN 推向 Yes，说明这些坐标对正确 negative decision 很重要；但把 FP 救回来很难，64 个 FP 中只 rescue 了 3 个，主要是 borderline case。

5. **应用上，geometry 更适合做 selective routing，而不是 standalone detector。**
   在 predicted-Yes 子集上，low-margin + geometry 可以把 warning precision 从 base rate 约 0.089 提升到 0.324–0.327；gated ICD/VCD 也能在减少 FP 的同时更好保护 TP，但 strict transfer 和 AMBER 上效果较弱，所以不能过度声称泛化。

---

# 三、推荐汇报结构：12 页，15–20 分钟

下面是我建议的完整汇报 PPT 结构。你可以直接按这个做。

---

## Slide 1：标题页

**标题：**

> 基于盲参考差分的 VLM 幻觉校正几何分析

**副标题：**

> Dominant visual correction directions are not hallucination decision directions

**你讲的时候说：**

“最近这段时间我主要在研究一个问题：当视觉语言模型发生幻觉时，相关信号到底在 hidden state 的什么位置？我们通常会认为模型看到图像后，hidden state 的主要变化方向最重要。但我的实验发现，主方差方向并不是幻觉判别方向，真正有用的信号反而在 residual/tail 坐标中。”

---

## Slide 2：研究问题

**标题：**

> 研究问题：视觉证据如何改变 VLM 的内部表示？

**页面内容：**

传统问题：

> 模型为什么在不存在物体时仍然回答 Yes？

我的切入点：

> 不直接从输出 token 入手，而是比较模型在两种条件下的 hidden state。

两种条件：

```text
Image + Question  → z_img
Question only     → z_blind
Difference        → d = z_blind - z_img
```

核心问题：

> 这个 difference space 中，哪些方向与 hallucination 有关？

**你讲的时候说：**

“我不是先把它当作一个检测器，而是把它当作机制分析工具。对于同一个问题，我分别让模型在图文条件和纯文本条件下前向传播，然后比较两者 hidden state 的差异。这个差异可以理解为视觉证据对模型内部状态施加的 correction。”

---

## Slide 3：任务边界，一定要讲清楚

**标题：**

> 先区分三个不同问题

**页面表格：**

| 任务        | 定义                                | 作用     | 不能说明什么              |
| --------- | --------------------------------- | ------ | ------------------- |
| FP vs TN  | ground-truth=No 中，错误 Yes vs 正确 No | 机制分析   | 不是部署检测任务            |
| FP vs TP  | predicted-Yes 中，错误 Yes vs 正确 Yes  | 部署风险识别 | 不等价于 FP/TN          |
| FP rescue | 通过干预把 FP 改成 No                    | 因果修正探索 | 目前不能证明可靠 mitigation |

**你讲的时候说：**

“这个边界非常重要。FP vs TN 是我的主要机制显微镜，它能告诉我们正确拒绝和错误接受在内部几何上有什么区别。但真实部署时，我们不知道 ground truth，所以真正部署相关的是 predicted-Yes 里的 FP vs TP。至于把 FP 直接修回来，这是更强的干预问题，目前结果比较弱。”

这一页很关键，它能防止听众误解你在“做一个新检测器”。

---

## Slide 4：方法总览

**标题：**

> Blind-Reference Differencing

**页面内容：**

流程图建议：

```text
Input image + question
        ↓
VLM forward
        ↓
z_img at layer L

Question only
        ↓
VLM forward
        ↓
z_blind at layer L

d = z_blind - z_img
        ↓
SVD / tail bands / supervised subspaces / intervention / gate
```

**主要分析对象：**

* Layers: L16 / L20 / L24 / L28 / L32
* Readout: last prompt token, last-4 mean, last-8 mean
* Main task: FP vs TN on POPE
* External: AMBER
* Cross-model: LLaVA-7B/13B, Qwen2-VL, InternVL

**你讲的时候说：**

“方法本身很简单，但后面所有实验都围绕这个 difference matrix 展开。我们先看它的谱结构，再看哪些坐标能区分 FP/TN，再做条件对照、因果干预、跨模型和选择性路由。”

---

## Slide 5：发现一：correction space 有强低秩结构

**标题：**

> Finding 1: Blind-image differences form a strong low-rank backbone

**页面内容：**

可以放一张 explained variance curve。

文字：

* Top-4 SVD directions explain a large fraction of variance.
* L8/L24/L32 top-4 explained variance roughly 72.7%–88.6%.
* Split-half stability is strongest at small K.

**你讲的时候说：**

“第一步我发现 difference matrix 有非常强的低秩结构。也就是说，模型从 question-only 到 image-question 的变化，并不是均匀分布在所有维度上，而是被少数主方向支配。直觉上这似乎说明 top SVD directions 很重要，但接下来的结果正好相反。”

---

## Slide 6：发现二：主方差方向不是幻觉判别方向

**标题：**

> Finding 2: Variance is not discrimination

**页面内容：**

建议放一个最重要的图：

* x-axis: K 或 explained variance
* left/right: explained variance vs AUROC
* 标出 top-4 方差高但 AUROC 低

核心数字：

| Feature                  |                        结果 |
| ------------------------ | ------------------------: |
| Top-4 explained variance |     very high, often >80% |
| L24 top-4 AUROC          |               around 0.47 |
| Full difference AUROC    | around 0.72 in multi-seed |
| Top-256 > Top-4          |        robust improvement |

**你讲的时候说：**

“这是目前最核心的发现。虽然 top-4 方向解释了大部分方差，但它们几乎不能区分 FP/TN，甚至在一些层上低于随机。而随着 K 增大到 64、128、256，判别能力才逐渐出现。多种子实验里 full difference 一直最强，而 top-4 非常弱。这说明主方差方向不是 hallucination decision direction。”

这一页是整场汇报的核心，必须讲慢一点。

---

## Slide 7：为什么会这样？Top backbone 主要编码“有没有图”，不是“证据是否支持回答”

**标题：**

> Interpreting the mismatch: top backbone tracks image-conditioning, not evidence correctness

**页面内容：**

用一个直观分解：

[
d = a v_{\text{image}} + b v_{\text{evidence}} + \epsilon
]

解释：

* (v_{\text{image}})：图像条件带来的大变化；
* (v_{\text{evidence}})：证据是否支持回答；
* (a) 方差大，所以 SVD 抓到 (v_{\text{image}})；
* FP/TN 标签主要依赖 (v_{\text{evidence}})，所以 top variance 不判别。

配合 Stage B 结果：

* Top-backbone 能分离 image-conditioned vs blind；
* 但不能分离 matched vs wrong evidence；
* Tail/supervised view 对 matched/random/adversarial 更敏感。

**你讲的时候说：**

“我认为这个现象可以这样理解：模型看到图像后，最大的 hidden-state 变化可能只是‘进入视觉条件模式’，比如场景、视觉属性、图像存在性。但幻觉判别需要的是更细粒度的问题：图像证据是否支持当前语言判断。这个信号方差可能不大，所以不会出现在 top SVD directions，而是分布在 tail 或 residual 坐标里。”

---

## Slide 8：发现三：Residual/Tail coordinates 更接近 evidence-sensitive signal

**标题：**

> Finding 3: Residual/tail coordinates are more evidence-sensitive

**页面内容：**

展示 matched / random / adversarial 条件对比。

可以放表：

| View                      | 观察                              |
| ------------------------- | ------------------------------- |
| Top-backbone energy       | 区分 image-conditioned vs blind   |
| Tail band 257–1024        | 对 matched vs mismatch 更敏感       |
| Supervised decision score | 只在 matched evidence 下明显区分 FP/TN |
| TN matched-specific tail  | 强于 FP                           |

**你讲的时候说：**

“为了确认 tail 不是随机噪声，我做了条件几何分析：matched image、random image、adversarial image 和 blind。结果显示 top-backbone 更像是在看有没有图，而 residual/tail 和 supervised decision score 更能反映证据条件是否匹配。尤其是 TN 样本中 matched-specific tail 增强更明显，这支持 tail 坐标与正确拒绝有关。”

---

## Slide 9：发现四：Tail coordinates 对正确拒绝有因果必要性

**标题：**

> Finding 4: Tail ablation causally disrupts correct negative decisions

**页面内容：**

放 L24 tail ablation dose curve：

| Alpha | L24 Yes Rate | Median Margin |
| ----: | -----------: | ------------: |
|     4 |        0.000 |        -0.750 |
|     5 |        0.125 |        -0.328 |
|     6 |        0.562 |        +0.016 |
|     7 |        0.938 |        +0.391 |
|     8 |        1.000 |        +0.934 |

再放一句：

> Norm-matched random tail control keeps Yes rate at 0 under last-token setting.

**你讲的时候说：**

“这部分是目前最干净的因果证据。我对 L24 的 tail slice 做 ablation，发现随着 alpha 增大，原本正确回答 No 的 TN 会逐渐翻转成 Yes。也就是说，这些 tail 坐标对模型保持正确拒绝是必要的。注意这里我只说 necessary for correct negative decisions，不说它能可靠修复幻觉。”

---

## Slide 10：负结果：FP rescue 很弱，所以不能包装成 mitigation

**标题：**

> Negative result: rescue is boundary-local, not reliable mitigation

**页面内容：**

核心结果：

* 64 FP samples: only 3 rescued
* Stage M: 2/32 FP rescued
* 30/32 margin improved but answer unchanged
* rescued samples are borderline cases

结论：

> Tail coordinates are necessary for TN, but not sufficient for robust FP rescue.

**你讲的时候说：**

“我也尝试了反方向：既然 tail 对正确拒绝重要，那能不能把 FP 拉回 No？结果很弱。虽然很多样本的 logit margin 有改善，但 decoded answer 大多不变，真正翻转的基本都是 margin 很小的 borderline case。所以这项工作不能定位为一个可靠的 hallucination mitigation 方法。它更适合作为机制分析。”

这页要主动讲，这样显得诚实，也避免老师/审稿人追问时被动。

---

## Slide 11：部署视角：geometry 可用于 selective warning/routing

**标题：**

> Deployment view: geometry helps selective routing, especially with margin

**页面内容：**

讲 predicted-Yes FP vs TP。

核心表：

| Method             |     Trigger |   FP Recall |   TP Damage | Warning Precision |
| ------------------ | ----------: | ----------: | ----------: | ----------------: |
| Random             | 0.156–0.223 | 0.155–0.233 | 0.156–0.222 |       0.088–0.093 |
| PLS only           |       0.156 |       0.396 |       0.133 |             0.226 |
| Low-margin + PLS   |       0.176 |       0.642 |       0.131 |             0.324 |
| Low-margin + FullD |       0.180 |       0.660 |       0.133 |             0.327 |

再放 gated ICD/VCD：

| Operator + Gate            | FP Reduction | TP Preserved | Acc Delta |
| -------------------------- | -----------: | -----------: | --------: |
| ICD blind + low-margin+PLS |        0.321 |        0.965 |    -0.001 |

**你讲的时候说：**

“虽然 geometry 不是最强 standalone detector，但它在部署上有一个比较实际的用法：作为 selective gate。也就是只在模型预测 Yes 的样本里，挑出更可疑的部分做 warning、abstention 或 VCD/ICD。最稳定的结果是 low-margin 加 geometry，能把 warning precision 从 base rate 约 0.089 提升到 0.32 左右。对 ICD/VCD 来说，选择性触发也能避免 always-on 对 TP 的损害。”

---

## Slide 12：跨模型和边界：不是普适神药

**标题：**

> Cross-model audit: partial transfer and important failure modes

**页面内容：**

总结：

| Model                   | 观察                               |
| ----------------------- | -------------------------------- |
| LLaVA-1.5-7B            | 主实验成立                            |
| LLaVA-1.5-13B           | 方差-判别解耦复现                        |
| Qwen2-VL / Qwen2.5-VL   | geometry 中等有效，但弱于 margin entropy |
| InternVL2 / InternVL2.5 | FP/TN 几乎完美，但 FP/TP 部署失败          |

重点句：

> FP/TN separability does not necessarily imply deployable hallucination-risk separability.

**你讲的时候说：**

“跨模型结果提醒我们，这不是一个普适神药。LLaVA-family 的模式比较稳定，Qwen 上有中等信号，但 InternVL 是很重要的失败案例：它在 FP/TN 上几乎完美，但在 predicted-Yes 的 FP/TP 任务上失败。这说明内部几何可分性不等于真实部署风险可分性。这一点反而帮助我们更清楚地定义问题边界。”

---

## Slide 13：总结页

**标题：**

> Takeaways

**页面内容：**

只放 5 句话：

1. Blind-reference differencing gives a useful lens for studying visual-evidence correction.
2. Dominant correction directions explain variance, but not hallucination decisions.
3. Hallucination-relevant signals live in residual/tail and evidence-sensitive coordinates.
4. Tail coordinates are causally necessary for correct negative decisions, but FP rescue is weak.
5. Geometry is best positioned as mechanistic evidence and a complementary selective-routing signal.

**你讲的时候说：**

“总结来说，我现在不会把这项工作包装成一个新的强幻觉修正方法。更准确的定位是：它揭示了 VLM 中 visual evidence correction 的分层结构，特别是主方差方向和幻觉判别方向之间的解耦。这个发现有机制价值，也能为选择性风险路由提供辅助信号。”

---

## Slide 14：下一步工作

**标题：**

> Next steps

**页面内容：**

建议只讲 4 个，不要太多：

1. **补 margin-only vs geometry-only vs margin+geometry 的统一对照。**
   解决“是不是 margin 已经够了”的问题。

2. **扩大 tail ablation 因果实验。**
   从 pilot 变成更稳的主文证据，例如更多 TN 样本和 LLaVA-13B 复现。

3. **进一步审计 InternVL failure case。**
   解释为什么 FP/TN 可分但 FP/TP 部署失败。

4. **补一个开放式生成幻觉评估。**
   避免工作被认为只是 POPE yes/no artifact。

**你讲的时候说：**

“下一步我不准备盲目增加很多新实验，而是围绕审稿人最可能质疑的地方收口：强基线、因果证据、跨架构边界和开放式外部评估。”

---

# 四、如果时间只有 8–10 分钟，就压缩成 7 页

如果这是组会简短汇报，用下面这个版本：

1. **Motivation：VLM 幻觉信号藏在哪里？**
2. **Method：blind-reference differencing**
3. **Main finding：variance ≠ discrimination**
4. **Evidence-sensitive tail geometry**
5. **Causal tail ablation + weak rescue**
6. **Selective routing utility + limitations**
7. **Takeaways & next steps**

不要讲所有 Stage 名称。Stage 名称只适合放 appendix。

---

# 五、汇报时最容易讲乱的地方，我帮你改成通顺说法

## 原来容易讲乱的版本

“我先做了 Stage A，然后 Stage C，接着 Stage B，之后 E、G、J、K、L、M、N、P、O、T……”

这样听众会完全迷失。

## 改成故事线版本

“我围绕一个问题展开：blind-image difference 里哪些方向与幻觉有关？
第一步，我发现这个 difference space 有强低秩结构。
第二步，我发现低秩主方向并不判别幻觉，这就是方差-判别解耦。
第三步，通过条件几何和 tail 分析，我发现证据相关信号主要在 residual/tail 坐标。
第四步，通过 tail ablation，我验证这些坐标对正确拒绝有因果必要性。
第五步，我进一步看部署场景，发现 geometry 单独不是最强 detector，但可以辅助 selective warning 和 VCD/ICD routing。
最后，跨模型结果表明这个现象有边界，尤其 InternVL 暴露了 FP/TN 与 FP/TP 的差异。”

这个版本清晰很多。

---

# 六、这次汇报应该避免的说法

这些话不要说：

> “我们提出了一个新的幻觉检测方法，效果很好。”

不稳，因为 margin/entropy baseline 很强。

> “我们可以修正幻觉。”

不稳，因为 FP rescue 很弱。

> “这个机制在所有 VLM 上普遍成立。”

不稳，因为 InternVL 是 failure case。

> “Top SVD direction 是 grounding direction。”

不稳，因为你的结果显示 top-backbone 主要是 image-conditioning，不是 evidence correctness。

建议说：

> “我们用 blind-reference differencing 作为机制分析工具。”
> “主方差方向不是幻觉判别方向。”
> “Residual/tail coordinates 更接近 evidence-sensitive correction。”
> “Tail 对正确拒绝有因果必要性，但不等于可以可靠 rescue FP。”
> “Geometry 更适合作为 margin/confidence 的补充路由信号。”

---

# 七、可以直接作为汇报开场白

你可以这样开头：

> “今天我汇报最近关于 VLM 幻觉内部机制的一组实验。最开始我的想法是：当模型看到图像之后，相比只看到问题，它的 hidden state 会发生某种 correction。如果这种 correction 与视觉证据有关，那么幻觉样本和非幻觉样本的 correction geometry 应该不同。
>
> 所以我构造了一个 blind-reference difference，也就是用 question-only 的 hidden state 减去 image-question 的 hidden state。然后我围绕这个 difference space 做了谱分析、FP/TN probe、条件几何、因果干预、跨模型验证和选择性路由。
>
> 目前最核心的发现可以概括为一句话：图像带来的最大变化方向，并不是幻觉判别方向。真正与幻觉相关的信号，更多出现在 residual/tail correction coordinates 中。”

---

# 八、可以直接作为汇报结尾

结尾可以这样说：

> “总的来说，这项工作的价值不在于提出一个直接超过 output confidence 的幻觉检测器，而在于提供了一个机制视角：VLM 的 visual correction space 是分层的。低秩 backbone 解释了大部分图文条件差异，但它主要反映是否进入视觉条件模式；真正与正确拒绝和幻觉风险相关的信号，分布在 residual/tail 和 evidence-sensitive coordinates 中。
>
> 这也解释了为什么简单看 top SVD directions 会错过关键判别信号。后续我会重点补强三个方面：第一，和 margin-only 的严格对照；第二，扩大 tail ablation 的因果实验；第三，进一步解释跨架构 failure case，尤其是 InternVL 中 FP/TN 可分但 FP/TP 失败的问题。”

---

# 九、我建议你最终汇报的“一句话中心论点”

你整场汇报必须反复回到这句话：

> **VLM 中最大的 visual correction directions 并不是 hallucination decision directions；幻觉相关信号主要藏在 residual/tail evidence-sensitive coordinates 中。**

只要听众记住这句话，你这次汇报就是成功的。
