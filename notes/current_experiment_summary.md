下面我给你一版优化后的计划书，重点修正五点：

1. 明确 probe / calibration / test 划分，避免泄露；
2. 明确 gate 部署对象是“模型预测 Yes 的样本”，因此必须评估 TP Damage；
3. 强调 geometry gate 的价值不是 verification prompt 本身，而是“比 random/margin 更精准地选择该修正的样本”；
4. 实验分成必做、应做、有余力再做；
5. 贡献表述从“routing”升级为“complementary risk signal”。

---

# 计划书 v2：Correction-Geometry Guided Selective Correction

## 1. 核心目标

本文不把 correction geometry 写成一个强检测器，也不把它写成可靠 mitigation 方法，而是将其落实为一种**选择性修正策略**：

> Correction geometry provides a complementary internal risk signal to output confidence, enabling selective verification or contrastive correction with lower false-trigger rates than margin-based or random routing.

中文理解：

> 校正几何提供了一个与输出置信度互补的内部风险信号，可以帮助我们判断哪些样本需要额外验证或对比解码，从而减少不必要的干预，并尽量降低对正确 Yes 样本的误伤。

这个目标和你已有发现是匹配的：你已经证明 top variance 不是 hallucination decision geometry，FP/TN 信号更多分布在 full difference、PLS/Fisher、residual/tail coordinates 中；但同时你也知道 logits margin 很强、FP rescue 不可靠，所以落地方式必须是“选择性处理”，而不是“强行全局修复”。

---

# 2. 方法设计

## 2.1 基本表示

对每个样本提取两种 hidden state：

```text
z_img   = hidden_state(image + question)
z_blind = hidden_state(question only)
```

定义 blind-reference correction difference：

```text
d = z_blind - z_img
```

主层使用：

```text
L24
```

辅助层：

```text
L20 / L32
```

L24 作为主层，是因为你已有结果中 L24 full difference、PLS、tail ablation 都比较关键；L32 可以作为 late-layer arbitration 的补充。

---

## 2.2 Geometry risk score

建议保留三个主分数，不要太多。

### Score A：Full-Difference Risk

```text
s_full = Logistic(d)
```

训练任务仍然是：

```text
FP vs TN
```

也就是在 ground-truth = No 的样本中区分：

```text
FP: 模型错误回答 Yes
TN: 模型正确回答 No
```

但这里要明确：**这个 probe 只是学习“错误 Yes 与正确 No 的内部差异”，部署时不能假设 ground truth 已知。**

---

### Score B：Tail / Residual Risk

对差分矩阵 `D` 做 SVD，取 residual/tail band：

```text
P_tail(d) = SVD coordinates 257-1024
```

构造：

```text
s_tail_energy = ||P_tail(d)||²
s_tail_probe  = Logistic(P_tail(d))
```

这对应你的机制发现：matched evidence 与 mismatch 的差异更明显体现在 residual/tail，而不是 top variance backbone。

---

### Score C：PLS / Fisher Risk

```text
s_pls = w_pls^T d
```

主设定可以用：

```text
L24, K=32 PLS
```

因为 PLS/Fisher 可以作为更 compact 的 FP/TN decision subspace，但它的稳定性可能弱于 full difference，所以建议作为对照分数，而不是唯一主分数。

---

# 3. Gate 设计：必须对齐真实部署场景

这是计划书 v2 最重要的修改。

## 3.1 实际部署时你知道什么？

真实部署时你不知道 ground truth。你只知道：

```text
模型预测 Yes 或 No
模型输出 margin / entropy
geometry risk score
```

对于 POPE object hallucination，最重要的是：

```text
模型预测 Yes 的样本 = TP + FP
```

你的 gate 面对的是：

```text
TP: 正确 Yes
FP: 错误 Yes，即幻觉
```

所以 gate 的真正目标不是单纯区分 FP/TN，而是：

> 在模型预测 Yes 的样本中，尽量触发 FP，尽量不要触发 TP。

因此，实验中必须把 **TP Damage** 提升为核心指标。

---

## 3.2 Gate 公式

最小版本：

```text
G_geo(x) = 1 if model predicts Yes and s_geo(x) > τ_geo
```

组合版本：

```text
G_margin+geo(x) = 1 if model predicts Yes and 
                  (s_geo(x) > τ_geo or margin is uncertain / suspicious)
```

也可以测试更严格版本：

```text
G_strict(x) = 1 if model predicts Yes and 
              s_geo(x) > τ_geo and margin_yes > τ_margin
```

解释：

* `model predicts Yes`：只在可能发生 object hallucination 的样本上触发；
* `s_geo(x) > τ_geo`：内部 correction geometry 显示视觉证据修正异常；
* `margin`：输出置信度信号，用于和 geometry 做互补。

---

# 4. 数据划分与阈值校准

这部分必须写清楚。

建议使用下面这种清晰划分：

## 方案 A：按 POPE subset 划分

```text
Probe Train: POPE random
Calibration: POPE popular
Test: POPE adversarial
External: AMBER
```

用途：

| 数据               | 用途                                     |
| ---------------- | -------------------------------------- |
| POPE random      | 训练 FP/TN probe、SVD subspace、PLS/Fisher |
| POPE popular     | 选择 gate 阈值 τ，选择 trigger rate           |
| POPE adversarial | 固定所有参数后最终测试                            |
| AMBER            | 外部验证                                   |

优点：最干净，容易向审稿人解释。

缺点：random/popular/adversarial 分布不同，可能会让训练更难，但这反而更严格。

---

## 方案 B：按每个 subset 内部划分

如果你担心用 random 训练、adversarial 测试太难，也可以：

```text
Train: 每个 subset 的 50%
Calibration: 每个 subset 的 20%
Test: 每个 subset 的 30%
```

但要保证：

```text
训练 probe 的样本 ≠ 选择 τ 的样本 ≠ 最终测试样本
```

我更推荐方案 A，因为更清楚，也更能支撑 generalization。

---

# 5. 触发后的处理方式

## 5.1 必做：Geometry-Gated Verification

如果：

```text
G(x) = 0
```

保留原始回答。

如果：

```text
G(x) = 1
```

触发 verification prompt：

```text
Please verify the visual evidence before answering.
Answer "Yes" only if the queried object is clearly visible in the image.
If the visual evidence is insufficient or unclear, answer "No".

Question: {question}
```

这里要明确分离两个贡献：

### 贡献 1：Verification prompt 本身是否有效？

比较：

```text
Original vs Always Verification
```

这说明“换 prompt 重新检查”本身能不能降低 FP。

### 贡献 2：Geometry gate 是否精准？

比较：

```text
Geometry-gated Verification vs Random-gated Verification
```

要求它们有相同 trigger rate。

这才是你的真正贡献：

> 在同样只触发 20% 样本的情况下，geometry gate 是否比 random gate 找到了更多 FP、更少误伤 TP？

---

## 5.2 有余力再做：Geometry-Gated VCD / ICD

如果 gated verification 有正结果，再做 VCD/ICD。

不要先做 hidden-state filtered decoding，工程风险太高。先做 gated VCD：

```text
G(x) = 0: 原始 greedy decoding
G(x) = 1: 使用 VCD / ICD
```

核心比较：

```text
Original
Always-on VCD
Margin-gated VCD
Geometry-gated VCD
Margin+Geometry-gated VCD
Random-gated VCD
```

重点不是超过 always-on VCD，而是：

> 用更低 trigger rate 达到接近的 FP reduction，或者在相同 FP reduction 下减少 TP damage / 额外计算。

---

# 6. 核心指标

## 6.1 Gate 精准性指标

这是最重要的一组。

| 指标                            | 含义               |
| ----------------------------- | ---------------- |
| Trigger Rate                  | 触发比例             |
| Triggered FP Ratio            | 触发样本里 FP 占比      |
| FP Recall among predicted-Yes | 在所有 FP 中抓住了多少    |
| TP Damage                     | 被误触发的 TP 比例      |
| Precision of Gate             | 触发样本中真正需要修正的比例   |
| FP Reduction per Trigger      | 每触发一个样本带来的 FP 降低 |

其中最关键的是：

```text
FP Reduction ↑
TP Damage ↓
```

你要把它们并列作为主指标。

---

## 6.2 最终回答质量指标

| 指标            | 含义                       |
| ------------- | ------------------------ |
| Accuracy      | 总准确率                     |
| F1            | yes/no 平衡性能              |
| Precision     | Yes 回答的可靠性               |
| Recall        | 正确 Yes 保留程度              |
| FP Rate       | 幻觉率                      |
| TN Preserved  | 正确 No 是否保留               |
| TP Preserved  | 正确 Yes 是否保留              |
| Extra Compute | 额外 forward / decoding 成本 |

注意：
如果 verification prompt 降低 FP 但大量伤害 TP，不能说方法成功。你要强调 trade-off。

---

# 7. 必做实验

## 实验 1：Geometry 是否补充 margin？

这是最小闭环第一步。

比较：

```text
margin-only
geometry-only
margin + geometry
```

建议 geometry 分数只选三个：

```text
s_full
s_tail_probe
s_pls
```

指标：

```text
AUROC
AUPRC
Risk-Coverage AUC
FP@fixed coverage
TP Damage under predicted-Yes gate
```

注意：
这里不能只看 FP/TN AUROC。必须额外看：

```text
在 predicted-Yes = TP + FP 子集上，
geometry 能不能区分 FP 和 TP？
```

这正是那份意见指出的漏洞。

---

## 实验 2：Gated Verification 主实验

触发率建议固定三个：

```text
10%
20%
30%
```

在每个触发率下比较：

```text
Random-gated Verification
Margin-gated Verification
Geometry-gated Verification
Margin+Geometry-gated Verification
```

主结论优先级：

### 第一优先级

```text
Geometry-gated 是否在相同 trigger rate 下比 Random-gated 抓住更多 FP、误伤更少 TP？
```

### 第二优先级

```text
Geometry-gated 是否优于 Margin-gated？
```

### 第三优先级

```text
Margin+Geometry 是否优于 Margin-only？
```

这个顺序很重要。因为你的方法不一定总能打败 margin，但至少应该比 random gate 显著更精准。

---

## 实验 3：Gate 来源消融

比较：

```text
Top-4 SVD gate
Tail 257-1024 gate
Full-difference gate
PLS/Fisher gate
Random-subspace gate
```

目标：

> 证明不是任意 hidden feature 都能 gate，而是 residual/tail/full-difference 这些由机制分析识别出的坐标更有效。

这能直接呼应你的核心发现：

> dominant variance directions are not decision geometry.

如果 top-4 gate 明显弱，而 tail/full/PLS gate 更好，这就是非常漂亮的“机制指导方法设计”证据。

---

# 8. 应做实验

## 实验 4：Margin-bin / Overconfidence 分析

按 yes/no margin 分桶：

```text
low-margin
medium-margin
high-margin
```

在每个桶里分析：

```text
geometry score 的 FP/TP 区分能力
geometry-gated 的 FP recall
geometry-gated 的 TP damage
```

预期有三种写法。

### 理想情况

如果 high-margin FP 中 geometry 仍有效：

> correction geometry detects overconfident hallucinations missed by output confidence.

### 中等情况

如果主要在 medium-margin 样本有效：

> correction geometry is most useful for boundary-near hallucination handling.

### 较差情况

如果没有明显分桶优势：

> geometry provides limited complementarity to output confidence, suggesting that hidden correction geometry and output confidence are partially aligned in this setting.

---

## 实验 5：LLaVA-1.5-13B 复现

你已有 13B 相关结果，可以低成本补一张表：

```text
7B gate result vs 13B gate result
```

重点看：

```text
top-4 gate 是否仍弱
tail/full/PLS gate 是否仍更有效
geometry-gated 是否仍比 random-gated 精准
```

这能把结论从 7B 稍微推到 checkpoint-level recurrence。你的已有总结里也说明，13B 复现了 full difference 强、top-4 弱、tail gap 存在这些 qualitative pattern。

---

# 9. 有余力再做实验

## 实验 6：Gated VCD / ICD

只有在 Gated Verification 有效果后再做。

比较：

```text
Original
Always VCD
Random-gated VCD
Margin-gated VCD
Geometry-gated VCD
Margin+Geometry-gated VCD
```

核心指标：

```text
FP Reduction
TP Damage
Trigger Rate
Extra Compute
FP Reduction per Trigger
```

---

## 实验 7：AMBER 外部验证

用 POPE 学到的 score / threshold，在 AMBER 上测试：

```text
risk transfer
predicted-positive risk
gated verification
```

语气要克制。你已有 AMBER transfer 是 above-chance but modest，不适合写成强泛化。

---

## 实验 8：Qwen2-VL / InternVL

如果想冲更高上限，再补一个跨架构模型。

最小只跑：

```text
top-4 vs full/tail risk
predicted-Yes FP/TP gate precision
gated verification at 10/20/30%
```

不要一开始做完整 intervention。

---

# 10. 实验优先级总表

## 必做

| 实验                                    | 目的                           |
| ------------------------------------- | ---------------------------- |
| margin vs geometry vs margin+geometry | 证明 geometry 是否有互补风险信号        |
| predicted-Yes 子集 FP/TP 分析             | 修复 probe 训练目标与部署场景不匹配的问题     |
| Gated Verification                    | 证明 geometry gate 有实际 utility |
| Random gate 对照                        | 证明不是随机触发就有效                  |
| Top-4 vs tail/full/PLS gate           | 证明机制发现能指导 gate 设计            |

---

## 应做

| 实验                    | 目的                             |
| --------------------- | ------------------------------ |
| Margin-bin analysis   | 说明 geometry 在哪类样本上有价值          |
| LLaVA-1.5-13B gate 复现 | 增强 checkpoint-level generality |

---

## 有余力再做

| 实验                  | 目的     |
| ------------------- | ------ |
| Gated VCD / ICD     | 更强落地版本 |
| AMBER               | 外部验证   |
| Qwen2-VL / InternVL | 跨架构泛化  |

---

# 11. 最小可执行闭环

你现在最应该先做这个闭环：

```text
1. 用 POPE random 训练 full/tail/PLS geometry risk。
2. 用 POPE popular 选 threshold，使 trigger rate = 10%、20%、30%。
3. 在 POPE adversarial 上测试 predicted-Yes 子集中的：
   - FP recall
   - TP damage
   - gate precision
4. 做 Gated Verification：
   - random gate
   - margin gate
   - geometry gate
   - margin+geometry gate
5. 做 top-4 vs tail/full/PLS gate 消融。
```

如果这个闭环成立，你就能写出一个非常清楚的落地贡献：

> Correction geometry provides a complementary risk signal to output confidence and enables selective verification that catches more hallucinated Yes responses with lower TP damage than random or top-variance routing.

---

# 12. 最终论文贡献表述

建议改成四条：

1. We introduce blind-reference differencing to analyze visual-evidence correction geometry in LVLMs.
2. We show that dominant variance directions are not hallucination decision directions.
3. We identify residual/tail correction coordinates that are evidence-sensitive and causally relevant to faithful negative decisions.
4. We show that correction geometry provides complementary risk signal to output confidence, enabling selective verification with lower false-trigger rates than random or top-variance routing.

注意第 4 条不要写得太满。
不要说：

```text
we solve hallucination mitigation
```

而说：

```text
we enable selective verification / selective correction
```

---

# 13. 最终判断

这份意见里最重要的修改就是：

> 不能只训练 FP/TN probe，然后直接拿去 gate predicted-Yes 样本；必须显式评估 TP 是否被误伤。

这个修改非常关键，也很容易做。

所以优化后的计划书核心变成：

```text
机制发现：
top variance ≠ decision geometry；
residual/tail/full difference 更接近 hallucination-sensitive correction。

方法落地：
用这些 geometry score 在 predicted-Yes 样本中选择性触发 verification。

核心证明：
在相同 trigger rate 下，
geometry gate 比 random gate / top-4 gate 抓住更多 FP，
同时比粗糙 gate 更少伤害 TP。
```

只要这个结果能跑出来，你这篇文章就不会显得空洞了。

---

# 14. 2026-05-07 跨架构 user-content readout 结果分析

本轮结果位于：

```text
outputs/stage_o_cross_model_user_readout/
```

新增部署视角分析：

```text
outputs/stage_o_cross_model_user_readout/audit/predicted_yes_gate_summary.csv
outputs/stage_o_cross_model_user_readout/audit/predicted_yes_gate_trigger_rates.csv
```

## 14.1 读出修正确认有效

四个模型的 hidden readout 都是：

```text
last_user_content_token
```

Qwen2-VL / Qwen2.5-VL 的 FP/TN probe 不再出现之前接近 1.0 的异常结果：

| 模型 | best difference AUROC |
| --- | ---: |
| Qwen2-VL-7B | 0.772 |
| Qwen2.5-VL-7B | 0.771 |

这说明之前 Qwen 的异常高分基本来自 assistant generation prompt / next-token 读出混入，而不是稳定的跨模型 correction geometry。

## 14.2 InternVL 仍然异常强，需要单独审计

InternVL2 / InternVL2.5 的 raw_img 和 difference 在 FP/TN 上仍然接近完美：

| 模型 | best raw_img AUROC | best difference AUROC |
| --- | ---: | ---: |
| InternVL2-8B | 0.999 | 0.999 |
| InternVL2.5-8B | 0.998 | 0.998 |

但这个强结果不能直接解释成 mitigation 能力。部署视角下，在 test split 的 predicted-Yes 样本中，full difference 对 FP-vs-TP 是反向的：

| 模型 | difference FP-vs-TN AUROC | difference FP-vs-TP AUROC |
| --- | ---: | ---: |
| InternVL2-8B | 0.997 | 0.218 |
| InternVL2.5-8B | 0.998 | 0.126 |

top 10% geometry trigger 对 InternVL 两个模型都没有抓到 FP：

| 模型 | trigger | FP caught | TP damage |
| --- | ---: | ---: | ---: |
| InternVL2-8B | 53 | 0 | 53 |
| InternVL2.5-8B | 64 | 0 | 64 |

因此 InternVL 的 FP/TN separability 更像是在区分“模型会回答 Yes 还是 No”的内部状态，而不是能安全路由 predicted-Yes 幻觉样本的 risk signal。

## 14.3 部署 gate 的关键结论

在 predicted-Yes test subset 上，margin entropy / low yes-margin 目前比 geometry gate 更可靠：

| 模型 | best geometry FP-vs-TP AUROC | margin entropy FP-vs-TP AUROC |
| --- | ---: | ---: |
| Qwen2-VL-7B | 0.703 raw_blind / 0.675 difference | 0.869 |
| Qwen2.5-VL-7B | 0.765 raw_blind / 0.745 difference | 0.883 |
| InternVL2-8B | 0.698 raw_blind / 0.218 difference | 0.883 |
| InternVL2.5-8B | 0.697 raw_blind / 0.126 difference | 0.903 |

top 10% trigger 时：

| 模型 | score | FP recall | TP damage rate |
| --- | --- | ---: | ---: |
| Qwen2-VL-7B | difference | 0.286 | 0.093 |
| Qwen2-VL-7B | margin entropy | 0.571 | 0.082 |
| Qwen2.5-VL-7B | difference | 0.400 | 0.085 |
| Qwen2.5-VL-7B | margin entropy | 0.360 | 0.087 |
| InternVL2-8B | difference | 0.000 | 0.103 |
| InternVL2-8B | margin entropy | 0.500 | 0.086 |
| InternVL2.5-8B | difference | 0.000 | 0.109 |
| InternVL2.5-8B | margin entropy | 0.511 | 0.068 |

当前不能声称 geometry gate 已经优于 margin gate。更稳妥的表述是：

```text
Corrected user-content readout removes the Qwen prompt-readout artifact.
Cross-architecture evidence for FP/TN geometry is mixed:
Qwen shows moderate difference signal, while InternVL shows very strong FP/TN separability that does not transfer to predicted-Yes FP-vs-TP gating.
For deployment, margin entropy remains the strongest single gate; geometry should next be tested only as a complementary signal combined with margin, not as a replacement.
```

## 14.4 下一步

下一步应做真正的 selective gate 实验：

1. 训练目标改成 predicted-Yes 内的 FP vs TP，或至少报告 FP/TN-trained score 在 FP-vs-TP 上的表现。
2. 用 calibration split 选 threshold，不用 test split 选 top rate。
3. 比较 random、margin entropy、geometry、margin+geometry。
4. 主指标固定为 FP recall、TP damage、gate precision。
5. 对 InternVL 追加审计：检查 readout token、raw_img 是否含有输出决策状态，必要时改用更早层或 pre-answer-free 的 forward 设定。
