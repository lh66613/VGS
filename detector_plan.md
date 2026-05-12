
---

# 幻觉检测器完善计划书

## 一、总体目标

当前我们已经有一个基于 blind-reference differencing 的检测思路：

[
z_{\text{img}} = \text{hidden state}(\text{image + question})
]

[
z_{\text{blind}} = \text{hidden state}(\text{question only})
]

[
d = z_{\text{blind}} - z_{\text{img}}
]

已有结果显示，blind-reference difference 中确实存在 FP/TN 判别信号，并且 full difference、多种子下的 full-diff probe、PLS/Fisher、tail/residual 子空间等都有一定效果；同时 top-4 主方差方向虽然解释大量方差，但判别能力弱。已有总结中也记录了：full difference 在 5 seeds 上平均 AUROC 约 0.721，而 L24 top-4 约 0.471；Stage T 中 low-margin+geometry 在 predicted-Yes 设置下能把 warning precision 从 base rate 约 0.089 提升到约 0.324–0.327。

但是目前 detector 还没有成为一个严格完整的方法，因为还缺少：

1. 和其他**需要训练的检测器**公平比较；
2. 不只是 AUROC，还要有 ACC、F1、AUPRC、速度、显存、训练成本等指标；
3. 证明 correction subspace 的必要性；
4. 证明不是直接在原 hidden representation 上训练 probe 就足够；
5. 区分机制任务 FP/TN 和部署任务 FP/TP；
6. 给出统一协议、统一表格和清晰结论。

本计划的最终目标是把检测器路线补强到可以回答：

> 我们的 detector 到底在什么任务上有效？相比 raw representation probe、output confidence probe、普通 PCA/SVD/随机子空间，它的优势是什么？subspace 是否必要？它是否具有实际部署价值？

---

# 二、检测器路线的最终定位

建议不要把 detector 写成：

> 我们提出了最强幻觉检测器。

而应写成：

> 我们提出并系统评估了一类基于 blind-reference correction geometry 的 hallucination risk detector。它的价值不只在于分数，而在于把 output confidence、raw hidden-state probe 和 correction-subspace probe 统一比较，证明 residual/tail/evidence-sensitive subspace 在某些场景下提供了更稳定、更低维、更可解释、可用于选择性路由的风险信号。

也就是说，检测器的主张分三层：

## 主张 1：paired correction difference 比单独 raw hidden state 更有信息

要证明：

[
d = z_{\text{blind}} - z_{\text{img}}
]

比单独使用：

[
z_{\text{img}},\quad z_{\text{blind}}
]

更适合捕获 hallucination-relevant signal。

---

## 主张 2：子空间不是装饰，而是有必要性

要证明：

* top variance directions 不够；
* residual/tail 或 supervised evidence-sensitive subspace 更相关；
* subspace 可以在低维下接近或超过 full representation；
* subspace 在稳定性、迁移性、解释性、速度上有优势。

---

## 主张 3：部署场景中，geometry detector 不是替代 margin，而是补充 margin

目前已有证据显示 margin/entropy 非常强，尤其在某些子集上 yes/no margin 甚至达到 AUROC=1.000，binary entropy 也很强；所以不能硬说 geometry 全面优于 output confidence。

更稳的目标是证明：

> margin-only 已经强，但 margin+geometry 在 predicted-Yes 风险筛选、warning precision、TP damage 控制、选择性 VCD/ICD routing 上提供额外收益。

---

# 三、任务定义

必须先把 detector 任务拆开，否则容易混乱。

## Task A：FP vs TN，机制型检测任务

**定义：**

在 ground-truth=No 的样本中：

* FP：模型错误回答 Yes；
* TN：模型正确回答 No。

**意义：**

这是 object hallucination 最直接的形式：

> 图像里没有某物，但模型说有。

**用途：**

用来分析 correction geometry 是否能区分“错误接受”和“正确拒绝”。

**注意：**

这不是完整部署任务，因为实际部署时我们不知道 GT=No。

---

## Task B：FP vs TP，部署型风险检测任务

**定义：**

在模型预测 Yes 的样本中：

* FP：模型说 Yes，但 GT=No；
* TP：模型说 Yes，且 GT=Yes。

**意义：**

这是实际部署中最关键的任务：

> 模型已经说 Yes，我如何判断这个 Yes 是否可疑？

**用途：**

用于 warning、abstention、selective VCD/ICD、人工复核触发。

**这是 detector 路线最应该重点补强的任务。**

---

## Task C：Error vs Correct，整体错误检测任务

**定义：**

所有样本中：

* Error：FP + FN；
* Correct：TP + TN。

**意义：**

评估 detector 是否只是 hallucinated-Yes 专用，还是可以做一般 correctness detection。

**用途：**

作为补充结果，不建议作为主任务。

---

## Task D：FN vs TP，对称对照任务

**定义：**

在 ground-truth=Yes 的样本中：

* FN：模型错误回答 No；
* TP：模型正确回答 Yes。

**意义：**

检查你的 correction geometry 是 hallucination-specific，还是 general correctness signal。

**用途：**

作为 specificity control。

---

# 四、数据集与划分协议

## 4.1 主数据集：POPE COCO

使用 POPE random / popular / adversarial 三个 split。

建议设置两套协议。

---

## Protocol 1：Strict subset-transfer protocol

这是最严格、最应该放主文的协议。

```text
Train: POPE random
Calibration: POPE popular
Test: POPE adversarial
```

用途：

* 训练 probe；
* 在 calibration 上选择 threshold、K、layer、readout；
* 在 test 上只报告一次结果。

优点：

* 可以防止 cherry-picking；
* 更符合审稿标准；
* 能证明跨 split 泛化。

已有 Stage T 中已经有这个协议，例如 random → popular → adversarial 的 20% trigger 结果，但当前还需要补齐更多 baselines 和指标。

---

## Protocol 2：Fixed split / repeated stratified split

这是辅助协议。

建议：

* 5 seeds；
* train / val / test = 60 / 20 / 20；
* 保持 outcome 比例一致；
* 每个 seed 重新训练 probe；
* 报 mean ± std 和 95% CI。

用途：

* 检查统计稳定性；
* 得到更充分的 error bar；
* 和已有 Stage P 多种子鲁棒性衔接。

---

## 4.2 外部数据集：AMBER

AMBER 作为 external transfer。

设置：

```text
Train: POPE
Test: AMBER
```

不在 AMBER 上调参，最多只做：

* zero-shot transfer；
* 或者 train-on-POPE, calibrate-on-small-AMBER-val, test-on-AMBER-test。

已有结果显示 POPE-trained geometry 到 AMBER 的 AUROC 大约在 0.63–0.665，属于 above chance but modest，需要如实报告。

---

## 4.3 可选外部数据集

如果时间够，可以补：

* HallusionBench；
* MMHal-Bench；
* CHAIR / COCO caption hallucination；
* MME hallucination subset。

但 detector 第一阶段不强制。先把 POPE + AMBER 做扎实。

---

# 五、模型范围

## 第一阶段：主模型

先用：

```text
LLaVA-1.5-7B
```

原因：

* 已有结果最完整；
* 实验成本低；
* 方便快速补齐 detector 表格。

---

## 第二阶段：checkpoint-level 复现

补：

```text
LLaVA-1.5-13B
```

已有结果显示 LLaVA-13B 上 full diff 强、top-4 弱，方差-判别解耦可以复现。

---

## 第三阶段：跨架构压力测试

再补：

```text
Qwen2-VL-7B
Qwen2.5-VL-7B
InternVL2-8B
InternVL2.5-8B
```

注意：

* Qwen 上 geometry 信号中等，但 margin entropy 更强；
* InternVL 在 FP/TN 上几乎完美，但 predicted-Yes FP/TP 失败，是重要 failure case。

检测器论文里一定要区分：

> FP/TN separability 不等于 deployable FP/TP risk separability。

---

# 六、特征组设计

这是本计划最核心的部分。导师问“子空间必要性”，就必须构造完整 feature family。

---

## Group 0：无训练 black-box baselines

这些必须放，而且应该放在第一组。

| 名称                            | 特征                                     | 说明                                   |   |      |
| ----------------------------- | -------------------------------------- | ------------------------------------ | - | ---- |
| Yes/No margin                 | (\ell_{\text{Yes}} - \ell_{\text{No}}) | 最强简单基线                               |   |      |
| Absolute margin               | (                                      | \ell_{\text{Yes}} - \ell_{\text{No}} | ) | 置信度型 |
| Binary entropy                | Yes/No 概率熵                             | 不确定性                                 |   |      |
| Max probability               | 最大 token probability                   | 通用置信度                                |   |      |
| Full-vocab entropy            | 全词表熵                                   | 生成不确定性                               |   |      |
| Answer length / refusal stats | 可选                                     | 作为弱基线                                |   |      |

目的：

> 证明我们的 detector 不是在回避最强简单置信度基线。

---

## Group 1：可训练 output-level baselines

这些是公平训练型检测器。

| 名称                          | 输入特征                        | 模型                             |
| --------------------------- | --------------------------- | ------------------------------ |
| Logistic-margin             | margin, entropy, max prob   | Logistic regression            |
| Logistic-logits             | Yes logit, No logit, margin | Logistic regression            |
| Top-k logits probe          | top-10 / top-50 logits      | Logistic / MLP                 |
| Confidence MLP              | confidence statistics       | 小 MLP                          |
| Calibrated confidence model | margin + entropy            | Logistic / temperature scaling |

目的：

> 如果只用输出层信息训练一个轻量模型，是否已经足够？

---

## Group 2：raw representation baselines

这是回答导师问题的关键。

| 名称                | 输入特征                                    |
| ----------------- | --------------------------------------- |
| Raw image probe   | (z_{\text{img}})                        |
| Raw blind probe   | (z_{\text{blind}})                      |
| Raw concat probe  | ([z_{\text{img}}; z_{\text{blind}}])    |
| Raw diff probe    | (d = z_{\text{blind}} - z_{\text{img}}) |
| Raw pair features | ([z_{\text{img}}; z_{\text{blind}}; d]) |

模型包括：

* Logistic regression；
* Ridge logistic；
* Linear SVM；
* 小 MLP，作为非线性上限。

目的：

> 证明直接在原表征空间做 probe 到底行不行。

如果 raw diff 最强，也没关系。那可以把贡献写成：

> blind-reference difference 本身是有效特征；subspace 进一步提供低维、稳定、解释性。

---

## Group 3：普通降维 baselines

这些用于防止审稿人说“你只是做了降维”。

| 名称                        | 说明                  |
| ------------------------- | ------------------- |
| Random projection         | 随机 K 维子空间           |
| PCA on (z_{\text{img}})   | 普通 image hidden PCA |
| PCA on (z_{\text{blind}}) | 普通 blind hidden PCA |
| PCA on (d)                | 普通 diff PCA         |
| PCA-whitened diff         | whiten 后 probe      |

目的：

> 排除“任意低维投影都可以”的可能。

---

## Group 4：correction-space subspace methods

这是你的方法族。

| 名称                   | 说明                            |
| -------------------- | ----------------------------- |
| Top-SVD K            | (d) 的 top-K SVD directions    |
| Mid-SVD band         | 例如 64–256                     |
| Tail band            | 例如 257–1024                   |
| Top-complement       | 移除 top directions 后的 residual |
| Full SVD coordinates | 完整 SVD 坐标                     |
| PLS FP/TN            | 监督判别子空间                       |
| Fisher / LDA         | 监督判别方向                        |
| Contrastive PCA      | matched-vs-mismatch 子空间       |

已有结果显示，top-4 虽有高方差但判别弱，而 full SVD coordinates、PLS、full difference 等更有效；Stage L 中 PLS FP/TN 是紧凑检测子空间里较强的方案。

---

## Group 5：combined detectors

最终部署可能不是 geometry-only，而是组合模型。

| 名称                | 输入                                  |
| ----------------- | ----------------------------------- |
| Margin + raw diff | output confidence + (d)             |
| Margin + PLS      | output confidence + PLS score       |
| Margin + tail     | output confidence + tail score      |
| Margin + fullD    | output confidence + full diff score |
| Stacked logistic  | 多组 score 融合                         |
| Two-stage gate    | low-margin 先筛，再 geometry 排序         |

目的：

> 证明 geometry 是否提供 margin 之外的互补信息。

这是最重要的一组。

---

# 七、模型训练设置

为了公平，所有训练型 detector 采用统一设置。

## 7.1 Linear probe

默认模型：

```text
Logistic Regression with L2 regularization
```

超参数：

* (C \in {0.01, 0.1, 1, 10, 100})
* class_weight = balanced / none 都试；
* solver = liblinear / lbfgs；
* threshold 在 calibration set 上选择。

---

## 7.2 MLP probe

作为高容量 probe 上限，不作为主方法。

结构建议：

```text
Input dim → Linear(256) → ReLU → Dropout(0.1) → Linear(1)
```

或者更简单：

```text
Input dim → Linear(64) → ReLU → Linear(1)
```

目的：

> 检查线性不可分是否限制性能。

---

## 7.3 Threshold selection

不能在 test 上选 threshold。

建议三种 threshold：

1. **F1-optimal threshold**：在 calibration 上最大化 F1；
2. **fixed trigger rate threshold**：例如 10%、20%、30%；
3. **high precision threshold**：例如 precision ≥ 0.5 时最大 recall。

部署任务中优先 fixed trigger rate，因为最直观。

---

# 八、评价指标

不能只报 AUROC。

## 8.1 Ranking metrics

| 指标                | 用途                    |
| ----------------- | --------------------- |
| AUROC             | 总体排序能力                |
| AUPRC             | 类别不平衡时更重要             |
| Average Precision | 和 AUPRC 类似            |
| Partial AUROC     | 高 precision 或低 FPR 区域 |

---

## 8.2 Classification metrics

在 calibration 上选 threshold 后，test 上报告：

| 指标                | 用途                  |
| ----------------- | ------------------- |
| Accuracy          | 总体正确率               |
| Balanced Accuracy | 类别不平衡更稳             |
| Precision         | 被判为风险样本中有多少是真的      |
| Recall            | 捕获多少风险样本            |
| F1                | precision/recall 平衡 |
| MCC               | 二分类综合指标             |
| Confusion matrix  | 直观展示 FP/TP damage   |

---

## 8.3 Deployment metrics

对 Task B，必须报告：

| 指标                       | 含义                      |
| ------------------------ | ----------------------- |
| Trigger rate             | 触发比例                    |
| FP Recall                | 捕获多少 FP                 |
| TP Damage                | 错伤多少 TP                 |
| Warning Precision        | 触发样本里 FP 占比             |
| FP Reduction per Trigger | 单位触发收益                  |
| Risk-coverage curve      | selective prediction 曲线 |

已有 Stage T 中已经用了 FP Recall、TP Damage、Gate Precision 等核心指标，可以沿用并补全。

---

## 8.4 Calibration metrics

建议补：

| 指标                  | 含义        |
| ------------------- | --------- |
| ECE                 | 风险概率校准误差  |
| Brier score         | 概率预测质量    |
| Reliability diagram | 可视化校准     |
| NLL                 | 概率模型负对数似然 |

如果 detector 要作为风险分数，校准指标很重要。

---

## 8.5 Efficiency metrics

导师提到速度，所以必须报。

| 指标                        | 说明                 |
| ------------------------- | ------------------ |
| Feature dim               | 特征维度               |
| Train time                | probe 训练耗时         |
| Inference time per sample | 每样本检测耗时            |
| Extra forward count       | 是否需要 blind forward |
| GPU memory                | 峰值显存               |
| Storage cost              | hidden state 存储成本  |
| Projection cost           | 子空间投影耗时            |

尤其要诚实说明：

* margin baseline 几乎免费；
* raw hidden probe 需要一次 image forward；
* blind-reference detector 需要额外 question-only forward；
* 但如果用于 selective VCD/ICD，可能通过减少 VCD/ICD 触发节省总成本。

---

# 九、核心实验设计

## Experiment 1：主检测性能比较

### 目标

回答：

> 我们的 detector 相比 black-box、output-level trainable、raw hidden-state probe、subspace probe，到底怎么样？

### 任务

* Task A：FP vs TN；
* Task B：FP vs TP；
* Task C：Error vs Correct，可选。

### 表格模板

| Feature / Method       |      Dim | Train? | AUROC | AUPRC | ACC | BAcc | F1 | MCC | Time/sample |
| ---------------------- | -------: | ------ | ----: | ----: | --: | ---: | -: | --: | ----------: |
| Yes/No margin          |        1 | No     |       |       |     |      |    |     |             |
| Binary entropy         |        1 | No     |       |       |     |      |    |     |             |
| Logistic output stats  |        5 | Yes    |       |       |     |      |    |     |             |
| Raw (z_{\text{img}})   |     4096 | Yes    |       |       |     |      |    |     |             |
| Raw (z_{\text{blind}}) |     4096 | Yes    |       |       |     |      |    |     |             |
| Raw concat             |     8192 | Yes    |       |       |     |      |    |     |             |
| Raw diff (d)           |     4096 | Yes    |       |       |     |      |    |     |             |
| Random-K               |        K | Yes    |       |       |     |      |    |     |             |
| PCA-K                  |        K | Yes    |       |       |     |      |    |     |             |
| Top-SVD-K              |        K | Yes    |       |       |     |      |    |     |             |
| Tail band              |        K | Yes    |       |       |     |      |    |     |             |
| PLS-32                 |       32 | Yes    |       |       |     |      |    |     |             |
| Fisher-64              |       64 | Yes    |       |       |     |      |    |     |             |
| Margin + PLS           |       33 | Yes    |       |       |     |      |    |     |             |
| Margin + FullD         | 2 scores | Yes    |       |       |     |      |    |     |             |

### 成功标准

至少满足以下之一：

1. geometry detector 在 Task B 上优于 raw representation probe；
2. margin+geometry 明显优于 margin-only；
3. subspace detector 用更低维度达到接近 raw diff 的性能；
4. subspace detector 在 strict transfer 上比 raw probe 更稳定；
5. subspace detector 在 speed/维度/解释性上有明显优势。

---

## Experiment 2：子空间必要性实验

### 目标

回答导师问题：

> 为什么不直接在原表征空间做探测？

### 对比组

固定同一层、同一 readout、同一 split：

| 方法                           | 说明        |
| ---------------------------- | --------- |
| Raw (z_{\text{img}}) probe   | 直接图文表征    |
| Raw (z_{\text{blind}}) probe | 直接文本表征    |
| Raw (d) probe                | 原始差分      |
| PCA on (d)                   | 普通降维      |
| Random subspace              | 随机低维      |
| Top-SVD                      | 主方差       |
| Tail/residual                | 去掉主方差后的信号 |
| PLS/Fisher                   | 监督判别子空间   |

### 要报告

1. AUROC / AUPRC；
2. Dim-performance curve；
3. Stability across seeds；
4. Cross-split transfer；
5. Projection interpretability；
6. Feature dimension and cost。

### 关键图

画一张：

```text
x-axis: feature dimension K
y-axis: AUROC / AUPRC
curves:
- random
- PCA
- top-SVD
- tail
- PLS
- raw full diff horizontal line
```

这张图可以直接回答“子空间是否必要”。

---

## Experiment 3：margin 互补性实验

### 目标

回答：

> geometry 只是 margin 的重复，还是能提供额外信息？

### 设计 1：margin-only vs geometry-only vs margin+geometry

固定 predicted-Yes Task B。

| Gate            |   Trigger | FP Recall | TP Damage | Warning Precision | AUPRC |
| --------------- | --------: | --------: | --------: | ----------------: | ----: |
| Random          | 10/20/30% |           |           |                   |       |
| Margin-only     | 10/20/30% |           |           |                   |       |
| Geometry-only   | 10/20/30% |           |           |                   |       |
| Margin+Geometry | 10/20/30% |           |           |                   |       |

### 设计 2：margin-bin analysis

把样本按 margin 分桶：

| Margin bin | 样本数 | FP rate | Geometry AUROC | Geometry AUPRC |
| ---------- | --: | ------: | -------------: | -------------: |
| very low   |     |         |                |                |
| low        |     |         |                |                |
| medium     |     |         |                |                |
| high       |     |         |                |                |

如果 geometry 在 low/medium margin 内仍有区分能力，就说明它不是 margin 的简单复制。

### 设计 3：residual prediction

训练 margin-only detector，找出它漏掉的 FP，看 geometry 是否能补抓。

报告：

* margin-only missed FP count；
* geometry top-20% 能抓回多少；
* random top-20% 抓回多少；
* case study。

### 成功标准

最理想：

> margin+geometry 在相同 trigger rate 下，比 margin-only 有更高 FP Recall 或 Warning Precision，同时 TP Damage 不显著增加。

如果提升有限，也可以写成：

> geometry 的主要价值不是全面超过 margin，而是在低维、可解释、选择性路由中提供补充信号。

---

## Experiment 4：部署式 selective warning 评估

### 目标

把 detector 从 AUROC 转到实际使用。

### 设置

只考虑 predicted-Yes 样本。

检测器输出 risk score。

选择 top 10%、20%、30% 作为 warning 样本。

### 报告

| Method       | Trigger | FP Recall | TP Damage | Warning Precision | Relative Precision Gain |
| ------------ | ------: | --------: | --------: | ----------------: | ----------------------: |
| Random       |         |           |           |                   |                         |
| Margin-only  |         |           |           |                   |                         |
| PLS-only     |         |           |           |                   |                         |
| FullD-only   |         |           |           |                   |                         |
| Margin+PLS   |         |           |           |                   |                         |
| Margin+FullD |         |           |           |                   |                         |

已有结果中，low-margin+geometry 的 warning precision 已经能提升到约 0.324–0.327，可以作为目前正面结果的基础，但需要补 margin-only 和完整对照。

---

## Experiment 5：跨 split 与跨数据集泛化

### 目标

防止 detector 只是在一个 split 上过拟合。

### 设置

主表：

| Train  | Calib   | Test        | Method     | AUROC | AUPRC | F1 | Warning Precision |
| ------ | ------- | ----------- | ---------- | ----: | ----: | -: | ----------------: |
| random | popular | adversarial | margin     |       |       |    |                   |
| random | popular | adversarial | raw diff   |       |       |    |                   |
| random | popular | adversarial | PLS        |       |       |    |                   |
| random | popular | adversarial | margin+PLS |       |       |    |                   |
| POPE   | none    | AMBER       | margin     |       |       |    |                   |
| POPE   | none    | AMBER       | PLS        |       |       |    |                   |

### 结论写法

如果 AMBER modest：

> External transfer remains above chance but modest, suggesting that the detector captures partially transferable risk geometry but still depends on dataset/task format.

已有 AMBER 结果本来就是 modest，不能过度包装。

---

## Experiment 6：跨模型 detector 审计

### 目标

证明 detector 在不同模型上的边界。

### 统一表格

| Model       | Task  | Margin AUROC | Raw diff AUROC | PLS AUROC | Margin+PLS AUROC | Warning Precision | Comment     |
| ----------- | ----- | -----------: | -------------: | --------: | ---------------: | ----------------: | ----------- |
| LLaVA-7B    | FP/TN |              |                |           |                  |                   | main        |
| LLaVA-13B   | FP/TN |              |                |           |                  |                   | replication |
| Qwen2-VL    | FP/TP |              |                |           |                  |                   | moderate    |
| Qwen2.5-VL  | FP/TP |              |                |           |                  |                   | moderate    |
| InternVL2   | FP/TP |              |                |           |                  |                   | failure     |
| InternVL2.5 | FP/TP |              |                |           |                  |                   | failure     |

已有结果显示 Qwen 的 geometry 有中等信号，但 margin entropy 更强；InternVL 的 FP/TN 近乎完美但 FP/TP 失败。

这一部分不要怕暴露失败，应该写成：

> Cross-model audits reveal the boundary of correction-geometry detectors.

---

# 十、统计显著性与可靠性

所有主结果必须带统计检验。

## 10.1 Bootstrap CI

对以下指标做 1000 次 bootstrap：

* AUROC；
* AUPRC；
* F1；
* FP Recall；
* TP Damage；
* Warning Precision；
* ACC delta。

报告 95% CI。

---

## 10.2 Paired comparison

比较两个 detector 时，不要只报数值。

例如：

* margin-only vs margin+PLS；
* raw diff vs PLS；
* top-SVD vs tail；
* random vs geometry。

报告：

[
\Delta \text{AUROC}
]

[
\Delta \text{AUPRC}
]

[
\Delta \text{Warning Precision}
]

及其 CI。

---

## 10.3 多 seed

至少 5 seeds：

```text
13, 17, 23, 29, 31
```

已有 Stage P 已经使用过这组 seeds，可以延续。

---

# 十一、速度与成本评估

导师提到速度，这部分要认真做。

## 11.1 成本拆解

| Method                     | Image forward |  Blind forward | Extra operator | Projection | Probe | Total overhead |
| -------------------------- | ------------: | -------------: | -------------: | ---------: | ----: | -------------: |
| Margin                     |             1 |              0 |              0 |          0 |     0 |         lowest |
| Raw (z_{\text{img}}) probe |             1 |              0 |              0 |          0 |  tiny |            low |
| Blind diff probe           |             1 |              1 |              0 |       tiny |  tiny |         medium |
| VCD always-on              |             1 |    1 distorted |       decoding |         no |    no |           high |
| Selective geometry + VCD   |       1+blind | triggered only |        partial |       tiny |  tiny |       variable |

## 11.2 要报告的数字

* feature extraction time；
* detector inference time；
* total time per 1000 samples；
* GPU memory peak；
* hidden state storage size；
* training time；
* VCD/ICD trigger savings。

如果你的 detector 本身比 margin 慢，可以这样解释：

> Geometry detector is not intended to replace free confidence baselines, but to support selective routing where the cost of downstream correction is much higher.

---

# 十二、最终论文表格规划

## Table 1：Task A 主检测结果，FP vs TN

比较所有 feature families。

---

## Table 2：Task B 部署检测结果，FP vs TP

重点比较：

* random；
* margin-only；
* geometry-only；
* margin+geometry；
* raw hidden probe；
* subspace probe。

---

## Table 3：子空间必要性

显示：

* raw full diff；
* random；
* PCA；
* top-SVD；
* tail；
* PLS/Fisher；
* dim；
* stability；
* speed。

---

## Table 4：跨数据集 / 跨模型泛化

POPE → AMBER，LLaVA → Qwen/InternVL。

---

## Table 5：速度与成本

包括 forward 次数、特征维度、训练时间、推理时间、显存。

---

# 十三、最终图规划

## Figure 1：Detector pipeline

展示：

```text
image+question → z_img
question only → z_blind
d = z_blind - z_img
projection / subspace
risk score
warning / selective routing
```

---

## Figure 2：子空间维度 vs 性能曲线

x-axis: K
y-axis: AUROC/AUPRC
curves:

* random；
* PCA；
* top-SVD；
* tail；
* PLS；
* raw diff horizontal line。

---

## Figure 3：margin 互补性

可以画：

* margin-only vs margin+geometry PR curve；
* margin-bin geometry AUROC；
* risk-coverage curve。

---

## Figure 4：跨模型 detector map

显示不同模型上：

* FP/TN；
* FP/TP；
* margin；
* geometry。

突出 InternVL failure。

---

# 十四、阶段性执行计划

## 第 1 周：统一数据与特征缓存

目标：

* 统一 POPE random/popular/adversarial；
* 固定 readout；
* 缓存 (z_{\text{img}})、(z_{\text{blind}})、(d)、logits、margin、entropy；
* 标准化 outcome label：TP/TN/FP/FN。

产出：

```text
features/
  llava15_7b/
    pope_random/
    pope_popular/
    pope_adversarial/
      logits.npy
      z_img_Lxx.npy
      z_blind_Lxx.npy
      diff_Lxx.npy
      labels.csv
```

---

## 第 2 周：完成 baseline detector 大表

目标：

* black-box baselines；
* output-level trainable baselines；
* raw representation probes；
* raw diff probes；
* basic subspace probes。

产出：

```text
results/detector_baseline_table.csv
results/detector_baseline_summary.md
```

---

## 第 3 周：完成子空间必要性实验

目标：

* random / PCA / top-SVD / tail / PLS / Fisher；
* dimension curve；
* seed robustness；
* bootstrap CI。

产出：

```text
figures/dim_vs_auc.png
tables/subspace_necessity.csv
```

---

## 第 4 周：完成 margin 互补性与部署 warning

目标：

* margin-only vs geometry-only vs margin+geometry；
* trigger rate 10/20/30；
* warning precision；
* risk-coverage；
* margin-bin analysis。

产出：

```text
tables/deployment_warning.csv
figures/margin_complementarity.png
figures/risk_coverage.png
```

---

## 第 5 周：外部泛化与跨模型最小实验

目标：

* AMBER zero-shot transfer；
* LLaVA-13B replication；
* Qwen / InternVL 统一最小表。

产出：

```text
tables/external_transfer.csv
tables/cross_model_detector.csv
```

---

## 第 6 周：整理论文级结果

目标：

* 写 detector 部分；
* 确定主表；
* 整理 appendix；
* 明确哪些结果主打，哪些作为限制。

产出：

```text
detector_section_draft.md
detector_figures/
detector_tables/
```

---

# 十五、成功标准

我建议你提前设定 success criteria。

## 强成功

满足以下条件：

1. Task B 上 margin+geometry 稳定优于 margin-only；
2. strict subset-transfer 中仍有明显提升；
3. subspace detector 以低维达到或超过 raw diff；
4. AMBER transfer above chance；
5. LLaVA-13B 复现；
6. speed/cost 有合理解释。

可以主张：

> correction-geometry detector provides complementary deployable hallucination risk signal.

---

## 中等成功

满足：

1. Task A 上 correction geometry 明显有效；
2. Task B 上 geometry-only 弱于 margin，但 margin+geometry 略有提升；
3. subspace 低维、可解释、稳定；
4. 外部迁移 modest。

可以主张：

> detector 不是最强 confidence baseline，但 correction subspace 是有用的机制性风险特征，可辅助部署。

---

## 弱成功

如果结果显示：

* margin-only 全面胜出；
* raw diff 全面胜过所有 subspace；
* margin+geometry 没有提升；
* 外部迁移弱。

那就不要主打 detector 方法论文。把 detector 降级为机制论文里的 supporting experiment：

> correction geometry contains hallucination-related information, but output confidence remains a stronger practical detector.

这也不是失败，只是定位要变。

---

# 十六、我建议你优先完成的最小实验包

如果你想最快给导师一个有力反馈，先做这个最小包：

## Minimal Package

1. **Task B predicted-Yes FP vs TP**
2. **LLaVA-1.5-7B**
3. **Strict split：random train / popular calib / adversarial test**
4. **Baselines：**

   * margin-only；
   * entropy；
   * output logistic；
   * raw (z_{\text{img}})；
   * raw (d)；
   * random subspace；
   * PCA；
   * top-SVD；
   * tail；
   * PLS；
   * margin+PLS；
   * margin+FullD。
5. **Metrics：**

   * AUROC；
   * AUPRC；
   * ACC；
   * F1；
   * Trigger 10/20/30；
   * FP Recall；
   * TP Damage；
   * Warning Precision；
   * Time/sample。
6. **图：**

   * dimension vs AUROC；
   * margin-only vs margin+geometry PR curve；
   * trigger tradeoff curve。

这个最小包完成后，你就能回答导师最核心的问题：

> 检测器和其他训练型模型比怎么样？子空间有没有必要？是否比直接原空间 probe 更有优势？是否有部署价值？

---

# 十七、最终建议

你现在做检测器路线是合理的，因为它比直接证明机制因果闭环更快，也更容易形成完整实验。

但你一定要避免两个陷阱：

1. **不要只报 AUROC。**
   要把它变成完整 detector benchmark。

2. **不要硬说 geometry 全面优于 margin。**
   更稳的说法是：geometry 提供 correction-space 风险信号，尤其在 margin+geometry、低维子空间、选择性路由中有价值。

最终 detector 论文的核心结论可以写成：

> Blind-reference correction geometry provides a compact and interpretable risk signal for VLM hallucination detection. While output confidence remains a strong baseline, correction-space subspaces—especially residual/tail and supervised PLS/Fisher coordinates—offer complementary information, lower-dimensional probes, and useful selective-routing behavior under deployment-style predicted-Yes evaluation.

这条路线既回应了导师的要求，也不会过度承诺。
