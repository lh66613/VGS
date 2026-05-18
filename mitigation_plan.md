下面给你一份**“利用机理实现幻觉缓解”的任务书**。我会把它设计成一个可以直接执行的实验计划，目标不是继续做零散尝试，而是完成导师要求的那个闭环：

> **从机制发现 → 解释现有 VCD/ICD 的不足 → 设计 geometry-guided 缓解方法 → 证明它比原始 VCD/ICD 更好。**

你现在 detector 线暂时已经有一个阶段性结论：POPE 内部 strict split 上 margin+tail/full 有显著增益，但 AMBER 外部迁移不稳，所以 detector 暂时可以作为辅助结果保留。
接下来机理缓解路线的核心是：**不能只证明“破坏 tail 会让 TN 变坏”，还要证明“利用这些几何结构可以更好地减少 FP，同时少伤 TP”。**

---

# 基于 Correction Geometry 的幻觉缓解任务书

## 一、任务目标

本任务的总目标是：

> 基于 blind-reference correction geometry，设计一种比原始 VCD/ICD 更有针对性的幻觉缓解方法，在减少 false positive hallucination 的同时，尽可能保留 true positive answer。

更具体地说，要回答三个问题：

1. **机制解释问题**
   VCD/ICD 这类 contrastive decoding 为什么能 work？它的有效成分和无效成分分别落在 correction spectrum 的什么位置？

2. **方法改进问题**
   如果主方差方向不是幻觉判别方向，那么是否可以过滤掉无关 correction component，只保留 hallucination-relevant component，从而比普通 VCD/ICD 更好？

3. **因果闭环问题**
   我们能否不仅“破坏正确样本”，还可以“缓解错误样本”，即减少 FP，同时保持 TP？

---

# 二、核心假设

根据已有发现，可以提出三个假设。

## 假设 H1：VCD/ICD 的 full contrastive correction 混入了大量非判别方向

你已有结果显示，blind-reference difference 存在强低秩结构，但 top dominant variance directions 与 hallucination discrimination 并不一致；判别信号更多出现在 residual/tail、mid spectral bands 或 evidence-sensitive coordinates。已有完整实验总结中也记录了 top-4 高方差但 AUROC 弱、tail/residual 与 FP/TN 相关、L24 tail ablation 能破坏 TN 正确拒绝等发现。

因此可以推测：

> 普通 VCD/ICD 使用完整 contrastive signal 时，可能同时放大了有用的幻觉修正方向和无关的 image-conditioning backbone，从而导致 TP damage 或过度保守。

---

## 假设 H2：过滤到 correction spectrum 的有效子空间后，可以改善 VCD/ICD 的 FP-TP tradeoff

也就是说，不一定要让 FP reduction 绝对最大，而是要做到：

* 同等 FP reduction 下，TP preserved 更高；
* 同等 TP damage 下，FP reduction 更高；
* 同等效果下，触发比例更低；
* 同等效果下，计算成本更低。

这是最适合证明“比 VCD 更好”的方式。

---

## 假设 H3：直接 hidden-state rescue 很难，但 geometry-guided decoding / routing 更可行

你之前 direct FP rescue 弱，说明单层单方向 steering 很难把已经形成的错误 Yes 改成 No。已有总结中也指出，FP rescue 主要只对 borderline case 有效，不宜包装成可靠 mitigation。

因此这次不要一开始就主打 hidden rescue，而应该优先做更稳定的三类方法：

1. **Subspace-filtered VCD/ICD**
2. **Geometry-guided selective VCD/ICD**
3. **Geometry-guided logit correction / reranking**

---

# 三、总体实验路线

整条机理缓解路线分为五个阶段。

```text
Stage 1: VCD/ICD correction decomposition
        ↓
Stage 2: Subspace-filtered VCD/ICD
        ↓
Stage 3: Geometry-guided selective routing
        ↓
Stage 4: Logit-level correction / reranking
        ↓
Stage 5: Cross-dataset / cross-model validation
```

每一阶段都有明确的通过标准。

---

# 四、Stage 1：VCD/ICD correction decomposition

## 4.1 目标

先不要急着改方法，第一步是证明：

> VCD/ICD 的 correction signal 是否真的包含大量非幻觉判别成分？

如果这一步成立，后面的 subspace-filtered VCD 才有理论基础。

---

## 4.2 实验对象

选择几个 operator：

| Operator      | 说明                          |
| ------------- | --------------------------- |
| VCD-gray      | 灰图对比                        |
| VCD-blur      | 模糊图对比                       |
| VCD-diffusion | diffusion noise 对比          |
| ICD-blind     | image vs blind/text-only 对比 |

已有 Stage T 中，ICD-blind 和 VCD 系列已经做过 always-on 与 selective routing 对比，其中 ICD-blind 的 always-on FP reduction 较强但会带来 TP damage，selective routing 能改善 tradeoff。

---

## 4.3 需要计算的 correction

对于每个样本、每层 (L)、每个 readout position，计算：

[
\Delta h_{\text{VCD}} = h_{\text{orig}} - h_{\text{neg}}
]

其中：

* (h_{\text{orig}})：正常 image+question hidden state；
* (h_{\text{neg}})：VCD/ICD 的负条件 hidden state，例如 gray/blur/diffusion/blind；
* 对 ICD-blind，(h_{\text{neg}}) 就是 question-only / blind hidden state。

也可以在 logits 层计算：

[
\Delta \ell_{\text{VCD}} = \ell_{\text{orig}} - \ell_{\text{neg}}
]

但优先做 hidden-level decomposition，因为它能和 correction space 对齐。

---

## 4.4 投影到 correction spectrum

将 (\Delta h_{\text{VCD}}) 投影到你已经构建的子空间：

| 子空间            | 含义                                   |
| -------------- | ------------------------------------ |
| Top 1–4        | dominant image-conditioning backbone |
| Band 5–16      | early transferable spectral signal   |
| Band 17–64     | mid spectral signal                  |
| Band 65–256    | deeper spectral signal               |
| Tail 257–1024  | residual/tail risk signal            |
| Top-complement | 去掉 dominant backbone 后的 residual     |
| PLS/Fisher     | supervised discriminative directions |
| Random band    | 控制组                                  |

计算每个 band 的：

### Energy fraction

[
r_b = \frac{|P_b \Delta h|^2}{|\Delta h|^2}
]

表示 VCD correction 的能量主要落在哪里。

### Logit contribution

[
\Delta \ell_b = W_U P_b \Delta h
]

重点看：

[
\Delta m_b = \Delta \ell_b(\text{No}) - \Delta \ell_b(\text{Yes})
]

表示该 band 是否推动模型从 Yes 转向 No。

### Outcome correlation

分别在 FP、TP、TN 上统计：

* band energy；
* band logit contribution；
* band contribution 与 VCD 成功/失败的相关性；
* band contribution 与 TP damage 的相关性。

---

## 4.5 关键分析问题

你要回答：

1. VCD/ICD 的 correction energy 是否主要落在 top backbone？
2. top backbone component 是否对 FP reduction 贡献弱？
3. residual/tail component 是否更能预测 FP 被修正？
4. TP damage 是否与 top-backbone over-correction 有关？
5. ICD-blind 是否比 VCD-gray/blur/diffusion 更接近你的 blind-reference correction space？

---

## 4.6 预期正结果

最理想结果是：

> VCD/ICD 的大部分 correction energy 落在 top backbone，但 FP reduction 主要由 residual/tail 或 top-complement component 贡献；TP damage 与 top-backbone component 更相关。

如果得到这个结果，就可以强力支撑：

> 普通 VCD 能 work，但它不是 targeted correction；它混入了大量非判别视觉条件变化。

---

## 4.7 阶段产出

文件建议：

```text
outputs/mechanism_mitigation/stage1_vcd_decomposition/
  vcd_band_energy.csv
  vcd_band_logit_contribution.csv
  vcd_success_failure_analysis.csv
  vcd_tp_damage_analysis.csv
  figures/
    vcd_energy_by_band.png
    vcd_logit_effect_by_band.png
    vcd_success_vs_band_score.png
```

---

# 五、Stage 2：Subspace-filtered VCD/ICD

## 5.1 目标

设计核心改进方法：

> 不使用完整 VCD/ICD correction，而是只使用 correction spectrum 中更相关的子空间成分。

---

## 5.2 方法定义

普通 VCD/ICD 近似使用完整 correction：

[
h' = h_{\text{orig}} + \alpha (h_{\text{orig}} - h_{\text{neg}})
]

你的方法改成：

[
h' = h_{\text{orig}} + \alpha P_{\mathcal{S}}(h_{\text{orig}} - h_{\text{neg}})
]

其中 (\mathcal{S}) 是某个子空间。

可选子空间包括：

| 方法名            | 子空间             |
| -------------- | --------------- |
| Top4-VCD       | Top 1–4         |
| Top16-VCD      | Top 1–16        |
| Band5-16-VCD   | Band 5–16       |
| Tail-VCD       | Tail 257–1024   |
| Complement-VCD | 去掉 Top 1–4      |
| Full-VCD       | 原始完整 correction |
| Random-VCD     | 随机同维子空间         |
| PLS-VCD        | PLS/Fisher 子空间  |

---

## 5.3 两种实现方式

### 方式 A：Hidden-state subspace intervention

在某一层 (L) 的 hidden state 上做：

[
h_L' = h_L + \alpha P_{\mathcal{S}} \Delta h_L
]

然后继续 forward / decode。

优点：

* 更接近机制；
* 可以证明 hidden correction 有因果效应。

缺点：

* 实现复杂；
* 容易不稳定；
* 需要 forward hook。

---

### 方式 B：Logit-level subspace correction

先将 subspace hidden correction 映射到 logits：

[
\Delta \ell_{\mathcal{S}} = W_U P_{\mathcal{S}}\Delta h_L
]

然后做：

[
\ell' = \ell_{\text{orig}} + \alpha \Delta \ell_{\mathcal{S}}
]

对于 POPE yes/no，可以更简单地只修正 Yes/No margin：

[
m' = m + \alpha \Delta m_{\mathcal{S}}
]

其中：

[
m = \ell_{\text{No}} - \ell_{\text{Yes}}
]

优点：

* 稳定；
* 快；
* 更容易出正结果；
* 更适合第一阶段验证。

缺点：

* 不如 hidden intervention 机制性强；
* 主要适合 yes/no benchmark。

**建议先做方式 B，再做方式 A。**

---

## 5.4 实验设置

主数据集：

```text
POPE random/popular/adversarial
```

主协议：

```text
Train subspace: random
Calibrate alpha/threshold: popular
Test: adversarial
```

不要在 test 上调 alpha。

调参范围：

```text
alpha ∈ {0.25, 0.5, 1, 2, 4}
layer ∈ {16, 20, 24, 28, 32}
subspace ∈ {top4, top16, band5-16, tail, complement, full, random}
```

为了防止过拟合，建议先固定：

```text
layer = 24
readout = last_prompt_token 或 last_8_prompt_mean
```

然后再做 layer sweep。

---

## 5.5 Baselines

必须比较：

| Baseline               | 说明                   |
| ---------------------- | -------------------- |
| Base                   | 不使用 correction       |
| Full VCD               | 原始完整 VCD             |
| Full ICD               | 原始完整 ICD             |
| Random-subspace VCD    | 随机同维控制               |
| Top4-VCD               | 主方差方向控制              |
| Margin-only correction | 只用 output confidence |
| Detector-gated VCD     | 你检测器已有结果             |
| Oracle gate            | 上界，只作为参考             |

---

## 5.6 评价指标

不能只看 accuracy。

重点指标：

| 指标                         | 含义                       |
| -------------------------- | ------------------------ |
| FP reduction               | 原本 FP 中被修正为 No 的比例       |
| TP preserved               | 原本 TP 中仍保持 Yes 的比例       |
| TN preserved               | 原本 TN 不被破坏的比例            |
| Accuracy delta             | 总体准确率变化                  |
| Yes rate change            | 是否过度减少 Yes               |
| No bias                    | 是否变成一味回答 No              |
| Unknown / invalid rate     | 是否产生异常输出                 |
| FP reduction per TP damage | tradeoff 指标              |
| Cost                       | 额外 forward / decoding 成本 |

最重要的是 Pareto tradeoff：

[
\text{FP reduction} \uparrow,\quad \text{TP preserved} \uparrow
]

---

## 5.7 成功标准

### 强成功

满足：

> Tail/Complement/Band-filtered VCD 在相同 TP preserved 下，FP reduction 高于 full VCD；或者在相同 FP reduction 下，TP damage 更低。

例如：

| Method   | FP Reduction | TP Preserved | Accuracy Delta |
| -------- | -----------: | -----------: | -------------: |
| Full VCD |         0.30 |         0.91 |          -0.02 |
| Tail-VCD |         0.28 |         0.96 |           0.00 |

这就是非常好的正结果。

---

### 中等成功

满足：

> Subspace-filtered VCD 不一定超过 full VCD 的 FP reduction，但显著减少 TP damage，形成更好的 Pareto point。

这也可以写。

---

### 弱成功

如果 subspace-filtered VCD 没有超过 full VCD，但不同 bands 显示出清晰 tradeoff：

* top4 容易 damage TP；
* tail 更保守；
* complement 更平衡。

也可以作为机制证据。

---

## 5.8 阶段产出

```text
outputs/mechanism_mitigation/stage2_subspace_vcd/
  subspace_vcd_results.csv
  alpha_sweep.csv
  pareto_frontier.csv
  band_comparison.csv
  figures/
    fp_reduction_vs_tp_preserved.png
    alpha_tradeoff_by_band.png
    pareto_frontier.png
```

---

# 六、Stage 3：Geometry-guided selective VCD/ICD

## 6.1 目标

如果 subspace-filtered intervention 不够强，selective routing 仍然可能成功。

核心问题：

> 不一定修改 VCD 本身，而是用 geometry detector 决定什么时候触发 VCD/ICD。

你已经有相关基础结果：POPE 内部 strict split 中，margin+tail/full 的 warning precision 和 AUPRC 显著优于 margin-only，但 AMBER 迁移不稳。
这可以直接服务于 selective mitigation。

---

## 6.2 方法

先用 detector 得到 risk score：

[
r(x) = f(\text{margin}, \text{tail/full correction features})
]

然后只对 top-risk 样本触发 VCD/ICD：

[
\text{if } r(x) > \tau,\quad \text{apply VCD/ICD}
]

否则使用 base answer。

---

## 6.3 Gate variants

比较：

| Gate             | 说明              |
| ---------------- | --------------- |
| Random gate      | 随机触发            |
| Margin-only gate | 低 margin 触发     |
| Tail-only gate   | tail risk score |
| Margin+tail gate | 当前最强            |
| Margin+full gate | 当前最强            |
| Oracle gate      | 理论上界            |

---

## 6.4 Operator variants

| Operator              | 说明 |
| --------------------- | -- |
| VCD-gray              |    |
| VCD-blur              |    |
| VCD-diffusion         |    |
| ICD-blind             |    |
| Subspace-filtered VCD |    |
| Subspace-filtered ICD |    |

重点组合：

```text
margin+tail gate + ICD-blind
margin+tail gate + VCD-diffusion
margin+tail gate + Tail-VCD
margin+full gate + ICD-blind
```

---

## 6.5 评价表格

| Method                | Trigger | FP Reduction | TP Preserved | Accuracy Delta | Cost Saved |
| --------------------- | ------: | -----------: | -----------: | -------------: | ---------: |
| Always-on ICD         |    100% |              |              |                |          0 |
| Random-gated ICD      |     20% |              |              |                |        80% |
| Margin-gated ICD      |     20% |              |              |                |        80% |
| Margin+tail gated ICD |     20% |              |              |                |        80% |
| Margin+full gated ICD |     20% |              |              |                |        80% |
| Oracle-gated ICD      |     20% |              |              |                |        80% |

---

## 6.6 成功标准

强成功：

> 在 20% 或 10% trigger 下，geometry-gated ICD/VCD 接近 always-on 的 FP reduction，但明显更高 TP preserved 和更小 accuracy drop。

例如：

| Method                | Trigger | FP Reduction | TP Preserved | Acc Delta |
| --------------------- | ------: | -----------: | -----------: | --------: |
| Always-on ICD         |    100% |         0.34 |         0.91 |     -0.02 |
| Margin+tail gated ICD |     20% |    0.28–0.32 |         0.96 |      0.00 |

这就能说明：

> 机制发现可以指导更低成本、更低损伤的幻觉缓解。

---

## 6.7 阶段产出

```text
outputs/mechanism_mitigation/stage3_selective_vcd/
  gated_operator_results.csv
  trigger_sweep.csv
  cost_benefit_table.csv
  figures/
    fp_reduction_tp_preserved_tradeoff.png
    selective_vs_always_on.png
```

---

# 七、Stage 4：Geometry-guided logit correction / reranking

## 7.1 目标

如果 hidden-level subspace VCD 不稳定，可以做更稳的 logit-level 缓解。

核心思想：

> 对高风险 predicted-Yes 样本，直接修正 Yes/No margin，而不是大幅编辑 hidden state。

---

## 7.2 方法 A：risk-aware Yes suppression

对于 predicted-Yes 样本，计算 geometry risk (r(x))，然后：

[
\ell'*{\text{Yes}} = \ell*{\text{Yes}} - \lambda r(x)
]

或者：

[
m' = m - \lambda r(x)
]

其中：

[
m = \ell_{\text{Yes}} - \ell_{\text{No}}
]

只对 high-risk 样本启用，避免伤害所有 TP。

---

## 7.3 方法 B：band-specific logit contribution

使用 band-specific correction：

[
\Delta m_{\mathcal{S}} = \Delta \ell_{\mathcal{S}}(\text{No}) - \Delta \ell_{\mathcal{S}}(\text{Yes})
]

然后：

[
m' = m + \alpha \Delta m_{\mathcal{S}}
]

比较：

* top4 logit correction；
* top16 logit correction；
* tail logit correction；
* full logit correction；
* random correction。

---

## 7.4 方法 C：candidate reranking

生成多个候选：

| Candidate           | 来源                  |
| ------------------- | ------------------- |
| base answer         | 原始模型                |
| VCD answer          | VCD                 |
| ICD answer          | ICD                 |
| conservative answer | verification prompt |
| subspace-VCD answer | 你的方法                |

然后用 risk score 或 consistency score 选择。

优点：

* 不直接编辑 hidden；
* 更稳定；
* 可以扩展到开放式任务。

---

## 7.5 成功标准

强成功：

> logit correction / reranking 能减少 FP，并且 TP preserved 明显高于 always-on VCD/ICD。

中等成功：

> 只在 low-margin/high-risk 样本上有效，作为 selective correction 辅助。

---

## 7.6 阶段产出

```text
outputs/mechanism_mitigation/stage4_logit_correction/
  logit_correction_alpha_sweep.csv
  reranking_results.csv
  high_risk_subset_results.csv
  figures/
    margin_shift_fp_tp.png
    correction_tradeoff.png
```

---

# 八、Stage 5：外部验证与失败分析

## 8.1 目标

防止方法只在 POPE 上有效。

---

## 8.2 数据集

最低要求：

| Dataset                   | 用途    |
| ------------------------- | ----- |
| POPE adversarial          | 主测试   |
| AMBER existence           | 外部测试  |
| AMBER attribute/relation  | 压力测试  |
| 可选 MMHal / HallusionBench | 开放式生成 |

AMBER 要特别小心，因为 detector 的 AMBER transfer 已经显示：margin-only 强，POPE-trained geometry 加进去可能伤害 warning precision。

因此在 mitigation 中也要诚实检验：

> POPE 上有效的 geometry-guided correction 是否在 AMBER 上仍有效？

---

## 8.3 外部评估设置

两种协议：

### Zero-shot transfer

```text
Train/Calibrate on POPE
Test on AMBER
```

这是最严格的。

### Light calibration

```text
Train subspace on POPE
Calibrate alpha / gate threshold on small AMBER val
Test on AMBER test
```

如果 zero-shot 不行，light calibration 可能仍然说明：

> 方法需要任务校准，但机制特征可复用。

---

## 8.4 评价指标

除了 FP reduction / TP preserved，还要看：

* overall accuracy；
* yes rate shift；
* answer distribution shift；
* task-specific performance；
* whether method over-suppresses Yes；
* open-ended hallucination rate。

---

## 8.5 阶段产出

```text
outputs/mechanism_mitigation/stage5_external/
  amber_zero_shot_results.csv
  amber_calibrated_results.csv
  task_group_analysis.csv
  failure_cases.md
```

---

# 九、最小可行实验包

如果你想快速开始，不要一口气做完所有东西。建议先完成一个 **MVP**。

## MVP 目标

证明：

> geometry-guided selective ICD/VCD 或 subspace-filtered ICD/VCD 在 POPE adversarial 上，比原始 always-on VCD/ICD 有更好的 FP/TP tradeoff。

---

## MVP 设置

### Dataset

```text
Train: POPE random
Calib: POPE popular
Test: POPE adversarial
```

### Model

```text
LLaVA-1.5-7B
```

### Layer / readout

先固定：

```text
Layer = 24
Readout = last_prompt_token 或 last_8_prompt_mean
```

### Subspaces

只做 5 个：

```text
full
top4
top16
tail257-1024
top4-complement
random-tail-dim
```

### Operators

只做 2 个：

```text
ICD-blind
VCD-diffusion
```

### Gate

只做 4 个：

```text
always-on
random 20%
margin-only 20%
margin+tail 20%
```

### Metrics

```text
FP reduction
TP preserved
accuracy delta
yes rate
trigger rate
cost
```

---

## MVP 成功标准

你只需要达到下面任意一个：

### 成功标准 1：Selective success

```text
margin+tail gated ICD @20%
≈ always-on ICD 的 FP reduction
但 TP preserved 更高，accuracy delta 更小
```

### 成功标准 2：Subspace success

```text
tail/complement ICD
在相同 TP damage 下 FP reduction 高于 full ICD
```

### 成功标准 3：Cost success

```text
20% trigger 的 selective ICD
达到 always-on 70% 以上的 FP reduction
但节省约 80% 下游 correction 成本
```

---

# 十、主结果表格设计

最终如果要写进论文，建议至少有四张表。

## Table 1：VCD/ICD decomposition

| Operator | Band | Energy Fraction | FP Correction Contribution | TP Damage Contribution |
| -------- | ---- | --------------: | -------------------------: | ---------------------: |

---

## Table 2：Subspace-filtered VCD/ICD

| Method | Subspace | FP Reduction | TP Preserved | Accuracy Delta | Yes Rate |
| ------ | -------- | -----------: | -----------: | -------------: | -------: |

---

## Table 3：Selective routing

| Gate | Operator | Trigger | FP Reduction | TP Preserved | Cost Saved |
| ---- | -------- | ------: | -----------: | -----------: | ---------: |

---

## Table 4：External validation

| Dataset | Method | FP Reduction | TP Preserved | Accuracy Delta | Note |
| ------- | ------ | -----------: | -----------: | -------------: | ---- |

---

# 十一、核心图设计

## Figure 1：方法图

展示：

```text
VCD correction Δh
       ↓
project into correction spectrum
       ↓
keep / suppress selected bands
       ↓
subspace-filtered decoding
```

---

## Figure 2：VCD correction energy vs useful effect

x-axis: spectral band
left y-axis: energy fraction
right y-axis: FP correction / TP damage contribution

目标：证明 energy 和 useful effect 不一致。

---

## Figure 3：Pareto frontier

x-axis:

```text
TP damage = 1 - TP preserved
```

y-axis:

```text
FP reduction
```

比较：

* base；
* full VCD；
* ICD；
* random-subspace；
* top4；
* tail；
* complement；
* selective margin+tail。

这是最重要的图。

---

## Figure 4：Selective cost-benefit curve

x-axis: trigger rate
y-axis: FP reduction / TP preserved / accuracy delta

比较：

* random gate；
* margin gate；
* margin+tail gate；
* oracle gate。

---

# 十二、可能结果与对应策略

## 情况 A：subspace-filtered VCD 明显优于 VCD

这是最理想情况。

论文可以主打：

> correction-spectrum analysis reveals why VCD is suboptimal and enables a better subspace-filtered contrastive decoder.

---

## 情况 B：subspace-filtered VCD 不强，但 selective routing 强

这也很好。

论文主张改成：

> correction geometry is more useful for deciding when to apply correction than for directly replacing the correction operator.

即：

> 它不是新的 VCD，而是 VCD 的智能触发器。

---

## 情况 C：POPE 有效，AMBER 无效

这和 detector 一样，说明外部迁移有限。

写法：

> mechanism-guided mitigation is effective under calibrated POPE-style existence QA but does not yet generalize robustly to heterogeneous hallucination settings.

这可以作为限制，不必硬撑。

---

## 情况 D：所有 mitigation 都不如 VCD

如果出现这种情况，说明机理方向需要回到机制论文，不适合主打方法。

那就写：

> correction geometry explains risk and VCD behavior but is insufficient for direct mitigation.

然后回到 detector/机制分析叙事。

---

# 十三、执行时间表

## 第 1 周：VCD/ICD decomposition

完成：

* 计算 (\Delta h_{\text{VCD}})；
* 投影到 spectral bands；
* 分析 energy / logit contribution / outcome correlation。

产出：

```text
stage1_vcd_decomposition_summary.md
```

---

## 第 2 周：logit-level subspace VCD

完成：

* full/top4/top16/tail/complement/random；
* alpha sweep；
* POPE adversarial test。

产出：

```text
stage2_subspace_logit_vcd_results.csv
pareto_frontier.png
```

---

## 第 3 周：hidden-level subspace intervention

完成：

* 选前一周最好的两个子空间；
* 在 L24/L32 做 hidden intervention；
* 对比 logit-level 方法。

产出：

```text
stage2_hidden_subspace_vcd_results.csv
```

---

## 第 4 周：selective routing

完成：

* margin-only gate；
* margin+tail gate；
* random gate；
* always-on；
* VCD/ICD operator comparison。

产出：

```text
stage3_selective_operator_results.csv
```

---

## 第 5 周：外部验证

完成：

* AMBER existence；
* AMBER overall；
* optional small open-ended set。

产出：

```text
stage5_external_results.csv
failure_analysis.md
```

---

## 第 6 周：整理主结论

完成：

* 主表；
* Pareto 图；
* case study；
* 写作草稿。

产出：

```text
mechanism_mitigation_section_draft.md
```

---

# 十四、最终判断标准

这个任务最终是否成功，看三个问题：

## 问题 1：是否比 VCD/ICD 更好？

至少在一个明确指标上形成 Pareto 优势：

```text
same FP reduction, higher TP preserved
or
same TP preserved, higher FP reduction
or
same utility, lower trigger/cost
```

---

## 问题 2：是否证明了子空间有用？

必须有：

```text
tail/complement/top16/random/full 对比
```

如果 tail/complement 比 random 强，且比 full 更少伤 TP，就说明子空间有用。

---

## 问题 3：是否能闭合机制链条？

最终要能讲出这条链：

```text
top variance directions are not hallucination decision directions
        ↓
full contrastive correction mixes useful and non-useful components
        ↓
subspace filtering / geometry gating removes part of the non-useful correction
        ↓
FP reduction vs TP preserved tradeoff improves
```

这就是导师要的“强证据”。

---

# 十五、最建议你现在立刻做的版本

如果你明天就开始，我建议先做这个最小版本：

1. **只用 LLaVA-1.5-7B + POPE random/popular/adversarial。**
2. **只做 ICD-blind，因为它和 blind-reference correction 最一致。**
3. **先做 logit-level correction，不要一开始上 hidden hook。**
4. **比较 full / top4 / top16 / tail / complement / random。**
5. **画 FP reduction vs TP damage 的 Pareto 图。**
6. **再做 margin+tail gate + ICD-blind 的 selective routing。**

如果这一步跑出来：

> margin+tail gated ICD 或 tail/complement ICD 比 always-on ICD 更少伤 TP，同时保留大部分 FP reduction，

那机理缓解路线就可以继续推进。

---

# 十六、一句话总结这个任务书

这条任务线的核心不是继续证明“我发现了一个空间”，而是证明：

> **这个空间能告诉我们 VCD/ICD 哪些 correction 有用、哪些有害，并且据此构造出更好的幻觉缓解策略。**

只要最终能形成一个更好的 Pareto tradeoff，你就完成了导师要求的“机理 → 方法有效性”的闭环。
