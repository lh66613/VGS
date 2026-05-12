# VGS 完整实验总结

> **项目全称**: Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination
> **生成日期**: 2026-05-08
> **主模型**: LLaVA-1.5-7B / 13B; 跨架构扩展: Qwen2-VL / Qwen2.5-VL / InternVL2 / InternVL2.5
> **主基准**: POPE COCO (random / popular / adversarial, 9000 samples); 外部基准: AMBER discriminative (14216 samples)

---

## 一、核心方法论：Blind-Reference Differencing

对每个样本提取两种 hidden state：

```
z_img   = hidden_state(image + question)
z_blind = hidden_state(question only)
```

定义 blind-reference correction difference：

```
d = z_blind - z_img
```

主读层：L24；辅助读层：L20 / L32 / L16 / L28。
主读出位置：`last_prompt_token`，辅助：`last_4_prompt_mean` / `last_8_prompt_mean`。
核心任务：在 ground-truth=No 的样本中区分 FP（模型错误回答 Yes）与 TN（模型正确回答 No）。

---

## 二、各阶段实验详细结果

### Stage A：差分校谱分析 (2026-04-22)

**目标**: 分析盲-图差分矩阵的低秩结构。

**主要发现**:
- 差分矩阵存在强低秩结构：Top-4 SVD 方向解释 72.7%-88.6% 方差（L8: 88.6%, L24: 87.7%, L32: 72.7%）
- L32 方差集中度明显低于中间层
- Split-half 稳定性在 K=4 时最强

---

### Stage C：FP/TN Probe 与深层诊断 (2026-04-22 ~ 2026-04-23)

**目标**: 评估差分几何对 FP/TN 的区分能力。

#### Stage C 基础 (2026-04-22)
- 最强 full difference AUROC: L24 = **0.6936**
- 最强 projected K=64: L20 = **0.6338**
- Full difference 特征优于低 K 投影特征

#### Stage C Deep (2026-04-22)
**核心发现：方差与判别性能不同步**

| Layer | K=4 AUROC | K=64 AUROC | K=128 AUROC | K=256 AUROC |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.4689 | 0.5528 | 0.6091 | 0.6374 |
| 12 | 0.4650 | 0.5899 | 0.6163 | 0.6526 |
| 16 | 0.4653 | 0.6242 | 0.6761 | 0.6862 |
| 20 | **0.5570** | **0.6338** | **0.6846** | **0.6948** |
| 24 | 0.4637 | 0.6192 | 0.6539 | 0.6496 |
| 28 | 0.4807 | 0.6185 | 0.6352 | 0.6515 |
| 32 | 0.5005 | 0.5652 | 0.5900 | 0.6185 |

对应 Top-K 解释方差（K=4 已达 72.7%-88.6%，K=256 达 95.8%-98.8%），但 AUROC 仍远低于 full difference。
- L20 是 top-K 投影最强的层，K=256 AUROC 0.6948
- L24 在 full difference 上最强但 top-K 投影 AUROC 仅 0.6539
- 判別性能主要在 K=64-256 范围增长

#### Stage C Supervised (2026-04-23)
**核心发现：监督判别方向近乎正交于 top SVD 方向**

- Logistic / LDA/Fisher 1D 判别方向与 top SVD 方向接近正交
- L20 logistic weight 到 top-SVD 的投影相似度：K=4 仅 0.0004, K=256 仅 0.0734
- LDA/Fisher 投影相似度更小：K=4 基本为 0, K=256 仅 0.0123
- PLS-8 与 SVD 基础的对齐更强（K=256 达 0.7753），但 PLS 稳定性较弱

Extended K 最佳结果：
| Layer | Best K | Best AUROC |
| --- | ---: | ---: |
| 8 | 512 | 0.6919 |
| 16 | 256 | 0.6862 |
| 20 | 256 | **0.6948** |
| 24 | 1024 | 0.6603 |
| 32 | 1024 | 0.6532 |

#### Stage C Coordinate Control (2026-04-23)
**核心发现：Full SVD 坐标增益不是随机旋转效应**

| Layer | Raw full diff | PCA-whitened | Full SVD coords | Random orthogonal |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.6869 | 0.6692 | **0.7343** | 0.6776 |
| 24 | 0.6936 | 0.6331 | **0.7096** | 0.6896 |
| 32 | 0.6694 | 0.6612 | **0.7139** | 0.6711 |

- Full SVD 坐标在所有层都优于 raw full difference
- 密集随机正交旋转接近 raw full difference，而非 SVD 坐标
- 移除 top variance 方向（1-1024 全部移除后），AUROC 仍保留：L20 0.7232, L24 0.6809

---

### Stage B：条件几何分析 (2026-04-23)

**目标**: 分析匹配/随机/对抗/盲四种条件下的 correction geometry 差异。

**设置**: L20/L24/L32, 512 样本（256 FP + 256 TN）

#### Top-Backbone 观察
- Top-backbone 能量能清晰分离 image-conditioned 与 blind，但不能分离正确证据与错误证据
- K=256 matched-minus-random 均值: L20 -587.1, L24 -520.8, L32 +268.1
- Top-backbone 能量主要由"是否有图"决定，而非"证据是否正确"

#### Residual/Tail 观察（关键发现）
- Tail band 257-1024 对 matched vs mismatch 更敏感：

| Layer | Tail matched-random delta | Tail matched-adversarial delta |
| ---: | ---: | ---: |
| 20 | +5.7 | +11.8 |
| 24 | +14.2 | +25.7 |
| 32 | +17.0 | +39.1 |

- TN 样本的 matched-specific tail 增强远强于 FP 样本：
  - L32 matched-random: FP -10.2 vs TN **+44.2**

#### Supervised Decision 观察
- 固定 FP-vs-TN 监督决策分数在 matched evidence 下区分 FP/TN，在 random/adversarial 下不区分
- Logistic FP-minus-TN 均值差：
  - L24: matched +0.925, random +0.155, adversarial +0.231

#### Condition Subspace 观察
- K=4 condition SVD 基础高度相似（跨条件稳定）
- K=64/256 时，条件特定基础开始分化
- K=256 matched vs adversarial 投影相似度约 0.48（非完美相同）

**结论**: 几何是 image-condition-sensitive 的，证据相关信号在 residual/tail 和 supervised decision view 中更强。幻觉关联的是 matched evidence 下扭曲的条件特定 correction geometry。

---

### Stage E：因果干预 (2026-04-23 ~ 2026-04-24)

**目标**: 通过干预 hidden state 验证因果关联。

#### E0 预检与初版 Pilot (2026-04-23)
- No-op hook: logit 完全不变（max delta 0.0）
- Random perturbation: logit 有变化但 decoded text 不变
- 初步 FP/TN 干预: alpha ≤ 1.0 时 decoded answer 无变化 → null result

#### Stronger-Alpha Pilot (2026-04-23)
- Alpha 扩展到 1/2/4/8
- **首个因果信号**: L24 tail ablation `257-1024`
  - alpha 8: 16/16 TN → `Yes`（TN accuracy 从 1.0 降至 0.0）
- Random tail control 在 alpha 4/8 时全部变成 `unknown`
- FP steering 在所有 alpha 下仍失败

#### Clean Tail Dose Analysis (2026-04-23)
**核心因果结果 — L24 tail ablation 剂量曲线**:

| Layer | alpha 4 | alpha 5 | alpha 6 | alpha 7 | alpha 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L20 Yes rate | 0.0625 | 0.3750 | 0.7500 | 0.8750 | 0.5625 |
| L20 median margin | -0.5703 | -0.0938 | +0.2812 | +0.6289 | +1.6484 |
| L24 Yes rate | 0.0000 | 0.1250 | 0.5625 | 0.9375 | 1.0000 |
| L24 median margin | -0.7500 | -0.3281 | +0.0156 | +0.3906 | +0.9336 |

- L24 翻转到 Yes 的 median first-Yes alpha = 6.0
- Norm-matched random tail control 在 last_token 下保持 0 Yes rate
- L24 是更干净的因果证据层

#### Granularity Pilots (2026-04-23)
- `full_sequence` tail ablation 比 `last_token` 效果更强（median margin 4.7402 vs 0.9414）
- `max_new_tokens=1` 控制后，full_sequence 效果仍存活
- 但 full_sequence control 稳定性不足，last_token + norm-matched control 仍是清洁证据

#### FP Rescue 系列 (2026-04-23 ~ 2026-04-24)

**第一轮 Rescue (2026-04-23)**:
- 名义上的 supervised directions（reduce_logistic/lda_fp_score, subtract_fp_shift）不仅无效，还**反方向推动** margin
- `add_tn_correction` 是唯一有微弱正向效果的方向（+0.0312 median gain）
- 仅 1/16 FP 翻转为 No (coco:popular:336)

**反向 Rescue (2026-04-23)**:
- 名义方向确实符号反了；`reverse_logistic_fp_direction` 是当时最强方向
- alpha 8 median gain: +0.0625
- 仍只有 1/16 翻转为 No

**Local Rescue (2026-04-23)**:
- 实现 sample-conditioned 方向：`local_knn_tn_correction`、`question_tn_correction`、`object_tn_correction`
- 初版因索引 bug 未正确运行；修正后重跑
- 所有 local 方向都有 rescue signal，但 `reverse_logistic_fp_direction` 仍最强

alpha 8 median gain in logit(No)-logit(Yes):

| Direction | last_token | full_sequence |
| --- | ---: | ---: |
| `reverse_logistic_fp_direction` | +0.0625 | +0.0625 |
| `local_knn_tn_correction` | +0.0391 | +0.0391 |
| `question_tn_correction` | +0.0312 | +0.0312 |
| `object_tn_correction` | +0.0312 | +0.0312 |
| `random_rescue_control` | +0.0156 | +0.0156 |

**Multi-Layer Rescue Sweep (2026-04-23)**:
- L20/L24/L32 三层对比，核心发现：

| Layer | Best direction | Median gain | 控制干净？ |
| --- | --- | ---: | --- |
| L20 | reverse_logistic | +0.0469 | 否（random control 也翻转） |
| L24 | reverse_logistic | +0.0625 | 是 |
| L32 | **question/object_tn_correction** | **+0.0703** | 是 |

- L32 是首个 local TN-conditioned 方向超越 global reverse-logistic 的层

**Expanded Sample Rescue (2026-04-23 ~ 2026-04-24)**:
- 扩展到 32 样本再到 64 样本
- 层分化模式稳定复现：
  - L24: global reverse-logistic 最强
  - L32: local TN-conditioned 最强
- 64 样本 decoded rescue: 仅 3/64（coco:popular:336, 1348, 162），全部 baseline margin = 0.0625

**结论**: L24 tail 坐标对 TN 正确决策有因果必要性（ablating → flip to Yes）。FP rescue 效果非常弱且仅限 borderline 案例。L32 local TN-conditioned 方向是最有希望的 rescue 方向，但整体 rescue 不应被定位为可靠的 mitigation 方法。

---

### Stage G：语义投影 (2026-04-24)

**目标**: 通过 LM head 将几何方向投影到词汇空间，理解其语义内容。

**三类几何对象的语义指纹**:

| 几何对象 | 语义解读 |
| --- | --- |
| top-SVD backbone (L24) | 宽泛的视觉/场景轴：sky/clouds, trees/leaves, window/wall/floor |
| L24 tail 257-1024 | **更偏具体物体**：horse, cat, bus, cow, motor → 与 tail ablation 故事一致 |
| L32 local TN rescue | **决策/仲裁轴**：positive=with/over/near/large, negative=yes/no/despite/usually |

**Sample-Level 检查**:
- 单个语义方向不是强 hallucination detector
- 最强 FP/TN 单轴对比: L20_svd_8 AUC 仅 0.562
- 几何是**部分可解释的 grounding-related correction geometry**，不是单一语义幻觉坐标

---

### Stage J：破坏性控制 (2026-04-25)

**目标**: 验证 FP/TN 信号是否真正来自配对的视觉证据。

**Spectrum 控制**:
- Image-shuffled 和 blind-shuffled 差异的光谱与 real matched 几乎相同（有效秩偏差 < 10）
- 因此低秩 backbone 本身不足以证明 paired visual grounding

**Probe 控制**:
- Label shuffle → AUROC 接近 chance（0.498-0.508），证明信号与标签关联
- Image-shuffle 和 Gaussian 控制在 L24/L32 远弱于 real matched
- Blind-shuffle 控制在 L20/L32 仍 non-trivial

**Random Subspace 控制**:
- Plain SVD top-K 并不一致优于随机正交 K 子空间
- PCA on `z_blind` 意外地强，常优于 plain SVD top-K

**结论**: Real blind-image differences 非高斯、稳定、含 FP/TN 信号。但低秩 backbone 不是配对视觉基础的充分证据。应用 "paired correction geometry contains hallucination-relevant residual information" 而非 "low-rank backbone proves visual grounding"。

---

### Stage K：Token 位置鲁棒性 (2026-04-25)

**目标**: 验证信号是否对读出位置鲁棒。

**核心结果**:

| Position | L16 AUROC | L20 AUROC | L24 AUROC | L32 AUROC |
| --- | ---: | ---: | ---: | ---: |
| last_prompt_token | 0.6595 | 0.6588 | 0.6518 | 0.6197 |
| last_4_prompt_mean | 0.6848 | 0.7075 | **0.7108** | 0.6798 |
| last_8_prompt_mean | 0.7094 | **0.7377** | 0.7319 | 0.7023 |

- `last_4_prompt_mean` 和 `last_8_prompt_mean` 系统性地优于 single-token
- `last_8_prompt_mean` 极端集中：L16 K=4 解释方差 0.9992，但 K=4 AUROC 仍然弱
- **方差-判別解耦模式在不同读出位置下都保持**

**Condition Geometry Across Readouts**:
- 监督 logistic condition score 在所有 readout 下都分离 matched vs mismatch
- Tail band condition separation 在 single-token 下最强（L32 matched-adversarial delta +39.09）
- Pooled readouts 提升了检测但减弱了 tail condition gaps

**结论**: 信号对 token 位置鲁棒。Pooled prompt readouts 检测更强；single-token residual/tail 坐标条件几何解释更清爽。

---

### Stage L：证据特定子空间 (2026-04-25)

**目标**: 比较不同子空间构建方法。

**最佳 FP/TN Probe**:

| Method | Best layer | Best K | Best AUROC | Notes |
| --- | ---: | ---: | ---: | --- |
| **PLS FP/TN** | 24 | 32 | **0.7196** | 最强检测 |
| Fisher FP/TN | 20 | 64 | 0.6654 | 比 PLS 更稳定 |
| Plain SVD | 20 | 32 | 0.6103 | 稳定但弱 |
| Contrastive PCA | 20 | 64 | 0.6029 | 条件分离好于 FP/TN 检测 |
| Generalized | 20 | 8 | 0.5798 | 弱 |
| Mat-vs-Adv Log | 24 | 64 | 0.5757 | 弱 |

**Condition-Gap 结果**:
- Contrastive PCA 产生最大的 matched-vs-mismatch score gaps
- L32 K=64 contrastive PCA: matched-random delta **1030.3**, matched-adversarial delta **1014.4**

**稳定性**:
- Plain SVD 最稳定（split-half 相似度 > 0.99）
- PLS 检测最强但 split-half 稳定性弱（0.4-0.5）

**核心三分法**:
1. **PLS** → 最强紧凑 hallucination-detection subspace
2. **Contrastive PCA** → 最强 evidence-condition separation subspace
3. **Plain SVD** → 最稳定 dominant correction backbone，但非最佳判别对象

---

### Stage M：记忆库与局部 Rescue (2026-04-25)

**目标**: 用 train-set 记忆库进行 sample-conditioned 的 steering。

**设置**: L32, 32 FP / 32 TN / 32 TP, alpha 2/4/8, 多种检索模式

**Baseline Margins**:
| Outcome | n | Mean yes-no margin | Median |
| --- | ---: | ---: | ---: |
| FP | 32 | +0.7324 | +0.6328 |
| TN | 32 | -2.0103 | -2.2422 |
| TP | 32 | +2.3003 | +2.6484 |

**Rescue 结果** (alpha 8):
- 2/32 FP rescued（coco:popular:2714 "person", coco:popular:966 "chair"）
- Baseline margin 分别为 0.0156 和 0.0313
- TN damage = 0; TP damage 仅限 1 个 borderline 样本

**Rescue Failure 分析**:
| Label | Count |
| --- | ---: |
| margin improved but answer unchanged | 30 |
| rescued to correct No | 2 |
| no effect / damaged / wrong direction | 0 |

**核心发现**:
- Global/random/local TN 控制在 rescue rate 上竞争激烈
- Rescue 主要是 **boundary-local first-token steerability**，不是可靠的局部检索 rescue
- 30/32 FPs 的 margin 改善了但 answer 未变，只有极其 borderline 的样本被 rescue

---

### Stage N：AMBER 外部验证 (2026-04-25 ~ 2026-04-27)

**目标**: 用 POPE 训练的 geometry 零样本迁移到 AMBER。

**AMBER 全量预测质量** (14216 samples):
| Dimension | N | Accuracy |
| --- | ---: | ---: |
| attribute | 7628 | 0.798 |
| existence | 4924 | 0.878 |
| relation | 1664 | 0.712 |
| overall | 14216 | 0.816 |

**最佳 Transfer 行** (POPE-trained → AMBER):
| Layer | Dimension | Feature | FP AUROC |
| ---: | --- | --- | ---: |
| 24 | relation | evidence_fisher_fp_tn_k64 | 0.665 |
| 24 | relation | pope_probe_top_256 | 0.664 |
| 24 | existence | pope_probe_top_64 | 0.663 |
| 20 | existence | pope_probe_top_4 | 0.661 |
| 24 | existence | evidence_fisher_fp_tn_k16 | 0.657 |

- Top AMBER transfer rows 在 0.63-0.665 范围
- Evidence-specific Fisher/PLS transfer 有竞争力
- Raw tail energy 弱（最佳仅 0.561）

**结论**: POPE-trained risk geometry transfers above chance to AMBER，但效果 modest。适合写为 modest external validity，不宜写为强泛化。

---

### Stage P：多种子鲁棒性 (2026-04-27)

**目标**: 验证 FP/TN detection 的统计稳定性。

**5 种子结果** (seeds 13/17/23/29/31):

| Layer | Feature | Mean AUROC | Std | 95% CI |
| ---: | --- | ---: | ---: | --- |
| 24 | full_diff | **0.721** | 0.027 | 0.699-0.741 |
| 20 | full_diff | **0.720** | 0.028 | 0.696-0.741 |
| 16 | full_diff | 0.714 | 0.019 | 0.699-0.727 |
| 32 | full_diff | 0.703 | 0.021 | 0.685-0.717 |
| 20 | top_256 | 0.677 | 0.035 | 0.653-0.706 |
| 32 | tail_257_1024 | 0.667 | 0.028 | 0.646-0.686 |
| 24 | tail_257_1024 | 0.656 | 0.046 | 0.622-0.691 |
| 24 | top_4 | **0.471** | 0.007 | 0.466-0.476 |

**配对 Bootstrap 结论**:
- `top_256 > top_4`: L24 delta +0.193, 95% CI 0.155-0.227
- `full_diff > top_256`: L24 delta +0.053, 95% CI 0.032-0.075
- `full_diff` 在所有 5 种子的所有层都是 rank 1

**结论**: 方差-判別解耦对种子鲁棒。Full difference 始终是最强检测器。Tail/residual 信号稳定但不应被定位为最佳 standalone detector。

---

### Stage O：跨模型与跨架构复现 (2026-05-04 ~ 2026-05-07)

#### LLaVA-1.5-13B 复现 (2026-05-04)
**结果：acceptable-to-strong checkpoint-level replication**

| Layer | Full diff AUROC | Top-4 AUROC |
| ---: | ---: | ---: |
| 20 | **0.736** | 0.549 |
| 24 | 0.726 | 0.535 |
| 32 | 0.723 | 0.552 |

- 方差-判別 mismatch 清晰复现
- Tail 条件几何部分复现（adversarial tail deltas 全部为正：L20 18.13, L24 28.86, L32 75.50）
- 最佳 projected: L32 K=128 AUROC 0.699

#### Phase 3 跨架构审计 (2026-05-06 ~ 2026-05-07)

**Assistant-Prompt Readout 污染问题 (2026-05-06)**:
- Qwen/InternVL 初版结果异常完美（AUROC ≈ 1.000）
- 根因：`last_prompt_token` 在 assistant generation prompt 位置，直接暴露 next-token 决策
- 修复：切换到 `last_user_content_token`

**User-Content Readout 修正后 (2026-05-07)**:

| 模型 | Best difference AUROC | Margin entropy FP-vs-TP AUROC |
| --- | ---: | ---: |
| Qwen2-VL-7B | **0.772** | 0.869 |
| Qwen2.5-VL-7B | 0.771 | 0.883 |
| InternVL2-8B | **0.999** | 0.883 |
| InternVL2.5-8B | 0.998 | 0.903 |

**InternVL 深入审计 (2026-05-07)**:
- 即使在 `last_question_token` 读出下，InternVL FP/TN AUROC 仍 0.998-1.000
- 但部署视角（predicted-Yes FP-vs-TP）：
  - InternVL2-8B difference FP-vs-TP AUROC: **0.187**（top 10% trigger 抓到 1/18 FP）
  - InternVL2.5-8B difference FP-vs-TP AUROC: **0.121**（top 10% trigger 抓到 0/47 FP）
- InternVL 的 FP/TN separability 更像在区分的"模型是否会回答 Yes/No 的内部状态"，而非可部署的 hallucination risk signal
- **InternVL 应被报告为 geometry-only gating 的警示性跨架构失败案例**

**跨架构部署 Gate 结论**:
- Margin entropy 在所有四个模型上都优于 geometry gate（FP-vs-TP）
- Qwen 的 geometry 信号处于中等水平（AUROC 0.67-0.77），有一定互补价值
- InternVL 的 geometry 信号在部署设置中基本无用

---

### Stage R：Case Panels 与语义指纹 (2026-05-04)

- **Case panels (R2)**: 22 个可读案例，涵盖 successful TN / weak FP / rescued FP / unrescued FP / adversarial mismatch / semantic extremes
- **Semantic fingerprints (R1)**: 28 个投影几何对象，含 top-SVD backbone + L24 tail slice + L32 local rescue + Stage L 证据子空间
  - Top-SVD backbone：宽视觉/场景/属性轴
  - L24 tail：物重，与 ablation 故事一致
  - L32 rescue：关系/语境/决策仲裁方向
- 无单个 projected direction 是强 hallucination detector

---

### Stage S：基线定位 (2026-05-04)

**检测基线对比**:
| Method | AUROC |
| --- | ---: |
| yes/no margin | **1.000** (stage M subset) |
| binary entropy | 0.884 |
| paired full difference (5-seed) | **0.721** |
| PLS FP/TN | 0.720 |
| raw blind hidden probe | 0.672 |
| raw image hidden probe | 0.651 |

- Paired-difference 不应写为纯检测 leaderboard win
- 更强的主张是 mechanistic：配对差分定位了幻觉相关信号，并揭示了为什么 top-variance 不是 decision geometry

**Mitigation/Rescue 基线**: 当前 Stage M rescue 弱且 boundary-local，local/global/random 控制在 gated subset 上接近。

---

### Stage Q：论文资产生成 (2026-04-27)

生成 5 张图和 4 张表：
- Figure 1: 方法总览
- Figure 2: 方差 vs AUROC 解耦
- Figure 3: 匹配 vs 不匹配条件几何
- Figure 4: 干预剂量曲线
- Figure 5: 分层几何总结
- Table 1-4: POPE 总结 / 特征比较 / 破坏性控制 / 干预总结

---

### Stage T：选择性修正 (2026-05-04 ~ 2026-05-07)

**目标**: 用 correction geometry 作为风险信号进行选择性 gate → verification/VCD/ICD。

#### Gate 设计
- 仅在模型预测 Yes 的样本上触发
- 核心度量：FP Recall、TP Damage、Gate Precision、FP Reduction per Trigger

#### Strict Subset-Transfer Protocol
Train: POPE random → Calibration: POPE popular → Test: POPE adversarial

At 20% trigger rate：
| Score | Triggered FP Ratio | FP Recall | TP Damage |
| --- | ---: | ---: | ---: |
| random64_probe | 0.202 | 0.296 | 0.184 |
| tail_257_1024_energy | 0.174 | 0.259 | 0.193 |
| pls32_probe | 0.165 | 0.238 | 0.189 |
| full_probe | 0.148 | 0.217 | 0.196 |
| top_4_probe | 0.127 | 0.185 | 0.200 |

#### Fixed-Split Protocol（主要正面结果）
At 20% predicted-Yes trigger rate：
| Score | Triggered FP Ratio | FP Recall | TP Damage |
| --- | ---: | ---: | ---: |
| **pls32_probe** | 0.226 | **0.396** | 0.133 |
| **tail_257_1024_probe** | 0.216 | **0.472** | 0.168 |
| **full_probe** | 0.207 | **0.453** | 0.169 |
| random64_probe | 0.164 | 0.396 | 0.197 |
| top_64_probe | 0.125 | 0.302 | 0.206 |
| top_4_probe | 0.025 | 0.057 | 0.215 |

**PLS/tail/full 选择远超 random 的 FPs，top-4 仍然弱** — 这是最干净的正面 gate 结果。

#### Verification（提示词修正）
- 所有 verification prompt（legacy/forced_evidence/conservative/internal_rationale）只能改变 ~16/163 FPs
- 更强的 prompt 反而让模型更倾向说 Yes
- **Prompt 修正不是强 correction operator**，但 gate 精准性 claim 存活

#### Selective Warning / Abstention
**最稳定的部署故事**:

| Target | Method | Trigger rate | FP recall | TP damage | Warning precision |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.20 | Low-margin+FullD | 0.180 | 0.660 | 0.133 | **0.327** |
| 0.20 | Low-margin+PLS | 0.176 | 0.642 | 0.131 | **0.324** |
| 0.20 | PLS only | 0.156 | 0.396 | 0.133 | 0.226 |
| 0.20 | Random | 0.156-0.223 | 0.155-0.233 | 0.156-0.222 | 0.088-0.093 |
| 0.20 | High-margin | 0.196 | 0.000 | 0.215 | 0.000 |

Base predicted-Yes FP prevalence ≈ 0.089；low-margin+geometry 将 warning precision 提升到 0.324-0.327。

#### Gated VCD/ICD（最强选择性修正结果）

**四个 operator 的 always-on 效果** (fixed-split test, 596 predicted-Yes):
| Operator | Always FP reduction | TP preserved | Accuracy delta |
| --- | ---: | ---: | ---: |
| `icd_blind` | 0.340 | 0.912 | -0.022 |
| `vcd_diffusion` | 0.302 | 0.912 | -0.024 |
| `vcd_gray` | 0.264 | 0.890 | -0.034 |
| `vcd_blur` | 0.151 | 0.971 | -0.006 |

**最佳 Selective Rows**:

| Operator | Best gate | Target | Trigger | FP reduction | TP preserved | Acc delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `icd_blind` | low-margin+PLS | 0.20 | 0.176 | **0.321** | 0.965 | **-0.001** |
| `icd_blind` | low-margin+tail | 0.30 | 0.297 | **0.340** | 0.937 | -0.012 |
| `icd_blind` | full_probe | 0.30 | 0.262 | 0.245 | 0.985 | **+0.004** |
| `icd_blind` | margin+full | 0.30 | 0.267 | 0.170 | 0.994 | **+0.004** |

**Bootstrap 95% CI 关键行**:
| Operator / gate | Target | Metric | Point | 95% CI |
| --- | ---: | --- | ---: | ---: |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | FP reduction | 0.321 | [0.196, 0.452] |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | TP preserved | 0.965 | [0.949, 0.980] |
| `icd_blind + full_probe` | 0.30 | FP reduction | 0.245 | [0.130, 0.370] |
| `icd_blind + full_probe` | 0.30 | Accuracy delta | +0.004 | [-0.003, 0.010] |

**Tradeoff 洞察**:
- **最大化 FP 捕获** → `low_margin+geometry`（icd_blind+low_margin+tail@30%: FP reduction 0.340，匹配 always-on 但省 70.3% 计算）
- **保护 TP** → `geometry-only` 或 `high_margin+geometry`（icd_blind+full@30%: TP preserved 0.985, accuracy 略升）
- Random gate 在相同 trigger 下远弱（best random FP reduction 仅 0.120）

#### 必补实验：Fixed-Trigger Margin-only vs Geometry-only vs Margin+Geometry

新增输出：
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_margin_geometry_fixed_trigger_ablation.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_margin_geometry_fixed_trigger_ablation.md`

设置：fixed-split held-out POPE test，L24，只在 predicted-`Yes` pool 上精确固定 trigger budget（596 个 predicted-`Yes`，其中 53 FP / 543 TP）。`Margin-only` 使用 reviewer-relevant 的 `low_margin_probe`；`Margin + ...` 使用 `low_margin_plus_*`。

Canonical `icd_blind` 结论：

| Target | Best margin+geometry | FP recall | TP damage | Warning precision | ICD/VCD FP reduction | TP preserved | Acc delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | margin+full/PLS/tail | 0.434 | 0.068 | 0.383 | 0.208-0.264 | 0.982-0.993 | +0.001~+0.005 |
| 0.20 | margin+full/PLS | 0.679 | 0.155 | 0.300 | 0.321-0.340 | 0.954-0.965 | -0.005~-0.001 |
| 0.30 | margin+PLS | 0.849 | 0.247 | 0.251 | 0.340 | 0.932 | -0.014 |

Against `Margin-only`:
- 10%: margin+geometry 明显更好（FP recall 0.434 vs 0.396；warning precision 0.383 vs 0.350；TP damage 0.068 vs 0.072）。
- 20%: margin+full/PLS 小幅更好（FP recall 0.679 vs 0.660；precision 0.300 vs 0.292），且 actual ICD/VCD 的 TP/accuracy tradeoff 更好。
- 30%: margin+PLS 在 FP recall 上略优（0.849 vs 0.830），但整体是 margin 主导，geometry 增益不应夸大。

写法决策：Stage T 可以保留为 **complementary risk signal / selective routing tradeoff**。不能写成 geometry 全面优于 margin；更稳的表述是：low-margin 是强基线，geometry 在低预算 warning precision 和 TP-preserving correction routing 上提供增量价值。

#### Strict Subset-Transfer VCD/ICD（压力测试）
- 效果弱于 fixed-split，但 selective routing 仍好于 always-on
- Best: `icd_blind + pls32_probe` @20%: FP reduction 0.079, TP preserved 0.988

#### AMBER External Warning / Gated Operator
- Tail energy 是最强外部 transfer score
- `icd_blind + tail_257_1024_energy` @20%: FP reduction 0.100, TP preserved 0.952
- External top-rate 评估下 tail energy 显著优于 random

---

## 三、跨所有阶段的关键发现总结

### 强证据支持的发现

1. **方差-判別解耦**（Stage C/P）：Top variance directions 不是 hallucination decision directions。Top-4 解释 >80% 方差但 AUROC ≈ 0.47。此模式对 layer、seed、readout position、checkpoint (7B/13B) 全部鲁棒。

2. **Residual/Tail 坐标的信号**（Stage B/C/E）：FP/TN 判别信号分布在 full difference、mid/high-dimensional SVD coordinates (K=64-256)、tail bands (257-1024) 和 PLS/Fisher 监督子空间中。

3. **L24 tail 的因果必要性**（Stage E）：Ablating L24 `257-1024` tail slice 以剂量依赖方式将 TN → Yes，norm-matched control 保持 No。此效应是当前最清洁的因果证据。

4. **条件敏感几何**（Stage B）：Matched-vs-mismatch separation 在 tail/supervised view 中而非 top-backbone energy 中显示。TN 显示更强的 matched-specific tail 修正。

5. **选择性 Gate 的 utility**（Stage T）：PLS/tail/full probe 在 predicted-Yes 子集中选择的 FP 远超 random/top-4 gate。Low-margin+geometry 将 warning precision 从 0.089 (base rate) 提升到 0.327。Selective VCD/ICD routing 将 always-on 的破坏性 operator 转化为有用的部署选择。

6. **LLaVA-family 复现**（Stage O/13B）：Full difference 强、top-4 弱、adversarial tail gap 存在 → 模式在 7B/13B 间复现。

### 部分支持的发现

7. **语义可解释性**（Stage G）：Top backbone = 宽视觉轴；Tail = 物重方向；L32 rescue = 决策仲裁方向。但无单个语义方向是强 detector。

8. **AMBER 外部迁移**（Stage N/T）：Above-chance but modest（top rows ≈ 0.63-0.67）。Tail energy 外部最强。

### 不支持的发现 / 限制

9. **FP Rescue 弱**（Stage E/M）：仅在 baseline margin ≈ 0.015-0.062 的 borderline FP 上翻转。30/32 FPs 的 margin 改善但 answer 不变。Global/random 控制与 local rescue 竞争。

10. **InternVL 跨架构失败**（Phase 3）：InternVL 有近乎完美的 FP/TN separability（AUROC ≈ 1.0），但在 predicted-Yes FP-vs-TP 上失败（AUROC ≈ 0.12-0.22）。是 geometry-only gating 的警示案例。

11. **Qwen 中等信号**（Phase 3）：Qwen2-VL / Qwen2.5-VL difference AUROC ≈ 0.77，Geometry 在部署设置中不及 margin entropy（FP-vs-TP AUROC 0.67-0.77 vs 0.87-0.88）。

12. **Prompt Verification 弱**（Stage T）：Prompt 修正只能改变 ~10% FPs；更强的 prompt 反使模型更倾向说 Yes。

---

## 四、当前论文定位

### 论文类型：机制分析论文（非 mitigation 方法论文）

**题目**: "Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination"

### 四项核心贡献

1. 引入 blind-reference differencing 分析 LVLM 中的 visual-evidence correction geometry
2. 证明 dominant variance directions 不是 hallucination decision directions
3. 识别 residual/tail correction coordinates 具有证据敏感性和因果相关性
4. 证明 correction geometry 提供与 output confidence 互补的风险信号，支持选择性验证/VCD 路由

### 部署叙事（Stage T 后）

- **Warning/Abstention**: low-margin+geometry 是最强部署风险路由器
- **Correction**: geometry-only 或 high-margin+geometry 是 TP-preserving VCD/ICD 路由器
- **主要限制**: 最强 utility 证据是 fixed-split held-out POPE；strict subset-transfer 和 AMBER 压力测试有支持性但较弱

---

## 五、实验阶段索引

| Stage | 日期 | 描述 | 关键产出路径 |
| --- | --- | --- | --- |
| A | 04-22 | 差分校谱 | `outputs/svd/` |
| C | 04-22~23 | FP/TN Probe 与坐标控制 | `outputs/stage_c_*/` |
| B | 04-23 | 条件几何（matched/mismatch/blind） | `outputs/stage_b/` |
| E | 04-23~24 | 因果干预与 Rescue | `outputs/interventions/` |
| G | 04-24 | 语义投影到词汇空间 | `outputs/semantics/` |
| J | 04-25 | 破坏性与随机控制 | `outputs/stage_j_controls/` |
| K | 04-25 | Token 位置鲁棒性 | `outputs/stage_k_*/` |
| L | 04-25 | 证据特定子空间 | `outputs/stage_l_evidence_subspace/` |
| M | 04-25 | 记忆库与局部 Rescue | `outputs/stage_m_local_rescue/` |
| N | 04-25~27 | AMBER 外部验证 | `outputs/stage_n_external*/` |
| P | 04-27 | 多种子统计鲁棒性 | `outputs/stage_p_stats/` |
| Q | 04-27 | 论文图/表资产生成 | `outputs/paper_figures/`, `outputs/paper_tables/` |
| R | 05-04 | Case Panels 与语义指纹 | `outputs/case_studies/`, `outputs/stage_r_semantics/` |
| S | 05-04 | 基线定位 | `outputs/stage_s_baselines/` |
| O | 05-04~07 | 13B 复现 + Phase 3 跨架构 | `outputs/stage_o_cross_model*/` |
| T | 05-04~07 | 选择性修正 Gate/VCD/ICD | `outputs/stage_t_selective_correction*/` |

---

## 六、当前待办与开放问题

1. **跨架构选择性 gate**: 需要在 Qwen/InternVL 上运行完整的 selective gate + VCD pipeline
2. **InternVL 审计**: 检查更早层或 pre-answer-free forward 设定，确认其强 FP/TN 分离的真正来源
3. **Representation editing pilot**: 已有 CPU 准备的 direction bank（`outputs/representation_editing_prep/`），尚未 GPU 评估
4. **Evidence-specific steering**: Stage L 的 PLS/Fisher/contrastive PCA 方向尚未接入干预 pipeline
5. **Compute overhead measurement**: VCD/ICD 的选择性路由计算成本需要更细致的量化
6. **论文写作**: Stage Q 的图和表可以开始用于论文草案
