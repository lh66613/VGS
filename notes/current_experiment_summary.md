# 当前实验结果系统性总结

生成日期：2026-05-04  
主线题目建议：**Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination**

本文档基于 `TODO_list.md`、`TODO_list_extend.md`、`notes/findings.md`、`notes/final_paper_framing_decision.md` 以及当前 `outputs/` 下的实验结果整理。写法刻意接近论文中的 Method / Experiments / Results，方便后续直接改成论文草稿或答辩材料。

---

## 1. 总览结论

当前证据最适合支撑一篇**机制分析论文**，而不是一个强 mitigation method 论文。

核心结论可以概括为：

> 通过 paired blind-reference differencing，即比较 image+question 隐状态和 blind/text-only 隐状态，能够揭示 LLaVA 中与视觉证据校正、幻觉判别和边界决策相关的一组分层几何结构。该结构不是一个单一、低维、通用的 visual grounding subspace，而更像是 **evidence-sensitive correction geometry**：高方差主干、mid/tail 残差坐标和 late-layer 局部仲裁方向分别承担不同角色。

最稳妥的贡献表述：

- blind-image difference 中存在稳定、非高斯、可解释的 correction geometry；
- 最大方差方向并不是最强 hallucination decision geometry；
- FP/TN 判别信号分布在中高维、mid/tail 或全差分坐标中；
- matched visual evidence 与 random/adversarial mismatch 的几何响应不同，尤其在 residual/tail 和 supervised decision score 视角下更明显；
- L24 residual/tail 坐标对正确 TN 的 first-token yes/no 决策有因果相关性；
- FP rescue 可以移动边界样本的 margin，但还不是可靠的缓解方法；
- AMBER 和 LLaVA-1.5-13B 结果支持 modest external / checkpoint-level generalization，但还不能声称跨架构通用。

推荐名称：

- 主对象：**evidence-sensitive correction geometry**
- 机制表达：**hallucination-sensitive residual correction coordinates**
- 总体 framing：**layered visual-evidence correction geometry**

需要避免的名称：

- visual grounding subspace
- universal hallucination subspace
- causal grounding direction
- reliable hallucination mitigation method

---

## 2. Original Idea

最初想法来自一个 paired reference 设定。对每个样本，在同一个模型和同一个问题下提取两种隐状态：

- `z_img`: image + question 条件下的 hidden state；
- `z_blind`: blind / text-only 条件下的 hidden state。

定义每一层的差分矩阵：

```text
D^(l) = { z_blind_i^(l) - z_img_i^(l) }_{i=1}^N
```

最早的假设是：这个 blind-reference difference 可能捕获模型从语言先验状态向视觉证据修正状态移动的几何结构。如果这个结构真实存在，它应该至少满足四个问题：

1. 是否是真实稳定结构，而不是噪声或 pipeline artifact？
2. 是否 evidence-specific，而不只是“有图像输入”带来的 generic modality shift？
3. 是否与 hallucination 相关，而不只是最大方差方向？
4. 是否能被干预影响模型行为，还是只能诊断？

当前实验的主要变化是：原始“低维 visual grounding subspace”的强假设被修正为更谨慎也更准确的“分层 correction geometry”。这是一个好变化，论文味道更稳了。

---

## 3. Method

### 3.1 Model and Data

主实验：

- 模型：`LLaVA-1.5-7B`，HuggingFace `LlavaForConditionalGeneration`
- checkpoint：`/data/lh/ModelandDataset/llava-1.5-7b-hf`
- 主 benchmark：POPE COCO，三种 subset：
  - random
  - popular
  - adversarial
- 样本数：9000
- 主要 layers：8 / 12 / 16 / 20 / 24 / 28 / 32
- 主 readout：`last_prompt_token`
- robustness readout：`last_4_prompt_mean`，另测试 `last_8_prompt_mean`

外部验证：

- AMBER discriminative full set：14216 rows
- 额外 checkpoint：`LLaVA-1.5-13B`，artifact alias 为 `llava_13b`

### 3.2 Representation Construction

对每个样本和每层抽取：

- image-conditioned representation: `z_img`
- blind/text-only representation: `z_blind`
- paired correction vector: `d = z_blind - z_img`

之后对 `D` 做：

- SVD，得到 top variance backbone；
- SVD coordinate projection；
- residual / tail band 评分，例如 `257-1024`；
- full difference vector probe；
- evidence-specific subspace extraction。

### 3.3 Geometry Families

当前比较过的 geometry / feature family 包括：

- raw `z_img`
- raw `z_blind`
- full difference `z_blind - z_img`
- top-K SVD coordinates
- full SVD coordinates
- SVD residual/tail band，例如 `257-1024`
- random projection / random orthogonal subspace
- image-state PCA / blind-state PCA
- evidence-specific subspaces：
  - contrastive PCA
  - generalized matched-vs-mismatch eigenspace
  - Fisher FP/TN
  - PLS FP/TN
  - matched-vs-adversarial logistic direction

### 3.4 Probing and Metrics

主判别任务是 POPE 的 FP-vs-TN，即在 ground-truth `no` 样本中区分：

- FP: 模型 hallucinated `yes`
- TN: 模型正确回答 `no`

主要指标：

- AUROC
- AUPRC
- accuracy / F1
- split-half subspace stability
- explained variance
- effective rank
- matched-vs-random / matched-vs-adversarial condition gap
- yes/no first-token margin shift
- rescue / damage / unknown rate

Stage P 进一步使用 5 seeds 和 stratified paired bootstrap 检查稳定性。

### 3.5 Condition Geometry

Stage B/K/O 构造了四种视觉证据条件：

- matched image
- random mismatch image
- adversarial mismatch image
- blind / no image

目标是区分：

- image presence movement：只要有图就发生的大幅移动；
- evidence-correct correction：matched visual evidence 特有的修正；
- hallucination-sensitive residual：FP/TN 在 matched evidence 下的异常差异。

### 3.6 Intervention

Stage E/M 主要验证因果相关性：

- TN tail ablation：对原本正确的 TN 样本，移除 L24 residual/tail SVD slice `257-1024`；
- FP rescue / steering：对 FP 样本沿 supervised、global TN、local TN、retrieval-based correction direction 移动；
- first-token 评估：重点看 `logit(No) - logit(Yes)` 和 decoded yes/no flip；
- Stage M 使用 train-only memory bank，避免 test leakage，并加入 random TN retrieval control。

---

## 4. Experiments

### 4.1 Stage A: Difference Geometry and Spectrum

目的：证明 `D = z_blind - z_img` 不是随机噪声。

Artifacts:

- `outputs/svd/`
- `outputs/plots/spectrum_layer_*.png`
- `outputs/svd/effective_rank_summary.csv`

主要发现：

- `D` 具有明显非高斯、低秩集中的谱结构；
- top-4 directions 在很多层解释了很高方差；
- L32 的谱比中间层更分散；
- 但后续控制显示：低秩谱本身不能证明 paired visual grounding，因为 image/blind shuffle 也保留了类似谱。

### 4.2 Stage C/P: Hallucination Detection and K Sensitivity

目的：测试差分几何是否区分 FP/TN。

Artifacts:

- `outputs/probes/`
- `outputs/stage_c_deep/`
- `outputs/stage_c_supervised/`
- `outputs/stage_c_coordinate_control/`
- `outputs/stage_p_stats/`
- `outputs/paper_tables/table2_feature_comparison.csv`

关键现象：

- full difference 是最稳定的 hidden-state detector；
- top-4 SVD directions 尽管解释大部分方差，但 FP/TN AUROC 接近 chance；
- AUROC 主要在 K=128/256 或 full difference 时上升；
- supervised PLS/Fisher 可提取更 compact 的 detection subspace。

代表结果：

| Setting | Result |
| --- | ---: |
| POPE full difference, single run L24 | AUROC 0.694 |
| Coordinate-control full SVD coords, L20 | AUROC 0.734 |
| Stage L PLS FP/TN, L24 K=32 | AUROC 0.720 |
| Stage P full difference, L24 5-seed mean | AUROC 0.721, 95% CI 0.699-0.741 |
| Stage P top-256, L20 5-seed mean | AUROC 0.677 |
| Stage P tail 257-1024, L32 5-seed mean | AUROC 0.667 |
| Stage P top-4, L24 5-seed mean | AUROC 0.471 |

解释：

> 幻觉相关信号不是最大方差方向，也不是一个很小的 top-K compact subspace。它更分布式，full difference、full SVD coordinates、PLS/Fisher 或 tail coordinates 都能捕获部分信号。

### 4.3 Stage B: Matched vs Mismatched Evidence

目的：证明 geometry 与正确视觉证据相关，而不只是图像输入带来的 generic shift。

Artifacts:

- `outputs/stage_b/stage_b_condition_score_summary.csv`
- `outputs/stage_b/stage_b_pairwise_condition_deltas.csv`
- `outputs/stage_b/stage_b_condition_subspace_similarity.csv`

关键发现：

- top-backbone energy 主要反映 image-conditioned movement，不稳定区分 matched 与 mismatch；
- residual/tail band `257-1024` 对 matched evidence 更敏感；
- supervised FP/TN decision score 在 matched evidence 下区分 FP/TN，在 mismatch 下显著弱化；
- TN 的 matched-specific residual/tail correction 更强，FP 更弱或异常。

代表 tail gap：

| Layer | Tail matched-random | Tail matched-adversarial |
| ---: | ---: | ---: |
| L20 | +5.7 | +11.8 |
| L24 | +14.2 | +25.7 |
| L32 | +17.0 | +39.1 |

代表 supervised logistic FP-minus-TN gap：

| Layer | Matched | Random mismatch | Adversarial mismatch |
| ---: | ---: | ---: | ---: |
| L20 | +0.623 | +0.102 | +0.133 |
| L24 | +0.925 | +0.155 | +0.231 |
| L32 | +0.834 | +0.101 | +0.196 |

解释：

> evidence-specific 信号主要体现在 residual/tail 和 supervised decision geometry，而不是简单 top variance energy。

### 4.4 Stage J: Destructive and Random Controls

目的：防止把 artifact 误读成 grounding。

Artifacts:

- `outputs/stage_j_controls/shuffle_spectrum_summary.csv`
- `outputs/stage_j_controls/shuffle_probe_summary.csv`
- `outputs/stage_j_controls/random_subspace_control.csv`
- `outputs/plots/stage_j_real_vs_shuffle_*.png`

关键发现：

- real matched `D` 比 Gaussian control 更真实、更有信号；
- 但 image-shuffle / blind-shuffle 的 singular spectrum 与 real matched 非常接近；
- label-shuffled probe 接近 chance，说明 FP/TN 信号不是纯分类器 artifact；
- blind-shuffle control 仍有非平凡信号，说明 text/blind side 也包含 hallucination-relevant 信息；
- random subspace 与 plain SVD top-K 接近，plain top-SVD 不能被说成唯一有效子空间。

代表 AUROC：

| Layer | Real full SVD | Image shuffle | Blind shuffle | Gaussian | Label shuffle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| L20 | 0.606 | 0.526 | 0.652 | 0.570 | 0.508 |
| L24 | 0.684 | 0.515 | 0.617 | 0.500 | 0.498 |
| L32 | 0.677 | 0.502 | 0.643 | 0.476 | 0.449 |

解释：

> 低秩谱是 correction geometry 的背景结构，但不是 paired grounding 的充分证据。更强的证据来自 FP/TN probe、matched-vs-mismatch gap、tail behavior 和 intervention。

### 4.5 Stage K: Token-Position Robustness

目的：回答“为什么读 last prompt token”。

Artifacts:

- `outputs/stage_k_positions/position_probe_summary.csv`
- `outputs/stage_k_positions/position_spectrum_summary.csv`
- `notes/readout_position_decision.md`

结果：

| Readout | Full-difference AUROC pattern |
| --- | --- |
| `last_prompt_token` | 0.620-0.659 |
| `last_4_prompt_mean` | 0.680-0.711 |
| `last_8_prompt_mean` | 0.702-0.738 |

关键解释：

- pooled prompt readout 提高 FP/TN detection；
- 但 `last_prompt_token` 与已有 Stage A/B/C/E 连续，仍作为主 paper protocol；
- `last_8_prompt_mean` 中早/中层 top-4 explained variance 接近 1.0，但 top-4 AUROC 仍弱，进一步支持“variance geometry != decision geometry”。

### 4.6 Stage L: Evidence-Specific Subspaces

目的：从 plain SVD 升级为更贴近 evidence / hallucination 的 subspace extraction。

Artifacts:

- `outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv`
- `outputs/stage_l_evidence_subspace/evidence_subspace_condition_gap.csv`
- `outputs/stage_l_evidence_subspace/evidence_subspace_stability.csv`
- `notes/naming_decision.md`

代表结果：

| Method | Best layer | Best K | Best AUROC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| PLS FP/TN | L24 | 32 | 0.720 | 最强 compact detection |
| Fisher FP/TN | L20 | 64 | 0.665 | 稳定但较弱 |
| Plain SVD | L20 | 32 | 0.610 | 稳定主干，不是最佳判别 |
| Contrastive PCA | L20 | 64 | 0.603 | FP/TN 弱，但 condition gap 强 |

特别重要：

- Contrastive PCA 在 L32 matched-vs-mismatch condition gap 最强；
- PLS detection 强但 split-half stability 较弱；
- Plain SVD stability 最强，但 detection 较弱。

解释：

> 不存在一个“所有任务都最好”的子空间。检测、证据区分、稳定主干、因果干预分别偏向不同 geometry。

### 4.7 Stage E/M: Causal Intervention and Rescue

目的：测试 geometry 是否影响模型行为。

Artifacts:

- `outputs/interventions/`
- `outputs/stage_m_local_rescue/`
- `notes/rescue_failure_analysis.md`
- `outputs/paper_tables/table4_intervention.csv`

TN tail ablation 的结论最强：

- L24 tail `257-1024` ablation 对正确 TN 的 first-token `No` 决策有剂量效应；
- `max_new_tokens=1` 下，L24 full-sequence alpha 6 让 8/8 TN 翻为 `Yes`；
- last-token alpha 6 让 5/8 TN 翻为 `Yes`；
- norm-matched random-tail last-token control 在 alpha 4/5/6 保持 8/8 为 `No`。

代表表：

| Intervention | Granularity | Alpha | Yes rate |
| --- | --- | ---: | ---: |
| L24 tail ablation | full_sequence | 5 | 0.50 |
| L24 tail ablation | full_sequence | 6 | 1.00 |
| L24 tail ablation | last_token | 5 | 0.25 |
| L24 tail ablation | last_token | 6 | 0.625 |
| norm-matched random tail | last_token | 6 | 0.00 |

FP rescue 的结论要谨慎：

- global supervised rescue 起初方向存在 sign mismatch；
- sign-reversed logistic 和 local TN-conditioned directions 能让 margin 朝 `No` 移动；
- L32 local TN-conditioned rescue 在 Stage E larger-sample check 中可救回少数 borderline FP；
- Stage M memory-bank/gated rescue 中，仅 2/32 FP 被救回，且 rescued samples baseline margin 分别只有 0.015625 和 0.03125；
- random/global/local TN controls 在小 gated subset 上相互接近，因此不能声称 local retrieval 是必要机制。

解释：

> intervention 支持“causal relevance”，尤其是 L24 tail 对 TN faithfulness；但当前 rescue 只是 boundary-local first-token steerability，不是 reliable mitigation。

### 4.8 Stage N: External Validity on AMBER

目的：验证 POPE-trained geometry 是否只适用于 POPE。

Artifacts:

- `outputs/stage_n_external_full/`
- `outputs/stage_n_external_full/external_category_summary.csv`
- `notes/external_benchmark_choice.md`

AMBER full prediction quality：

| Dimension | N | Accuracy | Outcomes |
| --- | ---: | ---: | --- |
| attribute | 7628 | 0.798 | 3044 TP / 3044 TN / 770 FP / 770 FN |
| existence | 4924 | 0.878 | 4325 TN / 599 FP |
| relation | 1664 | 0.712 | 552 TP / 632 TN / 57 FP / 423 FN |
| overall | 14216 | 0.816 | 3596 TP / 8001 TN / 1426 FP / 1193 FN |

强 transfer rows：

| Layer | Dimension | Feature | FP AUROC |
| ---: | --- | --- | ---: |
| L24 | relation | Fisher FP/TN K=64 risk | 0.665 |
| L24 | relation | POPE top-256 risk | 0.664 |
| L24 | existence | POPE top-64 risk | 0.663 |
| L20 | existence | POPE top-4 risk | 0.661 |
| L24 | attribute | PLS FP/TN K=8 risk | 0.633 |

解释：

- full AMBER 确认 above-chance transfer；
- pilot 高估了峰值，full set 更保守；
- raw energy / raw tail energy transfer 弱，best tail-energy FP AUROC 约 0.561；
- POPE-trained risk geometry modestly transfers beyond POPE；
- 外部有效性存在，但不是很强。

### 4.9 Stage O: LLaVA-1.5-13B Replication

目的：检查是否只是 7B 单 checkpoint artifact。

Artifacts:

- `outputs/stage_o_cross_model/llava_13b/`
- `notes/cross_model_replication.md`

结果：

- POPE accuracy: 0.871
- FP / TN / TP / FN: 468 / 4032 / 3804 / 696
- full difference AUROC：
  - L20: 0.736
  - L24: 0.726
  - L32: 0.723
- top-4 projected AUROC 仍弱：
  - L20: 0.549
  - L24: 0.535
  - L32: 0.552
- best projected row: L32 K=128 AUROC 0.699
- tail matched-adversarial gap：
  - L20: +18.126
  - L24: +28.860
  - L32: +75.504

解释：

> 7B 上的核心 qualitative pattern 在 13B checkpoint 上复现：full difference 强、top-variance 不等于 decision、tail adversarial gap 存在。但这仍是 LLaVA-family replication，不是跨架构通用性。

### 4.10 Stage R/S/T: Semantic, Baseline, Final Framing

Semantic artifacts:

- `outputs/stage_r_semantics/semantic_fingerprint_summary.csv`
- `outputs/stage_r_semantics/semantic_sample_panels/`
- `notes/semantic_interpretation_conclusion.md`
- `notes/case_studies.md`

语义结论：

- top-SVD backbone 多为 broad visual-scene / attribute / count / action / spatial fingerprints；
- L24 tail `257-1024` 更 object-heavy，与 tail ablation 故事一致；
- L32 local TN rescue directions 更像 relational/contextual 或 yes-no arbitration；
- 单个 semantic direction 不是强 detector，最强 single-direction FP/TN AUC 约 0.562。

Baseline artifacts:

- `outputs/stage_s_baselines/detection_baseline_comparison.csv`
- `outputs/stage_s_baselines/mitigation_baseline_comparison.csv`
- `notes/baseline_positioning.md`

baseline 结论：

- Stage M first-token subset 上 yes/no margin baseline AUROC = 1.000，entropy AUROC = 0.884；
- paired full difference 5-seed AUROC = 0.721；
- 因此不能把本文写成纯 detection leaderboard win；
- 应强调 paired differencing 的机制解释力，而不是简单追求 AUROC 第一。

Final framing:

- `notes/final_paper_framing_decision.md` 已明确：mechanistic analysis paper。

---

## 5. Results by Claim

| Claim | Current status | Evidence |
| --- | --- | --- |
| `D = z_blind - z_img` 存在稳定几何结构 | 支持，但需限定 | SVD 非高斯、稳定；但 shuffle spectrum 也类似 |
| top variance directions 是 grounding subspace | 不支持 | top-4 explained variance 高但 AUROC 弱；shuffle 保留 spectrum |
| hallucination signal 与 difference geometry 相关 | 支持 | full difference 5-seed AUROC 0.721；PLS 0.720 |
| hallucination signal 是低维 top-K compact | 不支持或较弱 | K=4 弱；K=128/256/full 才明显 |
| matched evidence 与 mismatch geometry 不同 | 支持 | Stage B tail gap、supervised score gap、condition subspace divergence |
| residual/tail coordinates 有因果相关性 | 支持，尤其 L24 TN | tail ablation dose curve 可将 TN `No` 翻为 `Yes` |
| FP rescue 是可靠 mitigation | 不支持 | rescue 只影响 borderline cases；random/global/local 控制接近 |
| 结果可转移到 AMBER | modest 支持 | full AMBER top rows 约 0.63-0.665 AUROC |
| 结果可跨模型复现 | checkpoint-level 支持 | LLaVA-1.5-13B 复现主要 qualitative pattern |
| 可解释语义方向存在 | 部分支持 | semantic fingerprints coherent，但 single direction detector 弱 |

---

## 6. Paper-Level Interpretation

### 6.1 最推荐的故事线

论文可以围绕三层 geometry 展开：

1. **Dominant image-induced correction backbone**  
   blind-image difference 有明显高方差主干，捕获图像输入引发的大尺度视觉/语义移动。但这个主干不是 hallucination decision geometry。

2. **Hallucination-sensitive residual / evidence correction coordinates**  
   FP/TN 信号主要分布在 mid/high-dimensional difference space、full SVD coordinates、PLS/Fisher subspaces 和 tail bands 中。matched evidence 与 mismatch 的差异在这些 residual/tail 或 supervised views 中更明显。

3. **Boundary-local late arbitration / rescue geometry**  
   L24 tail ablation 影响 TN first-token faithfulness；L32 local TN-conditioned directions 可以推动少数 borderline FP toward `No`，但效果窄、弱、控制竞争强。

### 6.2 推荐论文 claim

可以写：

- Blind-reference differencing reveals hallucination-relevant correction geometry in LVLMs.
- Dominant variance directions are not the decision geometry.
- Evidence-sensitive residual/tail coordinates distinguish matched visual evidence from mismatched evidence.
- L24 residual/tail correction coordinates are causally relevant for correct negative decisions.
- The geometry has partial semantic fingerprints and modest transfer beyond POPE.

不要写：

- We find a universal visual grounding subspace.
- Top singular directions are the hallucination directions.
- The method is state-of-the-art hallucination detection.
- The intervention is a reliable hallucination mitigation method.
- The mechanism is universal across LVLM architectures.

---

## 7. Limitations

1. **Dominant low-rank spectrum is not pair-specific enough**  
   image/blind shuffle controls preserve similar singular spectra, so spectrum alone不能作为 grounding 证据。

2. **Plain top-SVD is not uniquely discriminative**  
   random subspaces、blind-side representations 和 full difference 都有强信号。

3. **Detection baselines can be stronger when logits are available**  
   Stage M first-token margin baseline 在小 subset 上 AUROC = 1.000。

4. **FP rescue 仍然弱**  
   当前 steering 主要救 borderline FP；high-margin hallucination 基本不动。

5. **External transfer modest**  
   AMBER full set 有 above-chance transfer，但峰值约 0.665，不是强外部 detector。

6. **Cross-model replication 仍在同一模型家族内**  
   LLaVA-1.5 7B/13B 支持 checkpoint-level recurrence，但不支持跨架构 generality。

7. **Semantic interpretation 是 fingerprint，不是 circuit proof**  
   vocabulary projection 和 sample extremes 有帮助，但不能证明完整机制回路。

---

## 8. 下一步建议

如果目标是尽快写论文：

- 直接以 mechanism paper 写作，使用 Stage Q 已生成 figures/tables；
- 主标题采用 layered correction geometry；
- 把 intervention 放在 causal probe / boundary-local evidence，而不是 method section 的主打 mitigation；
- 把 AMBER 和 13B 放在 external / replication section，语气保持 modest。

如果目标是冲更强的 venue 或把方法做硬：

- 跑一个不同架构 LVLM，例如 Qwen2-VL / InternVL / LLaVA-NeXT；
- 补 VCD/ICD 或 CLIP/image-text similarity baseline；
- 将 Stage L evidence-specific directions 接入 intervention；
- 测 compute overhead；
- 尝试 representation editing direction bank 的 GPU evaluation，当前 artifact 已在 `outputs/representation_editing_prep/`。

---

## 9. Key Artifact Index

主记录：

- `notes/findings.md`
- `notes/experiment_log.md`
- `notes/final_paper_framing_decision.md`
- `notes/naming_decision.md`
- `notes/baseline_positioning.md`
- `notes/cross_model_replication.md`

主表格和图：

- `outputs/paper_tables/stage_q_asset_index.md`
- `outputs/paper_tables/table1_pope_summary.csv`
- `outputs/paper_tables/table2_feature_comparison.csv`
- `outputs/paper_tables/table3_controls.csv`
- `outputs/paper_tables/table4_intervention.csv`
- `outputs/paper_figures/fig1_method_overview.pdf`
- `outputs/paper_figures/fig2_variance_vs_auroc.pdf`
- `outputs/paper_figures/fig3_condition_geometry.pdf`
- `outputs/paper_figures/fig4_intervention_dose.pdf`
- `outputs/paper_figures/fig5_layered_geometry.pdf`

关键实验结果：

- `outputs/svd/`
- `outputs/stage_b/`
- `outputs/stage_c_deep/`
- `outputs/stage_c_supervised/`
- `outputs/stage_j_controls/`
- `outputs/stage_k_positions/`
- `outputs/stage_l_evidence_subspace/`
- `outputs/interventions/`
- `outputs/stage_m_local_rescue/`
- `outputs/stage_n_external_full/`
- `outputs/stage_o_cross_model/llava_13b/`
- `outputs/stage_p_stats/`
- `outputs/stage_r_semantics/`
- `outputs/stage_s_baselines/`

---

## 10. One-Paragraph Abstract Draft

We study hallucination in LLaVA through blind-reference differencing, comparing hidden states from image-conditioned and text-only inputs on paired visual question-answering samples. Across POPE, the resulting difference geometry is stable and highly structured, but its dominant variance directions are not the most hallucination-discriminative coordinates. Instead, FP/TN signal is distributed across full difference representations, evidence-specific supervised subspaces, and residual/tail coordinates. Matched visual evidence separates from random and adversarial mismatches most clearly in residual/tail and supervised decision views, while ablation of L24 tail coordinates causally disrupts correct negative answers. Rescue experiments show only boundary-local steerability rather than reliable mitigation. External AMBER transfer and LLaVA-1.5-13B replication support the geometry as a recurring but not universal pattern. These results motivate a layered correction-geometry view of LVLM hallucination, rather than a single universal visual grounding subspace.

