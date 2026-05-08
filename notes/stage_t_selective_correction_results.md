# Stage T Selective Correction Results

## What Was Added

Stage T implements the Plan_extend selective-correction loop:

- train correction-geometry risk scores without using the held-out test split;
- calibrate thresholds on a separate calibration split;
- deploy gates only on model-predicted `Yes` samples;
- report FP capture and TP damage together;
- compare against same-trigger-count random gates;
- generate verification prompt plans for the gated samples.

Main entry points:

```bash
bash scripts/run_cpu_stage_t_selective_correction.sh
bash scripts/run_cpu_stage_t_selective_correction_fixed_ids.sh
bash scripts/run_gpu_stage_t_verification.sh
bash scripts/run_gpu_stage_t_prompt_sweep.sh
bash scripts/run_gpu_stage_t_vcd_sweep.sh
bash scripts/run_gpu_stage_t_margin_gates.sh
bash scripts/run_gpu_stage_t_strict_vcd_sweep.sh
bash scripts/run_gpu_stage_t_amber_vcd_sweep.sh
python scripts/build_stage_t_operator_upper_bound.py --stage-t-dir outputs/stage_t_selective_correction_fixed_ids
python scripts/build_stage_t_external_warning.py --stage-t-dir outputs/stage_t_selective_correction_fixed_ids
python scripts/bootstrap_stage_t_vcd_results.py --operator icd_blind
```

## Strict Subset-Transfer Protocol

Path: `outputs/stage_t_selective_correction/`

Protocol:

```text
Probe/SVD/PLS train: POPE random
Gate calibration:    POPE popular
Held-out test:       POPE adversarial
```

At 20% predicted-Yes target trigger rate on adversarial:

| Score | Triggered FP Ratio | FP Recall | TP Damage |
| --- | ---: | ---: | ---: |
| random64_probe | 0.202 | 0.296 | 0.184 |
| tail_257_1024_energy | 0.174 | 0.259 | 0.193 |
| pls32_probe | 0.165 | 0.238 | 0.189 |
| full_probe | 0.148 | 0.217 | 0.196 |
| top_4_probe | 0.127 | 0.185 | 0.200 |

Interpretation: the strict random-to-adversarial transfer setting is hard. Tail
energy and PLS beat same-trigger random routing, but the random subspace probe is
also strong here, so this should be framed as a stress test rather than the main
positive method result.

## Repository Fixed-Split Protocol

Path: `outputs/stage_t_selective_correction_fixed_ids/`

Protocol:

```text
Probe/SVD/PLS train: outputs/splits/pope_train_ids.json
Gate calibration:    outputs/splits/pope_val_ids.json
Held-out test:       outputs/splits/pope_test_ids.json
```

At 20% predicted-Yes target trigger rate:

| Score | Triggered FP Ratio | FP Recall | TP Damage |
| --- | ---: | ---: | ---: |
| pls32_probe | 0.226 | 0.396 | 0.133 |
| tail_257_1024_probe | 0.216 | 0.472 | 0.168 |
| full_probe | 0.207 | 0.453 | 0.169 |
| random64_probe | 0.164 | 0.396 | 0.197 |
| top_64_probe | 0.125 | 0.302 | 0.206 |
| top_4_probe | 0.025 | 0.057 | 0.215 |

Same-trigger-count random gates have about `0.089` triggered FP ratio at this
rate. This is the cleanest positive result: PLS, tail, and full-difference gates
select substantially more FPs than random routing, while top-4 remains weak.

## External AMBER Check

Using fixed-split POPE-trained scores on AMBER predicted-Yes samples:

| Score | AUROC | AUPRC |
| --- | ---: | ---: |
| tail_257_1024_energy | 0.595 | 0.422 |
| pls32_probe | 0.544 | 0.299 |
| full_probe | 0.530 | 0.292 |
| top_4_probe | 0.376 | 0.221 |

This supports a modest transfer claim, with tail energy strongest externally.

## Verification Plan

Generated plans:

- `outputs/stage_t_selective_correction/stage_t_verification_samples.jsonl`
- `outputs/stage_t_selective_correction/stage_t_verification_pool.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_samples.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_pool.jsonl`

The `samples` file is the union of currently gated samples. The `pool` file is
all held-out predicted-Yes samples, useful if exact random-gated verification
controls should be evaluated from the same second-pass generations.

## Actual Strict Verification Result

User-run output:

- `outputs/stage_t_selective_correction/stage_t_verification_predictions.jsonl`
- `outputs/stage_t_selective_correction/stage_t_actual_verification_metrics.csv`

The strict subset-transfer verification run covered `1152` gated adversarial
samples. The verification prompt rarely changed the model answer:

| Original outcome | Verification Yes | Verification No |
| --- | ---: | ---: |
| TP | 974 | 15 |
| FP | 147 | 16 |

So the gate can select riskier samples, but the current verification prompt is
not a strong correction operator. The best actual FP reduction in this run is
small:

| Score | Target trigger rate | Actual FP reduction | TP preserved | Accuracy after |
| --- | ---: | ---: | ---: | ---: |
| random64_probe | 0.30 | 0.042 | 0.996 | 0.839 |
| random64_probe | 0.20 | 0.032 | 0.998 | 0.839 |
| pls32_probe | 0.30 | 0.032 | 0.997 | 0.838 |
| full_probe | 0.30 | 0.021 | 0.998 | 0.838 |
| full_probe | 0.20 | 0.016 | 0.998 | 0.838 |

Interpretation: this is a useful negative result for the correction operator.
Stage T currently supports the gate/risk-signal claim more strongly than the
prompt-based mitigation claim. The next correction experiment should either run
the fixed-split gated samples or replace the verification prompt with a stronger
operator such as VCD/ICD.

To run fixed-split verification from a CUDA-visible shell:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
TEST_SUBSET=test \
SPLIT_DIR=outputs/splits \
bash scripts/run_gpu_stage_t_verification.sh
```

## Actual Fixed-Split Verification Result

User-run output:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_predictions.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_actual_verification_metrics.csv`

The fixed-split verification run covered `513` gated held-out predicted-Yes
samples. The prompt still changes only a small number of answers:

| Original outcome | Verification Yes | Verification No |
| --- | ---: | ---: |
| TP | 460 | 7 |
| FP | 37 | 9 |

However, because the fixed-split gate selects cleaner FP-enriched sets, the
actual correction result is more useful than the strict subset-transfer run.
At 30% predicted-Yes trigger rate:

| Score | Actual FP reduction | TP preserved | Accuracy after | F1 after |
| --- | ---: | ---: | ---: | ---: |
| pls32_probe | 0.132 | 0.994 | 0.866 | 0.856 |
| tail_257_1024_probe | 0.132 | 0.994 | 0.866 | 0.856 |
| full_probe | 0.113 | 0.994 | 0.865 | 0.856 |
| tail_257_1024_energy | 0.075 | 0.994 | 0.864 | 0.854 |
| top_64_probe | 0.075 | 0.991 | 0.862 | 0.853 |
| random64_probe | 0.057 | 0.993 | 0.862 | 0.853 |
| top_4_probe | 0.000 | 1.000 | 0.863 | 0.854 |

Interpretation: the actual prompt-based correction is still far weaker than the
oracle gate upper bound, but the fixed-split result preserves the main Stage T
story: PLS/tail/full-difference gates are more useful correction routers than
top-variance or random subspace gates. This is the best current evidence for
writing selective verification as a modest utility result rather than only a
risk-score analysis.

## Strong Prompt Sweep

The verification runner now supports three stronger prompt variants:

| Variant | Intent |
| --- | --- |
| `forced_evidence` | Ask whether the queried object is visibly present and forbid prior-likelihood reasoning. |
| `conservative` | Answer Yes only under clear visual support; uncertain or partial evidence maps to No. |
| `internal_rationale` | Ask the model to internally verify direct visibility, but output only Yes/No. |

Recommended fixed-split command:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
TEST_SUBSET=test \
SPLIT_DIR=outputs/splits \
bash scripts/run_gpu_stage_t_prompt_sweep.sh
```

Each variant writes separate predictions and metrics, for example:

- `stage_t_verification_predictions_forced_evidence.jsonl`
- `stage_t_actual_verification_metrics_forced_evidence.csv`

The prompt sweep also rebuilds:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_operator_upper_bound_gap.csv`

This table joins gate potential and actual operator performance:

```text
Gate score
Triggered FP Ratio
Oracle FP Reduction
Actual FP Reduction
TP Damage / TP Preserved
Operator Realization Ratio
```

Use it to state the current tradeoff precisely: geometry gates expose useful
correction opportunities, while the prompt operator determines how much of that
upper bound is actually realized.

Actual fixed-split prompt-sweep outputs:

- `stage_t_verification_predictions_forced_evidence.jsonl`
- `stage_t_verification_predictions_conservative.jsonl`
- `stage_t_verification_predictions_internal_rationale.jsonl`
- `stage_t_actual_verification_metrics_forced_evidence.csv`
- `stage_t_actual_verification_metrics_conservative.csv`
- `stage_t_actual_verification_metrics_internal_rationale.csv`

Across the same `513` fixed-split gated samples, the stronger prompts did not
improve the correction operator. They made the model even more likely to repeat
`Yes`:

| Prompt variant | TP -> Yes | TP -> No | FP -> Yes | FP -> No |
| --- | ---: | ---: | ---: | ---: |
| `legacy` | 460 | 7 | 37 | 9 |
| `forced_evidence` | 465 | 2 | 43 | 3 |
| `conservative` | 466 | 1 | 44 | 2 |
| `internal_rationale` | 464 | 3 | 40 | 6 |

At 30% predicted-Yes trigger rate, the operator-upper-bound table shows the
same pattern:

| Prompt | Gate | Triggered FP Ratio | Oracle FP Reduction | Actual FP Reduction | TP Preserved |
| --- | --- | ---: | ---: | ---: | ---: |
| `legacy` | `pls32_probe` | 0.189 | 0.604 | 0.132 | 0.994 |
| `legacy` | `tail_257_1024_probe` | 0.183 | 0.566 | 0.132 | 0.994 |
| `legacy` | `full_probe` | 0.167 | 0.491 | 0.113 | 0.994 |
| `internal_rationale` | `pls32_probe` | 0.189 | 0.604 | 0.094 | 0.996 |
| `internal_rationale` | `tail_257_1024_probe` | 0.183 | 0.566 | 0.094 | 0.996 |
| `forced_evidence` | `tail_257_1024_probe` | 0.183 | 0.566 | 0.057 | 0.998 |
| `conservative` | `tail_257_1024_probe` | 0.183 | 0.566 | 0.038 | 0.998 |

Interpretation: prompt wording alone is not enough here. `internal_rationale` is
the best of the stronger variants, but it is still below the original legacy
verification prompt. This strengthens the operator-gap story: PLS/tail/full
gates identify a high-potential intervention set, but the current second-pass
prompt realizes only a small fraction of the oracle correction opportunity.

Next decision: move to the Step 3 gated VCD/ICD experiment on the fixed-split
test set. The routing comparison should keep the same Stage T gates and swap in
a stronger decoding-time operator:

| Method | Routing |
| --- | --- |
| Original | no second-pass decoding |
| Always VCD/ICD | all predicted-Yes samples |
| Random-gated VCD/ICD | same trigger counts as geometry gates |
| Geometry-gated VCD/ICD | `pls32_probe`, `tail_257_1024_probe`, `full_probe` |
| Top-4-gated VCD/ICD | `top_4_probe` |
| Margin-gated VCD/ICD | output-confidence gate |

## Selective Warning / Abstention

Because prompt-based answer correction is weak, the gate can already be used as
a deployment-time warning/abstention router: for gated predicted-`Yes` samples,
emit that the answer is visually unsupported or uncertain instead of forcing the
model to flip to `No`.

Generated outputs:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_selective_warning_metrics.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_selective_warning_metrics.md`

Command:

```bash
python scripts/build_stage_t_selective_warning.py \
  --stage-t-dir outputs/stage_t_selective_correction_fixed_ids
```

After adding LLaVA first-token margin scores, warning has two useful operating
points. Pure high-margin gating is the wrong direction for predicted-`Yes`
risk routing: it selects very confident `Yes` answers and captures `0` FPs on
the held-out test set. Low-margin gating, especially low-margin+geometry,
is the strongest warning baseline.

| Target | Method | Trigger rate | FP captured | FP recall | TP damage | Warning precision | Compute saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | Low-margin+FullD warning | 0.180 | 35 | 0.660 | 0.133 | 0.327 | 0.820 |
| 0.20 | Low-margin+PLS warning | 0.176 | 34 | 0.642 | 0.131 | 0.324 | 0.824 |
| 0.20 | Low-margin warning | 0.223 | 39 | 0.736 | 0.173 | 0.293 | 0.777 |
| 0.20 | PLS warning | 0.156 | 21 | 0.396 | 0.133 | 0.226 | 0.844 |
| 0.20 | Random warning | 0.156-0.223 | 8-12 | 0.155-0.233 | 0.156-0.222 | 0.088-0.093 | 0.777-0.844 |
| 0.20 | Margin warning | 0.196 | 0 | 0.000 | 0.215 | 0.000 | 0.804 |
| 0.30 | Low-margin+PLS warning | 0.273 | 43 | 0.811 | 0.221 | 0.264 | 0.727 |
| 0.30 | Low-margin+FullD warning | 0.290 | 42 | 0.792 | 0.241 | 0.243 | 0.710 |
| 0.30 | Low-margin warning | 0.352 | 46 | 0.868 | 0.302 | 0.219 | 0.648 |
| 0.30 | PLS warning | 0.284 | 32 | 0.604 | 0.252 | 0.189 | 0.716 |
| 0.30 | Random warning | 0.262-0.352 | 14-18 | 0.255-0.349 | 0.262-0.353 | 0.087-0.090 | 0.648-0.738 |
| 0.30 | Margin warning | 0.262 | 0 | 0.000 | 0.287 | 0.000 | 0.738 |

Interpretation: selective warning is currently the most stable deployment story.
The base predicted-`Yes` FP prevalence is only about `0.089`; low-margin+geometry
raises warning precision to `0.232-0.327` while still using only `17-30%` of
always-on warning compute. This directly answers the confidence-margin baseline:
confidence is useful, but the useful direction is low margin, and combining it
with correction geometry improves precision and TP damage relative to pure
low-margin warning.

## Gated VCD / ICD Scaffold

The repository now has a minimal fixed-split gated VCD/ICD path:

- `src/vgs/vcd.py`
- `scripts/run_stage_t_vcd_eval.py`
- `scripts/analyze_stage_t_vcd_results.py`
- `scripts/run_gpu_stage_t_vcd.sh`

The operator supports:

| Operator | Reference distribution |
| --- | --- |
| `vcd_diffusion` | diffusion-noised preprocessed image tensor |
| `vcd_blur` | Gaussian-blurred image ablation |
| `vcd_gray` | grayscale image |
| `icd_blind` | text-only blind prompt |

Recommended canonical first run from a CUDA-visible shell:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
TEST_SUBSET=test \
SPLIT_DIR=outputs/splits \
VCD_OPERATOR=vcd_diffusion \
VCD_DECODE_STRATEGY=sample \
VCD_NOISE_STEP=500 \
TARGET_RATES="0.2 0.3" \
bash scripts/run_gpu_stage_t_vcd.sh
```

To sweep the canonical diffusion VCD plus two ablations:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
TEST_SUBSET=test \
SPLIT_DIR=outputs/splits \
bash scripts/run_gpu_stage_t_vcd_sweep.sh
```

This runs VCD/ICD once on the held-out predicted-`Yes` pool and then evaluates
the same decoded outputs under:

- Original
- Always VCD/ICD
- Random-gated VCD/ICD
- Top-4-gated VCD/ICD
- PLS/Tail/FullD-gated VCD/ICD
- Margin and Margin+Geometry gates when margin scores are available

Metrics written by `analyze_stage_t_vcd_results.py`:

- FP reduction
- TP preserved
- FP reduction per trigger
- Accuracy / F1
- extra compute fraction versus always-on VCD/ICD
- compute saved versus always-on VCD/ICD
- gap to always-on VCD/ICD FP reduction

## Actual Blur-VCD Ablation Result

User-run output:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_vcd_blur.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_metrics_vcd_blur.csv`

Run configuration:

```text
Operator:     vcd_blur
Alpha:        1.0
Beta:         0.1
Blur radius:  5.0
Pool:         fixed-split held-out predicted-Yes samples
Pool size:    596
```

The blur-VCD operator changed only a small number of predicted-`Yes` answers:

| Original outcome | VCD Yes | VCD No |
| --- | ---: | ---: |
| TP | 527 | 16 |
| FP | 45 | 8 |

Always-on blur VCD reduces `8/53` FPs, but it also damages `16/543` TPs:

| Method | Trigger rate | FP reduction | TP preserved | FP reduction / trigger | Accuracy after | F1 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.000 | 0.000 | 1.000 | - | 0.863 | 0.854 |
| Always VCD | 1.000 | 0.151 | 0.971 | 0.013 | 0.857 | 0.845 |

At selective trigger rates, the best result is `tail_257_1024_probe` at the
30% target. It captures `5/8` of the FPs that always-on VCD can fix while using
only `27.5%` of the predicted-Yes VCD compute:

| Target | Method | Trigger rate | FP reduced | FP reduction | TP preserved | FP reduction / trigger | Compute saved | Gap to Always |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | PLS-gated VCD | 0.156 | 2 | 0.038 | 0.993 | 0.022 | 0.844 | 0.113 |
| 0.20 | Tail-gated VCD | 0.195 | 2 | 0.038 | 0.991 | 0.017 | 0.805 | 0.113 |
| 0.20 | FullD-gated VCD | 0.195 | 2 | 0.038 | 0.994 | 0.017 | 0.805 | 0.113 |
| 0.20 | Top-4-gated VCD | 0.201 | 0 | 0.000 | 0.996 | 0.000 | 0.799 | 0.151 |
| 0.30 | Tail-gated VCD | 0.275 | 5 | 0.094 | 0.989 | 0.030 | 0.725 | 0.057 |
| 0.30 | PLS-gated VCD | 0.284 | 3 | 0.057 | 0.989 | 0.018 | 0.716 | 0.094 |
| 0.30 | FullD-gated VCD | 0.262 | 3 | 0.057 | 0.987 | 0.019 | 0.738 | 0.094 |
| 0.30 | Top-4-gated VCD | 0.289 | 1 | 0.019 | 0.993 | 0.006 | 0.711 | 0.132 |

Same-trigger random-gated VCD is weaker. At the 30% target, random gates reduce
only about `0.039-0.043` of FPs with FP reduction per trigger around `0.013`,
while tail-gated VCD reaches `0.094` FP reduction and `0.030` FP reduction per
trigger. The tail gate therefore gets about `2.27x` the always-on FP reduction
per trigger and closes `62.5%` of always-on VCD's FP-reduction opportunity with
`27.5%` of the compute.

Implementation audit: this run should be treated as a blur ablation, not the
canonical VCD result. The official VCD setup contrasts the original image
against diffusion-noised image tensors and samples from the contrastive
distribution. The initial Stage T implementation used a raw-image Gaussian blur
proxy and greedy argmax decoding. The contrastive formula was directionally
correct, but the perturbation and decoding mode were not the canonical VCD
configuration.

Interpretation: `vcd_blur` is a weak correction operator and always-on blur-VCD
hurts overall accuracy/F1 in this fixed-split run. However, the routing claim
survives: geometry gates, especially the tail probe, concentrate the limited
blur-VCD corrections better than random/top-variance gates and preserve more
TPs than always-on blur-VCD. The next result to report as VCD should use
`VCD_OPERATOR=vcd_diffusion` with `VCD_DECODE_STRATEGY=sample`.

## Actual VCD / ICD Sweep Result

User-run outputs:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_vcd_diffusion.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_metrics_vcd_diffusion.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_vcd_gray.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_metrics_vcd_gray.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_icd_blind.jsonl`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_metrics_icd_blind.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_operator_comparison.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_operator_comparison.md`

Command used by the sweep:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
TEST_SUBSET=test \
SPLIT_DIR=outputs/splits \
bash scripts/run_gpu_stage_t_vcd_sweep.sh
```

Prediction-change matrix on the `596` held-out predicted-`Yes` pool:

| Operator | TP -> Yes | TP -> No | FP -> Yes | FP -> No |
| --- | ---: | ---: | ---: | ---: |
| `vcd_diffusion` | 495 | 48 | 37 | 16 |
| `vcd_gray` | 483 | 60 | 39 | 14 |
| `icd_blind` | 495 | 48 | 35 | 18 |
| `vcd_blur` | 527 | 16 | 45 | 8 |

Always-on operators reduce more FPs than blur, but they are too aggressive and
hurt overall metrics:

| Operator | Always FP reduction | Always TP preserved | Accuracy after | F1 after | Accuracy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `icd_blind` | 0.340 | 0.912 | 0.841 | 0.822 | -0.022 |
| `vcd_diffusion` | 0.302 | 0.912 | 0.839 | 0.820 | -0.024 |
| `vcd_gray` | 0.264 | 0.890 | 0.829 | 0.807 | -0.034 |
| `vcd_blur` | 0.151 | 0.971 | 0.857 | 0.845 | -0.006 |

Selective routing fixes this tradeoff, but the best row depends on the
deployment objective. After adding margin gates, low-margin+geometry maximizes
FP reduction and FP reduction per trigger, while geometry-only and
high-margin+geometry rows preserve TPs and overall accuracy better.

| Operator | Useful row | Target | Trigger rate | FP reduction | TP preserved | FP reduction / trigger | Accuracy after | F1 after | Accuracy delta | Compute saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `icd_blind` | Low-margin+Tail-gated VCD/ICD | 0.30 | 0.297 | 0.340 | 0.937 | 0.102 | 0.851 | 0.835 | -0.012 | 0.703 |
| `icd_blind` | Low-margin+PLS-gated VCD/ICD | 0.20 | 0.176 | 0.321 | 0.965 | 0.162 | 0.861 | 0.849 | -0.001 | 0.824 |
| `icd_blind` | PLS-gated VCD/ICD | 0.30 | 0.284 | 0.283 | 0.976 | 0.089 | 0.864 | 0.853 | +0.001 | 0.716 |
| `icd_blind` | FullD-gated VCD/ICD | 0.30 | 0.262 | 0.245 | 0.985 | 0.083 | 0.867 | 0.856 | +0.004 | 0.738 |
| `icd_blind` | Margin+FullD-gated VCD/ICD | 0.30 | 0.267 | 0.170 | 0.994 | 0.057 | 0.867 | 0.858 | +0.004 | 0.733 |
| `vcd_diffusion` | Low-margin+PLS-gated VCD/ICD | 0.30 | 0.273 | 0.283 | 0.948 | 0.092 | 0.853 | 0.839 | -0.010 | 0.727 |
| `vcd_diffusion` | Low-margin+FullD-gated VCD/ICD | 0.20 | 0.180 | 0.264 | 0.969 | 0.131 | 0.861 | 0.848 | -0.002 | 0.820 |
| `vcd_diffusion` | PLS-gated VCD/ICD | 0.30 | 0.284 | 0.226 | 0.976 | 0.071 | 0.862 | 0.851 | -0.001 | 0.716 |
| `vcd_diffusion` | Margin+Tail-gated VCD/ICD | 0.30 | 0.268 | 0.151 | 0.994 | 0.050 | 0.867 | 0.857 | +0.004 | 0.732 |
| `vcd_gray` | Low-margin+PLS-gated VCD/ICD | 0.30 | 0.273 | 0.245 | 0.930 | 0.080 | 0.844 | 0.828 | -0.019 | 0.727 |

Random-gated VCD/ICD is much weaker. At the 30% target, the best random rows
reach only:

| Operator | Best random FP reduction | Best random TP preserved | Best random FP reduction / trigger |
| --- | ---: | ---: | ---: |
| `icd_blind` | 0.120 | 0.969 | 0.030 |
| `vcd_diffusion` | 0.109 | 0.969 | 0.027 |
| `vcd_gray` | 0.096 | 0.961 | 0.024 |
| `vcd_blur` | 0.053 | 0.990 | 0.013 |

Interpretation: this is the strongest selective-correction result so far.
Always-on VCD/ICD is too damaging, but selective routing converts the same
operator into useful deployment choices. If the goal is maximum hallucinated-Yes
suppression, `low_margin+geometry` reaches the highest FP reduction per trigger
and can match the always-on `icd_blind` FP reduction while saving `70.3%` of
predicted-Yes operator compute. If the goal is balanced utility, geometry-only
and high-margin+geometry are safer: `icd_blind + full_probe` preserves `98.5%`
of TPs and slightly improves accuracy/F1, while `icd_blind + margin_plus_full`
preserves `99.4%` of TPs with the best accuracy/F1 row.

## Bootstrap Confidence Intervals

Bootstrap outputs:

- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_icd_blind.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_icd_blind.md`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_diffusion.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_diffusion.md`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_gray.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_gray.md`

Command pattern:

```bash
python scripts/bootstrap_stage_t_vcd_results.py \
  --operator icd_blind \
  --vcd-predictions outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_icd_blind.jsonl \
  --n-bootstrap 2000 \
  --output-dir outputs/stage_t_selective_correction_fixed_ids
```

Key 95% CI rows:

| Operator / gate | Target | Metric | Point | 95% CI |
| --- | ---: | --- | ---: | ---: |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | FP reduction | 0.321 | [0.196, 0.452] |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | TP preserved | 0.965 | [0.949, 0.980] |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | Accuracy delta | -0.001 | [-0.010, 0.007] |
| `icd_blind + low_margin_plus_pls32_probe` | 0.20 | FP reduction / trigger | 0.162 | [0.093, 0.236] |
| `icd_blind + low_margin_plus_tail_257_1024_probe` | 0.30 | FP reduction | 0.340 | [0.211, 0.472] |
| `icd_blind + low_margin_plus_tail_257_1024_probe` | 0.30 | TP preserved | 0.937 | [0.917, 0.956] |
| `icd_blind + low_margin_plus_tail_257_1024_probe` | 0.30 | Accuracy delta | -0.012 | [-0.022, -0.001] |
| `icd_blind + full_probe` | 0.30 | FP reduction | 0.245 | [0.130, 0.370] |
| `icd_blind + full_probe` | 0.30 | TP preserved | 0.985 | [0.974, 0.994] |
| `icd_blind + full_probe` | 0.30 | Accuracy delta | 0.004 | [-0.003, 0.010] |
| `icd_blind + full_probe` | 0.30 | FP reduction / trigger | 0.083 | [0.040, 0.130] |
| `icd_blind + margin_plus_full_probe` | 0.30 | FP reduction | 0.170 | [0.078, 0.279] |
| `icd_blind + margin_plus_full_probe` | 0.30 | TP preserved | 0.994 | [0.987, 1.000] |
| `icd_blind + margin_plus_full_probe` | 0.30 | Accuracy delta | 0.004 | [0.000, 0.010] |
| `vcd_diffusion + low_margin_plus_pls32_probe` | 0.30 | FP reduction | 0.283 | [0.163, 0.408] |
| `vcd_diffusion + low_margin_plus_pls32_probe` | 0.30 | TP preserved | 0.948 | [0.930, 0.965] |
| `vcd_diffusion + low_margin_plus_full_probe` | 0.20 | FP reduction / trigger | 0.131 | [0.067, 0.200] |

Interpretation: do not overclaim the small accuracy gain. Most balanced
geometry-only accuracy CIs cross zero, and low-margin VCD rows often have
negative accuracy deltas despite strong FP reduction. The robust claim is a
tradeoff: low-margin+geometry is best for FP capture and warning, while
geometry-only or high-margin+geometry is best for TP-preserving selective
correction.

## Margin-Gate Follow-Up

Margin follow-up is complete on the fixed-split POPE test set.

Generated outputs:

- `outputs/margins/pope_margin_scores.csv`
- `outputs/margins/margin_baseline_metrics.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_gate_metrics.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_selective_warning_metrics.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_operator_comparison.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_icd_blind.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_diffusion.csv`
- `outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_bootstrap_ci_vcd_gray.csv`

Reproduction command:

```bash
MODEL_PATH=/data/lh/ModelandDataset/llava-1.5-7b-hf \
MODEL_FAMILY=llava \
TORCH_DTYPE=float16 \
STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
SPLIT_DIR=outputs/splits \
TEST_SUBSET=test \
bash scripts/run_gpu_stage_t_margin_gates.sh
```

Important audit detail: the old `margin_probe` is high yes/no margin. It is
excellent for separating predicted-`Yes` from predicted-`No` in the raw
margin baseline, but it is not the right risk direction inside the predicted-`Yes`
pool. On held-out predicted-`Yes` samples it captures `0` FPs at both 20% and
30% trigger targets. The reviewer-relevant confidence baseline is therefore
`low_margin_probe`, which scores low `Yes-No` margin as risky.

At the gate/warning level, low-margin+geometry is the strongest result:

| Target | Gate | Triggered FP ratio | FP recall | TP damage |
| --- | --- | ---: | ---: | ---: |
| 0.20 | `low_margin_plus_full_probe` | 0.327 | 0.660 | 0.133 |
| 0.20 | `low_margin_plus_pls32_probe` | 0.324 | 0.642 | 0.131 |
| 0.20 | `low_margin_probe` | 0.293 | 0.736 | 0.173 |
| 0.20 | `pls32_probe` | 0.226 | 0.396 | 0.133 |
| 0.20 | `margin_probe` | 0.000 | 0.000 | 0.215 |
| 0.30 | `low_margin_plus_pls32_probe` | 0.264 | 0.811 | 0.221 |
| 0.30 | `low_margin_plus_full_probe` | 0.243 | 0.792 | 0.241 |
| 0.30 | `low_margin_probe` | 0.219 | 0.868 | 0.302 |
| 0.30 | `pls32_probe` | 0.189 | 0.604 | 0.252 |
| 0.30 | `margin_probe` | 0.000 | 0.000 | 0.287 |

At the VCD/ICD level, the low-margin gates reveal a sharper tradeoff:

- For FP reduction, `low_margin+geometry` is best. `icd_blind + low_margin_plus_tail`
  at 30% reaches `0.340` FP reduction, matching always-on `icd_blind` while
  saving `70.3%` compute, but TP preserved drops to `0.937`.
- For balanced correction utility, geometry-only and high-margin+geometry are
  safer. `icd_blind + full_probe` at 30% gives `0.245` FP reduction with `0.985`
  TP preserved, and `icd_blind + margin_plus_full` gives `0.170` FP reduction
  with `0.994` TP preserved and the best F1 row.

Recommended writeup framing:

- warning / abstention result: report `low_margin+geometry` as the strongest
  deployment-time risk router;
- correction result: report geometry-only or high-margin+geometry as the
  TP-preserving VCD/ICD router;
- limitation: strongest utility evidence is fixed-split held-out POPE; strict
  subset-transfer and AMBER stress tests are supportive but weaker.

## Stress-Test Results

Two follow-up stress tests are complete.

### Strict Subset-Transfer Gated VCD/ICD

Purpose: test whether the fixed Stage T routing story survives the harder
random-to-adversarial protocol.

Command:

```bash
STAGE_T_DIR=outputs/stage_t_selective_correction \
TEST_SUBSET=adversarial \
SPLIT_DIR= \
TARGET_RATES="0.2 0.3" \
VCD_OPERATORS="vcd_diffusion vcd_gray icd_blind" \
bash scripts/run_gpu_stage_t_strict_vcd_sweep.sh
```

This reuses:

- `outputs/stage_t_selective_correction/stage_t_verification_pool.jsonl`
- `outputs/stage_t_selective_correction/stage_t_verification_gate_assignments.csv`

and writes strict-protocol VCD/ICD outputs and comparison tables back to:

- `outputs/stage_t_selective_correction/stage_t_vcd_predictions_*.jsonl`
- `outputs/stage_t_selective_correction/stage_t_vcd_metrics_*.csv`
- `outputs/stage_t_selective_correction/stage_t_vcd_operator_comparison.csv`
- `outputs/stage_t_selective_correction/stage_t_vcd_bootstrap_ci_*.csv`

Prediction-change matrix on the `1391` adversarial predicted-`Yes` pool:

| Operator | TP -> Yes | TP -> No | FP -> Yes | FP -> No |
| --- | ---: | ---: | ---: | ---: |
| `icd_blind` | 1093 | 109 | 137 | 52 |
| `vcd_diffusion` | 1100 | 102 | 153 | 36 |
| `vcd_gray` | 1081 | 121 | 143 | 46 |

Always-on VCD/ICD still damages accuracy/F1 under strict transfer:

| Operator | Always FP reduction | TP preserved | Accuracy delta |
| --- | ---: | ---: | ---: |
| `icd_blind` | 0.275 | 0.909 | -0.019 |
| `vcd_gray` | 0.243 | 0.899 | -0.025 |
| `vcd_diffusion` | 0.190 | 0.915 | -0.022 |

Best selective rows:

| Operator | Best gate | Target | Trigger rate | FP reduction | TP preserved | FP / trigger | Accuracy delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `icd_blind` | `pls32_probe` | 0.30 | 0.300 | 0.116 | 0.969 | 0.053 | -0.005 |
| `icd_blind` | `pls32_probe` | 0.20 | 0.196 | 0.079 | 0.988 | 0.055 | 0.000 |
| `vcd_gray` | `tail_257_1024_energy` | 0.30 | 0.303 | 0.090 | 0.976 | 0.040 | -0.004 |
| `vcd_diffusion` | `tail_257_1024_energy` | 0.30 | 0.303 | 0.085 | 0.978 | 0.038 | -0.004 |

Bootstrap 95% CI highlights:

| Operator / gate | Target | Metric | Point | 95% CI |
| --- | ---: | --- | ---: | ---: |
| `icd_blind + pls32_probe` | 0.20 | FP reduction | 0.079 | [0.043, 0.121] |
| `icd_blind + pls32_probe` | 0.20 | TP preserved | 0.988 | [0.981, 0.993] |
| `icd_blind + pls32_probe` | 0.20 | Accuracy delta | 0.000 | [-0.004, 0.003] |
| `icd_blind + pls32_probe` | 0.30 | FP reduction | 0.116 | [0.074, 0.166] |
| `vcd_diffusion + tail_257_1024_energy` | 0.30 | FP reduction | 0.085 | [0.047, 0.127] |
| `vcd_diffusion + tail_257_1024_energy` | 0.30 | TP preserved | 0.978 | [0.969, 0.986] |

Interpretation: strict subset-transfer is a weaker stress test. Selective
gating still improves the always-on compute/TP tradeoff, but geometry gates are
only modestly above same-trigger random, and accuracy gains should not be
claimed. The useful statement is that selective routing remains less damaging
than always-on VCD/ICD under the hard transfer protocol.

### AMBER External Warning / Gated Operator

Purpose: test external transfer without refitting Stage T gates.

Generated warning-transfer outputs:

- `outputs/stage_t_external_amber_fixed_ids/stage_t_external_warning_metrics.csv`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_external_warning_metrics.md`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_external_gate_assignments_external_top_rate.csv`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_external_gate_assignments_pope_calibrated_threshold.csv`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_external_vcd_pool.jsonl`

AMBER warning results on `5022` predicted-`Yes` samples:

| Policy | Target | Best score | Trigger rate | FP recall | TP damage | Warning precision | Random precision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| POPE-calibrated threshold | 0.20 | `tail_257_1024_energy` | 0.007 | 0.023 | 0.000 | 1.000 | 0.293 |
| POPE-calibrated threshold | 0.30 | `tail_257_1024_energy` | 0.073 | 0.147 | 0.043 | 0.574 | 0.283 |
| External top-rate | 0.20 | `tail_257_1024_energy` | 0.200 | 0.285 | 0.166 | 0.405 | 0.284 |
| External top-rate | 0.30 | `tail_257_1024_energy` | 0.300 | 0.393 | 0.263 | 0.372 | 0.285 |
| External top-rate | 0.20 | `pls32_probe` | 0.200 | 0.211 | 0.196 | 0.300 | 0.286 |
| External top-rate | 0.20 | `full_probe` | 0.200 | 0.203 | 0.199 | 0.288 | 0.284 |

Interpretation: external warning transfer is modest but real for tail energy,
especially under same-trigger top-rate evaluation. Margin baselines are not
available on AMBER yet because no AMBER first-token margin dump exists.

Gated AMBER VCD/ICD command:

```bash
SOURCE_STAGE_T_DIR=outputs/stage_t_selective_correction_fixed_ids \
EXTERNAL_OUTPUT_DIR=outputs/stage_t_external_amber_fixed_ids \
EXTERNAL_PREDICTIONS=outputs/stage_n_external_full/amber_predictions.jsonl \
GATE_POLICY=external_top_rate \
TEST_SUBSET=discriminative \
TARGET_RATES="0.2 0.3" \
VCD_OPERATORS="vcd_diffusion vcd_gray icd_blind" \
bash scripts/run_gpu_stage_t_amber_vcd_sweep.sh
```

Use `GATE_POLICY=pope_calibrated_threshold` for the deployment-threshold
variant. The top-rate variant is better for a controlled ranking-transfer
stress test; the calibrated-threshold variant tests threshold transfer under
distribution shift.

Gated VCD/ICD outputs:

- `outputs/stage_t_external_amber_fixed_ids/stage_t_vcd_predictions_*.jsonl`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_vcd_metrics_*.csv`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_vcd_operator_comparison.csv`
- `outputs/stage_t_external_amber_fixed_ids/stage_t_vcd_bootstrap_ci_*.csv`

Prediction-change matrix on the `5022` AMBER predicted-`Yes` pool:

| Operator | TP -> Yes | TP -> No | FP -> Yes | FP -> No |
| --- | ---: | ---: | ---: | ---: |
| `icd_blind` | 2439 | 1157 | 855 | 571 |
| `vcd_diffusion` | 2764 | 832 | 1070 | 356 |
| `vcd_gray` | 2781 | 815 | 1021 | 405 |

Always-on operators reduce FPs but damage TPs heavily:

| Operator | Always FP reduction | TP preserved | Accuracy delta |
| --- | ---: | ---: | ---: |
| `icd_blind` | 0.400 | 0.678 | -0.041 |
| `vcd_gray` | 0.284 | 0.773 | -0.029 |
| `vcd_diffusion` | 0.250 | 0.769 | -0.033 |

Best external-top-rate gated rows:

| Operator | Gate | Target | Trigger rate | FP reduction | TP preserved | FP / trigger | Accuracy delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `icd_blind` | `tail_257_1024_energy` | 0.20 | 0.200 | 0.100 | 0.952 | 0.142 | -0.002 |
| `icd_blind` | `tail_257_1024_energy` | 0.30 | 0.300 | 0.142 | 0.925 | 0.135 | -0.005 |
| `vcd_gray` | `tail_257_1024_energy` | 0.20 | 0.200 | 0.069 | 0.961 | 0.099 | -0.003 |
| `vcd_gray` | `tail_257_1024_energy` | 0.30 | 0.300 | 0.103 | 0.943 | 0.098 | -0.004 |
| `vcd_diffusion` | `tail_257_1024_energy` | 0.20 | 0.200 | 0.055 | 0.961 | 0.079 | -0.004 |
| `vcd_diffusion` | `tail_257_1024_energy` | 0.30 | 0.300 | 0.082 | 0.942 | 0.078 | -0.006 |

Same-trigger random at 30% reaches only `0.121`, `0.086`, and `0.076` FP
reduction for `icd_blind`, `vcd_gray`, and `vcd_diffusion` respectively, so
tail-energy routing improves the FP-reduction-per-trigger tradeoff externally.

Bootstrap 95% CI highlights:

| Operator / gate | Target | Metric | Point | 95% CI |
| --- | ---: | --- | ---: | ---: |
| `icd_blind + tail_257_1024_energy` | 0.20 | FP reduction | 0.100 | [0.086, 0.116] |
| `icd_blind + tail_257_1024_energy` | 0.20 | TP preserved | 0.952 | [0.945, 0.959] |
| `icd_blind + tail_257_1024_energy` | 0.20 | Accuracy delta | -0.002 | [-0.004, 0.000] |
| `icd_blind + tail_257_1024_energy` | 0.30 | FP reduction | 0.142 | [0.124, 0.160] |
| `icd_blind + tail_257_1024_energy` | 0.30 | TP preserved | 0.925 | [0.917, 0.934] |
| `vcd_gray + tail_257_1024_energy` | 0.30 | FP reduction | 0.103 | [0.088, 0.120] |
| `vcd_diffusion + tail_257_1024_energy` | 0.30 | FP reduction | 0.082 | [0.068, 0.097] |

Interpretation: AMBER gives the cleanest external-transfer evidence, but for
tail energy rather than PLS/full. Warning transfer is strong; gated VCD/ICD is
useful mainly as a compute-saving mitigation with mild accuracy cost. Do not
claim AMBER correction improves overall accuracy.
