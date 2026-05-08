# InternVL Readout Audit

Date: 2026-05-07

## Problem

In the corrected cross-architecture run, Qwen2-VL / Qwen2.5-VL dropped to
moderate FP/TN geometry scores after switching from assistant-prompt readout to
`last_user_content_token`. InternVL2 / InternVL2.5 still showed near-perfect
FP/TN separability:

```text
InternVL2-8B:   raw_img / difference AUROC ~= 0.999
InternVL2.5-8B: raw_img / difference AUROC ~= 0.998
```

However, those same full-difference scores failed in the actual deployment
setting, FP vs TP among model-predicted `Yes` samples. This indicates the
InternVL FP/TN probe is likely reading an answer-preparation state rather than a
usable hallucination-risk signal.

## Audit Finding

For InternVL, the current `last_user_content_token` is the final token of the
fixed instruction:

```text
Answer with yes or no only.
```

Tokenizer-only audit:

```text
InternVL2-8B:   question_idx token `?`, user_idx token `.`
InternVL2.5-8B: question_idx token `?`, user_idx token `.`
```

So `last_user_content_token` avoids the assistant marker, but it still lands
after the model has consumed a strong answer-format instruction. InternVL appears
more sensitive to this than Qwen.

A simple metadata check does not explain the near-perfect score: tile count /
image size / question length are weak, and even image-id metadata is far below
the hidden-state probe. The issue is more likely answer-state contamination than
dynamic-tiling metadata alone.

## Fix Added

`src/vgs/vlm_hf.py` now supports earlier question-end readouts:

```text
last_question_token
last_question_4_mean
last_question_8_mean
```

For InternVL image and blind paths, `_question_end_index` is computed explicitly
from the raw POPE question. For Qwen, the same mode uses the rendered prompt and
the raw question text.

This readout lands on the original question mark, before the fixed
`Answer with yes or no only.` instruction. It should be a cleaner test of whether
InternVL has genuine visual-evidence geometry rather than answer-state geometry.

## Rerun Command

Run from a CUDA-visible shell:

```bash
PHASE3_STEP=gpu bash scripts/run_phase3_internvl_question_readout_all.sh
PHASE3_STEP=cpu bash scripts/run_phase3_internvl_question_readout_all.sh
```

Outputs will go to:

```text
outputs/stage_o_cross_model_question_readout/
```

Expected interpretation:

- If InternVL drops from near-perfect FP/TN AUROC to a moderate range, the old
  InternVL result was mostly answer-instruction/readout contamination.
- If InternVL remains near-perfect even at `last_question_token`, then the signal
  is not explained by the fixed answer instruction and should be treated as an
  InternVL-specific mechanism, but still evaluated with predicted-Yes FP-vs-TP
  gates before making any mitigation claim.

## Rerun Result

The `last_question_token` rerun is available at:

```text
outputs/stage_o_cross_model_question_readout/
```

The result did **not** fix InternVL. FP/TN probe performance remains near
perfect even when reading at the original question mark:

| Model | Best raw_img AUROC | Best difference AUROC |
| --- | ---: | ---: |
| InternVL2-8B | 0.999 | 1.000 |
| InternVL2.5-8B | 0.998 | 0.999 |

Deployment-style predicted-Yes FP-vs-TP analysis was regenerated:

```text
outputs/stage_o_cross_model_question_readout/audit/predicted_yes_gate_summary.csv
outputs/stage_o_cross_model_question_readout/audit/predicted_yes_gate_trigger_rates.csv
```

The full-difference score is still not a usable gate:

| Model | Difference FP-vs-TN AUROC | Difference FP-vs-TP AUROC | Top-10% FP caught |
| --- | ---: | ---: | ---: |
| InternVL2-8B | 0.999 | 0.187 | 1 / 18 |
| InternVL2.5-8B | 0.999 | 0.121 | 0 / 47 |

Margin entropy remains much better in the predicted-Yes setting:

| Model | Margin entropy FP-vs-TP AUROC | Top-10% FP caught |
| --- | ---: | ---: |
| InternVL2-8B | 0.883 | 9 / 18 |
| InternVL2.5-8B | 0.903 | 24 / 47 |

Updated interpretation:

```text
The InternVL FP/TN signal is real in the sense that it survives an earlier
question-token readout, but it is not the deployable hallucination-risk signal
needed for selective correction. It separates FP from TN while ranking TP above
FP inside the model-predicted-Yes set. Therefore InternVL should be reported as
a cautionary cross-architecture failure case for geometry-only gating, not as a
successful replication of the Stage T utility claim.
```

For InternVL, the next useful experiment is not another nearby prompt readout.
It should either:

1. use margin-first or margin+geometry gates, or
2. train/evaluate directly on predicted-Yes FP vs TP, or
3. inspect earlier layers before visual evidence has been converted into a
   model answer state.
