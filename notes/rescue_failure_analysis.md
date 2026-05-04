# Stage M Rescue Failure Analysis

Artifacts:

- Taxonomy: `outputs/stage_m_local_rescue/rescue_failure_taxonomy.csv`
- Group summary: `outputs/stage_m_local_rescue/rescue_failure_group_summary.csv`

## Outcome Taxonomy

Total FP samples analyzed: 32

- margin_improved_answer_unchanged: 30
- rescued_to_correct_no: 2

## Margin Bins

- medium_abs_0.25_1.0: 16
- borderline_abs_le_0.25: 9
- high_abs_gt_1.0: 7

## Rescued Samples

- `coco:popular:2714` / object `person` / subset `popular`: baseline yes-no margin `0.015625`, best gain `0.109375`, best `margin_and_fp_risk` + `random_tn_mean_correction` at alpha `8.0`.
- `coco:popular:966` / object `chair` / subset `popular`: baseline yes-no margin `0.031250`, best gain `0.093750`, best `low_abs_margin` + `random_tn_mean_correction` at alpha `8.0`.

## Interpretation

- Rescue is concentrated in extremely low-margin FP samples.
- Most FP samples receive a positive margin nudge but do not cross the yes/no boundary.
- High-margin FP cases remain failures under this first-token steering setup, consistent with stronger language-prior or visual-evidence-insensitive hallucinations.
- Current retrieval diagnostics should be treated cautiously because this run did not include the `same_object_fp` retrieval mode during intervention.

Median baseline margin for margin-improved but unrescued FP samples: `0.656250`.
