# Representation Editing Prep

This is a CPU-prepared direction bank for future GPU-side activation-editing experiments. It is not a trained LoRA adapter.

Artifacts:

- Direction bank: `outputs/representation_editing_prep/editing_direction_bank.pt`
- Candidate eval plan: `outputs/representation_editing_prep/candidate_eval_plan.csv`
- Direction summary: `outputs/representation_editing_prep/direction_summary.csv`

Recommended first GPU test:

- Use L24/L32 `pls_fp_tn` or `fisher_fp_tn` projected directions.
- Start with `projected_tn_minus_fp` at alpha 2/4/8.
- Gate with `margin_and_fp_risk` and evaluate FP rescue plus TN/TP damage.

Top candidate rows:

- L24 `pls_fp_tn` `projected_tn_minus_fp` k=32: TN-FP projection `3.535`, source AUROC `0.720`
- L24 `pls_fp_tn` `projected_tn_mean` k=32: TN-FP projection `-0.121`, source AUROC `0.720`
- L20 `pls_fp_tn` `projected_tn_minus_fp` k=64: TN-FP projection `2.448`, source AUROC `0.714`
- L20 `pls_fp_tn` `projected_tn_mean` k=64: TN-FP projection `0.078`, source AUROC `0.714`
- L20 `pls_fp_tn` `projected_tn_minus_fp` k=32: TN-FP projection `2.447`, source AUROC `0.712`
- L20 `pls_fp_tn` `projected_tn_mean` k=32: TN-FP projection `0.090`, source AUROC `0.712`
- L24 `pls_fp_tn` `projected_tn_minus_fp` k=64: TN-FP projection `3.535`, source AUROC `0.711`
- L24 `pls_fp_tn` `projected_tn_mean` k=64: TN-FP projection `-0.090`, source AUROC `0.711`
- L20 `pls_fp_tn` `projected_tn_minus_fp` k=16: TN-FP projection `2.447`, source AUROC `0.703`
- L20 `pls_fp_tn` `projected_tn_mean` k=16: TN-FP projection `0.102`, source AUROC `0.703`
- L24 `pls_fp_tn` `projected_tn_minus_fp` k=16: TN-FP projection `3.534`, source AUROC `0.698`
- L24 `pls_fp_tn` `projected_tn_mean` k=16: TN-FP projection `-0.151`, source AUROC `0.698`

Interpretation:

- The bank turns Stage L subspaces into concrete normalized edit directions.
- It is suitable for a next-step activation-editing pilot.
- It does not by itself establish a mitigation method.
