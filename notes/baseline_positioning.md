# Baseline Positioning

## Detection

Available detection baselines include raw image/blind hidden-state probes, paired difference probes, SVD-coordinate probes, Stage L evidence-specific subspaces, and a small Stage M first-token margin/entropy subset.

Top available AUROC rows:

- `yes/no margin baseline` (stage_m_first_token_subset): AUROC `1.000`, notes: Higher yes-minus-no logit indicates FP risk.
- `binary entropy baseline` (stage_m_first_token_subset): AUROC `0.884`, notes: Higher entropy indicates uncertainty.
- `Stage M FP-risk score` (stage_m_first_token_subset): AUROC `0.827`, notes: Train-bank FP-risk score on the Stage M first-token subset.
- `paired full difference, 5-seed` (stage_p_multiseed): AUROC `0.721`, notes: Mean over 5 stratified split/logistic seeds.
- `evidence-specific pls_fp_tn` (stage_l_evidence_subspace): AUROC `0.720`, notes: Compact Stage L evidence-specific subspace.
- `paired full blind-image difference` (pope_full_probe): AUROC `0.694`, notes: Primary paired-difference detector.
- `top-256 SVD coordinates, 5-seed` (stage_p_multiseed): AUROC `0.677`, notes: Mean over 5 stratified split/logistic seeds.
- `raw blind-state hidden probe` (pope_full_probe): AUROC `0.672`, notes: Text-only hidden-state baseline.

Interpretation: the paired-difference family is competitive, but it should not be framed only as a leaderboard win. Its value is mechanistic: it explains where hallucination-related signal appears and why top-variance directions are misleading.

## Mitigation / Rescue

Available mitigation comparisons include Stage M first-token rescue rows and Stage T official VCD diffusion baseline/routing rows.

Top available rescue rows:

- `official VCD baseline (diffusion, always-on)` / `vcd_diffusion`: rescue `0.302`, TN preservation ``, TP preservation `0.912`.
- `official VCD + best FP-reduction gate` / `vcd_diffusion`: rescue `0.283`, TN preservation ``, TP preservation `0.948`.
- `random steering control` / `random_tn_mean_correction`: rescue `0.250`, TN preservation ``, TP preservation ``.
- `global mean correction` / `global_tn_mean_correction`: rescue `0.250`, TN preservation ``, TP preservation ``.
- `local TN correction` / `same_object_tn_mean_correction`: rescue `0.250`, TN preservation ``, TP preservation ``.
- `official VCD + best accuracy-preserving gate` / `vcd_diffusion`: rescue `0.151`, TN preservation ``, TP preservation `0.994`.
- `no intervention` / `baseline`: rescue `0.000`, TN preservation `1.000`, TP preservation `1.000`.
- `no intervention (Stage T VCD pool)` / `baseline`: rescue `0.000`, TN preservation ``, TP preservation `1.000`.

Interpretation: official always-on VCD reduces FPs but damages TPs; the selective routing rows preserve more TPs and expose the deployment tradeoff. Stage M steering remains boundary-local and weak.
