# Cross-Model Replication

Model alias: `llava_13b`
Artifact root: `outputs/stage_o_cross_model/llava_13b`

## Status

Stage O minimal replication artifacts are available.

## Best Available Probe Rows

- L20 `difference` k=full: AUROC `0.736`
- L24 `difference` k=full: AUROC `0.726`
- L32 `difference` k=full: AUROC `0.723`
- L32 `projected_difference` k=128: AUROC `0.699`
- L20 `projected_difference` k=128: AUROC `0.694`
- L24 `projected_difference` k=128: AUROC `0.693`
- L32 `raw_blind` k=full: AUROC `0.693`
- L20 `raw_blind` k=full: AUROC `0.692`

## Condition Geometry Snapshot

- L20 `full_l2_sq` `matched_minus_random_mismatch` mean delta `58.468`
- L20 `full_l2_sq` `matched_minus_adversarial_mismatch` mean delta `-15.131`
- L20 `band_257_1024_l2_sq` `matched_minus_random_mismatch` mean delta `3.622`
- L20 `band_257_1024_l2_sq` `matched_minus_adversarial_mismatch` mean delta `18.126`
- L24 `full_l2_sq` `matched_minus_random_mismatch` mean delta `86.245`
- L24 `full_l2_sq` `matched_minus_adversarial_mismatch` mean delta `-55.434`
- L24 `band_257_1024_l2_sq` `matched_minus_random_mismatch` mean delta `-0.711`
- L24 `band_257_1024_l2_sq` `matched_minus_adversarial_mismatch` mean delta `28.860`
- L32 `full_l2_sq` `matched_minus_random_mismatch` mean delta `282.764`
- L32 `full_l2_sq` `matched_minus_adversarial_mismatch` mean delta `-337.411`
- L32 `band_257_1024_l2_sq` `matched_minus_random_mismatch` mean delta `-14.696`
- L32 `band_257_1024_l2_sq` `matched_minus_adversarial_mismatch` mean delta `75.504`

## Interpretation Template

- Strong replication if variance/discrimination mismatch, mid-layer residual/tail strength, and matched-vs-mismatch gaps all reappear.
- Acceptable replication if only two of the three qualitative patterns reappear.
- If the pattern fails, report Stage O as a limitation rather than expanding the paper's generality claim.

## Final Interpretation

Stage O is an **acceptable-to-strong checkpoint-level replication** on `llava_13b`.

Supported:

- POPE performance is clean enough for analysis: accuracy `0.871`, with `468` FP, `4032` TN, `3804` TP, and no unknown outputs.
- The full blind-image difference is again the strongest available hidden-state detector: L20 AUROC `0.736`, L24 AUROC `0.726`, L32 AUROC `0.723`.
- The variance/discrimination mismatch clearly reappears. Top-4 SVD explains large variance, but top-4 projected AUROC remains weak: L20 `0.549`, L24 `0.535`, L32 `0.552`.
- Mid/late compact projected coordinates become useful at larger K: best projected-difference row is L32 K=128 AUROC `0.699`.
- Tail condition geometry reappears for adversarial mismatch: band 257-1024 matched-minus-adversarial deltas are positive at L20 `18.126`, L24 `28.860`, and L32 `75.504`.

Limitations:

- This is a LLaVA-family checkpoint replication, not a different-architecture LVLM replication.
- Matched-minus-random mismatch tail deltas are mixed: L20 `3.622`, L24 `-0.711`, L32 `-14.696`.
- Full L2 matched-minus-adversarial deltas are negative at all three layers, so the cleanest evidence gap is residual/tail-specific rather than full-space.

Paper consequence:

- The mechanism is no longer just a single-checkpoint LLaVA-1.5-7B artifact.
- The paper can cautiously claim recurrence across LLaVA-1.5 checkpoints.
- It still should not claim generality across LVLM architectures without a Qwen2-VL, InternVL, or LLaVA-NeXT replication.
