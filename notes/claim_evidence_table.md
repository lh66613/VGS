# Claim Evidence Table

| Claim | Required Stage | Evidence Artifact | Status | Notes |
| --- | --- | --- | --- | --- |
| The blind-reference difference matrix has stable low-rank structure. | Stage A | `outputs/svd/` | pending |  |
| The subspace is related to image-conditioned correction. | Stage B | `outputs/stage_b/stage_b_condition_score_summary.csv`; `outputs/stage_b/stage_b_pairwise_condition_deltas.csv`; `outputs/stage_b/stage_b_condition_subspace_similarity.csv` | supported | Residual/tail and supervised decision views separate matched evidence from mismatch conditions; top-backbone energy alone is not evidence-specific. |
| The subspace carries hallucination-relevant predictive signal. | Stage C | probe comparison table | pending |  |
| The geometry is organized across layers. | Stage D | layerwise summary and angle heatmap | pending |  |
| The subspace has causal influence on faithfulness. | Stage E | intervention pilot | pending |  |
