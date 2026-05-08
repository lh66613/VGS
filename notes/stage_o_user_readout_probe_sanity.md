# Stage O Probe Sanity

CSV: `outputs/stage_o_cross_model_user_readout/audit/probe_sanity.csv`

## Best Real-Label Rows

### internvl2_5_8b
- L32 `raw_img` AUROC `0.999` AUPRC `0.988`
- L32 `difference` AUROC `0.998` AUPRC `0.983`
- L24 `raw_img` AUROC `0.998` AUPRC `0.982`
- L20 `difference` AUROC `0.998` AUPRC `0.976`
- L20 `raw_img` AUROC `0.998` AUPRC `0.975`

### internvl2_8b
- L32 `raw_img` AUROC `0.999` AUPRC `0.965`
- L32 `top32_projected_difference` AUROC `0.998` AUPRC `0.946`
- L24 `raw_img` AUROC `0.997` AUPRC `0.924`
- L32 `difference` AUROC `0.997` AUPRC `0.918`
- L20 `difference` AUROC `0.997` AUPRC `0.924`

### qwen2_5_vl_7b
- L20 `difference` AUROC `0.750` AUPRC `0.303`
- L24 `difference` AUROC `0.749` AUPRC `0.334`
- L28 `difference` AUROC `0.745` AUPRC `0.279`
- L24 `raw_blind` AUROC `0.743` AUPRC `0.101`
- L20 `raw_blind` AUROC `0.735` AUPRC `0.100`

### qwen2_vl_7b
- L28 `difference` AUROC `0.674` AUPRC `0.110`
- L28 `raw_img` AUROC `0.670` AUPRC `0.127`
- L24 `raw_img` AUROC `0.668` AUPRC `0.130`
- L20 `raw_img` AUROC `0.632` AUPRC `0.132`
- L24 `raw_blind` AUROC `0.631` AUPRC `0.068`

## Label-Shuffle Check

- `internvl2_5_8b` max shuffled-label AUROC `0.924`
- `internvl2_8b` max shuffled-label AUROC `0.896`
- `qwen2_5_vl_7b` max shuffled-label AUROC `0.622`
- `qwen2_vl_7b` max shuffled-label AUROC `0.610`

Interpretation:

- If real-label AUROC stays near 1.0 under split-locked evaluation while shuffled-label AUROC collapses, the separability is not a train/test leakage bug.
- It can still be a readout-position confound if the representation is taken at the assistant generation prompt and therefore linearly exposes the next-token decision.
