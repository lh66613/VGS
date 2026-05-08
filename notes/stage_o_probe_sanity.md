# Stage O Probe Sanity

CSV: `outputs/stage_o_cross_model/audit/probe_sanity.csv`

## Best Real-Label Rows

### internvl2_5_8b
- L24 `top32_projected_difference` AUROC `1.000` AUPRC `1.000`
- L32 `difference` AUROC `1.000` AUPRC `1.000`
- L20 `top32_projected_difference` AUROC `1.000` AUPRC `1.000`
- L32 `raw_img` AUROC `1.000` AUPRC `1.000`
- L20 `raw_img` AUROC `1.000` AUPRC `0.999`

### internvl2_8b
- L20 `difference` AUROC `0.999` AUPRC `0.986`
- L20 `raw_img` AUROC `0.999` AUPRC `0.986`
- L24 `raw_img` AUROC `0.999` AUPRC `0.983`
- L24 `difference` AUROC `0.999` AUPRC `0.981`
- L32 `raw_img` AUROC `0.999` AUPRC `0.977`

### llava_13b
- L20 `difference` AUROC `0.744` AUPRC `0.299`
- L24 `difference` AUROC `0.731` AUPRC `0.290`
- L32 `difference` AUROC `0.718` AUPRC `0.274`
- L32 `raw_blind` AUROC `0.714` AUPRC `0.197`
- L24 `raw_blind` AUROC `0.711` AUPRC `0.200`

### qwen2_5_vl_7b
- L28 `difference` AUROC `1.000` AUPRC `0.987`
- L28 `raw_img` AUROC `0.999` AUPRC `0.986`
- L28 `top32_projected_difference` AUROC `0.999` AUPRC `0.987`
- L24 `top32_projected_difference` AUROC `0.999` AUPRC `0.982`
- L24 `difference` AUROC `0.999` AUPRC `0.976`

### qwen2_vl_7b
- L24 `raw_img` AUROC `0.999` AUPRC `0.979`
- L24 `top32_projected_difference` AUROC `0.999` AUPRC `0.975`
- L28 `difference` AUROC `0.999` AUPRC `0.973`
- L24 `difference` AUROC `0.999` AUPRC `0.972`
- L28 `raw_img` AUROC `0.999` AUPRC `0.969`

## Label-Shuffle Check

- `internvl2_5_8b` max shuffled-label AUROC `0.856`
- `internvl2_8b` max shuffled-label AUROC `0.909`
- `llava_13b` max shuffled-label AUROC `0.574`
- `qwen2_5_vl_7b` max shuffled-label AUROC `0.998`
- `qwen2_vl_7b` max shuffled-label AUROC `0.994`

Interpretation:

- If real-label AUROC stays near 1.0 under split-locked evaluation while shuffled-label AUROC collapses, the separability is not a train/test leakage bug.
- It can still be a readout-position confound if the representation is taken at the assistant generation prompt and therefore linearly exposes the next-token decision.
