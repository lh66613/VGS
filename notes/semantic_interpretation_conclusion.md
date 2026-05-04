# Semantic Interpretation Conclusion

## What The Semantic Fingerprints Support

- Top-SVD backbone directions often show broad visual-scene, attribute, count, action, or spatial vocabulary fingerprints.
- The L24 tail 257-1024 slice is object-heavy in vocabulary projection and aligns with the causal/tail-ablation story.
- L32 local TN rescue directions look more like relational/contextual or yes-no arbitration directions than object-presence detectors.
- Stage L evidence-specific subspaces are quantitatively useful, but they have not yet been vocabulary-projected in this artifact bundle.

## What They Do Not Support

- Do not claim that any single semantic direction is a strong hallucination detector.
- Do not claim that the token projection proves a full mechanistic circuit.
- Do not call the result a universal visual grounding subspace.

## Summary Counts

- Projected geometry objects: 28
- Stage L quantitative-only rows: 4

## Strongest Sample-Level FP/TN Contrasts

- `L20_svd_8` (top_svd_backbone): FP/TN AUC `0.562`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L24_svd_8` (top_svd_backbone): FP/TN AUC `0.451`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L32_svd_5` (top_svd_backbone): FP/TN AUC `0.454`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L24_svd_7` (top_svd_backbone): FP/TN AUC `0.456`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L24_svd_5` (top_svd_backbone): FP/TN AUC `0.539`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L20_svd_5` (top_svd_backbone): FP/TN AUC `0.471`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L32_svd_6` (top_svd_backbone): FP/TN AUC `0.528`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.
- `L20_svd_7` (top_svd_backbone): FP/TN AUC `0.527`, interpretation: Broad visual-scene/action backbone axis with interpretable but mixed vocabulary.

## Recommended Paper Wording

Use: **partially interpretable grounding-related correction geometry**. Avoid: **semantic hallucination coordinate**, **universal grounding subspace**, or claims that vocabulary projection alone establishes causality.
