# Naming Decision

## Current Name

Use **evidence-sensitive correction geometry** for the main object studied in this project.

## Why

The geometry is more than a generic blind-image difference: matched visual evidence separates from random/adversarial mismatch conditions in supervised decision scores and several evidence-specific coordinates.

It should not yet be called a **visual grounding subspace** because:

- destructive shuffle controls show the dominant low-rank spectrum is not uniquely paired-evidence specific;
- plain SVD top directions are not uniquely discriminative;
- cross-model and external-benchmark evidence is still pending;
- mitigation/rescue effects remain local rather than broadly strong.

## Supporting Stage L Result

Stage L separates two roles:

- PLS FP/TN directions are strongest for hallucination detection, reaching AUROC 0.7196 at L24 K=32.
- Contrastive PCA is strongest for matched-vs-mismatch condition gaps, especially L32.

This supports wording the result as layered and evidence-sensitive rather than as one universal compact subspace.

## Recommended Paper Wording

- Primary phrase: **evidence-sensitive correction geometry**
- Mechanistic phrase: **hallucination-sensitive residual correction coordinates**
- Broader framing: **layered visual-evidence correction geometry**

Avoid:

- visual grounding subspace
- universal hallucination subspace
- causal grounding direction
