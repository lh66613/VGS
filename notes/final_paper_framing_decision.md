# Final Paper Framing Decision

## Decision

The current evidence supports a **mechanistic analysis paper**, not a mitigation-method paper.

Recommended framing:

- Main contribution: blind-reference differencing reveals layered correction geometry in LVLM hallucination.
- Detection claim: paired differencing is mechanistically informative and competitive, but should not be framed as a pure AUROC leaderboard win.
- Intervention claim: steering/rescue provides causal evidence that the identified geometry can affect boundary cases, but it is not yet a reliable mitigation method.

## Why This Is The Right Framing

### Strong supporting evidence

- The five-seed Stage P rerun shows that full blind-image paired differencing is the strongest and most stable hidden-state representation among the tested paired/SVD variants.
- Stage Q tables and figures provide paper-ready evidence that top-variance SVD directions are not the decision geometry.
- Stage R semantic fingerprints suggest partially interpretable grounding-related correction geometry, especially in the L24 tail and L32 local rescue directions.
- Stage N full AMBER transfer is modest but nonzero, so the result is not purely a tiny POPE artifact.

### Constraints on stronger claims

- Stage S detection baselines show that yes/no margin and entropy can be very strong when first-token logits are available.
- Current Stage M rescue is weak and boundary-local.
- Random/global/local TN steering controls are competitive on the small gated rescue subset.
- TN/TP preservation is undefined for the best gated rescue rows because no clean controls passed the same gate.
- VCD/ICD, evidence-specific steering, compute overhead, and cross-model replication are not yet available.

## Recommended Paper Claims

Use these claims:

- Blind-reference differencing separates hallucination-related decision geometry from dominant activation variance.
- Hallucination signal is layered: broad high-variance directions, mid/late tail subspaces, and boundary-local correction directions play different roles.
- Semantic projection gives partial interpretability, but no single projected direction is a detector.
- Intervention results are causal probes of the geometry, not a deployed mitigation method.

Avoid these claims:

- Universal visual grounding subspace.
- Single semantic hallucination coordinate.
- State-of-the-art hallucination detection.
- Reliable hallucination mitigation.
- General LVLM mechanism across model families.

## Title Direction

Best current title style:

- "Blind-Reference Differencing Reveals Layered Correction Geometry in Vision-Language Hallucination"

Backup:

- "Dominant Difference Is Not Decision Geometry: A Blind-Reference Analysis of LVLM Hallucination"

## Remaining Work For A Stronger Paper

- Cross-model replication would upgrade the claim from a LLaVA-centered mechanism to a recurring LVLM pattern.
- Compute-overhead measurement would make the baseline comparison more complete.
- Evidence-specific steering would be needed before presenting intervention as a serious mitigation method.
- Semantic stability and image-region correlation would strengthen the interpretability section.
