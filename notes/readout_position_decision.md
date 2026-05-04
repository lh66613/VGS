# Readout Position Decision

## Decision

- Primary readout for the current main paper version: `last_prompt_token`.
- Secondary robustness readout for appendix: `last_4_prompt_mean`.

## Rationale

`last_prompt_token` remains the primary readout because all completed Stage A/B/C/E experiments were run with it, and it is the simplest single-token convention to reproduce in the HuggingFace LLaVA path.

`last_4_prompt_mean` is the strongest secondary readout because it improves FP-vs-TN discrimination while preserving the same qualitative pattern: top variance directions explain most variance, but hallucination-sensitive AUROC improves only at larger K.

## Stage K Evidence

- `last_prompt_token` and `first_answer_prefill` are identical in the current implementation because first-answer prefill is represented by the causal hidden state at the last prompt token before generation.
- Full-difference AUROC improves from the `last_prompt_token` range of 0.6197-0.6595 to:
  - `last_4_prompt_mean`: 0.6798-0.7108
  - `last_8_prompt_mean`: 0.7023-0.7377
- Best top-K result is `last_4_prompt_mean`, L24, K=128, AUROC 0.6947.
- `last_8_prompt_mean` has very high explained variance at early/mid layers, which makes it useful as a robustness check but less clean as a primary mechanistic readout.

## Caveat

Switching the primary readout to a pooled position would require rerunning Stage B condition geometry and Stage E interventions under that readout. Until those reruns are complete, `last_prompt_token` is the primary protocol and `last_4_prompt_mean` is the robustness appendix setting.
