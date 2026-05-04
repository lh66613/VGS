# Prompt Templates

## Image + Question

```text
USER: <image>
{question} Answer with yes or no only. ASSISTANT:
```

## Blind / Text Only

```text
USER: {question} Answer with yes or no only. ASSISTANT:
```

## Random Mismatch

```text
USER: <image>
{question} Answer with yes or no only. ASSISTANT:
```

The image path is replaced with a random unrelated sample image from the condition plan.

## Adversarial Mismatch

```text
USER: <image>
{question} Answer with yes or no only. ASSISTANT:
```

The image path is replaced with a same-question, opposite-label image when available.

## Current Check

- Text instruction is identical: `Answer with yes or no only.`
- Difference intended by the current HF path: image prompt includes `<image>` and image pixels; blind prompt omits both.
- Random and adversarial mismatch prompts reuse the same image+question text template; only the supplied image changes.
- The exact image prompt may be expanded by `processor.apply_chat_template` at runtime; keep the processor/model version fixed in reported runs.
