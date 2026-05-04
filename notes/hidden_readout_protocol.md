# Hidden Readout Protocol

## Current Primary Readout

- Hidden stream: `post_block` transformer hidden-state tuple from HuggingFace `LlavaForConditionalGeneration`.
- Current token position: `last_prompt_token`.
- Image path: full multimodal model forward with image tokens and text prompt.
- Blind path: language model forward on the text-only prompt.
- Difference convention: `D = z_blind - z_img`.

## Observed Artifact Metadata

| Layer | Readout position | Hidden stream | Samples | Hidden dim |
| ---: | --- | --- | ---: | ---: |
| 8 | last_prompt_token | post_block | 9000 | 4096 |
| 12 | last_prompt_token | post_block | 9000 | 4096 |
| 16 | last_prompt_token | post_block | 9000 | 4096 |
| 20 | last_prompt_token | post_block | 9000 | 4096 |
| 24 | last_prompt_token | post_block | 9000 | 4096 |
| 28 | last_prompt_token | post_block | 9000 | 4096 |
| 32 | last_prompt_token | post_block | 9000 | 4096 |

## Pending Robustness Positions

- `first_answer_prefill`
- `last_4_prompt_mean`
- `last_8_prompt_mean`
- `question_object_token_mean` if object-token localization is implemented.
- `image_adjacent_text_token` if accessible in the chosen implementation.
