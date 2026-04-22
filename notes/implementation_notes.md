# Implementation Notes

## Fixed Implementation Choices

- Model: `LLaVA-1.5-7B`
- Benchmark: `POPE`
- Weights: frozen
- Conda environment: `after`
- Model path: `/data/lh/ModelandDataset/llava-1.5-7b-hf`
- Implementation source: Hugging Face
- Loader type: `LlavaForConditionalGeneration`
- Transformers version observed in `after`: `4.37.2`
- Torch version observed in `after`: `2.2.1+cu121`
- CUDA availability observed in this shell: unavailable
- Processor compatibility: `AutoProcessor(..., use_fast=False)` still hits an
  `image_token` config mismatch under transformers 4.37.2, so the code falls
  back to manual `LlavaProcessor(CLIPImageProcessor, AutoTokenizer(use_fast=False))`.
- Main POPE family: `coco`
- Hidden stream: transformer post-block hidden state
- Primary readout position: TBD after token-position pilot

## Intervention Precheck

Fill this section before Stage E:

- Image-token insertion/fusion path:
- Transformer block module names:
- `output_hidden_states=True` behavior:
- Hook target:
- No-op hook equality:
- Random-direction perturbation result:
