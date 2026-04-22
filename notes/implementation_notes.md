# Implementation Notes

## Fixed Implementation Choices

- Model: `LLaVA-1.5-7B`
- Benchmark: `POPE`
- Weights: frozen
- Implementation source: TBD, but must remain fixed once selected
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
