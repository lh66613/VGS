# External Benchmark Choice

## Decision

Use **AMBER discriminative** as the first external-validity sanity check.

## Rationale

- It is closest to the current POPE yes/no setup, so the first transfer test isolates benchmark shift rather than task-format shift.
- It includes discriminative existence, attribute, and relation queries, which lets us test whether POPE-derived correction geometry transfers beyond object existence.
- The official AMBER repository exposes separate query files for all discriminative queries and for existence / attribute / relation subsets.
- The official evaluation script uses an annotation file with `truth` labels and reports discriminative accuracy / precision / recall / F1.

## Primary Protocol

1. Prepare AMBER discriminative rows without refitting any POPE subspace.
2. Run LLaVA-1.5-7B on the AMBER rows and save yes/no predictions.
3. Dump paired image/blind hidden states using the same `last_prompt_token` readout.
4. Apply POPE SVD bases directly to AMBER differences.
5. Evaluate transfer by correctness / hallucination outcome and by AMBER dimension.

## Source Notes

- Paper / HF page: https://huggingface.co/papers/2311.07397
- Official repository: https://github.com/junyangwang0410/AMBER
- Repository README lists discriminative query files including `query_discriminative.json`, `query_discriminative-existence.json`, `query_discriminative-attribute.json`, and `query_discriminative-relation.json`.

## Local Layout

- Queries: `data/amber/data/query/query_discriminative.json`
- Annotations: `data/amber/data/annotations.json`
- Images: `data/amber/image/AMBER_*.jpg`
