# Protocol Lock

## Fixed Artifacts

- Predictions: `outputs/predictions/pope_predictions.jsonl`
- Hidden states: `outputs/hidden_states`
- Train IDs: `outputs/splits/pope_train_ids.json`
- Validation IDs: `outputs/splits/pope_val_ids.json`
- Test IDs: `outputs/splits/pope_test_ids.json`

## Split Policy

- Seed: `42`
- Train / validation / test fractions: `0.70` / `0.15` / `0.15`
- Stratification keys: `subset`, `label`, `outcome`.
- Test labels must not be used for subspace extraction, classifier fitting, or intervention memory-bank construction.

## Split Counts

| Split | Count |
| --- | ---: |
| train | 6300 |
| val | 1350 |
| test | 1350 |
