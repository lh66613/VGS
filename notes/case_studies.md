# Case Studies

Human-readable case panels generated from Stage R.

## successful_tn_strong_tail

### case_01 — coco:random:2200

- Image: `data/pope/images/COCO_val2014_000000470699.jpg`
- Question: Is there an airplane in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `airplane`
- Geometry: L24 tail `1280.051`, L24 top-4 `8742.016`, FP-risk ``
- Intervention: none recorded
- Note: Correct No with a strong matched tail-correction score.

### case_02 — coco:popular:1248

- Image: `data/pope/images/COCO_val2014_000000383185.jpg`
- Question: Is there a car in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `car`
- Geometry: L24 tail `966.322`, L24 top-4 `9006.955`, FP-risk ``
- Intervention: none recorded
- Note: Correct No with a strong matched tail-correction score.

### case_03 — coco:popular:2054

- Image: `data/pope/images/COCO_val2014_000000167724.jpg`
- Question: Is there a dining table in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `dining table`
- Geometry: L24 tail `905.134`, L24 top-4 `9740.290`, FP-risk ``
- Intervention: none recorded
- Note: Correct No with a strong matched tail-correction score.

### case_04 — coco:random:786

- Image: `data/pope/images/COCO_val2014_000000369541.jpg`
- Question: Is there a skateboard in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `skateboard`
- Geometry: L24 tail `762.377`, L24 top-4 `8597.157`, FP-risk ``
- Intervention: none recorded
- Note: Correct No with a strong matched tail-correction score.

## fp_weak_matched_correction

### case_05 — coco:popular:2388

- Image: `data/pope/images/COCO_val2014_000000061507.jpg`
- Question: Is there a chair in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `chair`
- Geometry: L24 tail `12.796`, L24 top-4 `10464.449`, FP-risk ``
- Intervention: none recorded
- Note: False Positive with weak matched tail-correction score.

### case_06 — coco:popular:138

- Image: `data/pope/images/COCO_val2014_000000516916.jpg`
- Question: Is there a cup in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `cup`
- Geometry: L24 tail `13.954`, L24 top-4 `10302.149`, FP-risk ``
- Intervention: none recorded
- Note: False Positive with weak matched tail-correction score.

### case_07 — coco:popular:1780

- Image: `data/pope/images/COCO_val2014_000000245448.jpg`
- Question: Is there a car in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `car`
- Geometry: L24 tail `15.187`, L24 top-4 `10736.799`, FP-risk ``
- Intervention: none recorded
- Note: False Positive with weak matched tail-correction score.

### case_08 — coco:popular:1180

- Image: `data/pope/images/COCO_val2014_000000184338.jpg`
- Question: Is there a car in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `car`
- Geometry: L24 tail `16.916`, L24 top-4 `10881.961`, FP-risk ``
- Intervention: none recorded
- Note: False Positive with weak matched tail-correction score.

## fp_rescued_by_local_steering

### case_09 — coco:popular:2714

- Image: `data/pope/images/COCO_val2014_000000318550.jpg`
- Question: Is there a person in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `person`
- Geometry: L24 tail `31.898`, L24 top-4 `9115.821`, FP-risk `0.630`
- Intervention: margin_and_fp_risk/random_tn_mean_correction/alpha=8.0/after=no:TN
- Note: Borderline False Positive rescued by a Stage M steering setting.

### case_10 — coco:popular:966

- Image: `data/pope/images/COCO_val2014_000000419453.jpg`
- Question: Is there a chair in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `chair`
- Geometry: L24 tail `248.868`, L24 top-4 `8959.430`, FP-risk `0.424`
- Intervention: low_abs_margin/random_tn_mean_correction/alpha=8.0/after=no:TN
- Note: Borderline False Positive rescued by a Stage M steering setting.

## fp_not_rescued_high_score

### case_11 — coco:popular:1208

- Image: `data/pope/images/COCO_val2014_000000124930.jpg`
- Question: Is there a dining table in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `dining table`
- Geometry: L24 tail `440.395`, L24 top-4 `8514.499`, FP-risk `1.000`
- Intervention: high_fp_risk/same_object_tn_mean_correction/alpha=8.0/after=yes:FP
- Note: High-risk False Positive remains unrescued, useful for failure analysis.

### case_12 — coco:random:1708

- Image: `data/pope/images/COCO_val2014_000000301575.jpg`
- Question: Is there a toothbrush in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `toothbrush`
- Geometry: L24 tail `1256.523`, L24 top-4 `8131.922`, FP-risk `1.000`
- Intervention: margin_and_fp_risk/random_tn_mean_correction/alpha=8.0/after=yes:FP
- Note: High-risk False Positive remains unrescued, useful for failure analysis.

### case_13 — coco:adversarial:526

- Image: `data/pope/images/COCO_val2014_000000207205.jpg`
- Question: Is there a dining table in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `dining table`
- Geometry: L24 tail `214.316`, L24 top-4 `10130.919`, FP-risk `1.000`
- Intervention: high_fp_risk/same_object_tn_mean_correction/alpha=8.0/after=yes:FP
- Note: High-risk False Positive remains unrescued, useful for failure analysis.

### case_14 — coco:popular:2598

- Image: `data/pope/images/COCO_val2014_000000299986.jpg`
- Question: Is there a chair in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `chair`
- Geometry: L24 tail `237.903`, L24 top-4 `9304.438`, FP-risk `1.000`
- Intervention: high_fp_risk/random_tn_mean_correction/alpha=8.0/after=yes:FP
- Note: High-risk False Positive remains unrescued, useful for failure analysis.

## adversarial_mismatch_example

### case_15 — coco:adversarial:1110

- Image: `data/pope/images/COCO_val2014_000000040361.jpg`
- Question: Is there a baseball glove in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `baseball glove`
- Geometry: L24 tail `1041.977`, L24 top-4 `8357.713`, FP-risk ``
- Intervention: none recorded
- Note: Adversarial subset case with a large matched-vs-adversarial tail-score difference.

### case_16 — coco:adversarial:2800

- Image: `data/pope/images/COCO_val2014_000000355776.jpg`
- Question: Is there a dining table in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `dining table`
- Geometry: L24 tail `704.149`, L24 top-4 `9605.644`, FP-risk ``
- Intervention: none recorded
- Note: Adversarial subset case with a large matched-vs-adversarial tail-score difference.

### case_17 — coco:adversarial:580

- Image: `data/pope/images/COCO_val2014_000000287305.jpg`
- Question: Is there a truck in the image?
- Ground truth / model answer / outcome: `no` / `yes` / `FP`
- Target object: `truck`
- Geometry: L24 tail `517.296`, L24 top-4 `8431.327`, FP-risk ``
- Intervention: none recorded
- Note: Adversarial subset case with a large matched-vs-adversarial tail-score difference.

### case_18 — coco:adversarial:184

- Image: `data/pope/images/COCO_val2014_000000291936.jpg`
- Question: Is there a motorcycle in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `motorcycle`
- Geometry: L24 tail `86.394`, L24 top-4 `10749.291`, FP-risk ``
- Intervention: none recorded
- Note: Adversarial subset case with a large matched-vs-adversarial tail-score difference.

## semantic_direction_extreme

### case_19 — coco:adversarial:1407

- Image: `data/pope/images/COCO_val2014_000000511341.jpg`
- Question: Is there a person in the image?
- Ground truth / model answer / outcome: `yes` / `yes` / `TP`
- Target object: `person`
- Geometry: L24 tail ``, L24 top-4 ``, FP-risk ``
- Intervention: none recorded
- Note: Extreme sample for an interpreted semantic geometry object.

### case_20 — coco:random:917

- Image: `data/pope/images/COCO_val2014_000000539251.jpg`
- Question: Is there a potted plant in the image?
- Ground truth / model answer / outcome: `yes` / `yes` / `TP`
- Target object: `potted plant`
- Geometry: L24 tail ``, L24 top-4 ``, FP-risk ``
- Intervention: none recorded
- Note: Extreme sample for an interpreted semantic geometry object.

### case_21 — coco:random:1840

- Image: `data/pope/images/COCO_val2014_000000333756.jpg`
- Question: Is there a refrigerator in the image?
- Ground truth / model answer / outcome: `no` / `no` / `TN`
- Target object: `refrigerator`
- Geometry: L24 tail ``, L24 top-4 ``, FP-risk ``
- Intervention: none recorded
- Note: Extreme sample for an interpreted semantic geometry object.

### case_22 — coco:adversarial:901

- Image: `data/pope/images/COCO_val2014_000000430052.jpg`
- Question: Is there a vase in the image?
- Ground truth / model answer / outcome: `yes` / `yes` / `TP`
- Target object: `vase`
- Geometry: L24 tail ``, L24 top-4 ``, FP-risk ``
- Intervention: none recorded
- Note: Extreme sample for an interpreted semantic geometry object.
