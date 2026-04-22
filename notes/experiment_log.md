# Experiment Log

| Time UTC | Stage | Status | Command | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-22T09:58:13+00:00 | analyze_spectrum | dry_run | `scripts/analyze_spectrum.py --dry-run --layers 8 12 --output-dir outputs/svd` | outputs/svd/analyze_spectrum_summary.json |
| 2026-04-22T10:33:14+00:00 | validate_pope_data | ok | `scripts/validate_pope_data.py` | outputs/predictions/validate_pope_data_summary.json |
| 2026-04-22T10:33:38+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --model-path /data/lh/ModelandDataset/llava-1.5-7b-hf --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:35:10+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:36:17+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:36:17+00:00 | validate_pope_data | ok | `scripts/validate_pope_data.py` | outputs/predictions/validate_pope_data_summary.json |
