# Experiment Log

| Time UTC | Stage | Status | Command | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-22T09:58:13+00:00 | analyze_spectrum | dry_run | `scripts/analyze_spectrum.py --dry-run --layers 8 12 --output-dir outputs/svd` | outputs/svd/analyze_spectrum_summary.json |
| 2026-04-22T10:33:14+00:00 | validate_pope_data | ok | `scripts/validate_pope_data.py` | outputs/predictions/validate_pope_data_summary.json |
| 2026-04-22T10:33:38+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --model-path /data/lh/ModelandDataset/llava-1.5-7b-hf --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:35:10+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:36:17+00:00 | run_pope_eval | dry_run | `scripts/run_pope_eval.py --dry-run --max-samples 2` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:36:17+00:00 | validate_pope_data | ok | `scripts/validate_pope_data.py` | outputs/predictions/validate_pope_data_summary.json |
| 2026-04-22T10:46:39+00:00 | create_smoke_artifacts | ok | `scripts/create_smoke_artifacts.py --layers 8 12 --num-samples 96 --hidden-dim 64` | outputs/smoke/create_smoke_artifacts_summary.json |
| 2026-04-22T10:46:51+00:00 | build_difference_matrix | scaffold_ready | `scripts/build_difference_matrix.py --layers 8 12 --hidden-states-dir outputs/hidden_states_smoke --output-dir outputs/svd_smoke` | outputs/svd_smoke/build_difference_matrix_summary.json |
| 2026-04-22T10:47:05+00:00 | analyze_spectrum | scaffold_ready | `scripts/analyze_spectrum.py --layers 8 12 --matrix-dir outputs/svd_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke` | outputs/svd_smoke/analyze_spectrum_summary.json |
| 2026-04-22T10:47:17+00:00 | analyze_k_sensitivity | scaffold_ready | `scripts/analyze_k_sensitivity.py --layers 8 12 --k-grid 4 8 16 --svd-dir outputs/svd_smoke --matrix-dir outputs/svd_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke --repeats 5` | outputs/svd_smoke/analyze_k_sensitivity_summary.json |
| 2026-04-22T10:47:28+00:00 | train_probe | scaffold_ready | `scripts/train_probe.py --layers 8 12 --k-grid 4 8 16 --feature-family projected_difference random_difference difference --predictions outputs/predictions/smoke_predictions.jsonl --hidden-states-dir outputs/hidden_states_smoke --svd-dir outputs/svd_smoke --output-dir outputs/probes_smoke` | outputs/probes_smoke/train_probe_summary.json |
| 2026-04-22T10:47:44+00:00 | compare_features | scaffold_ready | `scripts/compare_features.py --probe-dir outputs/probes_smoke --output-dir outputs/probes_smoke` | outputs/probes_smoke/compare_features_summary.json |
| 2026-04-22T10:47:44+00:00 | layerwise_analysis | scaffold_ready | `scripts/layerwise_analysis.py --layers 8 12 --k-grid 4 8 16 --svd-dir outputs/svd_smoke --probe-dir outputs/probes_smoke --output-dir outputs/svd_smoke --plot-dir outputs/plots_smoke` | outputs/svd_smoke/layerwise_analysis_summary.json |
| 2026-04-22T10:48:46+00:00 | dump_hidden_states | dry_run | `scripts/dump_hidden_states.py --dry-run --layers 8 12 --predictions outputs/predictions/smoke_predictions.jsonl --max-samples 2` | outputs/hidden_states/dump_hidden_states_summary.json |
| 2026-04-22T10:52:33+00:00 | run_pope_eval | scaffold_ready | `scripts/run_pope_eval.py --max-samples 10` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T10:52:57+00:00 | dump_hidden_states | scaffold_ready | `scripts/dump_hidden_states.py --layers 8 12 16 20 24 28 32 --predictions outputs/predictions/pope_predictions.jsonl --max-samples 10` | outputs/hidden_states/dump_hidden_states_summary.json |
| 2026-04-22T11:25:13+00:00 | run_pope_eval | scaffold_ready | `scripts/run_pope_eval.py` | outputs/predictions/run_pope_eval_summary.json |
| 2026-04-22T11:56:14+00:00 | dump_hidden_states | scaffold_ready | `scripts/dump_hidden_states.py --layers 8 12 16 20 24 28 32 --predictions outputs/predictions/pope_predictions.jsonl` | outputs/hidden_states/dump_hidden_states_summary.json |
| 2026-04-22T12:15:44+00:00 | create_smoke_artifacts | ok | `scripts/create_smoke_artifacts.py --layers 8 --num-samples 8 --hidden-dim 16 --predictions outputs/predictions/progress_smoke_predictions.jsonl --hidden-states-dir outputs/hidden_states_progress_smoke --output-dir outputs/progress_smoke` | outputs/progress_smoke/create_smoke_artifacts_summary.json |
| 2026-04-22T12:15:46+00:00 | validate_pope_data | ok | `scripts/validate_pope_data.py` | outputs/predictions/validate_pope_data_summary.json |
| 2026-04-22T12:16:19+00:00 | build_difference_matrix | scaffold_ready | `scripts/build_difference_matrix.py --layers 8 12 16 20 24 28 32` | outputs/svd/build_difference_matrix_summary.json |
| 2026-04-22T12:17:07+00:00 | analyze_spectrum | scaffold_ready | `scripts/analyze_spectrum.py --layers 8 12 16 20 24 28 32` | outputs/svd/analyze_spectrum_summary.json |
| 2026-04-22T12:39:56+00:00 | analyze_k_sensitivity | scaffold_ready | `scripts/analyze_k_sensitivity.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64 --repeats 5 --stability-method randomized` | outputs/svd/analyze_k_sensitivity_summary.json |
| 2026-04-22T12:41:05+00:00 | analyze_k_sensitivity | scaffold_ready | `scripts/analyze_k_sensitivity.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64 --repeats 3 --stability-method randomized --stability-sample-size 1024` | outputs/svd/analyze_k_sensitivity_summary.json |
| 2026-04-22T12:54:05+00:00 | train_probe | scaffold_ready | `scripts/train_probe.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64 --feature-family raw_img raw_blind difference projected_difference random_difference pca_img` | outputs/probes/train_probe_summary.json |
| 2026-04-22T12:54:18+00:00 | compare_features | scaffold_ready | `scripts/compare_features.py` | outputs/probes/compare_features_summary.json |
| 2026-04-22T12:58:07+00:00 | layerwise_analysis | scaffold_ready | `scripts/layerwise_analysis.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64` | outputs/svd/layerwise_analysis_summary.json |
| 2026-04-22T12:58:35+00:00 | layerwise_analysis | scaffold_ready | `scripts/layerwise_analysis.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 48 64` | outputs/svd/layerwise_analysis_summary.json |
| 2026-04-22T13:20:00+00:00 | analyze_stage_c_deep | ok | `scripts/analyze_stage_c_deep.py --layers 8 12 16 20 24 28 32 --k-grid 4 8 16 32 64 128 256 --focus-layers 20 24 28 32` | outputs/stage_c_deep/analyze_stage_c_deep_summary.json |
