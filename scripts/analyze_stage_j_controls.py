#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_k_args, add_layer_args, resolve_k_grid, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_j import analyze_stage_j_controls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage J destructive shuffle and random-subspace controls."
    )
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.set_defaults(k_grid=["4", "64", "128", "256"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--output-dir", default="outputs/stage_j_controls")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--stability-sample-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    layers = resolve_layers(args)
    k_grid = resolve_k_grid(args)
    payload = {
        "layers": layers,
        "k_grid": k_grid,
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
        "split_dir": args.split_dir,
        "repeats": args.repeats,
        "random_repeats": args.random_repeats,
        "stability_sample_size": args.stability_sample_size,
        "max_iter": args.max_iter,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_j_controls(
                layers,
                k_grid,
                args.predictions,
                args.hidden_states_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
                args.repeats,
                None if args.stability_sample_size <= 0 else args.stability_sample_size,
                args.random_repeats,
                args.max_iter,
                args.split_dir,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_j_controls_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_j_controls",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
