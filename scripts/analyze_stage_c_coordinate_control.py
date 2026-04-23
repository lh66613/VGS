#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.analysis import analyze_stage_c_coordinate_control
from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a same-split control for full-difference vs full-space rotated coordinates."
    )
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--output-dir", default="outputs/stage_c_coordinate_control")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    layers = resolve_layers(args)
    payload = {
        "layers": layers,
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
        "standardize": not args.no_standardize,
        "max_iter": args.max_iter,
        "C": args.C,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_c_coordinate_control(
                layers,
                args.predictions,
                args.hidden_states_dir,
                args.svd_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
                standardize=not args.no_standardize,
                max_iter=args.max_iter,
                c_value=args.C,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_c_coordinate_control_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_c_coordinate_control",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
