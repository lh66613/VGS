#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_k_args, add_layer_args, resolve_k_grid, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_k import analyze_stage_k_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage K token-position robustness artifacts.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.set_defaults(layers=["16", "20", "24", "32"])
    parser.set_defaults(k_grid=["4", "64", "128", "256"])
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["last_prompt_token", "first_answer_prefill", "last_4_prompt_mean", "last_8_prompt_mean"],
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-root", default="outputs/stage_k_hidden")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--output-dir", default="outputs/stage_k_positions")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    payload = {
        "positions": args.positions,
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "predictions": args.predictions,
        "hidden_root": args.hidden_root,
        "split_dir": args.split_dir,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
        "max_iter": args.max_iter,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_k_positions(
                args.positions,
                resolve_layers(args),
                resolve_k_grid(args),
                args.predictions,
                args.hidden_root,
                args.split_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
                args.max_iter,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_k_positions_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_k_positions",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
