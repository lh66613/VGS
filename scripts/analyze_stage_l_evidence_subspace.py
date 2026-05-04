#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_k_args, add_layer_args, resolve_k_grid, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_l import analyze_stage_l_evidence_subspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage L evidence-specific subspace extraction.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.set_defaults(k_grid=["4", "8", "16", "32", "64"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--output-dir", default="outputs/stage_l_evidence_subspace")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    payload = {
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "condition_hidden_dir": args.condition_hidden_dir,
        "condition_plan": args.condition_plan,
        "split_dir": args.split_dir,
        "ridge": args.ridge,
        "max_iter": args.max_iter,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_l_evidence_subspace(
                resolve_layers(args),
                resolve_k_grid(args),
                args.predictions,
                args.hidden_states_dir,
                args.condition_hidden_dir,
                args.condition_plan,
                args.split_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
                args.ridge,
                args.max_iter,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_l_evidence_subspace_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_l_evidence_subspace",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
