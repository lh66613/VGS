#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_n import analyze_external_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage N external transfer using POPE SVD bases.")
    add_layer_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.add_argument("--predictions", default="outputs/stage_n_external/amber_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/stage_n_external/amber_hidden")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--pope-predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--pope-hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument("--evidence-methods", nargs="*", default=["pls_fp_tn", "fisher_fp_tn"])
    parser.add_argument("--evidence-k-grid", nargs="*", default=["4", "8", "16", "32", "64"])
    parser.add_argument("--k-grid", nargs="*", default=["4", "64", "256"])
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--output-dir", default="outputs/stage_n_external")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    k_grid = [int(item) for value in args.k_grid for item in value.split(",") if item]
    evidence_k_grid = [int(item) for value in args.evidence_k_grid for item in value.split(",") if item]
    payload = {
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
        "pope_predictions": args.pope_predictions,
        "pope_hidden_states_dir": args.pope_hidden_states_dir,
        "split_dir": args.split_dir,
        "condition_hidden_dir": args.condition_hidden_dir,
        "condition_plan": args.condition_plan,
        "evidence_methods": args.evidence_methods,
        "evidence_k_grid": evidence_k_grid,
        "layers": resolve_layers(args),
        "k_grid": k_grid,
        "tail_band": args.tail_band,
        "seed": args.seed,
        "ridge": args.ridge,
    }
    payload.update(
        analyze_external_transfer(
            args.predictions,
            args.hidden_states_dir,
            args.svd_dir,
            args.pope_predictions,
            args.pope_hidden_states_dir,
            args.split_dir,
            args.output_dir,
            args.plot_dir,
            resolve_layers(args),
            k_grid,
            (tail_start, tail_end),
            args.condition_hidden_dir,
            args.condition_plan,
            args.evidence_methods,
            evidence_k_grid,
            args.seed,
            args.ridge,
        )
    )
    summary_path = write_json(Path(args.output_dir) / "analyze_stage_n_external_transfer_summary.json", payload)
    append_experiment_log(args.log_path, "analyze_stage_n_external_transfer", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
