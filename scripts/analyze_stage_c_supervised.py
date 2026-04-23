#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.analysis import analyze_stage_c_supervised
from vgs.cli import add_common_args, add_k_args, add_layer_args, resolve_k_grid, resolve_layers
from vgs.io import append_experiment_log, write_json


def _parse_bands(values: list[str]) -> list[tuple[int, int]]:
    bands = []
    for value in values:
        if "-" not in value:
            raise ValueError(f"Band must use start-end form: {value}")
        start, end = value.split("-", 1)
        bands.append((int(start), int(end)))
    return bands


def _parse_ints(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item) for item in value.split(",") if item)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SVD backbone against supervised subspaces and cumulative/exclusion probes."
    )
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.add_argument("--focus-layers", nargs="*", default=["20", "24", "28", "32"])
    parser.add_argument("--exclusion-k-grid", nargs="*", default=["4", "16", "64", "256", "512", "1024"])
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["1-4", "17-32", "33-64", "65-128", "129-256", "257-512", "513-1024"],
    )
    parser.add_argument("--pls-components", type=int, default=8)
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--output-dir", default="outputs/stage_c_supervised")
    args = parser.parse_args()

    layers = resolve_layers(args)
    k_grid = resolve_k_grid(args)
    focus_layers = [int(layer) for layer in args.focus_layers]
    exclusion_k_grid = _parse_ints(args.exclusion_k_grid)
    bands = _parse_bands(args.bands)
    payload = {
        "layers": layers,
        "focus_layers": focus_layers,
        "k_grid": k_grid,
        "exclusion_k_grid": exclusion_k_grid,
        "bands": args.bands,
        "pls_components": args.pls_components,
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_c_supervised(
                layers,
                focus_layers,
                k_grid,
                exclusion_k_grid,
                bands,
                args.predictions,
                args.hidden_states_dir,
                args.svd_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
                args.pls_components,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_c_supervised_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_c_supervised",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
