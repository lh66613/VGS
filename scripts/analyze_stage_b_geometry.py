#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_b import analyze_stage_b_geometry


def _parse_ints(values: list[str]) -> list[int]:
    parsed = []
    for value in values:
        parsed.extend(int(item) for item in value.split(",") if item)
    return parsed


def _parse_bands(values: list[str]) -> list[tuple[int, int]]:
    bands = []
    for value in values:
        if "-" not in value:
            raise ValueError(f"Band must use start-end form: {value}")
        start, end = value.split("-", 1)
        bands.append((int(start), int(end)))
    return bands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Stage B matched/mismatched/blind correction geometry."
    )
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--top-k-grid", nargs="*", default=["4", "64", "256"])
    parser.add_argument("--tail-bands", nargs="*", default=["257-1024", "65-128", "129-256"])
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--reference-predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--reference-hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--output-dir", default="outputs/stage_b")
    args = parser.parse_args()

    layers = resolve_layers(args)
    top_k_grid = _parse_ints(args.top_k_grid)
    tail_bands = _parse_bands(args.tail_bands)
    payload = {
        "layers": layers,
        "top_k_grid": top_k_grid,
        "tail_bands": args.tail_bands,
        "condition_plan": args.condition_plan,
        "condition_hidden_dir": args.condition_hidden_dir,
        "svd_dir": args.svd_dir,
        "reference_predictions": args.reference_predictions,
        "reference_hidden_states_dir": args.reference_hidden_states_dir,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_b_geometry(
                layers,
                top_k_grid,
                tail_bands,
                args.condition_plan,
                args.condition_hidden_dir,
                args.svd_dir,
                args.reference_predictions,
                args.reference_hidden_states_dir,
                args.output_dir,
                args.plot_dir,
                args.seed,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "analyze_stage_b_geometry_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "analyze_stage_b_geometry",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
