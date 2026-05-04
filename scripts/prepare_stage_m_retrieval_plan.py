#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_m import prepare_stage_m_retrieval_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage M train-bank retrieval plan.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--memory-bank", default="outputs/stage_m_local_rescue/memory_bank_train.pt")
    parser.add_argument("--target-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--outcomes", nargs="+", default=["FP", "TN", "TP"])
    parser.add_argument("--max-targets-per-outcome", type=int, default=64)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--max-svd-coords", type=int, default=1024)
    parser.add_argument("--exclude-same-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/stage_m_local_rescue")
    args = parser.parse_args()

    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    payload = {
        "layers": resolve_layers(args),
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
        "split_dir": args.split_dir,
        "memory_bank": args.memory_bank,
        "target_split": args.target_split,
        "outcomes": args.outcomes,
        "max_targets_per_outcome": args.max_targets_per_outcome,
        "k_neighbors": args.k_neighbors,
        "tail_band": args.tail_band,
        "max_svd_coords": args.max_svd_coords,
        "exclude_same_image": args.exclude_same_image,
    }
    if not args.dry_run:
        payload.update(
            prepare_stage_m_retrieval_plan(
                resolve_layers(args),
                args.predictions,
                args.hidden_states_dir,
                args.svd_dir,
                args.split_dir,
                args.memory_bank,
                args.output_dir,
                args.target_split,
                args.outcomes,
                args.max_targets_per_outcome,
                args.k_neighbors,
                (tail_start, tail_end),
                args.max_svd_coords,
                args.exclude_same_image,
                args.seed,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "prepare_stage_m_retrieval_plan_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "prepare_stage_m_retrieval_plan",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
