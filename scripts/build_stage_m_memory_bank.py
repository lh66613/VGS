#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_m import build_stage_m_memory_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage M train-only local rescue memory bank.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["20", "24", "32"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--max-svd-coords", type=int, default=1024)
    parser.add_argument("--output-dir", default="outputs/stage_m_local_rescue")
    args = parser.parse_args()

    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    payload = {
        "layers": resolve_layers(args),
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
        "split_dir": args.split_dir,
        "tail_band": args.tail_band,
        "max_svd_coords": args.max_svd_coords,
    }
    if not args.dry_run:
        payload.update(
            build_stage_m_memory_bank(
                resolve_layers(args),
                args.predictions,
                args.hidden_states_dir,
                args.svd_dir,
                args.split_dir,
                args.output_dir,
                (tail_start, tail_end),
                args.max_svd_coords,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "build_stage_m_memory_bank_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "build_stage_m_memory_bank",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
