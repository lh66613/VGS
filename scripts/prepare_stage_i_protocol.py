#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.protocol import prepare_stage_i_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed splits and protocol-lock notes.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--output-dir", default="outputs/splits")
    parser.add_argument("--notes-dir", default="notes")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    payload = {
        "layers": resolve_layers(args),
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "output_dir": args.output_dir,
        "notes_dir": args.notes_dir,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
    }
    if not args.dry_run:
        payload.update(
            prepare_stage_i_protocol(
                args.predictions,
                args.hidden_states_dir,
                args.output_dir,
                args.notes_dir,
                args.seed,
                args.train_frac,
                args.val_frac,
                resolve_layers(args),
            )
        )

    summary_path = write_json(Path(args.output_dir) / "prepare_stage_i_protocol_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "prepare_stage_i_protocol",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
