#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args
from vgs.io import append_experiment_log, write_json
from vgs.stage_b import prepare_stage_b_condition_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare matched/random/adversarial image conditions for Stage B."
    )
    add_common_args(parser)
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--outcomes", nargs="*", default=["FP", "TN"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-outcome", type=int, default=256)
    parser.add_argument("--allow-missing-adversarial", action="store_true")
    parser.add_argument("--output-dir", default="outputs/stage_b")
    args = parser.parse_args()

    payload = {
        "predictions": args.predictions,
        "outcomes": args.outcomes,
        "max_samples": args.max_samples,
        "max_samples_per_outcome": args.max_samples_per_outcome,
        "require_adversarial": not args.allow_missing_adversarial,
    }
    if not args.dry_run:
        payload.update(
            prepare_stage_b_condition_plan(
                args.predictions,
                args.output_dir,
                args.seed,
                args.outcomes,
                max_samples=args.max_samples,
                max_samples_per_outcome=args.max_samples_per_outcome,
                require_adversarial=not args.allow_missing_adversarial,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "prepare_stage_b_conditions_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "prepare_stage_b_conditions",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
