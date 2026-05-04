#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_json
from vgs.stage_n import prepare_amber_discriminative_plan, write_external_benchmark_choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AMBER discriminative plan for Stage N.")
    parser.add_argument("--query", default="data/amber/data/query/query_discriminative.json")
    parser.add_argument("--annotation", default="data/amber/data/annotations.json")
    parser.add_argument("--images-dir", default="data/amber/image")
    parser.add_argument("--dimension", default="discriminative")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-dimension-label", type=int, default=None)
    parser.add_argument("--dimensions", nargs="*", default=None, choices=["existence", "attribute", "relation", "discriminative"])
    parser.add_argument("--output-dir", default="outputs/stage_n_external")
    parser.add_argument("--choice-note", default="notes/external_benchmark_choice.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    choice_path = write_external_benchmark_choice(args.choice_note)
    payload = {
        "query": args.query,
        "annotation": args.annotation,
        "images_dir": args.images_dir,
        "dimension": args.dimension,
        "max_samples": args.max_samples,
        "max_per_dimension_label": args.max_per_dimension_label,
        "dimensions": args.dimensions,
        "choice_note": str(choice_path),
    }
    if not args.dry_run:
        payload.update(
            prepare_amber_discriminative_plan(
                args.query,
                args.annotation,
                args.images_dir,
                args.output_dir,
                args.dimension,
                args.max_samples,
                args.max_per_dimension_label,
                args.dimensions,
                13,
            )
        )
    summary_path = write_json(Path(args.output_dir) / "prepare_stage_n_amber_summary.json", payload)
    append_experiment_log(args.log_path, "prepare_stage_n_amber", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
