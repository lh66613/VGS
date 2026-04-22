#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.smoke import create_smoke_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny fake artifacts for CPU pipeline smoke tests.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--predictions", default="outputs/predictions/smoke_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states_smoke")
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()

    payload = create_smoke_artifacts(
        resolve_layers(args),
        args.num_samples,
        args.hidden_dim,
        args.predictions,
        args.hidden_states_dir,
        args.seed,
    )
    summary_path = write_json(Path(args.output_dir) / "create_smoke_artifacts_summary.json", payload)
    append_experiment_log(args.log_path, "create_smoke_artifacts", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
