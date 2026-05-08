#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.io import append_experiment_log, write_json
from vgs.stage_t import analyze_stage_t_selective_correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage T correction-geometry guided selective correction analysis."
    )
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["24"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction")
    parser.add_argument("--train-subset", default="random")
    parser.add_argument("--calibration-subset", default="popular")
    parser.add_argument("--test-subset", default="adversarial")
    parser.add_argument(
        "--split-policy",
        choices=["subset_transfer", "fixed_ids"],
        default="subset_transfer",
        help="Use Plan A subset transfer, or the repository fixed train/val/test split files.",
    )
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[4, 64])
    parser.add_argument("--pls-k", type=int, default=32)
    parser.add_argument("--random-dim", type=int, default=64)
    parser.add_argument("--trigger-rates", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--margin-scores", default=None)
    parser.add_argument("--external-predictions", default="outputs/stage_n_external_full/amber_predictions.jsonl")
    parser.add_argument("--external-hidden-states-dir", default="outputs/stage_n_external_full/amber_hidden")
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip optional AMBER/external transfer scoring even if artifacts exist.",
    )
    args = parser.parse_args()

    tail_band = _parse_band(args.tail_band)
    split_dir = _existing_path(args.split_dir) if args.split_policy == "fixed_ids" else None
    train_subset = "train" if args.split_policy == "fixed_ids" else args.train_subset
    calibration_subset = "calibration" if args.split_policy == "fixed_ids" else args.calibration_subset
    test_subset = "test" if args.split_policy == "fixed_ids" else args.test_subset
    external_predictions = None if args.no_external else _existing_path(args.external_predictions)
    external_hidden = None if args.no_external else _existing_path(args.external_hidden_states_dir)
    payload = {
        "layers": resolve_layers(args),
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "output_dir": args.output_dir,
        "split_policy": args.split_policy,
        "split_dir": split_dir,
        "train_subset": train_subset,
        "calibration_subset": calibration_subset,
        "test_subset": test_subset,
        "tail_band": args.tail_band,
        "top_k_grid": args.top_k_grid,
        "pls_k": args.pls_k,
        "random_dim": args.random_dim,
        "trigger_rates": args.trigger_rates,
        "max_iter": args.max_iter,
        "margin_scores": args.margin_scores,
        "external_predictions": external_predictions,
        "external_hidden_states_dir": external_hidden,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_t_selective_correction(
                layers=resolve_layers(args),
                predictions_path=args.predictions,
                hidden_states_dir=args.hidden_states_dir,
                output_dir=args.output_dir,
                train_subset=train_subset,
                calibration_subset=calibration_subset,
                test_subset=test_subset,
                tail_band=tail_band,
                top_k_grid=args.top_k_grid,
                pls_k=args.pls_k,
                random_dim=args.random_dim,
                trigger_rates=args.trigger_rates,
                seed=args.seed,
                max_iter=args.max_iter,
                margin_scores_path=args.margin_scores,
                split_dir=split_dir,
                external_predictions_path=external_predictions,
                external_hidden_states_dir=external_hidden,
            )
        )

    summary_path = write_json(
        Path(args.output_dir) / "analyze_stage_t_selective_correction_summary.json",
        payload,
    )
    append_experiment_log(
        args.log_path,
        "analyze_stage_t_selective_correction",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


def _parse_band(text: str) -> tuple[int, int]:
    try:
        start, end = text.split("-", 1)
        return int(start), int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected band like 257-1024") from exc


def _existing_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if Path(path).exists() else None


if __name__ == "__main__":
    main()
