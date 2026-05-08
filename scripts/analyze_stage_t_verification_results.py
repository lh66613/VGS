#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json
from vgs.pope import classify_outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate actual Stage T verification prompt outputs.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument(
        "--gate-assignments",
        default="outputs/stage_t_selective_correction/stage_t_verification_gate_assignments.csv",
    )
    parser.add_argument(
        "--verification-predictions",
        default="outputs/stage_t_selective_correction/stage_t_verification_predictions.jsonl",
    )
    parser.add_argument("--test-subset", default="adversarial")
    parser.add_argument(
        "--prompt-variant",
        default="legacy",
        help="Prompt variant label used for output filenames and audit columns.",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional fixed split directory; maps train/calibration/test to pope_train/val/test ids.",
    )
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = analyze_verification_results(
        predictions_path=args.predictions,
        gate_assignments_path=args.gate_assignments,
        verification_predictions_path=args.verification_predictions,
        test_subset=args.test_subset,
        split_dir=args.split_dir,
        prompt_variant=args.prompt_variant,
        output_dir=args.output_dir,
    )
    output_stem = _metrics_stem(args.prompt_variant)
    summary_path = write_json(
        Path(args.output_dir) / f"analyze_{output_stem}_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "analyze_stage_t_verification_results", summary_path, "ok")
    print(summary_path)


def analyze_verification_results(
    predictions_path: str | Path,
    gate_assignments_path: str | Path,
    verification_predictions_path: str | Path,
    test_subset: str,
    split_dir: str | Path | None,
    prompt_variant: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    all_predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    test_ids = _test_ids(all_predictions, test_subset, split_dir)
    predictions = {sample_id: all_predictions[sample_id] for sample_id in test_ids if sample_id in all_predictions}
    verification = {
        str(row["sample_id"]): row
        for row in read_jsonl(verification_predictions_path)
        if row.get("verification_parsed_prediction") in {"yes", "no"}
    }
    assignments = pd.read_csv(gate_assignments_path)
    rows: list[dict[str, Any]] = []
    original_quality = _quality([row["outcome"] for row in predictions.values()])
    group_cols = ["layer", "score", "target_trigger_rate_predicted_yes"]
    for key, group in assignments.groupby(group_cols, dropna=False):
        layer, score, target_rate = key
        triggered_ids = {str(item) for item in group["sample_id"].tolist()}
        available_ids = triggered_ids & set(verification)
        final_outcomes = []
        triggered_before = []
        triggered_after = []
        for sample_id, original in predictions.items():
            if sample_id in available_ids:
                parsed = verification[sample_id]["verification_parsed_prediction"]
                outcome = classify_outcome(parsed, str(original["label"]))
                final_outcomes.append(outcome)
                triggered_before.append(str(original["outcome"]))
                triggered_after.append(outcome)
            else:
                final_outcomes.append(str(original["outcome"]))
        after_quality = _quality(final_outcomes)
        before_trigger_counts = _counts(triggered_before)
        after_trigger_counts = _counts(triggered_after)
        rows.append(
            {
                "layer": layer,
                "score": score,
                "target_trigger_rate_predicted_yes": target_rate,
                "test_subset": test_subset,
                "prompt_variant": prompt_variant,
                "assigned_trigger_n": int(len(triggered_ids)),
                "verified_trigger_n": int(len(available_ids)),
                "missing_verification_n": int(len(triggered_ids - set(verification))),
                "triggered_fp_before": before_trigger_counts["FP"],
                "triggered_tp_before": before_trigger_counts["TP"],
                "triggered_fp_after": after_trigger_counts["FP"],
                "triggered_tp_after": after_trigger_counts["TP"],
                "accuracy_before": original_quality["accuracy"],
                "f1_before": original_quality["f1"],
                "fp_rate_before": original_quality["fp_rate"],
                "accuracy_after": after_quality["accuracy"],
                "f1_after": after_quality["f1"],
                "fp_rate_after": after_quality["fp_rate"],
                "actual_fp_reduction": (
                    (original_quality["FP"] - after_quality["FP"]) / original_quality["FP"]
                    if original_quality["FP"]
                    else math.nan
                ),
                "actual_tp_preserved": (
                    after_quality["TP"] / original_quality["TP"] if original_quality["TP"] else math.nan
                ),
            }
        )
    result_path = write_csv(Path(output_dir) / f"{_metrics_stem(prompt_variant)}.csv", rows, list(rows[0].keys()) if rows else [])
    return {
        "actual_verification_metrics_path": str(result_path),
        "num_rows": len(rows),
        "test_subset": test_subset,
        "split_dir": str(split_dir) if split_dir else "",
        "prompt_variant": prompt_variant,
        "num_test_predictions": len(predictions),
        "num_verification_predictions": len(verification),
    }


def _quality(outcomes: list[str]) -> dict[str, Any]:
    counts = _counts(outcomes)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    return {
        **counts,
        "accuracy": (tp + tn) / n if n else math.nan,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else math.nan,
        "fp_rate": fp / (fp + tn) if fp + tn else math.nan,
    }


def _counts(outcomes: list[str]) -> dict[str, int]:
    return {name: int(sum(outcome == name for outcome in outcomes)) for name in ["TP", "TN", "FP", "FN"]}


def _metrics_stem(prompt_variant: str) -> str:
    if prompt_variant == "legacy":
        return "stage_t_actual_verification_metrics"
    return f"stage_t_actual_verification_metrics_{prompt_variant}"


def _test_ids(
    predictions: dict[str, dict[str, Any]],
    test_subset: str,
    split_dir: str | Path | None,
) -> list[str]:
    if split_dir is not None:
        split_name = {"train": "train", "calibration": "val", "test": "test"}.get(test_subset)
        if split_name is not None:
            path = Path(split_dir) / f"pope_{split_name}_ids.json"
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return [str(sample_id) for sample_id in payload.get("sample_ids", [])]
    return [
        sample_id
        for sample_id, row in predictions.items()
        if str(row.get("subset", "")) == test_subset
    ]


if __name__ == "__main__":
    main()
