#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json


DEFAULT_SCORES = [
    "pls32_probe",
    "tail_257_1024_probe",
    "tail_257_1024_energy",
    "full_probe",
    "top_4_probe",
    "random64_probe",
    "margin_probe",
    "low_margin_probe",
    "margin_plus_pls32_probe",
    "margin_plus_tail_257_1024_probe",
    "margin_plus_full_probe",
    "low_margin_plus_pls32_probe",
    "low_margin_plus_tail_257_1024_probe",
    "low_margin_plus_full_probe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage T gated VCD/ICD outputs.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument(
        "--gate-assignments",
        default="outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_gate_assignments.csv",
    )
    parser.add_argument(
        "--vcd-predictions",
        default="outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_vcd_blur.jsonl",
    )
    parser.add_argument("--operator", default="vcd_blur")
    parser.add_argument("--test-subset", default="test")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument("--scores", nargs="*", default=DEFAULT_SCORES)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = analyze_vcd_results(
        predictions_path=args.predictions,
        gate_assignments_path=args.gate_assignments,
        vcd_predictions_path=args.vcd_predictions,
        operator=args.operator,
        test_subset=args.test_subset,
        split_dir=args.split_dir,
        target_rates=args.target_rates,
        selected_scores=args.scores,
        random_repeats=args.random_repeats,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary_path = write_json(
        Path(args.output_dir) / f"analyze_stage_t_vcd_results_{args.operator}_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "analyze_stage_t_vcd_results", summary_path, "ok")
    print(summary_path)


def analyze_vcd_results(
    predictions_path: str | Path,
    gate_assignments_path: str | Path,
    vcd_predictions_path: str | Path,
    operator: str,
    test_subset: str,
    split_dir: str | Path | None,
    target_rates: list[float],
    selected_scores: list[str],
    random_repeats: int,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    all_predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    test_ids = _test_ids(all_predictions, test_subset, split_dir)
    predictions = {sample_id: all_predictions[sample_id] for sample_id in test_ids if sample_id in all_predictions}
    predicted_yes_ids = [
        sample_id
        for sample_id, row in predictions.items()
        if str(row.get("parsed_prediction", "")).lower() == "yes"
        and str(row.get("outcome", "")) in {"FP", "TP"}
    ]
    vcd = {
        str(row["sample_id"]): row
        for row in read_jsonl(vcd_predictions_path)
        if row.get("vcd_parsed_prediction") in {"yes", "no"}
    }
    assignments = pd.read_csv(gate_assignments_path)
    target_keys = {_rate_key(rate) for rate in target_rates}
    assignments = assignments[
        assignments["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_keys)
    ].copy()
    available_scores = set(assignments["score"].dropna().astype(str))
    wanted_scores = [score for score in selected_scores if score in available_scores]
    missing_scores = [score for score in selected_scores if score not in available_scores]

    original_outcomes = {sample_id: str(row["outcome"]) for sample_id, row in predictions.items()}
    original_quality = _quality(list(original_outcomes.values()))
    rows: list[dict[str, Any]] = []

    always_ids = set(predicted_yes_ids) & set(vcd)
    always_metrics = _method_row(
        method="Always VCD/ICD",
        gate_family="always",
        score="always_predicted_yes",
        matched_score="",
        operator=operator,
        predictions=predictions,
        original_outcomes=original_outcomes,
        original_quality=original_quality,
        vcd=vcd,
        triggered_ids=always_ids,
        predicted_yes_n=len(predicted_yes_ids),
        target_rate=1.0,
        aggregation="deterministic",
    )
    rows.append(
        _original_row(
            operator=operator,
            original_quality=original_quality,
            predicted_yes_n=len(predicted_yes_ids),
        )
    )
    rows.append(always_metrics)
    always_fp_reduction = float(always_metrics["fp_reduction"])

    group_cols = ["layer", "score", "target_trigger_rate_predicted_yes"]
    for key, group in assignments.groupby(group_cols, dropna=False):
        layer, score, target_rate = key
        score = str(score)
        if score not in wanted_scores:
            continue
        triggered_ids = {str(item) for item in group["sample_id"].tolist()} & set(vcd)
        method_row = _method_row(
            method=_method_name(score),
            gate_family=_gate_family(score),
            score=score,
            matched_score="",
            operator=operator,
            predictions=predictions,
            original_outcomes=original_outcomes,
            original_quality=original_quality,
            vcd=vcd,
            triggered_ids=triggered_ids,
            predicted_yes_n=len(predicted_yes_ids),
            target_rate=float(target_rate),
            aggregation="deterministic",
            layer=layer,
        )
        method_row["gap_to_always_vcd_fp_reduction"] = always_fp_reduction - float(method_row["fp_reduction"])
        rows.append(method_row)
        rows.extend(
            _random_rows(
                layer=int(layer),
                matched_score=score,
                target_rate=float(target_rate),
                n_trigger=len(group),
                operator=operator,
                predictions=predictions,
                original_outcomes=original_outcomes,
                original_quality=original_quality,
                vcd=vcd,
                predicted_yes_ids=predicted_yes_ids,
                predicted_yes_n=len(predicted_yes_ids),
                always_fp_reduction=always_fp_reduction,
                random_repeats=random_repeats,
                seed=seed,
            )
        )

    for row in rows:
        row.setdefault("gap_to_always_vcd_fp_reduction", always_fp_reduction - float(row["fp_reduction"]))

    rows = sorted(
        rows,
        key=lambda item: (
            int(item.get("layer") or 0),
            float(item.get("target_trigger_rate_predicted_yes") or 0.0),
            str(item.get("method", "")),
            str(item.get("score", "")),
            str(item.get("matched_score", "")),
        ),
    )
    output_root = Path(output_dir)
    metrics_path = write_csv(
        output_root / f"stage_t_vcd_metrics_{operator}.csv",
        rows,
        _fieldnames(rows),
    )
    return {
        "operator": operator,
        "vcd_predictions_path": str(vcd_predictions_path),
        "metrics_path": str(metrics_path),
        "num_rows": len(rows),
        "num_test_predictions": len(predictions),
        "num_predicted_yes": len(predicted_yes_ids),
        "num_vcd_predictions": len(vcd),
        "available_selected_scores": wanted_scores,
        "missing_selected_scores": missing_scores,
        "random_repeats": random_repeats,
    }


def _method_row(
    method: str,
    gate_family: str,
    score: str,
    matched_score: str,
    operator: str,
    predictions: dict[str, dict[str, Any]],
    original_outcomes: dict[str, str],
    original_quality: dict[str, Any],
    vcd: dict[str, dict[str, Any]],
    triggered_ids: set[str],
    predicted_yes_n: int,
    target_rate: float,
    aggregation: str,
    layer: int | str = "",
) -> dict[str, Any]:
    final_outcomes = []
    before_trigger_counts = []
    after_trigger_counts = []
    for sample_id, original in predictions.items():
        if sample_id in triggered_ids:
            outcome = str(vcd[sample_id]["vcd_outcome"])
            final_outcomes.append(outcome)
            before_trigger_counts.append(original_outcomes[sample_id])
            after_trigger_counts.append(outcome)
        else:
            final_outcomes.append(original_outcomes[sample_id])
    after_quality = _quality(final_outcomes)
    before_counts = _counts(before_trigger_counts)
    after_counts = _counts(after_trigger_counts)
    trigger_n = len(triggered_ids)
    fp_reduced_n = original_quality["FP"] - after_quality["FP"]
    fp_reduction = fp_reduced_n / original_quality["FP"] if original_quality["FP"] else math.nan
    trigger_rate = trigger_n / predicted_yes_n if predicted_yes_n else math.nan
    return {
        "layer": layer,
        "operator": operator,
        "method": method,
        "gate_family": gate_family,
        "score": score,
        "matched_score": matched_score,
        "target_trigger_rate_predicted_yes": target_rate,
        "aggregation": aggregation,
        "trigger_n": trigger_n,
        "predicted_yes_n": predicted_yes_n,
        "trigger_rate_predicted_yes": trigger_rate,
        "extra_compute_fraction_vs_always": trigger_rate,
        "compute_saved_vs_always": 1.0 - trigger_rate if not math.isnan(trigger_rate) else math.nan,
        "triggered_fp_before": before_counts["FP"],
        "triggered_tp_before": before_counts["TP"],
        "triggered_fp_after": after_counts["FP"],
        "triggered_tp_after": after_counts["TP"],
        "fp_reduced_n": fp_reduced_n,
        "fp_reduction": fp_reduction,
        "tp_preserved": after_quality["TP"] / original_quality["TP"] if original_quality["TP"] else math.nan,
        "fp_reduction_per_trigger": fp_reduced_n / trigger_n if trigger_n else math.nan,
        "accuracy_before": original_quality["accuracy"],
        "f1_before": original_quality["f1"],
        "fp_rate_before": original_quality["fp_rate"],
        "accuracy_after": after_quality["accuracy"],
        "f1_after": after_quality["f1"],
        "fp_rate_after": after_quality["fp_rate"],
        "random_repeats": "",
        "metric_std_fp_reduction": "",
        "metric_std_tp_preserved": "",
        "metric_p05_fp_reduction": "",
        "metric_p95_fp_reduction": "",
    }


def _random_rows(
    layer: int,
    matched_score: str,
    target_rate: float,
    n_trigger: int,
    operator: str,
    predictions: dict[str, dict[str, Any]],
    original_outcomes: dict[str, str],
    original_quality: dict[str, Any],
    vcd: dict[str, dict[str, Any]],
    predicted_yes_ids: list[str],
    predicted_yes_n: int,
    always_fp_reduction: float,
    random_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = sorted(set(predicted_yes_ids) & set(vcd))
    if not candidates or n_trigger <= 0:
        return []
    rng = np.random.default_rng(seed + layer * 10007 + _stable_int(matched_score) + _rate_key(target_rate))
    metric_rows = []
    for _ in range(random_repeats):
        size = min(n_trigger, len(candidates))
        triggered = set(rng.choice(candidates, size=size, replace=False).tolist())
        metric_rows.append(
            _method_row(
                method="Random-gated VCD/ICD",
                gate_family="same_trigger_random",
                score="same_trigger_random",
                matched_score=matched_score,
                operator=operator,
                predictions=predictions,
                original_outcomes=original_outcomes,
                original_quality=original_quality,
                vcd=vcd,
                triggered_ids=triggered,
                predicted_yes_n=predicted_yes_n,
                target_rate=target_rate,
                aggregation="repeat",
                layer=layer,
            )
        )
    return [_aggregate_random_rows(metric_rows, random_repeats, always_fp_reduction)]


def _aggregate_random_rows(rows: list[dict[str, Any]], random_repeats: int, always_fp_reduction: float) -> dict[str, Any]:
    first = rows[0]
    numeric = [
        "trigger_n",
        "trigger_rate_predicted_yes",
        "extra_compute_fraction_vs_always",
        "compute_saved_vs_always",
        "triggered_fp_before",
        "triggered_tp_before",
        "triggered_fp_after",
        "triggered_tp_after",
        "fp_reduced_n",
        "fp_reduction",
        "tp_preserved",
        "fp_reduction_per_trigger",
        "accuracy_after",
        "f1_after",
        "fp_rate_after",
    ]
    out = {
        key: first[key]
        for key in [
            "layer",
            "operator",
            "method",
            "gate_family",
            "score",
            "matched_score",
            "target_trigger_rate_predicted_yes",
            "accuracy_before",
            "f1_before",
            "fp_rate_before",
        ]
    }
    out["aggregation"] = "mean"
    for key in numeric:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        out[key] = float(values.mean())
    fp_values = np.array([float(row["fp_reduction"]) for row in rows], dtype=float)
    tp_values = np.array([float(row["tp_preserved"]) for row in rows], dtype=float)
    out["random_repeats"] = random_repeats
    out["metric_std_fp_reduction"] = float(fp_values.std())
    out["metric_std_tp_preserved"] = float(tp_values.std())
    out["metric_p05_fp_reduction"] = float(np.quantile(fp_values, 0.05))
    out["metric_p95_fp_reduction"] = float(np.quantile(fp_values, 0.95))
    out["gap_to_always_vcd_fp_reduction"] = always_fp_reduction - float(out["fp_reduction"])
    return out


def _original_row(
    operator: str,
    original_quality: dict[str, Any],
    predicted_yes_n: int,
) -> dict[str, Any]:
    return {
        "layer": "",
        "operator": operator,
        "method": "Original",
        "gate_family": "none",
        "score": "",
        "matched_score": "",
        "target_trigger_rate_predicted_yes": 0.0,
        "aggregation": "deterministic",
        "trigger_n": 0,
        "predicted_yes_n": predicted_yes_n,
        "trigger_rate_predicted_yes": 0.0,
        "extra_compute_fraction_vs_always": 0.0,
        "compute_saved_vs_always": 1.0,
        "triggered_fp_before": 0,
        "triggered_tp_before": 0,
        "triggered_fp_after": 0,
        "triggered_tp_after": 0,
        "fp_reduced_n": 0,
        "fp_reduction": 0.0,
        "tp_preserved": 1.0,
        "fp_reduction_per_trigger": math.nan,
        "accuracy_before": original_quality["accuracy"],
        "f1_before": original_quality["f1"],
        "fp_rate_before": original_quality["fp_rate"],
        "accuracy_after": original_quality["accuracy"],
        "f1_after": original_quality["f1"],
        "fp_rate_after": original_quality["fp_rate"],
        "random_repeats": "",
        "metric_std_fp_reduction": "",
        "metric_std_tp_preserved": "",
        "metric_p05_fp_reduction": "",
        "metric_p95_fp_reduction": "",
        "gap_to_always_vcd_fp_reduction": "",
    }


def _quality(outcomes: list[str]) -> dict[str, Any]:
    counts = _counts(outcomes)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    unknown = counts["unknown"]
    n = tp + tn + fp + fn + unknown
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    return {
        **counts,
        "accuracy": (tp + tn) / n if n else math.nan,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else math.nan,
        "fp_rate": fp / (fp + tn) if fp + tn else math.nan,
    }


def _counts(outcomes: list[str]) -> dict[str, int]:
    names = ["TP", "TN", "FP", "FN", "unknown"]
    return {name: int(sum(outcome == name for outcome in outcomes)) for name in names}


def _test_ids(
    predictions: dict[str, dict[str, Any]],
    test_subset: str,
    split_dir: str | Path | None,
) -> list[str]:
    if split_dir:
        split_name = {"train": "train", "calibration": "val", "test": "test"}.get(test_subset)
        if split_name:
            path = Path(split_dir) / f"pope_{split_name}_ids.json"
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return [str(sample_id) for sample_id in payload.get("sample_ids", [])]
    return [
        sample_id
        for sample_id, row in predictions.items()
        if str(row.get("subset", "")) == test_subset
    ]


def _method_name(score: str) -> str:
    if score == "pls32_probe":
        return "PLS-gated VCD/ICD"
    if score == "tail_257_1024_probe":
        return "Tail-gated VCD/ICD"
    if score == "tail_257_1024_energy":
        return "Tail-energy-gated VCD/ICD"
    if score == "full_probe":
        return "FullD-gated VCD/ICD"
    if score == "top_4_probe":
        return "Top-4-gated VCD/ICD"
    if score == "top_64_probe":
        return "Top-64-gated VCD/ICD"
    if score == "random64_probe":
        return "Random-subspace-gated VCD/ICD"
    if score == "margin_probe":
        return "Margin-gated VCD/ICD"
    if score.startswith("margin_plus_"):
        return "Margin+Geometry-gated VCD/ICD"
    if score == "low_margin_probe":
        return "Low-margin-gated VCD/ICD"
    if score.startswith("low_margin_plus_"):
        return "Low-margin+Geometry-gated VCD/ICD"
    return f"{score}-gated VCD/ICD"


def _gate_family(score: str) -> str:
    if score.startswith("low_margin_plus_"):
        return "low_margin_plus_geometry"
    if score.startswith("margin_plus_"):
        return "margin_plus_geometry"
    if score == "low_margin_probe":
        return "low_margin"
    if score == "margin_probe":
        return "margin"
    if score.startswith("top_"):
        return "top_variance"
    if score.startswith("tail_") or score in {"pls32_probe", "full_probe"}:
        return "geometry"
    if score == "random64_probe":
        return "random_subspace"
    return "score_gate"


def _stable_int(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
