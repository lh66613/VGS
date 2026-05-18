#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json
from vgs.pope import classify_outcome


DEFAULT_SUBSPACES = [
    "full",
    "top4",
    "top16",
    "band5_16",
    "tail257_1024",
    "top4_complement",
    "random12",
    "random4_complement",
    "random_tail_dim",
]
DEFAULT_ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: offline logit-level subspace-filtered VCD/ICD. "
            "Uses m' = m_base + alpha * Delta m_subspace for POPE yes/no."
        )
    )
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--split-policy",
        choices=["subset_transfer", "fixed_ids"],
        default="fixed_ids",
    )
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--calibration-subset", default="popular")
    parser.add_argument("--test-subset", default="adversarial")
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/stage2_subspace_vcd")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = analyze_stage2(
        operator_geometry_path=args.operator_geometry,
        predictions_path=args.predictions,
        margin_scores_path=args.margin_scores,
        subspaces=args.subspaces,
        alphas=args.alphas,
        split_policy=args.split_policy,
        split_dir=args.split_dir,
        calibration_subset=args.calibration_subset,
        test_subset=args.test_subset,
        min_tp_preserved=args.min_tp_preserved,
        output_dir=args.output_dir,
    )
    summary_path = write_json(Path(args.output_dir) / "analyze_mechanism_mitigation_stage2_summary.json", result)
    append_experiment_log(args.log_path, "analyze_mechanism_mitigation_stage2", summary_path, "ok")
    print(summary_path)


def analyze_stage2(
    operator_geometry_path: str | Path,
    predictions_path: str | Path,
    margin_scores_path: str | Path | None,
    subspaces: list[str],
    alphas: list[float],
    split_policy: str,
    split_dir: str | Path | None,
    calibration_subset: str,
    test_subset: str,
    min_tp_preserved: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    margins = _load_margins(margin_scores_path)
    split_map = _load_split_map(split_dir) if split_policy == "fixed_ids" else {}
    calibration_name = "calibration" if split_policy == "fixed_ids" else calibration_subset
    test_name = "test" if split_policy == "fixed_ids" else test_subset
    df = pd.read_csv(operator_geometry_path)
    df["split"] = [
        split_map.get(str(row.sample_id), str(row.source_subset))
        for row in df.itertuples(index=False)
    ]
    df["base_no_minus_yes_logit"] = [
        _base_no_minus_yes(str(row.sample_id), row, margins)
        for row in df.itertuples(index=False)
    ]
    present_subspaces = [name for name in subspaces if f"dmargin_no_minus_yes_{name}" in df.columns]
    if not present_subspaces:
        raise ValueError("No requested subspace dmargin columns were present.")

    base_rows = []
    for split in sorted(df["split"].dropna().unique()):
        ids = sorted(set(df[df["split"] == split]["sample_id"].astype(str)))
        base_rows.append(_base_row(split, predictions, ids))

    sweep_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for (operator, layer, split), group in df.groupby(["operator", "layer", "split"], dropna=False):
        ids = [str(item) for item in group["sample_id"].tolist()]
        labels = [str(predictions[sample_id]["label"]) for sample_id in ids if sample_id in predictions]
        original_outcomes = [str(predictions[sample_id]["outcome"]) for sample_id in ids if sample_id in predictions]
        original_predictions = [
            str(predictions[sample_id].get("parsed_prediction", ""))
            for sample_id in ids
            if sample_id in predictions
        ]
        if not labels:
            continue
        for subspace in present_subspaces:
            dmargin = group[f"dmargin_no_minus_yes_{subspace}"].to_numpy(dtype=float)
            base_margin = group["base_no_minus_yes_logit"].to_numpy(dtype=float)
            for alpha in alphas:
                adjusted = base_margin + alpha * dmargin
                final_predictions = ["no" if value >= 0 else "yes" for value in adjusted]
                final_outcomes = [
                    classify_outcome(prediction, label)
                    for prediction, label in zip(final_predictions, labels)
                ]
                sample_rows.extend(
                    _sample_rows(
                        operator=str(operator),
                        layer=int(layer),
                        split=str(split),
                        subspace=subspace,
                        alpha=float(alpha),
                        ids=ids,
                        labels=labels,
                        original_predictions=original_predictions,
                        original_outcomes=original_outcomes,
                        final_predictions=final_predictions,
                        final_outcomes=final_outcomes,
                        base_margin=base_margin,
                        dmargin=dmargin,
                        adjusted_margin=adjusted,
                    )
                )
                sweep_rows.append(
                    _metric_row(
                        operator=str(operator),
                        layer=int(layer),
                        split=str(split),
                        subspace=subspace,
                        alpha=float(alpha),
                        ids=ids,
                        labels=labels,
                        original_outcomes=original_outcomes,
                        final_predictions=final_predictions,
                        final_outcomes=final_outcomes,
                    )
                )

    calibrated_rows = _calibrated_rows(sweep_rows, calibration_name, test_name, min_tp_preserved)
    pareto_rows = _pareto_rows([row for row in sweep_rows if row["split"] == test_name])
    band_rows = _band_comparison_rows([row for row in sweep_rows if row["split"] == test_name])

    output_root = Path(output_dir)
    base_path = write_csv(output_root / "base_results.csv", base_rows, _fieldnames(base_rows))
    sweep_path = write_csv(output_root / "alpha_sweep.csv", sweep_rows, _fieldnames(sweep_rows))
    sample_path = write_csv(output_root / "sample_predictions.csv", sample_rows, _fieldnames(sample_rows))
    results_path = write_csv(output_root / "subspace_vcd_results.csv", calibrated_rows, _fieldnames(calibrated_rows))
    pareto_path = write_csv(output_root / "pareto_frontier.csv", pareto_rows, _fieldnames(pareto_rows))
    band_path = write_csv(output_root / "band_comparison.csv", band_rows, _fieldnames(band_rows))
    figure_paths = _write_figures(output_root / "figures", [row for row in sweep_rows if row["split"] == test_name])
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "predictions_path": str(predictions_path),
        "margin_scores_path": str(margin_scores_path) if margin_scores_path else "",
        "split_policy": split_policy,
        "split_dir": str(split_dir) if split_dir else "",
        "calibration_split": calibration_name,
        "test_split": test_name,
        "subspaces": present_subspaces,
        "alphas": alphas,
        "base_results_path": str(base_path),
        "alpha_sweep_path": str(sweep_path),
        "sample_predictions_path": str(sample_path),
        "subspace_vcd_results_path": str(results_path),
        "pareto_frontier_path": str(pareto_path),
        "band_comparison_path": str(band_path),
        "figure_paths": figure_paths,
        "num_sweep_rows": len(sweep_rows),
        "num_sample_rows": len(sample_rows),
    }


def _metric_row(
    operator: str,
    layer: int,
    split: str,
    subspace: str,
    alpha: float,
    ids: list[str],
    labels: list[str],
    original_outcomes: list[str],
    final_predictions: list[str],
    final_outcomes: list[str],
) -> dict[str, Any]:
    original_counts = _counts(original_outcomes)
    final_counts = _counts(final_outcomes)
    original_accuracy = _accuracy(original_counts)
    final_accuracy = _accuracy(final_counts)
    original_fp = original_counts["FP"]
    original_tp = original_counts["TP"]
    original_tn = original_counts["TN"]
    fp_fixed = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "FP" and after == "TN")
    tp_kept = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "TP" and after == "TP")
    tn_kept = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "TN" and after == "TN")
    tp_damaged = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "TP" and after != "TP")
    yes_rate = sum(1 for item in final_predictions if item == "yes") / len(final_predictions) if final_predictions else math.nan
    no_rate = sum(1 for item in final_predictions if item == "no") / len(final_predictions) if final_predictions else math.nan
    return {
        "operator": operator,
        "layer": layer,
        "split": split,
        "method": _method_name(operator, subspace),
        "subspace": subspace,
        "alpha": alpha,
        "n": len(labels),
        "original_tp": original_counts["TP"],
        "original_tn": original_counts["TN"],
        "original_fp": original_counts["FP"],
        "original_fn": original_counts["FN"],
        "after_tp": final_counts["TP"],
        "after_tn": final_counts["TN"],
        "after_fp": final_counts["FP"],
        "after_fn": final_counts["FN"],
        "fp_reduced_n": fp_fixed,
        "tp_damaged_n": tp_damaged,
        "fp_reduction": fp_fixed / original_fp if original_fp else math.nan,
        "tp_preserved": tp_kept / original_tp if original_tp else math.nan,
        "tn_preserved": tn_kept / original_tn if original_tn else math.nan,
        "accuracy_before": original_accuracy,
        "accuracy_after": final_accuracy,
        "accuracy_delta": final_accuracy - original_accuracy,
        "yes_rate_after": yes_rate,
        "no_bias": no_rate,
        "fp_reduction_per_tp_damage": fp_fixed / tp_damaged if tp_damaged else math.inf if fp_fixed else math.nan,
    }


def _sample_rows(
    operator: str,
    layer: int,
    split: str,
    subspace: str,
    alpha: float,
    ids: list[str],
    labels: list[str],
    original_predictions: list[str],
    original_outcomes: list[str],
    final_predictions: list[str],
    final_outcomes: list[str],
    base_margin: np.ndarray,
    dmargin: np.ndarray,
    adjusted_margin: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(ids[: len(labels)]):
        rows.append(
            {
                "sample_id": sample_id,
                "operator": operator,
                "layer": layer,
                "split": split,
                "method": _method_name(operator, subspace),
                "subspace": subspace,
                "alpha": alpha,
                "label": labels[idx],
                "original_prediction": original_predictions[idx],
                "original_outcome": original_outcomes[idx],
                "final_prediction": final_predictions[idx],
                "final_outcome": final_outcomes[idx],
                "base_no_minus_yes_logit": float(base_margin[idx]),
                "dmargin_no_minus_yes": float(dmargin[idx]),
                "adjusted_no_minus_yes_logit": float(adjusted_margin[idx]),
            }
        )
    return rows


def _base_row(split: str, predictions: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    outcomes = [str(predictions[sample_id]["outcome"]) for sample_id in ids if sample_id in predictions]
    parsed = [str(predictions[sample_id].get("parsed_prediction", "")) for sample_id in ids if sample_id in predictions]
    counts = _counts(outcomes)
    yes_n = sum(1 for item in parsed if item == "yes")
    return {
        "split": split,
        "method": "Base",
        "n": len(outcomes),
        "tp": counts["TP"],
        "tn": counts["TN"],
        "fp": counts["FP"],
        "fn": counts["FN"],
        "accuracy": _accuracy(counts),
        "yes_rate": yes_n / len(parsed) if parsed else math.nan,
    }


def _calibrated_rows(
    sweep_rows: list[dict[str, Any]],
    calibration_split: str,
    test_split: str,
    min_tp_preserved: float,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(sweep_rows)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    keys = ["operator", "layer", "subspace"]
    test_lookup = {
        (row.operator, row.layer, row.subspace, row.alpha): row._asdict()
        for row in df[df["split"] == test_split].itertuples(index=False)
    }
    for key, group in df[df["split"] == calibration_split].groupby(keys, dropna=False):
        valid = group[group["tp_preserved"] >= min_tp_preserved].copy()
        if valid.empty:
            valid = group.copy()
        valid["calibration_score"] = valid["fp_reduction"].fillna(-1) - (1 - valid["tp_preserved"].fillna(0))
        best = valid.sort_values(
            ["calibration_score", "fp_reduction", "tp_preserved", "accuracy_delta"],
            ascending=[False, False, False, False],
        ).iloc[0]
        test_row = test_lookup.get((best.operator, int(best.layer), best.subspace, float(best.alpha)))
        if not test_row:
            continue
        out = dict(test_row)
        out["calibrated_on_split"] = calibration_split
        out["tested_on_split"] = test_split
        out["calibration_alpha"] = float(best.alpha)
        out["calibration_fp_reduction"] = float(best.fp_reduction)
        out["calibration_tp_preserved"] = float(best.tp_preserved)
        out["calibration_accuracy_delta"] = float(best.accuracy_delta)
        out["selection_rule"] = f"max fp_reduction - tp_damage with tp_preserved>={min_tp_preserved}; fallback same score"
        rows.append(out)
    return sorted(rows, key=lambda row: (row["operator"], row["layer"], row["subspace"]))


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    finite = [row for row in rows if np.isfinite(row["fp_reduction"]) and np.isfinite(row["tp_preserved"])]
    for row in finite:
        dominated = any(
            other is not row
            and other["fp_reduction"] >= row["fp_reduction"]
            and other["tp_preserved"] >= row["tp_preserved"]
            and (
                other["fp_reduction"] > row["fp_reduction"]
                or other["tp_preserved"] > row["tp_preserved"]
            )
            for other in finite
        )
        if not dominated:
            out = dict(row)
            out["pareto_axis_x_tp_damage"] = 1 - row["tp_preserved"]
            out["pareto_axis_y_fp_reduction"] = row["fp_reduction"]
            frontier.append(out)
    return sorted(frontier, key=lambda item: (item["operator"], item["layer"], item["pareto_axis_x_tp_damage"]))


def _band_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for key, group in df.groupby(["operator", "layer", "subspace"], dropna=False):
        best = group.sort_values(
            ["fp_reduction", "tp_preserved", "accuracy_delta"],
            ascending=[False, False, False],
        ).iloc[0]
        out.append(best.to_dict())
    return sorted(out, key=lambda row: (row["operator"], row["layer"], row["subspace"]))


def _load_margins(path: str | Path | None) -> dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "no_minus_yes_logit" not in df.columns:
        return {}
    return {str(row.sample_id): float(row.no_minus_yes_logit) for row in df.itertuples(index=False)}


def _load_split_map(split_dir: str | Path | None) -> dict[str, str]:
    if not split_dir:
        return {}
    root = Path(split_dir)
    mapping: dict[str, str] = {}
    for filename, split in [("pope_train_ids.json", "train"), ("pope_val_ids.json", "calibration"), ("pope_test_ids.json", "test")]:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for sample_id in payload.get("sample_ids", []):
            mapping[str(sample_id)] = split
    return mapping


def _base_no_minus_yes(sample_id: str, row: Any, margins: dict[str, float]) -> float:
    if sample_id in margins:
        return margins[sample_id]
    value = getattr(row, "orig_no_minus_yes_logit", math.nan)
    return float(value)


def _counts(outcomes: list[str]) -> dict[str, int]:
    return {key: sum(1 for item in outcomes if item == key) for key in ["TP", "TN", "FP", "FN", "unknown"]}


def _accuracy(counts: dict[str, int]) -> float:
    denom = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    return (counts["TP"] + counts["TN"]) / denom if denom else math.nan


def _method_name(operator: str, subspace: str) -> str:
    suffix = {
        "full": "Full",
        "top4": "Top4",
        "top16": "Top16",
        "band5_16": "Band5-16",
        "tail257_1024": "Tail257-1024",
        "top4_complement": "Top4-Complement",
        "random12": "Random12",
        "random4_complement": "Random4-Complement",
        "random_tail_dim": "RandomTailDim",
    }.get(subspace, subspace)
    return f"{suffix}-{operator}"


def _write_figures(figure_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if not rows:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    paths: list[str] = []
    path = figure_dir / "fp_reduction_vs_tp_preserved.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for (operator, subspace), group in df.groupby(["operator", "subspace"], dropna=False):
        ax.scatter(group["tp_preserved"], group["fp_reduction"], s=28, label=f"{operator}:{subspace}")
    ax.set_xlabel("TP Preserved")
    ax.set_ylabel("FP Reduction")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    path = figure_dir / "alpha_tradeoff_by_band.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for (operator, subspace), group in df.groupby(["operator", "subspace"], dropna=False):
        ordered = group.sort_values("alpha")
        ax.plot(ordered["alpha"], ordered["fp_reduction"], marker="o", label=f"{operator}:{subspace}")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("FP Reduction")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
