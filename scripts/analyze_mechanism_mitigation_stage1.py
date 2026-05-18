#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import os
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json


DEFAULT_BANDS = [
    "full",
    "top4",
    "top16",
    "band5_16",
    "band17_64",
    "band65_256",
    "tail257_1024",
    "top4_complement",
    "random_tail_dim",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: decompose VCD/ICD corrections by spectrum band.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv",
    )
    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/stage1_vcd_decomposition")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = analyze_stage1(args.operator_geometry, args.bands, args.output_dir)
    summary_path = write_json(Path(args.output_dir) / "analyze_mechanism_mitigation_stage1_summary.json", result)
    append_experiment_log(args.log_path, "analyze_mechanism_mitigation_stage1", summary_path, "ok")
    print(summary_path)


def analyze_stage1(
    operator_geometry_path: str | Path,
    bands: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    df = pd.read_csv(operator_geometry_path)
    present_bands = [
        band
        for band in bands
        if f"energy_frac_{band}" in df.columns and f"dmargin_no_minus_yes_{band}" in df.columns
    ]
    if not present_bands:
        raise ValueError("No requested bands were present in the operator geometry CSV.")

    energy_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    damage_rows: list[dict[str, Any]] = []
    for (operator, layer), group in df.groupby(["operator", "layer"], dropna=False):
        for band in present_bands:
            energy_col = f"energy_frac_{band}"
            margin_col = f"dmargin_no_minus_yes_{band}"
            for outcome, outcome_group in group.groupby("outcome", dropna=False):
                energy_rows.append(
                    {
                        "operator": operator,
                        "layer": layer,
                        "band": band,
                        "outcome": outcome,
                        "n": int(len(outcome_group)),
                        "mean_energy_fraction": _nanmean(outcome_group[energy_col]),
                        "median_energy_fraction": _nanmedian(outcome_group[energy_col]),
                    }
                )
                contribution_rows.append(
                    {
                        "operator": operator,
                        "layer": layer,
                        "band": band,
                        "outcome": outcome,
                        "n": int(len(outcome_group)),
                        "mean_no_minus_yes_contribution": _nanmean(outcome_group[margin_col]),
                        "median_no_minus_yes_contribution": _nanmedian(outcome_group[margin_col]),
                        "positive_contribution_rate": _positive_rate(outcome_group[margin_col]),
                    }
                )

            correlation_rows.extend(
                [
                    _metric_row(operator, layer, band, "fp_vs_tn_energy", group, energy_col, {"FP": 1, "TN": 0}),
                    _metric_row(operator, layer, band, "fp_vs_tn_dmargin", group, margin_col, {"FP": 1, "TN": 0}),
                    _metric_row(
                        operator,
                        layer,
                        band,
                        "predicted_yes_fp_vs_tp_energy",
                        group,
                        energy_col,
                        {"FP": 1, "TP": 0},
                    ),
                    _metric_row(
                        operator,
                        layer,
                        band,
                        "predicted_yes_fp_vs_tp_dmargin",
                        group,
                        margin_col,
                        {"FP": 1, "TP": 0},
                    ),
                ]
            )
            damage_rows.append(_tradeoff_proxy_row(operator, layer, band, group, margin_col))

    output_root = Path(output_dir)
    energy_path = write_csv(output_root / "vcd_band_energy.csv", energy_rows, _fieldnames(energy_rows))
    contribution_path = write_csv(
        output_root / "vcd_band_logit_contribution.csv",
        contribution_rows,
        _fieldnames(contribution_rows),
    )
    correlation_path = write_csv(
        output_root / "vcd_success_failure_analysis.csv",
        correlation_rows,
        _fieldnames(correlation_rows),
    )
    damage_path = write_csv(output_root / "vcd_tp_damage_analysis.csv", damage_rows, _fieldnames(damage_rows))
    figure_paths = _write_figures(output_root / "figures", energy_rows, contribution_rows)
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "bands": present_bands,
        "energy_path": str(energy_path),
        "logit_contribution_path": str(contribution_path),
        "success_failure_analysis_path": str(correlation_path),
        "tp_damage_analysis_path": str(damage_path),
        "figure_paths": figure_paths,
        "num_input_rows": int(len(df)),
    }


def _metric_row(
    operator: str,
    layer: int,
    band: str,
    metric: str,
    group: pd.DataFrame,
    value_col: str,
    outcome_labels: dict[str, int],
) -> dict[str, Any]:
    subset = group[group["outcome"].isin(outcome_labels)].copy()
    y = np.array([outcome_labels[item] for item in subset["outcome"]], dtype=np.int64)
    scores = subset[value_col].to_numpy(dtype=float)
    keep = np.isfinite(scores)
    y = y[keep]
    scores = scores[keep]
    return {
        "operator": operator,
        "layer": layer,
        "band": band,
        "metric": metric,
        "n": int(len(y)),
        "positive_n": int(np.sum(y == 1)) if len(y) else 0,
        "negative_n": int(np.sum(y == 0)) if len(y) else 0,
        "auroc": _safe_metric(y, scores, roc_auc_score),
        "auprc": _safe_metric(y, scores, average_precision_score),
        "mean_positive": float(np.mean(scores[y == 1])) if np.any(y == 1) else math.nan,
        "mean_negative": float(np.mean(scores[y == 0])) if np.any(y == 0) else math.nan,
    }


def _tradeoff_proxy_row(
    operator: str,
    layer: int,
    band: str,
    group: pd.DataFrame,
    value_col: str,
) -> dict[str, Any]:
    fp = group[group["outcome"] == "FP"][value_col].to_numpy(dtype=float)
    tp = group[group["outcome"] == "TP"][value_col].to_numpy(dtype=float)
    fp = fp[np.isfinite(fp)]
    tp = tp[np.isfinite(tp)]
    fp_help = float(np.mean(fp > 0)) if len(fp) else math.nan
    tp_damage = float(np.mean(tp > 0)) if len(tp) else math.nan
    return {
        "operator": operator,
        "layer": layer,
        "band": band,
        "fp_positive_contribution_rate": fp_help,
        "tp_positive_contribution_rate": tp_damage,
        "fp_minus_tp_positive_rate_gap": fp_help - tp_damage if not math.isnan(fp_help) and not math.isnan(tp_damage) else math.nan,
        "mean_fp_no_minus_yes_contribution": float(np.mean(fp)) if len(fp) else math.nan,
        "mean_tp_no_minus_yes_contribution": float(np.mean(tp)) if len(tp) else math.nan,
    }


def _safe_metric(y: np.ndarray, scores: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    try:
        return float(fn(y, scores))
    except ValueError:
        return math.nan


def _write_figures(
    figure_dir: Path,
    energy_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    energy = pd.DataFrame(energy_rows)
    contrib = pd.DataFrame(contribution_rows)
    if not energy.empty:
        summary = (
            energy.groupby(["operator", "band"], as_index=False)["mean_energy_fraction"]
            .mean()
            .sort_values(["operator", "mean_energy_fraction"], ascending=[True, False])
        )
        path = figure_dir / "vcd_energy_by_band.png"
        _bar_plot(summary, "mean_energy_fraction", "Mean Energy Fraction", path, plt)
        paths.append(str(path))
    if not contrib.empty:
        summary = (
            contrib.groupby(["operator", "band"], as_index=False)["mean_no_minus_yes_contribution"]
            .mean()
            .sort_values(["operator", "mean_no_minus_yes_contribution"], ascending=[True, False])
        )
        path = figure_dir / "vcd_logit_effect_by_band.png"
        _bar_plot(summary, "mean_no_minus_yes_contribution", "Mean No-Yes Contribution", path, plt)
        paths.append(str(path))
    return paths


def _bar_plot(df: pd.DataFrame, value_col: str, ylabel: str, path: Path, plt: Any) -> None:
    labels = [f"{row.operator}\n{row.band}" for row in df.itertuples(index=False)]
    values = df[value_col].to_numpy(dtype=float)
    fig_width = max(8, min(18, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(np.arange(len(labels)), values)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _nanmean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan


def _nanmedian(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else math.nan


def _positive_rate(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr > 0)) if len(arr) else math.nan


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
