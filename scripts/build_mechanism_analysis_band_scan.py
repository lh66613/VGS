#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from analyze_mechanism_mitigation_stage2 import analyze_stage2
from mechanism_analysis_common import (
    add_metric_rates,
    fieldnames,
    markdown_table,
    require_present_subspaces,
)


DEFAULT_SUBSPACES = [
    "full",
    "top4",
    "band1_12",
    "band5_16",
    "band9_20",
    "band13_24",
    "band17_28",
    "band29_40",
    "band41_52",
    "band53_64",
    "tail257_1024",
    "random12",
    "randcontig12_s00",
]
DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contiguous-band subspace ICD scan.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_band_scan(
        operator_geometry_path=Path(args.operator_geometry),
        predictions_path=Path(args.predictions),
        margin_scores_path=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        subspaces=args.subspaces,
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_band_scan_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_band_scan", summary_path, "ok")
    print(summary_path)


def build_band_scan(
    operator_geometry_path: Path,
    predictions_path: Path,
    margin_scores_path: Path,
    split_dir: Path,
    subspaces: list[str],
    alphas: list[float],
    min_tp_preserved: float,
    output_dir: Path,
) -> dict[str, Any]:
    geometry = pd.read_csv(operator_geometry_path)
    present_subspaces = require_present_subspaces(geometry, subspaces, operator_geometry_path)
    stage2_dir = output_dir / "stage2"
    analyze_stage2(
        operator_geometry_path=operator_geometry_path,
        predictions_path=predictions_path,
        margin_scores_path=margin_scores_path,
        subspaces=present_subspaces,
        alphas=alphas,
        split_policy="fixed_ids",
        split_dir=split_dir,
        calibration_subset="popular",
        test_subset="adversarial",
        min_tp_preserved=min_tp_preserved,
        output_dir=stage2_dir,
    )

    selected = pd.read_csv(stage2_dir / "subspace_vcd_results.csv")
    samples = pd.read_csv(stage2_dir / "sample_predictions.csv")
    table = add_metric_rates(selected, samples, split="test")
    table = _sort_subspaces(table, present_subspaces)
    rows = table.to_dict(orient="records")
    table_path = write_csv(output_dir / "band_scan_table.csv", rows, fieldnames(rows))
    random_summary_rows = _random_summary_rows(table)
    random_summary_path = write_csv(
        output_dir / "band_scan_random_summary.csv",
        random_summary_rows,
        fieldnames(random_summary_rows),
    )
    figure_paths = _write_figures(output_dir / "figures", table)
    report_path = _write_report(output_dir / "band_scan_report.md", table, figure_paths)
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "present_subspaces": present_subspaces,
        "stage2_dir": str(stage2_dir),
        "band_scan_table_path": str(table_path),
        "random_summary_path": str(random_summary_path),
        "report_path": str(report_path),
        "figure_paths": figure_paths,
        "num_rows": len(rows),
    }


def _sort_subspaces(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    rank = {name: idx for idx, name in enumerate(order)}
    out = df.copy()
    out["_rank"] = out["subspace"].map(rank).fillna(len(order)).astype(int)
    return out.sort_values(["operator", "layer", "_rank", "subspace"]).drop(columns=["_rank"])


def _random_summary_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    random_rows = table[
        table["subspace"].astype(str).str.match(r"(random12|randcontig12_s\d+)")
    ].copy()
    if random_rows.empty:
        return []
    rows: list[dict[str, Any]] = []
    for family, group in random_rows.groupby(random_rows["subspace"].astype(str).str.extract(r"^([a-z]+[0-9]*)", expand=False)):
        rows.append(
            {
                "family": family,
                "n": int(len(group)),
                "fp_reduction_mean": float(group["fp_reduction"].mean()),
                "fp_reduction_min": float(group["fp_reduction"].min()),
                "fp_reduction_max": float(group["fp_reduction"].max()),
                "tp_preserved_mean": float(group["tp_preserved"].mean()),
                "accuracy_delta_mean": float(group["accuracy_delta"].mean()),
            }
        )
    return rows


def _write_figures(figure_dir: Path, table: pd.DataFrame) -> list[str]:
    if table.empty:
        return []
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = table[table["operator"].astype(str) == "icd_blind"].copy()
    if plot_df.empty:
        plot_df = table.copy()
    labels = plot_df["subspace"].astype(str).tolist()
    x = np.arange(len(labels))
    path = figure_dir / "band_scan_metrics.png"
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.55 * len(labels)), 8), sharex=True)
    for ax, col, title in [
        (axes[0], "fp_reduction", "FP Reduction"),
        (axes[1], "tp_preserved", "TP Preserved"),
        (axes[2], "accuracy_delta", "Accuracy Delta"),
    ]:
        ax.bar(x, plot_df[col].astype(float))
        ax.set_ylabel(title)
        ax.axhline(0, color="black", linewidth=0.8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path)]


def _write_report(path: Path, table: pd.DataFrame, figure_paths: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "method",
        "subspace",
        "alpha",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
        "overall_yes_rate",
        "fp_yes_rate",
        "tp_yes_rate",
        "tn_yes_rate",
    ]
    lines = [
        "# Mechanism Analysis Band Scan",
        "",
        "Calibration split: fixed POPE calibration IDs. Test split: fixed POPE test IDs.",
        "",
        markdown_table(table[cols] if not table.empty else table),
        "",
    ]
    if figure_paths:
        lines.extend(["## Figures", "", *[f"- `{item}`" for item in figure_paths], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
