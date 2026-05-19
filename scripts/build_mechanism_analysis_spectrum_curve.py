#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from analyze_mechanism_mitigation_stage2 import analyze_stage2
from mechanism_analysis_common import add_metric_rates, fieldnames, markdown_table


DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build spectral curve from contiguous 12-direction windows.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument("--window-width", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-start", type=int, default=53)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/spectrum_curve_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_spectrum_curve(
        operator_geometry=Path(args.operator_geometry),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        window_width=args.window_width,
        stride=args.stride,
        max_start=args.max_start,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_spectrum_curve_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_spectrum_curve", summary_path, "ok")
    print(summary_path)


def build_spectrum_curve(
    operator_geometry: Path,
    predictions: Path,
    margin_scores: Path,
    split_dir: Path,
    alphas: list[float],
    min_tp_preserved: float,
    window_width: int,
    stride: int,
    max_start: int,
    output_dir: Path,
) -> dict[str, Any]:
    header = pd.read_csv(operator_geometry, nrows=0)
    present = _present_subspaces(header.columns)
    expected_windows = [f"band{start}_{start + window_width - 1}" for start in range(1, max_start + 1, stride)]
    window_subspaces = [name for name in expected_windows if name in present]
    random_subspaces = sorted(
        [name for name in present if re.fullmatch(r"random12(_s\d+)?", name)]
    )
    subspaces = list(dict.fromkeys(["full", *window_subspaces, *random_subspaces]))
    stage2_dir = output_dir / "stage2"
    analyze_stage2(
        operator_geometry_path=operator_geometry,
        predictions_path=predictions,
        margin_scores_path=margin_scores,
        subspaces=subspaces,
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
    selected = add_metric_rates(selected, samples, split="test")
    curve = _curve_rows(selected, expected_windows, random_subspaces)
    random_summary = _random_summary(selected, random_subspaces)
    baselines = _baseline_rows(selected)
    curve_path = write_csv(output_dir / "spectrum_curve.csv", curve, fieldnames(curve))
    random_path = write_csv(output_dir / "spectrum_random_control.csv", random_summary, fieldnames(random_summary))
    baseline_path = write_csv(output_dir / "spectrum_baselines.csv", baselines, fieldnames(baselines))
    figure_paths = _write_figures(output_dir / "figures", pd.DataFrame(curve), pd.DataFrame(random_summary), pd.DataFrame(baselines))
    report_path = _write_report(output_dir / "spectrum_curve_report.md", pd.DataFrame(curve), pd.DataFrame(random_summary), pd.DataFrame(baselines), figure_paths)
    return {
        "operator_geometry": str(operator_geometry),
        "stage2_dir": str(stage2_dir),
        "present_windows": window_subspaces,
        "missing_windows": [name for name in expected_windows if name not in window_subspaces],
        "random_subspaces": random_subspaces,
        "spectrum_curve_path": str(curve_path),
        "random_control_path": str(random_path),
        "baseline_path": str(baseline_path),
        "figure_paths": figure_paths,
        "report_path": str(report_path),
    }


def _present_subspaces(columns: pd.Index) -> set[str]:
    out = set()
    for column in columns:
        if column.startswith("dmargin_no_minus_yes_"):
            out.add(column.removeprefix("dmargin_no_minus_yes_"))
    return out


def _curve_rows(selected: pd.DataFrame, expected_windows: list[str], random_subspaces: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup = {str(row.subspace): row._asdict() for row in selected.itertuples(index=False)}
    for name in expected_windows:
        match = re.fullmatch(r"band(\d+)_(\d+)", name)
        start = int(match.group(1)) if match else np.nan
        end = int(match.group(2)) if match else np.nan
        source = lookup.get(name)
        row = {
            "subspace": name,
            "window_start": start,
            "window_end": end,
            "available": source is not None,
        }
        if source:
            row.update(
                {
                    "alpha": source["alpha"],
                    "fp_reduction": source["fp_reduction"],
                    "tp_damage": 1 - source["tp_preserved"],
                    "tp_preserved": source["tp_preserved"],
                    "accuracy_delta": source["accuracy_delta"],
                    "fp_yes_rate": source.get("fp_yes_rate", np.nan),
                    "tp_yes_rate": source.get("tp_yes_rate", np.nan),
                    "tn_yes_rate": source.get("tn_yes_rate", np.nan),
                }
            )
        rows.append(row)
    return rows


def _random_summary(selected: pd.DataFrame, random_subspaces: list[str]) -> list[dict[str, Any]]:
    view = selected[selected["subspace"].isin(random_subspaces)].copy()
    if view.empty:
        return []
    return [
        {
            "family": "random12",
            "n": int(len(view)),
            "fp_reduction_mean": float(view["fp_reduction"].mean()),
            "fp_reduction_std": float(view["fp_reduction"].std(ddof=0)),
            "tp_damage_mean": float((1 - view["tp_preserved"]).mean()),
            "tp_damage_std": float((1 - view["tp_preserved"]).std(ddof=0)),
            "tp_preserved_mean": float(view["tp_preserved"].mean()),
            "accuracy_delta_mean": float(view["accuracy_delta"].mean()),
        }
    ]


def _baseline_rows(selected: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    full = selected[selected["subspace"].astype(str) == "full"]
    if not full.empty:
        item = full.iloc[0]
        rows.append(
            {
                "baseline": "full_icd_tp_safe",
                "fp_reduction": float(item["fp_reduction"]),
                "tp_damage": float(1 - item["tp_preserved"]),
                "tp_preserved": float(item["tp_preserved"]),
                "accuracy_delta": float(item["accuracy_delta"]),
            }
        )
    rows.append(
        {
            "baseline": "always_icd_frozen",
            "fp_reduction": 0.340,
            "tp_damage": 1 - 0.912,
            "tp_preserved": 0.912,
            "accuracy_delta": -0.022,
        }
    )
    return rows


def _write_figures(figure_dir: Path, curve: pd.DataFrame, random_summary: pd.DataFrame, baselines: pd.DataFrame) -> list[str]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    if curve.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    available = curve[curve["available"] == True].copy()  # noqa: E712
    path = figure_dir / "spectrum_curve_fp_tp_damage.png"
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(available["window_start"], available["fp_reduction"], marker="o", label="FP reduction")
    ax1.set_xlabel("Window start singular direction")
    ax1.set_ylabel("FP reduction")
    ax2 = ax1.twinx()
    ax2.plot(available["window_start"], available["tp_damage"], marker="s", color="tab:red", label="TP damage")
    ax2.set_ylabel("TP damage")
    if not random_summary.empty:
        r = random_summary.iloc[0]
        ax1.axhline(float(r["fp_reduction_mean"]), color="tab:blue", linestyle="--", linewidth=1, label="random12 FP mean")
        ax1.fill_between(
            [available["window_start"].min(), available["window_start"].max()],
            float(r["fp_reduction_mean"] - r["fp_reduction_std"]),
            float(r["fp_reduction_mean"] + r["fp_reduction_std"]),
            color="tab:blue",
            alpha=0.12,
        )
    for _, row in baselines.iterrows():
        ax1.axhline(float(row["fp_reduction"]), color="gray", linestyle=":", linewidth=1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path)]


def _write_report(path: Path, curve: pd.DataFrame, random_summary: pd.DataFrame, baselines: pd.DataFrame, figure_paths: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    curve_cols = ["subspace", "window_start", "available", "alpha", "fp_reduction", "tp_damage", "tp_preserved", "accuracy_delta"]
    lines = [
        "# Spectrum Curve",
        "",
        "This view treats contiguous 12-direction bands as a spectral curve. Missing rows indicate windows not present in the current GPU dump.",
        "",
        "## Curve",
        "",
        markdown_table(curve, columns=curve_cols),
        "",
        "## Random Control",
        "",
        markdown_table(random_summary),
        "",
        "## Baselines",
        "",
        markdown_table(baselines),
        "",
    ]
    if figure_paths:
        lines.extend(["## Figures", "", *[f"- `{item}`" for item in figure_paths], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
