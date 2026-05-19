#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table, require_present_subspaces, safe_mean, safe_median


DEFAULT_SUBSPACES = [
    "full",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize yes/no logit shifts by outcome group.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument(
        "--band-scan-table",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/band_scan_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/logit_shift_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_logit_shift(
        operator_geometry_path=Path(args.operator_geometry),
        split_dir=Path(args.split_dir),
        split=args.split,
        subspaces=args.subspaces,
        band_scan_table=Path(args.band_scan_table),
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_logit_shift_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_logit_shift", summary_path, "ok")
    print(summary_path)


def build_logit_shift(
    operator_geometry_path: Path,
    split_dir: Path,
    split: str,
    subspaces: list[str],
    band_scan_table: Path,
    output_dir: Path,
) -> dict[str, Any]:
    df = pd.read_csv(operator_geometry_path)
    present = require_present_subspaces(df, subspaces, operator_geometry_path)
    split_map = _load_split_map(split_dir)
    df["split"] = df["sample_id"].astype(str).map(split_map).fillna(df.get("source_subset", ""))
    df = df[df["split"].astype(str) == split].copy()

    summary_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for (operator, layer), group in df.groupby(["operator", "layer"], dropna=False):
        for subspace in present:
            dmargin_col = f"dmargin_no_minus_yes_{subspace}"
            yes_col = f"dlogit_yes_{subspace}"
            no_col = f"dlogit_no_{subspace}"
            for outcome, outcome_group in group.groupby("outcome", dropna=False):
                dmargin = outcome_group[dmargin_col].astype(float)
                yes = outcome_group[yes_col].astype(float) if yes_col in outcome_group else pd.Series(dtype=float)
                no = outcome_group[no_col].astype(float) if no_col in outcome_group else pd.Series(dtype=float)
                summary_rows.append(
                    {
                        "operator": operator,
                        "layer": int(layer),
                        "split": split,
                        "subspace": subspace,
                        "outcome": outcome,
                        "n": int(len(outcome_group)),
                        "mean_dlogit_yes": safe_mean(yes),
                        "median_dlogit_yes": safe_median(yes),
                        "mean_dlogit_no": safe_mean(no),
                        "median_dlogit_no": safe_median(no),
                        "mean_dmargin_no_minus_yes": safe_mean(dmargin),
                        "median_dmargin_no_minus_yes": safe_median(dmargin),
                        "positive_no_minus_yes_rate": float((dmargin > 0).mean()) if len(dmargin) else np.nan,
                    }
                )
            gap_rows.append(_gap_row(operator, int(layer), split, subspace, group, dmargin_col))

    gap = pd.DataFrame(gap_rows)
    if band_scan_table.exists() and not gap.empty:
        scan = pd.read_csv(band_scan_table)
        merge_cols = ["operator", "layer", "subspace"]
        keep_cols = merge_cols + [
            "alpha",
            "fp_reduction",
            "tp_preserved",
            "accuracy_delta",
            "fp_yes_rate",
            "tp_yes_rate",
            "tn_yes_rate",
        ]
        gap = gap.merge(scan[[col for col in keep_cols if col in scan.columns]], on=merge_cols, how="left")

    summary_path = write_csv(output_dir / "logit_shift_summary.csv", summary_rows, fieldnames(summary_rows))
    gap_rows = gap.to_dict(orient="records")
    gap_path = write_csv(output_dir / "logit_shift_gap_table.csv", gap_rows, fieldnames(gap_rows))
    figure_paths = _write_figures(output_dir / "figures", gap)
    report_path = _write_report(output_dir / "logit_shift_report.md", gap, figure_paths)
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "split": split,
        "present_subspaces": present,
        "logit_shift_summary_path": str(summary_path),
        "logit_shift_gap_table_path": str(gap_path),
        "report_path": str(report_path),
        "figure_paths": figure_paths,
        "num_summary_rows": len(summary_rows),
        "num_gap_rows": len(gap_rows),
    }


def _gap_row(operator: str, layer: int, split: str, subspace: str, group: pd.DataFrame, dmargin_col: str) -> dict[str, Any]:
    fp = group[group["outcome"].astype(str) == "FP"][dmargin_col].astype(float)
    tp = group[group["outcome"].astype(str) == "TP"][dmargin_col].astype(float)
    tn = group[group["outcome"].astype(str) == "TN"][dmargin_col].astype(float)
    fn = group[group["outcome"].astype(str) == "FN"][dmargin_col].astype(float)
    fp_mean = safe_mean(fp)
    tp_mean = safe_mean(tp)
    return {
        "operator": operator,
        "layer": layer,
        "split": split,
        "subspace": subspace,
        "mean_dmargin_fp": fp_mean,
        "mean_dmargin_tp": tp_mean,
        "mean_dmargin_tn": safe_mean(tn),
        "mean_dmargin_fn": safe_mean(fn),
        "fp_tp_shift_gap": fp_mean - tp_mean if np.isfinite(fp_mean) and np.isfinite(tp_mean) else np.nan,
        "fp_positive_rate": float((fp > 0).mean()) if len(fp) else np.nan,
        "tp_positive_rate": float((tp > 0).mean()) if len(tp) else np.nan,
    }


def _load_split_map(split_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for filename, name in [
        ("pope_train_ids.json", "train"),
        ("pope_val_ids.json", "calibration"),
        ("pope_test_ids.json", "test"),
    ]:
        path = split_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample_id in payload.get("sample_ids", []):
            mapping[str(sample_id)] = name
    return mapping


def _write_figures(figure_dir: Path, gap: pd.DataFrame) -> list[str]:
    if gap.empty:
        return []
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = gap[gap["operator"].astype(str) == "icd_blind"].copy()
    if plot_df.empty:
        plot_df = gap.copy()
    labels = plot_df["subspace"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.28
    path = figure_dir / "logit_shift_gap_by_band.png"
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(labels)), 4.8))
    ax.bar(x - width, plot_df["mean_dmargin_fp"], width=width, label="FP")
    ax.bar(x, plot_df["mean_dmargin_tp"], width=width, label="TP")
    ax.bar(x + width, plot_df["fp_tp_shift_gap"], width=width, label="FP-TP gap")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta no-minus-yes margin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path)]


def _write_report(path: Path, gap: pd.DataFrame, figure_paths: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "subspace",
        "mean_dmargin_fp",
        "mean_dmargin_tp",
        "fp_tp_shift_gap",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
    ]
    lines = [
        "# Mechanism Analysis Logit Shift",
        "",
        "Positive margin means the subspace correction moves the next-token decision toward `No` over `Yes`.",
        "",
        markdown_table(gap[cols] if not gap.empty else gap),
        "",
    ]
    if figure_paths:
        lines.extend(["## Figures", "", *[f"- `{item}`" for item in figure_paths], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
