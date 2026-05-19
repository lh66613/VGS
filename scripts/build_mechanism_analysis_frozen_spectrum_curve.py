#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_WINDOWS = [f"band{start}_{start + 11}" for start in range(1, 54, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen stride-4 spectrum curve without GPU forward passes.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_operator_geometry/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_spectrum_curve_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_frozen_spectrum(
        operator_geometry=Path(args.operator_geometry),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_frozen_spectrum_curve_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_frozen_spectrum_curve", summary_path, "ok")
    print(summary_path)


def build_frozen_spectrum(
    operator_geometry: Path,
    predictions: Path,
    margin_scores: Path,
    split_dir: Path,
    alphas: list[float],
    min_tp_preserved: float,
    output_dir: Path,
) -> dict[str, Any]:
    frame = _load_frame(operator_geometry, predictions, margin_scores, split_dir)
    present = _present_subspaces(frame.columns)
    windows = [name for name in DEFAULT_WINDOWS if name in present]
    missing_windows = [name for name in DEFAULT_WINDOWS if name not in present]
    random12 = sorted(name for name in present if re.fullmatch(r"random12(_s\d+)?", name))
    randcontig12 = sorted(name for name in present if re.fullmatch(r"randcontig12_s\d+", name))
    subspaces = list(dict.fromkeys(["full", "top4", *windows, *random12, *randcontig12]))
    selected_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for subspace in subspaces:
        if f"dmargin_no_minus_yes_{subspace}" not in frame.columns:
            continue
        selected, sweep = _select_subspace(frame, subspace, alphas, min_tp_preserved)
        selected_rows.append(selected)
        alpha_rows.extend(sweep)

    selected = pd.DataFrame(selected_rows)
    curve_rows = _curve_rows(selected, DEFAULT_WINDOWS)
    random_summary = _random_summary(selected, random12, randcontig12)
    baselines = _baseline_rows(selected, frame)
    no_bias_rows = _no_bias_rows(selected, frame)
    curve_path = write_csv(output_dir / "frozen_spectrum_curve.csv", curve_rows, fieldnames(curve_rows))
    selected_path = write_csv(output_dir / "frozen_spectrum_selected.csv", selected_rows, fieldnames(selected_rows))
    alpha_path = write_csv(output_dir / "frozen_spectrum_alpha_sweep.csv", alpha_rows, fieldnames(alpha_rows))
    random_path = write_csv(output_dir / "frozen_spectrum_random_control.csv", random_summary, fieldnames(random_summary))
    baseline_path = write_csv(output_dir / "frozen_spectrum_baselines.csv", baselines, fieldnames(baselines))
    no_bias_path = write_csv(output_dir / "frozen_peak_no_bias_audit.csv", no_bias_rows, fieldnames(no_bias_rows))
    no_bias_report_path = _write_no_bias_report(
        output_dir / "frozen_peak_no_bias_audit_report.md",
        pd.DataFrame(no_bias_rows),
    )
    figure_paths = _write_figures(output_dir / "figures", pd.DataFrame(curve_rows), pd.DataFrame(random_summary), pd.DataFrame(baselines))
    report_path = _write_report(
        output_dir / "frozen_spectrum_curve_report.md",
        pd.DataFrame(curve_rows),
        pd.DataFrame(random_summary),
        pd.DataFrame(baselines),
        pd.DataFrame(no_bias_rows),
        figure_paths,
        missing_windows,
    )
    return {
        "operator_geometry": str(operator_geometry),
        "present_windows": windows,
        "missing_windows": missing_windows,
        "random12_subspaces": random12,
        "randcontig12_subspaces": randcontig12,
        "selected_path": str(selected_path),
        "alpha_sweep_path": str(alpha_path),
        "curve_path": str(curve_path),
        "random_control_path": str(random_path),
        "baseline_path": str(baseline_path),
        "no_bias_path": str(no_bias_path),
        "no_bias_report_path": str(no_bias_report_path),
        "figure_paths": figure_paths,
        "report_path": str(report_path),
    }


def _load_frame(operator_geometry: Path, predictions: Path, margin_scores: Path, split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(operator_geometry)
    df = df[(df["operator"].astype(str) == "icd_blind")].copy()
    pred_rows = {str(row["sample_id"]): row for row in read_jsonl(predictions)}
    margins = pd.read_csv(margin_scores)
    margin_lookup = {str(row.sample_id): float(row.no_minus_yes_logit) for row in margins.itertuples(index=False)}
    split_map = _load_split_map(split_dir)
    labels = []
    outcomes = []
    parsed = []
    splits = []
    base_margin = []
    for row in df.itertuples(index=False):
        sample_id = str(row.sample_id)
        pred = pred_rows.get(sample_id, {})
        labels.append(str(pred.get("label", getattr(row, "label", ""))))
        outcomes.append(str(pred.get("outcome", getattr(row, "outcome", ""))))
        parsed.append(str(pred.get("parsed_prediction", getattr(row, "parsed_prediction", ""))))
        splits.append(split_map.get(sample_id, "unassigned"))
        base_margin.append(float(margin_lookup.get(sample_id, getattr(row, "orig_no_minus_yes_logit", math.nan))))
    df["label_eval"] = labels
    df["original_outcome_eval"] = outcomes
    df["original_prediction_eval"] = parsed
    df["split_eval"] = splits
    df["base_no_minus_yes_logit"] = base_margin
    return df


def _select_subspace(
    frame: pd.DataFrame,
    subspace: str,
    alphas: list[float],
    min_tp_preserved: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep: list[dict[str, Any]] = []
    for split in ["calibration", "test"]:
        group = frame[frame["split_eval"].astype(str) == split]
        for alpha in alphas:
            sweep.append(_metric_row(group, subspace, alpha, split))
    calibration = pd.DataFrame([row for row in sweep if row["split"] == "calibration"])
    valid = calibration[calibration["tp_preserved"] >= min_tp_preserved].copy()
    if valid.empty:
        valid = calibration.copy()
    valid["selection_score"] = valid["fp_reduction"].fillna(-1) - (1 - valid["tp_preserved"].fillna(0))
    best = valid.sort_values(
        ["selection_score", "fp_reduction", "tp_preserved", "accuracy_delta"],
        ascending=[False, False, False, False],
    ).iloc[0]
    test_row = next(row for row in sweep if row["split"] == "test" and float(row["alpha"]) == float(best.alpha))
    selected = dict(test_row)
    selected.update(
        {
            "operator": "icd_blind",
            "layer": int(frame["layer"].iloc[0]),
            "calibration_alpha": float(best.alpha),
            "calibration_fp_reduction": float(best.fp_reduction),
            "calibration_tp_preserved": float(best.tp_preserved),
            "calibration_accuracy_delta": float(best.accuracy_delta),
            "selection_rule": f"max fp_reduction - tp_damage with tp_preserved>={min_tp_preserved}; fallback same score",
        }
    )
    return selected, sweep


def _metric_row(group: pd.DataFrame, subspace: str, alpha: float, split: str) -> dict[str, Any]:
    base = group["base_no_minus_yes_logit"].to_numpy(dtype=float)
    dmargin = group[f"dmargin_no_minus_yes_{subspace}"].to_numpy(dtype=float)
    adjusted = base + float(alpha) * dmargin
    final = np.where(adjusted >= 0, "no", "yes")
    original = group["original_outcome_eval"].astype(str).to_numpy()
    labels = group["label_eval"].astype(str).to_numpy()
    final_outcomes = np.array([_classify(pred, label) for pred, label in zip(final, labels)])
    original_counts = _counts(original)
    final_counts = _counts(final_outcomes)
    original_accuracy = _accuracy(original_counts)
    final_accuracy = _accuracy(final_counts)
    fp_fixed = int(((original == "FP") & (final_outcomes == "TN")).sum())
    tp_kept = int(((original == "TP") & (final_outcomes == "TP")).sum())
    tn_kept = int(((original == "TN") & (final_outcomes == "TN")).sum())
    tp_damaged = int(((original == "TP") & (final_outcomes != "TP")).sum())
    return {
        "split": split,
        "subspace": subspace,
        "alpha": float(alpha),
        "n": int(len(group)),
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
        "fp_reduction": fp_fixed / original_counts["FP"] if original_counts["FP"] else math.nan,
        "tp_preserved": tp_kept / original_counts["TP"] if original_counts["TP"] else math.nan,
        "tn_preserved": tn_kept / original_counts["TN"] if original_counts["TN"] else math.nan,
        "accuracy_before": original_accuracy,
        "accuracy_after": final_accuracy,
        "accuracy_delta": final_accuracy - original_accuracy,
        "overall_yes_rate": _yes_rate(final),
        "fp_yes_rate": _yes_rate(final[original == "FP"]),
        "tp_yes_rate": _yes_rate(final[original == "TP"]),
        "tn_yes_rate": _yes_rate(final[original == "TN"]),
        "fn_yes_rate": _yes_rate(final[original == "FN"]),
        "fp_reduction_per_tp_damage": fp_fixed / tp_damaged if tp_damaged else math.inf if fp_fixed else math.nan,
    }


def _curve_rows(selected: pd.DataFrame, windows: list[str]) -> list[dict[str, Any]]:
    rows = []
    lookup = {str(row.subspace): row._asdict() for row in selected.itertuples(index=False)}
    for name in windows:
        match = re.fullmatch(r"band(\d+)_(\d+)", name)
        row = {
            "subspace": name,
            "window_start": int(match.group(1)) if match else math.nan,
            "window_end": int(match.group(2)) if match else math.nan,
            "available": name in lookup,
        }
        if name in lookup:
            source = lookup[name]
            row.update(
                {
                    "alpha": source["alpha"],
                    "fp_reduction": source["fp_reduction"],
                    "tp_damage": 1 - source["tp_preserved"],
                    "tp_preserved": source["tp_preserved"],
                    "accuracy_delta": source["accuracy_delta"],
                    "fp_yes_rate": source["fp_yes_rate"],
                    "tp_yes_rate": source["tp_yes_rate"],
                    "tn_yes_rate": source["tn_yes_rate"],
                    "calibration_fp_reduction": source["calibration_fp_reduction"],
                    "calibration_tp_preserved": source["calibration_tp_preserved"],
                }
            )
        rows.append(row)
    return rows


def _random_summary(selected: pd.DataFrame, random12: list[str], randcontig12: list[str]) -> list[dict[str, Any]]:
    rows = []
    for family, names in [("random12", random12), ("random_contiguous12", randcontig12)]:
        view = selected[selected["subspace"].isin(names)].copy()
        if view.empty:
            continue
        rows.append(
            {
                "family": family,
                "n": int(len(view)),
                "fp_reduction_mean": float(view["fp_reduction"].mean()),
                "fp_reduction_std": float(view["fp_reduction"].std(ddof=0)),
                "tp_damage_mean": float((1 - view["tp_preserved"]).mean()),
                "tp_damage_std": float((1 - view["tp_preserved"]).std(ddof=0)),
                "tp_preserved_mean": float(view["tp_preserved"].mean()),
                "accuracy_delta_mean": float(view["accuracy_delta"].mean()),
            }
        )
    return rows


def _baseline_rows(selected: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label, subspace in [
        ("full_icd_tp_safe", "full"),
        ("top4", "top4"),
        ("band5_16", "band5_16"),
    ]:
        view = selected[selected["subspace"].astype(str) == subspace]
        if not view.empty:
            item = view.iloc[0]
            rows.append(
                {
                    "baseline": label,
                    "subspace": subspace,
                    "alpha": float(item["alpha"]),
                    "fp_reduction": float(item["fp_reduction"]),
                    "tp_damage": float(1 - item["tp_preserved"]),
                    "tp_preserved": float(item["tp_preserved"]),
                    "accuracy_delta": float(item["accuracy_delta"]),
                }
            )
    rows.append(
        {
            "baseline": "always_icd_frozen_table",
            "subspace": "full",
            "alpha": math.nan,
            "fp_reduction": 0.340,
            "tp_damage": 1 - 0.912,
            "tp_preserved": 0.912,
            "accuracy_delta": -0.022,
        }
    )
    base = frame[frame["split_eval"].astype(str) == "test"]
    rows.insert(
        0,
        {
            "baseline": "base_test_split",
            "subspace": "none",
            "alpha": 0.0,
            "fp_reduction": 0.0,
            "tp_damage": 0.0,
            "tp_preserved": 1.0,
            "accuracy_delta": 0.0,
            "n": int(len(base)),
        },
    )
    return rows


def _no_bias_rows(selected: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    test = frame[frame["split_eval"].astype(str) == "test"]
    original = test["original_outcome_eval"].astype(str).to_numpy()
    parsed = test["original_prediction_eval"].astype(str).to_numpy()
    rows.append(
        {
            "method": "Base",
            "subspace": "none",
            "alpha": 0.0,
            "fp_yes_rate": _yes_rate(parsed[original == "FP"]),
            "tp_yes_rate": _yes_rate(parsed[original == "TP"]),
            "tn_yes_rate": _yes_rate(parsed[original == "TN"]),
            "overall_yes_rate": _yes_rate(parsed),
            "fp_reduction": 0.0,
            "tp_preserved": 1.0,
            "accuracy_delta": 0.0,
        }
    )
    rows.append(
        {
            "method": "Always ICD",
            "subspace": "full",
            "alpha": math.nan,
            "fp_yes_rate": 0.660,
            "tp_yes_rate": 0.912,
            "tn_yes_rate": 0.000,
            "overall_yes_rate": 0.393,
            "fp_reduction": 0.340,
            "tp_preserved": 0.912,
            "accuracy_delta": -0.022,
        }
    )
    window_rows = selected[selected["subspace"].astype(str).str.fullmatch(r"band\d+_\d+", na=False)].copy()
    best_window = window_rows.sort_values(["fp_reduction", "tp_preserved", "accuracy_delta"], ascending=[False, False, False]).head(1)
    wanted = [
        ("Full ICD TP-safe", "full"),
        ("Band5-16", "band5_16"),
        ("Best window", str(best_window.iloc[0]["subspace"]) if not best_window.empty else ""),
        ("top4", "top4"),
    ]
    for method, subspace in wanted:
        view = selected[selected["subspace"].astype(str) == subspace]
        if view.empty:
            continue
        item = view.iloc[0]
        rows.append(_selected_no_bias_row(method, item))
    random_view = selected[selected["subspace"].astype(str).str.fullmatch(r"random12(_s\d+)?", na=False)]
    if not random_view.empty:
        rows.append(
            {
                "method": "Random12 mean",
                "subspace": "random12_family",
                "alpha": math.nan,
                "fp_yes_rate": float(random_view["fp_yes_rate"].mean()),
                "tp_yes_rate": float(random_view["tp_yes_rate"].mean()),
                "tn_yes_rate": float(random_view["tn_yes_rate"].mean()),
                "overall_yes_rate": float(random_view["overall_yes_rate"].mean()),
                "fp_reduction": float(random_view["fp_reduction"].mean()),
                "tp_preserved": float(random_view["tp_preserved"].mean()),
                "accuracy_delta": float(random_view["accuracy_delta"].mean()),
            }
        )
    return rows


def _selected_no_bias_row(method: str, item: pd.Series) -> dict[str, Any]:
    return {
        "method": method,
        "subspace": str(item["subspace"]),
        "alpha": float(item["alpha"]),
        "fp_yes_rate": float(item["fp_yes_rate"]),
        "tp_yes_rate": float(item["tp_yes_rate"]),
        "tn_yes_rate": float(item["tn_yes_rate"]),
        "overall_yes_rate": float(item["overall_yes_rate"]),
        "fp_reduction": float(item["fp_reduction"]),
        "tp_preserved": float(item["tp_preserved"]),
        "accuracy_delta": float(item["accuracy_delta"]),
    }


def _write_figures(
    figure_dir: Path,
    curve: pd.DataFrame,
    random_summary: pd.DataFrame,
    baselines: pd.DataFrame,
) -> list[str]:
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
    path = figure_dir / "frozen_spectrum_curve_fp_tp_damage.png"
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(available["window_start"], available["fp_reduction"], marker="o", label="FP reduction")
    ax1.set_xlabel("Window start singular direction")
    ax1.set_ylabel("FP reduction")
    ax2 = ax1.twinx()
    ax2.plot(available["window_start"], available["tp_damage"], marker="s", color="tab:red", label="TP damage")
    ax2.set_ylabel("TP damage")
    for _, row in random_summary.iterrows():
        if str(row["family"]) == "random12":
            ax1.axhline(float(row["fp_reduction_mean"]), color="tab:blue", linestyle="--", linewidth=1, label="random12 mean")
            ax1.fill_between(
                [available["window_start"].min(), available["window_start"].max()],
                float(row["fp_reduction_mean"] - row["fp_reduction_std"]),
                float(row["fp_reduction_mean"] + row["fp_reduction_std"]),
                color="tab:blue",
                alpha=0.12,
            )
        if str(row["family"]) == "random_contiguous12":
            ax1.axhline(float(row["fp_reduction_mean"]), color="tab:green", linestyle="--", linewidth=1, label="randcontig12 mean")
    for _, row in baselines.iterrows():
        if row.get("baseline") in {"full_icd_tp_safe", "top4", "always_icd_frozen_table"}:
            ax1.axhline(float(row["fp_reduction"]), color="gray", linestyle=":", linewidth=1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path)]


def _write_report(
    path: Path,
    curve: pd.DataFrame,
    random_summary: pd.DataFrame,
    baselines: pd.DataFrame,
    no_bias: pd.DataFrame,
    figure_paths: list[str],
    missing_windows: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    curve_cols = [
        "subspace",
        "window_start",
        "alpha",
        "fp_reduction",
        "tp_damage",
        "tp_preserved",
        "accuracy_delta",
        "fp_yes_rate",
        "tp_yes_rate",
        "tn_yes_rate",
    ]
    lines = [
        "# Frozen Spectrum Curve",
        "",
        "Unless otherwise stated, all main results use the frozen exact-reproduction pipeline.",
        "",
        "This table evaluates contiguous 12-direction stride-4 windows using the frozen hidden cache and reference geometry convention.",
        "",
        "## Curve",
        "",
        markdown_table(curve, columns=curve_cols),
        "",
        "## Random Controls",
        "",
        markdown_table(random_summary),
        "",
        "## Baselines",
        "",
        markdown_table(baselines),
        "",
        "## Peak-Level No-Bias Audit",
        "",
        markdown_table(no_bias),
        "",
    ]
    if missing_windows:
        lines.extend(["## Missing Windows", "", ", ".join(missing_windows), ""])
    if figure_paths:
        lines.extend(["## Figures", "", *[f"- `{item}`" for item in figure_paths], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_no_bias_report(path: Path, no_bias: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Frozen Peak-Level No-Bias Audit",
        "",
        "Unless otherwise stated, all main results use the frozen exact-reproduction pipeline.",
        "",
        "This audit checks whether spectral corrections mainly lower FP yes rate while preserving TP and TN behavior.",
        "",
        markdown_table(no_bias),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _present_subspaces(columns: pd.Index) -> set[str]:
    return {column.removeprefix("dmargin_no_minus_yes_") for column in columns if column.startswith("dmargin_no_minus_yes_")}


def _load_split_map(split_dir: Path) -> dict[str, str]:
    mapping = {}
    for filename, split in [("pope_train_ids.json", "train"), ("pope_val_ids.json", "calibration"), ("pope_test_ids.json", "test")]:
        payload = json.loads((split_dir / filename).read_text(encoding="utf-8"))
        for sample_id in payload.get("sample_ids", []):
            mapping[str(sample_id)] = split
    return mapping


def _classify(prediction: str, label: str) -> str:
    if prediction == "yes" and label == "yes":
        return "TP"
    if prediction == "no" and label == "no":
        return "TN"
    if prediction == "yes" and label == "no":
        return "FP"
    if prediction == "no" and label == "yes":
        return "FN"
    return "unknown"


def _counts(values: np.ndarray) -> dict[str, int]:
    return {name: int((values == name).sum()) for name in ["TP", "TN", "FP", "FN", "unknown"]}


def _accuracy(counts: dict[str, int]) -> float:
    denom = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    return (counts["TP"] + counts["TN"]) / denom if denom else math.nan


def _yes_rate(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    return float((values == "yes").mean())


if __name__ == "__main__":
    main()
