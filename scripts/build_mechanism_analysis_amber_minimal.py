#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json
from vgs.pope import classify_outcome

from mechanism_analysis_common import base_metrics_from_predictions, fieldnames, markdown_table, outcome_counts


DEFAULT_CANDIDATES = ["band1_12", "band5_16", "band9_20", "band13_24", "band17_28", "tail257_1024"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply POPE-calibrated subspace ICD to AMBER geometry.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_amber_icd/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/stage_n_external_full/amber_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins_amber/pope_margin_scores.csv")
    parser.add_argument(
        "--pope-band-scan",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/band_scan_table.csv",
    )
    parser.add_argument("--candidate-subspaces", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/amber_minimal",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_amber_minimal(
        operator_geometry_path=Path(args.operator_geometry),
        predictions_path=Path(args.predictions),
        margin_scores_path=Path(args.margin_scores),
        pope_band_scan_path=Path(args.pope_band_scan),
        candidate_subspaces=args.candidate_subspaces,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_amber_minimal_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_amber_minimal", summary_path, "ok")
    print(summary_path)


def build_amber_minimal(
    operator_geometry_path: Path,
    predictions_path: Path,
    margin_scores_path: Path,
    pope_band_scan_path: Path,
    candidate_subspaces: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    geometry = pd.read_csv(operator_geometry_path)
    predictions = pd.DataFrame(read_jsonl(predictions_path))
    margins = _load_margins(margin_scores_path)
    alpha_map = _pope_alpha_map(pope_band_scan_path)
    best_mid = _best_mid_band(pope_band_scan_path, candidate_subspaces)

    method_specs = [
        ("Full ICD", "full", alpha_map.get("full")),
        ("Band5-16 ICD", "band5_16", alpha_map.get("band5_16")),
        ("Best mid-band ICD", best_mid, alpha_map.get(best_mid)),
        ("Random12 ICD", "random12", alpha_map.get("random12")),
    ]

    rows = base_metrics_from_predictions(predictions, group_col="dimension").to_dict(orient="records")
    rows.extend(base_metrics_from_predictions(predictions, group_col=None).to_dict(orient="records"))
    for method, subspace, alpha in method_specs:
        if not subspace or alpha is None or f"dmargin_no_minus_yes_{subspace}" not in geometry.columns:
            continue
        rows.extend(_method_rows(method, subspace, float(alpha), geometry, predictions, margins))

    table_path = write_csv(output_dir / "amber_minimal_correction.csv", rows, fieldnames(rows))
    report_path = _write_report(output_dir / "amber_minimal_correction_report.md", pd.DataFrame(rows))
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "predictions_path": str(predictions_path),
        "pope_band_scan_path": str(pope_band_scan_path),
        "best_mid_band": best_mid,
        "amber_minimal_correction_path": str(table_path),
        "report_path": str(report_path),
        "num_rows": len(rows),
    }


def _method_rows(
    method: str,
    subspace: str,
    alpha: float,
    geometry: pd.DataFrame,
    predictions: pd.DataFrame,
    margins: dict[str, float],
) -> list[dict[str, Any]]:
    pred = predictions.set_index(predictions["sample_id"].astype(str)).to_dict(orient="index")
    sample_rows: list[dict[str, Any]] = []
    for row in geometry.itertuples(index=False):
        sample_id = str(row.sample_id)
        if sample_id not in pred:
            continue
        original = pred[sample_id]
        base_margin = margins.get(sample_id, float(getattr(row, "orig_no_minus_yes_logit", math.nan)))
        dmargin = float(getattr(row, f"dmargin_no_minus_yes_{subspace}"))
        adjusted = base_margin + alpha * dmargin
        final_prediction = "no" if adjusted >= 0 else "yes"
        sample_rows.append(
            {
                "dimension": str(original.get("dimension", getattr(row, "dimension", "all") or "all")),
                "label": str(original.get("label", "")),
                "original_outcome": str(original.get("outcome", "")),
                "final_prediction": final_prediction,
                "final_outcome": classify_outcome(final_prediction, str(original.get("label", ""))),
            }
        )
    sample_df = pd.DataFrame(sample_rows)
    rows: list[dict[str, Any]] = []
    for group_name, group in [*sample_df.groupby("dimension", dropna=False), ("all", sample_df)]:
        rows.append(_metric_row(method, subspace, alpha, str(group_name), group))
    return rows


def _metric_row(method: str, subspace: str, alpha: float, group_name: str, group: pd.DataFrame) -> dict[str, Any]:
    original_counts = outcome_counts(group["original_outcome"].astype(str))
    final_counts = outcome_counts(group["final_outcome"].astype(str))
    original_fp = original_counts["FP"]
    original_tp = original_counts["TP"]
    original_tn = original_counts["TN"]
    fp_fixed = int(((group["original_outcome"] == "FP") & (group["final_outcome"] == "TN")).sum())
    tp_kept = int(((group["original_outcome"] == "TP") & (group["final_outcome"] == "TP")).sum())
    tn_kept = int(((group["original_outcome"] == "TN") & (group["final_outcome"] == "TN")).sum())
    denom = sum(original_counts.values())
    final_acc = (final_counts["TP"] + final_counts["TN"]) / denom if denom else math.nan
    base_acc = (original_counts["TP"] + original_counts["TN"]) / denom if denom else math.nan
    return {
        "group": group_name,
        "method": method,
        "subspace": subspace,
        "alpha": alpha,
        "n": int(len(group)),
        "fp_reduction": fp_fixed / original_fp if original_fp else math.nan,
        "tp_preserved": tp_kept / original_tp if original_tp else math.nan,
        "tn_preserved": tn_kept / original_tn if original_tn else math.nan,
        "accuracy_delta": final_acc - base_acc if math.isfinite(final_acc) and math.isfinite(base_acc) else math.nan,
        "overall_yes_rate": float((group["final_prediction"] == "yes").mean()) if len(group) else math.nan,
        "fp_yes_rate": _yes_rate(group, "FP"),
        "tp_yes_rate": _yes_rate(group, "TP"),
        "tn_yes_rate": _yes_rate(group, "TN"),
    }


def _yes_rate(group: pd.DataFrame, outcome: str) -> float:
    subset = group[group["original_outcome"].astype(str) == outcome]
    return float((subset["final_prediction"] == "yes").mean()) if len(subset) else math.nan


def _load_margins(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "no_minus_yes_logit" not in df.columns:
        return {}
    return {str(row.sample_id): float(row.no_minus_yes_logit) for row in df.itertuples(index=False)}


def _pope_alpha_map(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(row.subspace): float(row.alpha) for row in df.itertuples(index=False)}


def _best_mid_band(path: Path, candidates: list[str]) -> str:
    if not path.exists():
        return "band5_16"
    df = pd.read_csv(path)
    view = df[df["subspace"].isin(candidates)].copy()
    if view.empty:
        return "band5_16"
    view = view.sort_values(["fp_reduction", "tp_preserved", "accuracy_delta"], ascending=[False, False, False])
    return str(view.iloc[0]["subspace"])


def _write_report(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group",
        "method",
        "subspace",
        "alpha",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
        "fp_yes_rate",
        "tp_yes_rate",
        "tn_yes_rate",
    ]
    lines = ["# AMBER Minimal Correction", "", markdown_table(table, columns=cols), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
