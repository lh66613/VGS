#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen flipped-subset logit shift for top spectral bands.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_operator_geometry/operator_geometry.csv",
    )
    parser.add_argument(
        "--selected-table",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_spectrum_curve_7b/frozen_spectrum_selected.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--subspaces", nargs="+", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_flipped_subset_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_frozen_flipped_subset(
        operator_geometry=Path(args.operator_geometry),
        selected_table=Path(args.selected_table),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        requested_subspaces=args.subspaces,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_frozen_flipped_subset_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_frozen_flipped_subset", summary_path, "ok")
    print(summary_path)


def build_frozen_flipped_subset(
    operator_geometry: Path,
    selected_table: Path,
    predictions: Path,
    margin_scores: Path,
    split_dir: Path,
    requested_subspaces: list[str] | None,
    output_dir: Path,
) -> dict[str, Any]:
    frame = _load_frame(operator_geometry, predictions, margin_scores, split_dir)
    selected = pd.read_csv(selected_table)
    subspaces = requested_subspaces or _default_subspaces(selected)
    rows: list[dict[str, Any]] = []
    for subspace in subspaces:
        match = selected[selected["subspace"].astype(str) == subspace]
        if match.empty or f"dmargin_no_minus_yes_{subspace}" not in frame.columns:
            continue
        alpha = float(match.iloc[0]["alpha"])
        rows.extend(_transition_rows(frame, subspace, alpha, group="FP"))
        rows.extend(_transition_rows(frame, subspace, alpha, group="TP"))
    table_path = write_csv(output_dir / "frozen_flipped_subset_logit_shift.csv", rows, fieldnames(rows))
    report_path = _write_report(output_dir / "frozen_flipped_subset_report.md", pd.DataFrame(rows), subspaces)
    return {
        "operator_geometry": str(operator_geometry),
        "selected_table": str(selected_table),
        "subspaces": subspaces,
        "table_path": str(table_path),
        "report_path": str(report_path),
        "num_rows": len(rows),
    }


def _default_subspaces(selected: pd.DataFrame) -> list[str]:
    windows = selected[selected["subspace"].astype(str).str.fullmatch(r"band\d+_\d+", na=False)].copy()
    top_windows = windows.sort_values(["fp_reduction", "tp_preserved", "accuracy_delta"], ascending=[False, False, False])[
        "subspace"
    ].astype(str).tolist()[:3]
    names = ["band5_16", *top_windows, "top4", "full", "random12"]
    return list(dict.fromkeys(names))


def _transition_rows(frame: pd.DataFrame, subspace: str, alpha: float, group: str) -> list[dict[str, Any]]:
    test = frame[(frame["split_eval"].astype(str) == "test") & (frame["original_outcome_eval"].astype(str) == group)].copy()
    if test.empty:
        return []
    dmargin = test[f"dmargin_no_minus_yes_{subspace}"].to_numpy(dtype=float)
    base = test["base_no_minus_yes_logit"].to_numpy(dtype=float)
    adjusted = base + alpha * dmargin
    final_prediction = np.where(adjusted >= 0, "no", "yes")
    labels = test["label_eval"].astype(str).to_numpy()
    final_outcome = np.array([_classify(pred, label) for pred, label in zip(final_prediction, labels)])
    if group == "FP":
        masks = [("Yes->No", final_outcome == "TN"), ("Yes->Yes", final_outcome == "FP")]
    else:
        masks = [("Yes->No", final_outcome == "FN"), ("Yes->Yes", final_outcome == "TP")]
    rows: list[dict[str, Any]] = []
    for transition, mask in masks:
        rows.append(
            {
                "subspace": subspace,
                "group": group,
                "transition": transition,
                "alpha": alpha,
                "n": int(mask.sum()),
                "mean_delta_no_yes": _safe_mean(dmargin[mask]),
                "mean_alpha_delta_no_yes": _safe_mean(alpha * dmargin[mask]),
                "median_delta_no_yes": _safe_median(dmargin[mask]),
                "mean_base_no_yes": _safe_mean(base[mask]),
                "mean_adjusted_no_yes": _safe_mean(adjusted[mask]),
            }
        )
    return rows


def _load_frame(operator_geometry: Path, predictions: Path, margin_scores: Path, split_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(operator_geometry)
    df = df[df["operator"].astype(str) == "icd_blind"].copy()
    pred_rows = {str(row["sample_id"]): row for row in read_jsonl(predictions)}
    margins = pd.read_csv(margin_scores)
    margin_lookup = {str(row.sample_id): float(row.no_minus_yes_logit) for row in margins.itertuples(index=False)}
    split_map = _load_split_map(split_dir)
    df["split_eval"] = [split_map.get(str(row.sample_id), "unassigned") for row in df.itertuples(index=False)]
    df["label_eval"] = [str(pred_rows.get(str(row.sample_id), {}).get("label", getattr(row, "label", ""))) for row in df.itertuples(index=False)]
    df["original_outcome_eval"] = [
        str(pred_rows.get(str(row.sample_id), {}).get("outcome", getattr(row, "outcome", ""))) for row in df.itertuples(index=False)
    ]
    df["base_no_minus_yes_logit"] = [
        float(margin_lookup.get(str(row.sample_id), getattr(row, "orig_no_minus_yes_logit", math.nan))) for row in df.itertuples(index=False)
    ]
    return df


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


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def _safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def _write_report(path: Path, table: pd.DataFrame, subspaces: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "subspace",
        "group",
        "transition",
        "n",
        "mean_delta_no_yes",
        "mean_alpha_delta_no_yes",
        "mean_base_no_yes",
        "mean_adjusted_no_yes",
    ]
    lines = [
        "# Frozen Flipped Subset Logit Shift",
        "",
        "Unless otherwise stated, all main results use the frozen exact-reproduction pipeline.",
        "",
        f"Selected subspaces: `{', '.join(subspaces)}`.",
        "",
        markdown_table(table, columns=columns),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
