#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from analyze_mechanism_mitigation_stage2 import analyze_stage2
from mechanism_analysis_common import add_metric_rates, fieldnames, markdown_table, require_present_subspaces


SINGLE_DIRECTIONS = [f"v{i}" for i in range(5, 17)]
CUMULATIVE_BANDS = ["band5_5", "band5_6", "band5_8", "band5_12", "band5_16", "band5_20"]
LEAVE_ONE_OUT = [f"band5_16_minus_v{i}" for i in range(5, 17)]
DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Band5-16 internal contribution.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/internal_contribution_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_internal_contribution(
        operator_geometry_path=Path(args.operator_geometry),
        predictions_path=Path(args.predictions),
        margin_scores_path=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(
        Path(args.output_dir) / "build_mechanism_analysis_internal_contribution_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "build_mechanism_analysis_internal_contribution", summary_path, "ok")
    print(summary_path)


def build_internal_contribution(
    operator_geometry_path: Path,
    predictions_path: Path,
    margin_scores_path: Path,
    split_dir: Path,
    alphas: list[float],
    min_tp_preserved: float,
    output_dir: Path,
) -> dict[str, Any]:
    subspaces = SINGLE_DIRECTIONS + CUMULATIVE_BANDS + LEAVE_ONE_OUT
    geometry = pd.read_csv(operator_geometry_path)
    present = require_present_subspaces(geometry, subspaces, operator_geometry_path)
    stage2_dir = output_dir / "stage2"
    analyze_stage2(
        operator_geometry_path=operator_geometry_path,
        predictions_path=predictions_path,
        margin_scores_path=margin_scores_path,
        subspaces=present,
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

    single = _sorted_subset(selected, SINGLE_DIRECTIONS)
    cumulative = _sorted_subset(selected, CUMULATIVE_BANDS)
    leave_one_out = _with_leave_one_out_importance(_sorted_subset(selected, LEAVE_ONE_OUT), selected)

    paths = {
        "single_direction_results": write_csv(
            output_dir / "single_direction_results.csv",
            single.to_dict(orient="records"),
            fieldnames(single.to_dict(orient="records")),
        ),
        "cumulative_results": write_csv(
            output_dir / "cumulative_results.csv",
            cumulative.to_dict(orient="records"),
            fieldnames(cumulative.to_dict(orient="records")),
        ),
        "leave_one_out_results": write_csv(
            output_dir / "leave_one_out_results.csv",
            leave_one_out.to_dict(orient="records"),
            fieldnames(leave_one_out.to_dict(orient="records")),
        ),
    }
    report_path = _write_report(output_dir / "internal_contribution_report.md", single, cumulative, leave_one_out)
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "present_subspaces": present,
        "stage2_dir": str(stage2_dir),
        **{key + "_path": str(value) for key, value in paths.items()},
        "report_path": str(report_path),
    }


def _sorted_subset(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rank = {name: idx for idx, name in enumerate(order)}
    out = df[df["subspace"].isin(order)].copy()
    if out.empty:
        return out
    out["_rank"] = out["subspace"].map(rank).fillna(len(rank)).astype(int)
    return out.sort_values(["operator", "layer", "_rank"]).drop(columns=["_rank"])


def _with_leave_one_out_importance(loo: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if loo.empty:
        return loo
    base = selected[selected["subspace"].astype(str) == "band5_16"]
    if base.empty:
        return loo
    base_row = base.iloc[0]
    out = loo.copy()
    out["removed_direction"] = out["subspace"].astype(str).str.extract(r"minus_(v\d+)", expand=False)
    out["fp_reduction_drop_vs_band5_16"] = float(base_row["fp_reduction"]) - out["fp_reduction"].astype(float)
    out["tp_preserved_delta_vs_band5_16"] = out["tp_preserved"].astype(float) - float(base_row["tp_preserved"])
    out["accuracy_delta_vs_band5_16"] = out["accuracy_delta"].astype(float) - float(base_row["accuracy_delta"])
    return out.sort_values("fp_reduction_drop_vs_band5_16", ascending=False)


def _write_report(path: Path, single: pd.DataFrame, cumulative: pd.DataFrame, leave_one_out: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "subspace",
        "alpha",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
        "fp_yes_rate",
        "tp_yes_rate",
        "tn_yes_rate",
    ]
    loo_cols = cols + ["removed_direction", "fp_reduction_drop_vs_band5_16"]
    lines = [
        "# Band5-16 Internal Contribution",
        "",
        "## Single Directions",
        "",
        markdown_table(single[[col for col in cols if col in single.columns]] if not single.empty else single),
        "",
        "## Cumulative Bands",
        "",
        markdown_table(cumulative[[col for col in cols if col in cumulative.columns]] if not cumulative.empty else cumulative),
        "",
        "## Leave One Out",
        "",
        markdown_table(
            leave_one_out[[col for col in loo_cols if col in leave_one_out.columns]]
            if not leave_one_out.empty
            else leave_one_out
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
