#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from analyze_mechanism_mitigation_stage2 import analyze_stage2
from mechanism_analysis_common import add_metric_rates, fieldnames, markdown_table


DEFAULT_SUBSPACES = ["full", "band5_16", "top4_complement", "random12"]
DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce frozen paper-table rows from reference geometry in the current CPU code.")
    parser.add_argument("--reference-geometry", default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/exact_reproduction",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_exact_reproduction(
        reference_geometry=Path(args.reference_geometry),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        subspaces=args.subspaces,
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_exact_reproduction_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_exact_reproduction", summary_path, "ok")
    print(summary_path)


def build_exact_reproduction(
    reference_geometry: Path,
    predictions: Path,
    margin_scores: Path,
    split_dir: Path,
    subspaces: list[str],
    alphas: list[float],
    min_tp_preserved: float,
    output_dir: Path,
) -> dict[str, Any]:
    stage2_dir = output_dir / "stage2"
    analyze_stage2(
        operator_geometry_path=reference_geometry,
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
    rows = selected.to_dict(orient="records")
    table_path = write_csv(output_dir / "exact_reproduction_table.csv", rows, fieldnames(rows))
    success = _success_row(selected)
    success_path = write_csv(output_dir / "exact_reproduction_success.csv", [success], fieldnames([success]))
    report_path = _write_report(output_dir / "exact_reproduction_report.md", selected, success)
    return {
        "reference_geometry": str(reference_geometry),
        "stage2_dir": str(stage2_dir),
        "exact_reproduction_table_path": str(table_path),
        "success_path": str(success_path),
        "report_path": str(report_path),
        "success": bool(success["success"]),
    }


def _success_row(selected: pd.DataFrame) -> dict[str, Any]:
    row = selected[
        (selected["operator"].astype(str) == "icd_blind")
        & (selected["subspace"].astype(str) == "band5_16")
    ]
    if row.empty:
        return {"target": "Band5-16 ICD", "success": False, "notes": "Missing row."}
    item = row.iloc[0]
    target = {"fp_reduction": 0.39622641509433965, "tp_preserved": 0.9594843462246777, "accuracy_delta": 0.0007407407407408195}
    return {
        "target": "Band5-16 ICD",
        "success": bool(
            abs(float(item["fp_reduction"]) - target["fp_reduction"]) < 1e-9
            and abs(float(item["tp_preserved"]) - target["tp_preserved"]) < 1e-9
            and abs(float(item["accuracy_delta"]) - target["accuracy_delta"]) < 1e-9
        ),
        "fp_reduction": float(item["fp_reduction"]),
        "target_fp_reduction": target["fp_reduction"],
        "tp_preserved": float(item["tp_preserved"]),
        "target_tp_preserved": target["tp_preserved"],
        "accuracy_delta": float(item["accuracy_delta"]),
        "target_accuracy_delta": target["accuracy_delta"],
        "alpha": float(item["alpha"]),
    }


def _write_report(path: Path, selected: pd.DataFrame, success: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    lines = [
        "# Frozen Exact Reproduction",
        "",
        f"Band5-16 exact target reproduced: `{success['success']}`.",
        "",
        markdown_table(selected, columns=columns),
        "",
        "## Success Row",
        "",
        markdown_table(pd.DataFrame([success])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
