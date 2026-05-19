#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


DEFAULT_SPECS = [
    "band5_16:FP",
    "band29_40:FP",
    "top4:TP",
    "full:FP",
    "full:TP",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down logit shift by flipped and unchanged subsets.")
    parser.add_argument(
        "--sample-predictions",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/stage2/sample_predictions.csv",
    )
    parser.add_argument(
        "--selected-table",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/band_scan_table.csv",
    )
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/flipped_subset_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_flipped_subset(
        sample_predictions=Path(args.sample_predictions),
        selected_table=Path(args.selected_table),
        specs=args.specs,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_flipped_subset_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_flipped_subset", summary_path, "ok")
    print(summary_path)


def build_flipped_subset(
    sample_predictions: Path,
    selected_table: Path,
    specs: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    samples = pd.read_csv(sample_predictions)
    selected = pd.read_csv(selected_table)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        subspace, group = _parse_spec(spec)
        selected_row = selected[
            (selected["operator"].astype(str) == "icd_blind")
            & (selected["subspace"].astype(str) == subspace)
        ]
        if selected_row.empty:
            continue
        alpha = float(selected_row.iloc[0]["alpha"])
        subset = samples[
            (samples["operator"].astype(str) == "icd_blind")
            & (samples["subspace"].astype(str) == subspace)
            & (samples["split"].astype(str) == "test")
            & (samples["original_outcome"].astype(str) == group)
            & (samples["alpha"].astype(float) == alpha)
        ].copy()
        rows.extend(_group_rows(subspace, group, alpha, subset))
    table_path = write_csv(output_dir / "flipped_subset_logit_shift.csv", rows, fieldnames(rows))
    report_path = _write_report(output_dir / "flipped_subset_report.md", pd.DataFrame(rows))
    return {
        "sample_predictions": str(sample_predictions),
        "selected_table": str(selected_table),
        "flipped_subset_path": str(table_path),
        "report_path": str(report_path),
        "num_rows": len(rows),
    }


def _group_rows(subspace: str, group: str, alpha: float, subset: pd.DataFrame) -> list[dict[str, Any]]:
    if group == "FP":
        statuses = [
            ("Yes->No", subset["final_outcome"].astype(str) == "TN"),
            ("Yes->Yes", subset["final_outcome"].astype(str) == "FP"),
        ]
    elif group == "TP":
        statuses = [
            ("Yes->No", subset["final_outcome"].astype(str) == "FN"),
            ("Yes->Yes", subset["final_outcome"].astype(str) == "TP"),
        ]
    else:
        statuses = [("changed", subset["final_outcome"].astype(str) != group), ("unchanged", subset["final_outcome"].astype(str) == group)]
    rows: list[dict[str, Any]] = []
    for changed, mask in statuses:
        data = subset[mask].copy()
        rows.append(
            {
                "subspace": subspace,
                "group": group,
                "changed": changed,
                "alpha": alpha,
                "n": int(len(data)),
                "mean_delta_no_yes": float(data["dmargin_no_minus_yes"].mean()) if len(data) else float("nan"),
                "mean_alpha_delta_no_yes": float((alpha * data["dmargin_no_minus_yes"]).mean()) if len(data) else float("nan"),
                "median_delta_no_yes": float(data["dmargin_no_minus_yes"].median()) if len(data) else float("nan"),
                "mean_base_no_yes": float(data["base_no_minus_yes_logit"].mean()) if len(data) else float("nan"),
                "mean_adjusted_no_yes": float(data["adjusted_no_minus_yes_logit"].mean()) if len(data) else float("nan"),
            }
        )
    return rows


def _parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Spec must be subspace:group, got {spec}")
    subspace, group = spec.split(":", 1)
    return subspace.strip(), group.strip()


def _write_report(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "subspace",
        "group",
        "changed",
        "n",
        "mean_delta_no_yes",
        "mean_alpha_delta_no_yes",
        "mean_base_no_yes",
        "mean_adjusted_no_yes",
    ]
    lines = [
        "# Flipped Subset Logit Shift",
        "",
        "Rows split FP/TP samples by whether the calibrated intervention changed the yes/no outcome.",
        "",
        markdown_table(table, columns=columns),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
