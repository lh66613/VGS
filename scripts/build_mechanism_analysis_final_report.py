#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_json

from mechanism_analysis_common import markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact mechanism-analysis report.")
    parser.add_argument(
        "--root-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis",
    )
    parser.add_argument("--margin-summary", default="outputs/margins/dump_pope_margins_summary.json")
    parser.add_argument(
        "--reference-geometry",
        default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv",
    )
    parser.add_argument(
        "--reference-band-scan",
        default="outputs/mechanism_mitigation/stage2_subspace_vcd/alpha_sweep.csv",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_final_report(
        Path(args.root_dir),
        Path(args.margin_summary),
        Path(args.reference_geometry),
        Path(args.reference_band_scan),
    )
    summary_path = write_json(Path(args.root_dir) / "build_mechanism_analysis_final_report_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_final_report", summary_path, "ok")
    print(summary_path)


def build_final_report(
    root_dir: Path,
    margin_summary: Path,
    reference_geometry: Path,
    reference_band_scan: Path,
) -> dict[str, Any]:
    sections: list[str] = [
        "# Mechanism Analysis Report",
        "",
        "This report freezes the existing paper-ready baseline and summarizes new mechanism-analysis outputs when present.",
        "",
    ]
    artifacts: dict[str, str] = {}
    sections.extend(_protocol_section(root_dir, margin_summary))
    sections.extend(_reference_drift_section(root_dir, reference_geometry, reference_band_scan))
    sections.extend(
        _section(
            root_dir / "drift_audit" / "drift_conclusion.csv",
            "Drift Diagnosis",
            columns=["priority", "finding", "evidence"],
        )
    )
    sections.extend(
        _section(
            root_dir / "exact_reproduction" / "exact_reproduction_success.csv",
            "Frozen Exact Reproduction",
        )
    )

    sections.extend(_section(root_dir / "frozen_baseline" / "frozen_main_table.csv", "Frozen Baseline"))
    sections.extend(_canonical_artifact_section(root_dir))
    sections.extend(
        _section(
            root_dir / "frozen_spectrum_curve_7b" / "frozen_spectrum_curve.csv",
            "Frozen Stride-4 Spectrum Curve",
            columns=[
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
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "frozen_spectrum_curve_7b" / "frozen_spectrum_random_control.csv",
            "Frozen Spectrum Random Controls",
        )
    )
    sections.extend(
        _section(
            root_dir / "frozen_spectrum_curve_7b" / "frozen_peak_no_bias_audit.csv",
            "Frozen Peak-Level No-Bias Audit",
            columns=[
                "method",
                "subspace",
                "alpha",
                "fp_yes_rate",
                "tp_yes_rate",
                "tn_yes_rate",
                "overall_yes_rate",
                "fp_reduction",
                "tp_preserved",
                "accuracy_delta",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "frozen_flipped_subset_7b" / "frozen_flipped_subset_logit_shift.csv",
            "Frozen Flipped Subset Logit Shift",
            columns=[
                "subspace",
                "group",
                "transition",
                "n",
                "mean_delta_no_yes",
                "mean_alpha_delta_no_yes",
                "mean_base_no_yes",
                "mean_adjusted_no_yes",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "frozen_split_spectral_selection_7b" / "split_robustness_summary.csv",
            "Frozen Split Spectral Selection",
            columns=[
                "pair",
                "setting",
                "subspace",
                "alpha",
                "fp_reduction",
                "tp_preserved",
                "accuracy_delta",
                "calibration_fp_reduction",
                "calibration_tp_preserved",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "band_scan_7b" / "band_scan_table.csv",
            "Drifted Vlm-Exp Contiguous Band Scan",
            columns=[
                "subspace",
                "alpha",
                "fp_reduction",
                "tp_preserved",
                "accuracy_delta",
                "fp_yes_rate",
                "tp_yes_rate",
                "tn_yes_rate",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "logit_shift_7b" / "logit_shift_gap_table.csv",
            "Logit Shift Decomposition",
            columns=[
                "subspace",
                "mean_dmargin_fp",
                "mean_dmargin_tp",
                "fp_tp_shift_gap",
                "fp_reduction",
                "tp_preserved",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "internal_contribution_7b" / "single_direction_results.csv",
            "Band5-16 Single Direction Contribution",
            columns=["subspace", "alpha", "fp_reduction", "tp_preserved", "accuracy_delta"],
        )
    )
    sections.extend(
        _section(
            root_dir / "internal_contribution_7b" / "cumulative_results.csv",
            "Band5-16 Cumulative Contribution",
            columns=["subspace", "alpha", "fp_reduction", "tp_preserved", "accuracy_delta"],
        )
    )
    sections.extend(
        _section(
            root_dir / "internal_contribution_7b" / "leave_one_out_results.csv",
            "Band5-16 Leave-One-Out Contribution",
            columns=[
                "removed_direction",
                "subspace",
                "fp_reduction",
                "fp_reduction_drop_vs_band5_16",
                "tp_preserved",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "spectrum_curve_7b" / "spectrum_curve.csv",
            "Drifted Vlm-Exp Spectrum Curve",
            columns=[
                "subspace",
                "window_start",
                "available",
                "alpha",
                "fp_reduction",
                "tp_damage",
                "tp_preserved",
                "accuracy_delta",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "spectrum_curve_7b" / "spectrum_random_control.csv",
            "Spectrum Random Control",
        )
    )
    sections.extend(
        _section(
            root_dir / "flipped_subset_7b" / "flipped_subset_logit_shift.csv",
            "Drifted Vlm-Exp Flipped Subset Logit Shift",
            columns=[
                "subspace",
                "group",
                "changed",
                "n",
                "mean_delta_no_yes",
                "mean_alpha_delta_no_yes",
                "mean_base_no_yes",
                "mean_adjusted_no_yes",
            ],
        )
    )
    sections.extend(
        _section(
            root_dir / "split_robustness_7b" / "split_robustness_summary.csv",
            "Drifted Vlm-Exp Split Robustness",
            columns=[
                "pair",
                "setting",
                "subspace",
                "fp_reduction",
                "tp_preserved",
                "accuracy_delta",
            ],
        )
    )

    output_path = root_dir / "mechanism_analysis_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    artifacts["report_path"] = str(output_path)
    return artifacts


def _protocol_section(root_dir: Path, margin_summary: Path) -> list[str]:
    geometry_summary = root_dir / "operator_geometry_7b_icd" / "operator_geometry_summary.json"
    if not geometry_summary.exists() or not margin_summary.exists():
        return ["## Protocol Audit", "", "_Pending: geometry or margin summary not found._", ""]
    geometry = json.loads(geometry_summary.read_text(encoding="utf-8"))
    margin = json.loads(margin_summary.read_text(encoding="utf-8"))
    geom_yes = geometry.get("yes_token_ids", [])
    geom_no = geometry.get("no_token_ids", [])
    margin_yes = margin.get("yes_token_ids", [])
    margin_no = margin.get("no_token_ids", [])
    status = "matched" if geom_yes == margin_yes and geom_no == margin_no else "mismatch"
    note = (
        "Geometry and margin token IDs match."
        if status == "matched"
        else "Geometry and margin token IDs differ; rerun geometry with explicit token IDs before making protocol-level comparisons to frozen paper tables."
    )
    rows = pd.DataFrame(
        [
            {
                "item": "geometry_yes_token_ids",
                "value": " ".join(map(str, geom_yes)),
            },
            {
                "item": "margin_yes_token_ids",
                "value": " ".join(map(str, margin_yes)),
            },
            {
                "item": "geometry_no_token_ids",
                "value": " ".join(map(str, geom_no)),
            },
            {
                "item": "margin_no_token_ids",
                "value": " ".join(map(str, margin_no)),
            },
        ]
    )
    return ["## Protocol Audit", "", f"Status: `{status}`. {note}", "", markdown_table(rows), ""]


def _canonical_artifact_section(root_dir: Path) -> list[str]:
    path = root_dir / "canonical_protocol.md"
    if not path.exists():
        return ["## Canonical Protocol", "", f"_Pending: `{path}` not found._", ""]
    return [
        "## Canonical Protocol",
        "",
        "Unless otherwise stated, all main results use the frozen exact-reproduction pipeline.",
        "",
        f"Artifact: `{path}`",
        "",
    ]


def _reference_drift_section(root_dir: Path, reference_geometry: Path, reference_band_scan: Path) -> list[str]:
    geometry = root_dir / "operator_geometry_7b_icd" / "operator_geometry.csv"
    current_scan = root_dir / "band_scan_7b" / "stage2" / "alpha_sweep.csv"
    if not geometry.exists() or not reference_geometry.exists():
        return ["## Reference Drift Audit", "", "_Pending: current or reference geometry not found._", ""]
    columns = [
        "sample_id",
        "operator",
        "orig_no_minus_yes_logit",
        "neg_no_minus_yes_logit",
        "delta_norm_sq",
        "energy_band5_16",
        "dmargin_no_minus_yes_band5_16",
        "dmargin_no_minus_yes_full",
    ]
    try:
        current = pd.read_csv(geometry, usecols=columns)
        reference = pd.read_csv(reference_geometry, usecols=columns)
    except ValueError as exc:
        return ["## Reference Drift Audit", "", f"_Could not compare geometry columns: `{exc}`._", ""]
    current = current[current["operator"].astype(str) == "icd_blind"]
    reference = reference[reference["operator"].astype(str) == "icd_blind"]
    merged = current.merge(reference, on=["sample_id", "operator"], suffixes=("_current", "_reference"))
    rows = []
    for column in columns[2:]:
        diff = (merged[f"{column}_current"] - merged[f"{column}_reference"]).abs()
        rows.append(
            {
                "column": column,
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )
    status = "matched" if all(row["max_abs_diff"] < 1e-5 for row in rows) else "drift"
    note = (
        "Current expanded geometry numerically matches the reference geometry."
        if status == "matched"
        else "Current expanded geometry differs from the reference frozen geometry; compare trends within this run, and use the frozen table only as a historical baseline."
    )
    alpha_rows = _reference_alpha_rows(current_scan, reference_band_scan)
    lines = ["## Reference Drift Audit", "", f"Status: `{status}`. {note}", "", markdown_table(pd.DataFrame(rows)), ""]
    if alpha_rows:
        lines.extend(["Band5-16 alpha=0.5 comparison:", "", markdown_table(pd.DataFrame(alpha_rows)), ""])
    return lines


def _reference_alpha_rows(current_scan: Path, reference_band_scan: Path) -> list[dict[str, Any]]:
    rows = []
    for label, path in [("current", current_scan), ("reference", reference_band_scan)]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        subset = df[
            (df["operator"].astype(str) == "icd_blind")
            & (df["subspace"].astype(str) == "band5_16")
            & (df["split"].astype(str) == "test")
            & (df["alpha"].astype(float) == 0.5)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "source": label,
                "fp_reduction": float(row["fp_reduction"]),
                "tp_preserved": float(row["tp_preserved"]),
                "accuracy_delta": float(row["accuracy_delta"]),
            }
        )
    return rows


def _section(path: Path, title: str, columns: list[str] | None = None) -> list[str]:
    lines = [f"## {title}", ""]
    if not path.exists():
        lines.extend([f"_Pending: `{path}` not found._", ""])
        return lines
    df = pd.read_csv(path)
    lines.extend([markdown_table(df, columns=columns), "", f"Artifact: `{path}`", ""])
    return lines


if __name__ == "__main__":
    main()
