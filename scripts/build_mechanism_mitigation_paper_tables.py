#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-ready mitigation tables from the mechanism mitigation follow-up package."
    )
    parser.add_argument("--followup-dir", default="outputs/mechanism_mitigation/followup")
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/paper_tables")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_tables(args.followup_dir, args.output_dir)
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_mitigation_paper_tables_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_mitigation_paper_tables", summary_path, "ok")
    print(summary_path)


def build_tables(followup_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    followup = Path(followup_dir)
    output = Path(output_dir)
    matched = pd.read_csv(followup / "matched_tp_safe_operating_points.csv")
    random = pd.read_csv(followup / "random_control_distribution.csv")
    yes = pd.read_csv(followup / "yes_rate_no_bias_audit.csv")
    bootstrap = pd.read_csv(followup / "bootstrap_comparisons.csv")
    reverse = pd.read_csv(followup / "reverse_split_results.csv")

    table_a = _table_a(matched, random, bootstrap)
    table_b = _table_b(random)
    table_c = _table_c(yes)
    table_d = _table_d(reverse)
    table_e = _table_e(matched, yes, bootstrap)

    table_a_path = write_csv(output / "table_a_tp_safe_mitigation.csv", table_a, _fieldnames(table_a))
    table_b_path = write_csv(output / "table_b_random_control_specificity.csv", table_b, _fieldnames(table_b))
    table_c_path = write_csv(output / "table_c_no_bias_audit.csv", table_c, _fieldnames(table_c))
    table_d_path = write_csv(output / "table_d_reverse_split_replication.csv", table_d, _fieldnames(table_d))
    table_e_path = write_csv(output / "table_e_best_vs_vcd_baseline.csv", table_e, _fieldnames(table_e))
    markdown_path = _write_markdown(
        output / "mechanism_mitigation_paper_tables.md",
        {
            "Table A: TP-Safe Mitigation Comparison": table_a,
            "Table B: Random Control Specificity": table_b,
            "Table C: Yes-Rate / No-Bias Audit": table_c,
            "Table D: Reverse Split Replication": table_d,
            "Table E: Best Method vs VCD and Baselines": table_e,
        },
    )
    comparison_path = _write_single_table_markdown(
        output / "best_vs_vcd_baseline_comparison.md",
        "Best Method vs VCD and Baselines",
        table_e,
    )
    return {
        "followup_dir": str(followup),
        "table_a_path": str(table_a_path),
        "table_b_path": str(table_b_path),
        "table_c_path": str(table_c_path),
        "table_d_path": str(table_d_path),
        "table_e_path": str(table_e_path),
        "markdown_path": str(markdown_path),
        "comparison_path": str(comparison_path),
    }


def _table_a(
    matched: pd.DataFrame,
    random: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "method": "Base",
            "fp_reduction": "",
            "tp_preserved": "",
            "accuracy_delta": "0.000",
            "notes": "No correction.",
        }
    ]
    always = _bootstrap_method_values(bootstrap, "Gated ICD vs Always ICD", use_a=False)
    if always:
        rows.append(
            {
                "method": "Always ICD",
                "fp_reduction": _fmt(always["fp_reduction"]),
                "tp_preserved": _fmt(always["tp_preserved"]),
                "accuracy_delta": _fmt(always["accuracy_delta"]),
                "notes": "Always-on contrast; higher TP damage.",
            }
        )
    for method, display, note in [
        ("Full ICD", "Full ICD TP-safe", "Calibrated under TP preserved >= 0.95."),
        ("Band5-16 ICD", "Band5-16 ICD", "Ours; TP-safe subspace-filtered ICD."),
        ("Top4-complement ICD", "Top4-complement ICD", "Backbone-removed ICD variant."),
    ]:
        row = _matched_row(matched, method)
        if row:
            rows.append(
                {
                    "method": display,
                    "fp_reduction": _fmt(row["test_fp_reduction"]),
                    "tp_preserved": _fmt(row["test_tp_preserved"]),
                    "accuracy_delta": _fmt(row["test_accuracy_delta"]),
                    "notes": f"{note} alpha={_fmt(row['selected_alpha'])}.",
                }
            )
    rand12 = _random_row(random, "Band5-16 ICD")
    if rand12:
        rows.extend(
            [
                {
                    "method": "Random12 mean",
                    "fp_reduction": _fmt(rand12["random_fp_mean"]),
                    "tp_preserved": _fmt(rand12["random_tp_mean"]),
                    "accuracy_delta": _fmt(rand12["random_acc_delta_mean"]),
                    "notes": "Mean over 20 random 12-dim controls.",
                },
                {
                    "method": "Random12 best",
                    "fp_reduction": _fmt(rand12["random_fp_max"]),
                    "tp_preserved": "",
                    "accuracy_delta": "",
                    "notes": "Best FP reduction among 20 random 12-dim controls.",
                },
            ]
        )
    gated = _bootstrap_method_values(bootstrap, "Gated ICD vs Always ICD", use_a=True)
    if gated:
        rows.append(
            {
                "method": "Gated ICD",
                "fp_reduction": _fmt(gated["fp_reduction"]),
                "tp_preserved": _fmt(gated["tp_preserved"]),
                "accuracy_delta": _fmt(gated["accuracy_delta"]),
                "notes": "Geometry-gated routing at ~30% trigger.",
            }
        )
    return rows


def _table_b(random: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = {
        "Band5-16 ICD": "Band5-16",
        "Top4-complement ICD": "Top4-comp",
        "Tail VCD-diffusion": "Tail VCD",
    }
    for row in random.itertuples(index=False):
        if row.target_method not in labels:
            continue
        rows.append(
            {
                "method": labels[row.target_method],
                "fp_reduction": _fmt(row.target_fp_reduction),
                "random_mean": _fmt(row.random_fp_mean),
                "random_range": f"{_fmt(row.random_fp_min)}-{_fmt(row.random_fp_max)}",
                "percentile": _fmt_percentile(row.target_fp_percentile),
                "beats": f"{int(row.target_outperforms_random_n)}/{int(row.n_random)}",
                "notes": f"Random family: {row.random_family}",
            }
        )
    return rows


def _table_c(yes: pd.DataFrame) -> list[dict[str, Any]]:
    wanted = {
        "Base": "Base",
        "Always ICD": "Always ICD",
        "Full ICD": "Full ICD TP-safe",
        "Band5-16 ICD": "Band5-16 ICD",
        "Random12 ICD": "Random12 ICD",
        "Top4-complement ICD": "Top4-complement ICD",
        "Gated ICD": "Gated ICD",
    }
    rows: list[dict[str, Any]] = []
    for method, display in wanted.items():
        subset = yes[yes["method_label"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "method": display,
                "overall_yes_rate": _fmt(row["overall_yes_rate"]),
                "tp_yes_rate": _fmt(row["tp_yes_rate"]),
                "fp_yes_rate": _fmt(row["fp_yes_rate"]),
                "tn_yes_rate": _fmt(row["tn_yes_rate"]),
                "fn_rate_after": _fmt(row["fn_rate_after"]),
                "accuracy": _fmt(row["accuracy_after"]),
            }
        )
    return rows


def _table_d(reverse: pd.DataFrame) -> list[dict[str, Any]]:
    wanted = {
        "Full-icd_blind": "Full ICD TP-safe",
        "Band5-16-icd_blind": "Band5-16 ICD",
        "Random12-icd_blind": "Random12 ICD",
        "Top4-Complement-icd_blind": "Top4-complement ICD",
        "RandomTailDim-icd_blind": "Random-tail ICD",
        "Full-vcd_diffusion": "Full VCD-diffusion",
        "Tail257-1024-vcd_diffusion": "Tail VCD-diffusion",
    }
    rows: list[dict[str, Any]] = []
    for method, display in wanted.items():
        subset = reverse[reverse["method"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "method": display,
                "calibrated_on": row.get("calibrated_on_split", ""),
                "tested_on": row.get("tested_on_split", ""),
                "alpha": _fmt(row.get("alpha", "")),
                "fp_reduction": _fmt(row.get("fp_reduction", "")),
                "tp_preserved": _fmt(row.get("tp_preserved", "")),
                "accuracy_delta": _fmt(row.get("accuracy_delta", "")),
                "notes": "Reverse split, TP-safe calibration.",
            }
        )
    return rows


def _table_e(
    matched: pd.DataFrame,
    yes: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> list[dict[str, Any]]:
    best = _matched_row(matched, "Band5-16 ICD")
    best_fp = float(best.get("test_fp_reduction", math.nan))
    rows: list[dict[str, Any]] = []

    base_yes = _yes_row(yes, "Base")
    if base_yes:
        rows.append(
            {
                "method": "Base",
                "family": "Baseline",
                "setting": "none",
                "fp_reduction": "",
                "fp_gap_vs_best": "",
                "tp_preserved": "",
                "accuracy_delta": "0.000",
                "overall_yes_rate": _fmt(base_yes["overall_yes_rate"]),
                "fp_yes_rate": _fmt(base_yes["fp_yes_rate"]),
                "takeaway": "No correction baseline.",
            }
        )

    for method, display, family, takeaway in [
        (
            "Full VCD-diffusion",
            "Full VCD-diffusion",
            "VCD",
            "TP-safe, but weak FP reduction and accuracy drop.",
        ),
        (
            "Tail VCD-diffusion",
            "Tail VCD-diffusion",
            "VCD",
            "Filtered VCD improves over Full VCD, still below Band5-16.",
        ),
        (
            "Full ICD",
            "Full ICD TP-safe",
            "ICD baseline",
            "Safe full-space ICD, lower FP reduction than Band5-16.",
        ),
        (
            "Band5-16 ICD",
            "Band5-16 ICD",
            "Ours",
            "Best TP-safe FP reduction with non-negative accuracy delta.",
        ),
    ]:
        row = _matched_row(matched, method)
        if not row:
            continue
        yes_row = _yes_row(yes, method)
        fp_reduction = float(row["test_fp_reduction"])
        rows.append(
            {
                "method": display,
                "family": family,
                "setting": f"alpha={_fmt(row['selected_alpha'])}",
                "fp_reduction": _fmt(fp_reduction),
                "fp_gap_vs_best": _fmt(fp_reduction - best_fp),
                "tp_preserved": _fmt(row["test_tp_preserved"]),
                "accuracy_delta": _fmt(row["test_accuracy_delta"]),
                "overall_yes_rate": _fmt(_metric_or_fallback(yes_row, "overall_yes_rate", row["test_yes_rate_after"])),
                "fp_yes_rate": _fmt(_metric_or_fallback(yes_row, "fp_yes_rate", 1.0 - fp_reduction)),
                "takeaway": takeaway,
            }
        )

    for comparison, use_a, display, family, setting, takeaway in [
        (
            "Gated ICD vs Always ICD",
            False,
            "Always ICD",
            "ICD baseline",
            "always-on",
            "Higher FP reduction than VCD, but unsafe TP damage.",
        ),
        (
            "Gated ICD vs Always ICD",
            True,
            "Gated ICD",
            "Ours",
            "~30% routed",
            "Same FP reduction as Always ICD with less TP damage.",
        ),
    ]:
        values = _bootstrap_method_values(bootstrap, comparison, use_a=use_a)
        yes_row = _yes_row(yes, display)
        if not values or not yes_row:
            continue
        fp_reduction = values["fp_reduction"]
        rows.append(
            {
                "method": display,
                "family": family,
                "setting": setting,
                "fp_reduction": _fmt(fp_reduction),
                "fp_gap_vs_best": _fmt(fp_reduction - best_fp),
                "tp_preserved": _fmt(values["tp_preserved"]),
                "accuracy_delta": _fmt(values["accuracy_delta"]),
                "overall_yes_rate": _fmt(yes_row["overall_yes_rate"]),
                "fp_yes_rate": _fmt(yes_row["fp_yes_rate"]),
                "takeaway": takeaway,
            }
        )
    return rows


def _bootstrap_method_values(
    bootstrap: pd.DataFrame,
    comparison: str,
    use_a: bool,
) -> dict[str, float]:
    subset = bootstrap[bootstrap["comparison"] == comparison]
    values: dict[str, float] = {}
    column = "point_a" if use_a else "point_b"
    for row in subset.itertuples(index=False):
        values[str(row.metric)] = float(getattr(row, column))
    return values


def _matched_row(matched: pd.DataFrame, method: str) -> dict[str, Any]:
    subset = matched[matched["method_label"] == method]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _random_row(random: pd.DataFrame, target_method: str) -> dict[str, Any]:
    subset = random[random["target_method"] == target_method]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _yes_row(yes: pd.DataFrame, method: str) -> dict[str, Any]:
    subset = yes[yes["method_label"] == method]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _metric_or_fallback(row: dict[str, Any], key: str, fallback: Any) -> Any:
    return row[key] if row and key in row else fallback


def _write_markdown(path: Path, tables: dict[str, list[dict[str, Any]]]) -> Path:
    lines = ["# Mechanism Mitigation Paper Tables", ""]
    for title, rows in tables.items():
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No rows.", ""])
            continue
        fieldnames = _fieldnames(rows)
        lines.append("| " + " | ".join(_title(field) for field in fieldnames) + " |")
        lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_single_table_markdown(path: Path, title: str, rows: list[dict[str, Any]]) -> Path:
    lines = [f"# {title}", ""]
    if rows:
        fieldnames = _fieldnames(rows)
        lines.append("| " + " | ".join(_title(field) for field in fieldnames) + " |")
        lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    else:
        lines.append("No rows.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.3f}"


def _fmt_percentile(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _title(field: str) -> str:
    return field.replace("_", " ").title()


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
