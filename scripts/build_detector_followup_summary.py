#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_json


KEY_METHODS = [
    "yes_no_margin",
    "margin_plus_top16_svd_diff",
    "margin_plus_tail_diff",
    "margin_plus_full_diff",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build follow-up detector summary tables.")
    parser.add_argument("--strict-dir", default="outputs/detector_minimal_package")
    parser.add_argument("--reverse-dir", default="outputs/detector_reverse_popular_random_adv")
    parser.add_argument("--amber-warning", default="outputs/stage_t_external_amber_fixed_ids/stage_t_external_warning_metrics.csv")
    parser.add_argument(
        "--amber-margin-warning",
        default="outputs/stage_t_external_amber_margin_detector/stage_t_external_warning_metrics.csv",
    )
    parser.add_argument("--output-dir", default="outputs/detector_followup")
    parser.add_argument("--notes-path", default="notes/detector_followup_summary.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_followup_summary(
        strict_dir=Path(args.strict_dir),
        reverse_dir=Path(args.reverse_dir),
        amber_warning_path=Path(args.amber_warning),
        amber_margin_warning_path=Path(args.amber_margin_warning),
        output_dir=Path(args.output_dir),
        notes_path=Path(args.notes_path),
    )
    summary_path = write_json(Path(args.output_dir) / "build_detector_followup_summary.json", result)
    append_experiment_log(args.log_path, "build_detector_followup_summary", summary_path, "ok")
    print(summary_path)


def build_followup_summary(
    strict_dir: Path,
    reverse_dir: Path,
    amber_warning_path: Path,
    amber_margin_warning_path: Path,
    output_dir: Path,
    notes_path: Path,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    protocol_df = _protocol_summary(strict_dir, reverse_dir)
    bootstrap_df = _bootstrap_main(strict_dir)
    trigger_df = _read_optional(strict_dir / "trigger_curve_table.csv")
    cost_df = _read_optional(strict_dir / "speed_cost_table.csv")
    amber_df = _amber_summary(amber_warning_path)
    amber_margin_df = _amber_margin_summary(amber_margin_warning_path)

    paths = {
        "protocol_summary": out / "protocol_replication_summary.csv",
        "bootstrap_main_table": out / "bootstrap_main_table.csv",
        "trigger_curve_table": out / "trigger_curve_table.csv",
        "speed_cost_table": out / "speed_cost_table.csv",
        "amber_geometry_external_summary": out / "amber_geometry_external_summary.csv",
        "amber_margin_external_summary": out / "amber_margin_external_summary.csv",
    }
    protocol_df.to_csv(paths["protocol_summary"], index=False)
    bootstrap_df.to_csv(paths["bootstrap_main_table"], index=False)
    trigger_df.to_csv(paths["trigger_curve_table"], index=False)
    cost_df.to_csv(paths["speed_cost_table"], index=False)
    amber_df.to_csv(paths["amber_geometry_external_summary"], index=False)
    amber_margin_df.to_csv(paths["amber_margin_external_summary"], index=False)

    note = _render_note(protocol_df, bootstrap_df, trigger_df, cost_df, amber_df, amber_margin_df, paths)
    ensure_dir(notes_path.parent)
    notes_path.write_text(note, encoding="utf-8")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "notes_path": str(notes_path),
        "num_protocol_rows": len(protocol_df),
        "num_amber_rows": len(amber_df),
        "num_amber_margin_rows": len(amber_margin_df),
    }


def _protocol_summary(strict_dir: Path, reverse_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, root in [
        ("random->popular->adversarial", strict_dir),
        ("popular->random->adversarial", reverse_dir),
    ]:
        baseline = _read_optional(root / "detector_baseline_table.csv")
        warning = _read_optional(root / "deployment_warning.csv")
        if baseline.empty:
            continue
        task_b = baseline[baseline["task"] == "task_b_pred_yes_fp_vs_tp"]
        warning_20 = warning[
            (warning["gate"] == "score_top_rate")
            & (warning["target_trigger_rate"].round(3) == 0.2)
        ] if not warning.empty else pd.DataFrame()
        for method in KEY_METHODS:
            match = task_b[task_b["method"] == method]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            warning_row = warning_20[warning_20["method"] == method]
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "feature_dim": row.get("feature_dim", ""),
                    "test_auroc": row.get("test_auroc", math.nan),
                    "test_auprc": row.get("test_auprc", math.nan),
                    "f1": row.get("f1", math.nan),
                    "mcc": row.get("mcc", math.nan),
                    "warning_precision_20pct": _cell(warning_row, "warning_precision"),
                    "fp_recall_20pct": _cell(warning_row, "fp_recall"),
                    "tp_damage_20pct": _cell(warning_row, "tp_damage"),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_main(strict_dir: Path) -> pd.DataFrame:
    return _read_optional(strict_dir / "bootstrap_main_table.csv")


def _amber_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "notes": f"Missing AMBER warning metrics: {path}",
                }
            ]
        )
    df = pd.read_csv(path)
    if df.empty:
        return df
    score_map = {
        "full_probe": "FullD geometry",
        "tail_257_1024_probe": "Tail geometry",
        "top_4_probe": "Top-4 geometry",
        "pls32_probe": "PLS geometry",
    }
    keep = df[
        (df["selection_policy"] == "external_top_rate")
        & (df["target_trigger_rate_predicted_yes"].isin([0.2, 0.3]))
        & (df["score"].isin(score_map))
    ].copy()
    rows = []
    for row in keep.itertuples(index=False):
        rows.append(
            {
                "dataset": "AMBER",
                "comparison_scope": "geometry-only; margin logits unavailable",
                "method": score_map[str(row.score)],
                "score": str(row.score),
                "target_trigger_rate": float(row.target_trigger_rate_predicted_yes),
                "actual_trigger_rate": float(row.trigger_rate_predicted_yes),
                "warning_precision": float(row.warning_precision),
                "fp_recall": float(row.fp_capture_rate),
                "tp_damage": float(row.tp_damage),
                "notes": "AMBER margin-only and margin+geometry require first-token yes/no logits, which are currently NaN in stage_t_external_scores.csv.",
            }
        )
    return pd.DataFrame(rows)


def _amber_margin_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "notes": f"Missing AMBER margin warning metrics: {path}",
                }
            ]
        )
    df = pd.read_csv(path)
    if df.empty:
        return df
    base = df[
        (df["selection_policy"] == "always")
        & (df["score"] == "always_predicted_yes")
    ]
    base_precision = _cell(base, "warning_precision")
    score_map = {
        "low_margin_probe": "margin-only",
        "low_margin_plus_top_16_probe": "margin+top16",
        "low_margin_plus_tail_257_1024_probe": "margin+tail",
        "low_margin_plus_full_probe": "margin+full",
        "top_16_probe": "top16-only",
        "tail_257_1024_probe": "tail-only",
        "full_probe": "full-only",
    }
    keep = df[
        (df["selection_policy"] == "external_top_rate")
        & (df["aggregation"] == "deterministic")
        & (df["target_trigger_rate_predicted_yes"].isin([0.2, 0.3]))
        & (df["score"].isin(score_map))
    ].copy()
    rows = []
    for row in keep.sort_values(
        ["target_trigger_rate_predicted_yes", "warning_precision"],
        ascending=[True, False],
    ).itertuples(index=False):
        precision = float(row.warning_precision)
        rows.append(
            {
                "dataset": "AMBER",
                "policy": "external_top_rate",
                "method": score_map[str(row.score)],
                "score": str(row.score),
                "target_trigger_rate": float(row.target_trigger_rate_predicted_yes),
                "actual_trigger_rate": float(row.trigger_rate_predicted_yes),
                "warning_precision": precision,
                "relative_precision_gain": precision / float(base_precision) if math.isfinite(float(base_precision)) and float(base_precision) else math.nan,
                "fp_recall": float(row.fp_capture_rate),
                "tp_damage": float(row.tp_damage),
                "base_pred_yes_fp_rate": float(base_precision),
            }
        )
    return pd.DataFrame(rows)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _cell(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df:
        return math.nan
    return df.iloc[0][column]


def _render_note(
    protocol_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    amber_df: pd.DataFrame,
    amber_margin_df: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = ["# Detector Follow-up Summary", "", "## Files", ""]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Protocol Replication", ""])
    lines.append(_markdown_table(protocol_df))
    lines.extend(["", "## Bootstrap Main Table", ""])
    lines.append(_markdown_table(bootstrap_df[["comparison", "metric", "delta", "ci95", "significant"]]) if not bootstrap_df.empty else "_Missing._")
    lines.extend(["", "## Trigger Curve", ""])
    lines.append(_markdown_table(trigger_df) if not trigger_df.empty else "_Missing._")
    lines.extend(["", "## Speed And Cost", ""])
    lines.append(_markdown_table(cost_df) if not cost_df.empty else "_Missing._")
    lines.extend(["", "## AMBER External Transfer", ""])
    lines.append(_markdown_table(amber_df) if not amber_df.empty else "_Missing._")
    lines.extend(["", "## AMBER Margin Deployment", ""])
    lines.append(_markdown_table(amber_margin_df) if not amber_margin_df.empty else "_Missing._")
    lines.extend(
        [
            "",
            "## Remaining AMBER Step",
            "",
            "AMBER margin-based deployment has been run if `amber_margin_external_summary` is populated. Re-run `bash scripts/run_gpu_detector_amber_deployment.sh` only when regenerating AMBER first-token margins or changing score sets.",
        ]
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Strict split supports margin+tail/full over margin-only with positive bootstrap CIs for AUROC, AUPRC, warning precision, and lower TP damage.",
            "- Reverse split reproduces the broad benefit in AUPRC and ranking, but warning precision gains are weaker and not uniformly significant.",
            "- AMBER margin transfer is now available. Low-margin is the strongest external warning signal; adding POPE-trained geometry reduces AMBER warning precision under fixed external trigger budgets, so report geometry external transfer as modest and not robust.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Empty._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    headers = [str(col) for col in formatted.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in formatted.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
