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
    parser = argparse.ArgumentParser(description="Build a compact mitigation MVP report from Stage 1/2/3 outputs.")
    parser.add_argument("--stage1-dir", default="outputs/mechanism_mitigation/stage1_vcd_decomposition")
    parser.add_argument("--stage2-dir", default="outputs/mechanism_mitigation/stage2_subspace_vcd")
    parser.add_argument("--stage3-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/mvp")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_report(args.stage1_dir, args.stage2_dir, args.stage3_dir, args.output_dir)
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_mitigation_mvp_report_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_mitigation_mvp_report", summary_path, "ok")
    print(summary_path)


def build_report(
    stage1_dir: str | Path,
    stage2_dir: str | Path,
    stage3_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    key_rows: list[dict[str, Any]] = []
    stage1_path = Path(stage1_dir) / "vcd_tp_damage_analysis.csv"
    stage2_path = Path(stage2_dir) / "subspace_vcd_results.csv"
    stage3_path = Path(stage3_dir) / "stage_t_vcd_operator_comparison.csv"

    stage1_rows = _stage1_rows(stage1_path)
    stage2_rows = _stage2_rows(stage2_path)
    stage3_rows = _stage3_rows(stage3_path)
    key_rows.extend(stage1_rows)
    key_rows.extend(stage2_rows)
    key_rows.extend(stage3_rows)

    key_path = write_csv(output_root / "mvp_key_results.csv", key_rows, _fieldnames(key_rows))
    report_path = _write_markdown(output_root / "mvp_summary.md", stage1_rows, stage2_rows, stage3_rows)
    return {
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage3_dir": str(stage3_dir),
        "key_results_path": str(key_path),
        "report_path": str(report_path),
        "num_key_rows": len(key_rows),
    }


def _stage1_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df["score"] = df["fp_minus_tp_positive_rate_gap"].fillna(-999)
    rows = []
    for row in df.sort_values("score", ascending=False).head(10).itertuples(index=False):
        rows.append(
            {
                "stage": "stage1_decomposition",
                "operator": row.operator,
                "method": row.band,
                "metric": "fp_minus_tp_positive_rate_gap",
                "value": row.fp_minus_tp_positive_rate_gap,
                "fp_reduction": "",
                "tp_preserved": "",
                "accuracy_delta": "",
                "note": f"L{row.layer}; FP positive rate {row.fp_positive_contribution_rate:.3f}, TP positive rate {row.tp_positive_contribution_rate:.3f}",
            }
        )
    return rows


def _stage2_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    df["tp_safe"] = df["tp_preserved"] >= 0.95
    df["rank_score"] = df["fp_reduction"].fillna(-1) - (1 - df["tp_preserved"].fillna(0))
    rows = []
    ranked = df.sort_values(
        ["tp_safe", "fp_reduction", "tp_preserved", "accuracy_delta"],
        ascending=[False, False, False, False],
    )
    for row in ranked.head(12).itertuples(index=False):
        caution = "" if bool(row.tp_safe) else "TP-unsafe fallback; do not present as mitigation win. "
        rows.append(
            {
                "stage": "stage2_subspace_logit",
                "operator": row.operator,
                "method": row.method,
                "metric": "calibrated_test_tradeoff_tp_safe" if bool(row.tp_safe) else "calibrated_test_tradeoff_tp_unsafe",
                "value": row.rank_score,
                "fp_reduction": row.fp_reduction,
                "tp_preserved": row.tp_preserved,
                "accuracy_delta": row.accuracy_delta,
                "note": f"{caution}L{row.layer}; alpha {row.alpha}; split {row.split}",
            }
        )
    return rows


def _stage3_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    keep = df[df["summary"].isin(["always", "best_gated_fp_reduction", "best_gated_accuracy"])].copy()
    rows = []
    for row in keep.sort_values(["operator", "summary"]).itertuples(index=False):
        rows.append(
            {
                "stage": "stage3_selective_routing",
                "operator": row.operator,
                "method": row.method,
                "metric": row.summary,
                "value": row.fp_reduction,
                "fp_reduction": row.fp_reduction,
                "tp_preserved": row.tp_preserved,
                "accuracy_delta": row.accuracy_delta_vs_original,
                "note": f"target {row.target_trigger_rate_predicted_yes}; compute saved {row.compute_saved_vs_always}",
            }
        )
    return rows


def _write_markdown(
    path: Path,
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    stage3_rows: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mechanism Mitigation MVP Summary",
        "",
        "This report packages the mitigation-plan MVP: correction decomposition, logit-level subspace filtering, and geometry-gated VCD/ICD routing.",
        "",
    ]
    _append_table(lines, "Stage 1: Decomposition Signals", stage1_rows)
    _append_table(lines, "Stage 2: Calibrated Subspace Correction", stage2_rows)
    _append_table(lines, "Stage 3: Selective Routing", stage3_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _append_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["No rows found. Run the corresponding stage first.", ""])
        return
    lines.extend(
        [
            "| Operator | Method | Metric | Value | FP Reduction | TP Preserved | Acc Delta | Note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['operator']}`",
                    str(row["method"]),
                    str(row["metric"]),
                    _fmt(row["value"]),
                    _fmt(row["fp_reduction"]),
                    _fmt(row["tp_preserved"]),
                    _fmt(row["accuracy_delta"]),
                    str(row["note"]),
                ]
            )
            + " |"
        )
    lines.append("")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
