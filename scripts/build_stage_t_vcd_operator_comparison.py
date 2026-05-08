#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage T VCD/ICD operator comparisons.")
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--operators", nargs="*", default=None)
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_operator_comparison(
        stage_t_dir=args.stage_t_dir,
        operators=args.operators,
        target_rates=args.target_rates,
    )
    summary_path = write_json(
        Path(args.stage_t_dir) / "build_stage_t_vcd_operator_comparison_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "build_stage_t_vcd_operator_comparison", summary_path, "ok")
    print(summary_path)


def build_operator_comparison(
    stage_t_dir: str | Path,
    operators: list[str] | None,
    target_rates: list[float],
) -> dict[str, Any]:
    root = Path(stage_t_dir)
    metric_paths = sorted(root.glob("stage_t_vcd_metrics_*.csv"))
    if operators:
        wanted = set(operators)
        metric_paths = [path for path in metric_paths if _operator_from_metrics_path(path) in wanted]
    target_keys = {_rate_key(rate) for rate in target_rates}
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for path in metric_paths:
        operator = _operator_from_metrics_path(path)
        df = pd.read_csv(path)
        original = df[df["method"] == "Original"].iloc[0]
        always = df[df["method"] == "Always VCD/ICD"].iloc[0]
        rows.append(_summary_row(operator, "original", original, original, always))
        rows.append(_summary_row(operator, "always", always, original, always))

        gated = df[
            df["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_keys)
            & ~df["method"].isin(["Original", "Always VCD/ICD"])
            & ~df["method"].str.startswith("Random-gated", na=False)
        ].copy()
        random = df[
            df["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_keys)
            & df["method"].str.startswith("Random-gated", na=False)
        ].copy()
        if not gated.empty:
            rows.append(_summary_row(operator, "best_gated_fp_reduction", _best(gated, "fp_reduction"), original, always))
            rows.append(_summary_row(operator, "best_gated_accuracy", _best(gated, "accuracy_after"), original, always))
            rows.append(
                _summary_row(
                    operator,
                    "best_gated_fp_reduction_per_trigger",
                    _best(gated, "fp_reduction_per_trigger"),
                    original,
                    always,
                )
            )
        if not random.empty:
            rows.append(_summary_row(operator, "best_random_fp_reduction", _best(random, "fp_reduction"), original, always))
            rows.append(
                _summary_row(
                    operator,
                    "best_random_fp_reduction_per_trigger",
                    _best(random, "fp_reduction_per_trigger"),
                    original,
                    always,
                )
            )
        count_rows.extend(_prediction_count_rows(root, operator))

    comparison_path = write_csv(
        root / "stage_t_vcd_operator_comparison.csv",
        rows,
        _fieldnames(rows),
    )
    counts_path = write_csv(
        root / "stage_t_vcd_operator_prediction_counts.csv",
        count_rows,
        _fieldnames(count_rows),
    )
    md_path = _write_markdown(root / "stage_t_vcd_operator_comparison.md", rows, count_rows)
    return {
        "stage_t_dir": str(root),
        "operators": sorted({row["operator"] for row in rows}),
        "target_rates": target_rates,
        "num_rows": len(rows),
        "comparison_path": str(comparison_path),
        "prediction_counts_path": str(counts_path),
        "markdown_path": str(md_path),
    }


def _summary_row(operator: str, summary: str, row: pd.Series, original: pd.Series, always: pd.Series) -> dict[str, Any]:
    fp_reduction = float(row["fp_reduction"])
    fp_per_trigger = float(row["fp_reduction_per_trigger"]) if pd.notna(row["fp_reduction_per_trigger"]) else float("nan")
    always_fp_reduction = float(always["fp_reduction"])
    return {
        "operator": operator,
        "summary": summary,
        "method": row["method"],
        "score": row.get("score", ""),
        "matched_score": row.get("matched_score", ""),
        "target_trigger_rate_predicted_yes": row["target_trigger_rate_predicted_yes"],
        "trigger_rate_predicted_yes": row["trigger_rate_predicted_yes"],
        "fp_reduced_n": row["fp_reduced_n"],
        "fp_reduction": fp_reduction,
        "tp_preserved": row["tp_preserved"],
        "fp_reduction_per_trigger": fp_per_trigger,
        "accuracy_after": row["accuracy_after"],
        "f1_after": row["f1_after"],
        "accuracy_delta_vs_original": float(row["accuracy_after"]) - float(original["accuracy_after"]),
        "f1_delta_vs_original": float(row["f1_after"]) - float(original["f1_after"]),
        "compute_saved_vs_always": row["compute_saved_vs_always"],
        "gap_to_always_vcd_fp_reduction": always_fp_reduction - fp_reduction,
    }


def _prediction_count_rows(root: Path, operator: str) -> list[dict[str, Any]]:
    path = root / f"stage_t_vcd_predictions_{operator}.jsonl"
    if not path.exists():
        return []
    counts: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (str(row.get("original_outcome", "")), str(row.get("vcd_parsed_prediction", "")))
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "operator": operator,
            "original_outcome": outcome,
            "vcd_parsed_prediction": prediction,
            "count": count,
        }
        for (outcome, prediction), count in sorted(counts.items())
    ]


def _best(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.sort_values([metric, "fp_reduction", "tp_preserved"], ascending=[False, False, False]).iloc[0]


def _write_markdown(path: Path, rows: list[dict[str, Any]], count_rows: list[dict[str, Any]]) -> Path:
    df = pd.DataFrame(rows)
    counts = pd.DataFrame(count_rows)
    lines = [
        "# Stage T VCD/ICD Operator Comparison",
        "",
        "This table summarizes always-on and best selective-routing rows across VCD/ICD operators.",
        "",
    ]
    if not counts.empty:
        lines.extend(
            [
                "## Prediction Changes",
                "",
                "| Operator | TP -> Yes | TP -> No | FP -> Yes | FP -> No |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for operator in sorted(counts["operator"].unique()):
            group = counts[counts["operator"] == operator]
            lookup = {
                (row.original_outcome, row.vcd_parsed_prediction): row.count
                for row in group.itertuples(index=False)
            }
            lines.append(
                f"| `{operator}` | {lookup.get(('TP', 'yes'), 0)} | {lookup.get(('TP', 'no'), 0)} | "
                f"{lookup.get(('FP', 'yes'), 0)} | {lookup.get(('FP', 'no'), 0)} |"
            )
        lines.append("")
    for summary in ["always", "best_gated_fp_reduction", "best_gated_accuracy", "best_random_fp_reduction"]:
        subset = df[df["summary"] == summary].copy()
        if subset.empty:
            continue
        lines.extend(
            [
                f"## {summary.replace('_', ' ').title()}",
                "",
                "| Operator | Method | Target | Trigger Rate | FP Reduction | TP Preserved | FP / Trigger | Acc After | F1 After | Acc Delta | Compute Saved |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset.sort_values(["fp_reduction", "accuracy_after"], ascending=[False, False]).itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.operator}`",
                        str(row.method),
                        _fmt(row.target_trigger_rate_predicted_yes),
                        _fmt(row.trigger_rate_predicted_yes),
                        _fmt(row.fp_reduction),
                        _fmt(row.tp_preserved),
                        _fmt(row.fp_reduction_per_trigger),
                        _fmt(row.accuracy_after),
                        _fmt(row.f1_after),
                        _fmt(row.accuracy_delta_vs_original),
                        _fmt(row.compute_saved_vs_always),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _operator_from_metrics_path(path: Path) -> str:
    return path.name.removeprefix("stage_t_vcd_metrics_").removesuffix(".csv")


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    return f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
