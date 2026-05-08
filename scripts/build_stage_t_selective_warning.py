#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


DEFAULT_SCORES = [
    "pls32_probe",
    "tail_257_1024_probe",
    "full_probe",
    "top_4_probe",
    "random64_probe",
    "margin_probe",
    "low_margin_probe",
    "margin_plus_pls32_probe",
    "margin_plus_tail_257_1024_probe",
    "margin_plus_full_probe",
    "low_margin_plus_pls32_probe",
    "low_margin_plus_tail_257_1024_probe",
    "low_margin_plus_full_probe",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Stage T selective warning/abstention metrics from calibrated gates."
    )
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument("--scores", nargs="*", default=DEFAULT_SCORES)
    parser.add_argument("--random-repeats-note", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.stage_t_dir)
    result = build_selective_warning_metrics(
        stage_t_dir=args.stage_t_dir,
        target_rates=args.target_rates,
        selected_scores=args.scores,
        output_dir=output_dir,
        random_repeats_note=args.random_repeats_note,
    )
    summary_path = write_json(output_dir / "build_stage_t_selective_warning_summary.json", result)
    append_experiment_log(args.log_path, "build_stage_t_selective_warning", summary_path, "ok")
    print(summary_path)


def build_selective_warning_metrics(
    stage_t_dir: str | Path,
    target_rates: list[float],
    selected_scores: list[str],
    output_dir: str | Path,
    random_repeats_note: int = 200,
) -> dict[str, Any]:
    root = Path(stage_t_dir)
    output_root = ensure_dir(output_dir)
    gate = pd.read_csv(root / "stage_t_gate_metrics.csv")
    random_gate = pd.read_csv(root / "stage_t_random_gate_metrics.csv")
    target_rates_set = {_rate_key(rate) for rate in target_rates}
    gate = gate[
        gate["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_rates_set)
    ].copy()
    random_gate = random_gate[
        random_gate["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_rates_set)
    ].copy()

    available_scores = set(gate["score"].dropna().astype(str))
    wanted_scores = [score for score in selected_scores if score in available_scores]
    missing_scores = [score for score in selected_scores if score not in available_scores]

    rows: list[dict[str, Any]] = []
    rows.extend(_original_and_always_rows(gate))
    for _, row in gate[gate["score"].isin(wanted_scores)].iterrows():
        rows.append(_gate_warning_row(row))
    rows.extend(_random_warning_rows(random_gate, gate, wanted_scores, random_repeats_note))

    rows = sorted(
        rows,
        key=lambda item: (
            int(item.get("layer") or 0),
            float(item.get("target_trigger_rate_predicted_yes") or 0.0),
            str(item.get("method", "")),
            str(item.get("score", "")),
            str(item.get("matched_score", "")),
        ),
    )
    metrics_path = write_csv(
        output_root / "stage_t_selective_warning_metrics.csv",
        rows,
        _fieldnames(rows),
    )
    md_path = _write_markdown_summary(
        output_root / "stage_t_selective_warning_metrics.md",
        rows,
        missing_scores,
    )
    return {
        "stage_t_dir": str(root),
        "target_rates": target_rates,
        "selected_scores": selected_scores,
        "available_selected_scores": wanted_scores,
        "missing_selected_scores": missing_scores,
        "num_rows": len(rows),
        "metrics_path": str(metrics_path),
        "markdown_path": str(md_path),
    }


def _original_and_always_rows(gate: pd.DataFrame) -> list[dict[str, Any]]:
    base_cols = ["layer", "split", "target_trigger_rate_predicted_yes"]
    rows: list[dict[str, Any]] = []
    for key, group in gate.groupby(base_cols, dropna=False):
        layer, split, target_rate = key
        ref = group.iloc[0]
        predicted_yes_n = int(ref["predicted_yes_n"])
        total_fp = _group_total_from_recall(group, "triggered_fp", "fp_recall_among_predicted_yes")
        total_tp = _group_total_from_recall(group, "triggered_tp", "tp_damage")
        always_precision = total_fp / predicted_yes_n if predicted_yes_n else math.nan
        common = {
            "layer": layer,
            "split": split,
            "target_trigger_rate_predicted_yes": target_rate,
            "matched_score": "",
            "random_repeats": "",
            "original_accuracy": ref["original_accuracy"],
            "original_f1": ref["original_f1"],
            "original_fp_rate": ref["original_fp_rate"],
            "oracle_flip_accuracy": "",
            "oracle_flip_f1": "",
            "oracle_fp_reduction": "",
            "oracle_tp_preserved": "",
        }
        rows.append(
            {
                **common,
                "method": "Original",
                "gate_family": "none",
                "score": "",
                "trigger_n": 0,
                "predicted_yes_n": predicted_yes_n,
                "trigger_rate_predicted_yes": 0.0,
                "extra_compute_fraction_vs_always": 0.0,
                "compute_saved_vs_always": 1.0,
                "fp_captured": 0,
                "fp_capture_rate": 0.0,
                "tp_damaged": 0,
                "tp_damage": 0.0,
                "tp_preserved": 1.0,
                "warning_precision": math.nan,
                "fp_captured_per_trigger": math.nan,
                "notes": "No warning is emitted.",
            }
        )
        rows.append(
            {
                **common,
                "method": "Always warning",
                "gate_family": "always",
                "score": "always_predicted_yes",
                "trigger_n": predicted_yes_n,
                "predicted_yes_n": predicted_yes_n,
                "trigger_rate_predicted_yes": 1.0,
                "extra_compute_fraction_vs_always": 1.0,
                "compute_saved_vs_always": 0.0,
                "fp_captured": total_fp,
                "fp_capture_rate": 1.0,
                "tp_damaged": total_tp,
                "tp_damage": 1.0,
                "tp_preserved": 0.0,
                "warning_precision": always_precision,
                "fp_captured_per_trigger": always_precision,
                "notes": "Upper warning coverage; also warns on every correct predicted-Yes.",
            }
        )
    return rows


def _gate_warning_row(row: pd.Series) -> dict[str, Any]:
    trigger_n = int(row["trigger_n"])
    triggered_fp = int(row["triggered_fp"])
    triggered_tp = int(row["triggered_tp"])
    trigger_rate = float(row["trigger_rate_predicted_yes"])
    return {
        "layer": row["layer"],
        "split": row["split"],
        "target_trigger_rate_predicted_yes": row["target_trigger_rate_predicted_yes"],
        "method": _method_name(str(row["score"])),
        "gate_family": _gate_family(str(row["score"])),
        "score": row["score"],
        "matched_score": "",
        "random_repeats": "",
        "trigger_n": trigger_n,
        "predicted_yes_n": int(row["predicted_yes_n"]),
        "trigger_rate_predicted_yes": trigger_rate,
        "extra_compute_fraction_vs_always": trigger_rate,
        "compute_saved_vs_always": 1.0 - trigger_rate,
        "fp_captured": triggered_fp,
        "fp_capture_rate": row["fp_recall_among_predicted_yes"],
        "tp_damaged": triggered_tp,
        "tp_damage": row["tp_damage"],
        "tp_preserved": 1.0 - float(row["tp_damage"]),
        "warning_precision": row["triggered_fp_ratio"],
        "fp_captured_per_trigger": triggered_fp / trigger_n if trigger_n else math.nan,
        "original_accuracy": row["original_accuracy"],
        "original_f1": row["original_f1"],
        "original_fp_rate": row["original_fp_rate"],
        "oracle_flip_accuracy": row["oracle_flip_accuracy"],
        "oracle_flip_f1": row["oracle_flip_f1"],
        "oracle_fp_reduction": row["oracle_fp_reduction"],
        "oracle_tp_preserved": row["oracle_tp_preserved"],
        "notes": "Selective warning/abstention; no answer rewriting required.",
    }


def _random_warning_rows(
    random_gate: pd.DataFrame,
    gate: pd.DataFrame,
    wanted_scores: list[str],
    random_repeats_note: int,
) -> list[dict[str, Any]]:
    gate_lookup = {
        (int(row.layer), str(row.score), _rate_key(row.target_trigger_rate_predicted_yes)): row
        for row in gate.itertuples(index=False)
    }
    total_lookup = {
        (int(layer), str(split), _rate_key(target_rate)): (
            _group_total_from_recall(group, "triggered_fp", "fp_recall_among_predicted_yes"),
            _group_total_from_recall(group, "triggered_tp", "tp_damage"),
        )
        for (layer, split, target_rate), group in gate.groupby(
            ["layer", "split", "target_trigger_rate_predicted_yes"],
            dropna=False,
        )
    }
    rows: list[dict[str, Any]] = []
    if random_gate.empty:
        return rows
    pivot = (
        random_gate.pivot_table(
            index=["layer", "matched_score", "split", "target_trigger_rate_predicted_yes", "n_trigger"],
            columns="metric",
            values=["mean", "std", "p05", "p95"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(part) for part in col if part != "").strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in pivot.columns
    ]
    for _, row in pivot.iterrows():
        matched_score = str(row["matched_score"])
        if matched_score not in wanted_scores:
            continue
        key = (int(row["layer"]), matched_score, _rate_key(row["target_trigger_rate_predicted_yes"]))
        ref = gate_lookup.get(key)
        if ref is None:
            continue
        n_trigger = int(row["n_trigger"])
        triggered_fp_ratio = float(row.get("mean_triggered_fp_ratio", math.nan))
        fp_capture = float(row.get("mean_fp_recall_among_predicted_yes", math.nan))
        tp_damage = float(row.get("mean_tp_damage", math.nan))
        total_fp, total_tp = total_lookup.get(
            (int(row["layer"]), str(row["split"]), _rate_key(row["target_trigger_rate_predicted_yes"])),
            (
                _total_from_recall(ref.triggered_fp, ref.fp_recall_among_predicted_yes),
                _total_from_recall(ref.triggered_tp, ref.tp_damage),
            ),
        )
        rows.append(
            {
                "layer": row["layer"],
                "split": row["split"],
                "target_trigger_rate_predicted_yes": row["target_trigger_rate_predicted_yes"],
                "method": "Random warning",
                "gate_family": "same_trigger_random",
                "score": "same_trigger_random",
                "matched_score": matched_score,
                "random_repeats": random_repeats_note,
                "trigger_n": n_trigger,
                "predicted_yes_n": int(ref.predicted_yes_n),
                "trigger_rate_predicted_yes": float(ref.trigger_rate_predicted_yes),
                "extra_compute_fraction_vs_always": float(ref.trigger_rate_predicted_yes),
                "compute_saved_vs_always": 1.0 - float(ref.trigger_rate_predicted_yes),
                "fp_captured": fp_capture * total_fp if not math.isnan(fp_capture) else math.nan,
                "fp_capture_rate": fp_capture,
                "tp_damaged": tp_damage * total_tp if not math.isnan(tp_damage) else math.nan,
                "tp_damage": tp_damage,
                "tp_preserved": 1.0 - tp_damage if not math.isnan(tp_damage) else math.nan,
                "warning_precision": triggered_fp_ratio,
                "fp_captured_per_trigger": triggered_fp_ratio,
                "original_accuracy": ref.original_accuracy,
                "original_f1": ref.original_f1,
                "original_fp_rate": ref.original_fp_rate,
                "oracle_flip_accuracy": "",
                "oracle_flip_f1": "",
                "oracle_fp_reduction": fp_capture,
                "oracle_tp_preserved": 1.0 - tp_damage if not math.isnan(tp_damage) else math.nan,
                "notes": "Mean same-trigger-count random warning baseline.",
            }
        )
    return rows


def _write_markdown_summary(
    path: Path,
    rows: list[dict[str, Any]],
    missing_scores: list[str],
) -> Path:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage T Selective Warning Metrics",
        "",
        "Selective warning treats gated predicted-Yes samples as visually unsupported or uncertain, without forcing a second answer.",
        "",
    ]
    if missing_scores:
        lines.extend(
            [
                "Unavailable requested scores:",
                "",
                *[f"- `{score}`" for score in missing_scores],
                "",
            ]
        )
    for rate in sorted(df["target_trigger_rate_predicted_yes"].dropna().unique()):
        subset = df[df["target_trigger_rate_predicted_yes"] == rate].copy()
        priority = [
            "PLS warning",
            "Tail warning",
            "FullD warning",
            "Top-4 warning",
            "Random warning",
            "Random-subspace warning",
            "Always warning",
            "Original",
        ]
        subset["priority"] = subset["method"].map({name: idx for idx, name in enumerate(priority)}).fillna(99)
        subset = subset.sort_values(["priority", "score", "matched_score"])
        lines.extend(
            [
                f"## Target Trigger Rate {float(rate):.2f}",
                "",
                "| Method | Score / Match | Trigger Rate | FP Captured | FP Recall | TP Damage | Warning Precision | Compute Saved |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset.itertuples(index=False):
            score = getattr(row, "score")
            matched = getattr(row, "matched_score")
            label = f"{score} -> {matched}" if matched else str(score)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(getattr(row, "method")),
                        label,
                        _fmt(getattr(row, "trigger_rate_predicted_yes")),
                        _fmt(getattr(row, "fp_captured")),
                        _fmt(getattr(row, "fp_capture_rate")),
                        _fmt(getattr(row, "tp_damage")),
                        _fmt(getattr(row, "warning_precision")),
                        _fmt(getattr(row, "compute_saved_vs_always")),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _method_name(score: str) -> str:
    if score == "pls32_probe":
        return "PLS warning"
    if score == "tail_257_1024_probe":
        return "Tail warning"
    if score == "tail_257_1024_energy":
        return "Tail-energy warning"
    if score == "full_probe":
        return "FullD warning"
    if score == "top_4_probe":
        return "Top-4 warning"
    if score == "top_64_probe":
        return "Top-64 warning"
    if score == "random64_probe":
        return "Random-subspace warning"
    if score == "margin_probe":
        return "Margin warning"
    if score.startswith("margin_plus_"):
        return "Margin+Geometry warning"
    if score == "low_margin_probe":
        return "Low-margin warning"
    if score.startswith("low_margin_plus_"):
        return "Low-margin+Geometry warning"
    return f"{score} warning"


def _gate_family(score: str) -> str:
    if score.startswith("low_margin_plus_"):
        return "low_margin_plus_geometry"
    if score.startswith("margin_plus_"):
        return "margin_plus_geometry"
    if score == "low_margin_probe":
        return "low_margin"
    if score == "margin_probe":
        return "margin"
    if score.startswith("top_"):
        return "top_variance"
    if score.startswith("tail_") or score in {"pls32_probe", "full_probe"}:
        return "geometry"
    if score == "random64_probe":
        return "random_subspace"
    return "score_gate"


def _total_from_recall(captured: Any, recall: Any) -> int:
    captured_f = float(captured)
    recall_f = float(recall)
    if recall_f <= 0 or math.isnan(recall_f):
        return 0
    return int(round(captured_f / recall_f))


def _group_total_from_recall(group: pd.DataFrame, captured_col: str, recall_col: str) -> int:
    for row in group.itertuples(index=False):
        total = _total_from_recall(getattr(row, captured_col), getattr(row, recall_col))
        if total > 0:
            return total
    return 0


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    if abs(number - round(number)) < 1e-9 and abs(number) >= 10:
        return str(int(round(number)))
    return f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
