#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


DEFAULT_GEOMETRY_SCORES = [
    "pls32_probe",
    "full_probe",
    "tail_257_1024_probe",
    "tail_257_1024_energy",
    "top_4_probe",
    "random64_probe",
]
DEFAULT_MARGIN_BIN_EDGES = [0.5, 1.5, 3.0]
DEFAULT_BIN_LABELS = ["very_low", "low", "medium", "high"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether geometry scores add local information beyond yes/no margin."
    )
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--scores-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--scores", nargs="*", default=DEFAULT_GEOMETRY_SCORES)
    parser.add_argument("--primary-score", default="pls32_probe")
    parser.add_argument("--margin-score", default="low_margin_probe")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument(
        "--bin-policy",
        choices=["fixed", "quantile"],
        default="fixed",
        help="Fixed uses --margin-bin-edges; quantile makes four equal-count bins in the analysis split.",
    )
    parser.add_argument("--margin-bin-edges", nargs="*", type=float, default=DEFAULT_MARGIN_BIN_EDGES)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--pair-max-margin-delta", type=float, default=0.0625)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.stage_t_dir)
    result = build_geometry_complementarity(
        stage_t_dir=args.stage_t_dir,
        scores_path=args.scores_path,
        output_dir=output_dir,
        layer=args.layer,
        split=args.split,
        calibration_split=args.calibration_split,
        scores=args.scores,
        primary_score=args.primary_score,
        margin_score=args.margin_score,
        target_rates=args.target_rates,
        bin_policy=args.bin_policy,
        margin_bin_edges=args.margin_bin_edges,
        max_pairs=args.max_pairs,
        pair_max_margin_delta=args.pair_max_margin_delta,
    )
    summary_path = write_json(
        output_dir / "analyze_stage_t_geometry_complementarity_summary.json",
        result,
    )
    append_experiment_log(
        args.log_path,
        "analyze_stage_t_geometry_complementarity",
        summary_path,
        "ok",
    )
    print(summary_path)


def build_geometry_complementarity(
    stage_t_dir: str | Path,
    scores_path: str | Path | None,
    output_dir: str | Path,
    layer: int,
    split: str,
    calibration_split: str,
    scores: list[str],
    primary_score: str,
    margin_score: str,
    target_rates: list[float],
    bin_policy: str = "fixed",
    margin_bin_edges: list[float] | None = None,
    max_pairs: int = 12,
    pair_max_margin_delta: float = 0.0625,
) -> dict[str, Any]:
    root = Path(stage_t_dir)
    output_root = ensure_dir(output_dir)
    scores_csv = Path(scores_path) if scores_path else root / "stage_t_scores.csv"
    df = pd.read_csv(scores_csv)

    required = {"layer", "subset", "outcome", "parsed_prediction", "yes_minus_no_logit"}
    missing_required = sorted(required.difference(df.columns))
    if missing_required:
        raise ValueError(f"{scores_csv} is missing required columns: {missing_required}")

    available_scores = [score for score in scores if score in df.columns]
    missing_scores = [score for score in scores if score not in df.columns]
    if primary_score not in available_scores:
        if primary_score in df.columns:
            available_scores.insert(0, primary_score)
        elif available_scores:
            primary_score = available_scores[0]
        else:
            raise ValueError("No requested geometry scores are available in stage_t_scores.csv")
    if margin_score not in df.columns:
        raise ValueError(f"{scores_csv} is missing margin score column: {margin_score}")

    pool = _predicted_yes_pool(df, layer, split)
    calibration_pool = _predicted_yes_pool(df, layer, calibration_split)
    if pool.empty:
        raise ValueError(f"No predicted-Yes FP/TP rows for layer={layer}, split={split}")
    if calibration_pool.empty:
        raise ValueError(
            f"No predicted-Yes FP/TP rows for layer={layer}, calibration_split={calibration_split}"
        )

    target_rates = sorted({float(rate) for rate in target_rates})
    margin_bin_edges = margin_bin_edges or DEFAULT_MARGIN_BIN_EDGES
    binned_pool, bin_info = _with_margin_bins(pool, bin_policy, margin_bin_edges)

    margin_bin_rows = _margin_bin_rows(
        binned_pool,
        layer=layer,
        split=split,
        scores=available_scores,
        target_rates=target_rates,
        bin_policy=bin_policy,
    )
    residual_rows = _residual_prediction_rows(
        pool=pool,
        calibration_pool=calibration_pool,
        layer=layer,
        split=split,
        calibration_split=calibration_split,
        scores=available_scores,
        margin_score=margin_score,
        target_rates=target_rates,
    )
    correlation_rows = _correlation_rows(
        pool=pool,
        layer=layer,
        split=split,
        scores=available_scores,
    )
    pair_rows = _same_margin_pair_rows(
        pool=pool,
        layer=layer,
        split=split,
        scores=available_scores,
        max_pairs=max_pairs,
        pair_max_margin_delta=pair_max_margin_delta,
    )

    margin_bin_path = write_csv(
        output_root / "stage_t_geometry_margin_bin_analysis.csv",
        margin_bin_rows,
        _fieldnames(margin_bin_rows),
    )
    residual_path = write_csv(
        output_root / "stage_t_geometry_residual_prediction.csv",
        residual_rows,
        _fieldnames(residual_rows),
    )
    correlation_path = write_csv(
        output_root / "stage_t_geometry_margin_correlations.csv",
        correlation_rows,
        _fieldnames(correlation_rows),
    )
    pair_path = write_csv(
        output_root / "stage_t_geometry_same_margin_pairs.csv",
        pair_rows,
        _fieldnames(pair_rows),
    )
    note_path = _write_markdown_summary(
        output_root / "stage_t_geometry_complementarity_summary.md",
        layer=layer,
        split=split,
        calibration_split=calibration_split,
        primary_score=primary_score,
        margin_score=margin_score,
        target_rates=target_rates,
        bin_info=bin_info,
        margin_bin_rows=margin_bin_rows,
        residual_rows=residual_rows,
        correlation_rows=correlation_rows,
        pair_rows=pair_rows,
    )

    return {
        "stage_t_dir": str(root),
        "scores_path": str(scores_csv),
        "output_dir": str(output_root),
        "layer": layer,
        "split": split,
        "calibration_split": calibration_split,
        "scores": available_scores,
        "missing_scores": missing_scores,
        "primary_score": primary_score,
        "margin_score": margin_score,
        "target_rates": target_rates,
        "bin_policy": bin_policy,
        "margin_bin_edges": margin_bin_edges,
        "bin_info": bin_info,
        "predicted_yes_n": int(len(pool)),
        "predicted_yes_fp_n": int((pool["outcome"] == "FP").sum()),
        "predicted_yes_tp_n": int((pool["outcome"] == "TP").sum()),
        "margin_bin_path": str(margin_bin_path),
        "residual_prediction_path": str(residual_path),
        "correlation_path": str(correlation_path),
        "same_margin_pair_path": str(pair_path),
        "summary_note_path": str(note_path),
    }


def _predicted_yes_pool(df: pd.DataFrame, layer: int, split: str) -> pd.DataFrame:
    keep = (
        (df["layer"] == layer)
        & (df["subset"].astype(str) == split)
        & (df["parsed_prediction"].astype(str) == "yes")
        & (df["outcome"].astype(str).isin(["FP", "TP"]))
        & (pd.to_numeric(df["yes_minus_no_logit"], errors="coerce").notna())
    )
    out = df[keep].copy()
    out["yes_minus_no_logit"] = pd.to_numeric(out["yes_minus_no_logit"], errors="coerce")
    if "binary_entropy" in out.columns:
        out["binary_entropy"] = pd.to_numeric(out["binary_entropy"], errors="coerce")
    return out


def _with_margin_bins(
    pool: pd.DataFrame,
    bin_policy: str,
    margin_bin_edges: list[float],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = pool.copy()
    if bin_policy == "quantile":
        labels = DEFAULT_BIN_LABELS
        ranked = out["yes_minus_no_logit"].rank(method="first")
        out["margin_bin"] = pd.qcut(ranked, q=4, labels=labels)
    else:
        edges = [-math.inf, *sorted(float(edge) for edge in margin_bin_edges), math.inf]
        labels = _bin_labels(len(edges) - 1)
        out["margin_bin"] = pd.cut(
            out["yes_minus_no_logit"],
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=True,
        )

    info: list[dict[str, Any]] = []
    for bin_name, group in out.groupby("margin_bin", observed=True):
        info.append(
            {
                "margin_bin": str(bin_name),
                "n": int(len(group)),
                "fp_n": int((group["outcome"] == "FP").sum()),
                "tp_n": int((group["outcome"] == "TP").sum()),
                "margin_min": _finite_min(group["yes_minus_no_logit"]),
                "margin_max": _finite_max(group["yes_minus_no_logit"]),
            }
        )
    return out, info


def _margin_bin_rows(
    pool: pd.DataFrame,
    layer: int,
    split: str,
    scores: list[str],
    target_rates: list[float],
    bin_policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_name, group in pool.groupby("margin_bin", observed=True):
        y = (group["outcome"] == "FP").astype(int).to_numpy()
        fp_n = int(np.sum(y == 1))
        tp_n = int(np.sum(y == 0))
        for score in scores:
            values = pd.to_numeric(group[score], errors="coerce").to_numpy(dtype=float)
            auroc = _safe_auroc(y, values)
            for rate in target_rates:
                capture = _capture_at_rate(group["outcome"].to_numpy(), values, rate)
                rows.append(
                    {
                        "layer": layer,
                        "split": split,
                        "bin_policy": bin_policy,
                        "margin_bin": str(bin_name),
                        "score": score,
                        "target_trigger_rate_within_bin": rate,
                        "n": int(len(group)),
                        "fp_n": fp_n,
                        "tp_n": tp_n,
                        "fp_rate": fp_n / len(group) if len(group) else math.nan,
                        "margin_min": _finite_min(group["yes_minus_no_logit"]),
                        "margin_max": _finite_max(group["yes_minus_no_logit"]),
                        "auroc_fp_vs_tp": auroc,
                        **capture,
                    }
                )
    return rows


def _residual_prediction_rows(
    pool: pd.DataFrame,
    calibration_pool: pd.DataFrame,
    layer: int,
    split: str,
    calibration_split: str,
    scores: list[str],
    margin_score: str,
    target_rates: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = pool["outcome"].astype(str).to_numpy()
    fp_total = int(np.sum(outcomes == "FP"))
    tp_total = int(np.sum(outcomes == "TP"))

    margin_values = pd.to_numeric(pool[margin_score], errors="coerce").to_numpy(dtype=float)
    calibration_margin = pd.to_numeric(
        calibration_pool[margin_score],
        errors="coerce",
    ).to_numpy(dtype=float)

    for rate in target_rates:
        margin_threshold = _threshold_at_rate(calibration_margin, rate)
        margin_trigger = np.isfinite(margin_values) & (margin_values >= margin_threshold)
        margin_capture = _capture_from_mask(outcomes, margin_trigger, fp_total, tp_total)
        residual_mask = ~margin_trigger
        residual_outcomes = outcomes[residual_mask]
        residual_fp_n = int(np.sum(residual_outcomes == "FP"))
        residual_tp_n = int(np.sum(residual_outcomes == "TP"))
        residual_base_fp_rate = (
            residual_fp_n / len(residual_outcomes) if len(residual_outcomes) else math.nan
        )

        for score in scores:
            values = pd.to_numeric(pool[score], errors="coerce").to_numpy(dtype=float)
            calibration_values = pd.to_numeric(
                calibration_pool[score],
                errors="coerce",
            ).to_numpy(dtype=float)
            geometry_threshold = _threshold_at_rate(calibration_values, rate)
            geometry_trigger = np.isfinite(values) & (values >= geometry_threshold)
            additional_trigger = residual_mask & geometry_trigger
            union_trigger = margin_trigger | geometry_trigger
            residual_values = values[residual_mask]
            residual_y = (residual_outcomes == "FP").astype(int)
            residual_top = _capture_at_rate(residual_outcomes, residual_values, rate)
            additional = _capture_from_mask(outcomes, additional_trigger, fp_total, tp_total)
            union = _capture_from_mask(outcomes, union_trigger, fp_total, tp_total)
            rows.append(
                {
                    "layer": layer,
                    "split": split,
                    "calibration_split": calibration_split,
                    "margin_score": margin_score,
                    "geometry_score": score,
                    "target_trigger_rate_predicted_yes": rate,
                    "predicted_yes_n": int(len(pool)),
                    "fp_total": fp_total,
                    "tp_total": tp_total,
                    "margin_threshold": margin_threshold,
                    "margin_trigger_n": margin_capture["trigger_n"],
                    "margin_warning_precision": margin_capture["warning_precision"],
                    "margin_fp_recall": margin_capture["fp_recall"],
                    "margin_tp_damage": margin_capture["tp_damage"],
                    "margin_missed_n": int(np.sum(residual_mask)),
                    "margin_missed_fp_n": residual_fp_n,
                    "margin_missed_tp_n": residual_tp_n,
                    "margin_missed_fp_rate": residual_base_fp_rate,
                    "geometry_threshold": geometry_threshold,
                    "residual_auroc_fp_vs_tp": _safe_auroc(residual_y, residual_values),
                    "residual_top_trigger_n": residual_top["trigger_n"],
                    "residual_top_warning_precision": residual_top["warning_precision"],
                    "residual_top_fp_recall": residual_top["fp_recall"],
                    "additional_trigger_n": additional["trigger_n"],
                    "additional_fp_caught": additional["triggered_fp"],
                    "additional_tp_triggered": additional["triggered_tp"],
                    "additional_warning_precision": additional["warning_precision"],
                    "margin_missed_fp_captured_rate": (
                        additional["triggered_fp"] / residual_fp_n
                        if residual_fp_n
                        else math.nan
                    ),
                    "union_trigger_n": union["trigger_n"],
                    "union_trigger_rate_predicted_yes": union["trigger_n"] / len(pool),
                    "union_warning_precision": union["warning_precision"],
                    "union_fp_recall": union["fp_recall"],
                    "union_tp_damage": union["tp_damage"],
                    "delta_fp_recall_vs_margin": union["fp_recall"] - margin_capture["fp_recall"],
                    "delta_tp_damage_vs_margin": union["tp_damage"] - margin_capture["tp_damage"],
                    "precision_lift_vs_margin_missed_base": (
                        additional["warning_precision"] / residual_base_fp_rate
                        if residual_base_fp_rate and not math.isnan(additional["warning_precision"])
                        else math.nan
                    ),
                }
            )
    return rows


def _correlation_rows(
    pool: pd.DataFrame,
    layer: int,
    split: str,
    scores: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references = ["yes_minus_no_logit"]
    if "binary_entropy" in pool.columns:
        references.append("binary_entropy")
    for score in scores:
        for reference in references:
            pair = pool[[score, reference]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append(
                {
                    "layer": layer,
                    "split": split,
                    "population": "predicted_yes_fp_vs_tp",
                    "score": score,
                    "reference": reference,
                    "n": int(len(pair)),
                    "pearson": _safe_corr(pair[score], pair[reference], method="pearson"),
                    "spearman": _safe_corr(pair[score], pair[reference], method="spearman"),
                }
            )
    return rows


def _same_margin_pair_rows(
    pool: pd.DataFrame,
    layer: int,
    split: str,
    scores: list[str],
    max_pairs: int,
    pair_max_margin_delta: float,
) -> list[dict[str, Any]]:
    fps = pool[pool["outcome"].astype(str) == "FP"].copy()
    tps = pool[pool["outcome"].astype(str) == "TP"].copy()
    rows: list[dict[str, Any]] = []
    if fps.empty or tps.empty:
        return rows

    for score in scores:
        candidates: list[dict[str, Any]] = []
        for _, fp in fps.iterrows():
            fp_margin = float(fp["yes_minus_no_logit"])
            fp_score = _to_float(fp[score])
            if math.isnan(fp_score):
                continue
            for _, tp in tps.iterrows():
                tp_score = _to_float(tp[score])
                if math.isnan(tp_score):
                    continue
                margin_delta = abs(fp_margin - float(tp["yes_minus_no_logit"]))
                candidates.append(
                    {
                        "layer": layer,
                        "split": split,
                        "score": score,
                        "fp_sample_id": str(fp.get("sample_id", "")),
                        "tp_sample_id": str(tp.get("sample_id", "")),
                        "fp_margin": fp_margin,
                        "tp_margin": float(tp["yes_minus_no_logit"]),
                        "margin_delta": margin_delta,
                        "fp_score": fp_score,
                        "tp_score": tp_score,
                        "score_delta_fp_minus_tp": fp_score - tp_score,
                        "fp_entropy": _to_float(fp.get("binary_entropy", math.nan)),
                        "tp_entropy": _to_float(tp.get("binary_entropy", math.nan)),
                        "fp_question": str(fp.get("question", "")),
                        "tp_question": str(tp.get("question", "")),
                        "fp_image": str(fp.get("image", "")),
                        "tp_image": str(tp.get("image", "")),
                        "fp_source_subset": str(fp.get("source_subset", "")),
                        "tp_source_subset": str(tp.get("source_subset", "")),
                    }
                )
        close = [row for row in candidates if row["margin_delta"] <= pair_max_margin_delta]
        ranked = close if close else candidates
        ranked = sorted(
            ranked,
            key=lambda row: (
                row["score_delta_fp_minus_tp"] <= 0,
                -row["score_delta_fp_minus_tp"],
                row["margin_delta"],
            ),
        )
        used_fp: set[str] = set()
        used_tp: set[str] = set()
        pair_rank = 1
        for row in ranked:
            if pair_rank > max_pairs:
                break
            if row["fp_sample_id"] in used_fp or row["tp_sample_id"] in used_tp:
                continue
            row = dict(row)
            row["pair_rank"] = pair_rank
            rows.append(row)
            used_fp.add(row["fp_sample_id"])
            used_tp.add(row["tp_sample_id"])
            pair_rank += 1
    return rows


def _capture_at_rate(outcomes: np.ndarray, values: np.ndarray, rate: float) -> dict[str, Any]:
    finite = np.isfinite(values)
    outcomes = outcomes[finite]
    values = values[finite]
    if len(outcomes) == 0:
        return _empty_capture()
    n_trigger = max(1, int(math.ceil(rate * len(outcomes))))
    n_trigger = min(n_trigger, len(outcomes))
    order = np.argsort(values)[::-1]
    chosen_mask = np.zeros(len(outcomes), dtype=bool)
    chosen_mask[order[:n_trigger]] = True
    return _capture_from_mask(
        outcomes,
        chosen_mask,
        int(np.sum(outcomes == "FP")),
        int(np.sum(outcomes == "TP")),
    )


def _capture_from_mask(
    outcomes: np.ndarray,
    mask: np.ndarray,
    fp_total: int,
    tp_total: int,
) -> dict[str, Any]:
    triggered = outcomes[mask]
    trigger_n = int(len(triggered))
    triggered_fp = int(np.sum(triggered == "FP"))
    triggered_tp = int(np.sum(triggered == "TP"))
    return {
        "trigger_n": trigger_n,
        "triggered_fp": triggered_fp,
        "triggered_tp": triggered_tp,
        "warning_precision": triggered_fp / trigger_n if trigger_n else math.nan,
        "fp_recall": triggered_fp / fp_total if fp_total else math.nan,
        "tp_damage": triggered_tp / tp_total if tp_total else math.nan,
    }


def _empty_capture() -> dict[str, Any]:
    return {
        "trigger_n": 0,
        "triggered_fp": 0,
        "triggered_tp": 0,
        "warning_precision": math.nan,
        "fp_recall": math.nan,
        "tp_damage": math.nan,
    }


def _threshold_at_rate(values: np.ndarray, rate: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan
    n_trigger = max(1, int(math.ceil(rate * len(finite))))
    n_trigger = min(n_trigger, len(finite))
    return float(np.sort(finite)[-n_trigger])


def _safe_auroc(y: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    y = y[finite]
    values = values[finite]
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return math.nan
    return float(roc_auc_score(y, values))


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return math.nan
    return float(left.corr(right, method=method))


def _bin_labels(n_bins: int) -> list[str]:
    if n_bins == 4:
        return DEFAULT_BIN_LABELS
    return [f"bin_{idx + 1}" for idx in range(n_bins)]


def _finite_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.min()) if len(numeric) else math.nan


def _finite_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if len(numeric) else math.nan


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_markdown_summary(
    path: Path,
    layer: int,
    split: str,
    calibration_split: str,
    primary_score: str,
    margin_score: str,
    target_rates: list[float],
    bin_info: list[dict[str, Any]],
    margin_bin_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> Path:
    ensure_dir(path.parent)
    primary_rate = target_rates[0] if target_rates else 0.2
    lines = [
        "# Stage T Geometry Complementarity Analysis",
        "",
        "## Protocol",
        "",
        f"- Layer: `{layer}`.",
        f"- Analysis split: `{split}` predicted-Yes FP/TP samples.",
        f"- Calibration split for residual gates: `{calibration_split}`.",
        f"- Primary geometry score for tables: `{primary_score}`.",
        f"- Margin-only risk score: `{margin_score}`.",
        "",
        "## Margin Bins",
        "",
        "| Bin | N | FP | TP | Margin min | Margin max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bin_info:
        lines.append(
            f"| {row['margin_bin']} | {row['n']} | {row['fp_n']} | {row['tp_n']} | "
            f"{_fmt(row['margin_min'])} | {_fmt(row['margin_max'])} |"
        )

    bin_df = pd.DataFrame(margin_bin_rows)
    selected_bins = bin_df[
        (bin_df["score"] == primary_score)
        & (bin_df["target_trigger_rate_within_bin"].round(6) == round(primary_rate, 6))
    ].copy()
    lines.extend(
        [
            "",
            f"## Margin-Bin Geometry Snapshot ({primary_score}, top {primary_rate:.0%})",
            "",
            "| Bin | N | FP rate | AUROC | FP recall | Warning precision |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_bins.itertuples(index=False):
        lines.append(
            f"| {row.margin_bin} | {row.n} | {_fmt(row.fp_rate)} | "
            f"{_fmt(row.auroc_fp_vs_tp)} | {_fmt(row.fp_recall)} | "
            f"{_fmt(row.warning_precision)} |"
        )

    residual_df = pd.DataFrame(residual_rows)
    selected_residual = residual_df[
        residual_df["target_trigger_rate_predicted_yes"].round(6) == round(primary_rate, 6)
    ].copy()
    selected_residual = selected_residual[
        selected_residual["geometry_score"].isin([primary_score, "full_probe", "tail_257_1024_probe"])
    ].head(8)
    lines.extend(
        [
            "",
            f"## Residual Prediction Snapshot (margin top {primary_rate:.0%})",
            "",
            "| Geometry score | Missed FP | Residual AUROC | Extra FP caught | Extra precision | Union FP recall | Union trigger rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_residual.itertuples(index=False):
        lines.append(
            f"| `{row.geometry_score}` | {row.margin_missed_fp_n} | "
            f"{_fmt(row.residual_auroc_fp_vs_tp)} | {row.additional_fp_caught} | "
            f"{_fmt(row.additional_warning_precision)} | {_fmt(row.union_fp_recall)} | "
            f"{_fmt(row.union_trigger_rate_predicted_yes)} |"
        )

    corr_df = pd.DataFrame(correlation_rows)
    selected_corr = corr_df[
        corr_df["score"].isin([primary_score, "full_probe", "tail_257_1024_probe"])
    ].copy()
    lines.extend(
        [
            "",
            "## Correlation Snapshot",
            "",
            "| Score | Reference | Pearson | Spearman |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in selected_corr.itertuples(index=False):
        lines.append(
            f"| `{row.score}` | `{row.reference}` | {_fmt(row.pearson)} | {_fmt(row.spearman)} |"
        )

    pair_df = pd.DataFrame(pair_rows)
    selected_pairs = pair_df[pair_df["score"] == primary_score].head(5) if not pair_df.empty else pair_df
    lines.extend(
        [
            "",
            f"## Same-Margin Pair Examples ({primary_score})",
            "",
            "| Rank | FP sample | TP sample | Margin delta | FP score | TP score | Score delta |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_pairs.itertuples(index=False):
        lines.append(
            f"| {row.pair_rank} | `{row.fp_sample_id}` | `{row.tp_sample_id}` | "
            f"{_fmt(row.margin_delta)} | {_fmt(row.fp_score)} | {_fmt(row.tp_score)} | "
            f"{_fmt(row.score_delta_fp_minus_tp)} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric):
        return ""
    return f"{numeric:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


if __name__ == "__main__":
    main()
