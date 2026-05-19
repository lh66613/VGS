from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["TP", "TN", "FP", "FN"]


def fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def requested_present_subspaces(df: pd.DataFrame, subspaces: list[str]) -> list[str]:
    return [name for name in subspaces if f"dmargin_no_minus_yes_{name}" in df.columns]


def require_present_subspaces(df: pd.DataFrame, subspaces: list[str], artifact: str | Path) -> list[str]:
    present = requested_present_subspaces(df, subspaces)
    missing = [name for name in subspaces if name not in present]
    if not present:
        raise ValueError(
            "None of the requested subspaces are present in "
            f"{artifact}. Missing examples: {', '.join(missing[:8])}"
        )
    return present


def add_metric_rates(
    selected: pd.DataFrame,
    sample_predictions: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    if selected.empty or sample_predictions.empty:
        return selected.copy()
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        subset = sample_predictions[
            (sample_predictions["operator"].astype(str) == str(row.operator))
            & (sample_predictions["layer"].astype(int) == int(row.layer))
            & (sample_predictions["subspace"].astype(str) == str(row.subspace))
            & np.isclose(sample_predictions["alpha"].astype(float), float(row.alpha))
            & (sample_predictions["split"].astype(str) == split)
        ]
        out = row._asdict()
        out.update(yes_rate_metrics(subset))
        rows.append(out)
    return pd.DataFrame(rows)


def yes_rate_metrics(sample_rows: pd.DataFrame) -> dict[str, float]:
    if sample_rows.empty:
        return {
            "overall_yes_rate": math.nan,
            "tp_yes_rate": math.nan,
            "fp_yes_rate": math.nan,
            "tn_yes_rate": math.nan,
            "fn_yes_rate": math.nan,
        }
    final = sample_rows["final_prediction"].astype(str)
    out: dict[str, float] = {
        "overall_yes_rate": _mean_yes(final),
    }
    for outcome in OUTCOMES:
        group = sample_rows[sample_rows["original_outcome"].astype(str) == outcome]
        out[f"{outcome.lower()}_yes_rate"] = _mean_yes(group["final_prediction"].astype(str))
    return out


def base_metrics_from_predictions(predictions: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    if group_col and group_col in predictions.columns:
        groups = [(str(key), group) for key, group in predictions.groupby(group_col, dropna=False)]
    else:
        groups = [("all", predictions)]
    rows: list[dict[str, Any]] = []
    for group_name, group in groups:
        outcomes = group["outcome"].astype(str)
        parsed = group["parsed_prediction"].astype(str)
        counts = outcome_counts(outcomes)
        rows.append(
            {
                "group": group_name,
                "method": "Base",
                "n": int(len(group)),
                "tp": counts["TP"],
                "tn": counts["TN"],
                "fp": counts["FP"],
                "fn": counts["FN"],
                "fp_reduction": 0.0,
                "tp_preserved": 1.0 if counts["TP"] else math.nan,
                "accuracy_delta": 0.0,
                "overall_yes_rate": _mean_yes(parsed),
                "tp_yes_rate": _mean_yes(parsed[outcomes == "TP"]),
                "fp_yes_rate": _mean_yes(parsed[outcomes == "FP"]),
                "tn_yes_rate": _mean_yes(parsed[outcomes == "TN"]),
                "fn_yes_rate": _mean_yes(parsed[outcomes == "FN"]),
                "accuracy": accuracy(counts),
            }
        )
    return pd.DataFrame(rows)


def outcome_counts(outcomes: pd.Series) -> dict[str, int]:
    return {key: int((outcomes == key).sum()) for key in OUTCOMES}


def accuracy(counts: dict[str, int]) -> float:
    denom = sum(counts.values())
    return (counts["TP"] + counts["TN"]) / denom if denom else math.nan


def safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else math.nan


def safe_median(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else math.nan


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_Missing._"
    view = df.copy()
    if columns:
        view = view[[col for col in columns if col in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(_title(col) for col in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _mean_yes(values: pd.Series) -> float:
    if len(values) == 0:
        return math.nan
    return float((values == "yes").mean())


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    text = str(value)
    return text.replace("|", "\\|")


def _title(name: str) -> str:
    return name.replace("_", " ").title()
