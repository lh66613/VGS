#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw yes/no margin baseline metrics for Stage O.")
    parser.add_argument("--margins", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--model-alias", default="")
    parser.add_argument("--output-dir", default="outputs/margins")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "margins": args.margins,
        "model_alias": args.model_alias,
        "output_dir": args.output_dir,
    }
    if not args.dry_run:
        payload.update(build_margin_baseline(args.margins, args.model_alias, args.output_dir))
    summary_path = write_json(Path(args.output_dir) / "build_stage_o_margin_baseline_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "build_stage_o_margin_baseline",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


def build_margin_baseline(
    margins_path: str | Path,
    model_alias: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    df = pd.read_csv(margins_path)
    base = df[df["outcome"].isin(["FP", "TN"])].copy()
    rows: list[dict[str, Any]] = []
    if not base.empty:
        y = (base["outcome"] == "FP").astype(int).to_numpy()
        for score, direction, note in [
            ("yes_minus_no_logit", "higher_means_fp", "Raw first-token logit(Yes)-logit(No)."),
            ("no_minus_yes_logit", "lower_means_fp", "Sign-flipped margin; AUROC uses FP-risk orientation."),
            ("binary_entropy", "higher_means_uncertain", "Binary yes/no entropy from first-token logits."),
        ]:
            values = pd.to_numeric(base[score], errors="coerce").to_numpy(dtype=np.float64)
            risk_score = -values if direction == "lower_means_fp" else values
            rows.append(_metric_row(model_alias, score, direction, y, values, risk_score, note))
    path = write_csv(Path(output_dir) / "margin_baseline_metrics.csv", rows, _fieldnames(rows))
    return {
        "metrics_path": str(path),
        "num_rows": int(len(df)),
        "num_fp_tn_rows": int(len(base)),
        "num_metric_rows": len(rows),
    }


def _metric_row(
    model_alias: str,
    score: str,
    direction: str,
    y: np.ndarray,
    values: np.ndarray,
    risk_score: np.ndarray,
    note: str,
) -> dict[str, Any]:
    mask = np.isfinite(risk_score)
    y = y[mask]
    values = values[mask]
    risk_score = risk_score[mask]
    if len(np.unique(y)) < 2:
        auroc = math.nan
        auprc = math.nan
    else:
        auroc = float(roc_auc_score(y, risk_score))
        auprc = float(average_precision_score(y, risk_score))
    if score == "yes_minus_no_logit":
        pred = (values >= 0).astype(int)
        threshold = 0.0
    elif score == "no_minus_yes_logit":
        pred = (values <= 0).astype(int)
        threshold = 0.0
    else:
        threshold = float(np.nanmedian(values)) if len(values) else math.nan
        pred = (values >= threshold).astype(int)
    return {
        "model_alias": model_alias,
        "baseline": score,
        "direction": direction,
        "target": "POPE FP-vs-TN",
        "n": int(len(y)),
        "num_positive": int(y.sum()),
        "threshold": threshold,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else math.nan,
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else math.nan,
        "notes": note,
    }


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
