#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import load_hidden_layer, load_svd, read_jsonl
from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage P multi-seed probes and bootstrap significance tests.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--output-dir", default="outputs/stage_p_stats")
    parser.add_argument("--protocol-note", default="notes/statistical_testing_protocol.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--layers", nargs="*", default=["16", "20", "24", "32"])
    parser.add_argument("--seeds", nargs="*", default=["13", "17", "23", "29", "31"])
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layers = [int(item) for value in args.layers for item in value.split(",") if item]
    seeds = [int(item) for value in args.seeds for item in value.split(",") if item]
    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    payload: dict[str, Any] = {
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
        "layers": layers,
        "seeds": seeds,
        "test_frac": args.test_frac,
        "tail_band": args.tail_band,
        "bootstrap_samples": args.bootstrap_samples,
        "max_iter": args.max_iter,
    }
    if not args.dry_run:
        payload.update(
            analyze_stage_p_stats(
                predictions_path=args.predictions,
                hidden_states_dir=args.hidden_states_dir,
                svd_dir=args.svd_dir,
                output_dir=args.output_dir,
                protocol_note=args.protocol_note,
                layers=layers,
                seeds=seeds,
                test_frac=args.test_frac,
                tail_band=(tail_start, tail_end),
                bootstrap_samples=args.bootstrap_samples,
                max_iter=args.max_iter,
            )
        )
    summary_path = write_json(Path(args.output_dir) / "analyze_stage_p_stats_summary.json", payload)
    append_experiment_log(args.log_path, "analyze_stage_p_stats", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


def analyze_stage_p_stats(
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    output_dir: str | Path,
    protocol_note: str | Path,
    layers: list[int],
    seeds: list[int],
    test_frac: float,
    tail_band: tuple[int, int],
    bootstrap_samples: int,
    max_iter: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    labels_by_id = _fp_tn_labels(read_jsonl(predictions_path))
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for layer in layers:
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(item) for item in hidden["sample_ids"]]
        keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
        y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
        diff = (hidden["z_blind"][keep].float() - hidden["z_img"][keep].float()).numpy()
        basis = load_svd(svd_dir, layer)["Vh"].float().numpy().T
        features = _feature_map(diff, basis, tail_band)
        for seed in seeds:
            train_idx, test_idx = _stratified_split(y, seed, test_frac)
            seed_feature_rows = []
            for feature, x in features.items():
                metrics, scores = _fit_score_probe(
                    x[train_idx],
                    x[test_idx],
                    y[train_idx],
                    y[test_idx],
                    seed,
                    max_iter,
                )
                row = {
                    "seed": seed,
                    "layer": layer,
                    "feature": feature,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "num_positive_test": int(y[test_idx].sum()),
                    **metrics,
                }
                metric_rows.append(row)
                seed_feature_rows.append(row)
                for local_idx, score in zip(test_idx, scores, strict=True):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "sample_id": sample_ids[keep[local_idx]],
                            "feature": feature,
                            "label": int(y[local_idx]),
                            "score": float(score),
                        }
                    )
            ranked = sorted(seed_feature_rows, key=lambda item: item["auroc"], reverse=True)
            for rank, row in enumerate(ranked, start=1):
                rank_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "feature": row["feature"],
                        "rank_within_layer": rank,
                        "auroc": row["auroc"],
                    }
                )
    summary_rows = _summary_rows(metric_rows, bootstrap_samples)
    significance_rows = _significance_rows(prediction_rows, bootstrap_samples)
    metric_path = write_csv(Path(output_dir) / "multiseed_probe_rows.csv", metric_rows, _fieldnames(metric_rows))
    summary_path = write_csv(
        Path(output_dir) / "multiseed_probe_summary.csv",
        summary_rows,
        _fieldnames(summary_rows),
    )
    rank_path = write_csv(Path(output_dir) / "multiseed_layer_rank.csv", rank_rows, _fieldnames(rank_rows))
    pred_path = write_csv(
        Path(output_dir) / "multiseed_probe_predictions.csv",
        prediction_rows,
        _fieldnames(prediction_rows),
    )
    significance_path = write_csv(
        Path(output_dir) / "significance_tests.csv",
        significance_rows,
        _fieldnames(significance_rows),
    )
    _write_protocol_note(protocol_note, layers, seeds, test_frac, tail_band, bootstrap_samples)
    return {
        "metric_rows_path": str(metric_path),
        "summary_path": str(summary_path),
        "rank_path": str(rank_path),
        "prediction_rows_path": str(pred_path),
        "significance_path": str(significance_path),
        "protocol_note": str(protocol_note),
        "num_metric_rows": len(metric_rows),
        "num_summary_rows": len(summary_rows),
        "num_significance_rows": len(significance_rows),
    }


def _fp_tn_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        if row.get("outcome") == "FP":
            labels[str(row["sample_id"])] = 1
        elif row.get("outcome") == "TN":
            labels[str(row["sample_id"])] = 0
    return labels


def _feature_map(diff: np.ndarray, basis: np.ndarray, tail_band: tuple[int, int]) -> dict[str, np.ndarray]:
    tail_start, tail_end = tail_band
    tail_end = min(tail_end, basis.shape[1])
    return {
        "full_diff": diff,
        "top_4": diff @ basis[:, :4],
        "top_64": diff @ basis[:, :64],
        "top_256": diff @ basis[:, :256],
        f"tail_{tail_start}_{tail_end}": diff @ basis[:, tail_start - 1 : tail_end],
    }


def _stratified_split(y: np.ndarray, seed: int, test_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        test_n = max(1, int(round(len(idx) * test_frac)))
        test_parts.append(idx[:test_n])
        train_parts.append(idx[test_n:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _fit_score_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_iter: int,
) -> tuple[dict[str, float], np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed, solver="lbfgs")
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(np.int64)
    metrics = {
        "auroc": float(roc_auc_score(y_test, scores)),
        "auprc": float(average_precision_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "n_iter": int(np.max(clf.n_iter_)),
    }
    return metrics, scores


def _summary_rows(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    result = []
    for (layer, feature), group in df.groupby(["layer", "feature"], sort=True):
        values = group["auroc"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(values, bootstrap_samples, seed=int(layer) + len(feature))
        result.append(
            {
                "layer": int(layer),
                "feature": feature,
                "num_seeds": int(len(values)),
                "auroc_mean": float(np.mean(values)),
                "auroc_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "auroc_min": float(np.min(values)),
                "auroc_max": float(np.max(values)),
                "auroc_ci95_low": ci_low,
                "auroc_ci95_high": ci_high,
                "auprc_mean": float(group["auprc"].mean()),
                "accuracy_mean": float(group["accuracy"].mean()),
                "f1_mean": float(group["f1"].mean()),
            }
        )
    return sorted(result, key=lambda row: row["auroc_mean"], reverse=True)


def _significance_rows(prediction_rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    df = pd.DataFrame(prediction_rows)
    comparisons = [
        ("top_256", "top_4"),
        ("tail_257_1024", "top_4"),
        ("full_diff", "top_256"),
        ("full_diff", "tail_257_1024"),
    ]
    rows = []
    for layer in sorted(df["layer"].unique()):
        for left, right in comparisons:
            layer_df = df[df["layer"] == layer]
            pivot = layer_df.pivot_table(
                index=["seed", "sample_id", "label"],
                columns="feature",
                values="score",
                aggfunc="first",
            ).reset_index()
            if left not in pivot or right not in pivot:
                continue
            y = pivot["label"].to_numpy(dtype=int)
            left_scores = pivot[left].to_numpy(dtype=float)
            right_scores = pivot[right].to_numpy(dtype=float)
            observed = float(roc_auc_score(y, left_scores) - roc_auc_score(y, right_scores))
            ci_low, ci_high, p_value = _bootstrap_auc_delta(
                y,
                left_scores,
                right_scores,
                bootstrap_samples,
                seed=int(layer) * 100 + len(left) + len(right),
            )
            rows.append(
                {
                    "layer": int(layer),
                    "metric": "auroc_delta",
                    "left": left,
                    "right": right,
                    "observed_delta": observed,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_p_two_sided": p_value,
                    "n_paired_predictions": int(len(y)),
                }
            )
    return rows


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(values) <= 1:
        value = float(values[0]) if len(values) else math.nan
        return value, value
    rng = np.random.default_rng(seed)
    draws = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _bootstrap_auc_delta(
    y: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = []
    for _ in range(n_boot):
        sample = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        deltas.append(float(roc_auc_score(y[sample], left_scores[sample]) - roc_auc_score(y[sample], right_scores[sample])))
    deltas_np = np.array(deltas)
    observed = float(roc_auc_score(y, left_scores) - roc_auc_score(y, right_scores))
    p_low = float(np.mean(deltas_np <= 0))
    p_high = float(np.mean(deltas_np >= 0))
    p_value = min(1.0, 2.0 * min(p_low, p_high))
    return float(np.percentile(deltas_np, 2.5)), float(np.percentile(deltas_np, 97.5)), p_value


def _write_protocol_note(
    path: str | Path,
    layers: list[int],
    seeds: list[int],
    test_frac: float,
    tail_band: tuple[int, int],
    bootstrap_samples: int,
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        "# Statistical Testing Protocol\n\n"
        "## Stage P Multi-Seed Probe Protocol\n\n"
        f"- Layers: `{layers}`\n"
        f"- Seeds: `{seeds}`\n"
        f"- Split: stratified FP/TN split with test fraction `{test_frac}`\n"
        "- Features: full blind-image difference, top-4/top-64/top-256 SVD coordinates, and tail SVD coordinates\n"
        f"- Tail band: `{tail_band[0]}-{tail_band[1]}`\n"
        "- Model: class-balanced logistic regression with per-feature standardization\n"
        "- Reported metrics: AUROC, AUPRC, accuracy, F1, seed mean/std/min/max, and bootstrap CI over seed means\n\n"
        "## Pairwise Significance Tests\n\n"
        f"- Bootstrap samples: `{bootstrap_samples}`\n"
        "- AUROC deltas use stratified paired bootstrap over FP/TN test predictions.\n"
        "- The p-value is a two-sided bootstrap sign probability around zero.\n"
        "- Because the same POPE samples can appear across different random splits, these tests should be treated as robustness diagnostics rather than final inferential statistics.\n",
        encoding="utf-8",
    )


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


if __name__ == "__main__":
    main()
