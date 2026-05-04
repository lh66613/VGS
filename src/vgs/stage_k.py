"""Stage K token-position robustness analysis."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from vgs.artifacts import load_hidden_layer, read_jsonl
from vgs.geometry import cumulative_explained_variance, effective_rank
from vgs.io import ensure_dir, write_csv


def analyze_stage_k_positions(
    positions: list[str],
    layers: list[int],
    k_grid: list[int],
    predictions_path: str | Path,
    hidden_root: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    labels_by_id = _fp_target_labels(read_jsonl(predictions_path))
    split_ids = _load_split_ids(split_dir)
    spectrum_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for position in tqdm(positions, desc="Stage K positions", unit="position"):
        hidden_dir = Path(hidden_root) / position
        for layer in tqdm(layers, desc=f"{position} layers", unit="layer", leave=False):
            payload = load_hidden_layer(hidden_dir, layer)
            sample_ids = [str(sample_id) for sample_id in payload["sample_ids"]]
            z_img = payload["z_img"].float().numpy()
            z_blind = payload["z_blind"].float().numpy()
            diff = z_blind - z_img
            train_all_idx = _indices_for_ids(sample_ids, split_ids["train"])
            train_matrix = diff[train_all_idx]
            basis, singular_values = _svd_basis(train_matrix)
            cumulative = cumulative_explained_variance(singular_values)
            spectrum_rows.append(
                {
                    "position": position,
                    "layer": layer,
                    "num_train_samples": int(train_matrix.shape[0]),
                    "hidden_dim": int(train_matrix.shape[1]),
                    "effective_rank": effective_rank(singular_values),
                    "explained_variance_k4": _explained_at(cumulative, 4),
                    "explained_variance_k64": _explained_at(cumulative, 64),
                    "explained_variance_k128": _explained_at(cumulative, 128),
                    "explained_variance_k256": _explained_at(cumulative, 256),
                }
            )

            train_label_idx = [
                idx
                for idx, sample_id in enumerate(sample_ids)
                if sample_id in labels_by_id and sample_id in split_ids["train"]
            ]
            test_label_idx = [
                idx
                for idx, sample_id in enumerate(sample_ids)
                if sample_id in labels_by_id and sample_id in split_ids["test"]
            ]
            y_train = np.array([labels_by_id[sample_ids[idx]] for idx in train_label_idx], dtype=np.int64)
            y_test = np.array([labels_by_id[sample_ids[idx]] for idx in test_label_idx], dtype=np.int64)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            diff_train = diff[train_label_idx]
            diff_test = diff[test_label_idx]
            probe_rows.extend(
                _probe_rows(
                    position,
                    layer,
                    diff_train,
                    diff_test,
                    y_train,
                    y_test,
                    basis,
                    cumulative,
                    k_grid,
                    seed,
                    max_iter,
                )
            )

    spectrum_path = write_csv(
        Path(output_dir) / "position_spectrum_summary.csv",
        spectrum_rows,
        _fieldnames(spectrum_rows),
    )
    probe_path = write_csv(
        Path(output_dir) / "position_probe_summary.csv",
        probe_rows,
        _fieldnames(probe_rows),
    )
    _plot_position_heatmap(probe_rows, Path(plot_dir))
    return {
        "spectrum_path": str(spectrum_path),
        "probe_path": str(probe_path),
        "num_spectrum_rows": len(spectrum_rows),
        "num_probe_rows": len(probe_rows),
    }


def _probe_rows(
    position: str,
    layer: int,
    diff_train: np.ndarray,
    diff_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    basis: np.ndarray,
    cumulative: np.ndarray,
    k_grid: list[int],
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "position": position,
            "layer": layer,
            "feature": "full_difference",
            "k": "full",
            "effective_k": diff_train.shape[1],
            "explained_variance": 1.0,
            "feature_dim": int(diff_train.shape[1]),
            **_fit_probe(diff_train, diff_test, y_train, y_test, seed, max_iter),
        }
    ]
    for k in k_grid:
        k_eff = min(k, basis.shape[1])
        rows.append(
            {
                "position": position,
                "layer": layer,
                "feature": "top_k_svd",
                "k": k,
                "effective_k": k_eff,
                "explained_variance": _explained_at(cumulative, k_eff),
                "feature_dim": k_eff,
                **_fit_probe(
                    diff_train @ basis[:, :k_eff],
                    diff_test @ basis[:, :k_eff],
                    y_train,
                    y_test,
                    seed,
                    max_iter,
                ),
            }
        )
    return rows


def _fit_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_iter: int,
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    probabilities = clf.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else math.nan,
        "auprc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "n_iter": int(np.max(clf.n_iter_)),
    }


def _svd_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt.T.astype(np.float32, copy=False), singular_values.astype(np.float64, copy=False)


def _fp_target_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = {}
    for row in rows:
        if row.get("outcome") == "FP":
            labels[str(row["sample_id"])] = 1
        elif row.get("outcome") == "TN":
            labels[str(row["sample_id"])] = 0
    return labels


def _load_split_ids(split_dir: str | Path) -> dict[str, set[str]]:
    root = Path(split_dir)
    splits = {}
    for split in ["train", "val", "test"]:
        with (root / f"pope_{split}_ids.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        splits[split] = {str(sample_id) for sample_id in payload["sample_ids"]}
    return splits


def _indices_for_ids(sample_ids: list[str], wanted_ids: set[str]) -> np.ndarray:
    return np.array([idx for idx, sample_id in enumerate(sample_ids) if sample_id in wanted_ids])


def _explained_at(cumulative: np.ndarray, k: int) -> float:
    if len(cumulative) == 0:
        return math.nan
    idx = min(max(k, 1), len(cumulative)) - 1
    return float(cumulative[idx])


def _plot_position_heatmap(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    sub = df[(df["feature"] == "top_k_svd") & (df["k"].astype(str) == "128")].copy()
    if sub.empty:
        sub = df[df["feature"] == "full_difference"].copy()
    pivot = sub.pivot_table(index="position", columns="layer", values="auroc", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(pivot.values, vmin=0.45, vmax=max(0.75, float(np.nanmax(pivot.values))))
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Layer")
    ax.set_title("Stage K Position Robustness AUROC")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_k_position_layer_heatmap.png", dpi=180)
    plt.close(fig)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []
