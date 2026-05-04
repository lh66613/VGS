"""Stage L evidence-specific subspace extraction."""

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
from scipy.linalg import eigh
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

from vgs.artifacts import load_condition_hidden_layer, load_hidden_layer, read_jsonl
from vgs.geometry import projection_similarity
from vgs.io import ensure_dir, write_csv


def analyze_stage_l_evidence_subspace(
    layers: list[int],
    k_grid: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    condition_hidden_dir: str | Path,
    condition_plan_path: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    ridge: float,
    max_iter: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    labels_by_id = _fp_target_labels(read_jsonl(predictions_path))
    split_ids = _load_split_ids(split_dir)
    plan_rows = read_jsonl(condition_plan_path)
    plan_by_id = {str(row["sample_id"]): row for row in plan_rows}

    probe_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage L layers", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
        z_img = hidden["z_img"].float().numpy()
        z_blind = hidden["z_blind"].float().numpy()
        diff = z_blind - z_img

        train_all_idx = _indices_for_ids(sample_ids, split_ids["train"])
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
        diff_train = diff[train_label_idx]
        diff_test = diff[test_label_idx]

        condition_payload = load_condition_hidden_layer(condition_hidden_dir, layer)
        condition_sample_ids = [str(sample_id) for sample_id in condition_payload["sample_ids"]]
        conditions = {
            condition: tensor.float().numpy()
            for condition, tensor in condition_payload["conditions"].items()
        }
        blind = conditions["blind"]
        matched = blind - conditions["matched"]
        random_mismatch = blind - conditions["random_mismatch"]
        adversarial_mismatch = blind - conditions["adversarial_mismatch"]
        condition_train_idx = [
            idx
            for idx, sample_id in enumerate(condition_sample_ids)
            if sample_id in split_ids["train"]
        ]
        if not condition_train_idx:
            condition_train_idx = list(range(len(condition_sample_ids)))

        max_k = min(max(k_grid), diff.shape[1], diff[train_all_idx].shape[0] - 1)
        bases = _build_method_bases(
            diff[train_all_idx],
            diff_train,
            y_train,
            matched[condition_train_idx],
            random_mismatch[condition_train_idx],
            adversarial_mismatch[condition_train_idx],
            max_k,
            seed + layer,
            ridge,
        )
        half_bases = _build_stability_bases(
            diff[train_all_idx],
            diff_train,
            y_train,
            matched[condition_train_idx],
            random_mismatch[condition_train_idx],
            adversarial_mismatch[condition_train_idx],
            max_k,
            seed + layer,
            ridge,
        )

        for method, basis in tqdm(bases.items(), desc=f"L{layer} methods", unit="method", leave=False):
            for k in k_grid:
                k_eff = min(k, basis.shape[1])
                if k_eff <= 0:
                    continue
                x_train = diff_train @ basis[:, :k_eff]
                x_test = diff_test @ basis[:, :k_eff]
                probe_rows.append(
                    {
                        "layer": layer,
                        "method": method,
                        "k": k,
                        "effective_k": k_eff,
                        "feature_dim": k_eff,
                        **_fit_probe(x_train, x_test, y_train, y_test, seed, max_iter),
                    }
                )
                condition_rows.extend(
                    _condition_gap_rows(
                        layer,
                        method,
                        k,
                        basis[:, :k_eff],
                        condition_sample_ids,
                        plan_by_id,
                        matched,
                        random_mismatch,
                        adversarial_mismatch,
                    )
                )
                if method in half_bases:
                    left, right = half_bases[method]
                    stability_rows.append(
                        {
                            "layer": layer,
                            "method": method,
                            "k": k,
                            "effective_k": min(k_eff, left.shape[1], right.shape[1]),
                            "projection_similarity": projection_similarity(left[:, :k_eff], right[:, :k_eff]),
                        }
                    )

    probe_path = write_csv(
        Path(output_dir) / "evidence_subspace_probe.csv",
        probe_rows,
        _fieldnames(probe_rows),
    )
    condition_path = write_csv(
        Path(output_dir) / "evidence_subspace_condition_gap.csv",
        condition_rows,
        _fieldnames(condition_rows),
    )
    stability_path = write_csv(
        Path(output_dir) / "evidence_subspace_stability.csv",
        stability_rows,
        _fieldnames(stability_rows),
    )
    _plot_probe(probe_rows, Path(plot_dir))
    return {
        "probe_path": str(probe_path),
        "condition_gap_path": str(condition_path),
        "stability_path": str(stability_path),
        "num_probe_rows": len(probe_rows),
        "num_condition_rows": len(condition_rows),
        "num_stability_rows": len(stability_rows),
    }


def _build_method_bases(
    diff_train_all: np.ndarray,
    diff_train_labeled: np.ndarray,
    y_train: np.ndarray,
    matched_train: np.ndarray,
    random_train: np.ndarray,
    adversarial_train: np.ndarray,
    max_k: int,
    seed: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    mismatch_train = np.concatenate([random_train, adversarial_train], axis=0)
    bases = {
        "plain_svd": _svd_basis(diff_train_all, max_k, seed),
        "contrastive_pca": _contrastive_basis(matched_train, mismatch_train, max_k),
        "generalized_matched_vs_mismatch": _generalized_basis(matched_train, mismatch_train, max_k, ridge),
        "fisher_fp_tn": _supervised_vector_plus_pca(diff_train_labeled, y_train, max_k, "fisher", seed),
        "pls_fp_tn": _pls_basis(diff_train_labeled, y_train, max_k),
        "matched_vs_adversarial_logistic": _condition_logistic_basis(
            matched_train,
            adversarial_train,
            max_k,
            seed,
        ),
    }
    return {name: _orthonormal_columns(basis)[:, :max_k] for name, basis in bases.items()}


def _build_stability_bases(
    diff_train_all: np.ndarray,
    diff_train_labeled: np.ndarray,
    y_train: np.ndarray,
    matched_train: np.ndarray,
    random_train: np.ndarray,
    adversarial_train: np.ndarray,
    max_k: int,
    seed: int,
    ridge: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    result = {}
    all_perm = rng.permutation(diff_train_all.shape[0])
    label_perm = rng.permutation(diff_train_labeled.shape[0])
    cond_perm = rng.permutation(matched_train.shape[0])
    all_a, all_b = np.array_split(all_perm, 2)
    label_a, label_b = np.array_split(label_perm, 2)
    cond_a, cond_b = np.array_split(cond_perm, 2)
    for method in [
        "plain_svd",
        "contrastive_pca",
        "generalized_matched_vs_mismatch",
        "fisher_fp_tn",
        "pls_fp_tn",
        "matched_vs_adversarial_logistic",
    ]:
        if method == "plain_svd":
            left = _svd_basis(diff_train_all[all_a], max_k, seed + 1)
            right = _svd_basis(diff_train_all[all_b], max_k, seed + 2)
        elif method == "contrastive_pca":
            left = _contrastive_basis(
                matched_train[cond_a],
                np.concatenate([random_train[cond_a], adversarial_train[cond_a]], axis=0),
                max_k,
            )
            right = _contrastive_basis(
                matched_train[cond_b],
                np.concatenate([random_train[cond_b], adversarial_train[cond_b]], axis=0),
                max_k,
            )
        elif method == "generalized_matched_vs_mismatch":
            left = _generalized_basis(
                matched_train[cond_a],
                np.concatenate([random_train[cond_a], adversarial_train[cond_a]], axis=0),
                max_k,
                ridge,
            )
            right = _generalized_basis(
                matched_train[cond_b],
                np.concatenate([random_train[cond_b], adversarial_train[cond_b]], axis=0),
                max_k,
                ridge,
            )
        elif method == "fisher_fp_tn":
            left = _supervised_vector_plus_pca(
                diff_train_labeled[label_a],
                y_train[label_a],
                max_k,
                "fisher",
                seed + 1,
            )
            right = _supervised_vector_plus_pca(
                diff_train_labeled[label_b],
                y_train[label_b],
                max_k,
                "fisher",
                seed + 2,
            )
        elif method == "pls_fp_tn":
            left = _pls_basis(diff_train_labeled[label_a], y_train[label_a], max_k)
            right = _pls_basis(diff_train_labeled[label_b], y_train[label_b], max_k)
        else:
            left = _condition_logistic_basis(
                matched_train[cond_a],
                adversarial_train[cond_a],
                max_k,
                seed + 1,
            )
            right = _condition_logistic_basis(
                matched_train[cond_b],
                adversarial_train[cond_b],
                max_k,
                seed + 2,
            )
        result[method] = (_orthonormal_columns(left)[:, :max_k], _orthonormal_columns(right)[:, :max_k])
    return result


def _svd_basis(matrix: np.ndarray, max_k: int, seed: int) -> np.ndarray:
    _, _, vt = randomized_svd(
        matrix,
        n_components=max_k,
        n_iter=4,
        random_state=seed,
    )
    return vt.T


def _contrastive_basis(matched: np.ndarray, mismatch: np.ndarray, max_k: int) -> np.ndarray:
    cov = _covariance(matched) - _covariance(mismatch)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return vectors[:, order[:max_k]]


def _generalized_basis(matched: np.ndarray, mismatch: np.ndarray, max_k: int, ridge: float) -> np.ndarray:
    cov_matched = _covariance(matched)
    cov_mismatch = _covariance(mismatch)
    scale = float(np.trace(cov_mismatch) / max(cov_mismatch.shape[0], 1))
    regularized = cov_mismatch + np.eye(cov_mismatch.shape[0], dtype=np.float64) * ridge * max(scale, 1e-6)
    values, vectors = eigh(
        cov_matched,
        regularized,
        subset_by_index=[cov_matched.shape[0] - max_k, cov_matched.shape[0] - 1],
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    return vectors[:, order]


def _supervised_vector_plus_pca(
    matrix: np.ndarray,
    y: np.ndarray,
    max_k: int,
    method: str,
    seed: int,
) -> np.ndarray:
    if method != "fisher":
        raise ValueError(method)
    positive = matrix[y == 1]
    negative = matrix[y == 0]
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    direction = direction[:, None]
    if max_k <= 1:
        return direction
    pca = _svd_basis(matrix, max_k - 1, seed)
    return np.concatenate([direction, pca], axis=1)


def _pls_basis(matrix: np.ndarray, y: np.ndarray, max_k: int) -> np.ndarray:
    n_components = min(max_k, matrix.shape[1], max(1, matrix.shape[0] - 1))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x_scaled, y.astype(np.float64))
    return pls.x_weights_ / np.maximum(scaler.scale_[:, None], 1e-12)


def _condition_logistic_basis(
    matched: np.ndarray,
    adversarial: np.ndarray,
    max_k: int,
    seed: int,
) -> np.ndarray:
    x = np.concatenate([matched, adversarial], axis=0)
    y = np.concatenate([np.ones(matched.shape[0], dtype=np.int64), np.zeros(adversarial.shape[0], dtype=np.int64)])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, solver="lbfgs")
    clf.fit(x_scaled, y)
    direction = (clf.coef_[0] / np.maximum(scaler.scale_, 1e-12))[:, None]
    if max_k <= 1:
        return direction
    pca = _svd_basis(x, max_k - 1, seed)
    return np.concatenate([direction, pca], axis=1)


def _condition_gap_rows(
    layer: int,
    method: str,
    k: int,
    basis: np.ndarray,
    sample_ids: list[str],
    plan_by_id: dict[str, dict[str, Any]],
    matched: np.ndarray,
    random_mismatch: np.ndarray,
    adversarial_mismatch: np.ndarray,
) -> list[dict[str, Any]]:
    matched_scores = np.sum((matched @ basis) ** 2, axis=1)
    random_scores = np.sum((random_mismatch @ basis) ** 2, axis=1)
    adversarial_scores = np.sum((adversarial_mismatch @ basis) ** 2, axis=1)
    rows = []
    for condition, scores in [
        ("matched", matched_scores),
        ("random_mismatch", random_scores),
        ("adversarial_mismatch", adversarial_scores),
    ]:
        rows.extend(_condition_summary_rows(layer, method, k, condition, sample_ids, plan_by_id, scores))
    rows.append(_delta_row(layer, method, k, "matched_minus_random_mismatch", matched_scores - random_scores))
    rows.append(
        _delta_row(
            layer,
            method,
            k,
            "matched_minus_adversarial_mismatch",
            matched_scores - adversarial_scores,
        )
    )
    return rows


def _condition_summary_rows(
    layer: int,
    method: str,
    k: int,
    condition: str,
    sample_ids: list[str],
    plan_by_id: dict[str, dict[str, Any]],
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            "layer": layer,
            "method": method,
            "k": k,
            "row_type": "condition_summary",
            "condition": condition,
            "comparison": "",
            "outcome": "ALL",
            "n": int(len(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        }
    ]
    for outcome in ["FP", "TN"]:
        keep = np.array([plan_by_id.get(sample_id, {}).get("outcome") == outcome for sample_id in sample_ids])
        if keep.any():
            values = scores[keep]
            rows.append(
                {
                    "layer": layer,
                    "method": method,
                    "k": k,
                    "row_type": "condition_outcome_summary",
                    "condition": condition,
                    "comparison": "",
                    "outcome": outcome,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                }
            )
    return rows


def _delta_row(layer: int, method: str, k: int, comparison: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "layer": layer,
        "method": method,
        "k": k,
        "row_type": "condition_delta",
        "condition": "",
        "comparison": comparison,
        "outcome": "ALL",
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


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
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed, solver="lbfgs")
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


def _covariance(matrix: np.ndarray) -> np.ndarray:
    centered = matrix.astype(np.float64, copy=False) - matrix.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(centered.shape[0] - 1, 1)


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


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


def _plot_probe(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in df.groupby("method"):
        best = group.groupby("k")["auroc"].mean().reset_index().sort_values("k")
        ax.plot(best["k"], best["auroc"], marker="o", label=method)
    ax.set_title("Stage L Evidence-Specific Subspaces")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean FP-vs-TN AUROC")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_l_plain_svd_vs_evidence_specific.png", dpi=180)
    plt.close(fig)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []
