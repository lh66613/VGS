"""Stage J destructive controls for blind-reference correction geometry."""

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
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

from vgs.artifacts import load_hidden_layer, read_jsonl
from vgs.geometry import cumulative_explained_variance, effective_rank, projection_similarity
from vgs.io import ensure_dir, write_csv


CONTROL_NAMES = [
    "real_matched",
    "image_shuffled",
    "blind_shuffled",
    "gaussian_matched",
]


def analyze_stage_j_controls(
    layers: list[int],
    k_grid: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    repeats: int,
    stability_sample_size: int | None,
    random_repeats: int,
    max_iter: int,
    split_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = _fp_target_labels(prediction_rows)
    split_ids = _load_split_ids(split_dir) if split_dir else None
    spectrum_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage J layers", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
        z_img = hidden["z_img"].float().numpy()
        z_blind = hidden["z_blind"].float().numpy()
        controls = _control_matrices(z_img, z_blind, seed + layer)

        if split_ids:
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
            split = None
            label_keep = []
            y = np.array([], dtype=np.int64)
        else:
            label_keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
            y = np.array([labels_by_id[sample_ids[idx]] for idx in label_keep], dtype=np.int64)
            if len(label_keep) and len(np.unique(y)) >= 2:
                split = _fixed_split(y, seed)
            else:
                split = None
            train_label_idx = []
            test_label_idx = []
            y_train = np.array([], dtype=np.int64)
            y_test = np.array([], dtype=np.int64)

        for control_name, matrix in tqdm(
            controls.items(),
            desc=f"L{layer} destructive controls",
            unit="control",
            leave=False,
        ):
            spectrum_matrix = (
                matrix[_indices_for_ids(sample_ids, split_ids["train"])]
                if split_ids
                else matrix
            )
            basis, singular_values = _svd_basis(spectrum_matrix)
            cumulative = cumulative_explained_variance(singular_values)
            spectrum_rows.append(_spectrum_row(layer, control_name, spectrum_matrix, singular_values, cumulative))
            stability_rows.extend(
                _stability_rows(
                    layer,
                    control_name,
                    spectrum_matrix,
                    k_grid,
                    seed + layer,
                    repeats,
                    stability_sample_size,
                )
            )
            if split_ids and len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                probe_rows.extend(
                    _probe_rows_prepared(
                        layer,
                        control_name,
                        matrix[train_label_idx],
                        matrix[test_label_idx],
                        y_train,
                        y_test,
                        basis,
                        seed,
                        max_iter,
                    )
                )
                if control_name == "real_matched":
                    probe_rows.extend(
                        _label_shuffle_rows_prepared(
                            layer,
                            matrix[train_label_idx],
                            matrix[test_label_idx],
                            y_train,
                            y_test,
                            basis,
                            seed,
                            max_iter,
                        )
                    )
                    random_rows.extend(
                        _random_subspace_rows_prepared(
                            layer,
                            matrix[train_label_idx],
                            matrix[test_label_idx],
                            z_img[train_label_idx],
                            z_img[test_label_idx],
                            z_blind[train_label_idx],
                            z_blind[test_label_idx],
                            y_train,
                            y_test,
                            basis,
                            k_grid,
                            seed,
                            random_repeats,
                            max_iter,
                        )
                    )
            elif split is not None:
                control_diff = matrix[label_keep]
                probe_rows.extend(
                    _probe_rows(
                        layer,
                        control_name,
                        control_diff,
                        y,
                        basis,
                        split,
                        seed,
                        max_iter,
                    )
                )
                if control_name == "real_matched":
                    probe_rows.extend(
                        _label_shuffle_rows(
                            layer,
                            control_diff,
                            y,
                            basis,
                            split,
                            seed,
                            max_iter,
                        )
                    )
                    random_rows.extend(
                        _random_subspace_rows(
                            layer,
                            control_diff,
                            z_img[label_keep],
                            z_blind[label_keep],
                            y,
                            basis,
                            k_grid,
                            split,
                            seed,
                            random_repeats,
                            max_iter,
                        )
                    )

    spectrum_path = write_csv(
        Path(output_dir) / "shuffle_spectrum_summary.csv",
        spectrum_rows,
        _fieldnames(spectrum_rows),
    )
    probe_path = write_csv(
        Path(output_dir) / "shuffle_probe_summary.csv",
        probe_rows,
        _fieldnames(probe_rows),
    )
    stability_path = write_csv(
        Path(output_dir) / "shuffle_stability_summary.csv",
        stability_rows,
        _fieldnames(stability_rows),
    )
    random_path = write_csv(
        Path(output_dir) / "random_subspace_control.csv",
        random_rows,
        _fieldnames(random_rows),
    )
    _plot_spectrum(spectrum_rows, Path(plot_dir))
    _plot_probe(probe_rows, Path(plot_dir))
    _plot_random_subspaces(random_rows, Path(plot_dir))
    return {
        "spectrum_path": str(spectrum_path),
        "probe_path": str(probe_path),
        "stability_path": str(stability_path),
        "random_subspace_path": str(random_path),
        "num_spectrum_rows": len(spectrum_rows),
        "num_probe_rows": len(probe_rows),
        "num_stability_rows": len(stability_rows),
        "num_random_subspace_rows": len(random_rows),
    }


def _control_matrices(z_img: np.ndarray, z_blind: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    image_perm = rng.permutation(z_img.shape[0])
    blind_perm = rng.permutation(z_blind.shape[0])
    real = z_blind - z_img
    gaussian = rng.normal(loc=float(real.mean()), scale=float(real.std()), size=real.shape)
    return {
        "real_matched": real,
        "image_shuffled": z_blind - z_img[image_perm],
        "blind_shuffled": z_blind[blind_perm] - z_img,
        "gaussian_matched": gaussian.astype(np.float32),
    }


def _svd_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt.T.astype(np.float32, copy=False), singular_values.astype(np.float64, copy=False)


def _spectrum_row(
    layer: int,
    control: str,
    matrix: np.ndarray,
    singular_values: np.ndarray,
    cumulative: np.ndarray,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "control": control,
        "num_samples": int(matrix.shape[0]),
        "hidden_dim": int(matrix.shape[1]),
        "effective_rank": effective_rank(singular_values),
        "top_singular_value": float(singular_values[0]) if len(singular_values) else math.nan,
        "explained_variance_k4": _explained_at(cumulative, 4),
        "explained_variance_k64": _explained_at(cumulative, 64),
        "explained_variance_k128": _explained_at(cumulative, 128),
        "explained_variance_k256": _explained_at(cumulative, 256),
    }


def _stability_rows(
    layer: int,
    control: str,
    matrix: np.ndarray,
    k_grid: list[int],
    seed: int,
    repeats: int,
    sample_size: int | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size > 0 and matrix.shape[0] > sample_size:
        matrix = matrix[rng.choice(matrix.shape[0], size=sample_size, replace=False)]
    rows = []
    for k in k_grid:
        k_eff = min(k, matrix.shape[0] // 2, matrix.shape[1])
        if k_eff <= 0:
            similarity = math.nan
            random_similarity = math.nan
        else:
            similarity = _split_half_stability(matrix, k_eff, repeats, rng)
            random_similarity = _random_stability(matrix.shape[1], k_eff, repeats, rng)
        rows.append(
            {
                "layer": layer,
                "control": control,
                "k": k,
                "effective_k": k_eff,
                "split_half_projection_similarity": similarity,
                "random_projection_similarity": random_similarity,
            }
        )
    return rows


def _split_half_stability(
    matrix: np.ndarray,
    k: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    sims = []
    n = matrix.shape[0]
    for _ in range(repeats):
        perm = rng.permutation(n)
        half = n // 2
        va = _randomized_basis(matrix[perm[:half]], k, rng)
        vb = _randomized_basis(matrix[perm[half : half * 2]], k, rng)
        sims.append(projection_similarity(va, vb))
    return float(np.mean(sims))


def _randomized_basis(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    _, _, vt = randomized_svd(
        matrix,
        n_components=k,
        n_iter=3,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    return vt.T


def _random_stability(dim: int, k: int, repeats: int, rng: np.random.Generator) -> float:
    sims = []
    for _ in range(repeats):
        qa = _random_basis(dim, k, rng)
        qb = _random_basis(dim, k, rng)
        sims.append(projection_similarity(qa, qb))
    return float(np.mean(sims))


def _probe_rows(
    layer: int,
    control: str,
    diff: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rows = []
    feature_specs = [("full_difference", "raw", diff), ("full_svd_coordinates", "svd", diff @ basis)]
    for k in [4, 64, 128, 256]:
        k_eff = min(k, basis.shape[1])
        feature_specs.append((f"top_k_svd_{k}", "top_k_svd", diff @ basis[:, :k_eff]))
    for feature, transform, x in feature_specs:
        rows.append(
            {
                "layer": layer,
                "control": control,
                "feature": feature,
                "transform": transform,
                "feature_dim": int(x.shape[1]),
                "num_samples": int(len(y)),
                "num_positive": int(y.sum()),
                **_fit_probe(x, y, split, seed, max_iter),
            }
        )
    return rows


def _probe_rows_prepared(
    layer: int,
    control: str,
    diff_train: np.ndarray,
    diff_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    basis: np.ndarray,
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rows = []
    feature_specs = [
        ("full_difference", "raw", diff_train, diff_test),
        ("full_svd_coordinates", "svd", diff_train @ basis, diff_test @ basis),
    ]
    for k in [4, 64, 128, 256]:
        k_eff = min(k, basis.shape[1])
        feature_specs.append(
            (
                f"top_k_svd_{k}",
                "top_k_svd",
                diff_train @ basis[:, :k_eff],
                diff_test @ basis[:, :k_eff],
            )
        )
    for feature, transform, x_train, x_test in feature_specs:
        rows.append(
            {
                "layer": layer,
                "control": control,
                "feature": feature,
                "transform": transform,
                "feature_dim": int(x_train.shape[1]),
                "num_samples": int(len(y_train) + len(y_test)),
                "num_positive": int(y_train.sum() + y_test.sum()),
                **_fit_prepared_probe(x_train, x_test, y_train, y_test, seed, max_iter),
            }
        )
    return rows


def _label_shuffle_rows(
    layer: int,
    diff: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + layer + 9001)
    shuffled = np.array(y, copy=True)
    rng.shuffle(shuffled)
    return [
        {
            "layer": layer,
            "control": "label_shuffled",
            "feature": "full_difference",
            "transform": "raw",
            "feature_dim": int(diff.shape[1]),
            "num_samples": int(len(shuffled)),
            "num_positive": int(shuffled.sum()),
            **_fit_probe(diff, shuffled, split, seed, max_iter),
        },
        {
            "layer": layer,
            "control": "label_shuffled",
            "feature": "full_svd_coordinates",
            "transform": "svd",
            "feature_dim": int(basis.shape[1]),
            "num_samples": int(len(shuffled)),
            "num_positive": int(shuffled.sum()),
            **_fit_probe(diff @ basis, shuffled, split, seed, max_iter),
        },
    ]


def _label_shuffle_rows_prepared(
    layer: int,
    diff_train: np.ndarray,
    diff_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    basis: np.ndarray,
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + layer + 9001)
    shuffled_train = np.array(y_train, copy=True)
    shuffled_test = np.array(y_test, copy=True)
    rng.shuffle(shuffled_train)
    rng.shuffle(shuffled_test)
    return [
        {
            "layer": layer,
            "control": "label_shuffled",
            "feature": "full_difference",
            "transform": "raw",
            "feature_dim": int(diff_train.shape[1]),
            "num_samples": int(len(shuffled_train) + len(shuffled_test)),
            "num_positive": int(shuffled_train.sum() + shuffled_test.sum()),
            **_fit_prepared_probe(diff_train, diff_test, shuffled_train, shuffled_test, seed, max_iter),
        },
        {
            "layer": layer,
            "control": "label_shuffled",
            "feature": "full_svd_coordinates",
            "transform": "svd",
            "feature_dim": int(basis.shape[1]),
            "num_samples": int(len(shuffled_train) + len(shuffled_test)),
            "num_positive": int(shuffled_train.sum() + shuffled_test.sum()),
            **_fit_prepared_probe(
                diff_train @ basis,
                diff_test @ basis,
                shuffled_train,
                shuffled_test,
                seed,
                max_iter,
            ),
        },
    ]


def _random_subspace_rows(
    layer: int,
    diff: np.ndarray,
    z_img: np.ndarray,
    z_blind: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
    k_grid: list[int],
    split: tuple[np.ndarray, np.ndarray],
    seed: int,
    repeats: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + layer + 1234)
    rows = []
    max_k = min(max(k_grid), diff.shape[1], diff.shape[0] - 1)
    pca_img = PCA(n_components=max_k, svd_solver="randomized", random_state=seed).fit(z_img[split[0]])
    pca_blind = PCA(n_components=max_k, svd_solver="randomized", random_state=seed).fit(z_blind[split[0]])
    for k in k_grid:
        k_eff = min(k, basis.shape[1], diff.shape[1])
        specs = [
            ("svd_top_k", "plain_svd", diff @ basis[:, :k_eff]),
            ("pca_img_top_k", "pca_img", pca_img.transform(z_img)[:, :k_eff]),
            ("pca_blind_top_k", "pca_blind", pca_blind.transform(z_blind)[:, :k_eff]),
        ]
        for feature, transform, x in specs:
            rows.append(_random_control_row(layer, k, feature, transform, x, y, split, seed, max_iter))
        for repeat in range(repeats):
            random_basis = _random_basis(diff.shape[1], k_eff, rng)
            rows.append(
                _random_control_row(
                    layer,
                    k,
                    f"random_orthogonal_{repeat}",
                    "random_orthogonal",
                    diff @ random_basis,
                    y,
                    split,
                    seed,
                    max_iter,
                )
            )
            start = int(rng.integers(0, max(1, basis.shape[1] - k_eff + 1)))
            rows.append(
                _random_control_row(
                    layer,
                    k,
                    f"random_svd_band_{repeat}",
                    "random_svd_band",
                    diff @ basis[:, start : start + k_eff],
                    y,
                    split,
                    seed,
                    max_iter,
                    band_start=start + 1,
                    band_end=start + k_eff,
                )
            )
    return rows


def _random_subspace_rows_prepared(
    layer: int,
    diff_train: np.ndarray,
    diff_test: np.ndarray,
    z_img_train: np.ndarray,
    z_img_test: np.ndarray,
    z_blind_train: np.ndarray,
    z_blind_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    basis: np.ndarray,
    k_grid: list[int],
    seed: int,
    repeats: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + layer + 1234)
    rows = []
    max_k = min(max(k_grid), diff_train.shape[1], diff_train.shape[0] - 1)
    pca_img = PCA(n_components=max_k, svd_solver="randomized", random_state=seed).fit(z_img_train)
    pca_blind = PCA(n_components=max_k, svd_solver="randomized", random_state=seed).fit(z_blind_train)
    for k in k_grid:
        k_eff = min(k, basis.shape[1], diff_train.shape[1])
        specs = [
            (
                "svd_top_k",
                "plain_svd",
                diff_train @ basis[:, :k_eff],
                diff_test @ basis[:, :k_eff],
            ),
            (
                "pca_img_top_k",
                "pca_img",
                pca_img.transform(z_img_train)[:, :k_eff],
                pca_img.transform(z_img_test)[:, :k_eff],
            ),
            (
                "pca_blind_top_k",
                "pca_blind",
                pca_blind.transform(z_blind_train)[:, :k_eff],
                pca_blind.transform(z_blind_test)[:, :k_eff],
            ),
        ]
        for feature, transform, x_train, x_test in specs:
            rows.append(
                _random_control_row_prepared(
                    layer,
                    k,
                    feature,
                    transform,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    seed,
                    max_iter,
                )
            )
        for repeat in range(repeats):
            random_basis = _random_basis(diff_train.shape[1], k_eff, rng)
            rows.append(
                _random_control_row_prepared(
                    layer,
                    k,
                    f"random_orthogonal_{repeat}",
                    "random_orthogonal",
                    diff_train @ random_basis,
                    diff_test @ random_basis,
                    y_train,
                    y_test,
                    seed,
                    max_iter,
                )
            )
            start = int(rng.integers(0, max(1, basis.shape[1] - k_eff + 1)))
            rows.append(
                _random_control_row_prepared(
                    layer,
                    k,
                    f"random_svd_band_{repeat}",
                    "random_svd_band",
                    diff_train @ basis[:, start : start + k_eff],
                    diff_test @ basis[:, start : start + k_eff],
                    y_train,
                    y_test,
                    seed,
                    max_iter,
                    band_start=start + 1,
                    band_end=start + k_eff,
                )
            )
    return rows


def _random_control_row(
    layer: int,
    k: int,
    feature: str,
    transform: str,
    x: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_iter: int,
    band_start: int | None = None,
    band_end: int | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "k": k,
        "feature": feature,
        "transform": transform,
        "feature_dim": int(x.shape[1]),
        "band_start": band_start or "",
        "band_end": band_end or "",
        "num_samples": int(len(y)),
        "num_positive": int(y.sum()),
        **_fit_probe(x, y, split, seed, max_iter),
    }


def _random_control_row_prepared(
    layer: int,
    k: int,
    feature: str,
    transform: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_iter: int,
    band_start: int | None = None,
    band_end: int | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "k": k,
        "feature": feature,
        "transform": transform,
        "feature_dim": int(x_train.shape[1]),
        "band_start": band_start or "",
        "band_end": band_end or "",
        "num_samples": int(len(y_train) + len(y_test)),
        "num_positive": int(y_train.sum() + y_test.sum()),
        **_fit_prepared_probe(x_train, x_test, y_train, y_test, seed, max_iter),
    }


def _fixed_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    stratify = y if min(np.bincount(y)) >= 2 else None
    return train_test_split(indices, test_size=0.3, random_state=seed, stratify=stratify)


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_iter: int,
) -> dict[str, float]:
    train_idx, test_idx = split
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y[train_idx])
    probabilities = clf.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y[test_idx], probabilities))
        if len(np.unique(y[test_idx])) > 1
        else math.nan,
        "auprc": float(average_precision_score(y[test_idx], probabilities)),
        "accuracy": float(accuracy_score(y[test_idx], predictions)),
        "f1": float(f1_score(y[test_idx], predictions, zero_division=0)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "n_iter": int(np.max(clf.n_iter_)),
    }


def _fit_prepared_probe(
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


def _random_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q[:, :k]


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
        path = root / f"pope_{split}_ids.json"
        with path.open("r", encoding="utf-8") as f:
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


def _plot_spectrum(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for control, group in df.groupby("control"):
        group = group.sort_values("layer")
        axes[0].plot(group["layer"], group["effective_rank"], marker="o", label=control)
        axes[1].plot(group["layer"], group["explained_variance_k4"], marker="o", label=control)
    axes[0].set_title("Effective Rank")
    axes[1].set_title("Explained Variance K=4")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_j_real_vs_shuffle_spectrum.png", dpi=180)
    plt.close(fig)


def _plot_probe(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    sub = df[df["feature"] == "full_svd_coordinates"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for control, group in sub.groupby("control"):
        group = group.sort_values("layer")
        ax.plot(group["layer"], group["auroc"], marker="o", label=control)
    ax.set_title("Stage J FP-vs-TN AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_j_real_vs_shuffle_auroc.png", dpi=180)
    plt.close(fig)


def _plot_random_subspaces(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    summary = df.groupby(["k", "transform"])["auroc"].mean().reset_index()
    for transform, group in summary.groupby("transform"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["auroc"], marker="o", label=transform)
    ax.set_title("Stage J Random Subspace Controls")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean AUROC")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_j_random_subspace_boxplot.png", dpi=180)
    plt.close(fig)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []
