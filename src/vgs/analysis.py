"""CPU-side matrix, spectrum, and probe analysis."""

from __future__ import annotations

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
from tqdm.auto import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from vgs.artifacts import (
    load_difference_matrix,
    load_hidden_layer,
    load_svd,
    read_jsonl,
    save_difference_matrix,
    save_svd,
)
from vgs.geometry import cumulative_explained_variance, effective_rank, projection_similarity
from vgs.io import ensure_dir, write_csv


def build_difference_matrices(
    layers: list[int],
    hidden_states_dir: str | Path,
    output_dir: str | Path,
    control: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    generator = torch.Generator().manual_seed(seed)
    for layer in tqdm(layers, desc="build D matrices", unit="layer"):
        payload = load_hidden_layer(hidden_states_dir, layer)
        z_img = payload["z_img"].float()
        z_blind = payload["z_blind"].float()
        sample_ids = list(payload["sample_ids"])
        if control == "none":
            matrix = z_blind - z_img
        elif control == "shuffle_image_question":
            perm = torch.randperm(z_img.shape[0], generator=generator)
            matrix = z_blind - z_img[perm]
        elif control == "shuffle_blind_image":
            perm = torch.randperm(z_blind.shape[0], generator=generator)
            matrix = z_blind[perm] - z_img
        elif control == "gaussian":
            ref = z_blind - z_img
            matrix = torch.randn(ref.shape, generator=generator) * ref.std() + ref.mean()
        else:
            raise ValueError(f"Unknown control: {control}")
        path = save_difference_matrix(
            output_dir,
            layer,
            sample_ids,
            matrix,
            metadata={"control": control, "source_hidden_states_dir": str(hidden_states_dir)},
        )
        rows.append(
            {
                "layer": layer,
                "control": control,
                "num_samples": int(matrix.shape[0]),
                "hidden_dim": int(matrix.shape[1]),
                "path": str(path),
            }
        )
    return rows


def analyze_spectra(
    layers: list[int],
    matrix_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    rows = []
    for layer in tqdm(layers, desc="analyze spectra", unit="layer"):
        payload = load_difference_matrix(matrix_dir, layer)
        matrix = payload["D"].float()
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        save_svd(output_dir, layer, list(payload.get("sample_ids", [])), singular_values, vh)
        s_np = singular_values.numpy()
        cumulative = cumulative_explained_variance(s_np)
        row = {
            "layer": layer,
            "num_samples": int(matrix.shape[0]),
            "hidden_dim": int(matrix.shape[1]),
            "top_singular_value": float(s_np[0]) if len(s_np) else math.nan,
            "effective_rank": effective_rank(s_np),
            "explained_variance_k4": _explained_at(cumulative, 4),
            "explained_variance_k8": _explained_at(cumulative, 8),
            "explained_variance_k16": _explained_at(cumulative, 16),
            "explained_variance_k32": _explained_at(cumulative, 32),
        }
        rows.append(row)
        _plot_spectrum(layer, s_np, cumulative, Path(plot_dir))

    write_csv(Path(output_dir) / "effective_rank_summary.csv", rows, list(rows[0].keys()) if rows else [])
    return rows


def analyze_k_sensitivity(
    layers: list[int],
    k_grid: list[int],
    svd_dir: str | Path,
    matrix_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    repeats: int = 10,
    stability_method: str = "randomized",
    stability_sample_size: int | None = 1024,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for layer in tqdm(layers, desc="K sensitivity layers", unit="layer"):
        svd = load_svd(svd_dir, layer)
        singular_values = svd["singular_values"].numpy()
        cumulative = cumulative_explained_variance(singular_values)
        matrix = load_difference_matrix(matrix_dir, layer)["D"].float().numpy()
        stability_matrix = _subsample_rows(matrix, stability_sample_size, rng)
        for k in tqdm(k_grid, desc=f"layer {layer} K grid", unit="K", leave=False):
            k_eff = min(k, stability_matrix.shape[0] // 2, stability_matrix.shape[1])
            stability = (
                _split_half_stability(
                    stability_matrix,
                    k_eff,
                    repeats,
                    rng,
                    method=stability_method,
                    desc=f"L{layer} K{k} split-half",
                )
                if k_eff > 0
                else math.nan
            )
            random_stability = _random_subspace_stability(matrix.shape[1], k_eff, repeats, rng) if k_eff > 0 else math.nan
            rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "effective_k": k_eff,
                    "explained_variance": _explained_at(cumulative, k),
                    "split_half_projection_similarity": stability,
                    "random_projection_similarity": random_stability,
                }
            )
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(Path(output_dir) / "k_sensitivity_summary.csv", rows, fieldnames)
    _plot_k_sensitivity(rows, Path(plot_dir))
    return rows


def train_probe_models(
    layers: list[int],
    k_grid: list[int],
    feature_families: list[str],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    output_dir: str | Path,
    seed: int,
) -> list[dict[str, Any]]:
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = _fp_target_labels(prediction_rows)
    rows = []
    total_tasks = sum(len(_family_k_values(family, k_grid)) for family in feature_families) * len(layers)
    progress = tqdm(total=total_tasks, desc="train probes", unit="model")
    for layer in layers:
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = list(hidden["sample_ids"])
        keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
        if not keep:
            continue
        y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
        z_img = hidden["z_img"][keep].float().numpy()
        z_blind = hidden["z_blind"][keep].float().numpy()
        diff = z_blind - z_img
        svd = load_svd(svd_dir, layer)
        basis = svd["Vh"].float().numpy().T
        for family in feature_families:
            for k in _family_k_values(family, k_grid):
                x = _build_feature(family, k, z_img, z_blind, diff, basis, seed)
                metrics = _fit_probe(x, y, seed)
                progress.update(1)
                rows.append(
                    {
                        "layer": layer,
                        "feature_family": family,
                        "k": k if k is not None else "full",
                        "num_samples": int(len(y)),
                        "num_positive": int(y.sum()),
                        **metrics,
                    }
                )
    progress.close()
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(Path(output_dir) / "probe_results.csv", rows, fieldnames)
    return rows


def compare_probe_features(probe_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(probe_dir) / "probe_results.csv"
    df = pd.read_csv(path)
    sort_cols = [col for col in ["auroc", "auprc", "accuracy", "f1"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    target = Path(output_dir) / "feature_comparison.csv"
    ensure_dir(target.parent)
    df.to_csv(target, index=False)
    return df.to_dict(orient="records")


def layerwise_summary(
    layers: list[int],
    k_grid: list[int],
    svd_dir: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
) -> dict[str, Any]:
    rows = []
    effective_rank_path = Path(svd_dir) / "effective_rank_summary.csv"
    k_path = Path(svd_dir) / "k_sensitivity_summary.csv"
    probe_path = Path(probe_dir) / "probe_results.csv"
    rank_df = pd.read_csv(effective_rank_path) if effective_rank_path.exists() else pd.DataFrame()
    k_df = pd.read_csv(k_path) if k_path.exists() else pd.DataFrame()
    probe_df = pd.read_csv(probe_path) if probe_path.exists() else pd.DataFrame()
    for layer in tqdm(layers, desc="layerwise summary", unit="layer"):
        row = {"layer": layer}
        if not rank_df.empty:
            match = rank_df[rank_df["layer"] == layer]
            if not match.empty:
                row.update(match.iloc[0].to_dict())
        if not k_df.empty:
            match = k_df[k_df["layer"] == layer]
            if not match.empty:
                best = match.sort_values("split_half_projection_similarity", ascending=False).iloc[0]
                row["best_stability_k"] = int(best["k"])
                row["best_split_half_similarity"] = float(best["split_half_projection_similarity"])
        if not probe_df.empty and "auroc" in probe_df:
            match = probe_df[probe_df["layer"] == layer]
            if not match.empty:
                best = match.sort_values("auroc", ascending=False, na_position="last").iloc[0]
                row["best_probe_family"] = best["feature_family"]
                row["best_probe_k"] = best["k"]
                row["best_probe_auroc"] = float(best["auroc"])
        rows.append(row)
    write_csv(Path(output_dir) / "layerwise_summary.csv", rows, sorted({k for row in rows for k in row}))
    angle_rows = _layer_angle_rows(layers, k_grid, svd_dir)
    if angle_rows:
        write_csv(Path(output_dir) / "layer_angle_summary.csv", angle_rows, list(angle_rows[0].keys()))
        _plot_layer_angles(angle_rows, Path(plot_dir))
    return {"summary_rows": rows, "angle_rows": angle_rows}


def analyze_stage_c_deep(
    layers: list[int],
    focus_layers: list[int],
    k_grid: list[int],
    bands: list[tuple[int, int]],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = _fp_target_labels(prediction_rows)
    topk_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage C deep layers", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = list(hidden["sample_ids"])
        keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
        if not keep:
            continue
        y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
        z_img = hidden["z_img"][keep].float().numpy()
        z_blind = hidden["z_blind"][keep].float().numpy()
        diff = z_blind - z_img
        svd = load_svd(svd_dir, layer)
        singular_values = svd["singular_values"].float().numpy()
        cumulative = cumulative_explained_variance(singular_values)
        basis = svd["Vh"].float().numpy().T

        full_metrics = _fit_probe(diff, y, seed)
        layer_top_rows = []
        for k in tqdm(k_grid, desc=f"L{layer} top-K probes", unit="K", leave=False):
            k_eff = min(k, basis.shape[1])
            x = diff @ basis[:, :k_eff]
            metrics = _fit_probe(x, y, seed)
            row = {
                "layer": layer,
                "feature": "top_k",
                "k": k,
                "effective_k": k_eff,
                "explained_variance": _explained_at(cumulative, k_eff),
                "num_samples": int(len(y)),
                "num_positive": int(y.sum()),
                **metrics,
            }
            topk_rows.append(row)
            layer_top_rows.append(row)

        if layer in focus_layers:
            for start, end in tqdm(bands, desc=f"L{layer} SVD bands", unit="band", leave=False):
                band_start = max(start, 1)
                band_end = min(end, basis.shape[1])
                if band_end < band_start:
                    continue
                width = band_end - band_start + 1
                band_basis = basis[:, band_start - 1 : band_end]
                band_feature = diff @ band_basis
                band_metrics = _fit_probe(band_feature, y, seed)
                band_rows.append(
                    {
                        "layer": layer,
                        "feature": "svd_band",
                        "band": f"{band_start}-{band_end}",
                        "start": band_start,
                        "end": band_end,
                        "width": width,
                        "explained_variance_start": _explained_at(cumulative, band_start),
                        "explained_variance_end": _explained_at(cumulative, band_end),
                        "explained_variance_delta": _explained_at(cumulative, band_end)
                        - _explained_at(cumulative, band_start - 1),
                        "num_samples": int(len(y)),
                        "num_positive": int(y.sum()),
                        **band_metrics,
                    }
                )
                random_basis = _random_basis(diff.shape[1], width, seed + layer + band_start)
                random_metrics = _fit_probe(diff @ random_basis, y, seed)
                band_rows.append(
                    {
                        "layer": layer,
                        "feature": "random_band_width",
                        "band": f"random_width_{width}_for_{band_start}-{band_end}",
                        "start": band_start,
                        "end": band_end,
                        "width": width,
                        "explained_variance_start": math.nan,
                        "explained_variance_end": math.nan,
                        "explained_variance_delta": math.nan,
                        "num_samples": int(len(y)),
                        "num_positive": int(y.sum()),
                        **random_metrics,
                    }
                )

        best_top = max(layer_top_rows, key=lambda row: row["auroc"]) if layer_top_rows else {}
        diagnostic_rows.append(
            {
                "layer": layer,
                "effective_rank": effective_rank(singular_values),
                "explained_variance_k4": _explained_at(cumulative, 4),
                "explained_variance_k64": _explained_at(cumulative, 64),
                "explained_variance_k256": _explained_at(cumulative, 256),
                "full_difference_auroc": full_metrics["auroc"],
                "full_difference_auprc": full_metrics["auprc"],
                "best_top_k": best_top.get("k"),
                "best_top_k_auroc": best_top.get("auroc"),
                "best_top_k_explained_variance": best_top.get("explained_variance"),
            }
        )

    topk_fields = list(topk_rows[0].keys()) if topk_rows else []
    band_fields = list(band_rows[0].keys()) if band_rows else []
    diagnostic_fields = list(diagnostic_rows[0].keys()) if diagnostic_rows else []
    write_csv(Path(output_dir) / "stage_c_topk_curve.csv", topk_rows, topk_fields)
    write_csv(Path(output_dir) / "stage_c_band_probe.csv", band_rows, band_fields)
    write_csv(Path(output_dir) / "stage_c_layer_diagnostics.csv", diagnostic_rows, diagnostic_fields)
    _plot_stage_c_topk(topk_rows, Path(plot_dir))
    _plot_stage_c_band_probe(band_rows, Path(plot_dir))
    _plot_stage_c_layer_diagnostics(diagnostic_rows, Path(plot_dir))
    return {
        "topk_rows": topk_rows,
        "band_rows": band_rows,
        "diagnostic_rows": diagnostic_rows,
    }


def analyze_stage_c_supervised(
    layers: list[int],
    focus_layers: list[int],
    k_grid: list[int],
    exclusion_k_grid: list[int],
    bands: list[tuple[int, int]],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    pls_components: int = 8,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = _fp_target_labels(prediction_rows)
    alignment_rows: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="supervised Stage C layers", unit="layer"):
        diff, y, basis, cumulative = _load_diff_labels_basis(
            layer, hidden_states_dir, svd_dir, labels_by_id
        )
        supervised_bases = _supervised_direction_bases(diff, y, seed, pls_components)
        for method, supervised_basis in supervised_bases.items():
            for k in k_grid:
                k_eff = min(k, basis.shape[1])
                similarity = _directed_projection_similarity(supervised_basis, basis[:, :k_eff])
                alignment_rows.append(
                    {
                        "layer": layer,
                        "method": method,
                        "svd_k": k,
                        "effective_svd_k": k_eff,
                        "supervised_dim": supervised_basis.shape[1],
                        "projection_similarity": similarity,
                        "projection_norm": math.sqrt(max(similarity, 0.0))
                        if supervised_basis.shape[1] == 1
                        else math.nan,
                        "principal_angle_degrees": math.degrees(
                            math.acos(min(1.0, math.sqrt(max(similarity, 0.0))))
                        )
                        if supervised_basis.shape[1] == 1
                        else math.nan,
                    }
                )

        for k in tqdm(k_grid, desc=f"L{layer} extended top-K", unit="K", leave=False):
            k_eff = min(k, basis.shape[1])
            coords = diff @ basis[:, :k_eff]
            metrics = _fit_probe(coords, y, seed)
            k_rows.append(
                {
                    "layer": layer,
                    "feature": "only_top_1_to_k",
                    "k": k,
                    "effective_k": k_eff,
                    "explained_variance": _explained_at(cumulative, k_eff),
                    "num_samples": int(len(y)),
                    "num_positive": int(y.sum()),
                    **metrics,
                }
            )

        if layer not in focus_layers:
            continue

        svd_coords = diff @ basis
        full_metrics = _fit_probe(svd_coords, y, seed)
        cumulative_rows.append(
            {
                "layer": layer,
                "feature": "full_svd_coordinates",
                "k": "full",
                "removed": "none",
                "remaining_dim": int(svd_coords.shape[1]),
                "explained_variance_removed": 0.0,
                **full_metrics,
            }
        )
        for k in tqdm(
            exclusion_k_grid,
            desc=f"L{layer} cumulative exclusion",
            unit="K",
            leave=False,
        ):
            k_eff = min(k, basis.shape[1])
            only_metrics = _fit_probe(svd_coords[:, :k_eff], y, seed)
            cumulative_rows.append(
                {
                    "layer": layer,
                    "feature": "only_top_1_to_k",
                    "k": k,
                    "removed": "complement",
                    "remaining_dim": k_eff,
                    "explained_variance_removed": 1.0 - _explained_at(cumulative, k_eff),
                    **only_metrics,
                }
            )
            remove_metrics = _fit_probe(svd_coords[:, k_eff:], y, seed)
            cumulative_rows.append(
                {
                    "layer": layer,
                    "feature": "remove_top_1_to_k",
                    "k": k,
                    "removed": f"1-{k_eff}",
                    "remaining_dim": int(svd_coords.shape[1] - k_eff),
                    "explained_variance_removed": _explained_at(cumulative, k_eff),
                    **remove_metrics,
                }
            )

        for start, end in tqdm(bands, desc=f"L{layer} band exclusion", unit="band", leave=False):
            band_start = max(start, 1)
            band_end = min(end, basis.shape[1])
            if band_end < band_start:
                continue
            band_slice = slice(band_start - 1, band_end)
            band_width = band_end - band_start + 1
            only_metrics = _fit_probe(svd_coords[:, band_slice], y, seed)
            band_rows.append(
                {
                    "layer": layer,
                    "feature": "only_band",
                    "band": f"{band_start}-{band_end}",
                    "start": band_start,
                    "end": band_end,
                    "width": band_width,
                    "remaining_dim": band_width,
                    "explained_variance_removed": 1.0
                    - (
                        _explained_at(cumulative, band_end)
                        - _explained_at(cumulative, band_start - 1)
                    ),
                    **only_metrics,
                }
            )
            kept = np.concatenate([svd_coords[:, : band_start - 1], svd_coords[:, band_end:]], axis=1)
            remove_metrics = _fit_probe(kept, y, seed)
            band_rows.append(
                {
                    "layer": layer,
                    "feature": "remove_band",
                    "band": f"{band_start}-{band_end}",
                    "start": band_start,
                    "end": band_end,
                    "width": band_width,
                    "remaining_dim": int(kept.shape[1]),
                    "explained_variance_removed": _explained_at(cumulative, band_end)
                    - _explained_at(cumulative, band_start - 1),
                    **remove_metrics,
                }
            )

    write_csv(
        Path(output_dir) / "stage_c_supervised_alignment.csv",
        alignment_rows,
        list(alignment_rows[0].keys()) if alignment_rows else [],
    )
    write_csv(
        Path(output_dir) / "stage_c_extended_k_curve.csv",
        k_rows,
        list(k_rows[0].keys()) if k_rows else [],
    )
    write_csv(
        Path(output_dir) / "stage_c_cumulative_exclusion.csv",
        cumulative_rows,
        list(cumulative_rows[0].keys()) if cumulative_rows else [],
    )
    write_csv(
        Path(output_dir) / "stage_c_band_exclusion.csv",
        band_rows,
        list(band_rows[0].keys()) if band_rows else [],
    )
    _plot_supervised_alignment(alignment_rows, Path(plot_dir))
    _plot_extended_k_curve(k_rows, Path(plot_dir))
    _plot_cumulative_exclusion(cumulative_rows, Path(plot_dir))
    _plot_band_exclusion(band_rows, Path(plot_dir))
    return {
        "alignment_rows": alignment_rows,
        "k_rows": k_rows,
        "cumulative_rows": cumulative_rows,
        "band_rows": band_rows,
    }


def analyze_stage_c_coordinate_control(
    layers: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
    standardize: bool = True,
    max_iter: int = 2000,
    c_value: float = 1.0,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = _fp_target_labels(prediction_rows)
    rows: list[dict[str, Any]] = []
    rotation_cache: dict[int, np.ndarray] = {}

    for layer in tqdm(layers, desc="coordinate-control layers", unit="layer"):
        diff, y, basis, _ = _load_diff_labels_basis(layer, hidden_states_dir, svd_dir, labels_by_id)
        if len(np.unique(y)) < 2:
            continue
        indices = np.arange(len(y))
        stratify = y if min(np.bincount(y)) >= 2 else None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=seed,
            stratify=stratify,
        )
        common = {
            "layer": layer,
            "num_samples": int(len(y)),
            "num_positive": int(y.sum()),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "standardize": bool(standardize),
            "solver": "lbfgs",
            "class_weight": "balanced",
            "C": float(c_value),
            "max_iter": int(max_iter),
        }

        feature_specs = [
            ("raw_full_difference", "identity", diff),
            ("full_svd_coordinates", "all_svd_coordinates", diff @ basis),
        ]
        dim = diff.shape[1]
        if dim not in rotation_cache:
            rotation_cache[dim] = _dense_random_orthogonal(dim, seed + dim)
        feature_specs.append(
            (
                "random_orthogonal_rotation",
                "dense_random_orthogonal",
                diff @ rotation_cache[dim],
            )
        )

        for feature_name, transform, feature_matrix in tqdm(
            feature_specs,
            desc=f"L{layer} coordinate controls",
            unit="feature",
            leave=False,
        ):
            metrics = _fit_probe_fixed_split(
                feature_matrix,
                y,
                train_idx,
                test_idx,
                seed,
                standardize=standardize,
                max_iter=max_iter,
                c_value=c_value,
            )
            rows.append(
                {
                    **common,
                    "feature": feature_name,
                    "transform": transform,
                    "feature_dim": int(feature_matrix.shape[1]),
                    **metrics,
                }
            )

        whitened_train, whitened_test = _train_pca_whiten(diff, train_idx, test_idx)
        whitened_metrics = _fit_prepared_probe(
            whitened_train,
            whitened_test,
            y[train_idx],
            y[test_idx],
            seed,
            standardize=standardize,
            max_iter=max_iter,
            c_value=c_value,
        )
        rows.append(
            {
                **common,
                "feature": "pca_whitened_difference",
                "transform": "train_split_pca_whitening",
                "feature_dim": int(whitened_train.shape[1]),
                **whitened_metrics,
            }
        )

    write_csv(
        Path(output_dir) / "stage_c_coordinate_control.csv",
        rows,
        list(rows[0].keys()) if rows else [],
    )
    _plot_coordinate_control(rows, Path(plot_dir))
    return {"coordinate_control_rows": rows}


def _plot_spectrum(layer: int, singular_values: np.ndarray, cumulative: np.ndarray, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(singular_values)
    axes[0].set_title(f"Layer {layer} singular values")
    axes[1].plot(singular_values / max(float(singular_values[0]), 1e-12))
    axes[1].set_title("Normalized")
    axes[2].plot(cumulative)
    axes[2].set_title("Cumulative variance")
    for ax in axes:
        ax.set_xlabel("Index")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / f"spectrum_layer_{layer}.png", dpi=160)
    plt.close(fig)


def _plot_k_sensitivity(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    ensure_dir(plot_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for layer, group in df.groupby("layer"):
        axes[0].plot(group["k"], group["explained_variance"], marker="o", label=f"L{layer}")
        axes[1].plot(group["k"], group["split_half_projection_similarity"], marker="o", label=f"L{layer}")
    axes[0].set_title("Explained variance vs K")
    axes[1].set_title("Split-half stability vs K")
    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "k_sensitivity_plot.png", dpi=160)
    plt.close(fig)


def _plot_layer_angles(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for k, group in tqdm(df.groupby("k"), desc="plot layer angles", unit="K"):
        pivot = group.pivot(index="layer_a", columns="layer_b", values="projection_similarity")
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(pivot.fillna(0).values, vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title(f"Layer subspace similarity K={k}")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(Path(plot_dir) / f"layer_angle_heatmap_k{k}.png", dpi=160)
        plt.close(fig)


def _plot_stage_c_topk(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for layer, group in df.groupby("layer"):
        group = group.sort_values("k")
        axes[0].plot(group["k"], group["auroc"], marker="o", label=f"L{layer}")
        axes[1].plot(group["k"], group["explained_variance"], marker="o", label=f"L{layer}")
    axes[0].set_title("Projected Difference AUROC vs K")
    axes[1].set_title("Explained Variance vs K")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_topk_auroc_explained_variance.png", dpi=180)
    plt.close(fig)

    for layer, group in df.groupby("layer"):
        group = group.sort_values("k")
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(group["k"], group["auroc"], marker="o", color="tab:blue", label="AUROC")
        ax1.set_xlabel("K")
        ax1.set_xscale("log", base=2)
        ax1.set_ylabel("AUROC", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(
            group["k"],
            group["explained_variance"],
            marker="s",
            color="tab:orange",
            label="Explained variance",
        )
        ax2.set_ylabel("Explained variance", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        fig.suptitle(f"Layer {layer}: AUROC vs Variance Growth")
        fig.tight_layout()
        fig.savefig(plot_dir / f"stage_c_layer_{layer}_auroc_vs_variance.png", dpi=180)
        plt.close(fig)


def _plot_stage_c_band_probe(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    svd_df = df[df["feature"] == "svd_band"].copy()
    random_df = df[df["feature"] == "random_band_width"].copy()
    if svd_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for layer, group in svd_df.groupby("layer"):
        group = group.sort_values("start")
        ax.plot(group["band"], group["auroc"], marker="o", label=f"L{layer} SVD band")
    if not random_df.empty:
        random_summary = random_df.groupby(["layer", "start"])["auroc"].mean().reset_index()
        band_names = svd_df.sort_values("start")["band"].drop_duplicates().tolist()
        for layer, group in random_summary.groupby("layer"):
            group = group.sort_values("start")
            ax.plot(band_names[: len(group)], group["auroc"], linestyle="--", alpha=0.55, label=f"L{layer} random")
    ax.set_title("Non Top-K SVD Band Probe")
    ax.set_xlabel("Singular direction band")
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_band_probe_auroc.png", dpi=180)
    plt.close(fig)


def _plot_stage_c_layer_diagnostics(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(df["layer"], df["explained_variance_k4"], marker="o", label="EV K=4")
    axes[0].plot(df["layer"], df["explained_variance_k64"], marker="o", label="EV K=64")
    axes[0].plot(df["layer"], df["explained_variance_k256"], marker="o", label="EV K=256")
    axes[0].set_title("Variance Concentration")
    axes[0].set_ylabel("Explained variance")
    axes[1].plot(df["layer"], df["full_difference_auroc"], marker="o", label="Full diff")
    axes[1].plot(df["layer"], df["best_top_k_auroc"], marker="o", label="Best top-K")
    axes[1].set_title("Probe AUROC")
    axes[1].set_ylabel("AUROC")
    axes[2].plot(df["layer"], df["effective_rank"], marker="o")
    axes[2].set_title("Effective Rank")
    axes[2].set_ylabel("Effective rank")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if ax is not axes[2]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_layer_diagnostics.png", dpi=180)
    plt.close(fig)


def _plot_supervised_alignment(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    for method, group in df.groupby("method"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for layer, layer_group in group.groupby("layer"):
            layer_group = layer_group.sort_values("svd_k")
            ax.plot(
                layer_group["svd_k"],
                layer_group["projection_similarity"],
                marker="o",
                label=f"L{layer}",
            )
        ax.set_title(f"Supervised Subspace Alignment: {method}")
        ax.set_xlabel("SVD top-K backbone")
        ax.set_ylabel("Projection similarity")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"stage_c_supervised_alignment_{method}.png", dpi=180)
        plt.close(fig)


def _plot_extended_k_curve(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for layer, group in df.groupby("layer"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["auroc"], marker="o", label=f"L{layer}")
    ax.set_title("Extended Top-K AUROC")
    ax.set_xlabel("K")
    ax.set_ylabel("AUROC")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_extended_topk_auroc.png", dpi=180)
    plt.close(fig)


def _plot_cumulative_exclusion(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for feature, ax in [("only_top_1_to_k", axes[0]), ("remove_top_1_to_k", axes[1])]:
        sub = df[df["feature"] == feature].copy()
        sub = sub[sub["k"] != "full"]
        if sub.empty:
            continue
        sub["k_numeric"] = sub["k"].astype(int)
        for layer, group in sub.groupby("layer"):
            group = group.sort_values("k_numeric")
            ax.plot(group["k_numeric"], group["auroc"], marker="o", label=f"L{layer}")
        ax.set_title(feature)
        ax.set_xlabel("K")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_cumulative_exclusion_auroc.png", dpi=180)
    plt.close(fig)


def _plot_band_exclusion(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for feature, ax in [("only_band", axes[0]), ("remove_band", axes[1])]:
        sub = df[df["feature"] == feature].copy()
        for layer, group in sub.groupby("layer"):
            group = group.sort_values("start")
            ax.plot(group["band"], group["auroc"], marker="o", label=f"L{layer}")
        ax.set_title(feature)
        ax.set_xlabel("Band")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_band_exclusion_auroc.png", dpi=180)
    plt.close(fig)


def _plot_coordinate_control(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    features = [
        "raw_full_difference",
        "pca_whitened_difference",
        "full_svd_coordinates",
        "random_orthogonal_rotation",
    ]
    x = np.arange(len(features))
    layers = sorted(df["layer"].unique())
    width = 0.8 / max(len(layers), 1)
    for offset, layer in enumerate(layers):
        group = df[df["layer"] == layer].set_index("feature")
        values = [group.loc[feature, "auroc"] if feature in group.index else math.nan for feature in features]
        ax.bar(x + offset * width, values, width=width, label=f"L{layer}")
    ax.set_xticks(x + width * (len(layers) - 1) / 2, labels=features, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Full-Space Coordinate Control")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "stage_c_coordinate_control_auroc.png", dpi=180)
    plt.close(fig)


def _explained_at(cumulative: np.ndarray, k: int) -> float:
    if len(cumulative) == 0:
        return math.nan
    idx = min(max(k, 1), len(cumulative)) - 1
    return float(cumulative[idx])


def _split_half_stability(
    matrix: np.ndarray,
    k: int,
    repeats: int,
    rng: np.random.Generator,
    method: str,
    desc: str,
) -> float:
    sims = []
    n = matrix.shape[0]
    for _ in tqdm(range(repeats), desc=desc, unit="split", leave=False):
        perm = rng.permutation(n)
        half = n // 2
        a = matrix[perm[:half]]
        b = matrix[perm[half : half * 2]]
        va = _top_right_singular_vectors(a, k, method, rng)
        vb = _top_right_singular_vectors(b, k, method, rng)
        sims.append(projection_similarity(va, vb))
    return float(np.mean(sims))


def _top_right_singular_vectors(
    matrix: np.ndarray,
    k: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "exact":
        return np.linalg.svd(matrix, full_matrices=False)[2].T[:, :k]
    if method == "randomized":
        _, _, vt = randomized_svd(
            matrix,
            n_components=k,
            n_iter=3,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        return vt.T[:, :k]
    raise ValueError(f"Unknown stability method: {method}")


def _random_subspace_stability(dim: int, k: int, repeats: int, rng: np.random.Generator) -> float:
    sims = []
    for _ in range(repeats):
        qa, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        qb, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        sims.append(projection_similarity(qa, qb))
    return float(np.mean(sims))


def _random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q[:, :k]


def _dense_random_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float32, copy=False)


def _load_diff_labels_basis(
    layer: int,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    labels_by_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden = load_hidden_layer(hidden_states_dir, layer)
    sample_ids = list(hidden["sample_ids"])
    keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
    y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
    z_img = hidden["z_img"][keep].float().numpy()
    z_blind = hidden["z_blind"][keep].float().numpy()
    diff = z_blind - z_img
    svd = load_svd(svd_dir, layer)
    basis = svd["Vh"].float().numpy().T
    cumulative = cumulative_explained_variance(svd["singular_values"].float().numpy())
    return diff, y, basis, cumulative


def _supervised_direction_bases(
    diff: np.ndarray,
    y: np.ndarray,
    seed: int,
    pls_components: int,
) -> dict[str, np.ndarray]:
    bases: dict[str, np.ndarray] = {}
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(diff)

    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    logistic.fit(x_scaled, y)
    logistic_w = logistic.coef_[0] / np.maximum(scaler.scale_, 1e-12)
    bases["logistic_weight"] = _orthonormal_columns(logistic_w[:, None])

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(x_scaled, y)
    lda_w = lda.coef_[0] / np.maximum(scaler.scale_, 1e-12)
    bases["lda_fisher"] = _orthonormal_columns(lda_w[:, None])

    n_components = min(pls_components, diff.shape[1], max(1, diff.shape[0] - 1))
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x_scaled, y.astype(np.float64))
    pls_weights = pls.x_weights_ / np.maximum(scaler.scale_[:, None], 1e-12)
    bases[f"pls_{n_components}"] = _orthonormal_columns(pls_weights)
    return bases


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


def _directed_projection_similarity(source_basis: np.ndarray, target_basis: np.ndarray) -> float:
    source = np.asarray(source_basis, dtype=np.float64)
    target = np.asarray(target_basis, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Subspace bases must be 2D arrays.")
    if source.shape[1] == 0:
        return 0.0
    overlap = source.T @ target
    return float(np.sum(overlap * overlap) / source.shape[1])


def _subsample_rows(
    matrix: np.ndarray,
    sample_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or matrix.shape[0] <= sample_size:
        return matrix
    indices = rng.choice(matrix.shape[0], size=sample_size, replace=False)
    return matrix[indices]


def _fp_target_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = {}
    for row in rows:
        outcome = row.get("outcome")
        if outcome == "FP":
            labels[str(row["sample_id"])] = 1
        elif outcome == "TN":
            labels[str(row["sample_id"])] = 0
    return labels


def _family_k_values(family: str, k_grid: list[int]) -> list[int | None]:
    if family in {"raw_img", "raw_blind", "difference"}:
        return [None]
    return k_grid


def _build_feature(
    family: str,
    k: int | None,
    z_img: np.ndarray,
    z_blind: np.ndarray,
    diff: np.ndarray,
    basis: np.ndarray,
    seed: int,
) -> np.ndarray:
    if family == "raw_img":
        return z_img
    if family == "raw_blind":
        return z_blind
    if family == "difference":
        return diff
    if family == "projected_difference":
        return diff @ basis[:, : int(k)]
    if family == "random_difference":
        rng = np.random.default_rng(seed + int(k))
        q, _ = np.linalg.qr(rng.normal(size=(diff.shape[1], int(k))))
        return diff @ q[:, : int(k)]
    if family == "pca_img":
        centered = z_img - z_img.mean(axis=0, keepdims=True)
        vh = np.linalg.svd(centered, full_matrices=False)[2]
        return centered @ vh[: int(k)].T
    raise ValueError(f"Unknown feature family: {family}")


def _fit_probe(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": math.nan, "auprc": math.nan, "accuracy": math.nan, "f1": math.nan}
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=stratify
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)
    probabilities = clf.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else math.nan,
        "auprc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def _fit_probe_fixed_split(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    standardize: bool,
    max_iter: int,
    c_value: float,
) -> dict[str, float]:
    return _fit_prepared_probe(
        x[train_idx],
        x[test_idx],
        y[train_idx],
        y[test_idx],
        seed,
        standardize=standardize,
        max_iter=max_iter,
        c_value=c_value,
    )


def _fit_prepared_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    standardize: bool,
    max_iter: int,
    c_value: float,
) -> dict[str, float]:
    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        C=c_value,
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
        "n_iter": int(np.max(clf.n_iter_)),
        "coef_norm": float(np.linalg.norm(clf.coef_)),
        "intercept": float(clf.intercept_[0]),
    }


def _train_pca_whiten(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x_train_raw = x[train_idx]
    x_test_raw = x[test_idx]
    mean = x_train_raw.mean(axis=0, keepdims=True)
    train_centered = x_train_raw - mean
    test_centered = x_test_raw - mean
    _, singular_values, vt = np.linalg.svd(train_centered, full_matrices=False)
    rank = int(np.sum(singular_values > eps))
    if rank == 0:
        return train_centered[:, :1] * 0.0, test_centered[:, :1] * 0.0
    components = vt[:rank].T
    scales = singular_values[:rank] / math.sqrt(max(train_centered.shape[0] - 1, 1))
    scales = np.maximum(scales, eps)
    return (train_centered @ components) / scales, (test_centered @ components) / scales


def _layer_angle_rows(layers: list[int], k_grid: list[int], svd_dir: str | Path) -> list[dict[str, Any]]:
    bases = {}
    for layer in tqdm(layers, desc="load layer bases", unit="layer"):
        svd = load_svd(svd_dir, layer)
        bases[layer] = svd["Vh"].float().numpy().T
    rows = []
    for k in tqdm(k_grid, desc="layer angle K grid", unit="K"):
        for layer_a in layers:
            for layer_b in layers:
                k_eff = min(k, bases[layer_a].shape[1], bases[layer_b].shape[1])
                rows.append(
                    {
                        "k": k,
                        "layer_a": layer_a,
                        "layer_b": layer_b,
                        "projection_similarity": projection_similarity(
                            bases[layer_a][:, :k_eff], bases[layer_b][:, :k_eff]
                        ),
                    }
                )
    return rows
