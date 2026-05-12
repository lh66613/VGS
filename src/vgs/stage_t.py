"""Stage T selective correction from correction-geometry risk scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from vgs.artifacts import load_hidden_layer, read_jsonl
from vgs.io import ensure_dir, write_csv, write_jsonl


ScoreFn = Callable[[np.ndarray], np.ndarray]
ExternalScoreFn = Callable[[np.ndarray, list[dict[str, Any]]], np.ndarray]


@dataclass(frozen=True)
class ScoreModel:
    name: str
    family: str
    scorer: ScoreFn
    train_orientation: float = 1.0
    external_scorer: ExternalScoreFn | None = None


def analyze_stage_t_selective_correction(
    layers: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    output_dir: str | Path,
    train_subset: str = "random",
    calibration_subset: str = "popular",
    test_subset: str = "adversarial",
    tail_band: tuple[int, int] = (257, 1024),
    top_k_grid: list[int] | None = None,
    pls_k: int = 32,
    random_dim: int = 64,
    trigger_rates: list[float] | None = None,
    seed: int = 13,
    max_iter: int = 2000,
    margin_scores_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    external_predictions_path: str | Path | None = None,
    external_hidden_states_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train geometry scores and evaluate split-locked selective gates.

    The primary protocol mirrors Plan_extend.md:
    POPE random trains FP-vs-TN probes/subspaces, POPE popular calibrates
    thresholds, and POPE adversarial is the held-out gate test.
    """

    top_k_grid = sorted(top_k_grid or [4, 64])
    trigger_rates = sorted(trigger_rates or [0.1, 0.2, 0.3])
    output_root = ensure_dir(output_dir)

    rows = read_jsonl(predictions_path)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    margin_by_id = _load_margin_scores(margin_scores_path)
    split_by_id = _load_protocol_split_map(split_dir)
    metadata_rows = [
        _metadata_row(row, margin_by_id.get(str(row["sample_id"]), {}), split_by_id)
        for row in rows
    ]
    verification_pool = {
        str(row["sample_id"]): _verification_sample_from_prediction(row)
        for row, meta in zip(rows, metadata_rows)
        if meta.get("subset") == test_subset
        and meta.get("parsed_prediction") == "yes"
        and meta.get("outcome") in {"FP", "TP"}
    }

    score_rows: list[dict[str, Any]] = []
    score_metric_rows: list[dict[str, Any]] = []
    gate_metric_rows: list[dict[str, Any]] = []
    random_gate_rows: list[dict[str, Any]] = []
    verification_assignments: list[dict[str, Any]] = []
    verification_samples: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []
    external_score_rows: list[dict[str, Any]] = []
    external_metric_rows: list[dict[str, Any]] = []

    for layer in layers:
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(item) for item in hidden["sample_ids"]]
        _require_sample_alignment(sample_ids, rows_by_id, hidden_states_dir, layer)
        diff = (hidden["z_blind"].float() - hidden["z_img"].float()).numpy()
        meta_by_hidden = [rows_by_id[sample_id] for sample_id in sample_ids]
        layer_metadata = [
            _metadata_row(row, margin_by_id.get(str(row["sample_id"]), {}), split_by_id)
            for row in meta_by_hidden
        ]
        train_idx = _subset_indices(layer_metadata, train_subset)
        train_label_idx, y_train = _label_indices(layer_metadata, train_subset, {"FP": 1, "TN": 0})
        if len(train_label_idx) == 0 or len(np.unique(y_train)) < 2:
            raise ValueError(f"Layer {layer} has insufficient FP/TN train labels in {train_subset}.")

        score_models, layer_train_rows = _fit_score_models(
            diff=diff,
            layer_metadata=layer_metadata,
            train_idx=train_idx,
            train_label_idx=train_label_idx,
            y_train=y_train,
            tail_band=tail_band,
            top_k_grid=top_k_grid,
            pls_k=pls_k,
            random_dim=random_dim,
            seed=seed + layer,
            max_iter=max_iter,
            margin_available=bool(margin_by_id),
        )
        threshold_rows.extend(layer_train_rows)
        layer_scores = {name: model.scorer(diff) for name, model in score_models.items()}

        for local_idx, sample_id in enumerate(sample_ids):
            row = {
                "layer": layer,
                "sample_id": sample_id,
                **layer_metadata[local_idx],
            }
            for name, values in layer_scores.items():
                row[name] = float(values[local_idx])
            score_rows.append(row)

        for score_name, values in layer_scores.items():
            score_metric_rows.extend(
                _score_metric_rows(
                    layer=layer,
                    score_name=score_name,
                    scores=values,
                    metadata=layer_metadata,
                    train_subset=train_subset,
                    calibration_subset=calibration_subset,
                    test_subset=test_subset,
                    trigger_rates=trigger_rates,
                )
            )

        for score_name, values in layer_scores.items():
            for target_rate in trigger_rates:
                threshold, calib_info = _calibrate_threshold(
                    scores=values,
                    metadata=layer_metadata,
                    subset=calibration_subset,
                    target_rate=target_rate,
                )
                gate_mask = _gate_mask(values, layer_metadata, test_subset, threshold)
                gate_row, projected_outcomes = _gate_metric_row(
                    layer=layer,
                    score_name=score_name,
                    gate_name="geometry_gate",
                    target_rate=target_rate,
                    threshold=threshold,
                    calib_info=calib_info,
                    metadata=layer_metadata,
                    subset=test_subset,
                    gate_mask=gate_mask,
                )
                gate_metric_rows.append(gate_row)
                random_gate_rows.extend(
                    _random_gate_rows(
                        layer=layer,
                        score_name=score_name,
                        target_rate=target_rate,
                        metadata=layer_metadata,
                        subset=test_subset,
                        n_trigger=int(gate_row["trigger_n"]),
                        seed=seed + layer + int(round(target_rate * 1000)),
                        repeats=200,
                    )
                )
                verification_assignments.extend(
                    _verification_assignment_rows(
                        layer=layer,
                        score_name=score_name,
                        target_rate=target_rate,
                        threshold=threshold,
                        metadata=layer_metadata,
                        sample_ids=sample_ids,
                        gate_mask=gate_mask,
                    )
                )
                _collect_verification_samples(
                    verification_samples,
                    metadata=layer_metadata,
                    sample_ids=sample_ids,
                    gate_mask=gate_mask,
                )

        if external_predictions_path and external_hidden_states_dir:
            external_rows = read_jsonl(external_predictions_path)
            external_by_id = {str(row["sample_id"]): row for row in external_rows}
            external_hidden = load_hidden_layer(external_hidden_states_dir, layer)
            external_ids = [str(item) for item in external_hidden["sample_ids"]]
            _require_sample_alignment(external_ids, external_by_id, external_hidden_states_dir, layer)
            external_diff = (
                external_hidden["z_blind"].float() - external_hidden["z_img"].float()
            ).numpy()
            external_metadata = [
                _metadata_row(external_by_id[sample_id], margin_by_id.get(sample_id, {}), {})
                for sample_id in external_ids
            ]
            external_scores = {
                name: _score_external_model(model, external_diff, external_metadata)
                for name, model in score_models.items()
            }
            for local_idx, sample_id in enumerate(external_ids):
                row = {
                    "layer": layer,
                    "sample_id": sample_id,
                    **external_metadata[local_idx],
                }
                for name, values in external_scores.items():
                    row[name] = float(values[local_idx])
                external_score_rows.append(row)
            for score_name, values in external_scores.items():
                external_metric_rows.extend(
                    _external_score_metric_rows(
                        layer=layer,
                        score_name=score_name,
                        scores=values,
                        metadata=external_metadata,
                    )
                )

    scores_path = write_csv(
        output_root / "stage_t_scores.csv",
        score_rows,
        _fieldnames(score_rows),
    )
    score_metrics_path = write_csv(
        output_root / "stage_t_score_metrics.csv",
        score_metric_rows,
        _fieldnames(score_metric_rows),
    )
    gate_metrics_path = write_csv(
        output_root / "stage_t_gate_metrics.csv",
        gate_metric_rows,
        _fieldnames(gate_metric_rows),
    )
    random_gate_path = write_csv(
        output_root / "stage_t_random_gate_metrics.csv",
        random_gate_rows,
        _fieldnames(random_gate_rows),
    )
    thresholds_path = write_csv(
        output_root / "stage_t_score_training_audit.csv",
        threshold_rows,
        _fieldnames(threshold_rows),
    )
    assignment_path = write_csv(
        output_root / "stage_t_verification_gate_assignments.csv",
        verification_assignments,
        _fieldnames(verification_assignments),
    )
    samples_path = write_jsonl(
        output_root / "stage_t_verification_samples.jsonl",
        sorted(verification_samples.values(), key=lambda row: row["sample_id"]),
    )
    pool_path = write_jsonl(
        output_root / "stage_t_verification_pool.jsonl",
        sorted(verification_pool.values(), key=lambda row: row["sample_id"]),
    )

    external_scores_path = ""
    external_metrics_path = ""
    if external_score_rows:
        external_scores_path = str(
            write_csv(
                output_root / "stage_t_external_scores.csv",
                external_score_rows,
                _fieldnames(external_score_rows),
            )
        )
        external_metrics_path = str(
            write_csv(
                output_root / "stage_t_external_score_metrics.csv",
                external_metric_rows,
                _fieldnames(external_metric_rows),
            )
        )

    note_path = _write_summary_note(
        output_root / "stage_t_selective_correction_summary.md",
        gate_metric_rows,
        random_gate_rows,
        score_metric_rows,
        train_subset,
        calibration_subset,
        test_subset,
        bool(margin_by_id),
    )

    return {
        "scores_path": str(scores_path),
        "score_metrics_path": str(score_metrics_path),
        "gate_metrics_path": str(gate_metrics_path),
        "random_gate_metrics_path": str(random_gate_path),
        "score_training_audit_path": str(thresholds_path),
        "verification_gate_assignments_path": str(assignment_path),
        "verification_samples_path": str(samples_path),
        "verification_pool_path": str(pool_path),
        "summary_note_path": str(note_path),
        "external_scores_path": external_scores_path,
        "external_metrics_path": external_metrics_path,
        "num_score_rows": len(score_rows),
        "num_score_metric_rows": len(score_metric_rows),
        "num_gate_metric_rows": len(gate_metric_rows),
        "num_random_gate_rows": len(random_gate_rows),
        "num_verification_samples": len(verification_samples),
        "num_verification_pool_samples": len(verification_pool),
        "layers": layers,
        "train_subset": train_subset,
        "calibration_subset": calibration_subset,
        "test_subset": test_subset,
        "tail_band": f"{tail_band[0]}-{tail_band[1]}",
        "top_k_grid": top_k_grid,
        "trigger_rates": trigger_rates,
        "margin_scores_available": bool(margin_by_id),
        "split_dir": str(split_dir) if split_dir else "",
    }


def _fit_score_models(
    diff: np.ndarray,
    layer_metadata: list[dict[str, Any]],
    train_idx: np.ndarray,
    train_label_idx: np.ndarray,
    y_train: np.ndarray,
    tail_band: tuple[int, int],
    top_k_grid: list[int],
    pls_k: int,
    random_dim: int,
    seed: int,
    max_iter: int,
    margin_available: bool,
) -> tuple[dict[str, ScoreModel], list[dict[str, Any]]]:
    models: dict[str, ScoreModel] = {}
    audit_rows: list[dict[str, Any]] = []

    full_model, full_audit = _fit_logistic_model(
        name="full_probe",
        family="full_difference",
        train_x=diff[train_label_idx],
        y_train=y_train,
        seed=seed,
        max_iter=max_iter,
    )
    models[full_model.name] = full_model
    audit_rows.append(full_audit)

    max_svd_k = min(
        max([tail_band[1], *top_k_grid]),
        diff.shape[1],
        max(1, diff[train_idx].shape[0] - 1),
    )
    _, _, vt = randomized_svd(
        diff[train_idx],
        n_components=max_svd_k,
        n_iter=4,
        random_state=seed,
    )
    basis = vt.T.astype(np.float32, copy=False)
    for k in top_k_grid:
        k_eff = min(k, basis.shape[1])
        if k_eff <= 0:
            continue
        model, audit = _fit_projected_probe(
            name=f"top_{k}_probe",
            family="top_svd_probe",
            basis=basis[:, :k_eff],
            diff=diff,
            train_label_idx=train_label_idx,
            y_train=y_train,
            seed=seed + k,
            max_iter=max_iter,
        )
        audit["effective_dim"] = k_eff
        models[model.name] = model
        audit_rows.append(audit)

    tail_start, tail_end = tail_band
    tail_start_idx = max(0, tail_start - 1)
    tail_end_idx = min(tail_end, basis.shape[1])
    if tail_end_idx > tail_start_idx:
        tail_basis = basis[:, tail_start_idx:tail_end_idx]
        tail_name = f"tail_{tail_start}_{tail_end}_probe"
        model, audit = _fit_projected_probe(
            name=tail_name,
            family="tail_svd_probe",
            basis=tail_basis,
            diff=diff,
            train_label_idx=train_label_idx,
            y_train=y_train,
            seed=seed + tail_end_idx,
            max_iter=max_iter,
        )
        audit["effective_dim"] = int(tail_basis.shape[1])
        models[model.name] = model
        audit_rows.append(audit)

        energy_train = np.sum((diff[train_label_idx] @ tail_basis) ** 2, axis=1)
        sign = _orientation_sign(energy_train, y_train)
        models[f"tail_{tail_start}_{tail_end}_energy"] = ScoreModel(
            name=f"tail_{tail_start}_{tail_end}_energy",
            family="tail_svd_energy",
            scorer=lambda x, b=tail_basis, s=sign: s * np.sum((x @ b) ** 2, axis=1),
            train_orientation=sign,
        )
        audit_rows.append(
            {
                "score_name": f"tail_{tail_start}_{tail_end}_energy",
                "family": "tail_svd_energy",
                "train_n": int(len(y_train)),
                "train_positive_n": int(np.sum(y_train == 1)),
                "train_negative_n": int(np.sum(y_train == 0)),
                "effective_dim": int(tail_basis.shape[1]),
                "train_orientation": sign,
                "train_auroc": _safe_metric(y_train, sign * energy_train, roc_auc_score),
                "train_auprc": _safe_metric(y_train, sign * energy_train, average_precision_score),
            }
        )

    pls_basis = _pls_basis(diff[train_label_idx], y_train, pls_k)
    if pls_basis.shape[1] > 0:
        model, audit = _fit_projected_probe(
            name=f"pls{pls_basis.shape[1]}_probe",
            family="pls_fp_tn_probe",
            basis=pls_basis,
            diff=diff,
            train_label_idx=train_label_idx,
            y_train=y_train,
            seed=seed + 17,
            max_iter=max_iter,
        )
        audit["effective_dim"] = int(pls_basis.shape[1])
        models[model.name] = model
        audit_rows.append(audit)

    random_basis = _random_orthonormal_basis(diff.shape[1], min(random_dim, diff.shape[1]), seed)
    model, audit = _fit_projected_probe(
        name=f"random{random_basis.shape[1]}_probe",
        family="random_subspace_probe",
        basis=random_basis,
        diff=diff,
        train_label_idx=train_label_idx,
        y_train=y_train,
        seed=seed + 29,
        max_iter=max_iter,
    )
    audit["effective_dim"] = int(random_basis.shape[1])
    models[model.name] = model
    audit_rows.append(audit)

    if margin_available:
        margin_train = np.array([layer_metadata[idx]["yes_minus_no_logit"] for idx in train_label_idx], dtype=float)
        if not np.isnan(margin_train).all():
            sign = _orientation_sign(margin_train, y_train)
            geometry_names = [
                name
                for name, model in models.items()
                if model.family not in {"output_margin", "margin_plus_geometry", "low_output_margin", "low_margin_plus_geometry"}
            ]
            models["margin_probe"] = ScoreModel(
                name="margin_probe",
                family="output_margin",
                scorer=lambda _x, meta=layer_metadata, s=sign: s
                * np.array([row["yes_minus_no_logit"] for row in meta], dtype=float),
                train_orientation=sign,
                external_scorer=lambda _x, meta, s=sign: s
                * np.array([row["yes_minus_no_logit"] for row in meta], dtype=float),
            )
            audit_rows.append(
                {
                    "score_name": "margin_probe",
                    "family": "output_margin",
                    "train_n": int(len(y_train)),
                    "train_positive_n": int(np.sum(y_train == 1)),
                    "train_negative_n": int(np.sum(y_train == 0)),
                    "effective_dim": 1,
                    "train_orientation": sign,
                    "train_auroc": _safe_metric(y_train, sign * margin_train, roc_auc_score),
                    "train_auprc": _safe_metric(y_train, sign * margin_train, average_precision_score),
                }
            )
            models["low_margin_probe"] = ScoreModel(
                name="low_margin_probe",
                family="low_output_margin",
                scorer=lambda _x, meta=layer_metadata: -np.array(
                    [row["yes_minus_no_logit"] for row in meta],
                    dtype=float,
                ),
                train_orientation=-1.0,
                external_scorer=lambda _x, meta: -np.array(
                    [row["yes_minus_no_logit"] for row in meta],
                    dtype=float,
                ),
            )
            audit_rows.append(
                {
                    "score_name": "low_margin_probe",
                    "family": "low_output_margin",
                    "train_n": int(len(y_train)),
                    "train_positive_n": int(np.sum(y_train == 1)),
                    "train_negative_n": int(np.sum(y_train == 0)),
                    "effective_dim": 1,
                    "train_orientation": -1.0,
                    "train_auroc": _safe_metric(y_train, -margin_train, roc_auc_score),
                    "train_auprc": _safe_metric(y_train, -margin_train, average_precision_score),
                }
            )
            for geo_name in geometry_names:
                geo_scores_train = models[geo_name].scorer(diff[train_label_idx])
                margin_scores_train = sign * margin_train
                geo_center, geo_scale = _center_scale(geo_scores_train)
                margin_center, margin_scale = _center_scale(margin_scores_train)

                def scorer(
                    x: np.ndarray,
                    geo=models[geo_name],
                    meta=layer_metadata,
                    margin_sign=sign,
                    gc=geo_center,
                    gs=geo_scale,
                    mc=margin_center,
                    ms=margin_scale,
                ) -> np.ndarray:
                    return _margin_plus_geometry_scores(x, meta, geo, margin_sign, gc, gs, mc, ms)

                def external_scorer(
                    x: np.ndarray,
                    meta: list[dict[str, Any]],
                    geo=models[geo_name],
                    margin_sign=sign,
                    gc=geo_center,
                    gs=geo_scale,
                    mc=margin_center,
                    ms=margin_scale,
                ) -> np.ndarray:
                    return _margin_plus_geometry_scores(x, meta, geo, margin_sign, gc, gs, mc, ms)

                models[f"margin_plus_{geo_name}"] = ScoreModel(
                    name=f"margin_plus_{geo_name}",
                    family="margin_plus_geometry",
                    scorer=scorer,
                    external_scorer=external_scorer,
                )
                low_margin_scores_train = -margin_train
                low_margin_center, low_margin_scale = _center_scale(low_margin_scores_train)

                def low_scorer(
                    x: np.ndarray,
                    geo=models[geo_name],
                    meta=layer_metadata,
                    gc=geo_center,
                    gs=geo_scale,
                    mc=low_margin_center,
                    ms=low_margin_scale,
                ) -> np.ndarray:
                    return _margin_plus_geometry_scores(x, meta, geo, -1.0, gc, gs, mc, ms)

                def external_low_scorer(
                    x: np.ndarray,
                    meta: list[dict[str, Any]],
                    geo=models[geo_name],
                    gc=geo_center,
                    gs=geo_scale,
                    mc=low_margin_center,
                    ms=low_margin_scale,
                ) -> np.ndarray:
                    return _margin_plus_geometry_scores(x, meta, geo, -1.0, gc, gs, mc, ms)

                models[f"low_margin_plus_{geo_name}"] = ScoreModel(
                    name=f"low_margin_plus_{geo_name}",
                    family="low_margin_plus_geometry",
                    scorer=low_scorer,
                    external_scorer=external_low_scorer,
                )

    return models, audit_rows


def _score_external_model(
    model: ScoreModel,
    diff: np.ndarray,
    metadata: list[dict[str, Any]],
) -> np.ndarray:
    if model.external_scorer is not None:
        return model.external_scorer(diff, metadata)
    return model.scorer(diff)


def _margin_plus_geometry_scores(
    x: np.ndarray,
    metadata: list[dict[str, Any]],
    geo: ScoreModel,
    margin_sign: float,
    geo_center: float,
    geo_scale: float,
    margin_center: float,
    margin_scale: float,
) -> np.ndarray:
    geo_z = (geo.scorer(x) - geo_center) / geo_scale
    margin_z = (
        margin_sign
        * np.array([row["yes_minus_no_logit"] for row in metadata], dtype=float)
        - margin_center
    ) / margin_scale
    return geo_z + margin_z


def _fit_logistic_model(
    name: str,
    family: str,
    train_x: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_iter: int,
) -> tuple[ScoreModel, dict[str, Any]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    train_scores = clf.predict_proba(x_train)[:, 1]

    def scorer(x: np.ndarray, fitted_scaler: StandardScaler = scaler, fitted_clf: LogisticRegression = clf) -> np.ndarray:
        return fitted_clf.predict_proba(fitted_scaler.transform(x))[:, 1]

    return (
        ScoreModel(name=name, family=family, scorer=scorer),
        {
            "score_name": name,
            "family": family,
            "train_n": int(len(y_train)),
            "train_positive_n": int(np.sum(y_train == 1)),
            "train_negative_n": int(np.sum(y_train == 0)),
            "effective_dim": int(train_x.shape[1]),
            "train_orientation": 1.0,
            "train_auroc": _safe_metric(y_train, train_scores, roc_auc_score),
            "train_auprc": _safe_metric(y_train, train_scores, average_precision_score),
        },
    )


def _fit_projected_probe(
    name: str,
    family: str,
    basis: np.ndarray,
    diff: np.ndarray,
    train_label_idx: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_iter: int,
) -> tuple[ScoreModel, dict[str, Any]]:
    train_x = diff[train_label_idx] @ basis
    model, audit = _fit_logistic_model(name, family, train_x, y_train, seed, max_iter)

    def scorer(x: np.ndarray, fitted_model: ScoreModel = model, fitted_basis: np.ndarray = basis) -> np.ndarray:
        return fitted_model.scorer(x @ fitted_basis)

    return ScoreModel(name=name, family=family, scorer=scorer), audit


def _score_metric_rows(
    layer: int,
    score_name: str,
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
    trigger_rates: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in [train_subset, calibration_subset, test_subset]:
        split_label = _split_role(split_name, train_subset, calibration_subset, test_subset)
        rows.append(
            _population_score_metrics(
                layer,
                score_name,
                split_label,
                split_name,
                "fp_vs_tn",
                scores,
                metadata,
                lambda row: row["subset"] == split_name and row["outcome"] in {"FP", "TN"},
                lambda row: int(row["outcome"] == "FP"),
                trigger_rates,
            )
        )
        rows.append(
            _population_score_metrics(
                layer,
                score_name,
                split_label,
                split_name,
                "predicted_yes_fp_vs_tp",
                scores,
                metadata,
                lambda row: row["subset"] == split_name and row["outcome"] in {"FP", "TP"},
                lambda row: int(row["outcome"] == "FP"),
                trigger_rates,
            )
        )
    return rows


def _external_score_metric_rows(
    layer: int,
    score_name: str,
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = sorted({row.get("dimension", "") for row in metadata})
    for dimension in ["ALL", *dimensions]:
        rows.append(
            _population_score_metrics(
                layer,
                score_name,
                "external",
                dimension,
                "fp_vs_tn",
                scores,
                metadata,
                lambda row, dim=dimension: (dim == "ALL" or row.get("dimension", "") == dim)
                and row["outcome"] in {"FP", "TN"},
                lambda row: int(row["outcome"] == "FP"),
                [0.1, 0.2, 0.3],
            )
        )
        rows.append(
            _population_score_metrics(
                layer,
                score_name,
                "external",
                dimension,
                "predicted_yes_fp_vs_tp",
                scores,
                metadata,
                lambda row, dim=dimension: (dim == "ALL" or row.get("dimension", "") == dim)
                and row["outcome"] in {"FP", "TP"},
                lambda row: int(row["outcome"] == "FP"),
                [0.1, 0.2, 0.3],
            )
        )
    return rows


def _population_score_metrics(
    layer: int,
    score_name: str,
    split_role: str,
    split_name: str,
    population: str,
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
    keep_fn: Callable[[dict[str, Any]], bool],
    label_fn: Callable[[dict[str, Any]], int],
    trigger_rates: list[float],
) -> dict[str, Any]:
    keep = np.array([bool(keep_fn(row)) for row in metadata])
    y = np.array([label_fn(row) for row, is_kept in zip(metadata, keep) if is_kept], dtype=np.int64)
    values = scores[keep]
    result: dict[str, Any] = {
        "layer": layer,
        "score": score_name,
        "split_role": split_role,
        "split": split_name,
        "population": population,
        "n": int(len(y)),
        "positive_n": int(np.sum(y == 1)) if len(y) else 0,
        "negative_n": int(np.sum(y == 0)) if len(y) else 0,
        "positive_rate": float(np.mean(y)) if len(y) else math.nan,
        "auroc": _safe_metric(y, values, roc_auc_score),
        "auprc": _safe_metric(y, values, average_precision_score),
        "risk_coverage_auc": _risk_coverage_auc(y, values),
        "ece_10": _ece_10(y, values),
    }
    for rate in trigger_rates:
        capture, damage, precision = _capture_damage_at_rate(y, values, rate)
        suffix = f"{int(round(rate * 100))}"
        result[f"positive_recall_at_{suffix}pct"] = capture
        result[f"negative_trigger_at_{suffix}pct"] = damage
        result[f"precision_at_{suffix}pct"] = precision
    return result


def _calibrate_threshold(
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
    subset: str,
    target_rate: float,
) -> tuple[float, dict[str, Any]]:
    mask = _predicted_yes_mask(metadata, subset)
    values = scores[mask]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, {"calibration_predicted_yes_n": 0, "calibration_trigger_n": 0}
    n_trigger = max(1, int(math.ceil(target_rate * len(values))))
    n_trigger = min(n_trigger, len(values))
    threshold = float(np.sort(values)[-n_trigger])
    triggered = values >= threshold
    return threshold, {
        "calibration_predicted_yes_n": int(len(values)),
        "calibration_trigger_n": int(np.sum(triggered)),
        "calibration_trigger_rate_predicted_yes": float(np.mean(triggered)),
    }


def _gate_mask(
    scores: np.ndarray,
    metadata: list[dict[str, Any]],
    subset: str,
    threshold: float,
) -> np.ndarray:
    if math.isnan(threshold):
        return np.zeros(len(metadata), dtype=bool)
    return _predicted_yes_mask(metadata, subset) & (scores >= threshold)


def _gate_metric_row(
    layer: int,
    score_name: str,
    gate_name: str,
    target_rate: float,
    threshold: float,
    calib_info: dict[str, Any],
    metadata: list[dict[str, Any]],
    subset: str,
    gate_mask: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    split_mask = np.array([row["subset"] == subset for row in metadata])
    pred_yes_mask = _predicted_yes_mask(metadata, subset)
    pred_yes_outcomes = [row["outcome"] for row, keep in zip(metadata, pred_yes_mask) if keep]
    split_outcomes = [row["outcome"] for row, keep in zip(metadata, split_mask) if keep]
    triggered_outcomes = [row["outcome"] for row, keep in zip(metadata, gate_mask) if keep]
    projected = [
        _projected_outcome(row["outcome"], bool(trigger))
        for row, in_split, trigger in zip(metadata, split_mask, gate_mask)
        if in_split
    ]
    quality = _quality_metrics(split_outcomes)
    projected_quality = _quality_metrics(projected)
    pred_yes_counts = _outcome_counts(pred_yes_outcomes)
    trigger_counts = _outcome_counts(triggered_outcomes)
    fp_total = pred_yes_counts["FP"]
    tp_total = pred_yes_counts["TP"]
    trigger_n = len(triggered_outcomes)
    row = {
        "layer": layer,
        "gate": gate_name,
        "score": score_name,
        "split": subset,
        "target_trigger_rate_predicted_yes": target_rate,
        "threshold": threshold,
        **calib_info,
        "split_n": int(np.sum(split_mask)),
        "predicted_yes_n": int(np.sum(pred_yes_mask)),
        "trigger_n": trigger_n,
        "trigger_rate_all": float(trigger_n / max(1, np.sum(split_mask))),
        "trigger_rate_predicted_yes": float(trigger_n / max(1, np.sum(pred_yes_mask))),
        "triggered_fp": trigger_counts["FP"],
        "triggered_tp": trigger_counts["TP"],
        "triggered_fp_ratio": float(trigger_counts["FP"] / trigger_n) if trigger_n else math.nan,
        "fp_recall_among_predicted_yes": float(trigger_counts["FP"] / fp_total) if fp_total else math.nan,
        "tp_damage": float(trigger_counts["TP"] / tp_total) if tp_total else math.nan,
        "original_accuracy": quality["accuracy"],
        "original_f1": quality["f1"],
        "original_fp_rate": quality["fp_rate"],
        "oracle_flip_accuracy": projected_quality["accuracy"],
        "oracle_flip_f1": projected_quality["f1"],
        "oracle_flip_fp_rate": projected_quality["fp_rate"],
        "oracle_fp_reduction": (
            float((quality["FP"] - projected_quality["FP"]) / quality["FP"])
            if quality["FP"]
            else math.nan
        ),
        "oracle_tp_preserved": (
            float(projected_quality["TP"] / quality["TP"]) if quality["TP"] else math.nan
        ),
        "fp_reduction_per_trigger": float(trigger_counts["FP"] / trigger_n) if trigger_n else math.nan,
    }
    return row, projected


def _random_gate_rows(
    layer: int,
    score_name: str,
    target_rate: float,
    metadata: list[dict[str, Any]],
    subset: str,
    n_trigger: int,
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    pred_yes_indices = np.flatnonzero(_predicted_yes_mask(metadata, subset))
    split_mask = np.array([row["subset"] == subset for row in metadata])
    split_outcomes = [row["outcome"] for row, keep in zip(metadata, split_mask) if keep]
    quality = _quality_metrics(split_outcomes)
    if n_trigger <= 0 or len(pred_yes_indices) == 0:
        return []
    n_trigger = min(n_trigger, len(pred_yes_indices))
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, float]] = []
    for _ in range(repeats):
        chosen = set(rng.choice(pred_yes_indices, size=n_trigger, replace=False).tolist())
        gate_mask = np.array([idx in chosen for idx in range(len(metadata))])
        triggered = [row["outcome"] for row, keep in zip(metadata, gate_mask) if keep]
        trigger_counts = _outcome_counts(triggered)
        projected = [
            _projected_outcome(row["outcome"], bool(trigger))
            for row, in_split, trigger in zip(metadata, split_mask, gate_mask)
            if in_split
        ]
        projected_quality = _quality_metrics(projected)
        pred_yes_outcomes = [
            row["outcome"] for row, keep in zip(metadata, _predicted_yes_mask(metadata, subset)) if keep
        ]
        pred_yes_counts = _outcome_counts(pred_yes_outcomes)
        metric_rows.append(
            {
                "triggered_fp_ratio": trigger_counts["FP"] / n_trigger,
                "fp_recall_among_predicted_yes": trigger_counts["FP"] / max(1, pred_yes_counts["FP"]),
                "tp_damage": trigger_counts["TP"] / max(1, pred_yes_counts["TP"]),
                "oracle_flip_accuracy": projected_quality["accuracy"],
                "oracle_flip_f1": projected_quality["f1"],
                "oracle_fp_reduction": (
                    (quality["FP"] - projected_quality["FP"]) / quality["FP"]
                    if quality["FP"]
                    else math.nan
                ),
            }
        )
    rows: list[dict[str, Any]] = []
    for metric in metric_rows[0]:
        values = np.array([row[metric] for row in metric_rows], dtype=float)
        rows.append(
            {
                "layer": layer,
                "baseline": "same_trigger_count_random_gate",
                "matched_score": score_name,
                "split": subset,
                "target_trigger_rate_predicted_yes": target_rate,
                "n_trigger": n_trigger,
                "repeats": repeats,
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "p05": float(np.nanpercentile(values, 5)),
                "p95": float(np.nanpercentile(values, 95)),
            }
        )
    return rows


def _verification_assignment_rows(
    layer: int,
    score_name: str,
    target_rate: float,
    threshold: float,
    metadata: list[dict[str, Any]],
    sample_ids: list[str],
    gate_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for sample_id, row, keep in zip(sample_ids, metadata, gate_mask):
        if not keep:
            continue
        rows.append(
            {
                "layer": layer,
                "score": score_name,
                "target_trigger_rate_predicted_yes": target_rate,
                "threshold": threshold,
                "sample_id": sample_id,
                "subset": row["subset"],
                "outcome": row["outcome"],
                "label": row["label"],
                "parsed_prediction": row["parsed_prediction"],
            }
        )
    return rows


def _collect_verification_samples(
    out: dict[str, dict[str, Any]],
    metadata: list[dict[str, Any]],
    sample_ids: list[str],
    gate_mask: np.ndarray,
) -> None:
    for sample_id, row, keep in zip(sample_ids, metadata, gate_mask):
        if not keep or sample_id in out:
            continue
        out[sample_id] = {
            "sample_id": sample_id,
            "subset": row["subset"],
            "image": row.get("image", ""),
            "image_path": row.get("image_path", ""),
            "question": row["question"],
            "verification_question": _verification_question(row["question"]),
            "label": row["label"],
            "original_parsed_prediction": row["parsed_prediction"],
            "original_outcome": row["outcome"],
        }


def _verification_sample_from_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "subset": str(row.get("subset", "")),
        "image": str(row.get("image", "")),
        "image_path": str(row.get("image_path", "")),
        "question": str(row.get("question", "")),
        "verification_question": _verification_question(str(row.get("question", ""))),
        "label": str(row.get("label", "")),
        "original_parsed_prediction": str(row.get("parsed_prediction", "")),
        "original_outcome": str(row.get("outcome", "")),
    }


def _verification_question(question: str) -> str:
    return (
        "Please verify the visual evidence before answering. "
        "Answer Yes only if the queried object is clearly visible in the image. "
        "If the visual evidence is insufficient or unclear, answer No. "
        f"Question: {question}"
    )


def _metadata_row(
    row: dict[str, Any],
    margin: dict[str, Any],
    split_by_id: dict[str, str],
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", ""))
    source_subset = str(row.get("subset", ""))
    return {
        "subset": split_by_id.get(sample_id, source_subset),
        "source_subset": source_subset,
        "dimension": str(row.get("dimension", "")),
        "label": str(row.get("label", "")),
        "outcome": str(row.get("outcome", "")),
        "parsed_prediction": str(row.get("parsed_prediction", "")),
        "question": str(row.get("question", "")),
        "image": str(row.get("image", "")),
        "image_path": str(row.get("image_path", "")),
        "yes_minus_no_logit": _maybe_float(margin.get("yes_minus_no_logit")),
        "binary_entropy": _maybe_float(margin.get("binary_entropy")),
    }


def _load_margin_scores(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        default_path = Path("outputs/margins/pope_margin_scores.csv")
        path = default_path if default_path.exists() else None
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        return {}
    return {str(row.sample_id): row._asdict() for row in df.itertuples(index=False)}


def _load_protocol_split_map(split_dir: str | Path | None) -> dict[str, str]:
    if split_dir is None:
        return {}
    root = Path(split_dir)
    mapping: dict[str, str] = {}
    split_names = {
        "train": "train",
        "val": "calibration",
        "test": "test",
    }
    for filename_split, stage_split in split_names.items():
        path = root / f"pope_{filename_split}_ids.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        payload = torch.load(path, map_location="cpu") if path.suffix == ".pt" else None
        if payload is None:
            import json

            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        for sample_id in payload.get("sample_ids", []):
            mapping[str(sample_id)] = stage_split
    return mapping


def _require_sample_alignment(
    sample_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    hidden_states_dir: str | Path,
    layer: int,
) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{hidden_states_dir}/layer_{layer}.pt has unknown sample_ids: {preview}")


def _subset_indices(metadata: list[dict[str, Any]], subset: str) -> np.ndarray:
    return np.array([idx for idx, row in enumerate(metadata) if row["subset"] == subset], dtype=np.int64)


def _label_indices(
    metadata: list[dict[str, Any]],
    subset: str,
    labels: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    y = []
    for idx, row in enumerate(metadata):
        if row["subset"] != subset or row["outcome"] not in labels:
            continue
        indices.append(idx)
        y.append(labels[row["outcome"]])
    return np.array(indices, dtype=np.int64), np.array(y, dtype=np.int64)


def _predicted_yes_mask(metadata: list[dict[str, Any]], subset: str) -> np.ndarray:
    return np.array(
        [
            row["subset"] == subset
            and row["parsed_prediction"] == "yes"
            and row["outcome"] in {"FP", "TP"}
            for row in metadata
        ],
        dtype=bool,
    )


def _projected_outcome(outcome: str, trigger: bool) -> str:
    if not trigger:
        return outcome
    if outcome == "FP":
        return "TN"
    if outcome == "TP":
        return "FN"
    return outcome


def _quality_metrics(outcomes: list[str]) -> dict[str, Any]:
    counts = _outcome_counts(outcomes)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    known = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        **counts,
        "n": known,
        "accuracy": (tp + tn) / known if known else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp / (fp + tn) if fp + tn else math.nan,
        "tn_preserved": tn / (tn + fp) if tn + fp else math.nan,
        "tp_preserved": recall,
    }


def _outcome_counts(outcomes: list[str]) -> dict[str, int]:
    return {name: int(sum(outcome == name for outcome in outcomes)) for name in ["TP", "TN", "FP", "FN"]}


def _capture_damage_at_rate(
    y: np.ndarray,
    scores: np.ndarray,
    rate: float,
) -> tuple[float, float, float]:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0:
        return math.nan, math.nan, math.nan
    n_trigger = max(1, int(math.ceil(rate * len(y))))
    order = np.argsort(scores)[::-1]
    chosen = order[:n_trigger]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    triggered_pos = int(np.sum(y[chosen] == 1))
    triggered_neg = int(np.sum(y[chosen] == 0))
    return (
        triggered_pos / positives if positives else math.nan,
        triggered_neg / negatives if negatives else math.nan,
        triggered_pos / n_trigger if n_trigger else math.nan,
    )


def _risk_coverage_auc(y: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or int(np.sum(y == 1)) == 0:
        return math.nan
    order = np.argsort(scores)[::-1]
    sorted_y = y[order]
    coverage = np.arange(1, len(y) + 1, dtype=float) / len(y)
    positive_recall = np.cumsum(sorted_y == 1) / np.sum(y == 1)
    return float(np.trapz(positive_recall, coverage))


def _ece_10(y: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0:
        return math.nan
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            keep = (scores >= lo) & (scores <= hi)
        else:
            keep = (scores >= lo) & (scores < hi)
        if not keep.any():
            continue
        ece += float(np.mean(keep) * abs(np.mean(scores[keep]) - np.mean(y[keep])))
    return ece


def _orientation_sign(values: np.ndarray, y: np.ndarray) -> float:
    pos = values[y == 1]
    neg = values[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    return 1.0 if float(np.nanmean(pos)) >= float(np.nanmean(neg)) else -1.0


def _pls_basis(matrix: np.ndarray, y: np.ndarray, max_k: int) -> np.ndarray:
    n_components = min(max_k, matrix.shape[1], max(1, matrix.shape[0] - 1))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x_scaled, y.astype(np.float64))
    basis = pls.x_weights_ / np.maximum(scaler.scale_[:, None], 1e-12)
    return _orthonormal_columns(basis)


def _random_orthonormal_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, k))
    return _orthonormal_columns(matrix).astype(np.float32, copy=False)


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float32, copy=False)


def _safe_metric(y: np.ndarray, scores: np.ndarray, metric: Callable[[Any, Any], float]) -> float:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return math.nan
    return float(metric(y, scores))


def _center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    return center, max(scale, 1e-12)


def _split_role(
    split: str,
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
) -> str:
    if split == train_subset:
        return "probe_train"
    if split == calibration_subset:
        return "calibration"
    if split == test_subset:
        return "heldout_test"
    return split


def _maybe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_summary_note(
    path: Path,
    gate_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
    margin_available: bool,
) -> Path:
    ensure_dir(path.parent)
    gate_df = pd.DataFrame(gate_rows)
    random_df = pd.DataFrame(random_rows)
    score_df = pd.DataFrame(score_rows)
    lines = [
        "# Stage T Selective Correction Summary",
        "",
        "## Protocol",
        "",
        f"- Probe/SVD/PLS train split: POPE `{train_subset}`.",
        f"- Gate threshold calibration split: POPE `{calibration_subset}`.",
        f"- Held-out test split: POPE `{test_subset}`.",
        "- Gate deployment population: model-predicted `Yes` samples only.",
        "- `oracle_flip_*` rows are routing upper bounds: they flip triggered predicted-Yes answers to `No` without running a second model pass.",
        f"- Margin scores available: `{margin_available}`.",
        "",
        "## Held-Out Gate Snapshot",
        "",
    ]
    if not gate_df.empty:
        preferred = gate_df[
            (gate_df["target_trigger_rate_predicted_yes"].round(3) == 0.2)
        ].copy()
        if preferred.empty:
            preferred = gate_df.copy()
        preferred = preferred.sort_values(
            ["layer", "triggered_fp_ratio", "fp_recall_among_predicted_yes"],
            ascending=[True, False, False],
        ).head(12)
        lines.append(
            "| Layer | Score | Target Rate | Triggered FP Ratio | FP Recall | TP Damage | Oracle F1 |"
        )
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in preferred.itertuples(index=False):
            lines.append(
                f"| {row.layer} | `{row.score}` | {row.target_trigger_rate_predicted_yes:.2f} | "
                f"{row.triggered_fp_ratio:.3f} | {row.fp_recall_among_predicted_yes:.3f} | "
                f"{row.tp_damage:.3f} | {row.oracle_flip_f1:.3f} |"
            )
    lines.extend(["", "## Random Gate Control", ""])
    if not random_df.empty:
        metric = random_df[random_df["metric"] == "triggered_fp_ratio"].copy()
        metric = metric.sort_values(["layer", "target_trigger_rate_predicted_yes", "mean"], ascending=[True, True, False]).head(12)
        lines.append("| Layer | Matched Score | Target Rate | Random Triggered FP Ratio Mean | P05 | P95 |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for row in metric.itertuples(index=False):
            lines.append(
                f"| {row.layer} | `{row.matched_score}` | {row.target_trigger_rate_predicted_yes:.2f} | "
                f"{row.mean:.3f} | {row.p05:.3f} | {row.p95:.3f} |"
            )
    lines.extend(["", "## Predicted-Yes AUROC Snapshot", ""])
    if not score_df.empty:
        pred_yes = score_df[
            (score_df["population"] == "predicted_yes_fp_vs_tp")
            & (score_df["split_role"] == "heldout_test")
        ].copy()
        pred_yes = pred_yes.sort_values(["layer", "auroc"], ascending=[True, False]).head(12)
        lines.append("| Layer | Score | N | FP Rate | AUROC | AUPRC |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for row in pred_yes.itertuples(index=False):
            lines.append(
                f"| {row.layer} | `{row.score}` | {row.n} | {row.positive_rate:.3f} | "
                f"{row.auroc:.3f} | {row.auprc:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []
