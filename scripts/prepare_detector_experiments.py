#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import load_hidden_layer, read_jsonl
from vgs.io import append_experiment_log, ensure_dir, write_json


TASKS = {
    "task_a_fp_vs_tn": {
        "display": "Task A: FP vs TN",
        "population": "fp_vs_tn",
        "labels": {"FP": 1, "TN": 0},
        "positive": "FP",
        "negative": "TN",
        "margin_risk": "yes_minus_no_logit",
    },
    "task_b_pred_yes_fp_vs_tp": {
        "display": "Task B: predicted-Yes FP vs TP",
        "population": "predicted_yes_fp_vs_tp",
        "labels": {"FP": 1, "TP": 0},
        "positive": "FP",
        "negative": "TP",
        "margin_risk": "low_yes_margin",
    },
}

DEFAULT_METHOD_ORDER = [
    "yes_no_margin",
    "binary_entropy",
    "output_logistic",
    "raw_img",
    "raw_blind",
    "raw_concat",
    "raw_diff",
    "random64_diff",
    "pca64_diff",
    "pca64_img",
    "top4_svd_diff",
    "top16_svd_diff",
    "top64_svd_diff",
    "top256_svd_diff",
    "tail_257_1024_diff",
    "pls32_diff",
    "margin_plus_top4_svd_diff",
    "margin_plus_top16_svd_diff",
    "margin_plus_top64_svd_diff",
    "margin_plus_top256_svd_diff",
    "margin_plus_full_diff",
    "margin_plus_pls32_diff",
    "margin_plus_tail_diff",
]

SPECTRAL_BANDS = [
    ("top_1_4", 1, 4, "dominant_backbone"),
    ("band_5_16", 5, 16, "early_useful_coordinates"),
    ("band_17_64", 17, 64, "mid_spectral"),
    ("band_65_256", 65, 256, "deeper_spectral"),
    ("tail_257_1024", 257, 1024, "tail"),
]


@dataclass(frozen=True)
class ScoreResult:
    name: str
    family: str
    feature_dim: int
    trainable: bool
    scores: np.ndarray
    fit_seconds: float
    score_seconds: float
    extra_blind_forward: int
    notes: str = ""

    @property
    def score_ms_per_sample(self) -> float:
        return 1000.0 * self.score_seconds / max(1, len(self.scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the detector minimal experiment package from cached POPE artifacts."
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--margins", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--output-dir", default="outputs/detector_minimal_package")
    parser.add_argument("--notes-path", default="notes/detector_experiment_prep.md")
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument("--train-subset", default="random")
    parser.add_argument("--calibration-subset", default="popular")
    parser.add_argument("--test-subset", default="adversarial")
    parser.add_argument("--top-k-grid", nargs="+", type=int, default=[4, 16, 64, 256])
    parser.add_argument("--dim-k-grid", nargs="+", type=int, default=[4, 16, 64, 256])
    parser.add_argument("--tail-start", type=int, default=257)
    parser.add_argument("--tail-end", type=int, default=1024)
    parser.add_argument("--pls-k", type=int, default=32)
    parser.add_argument("--random-dim", type=int, default=64)
    parser.add_argument("--trigger-rates", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--class-weights", nargs="+", default=["balanced", "none"])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-trigger-rate", type=float, default=0.2)
    parser.add_argument("--include-raw-pair", action="store_true")
    parser.add_argument("--skip-dim-curve", action="store_true")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "margins": args.margins,
        "output_dir": args.output_dir,
        "notes_path": args.notes_path,
        "layers": args.layers,
        "train_subset": args.train_subset,
        "calibration_subset": args.calibration_subset,
        "test_subset": args.test_subset,
        "top_k_grid": args.top_k_grid,
        "dim_k_grid": args.dim_k_grid,
        "tail_band": [args.tail_start, args.tail_end],
        "pls_k": args.pls_k,
        "random_dim": args.random_dim,
        "trigger_rates": args.trigger_rates,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_trigger_rate": args.bootstrap_trigger_rate,
        "include_raw_pair": args.include_raw_pair,
        "skip_dim_curve": args.skip_dim_curve,
    }
    if not args.dry_run:
        payload.update(
            prepare_detector_experiments(
                predictions_path=Path(args.predictions),
                hidden_states_dir=Path(args.hidden_states_dir),
                margins_path=Path(args.margins),
                output_dir=Path(args.output_dir),
                notes_path=Path(args.notes_path),
                layers=args.layers,
                train_subset=args.train_subset,
                calibration_subset=args.calibration_subset,
                test_subset=args.test_subset,
                top_k_grid=sorted(set(args.top_k_grid)),
                dim_k_grid=sorted(set(args.dim_k_grid)),
                tail_band=(args.tail_start, args.tail_end),
                pls_k=args.pls_k,
                random_dim=args.random_dim,
                trigger_rates=sorted(args.trigger_rates),
                c_grid=args.c_grid,
                class_weights=args.class_weights,
                max_iter=args.max_iter,
                seed=args.seed,
                random_repeats=args.random_repeats,
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_trigger_rate=args.bootstrap_trigger_rate,
                include_raw_pair=args.include_raw_pair,
                skip_dim_curve=args.skip_dim_curve,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "prepare_detector_experiments_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "prepare_detector_experiments",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


def prepare_detector_experiments(
    predictions_path: Path,
    hidden_states_dir: Path,
    margins_path: Path,
    output_dir: Path,
    notes_path: Path,
    layers: list[int],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
    top_k_grid: list[int],
    dim_k_grid: list[int],
    tail_band: tuple[int, int],
    pls_k: int,
    random_dim: int,
    trigger_rates: list[float],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    random_repeats: int,
    bootstrap_repeats: int,
    bootstrap_trigger_rate: float,
    include_raw_pair: bool,
    skip_dim_curve: bool,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    rows = read_jsonl(predictions_path)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    margin_by_id = _load_margin_rows(margins_path)

    baseline_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    dim_curve_rows: list[dict[str, Any]] = []
    spectral_band_rows: list[dict[str, Any]] = []
    pls_diagnostic_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    feature_audit_rows: list[dict[str, Any]] = []

    for layer in layers:
        with torch.no_grad():
            hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
        _require_alignment(sample_ids, rows_by_id, layer)
        metadata = [_metadata_row(rows_by_id[sample_id], margin_by_id.get(sample_id, {})) for sample_id in sample_ids]
        z_img = hidden["z_img"].float().numpy()
        z_blind = hidden["z_blind"].float().numpy()
        diff = z_blind - z_img
        output_features = _output_feature_matrix(metadata)

        train_population_idx = _subset_indices(metadata, train_subset)
        max_svd_k = _max_basis_k(
            requested=[*top_k_grid, *dim_k_grid, tail_band[1]],
            matrix=diff[train_population_idx],
        )
        max_pca_k = _max_basis_k(
            requested=[*dim_k_grid, min(64, diff.shape[1])],
            matrix=diff[train_population_idx],
        )
        svd_basis = _fit_svd_basis(diff[train_population_idx], max_svd_k, seed + layer, centered=False)
        pca_diff_basis = _fit_svd_basis(diff[train_population_idx], max_pca_k, seed + layer + 1, centered=True)
        pca_img_basis = _fit_svd_basis(z_img[train_population_idx], max_pca_k, seed + layer + 2, centered=True)
        random_basis = _random_basis(diff.shape[1], min(random_dim, diff.shape[1]), seed + layer)

        layer_context = {
            "sample_ids": sample_ids,
            "metadata": metadata,
            "z_img": z_img,
            "z_blind": z_blind,
            "diff": diff,
            "output_features": output_features,
            "svd_basis": svd_basis,
            "pca_diff_basis": pca_diff_basis,
            "pca_img_basis": pca_img_basis,
            "random_basis": random_basis,
            "train_population_idx": train_population_idx,
        }
        feature_audit_rows.extend(
            _feature_audit_rows(
                layer=layer,
                metadata=metadata,
                diff=diff,
                svd_basis=svd_basis,
                pca_diff_basis=pca_diff_basis,
                tail_band=tail_band,
            )
        )

        for task_name, task in TASKS.items():
            split = _task_split_indices(metadata, task["labels"], train_subset, calibration_subset, test_subset)
            if not _has_two_classes(split["y_train"]) or not _has_two_classes(split["y_test"]):
                baseline_rows.append(
                    {
                        "layer": layer,
                        "task": task_name,
                        "method": "SKIPPED",
                        "notes": "insufficient train/test class support",
                    }
                )
                continue
            score_results = _score_methods_for_task(
                layer_context=layer_context,
                task_name=task_name,
                task=task,
                split=split,
                top_k_grid=top_k_grid,
                tail_band=tail_band,
                pls_k=pls_k,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + layer,
                include_raw_pair=include_raw_pair,
            )
            for result in score_results:
                row, threshold_row = _baseline_metric_row(
                    layer=layer,
                    task_name=task_name,
                    task=task,
                    result=result,
                    split=split,
                    train_subset=train_subset,
                    calibration_subset=calibration_subset,
                    test_subset=test_subset,
                )
                baseline_rows.append(row)
                threshold_rows.append(threshold_row)

                if task_name == "task_b_pred_yes_fp_vs_tp":
                    warning_rows.extend(
                        _warning_rows(
                            layer=layer,
                            result=result,
                            split=split,
                            trigger_rates=trigger_rates,
                            random_repeats=random_repeats,
                            seed=seed + layer,
                        )
                    )
            if task_name == "task_b_pred_yes_fp_vs_tp":
                bootstrap_rows.extend(
                    _bootstrap_comparison_rows(
                        layer=layer,
                        task_name=task_name,
                        split=split,
                        score_results=score_results,
                        target_rate=bootstrap_trigger_rate,
                        repeats=bootstrap_repeats,
                        seed=seed + layer,
                    )
                )
            if not skip_dim_curve:
                dim_curve_rows.extend(
                    _dimension_curve_rows(
                        layer_context=layer_context,
                        layer=layer,
                        task_name=task_name,
                        task=task,
                        split=split,
                        dim_k_grid=dim_k_grid,
                        tail_band=tail_band,
                        c_grid=c_grid,
                        class_weights=class_weights,
                        max_iter=max_iter,
                        seed=seed + layer,
                    )
                )
            spectral_band_rows.extend(
                _spectral_band_rows(
                    layer_context=layer_context,
                    layer=layer,
                    task_name=task_name,
                    task=task,
                    split=split,
                    bands=SPECTRAL_BANDS,
                    c_grid=c_grid,
                    class_weights=class_weights,
                    max_iter=max_iter,
                    seed=seed + layer,
                    warning_target_rate=bootstrap_trigger_rate,
                )
            )
            pls_diagnostic_rows.extend(
                _pls_diagnostic_rows(
                    layer_context=layer_context,
                    layer=layer,
                    task_name=task_name,
                    split=split,
                    pls_k_grid=sorted(set([*dim_k_grid, pls_k])),
                    bands=SPECTRAL_BANDS,
                    c_grid=c_grid,
                    class_weights=class_weights,
                    max_iter=max_iter,
                    seed=seed + layer,
                )
            )

    baseline_df = _ordered_dataframe(baseline_rows, DEFAULT_METHOD_ORDER)
    warning_df = pd.DataFrame(warning_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    dim_curve_df = pd.DataFrame(dim_curve_rows)
    spectral_band_df = pd.DataFrame(spectral_band_rows)
    pls_diagnostic_df = pd.DataFrame(pls_diagnostic_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    feature_audit_df = pd.DataFrame(feature_audit_rows)

    paths = {
        "baseline_table": out / "detector_baseline_table.csv",
        "deployment_warning": out / "deployment_warning.csv",
        "threshold_audit": out / "threshold_audit.csv",
        "dimension_curve": out / "dimension_curve.csv",
        "spectral_band_curve": out / "spectral_band_curve.csv",
        "pls_diagnostics": out / "pls_diagnostics.csv",
        "bootstrap_comparisons": out / "bootstrap_comparisons.csv",
        "bootstrap_main_table": out / "bootstrap_main_table.csv",
        "trigger_curve_table": out / "trigger_curve_table.csv",
        "speed_cost_table": out / "speed_cost_table.csv",
        "feature_audit": out / "feature_audit.csv",
    }
    baseline_df.to_csv(paths["baseline_table"], index=False)
    warning_df.to_csv(paths["deployment_warning"], index=False)
    threshold_df.to_csv(paths["threshold_audit"], index=False)
    dim_curve_df.to_csv(paths["dimension_curve"], index=False)
    spectral_band_df.to_csv(paths["spectral_band_curve"], index=False)
    pls_diagnostic_df.to_csv(paths["pls_diagnostics"], index=False)
    bootstrap_df.to_csv(paths["bootstrap_comparisons"], index=False)
    bootstrap_main_df = _bootstrap_main_table(bootstrap_df)
    trigger_curve_df = _trigger_curve_table(warning_df)
    speed_cost_df = _speed_cost_table(baseline_df)
    bootstrap_main_df.to_csv(paths["bootstrap_main_table"], index=False)
    trigger_curve_df.to_csv(paths["trigger_curve_table"], index=False)
    speed_cost_df.to_csv(paths["speed_cost_table"], index=False)
    feature_audit_df.to_csv(paths["feature_audit"], index=False)

    note = _render_summary_note(
        baseline_df=baseline_df,
        warning_df=warning_df,
        dim_curve_df=dim_curve_df,
        spectral_band_df=spectral_band_df,
        pls_diagnostic_df=pls_diagnostic_df,
        bootstrap_df=bootstrap_df,
        bootstrap_main_df=bootstrap_main_df,
        trigger_curve_df=trigger_curve_df,
        speed_cost_df=speed_cost_df,
        feature_audit_df=feature_audit_df,
        paths=paths,
        train_subset=train_subset,
        calibration_subset=calibration_subset,
        test_subset=test_subset,
    )
    ensure_dir(notes_path.parent)
    notes_path.write_text(note, encoding="utf-8")
    summary_md = out / "detector_experiment_summary.md"
    summary_md.write_text(note, encoding="utf-8")

    return {
        "num_layers": len(layers),
        "num_baseline_rows": len(baseline_df),
        "num_warning_rows": len(warning_df),
        "num_dimension_curve_rows": len(dim_curve_df),
        "num_spectral_band_rows": len(spectral_band_df),
        "num_pls_diagnostic_rows": len(pls_diagnostic_df),
        "num_bootstrap_rows": len(bootstrap_df),
        "num_bootstrap_main_rows": len(bootstrap_main_df),
        "num_trigger_curve_rows": len(trigger_curve_df),
        "num_speed_cost_rows": len(speed_cost_df),
        "paths": {name: str(path) for name, path in paths.items()},
        "notes_path": str(notes_path),
        "summary_markdown_path": str(summary_md),
    }


def _score_methods_for_task(
    layer_context: dict[str, Any],
    task_name: str,
    task: dict[str, Any],
    split: dict[str, Any],
    top_k_grid: list[int],
    tail_band: tuple[int, int],
    pls_k: int,
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    include_raw_pair: bool,
) -> list[ScoreResult]:
    metadata = layer_context["metadata"]
    z_img = layer_context["z_img"]
    z_blind = layer_context["z_blind"]
    diff = layer_context["diff"]
    output_features = layer_context["output_features"]
    svd_basis = layer_context["svd_basis"]
    pca_diff_basis = layer_context["pca_diff_basis"]
    pca_img_basis = layer_context["pca_img_basis"]
    random_basis = layer_context["random_basis"]

    train_idx = split["train_idx"]
    y_train = split["y_train"]
    val_idx = split["val_idx"]
    y_val = split["y_val"]

    results: list[ScoreResult] = []
    margin_scores = _task_margin_scores(metadata, task["margin_risk"])
    results.append(
        ScoreResult(
            name="yes_no_margin",
            family="black_box_confidence",
            feature_dim=1,
            trainable=False,
            scores=margin_scores,
            fit_seconds=0.0,
            score_seconds=0.0,
            extra_blind_forward=0,
            notes=f"risk orientation: {task['margin_risk']}",
        )
    )
    results.append(
        ScoreResult(
            name="binary_entropy",
            family="black_box_confidence",
            feature_dim=1,
            trainable=False,
            scores=np.array([row["binary_entropy"] for row in metadata], dtype=float),
            fit_seconds=0.0,
            score_seconds=0.0,
            extra_blind_forward=0,
        )
    )
    results.append(
        _fit_score_result(
            name="output_logistic",
            family="trainable_output_baseline",
            x_all=output_features,
            train_idx=train_idx,
            y_train=y_train,
            val_idx=val_idx,
            y_val=y_val,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 3,
            extra_blind_forward=0,
        )
    )

    results.append(
        _fit_score_result(
            name="raw_img",
            family="raw_representation",
            x_all=z_img,
            train_idx=train_idx,
            y_train=y_train,
            val_idx=val_idx,
            y_val=y_val,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 5,
            extra_blind_forward=0,
        )
    )
    results.append(
        _fit_score_result(
            name="raw_blind",
            family="raw_representation",
            x_all=z_blind,
            train_idx=train_idx,
            y_train=y_train,
            val_idx=val_idx,
            y_val=y_val,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 7,
            extra_blind_forward=1,
        )
    )
    results.append(
        _fit_score_result(
            name="raw_concat",
            family="raw_representation",
            x_all=np.concatenate([z_img, z_blind], axis=1),
            train_idx=train_idx,
            y_train=y_train,
            val_idx=val_idx,
            y_val=y_val,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 11,
            extra_blind_forward=1,
        )
    )
    results.append(
        _fit_score_result(
            name="raw_diff",
            family="raw_correction_difference",
            x_all=diff,
            train_idx=train_idx,
            y_train=y_train,
            val_idx=val_idx,
            y_val=y_val,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 13,
            extra_blind_forward=1,
        )
    )
    if include_raw_pair:
        results.append(
            _fit_score_result(
                name="raw_pair",
                family="raw_representation",
                x_all=np.concatenate([z_img, z_blind, diff], axis=1),
                train_idx=train_idx,
                y_train=y_train,
                val_idx=val_idx,
                y_val=y_val,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 17,
                extra_blind_forward=1,
            )
        )

    if random_basis.shape[1] > 0:
        results.append(
            _fit_projected_result(
                name=f"random{random_basis.shape[1]}_diff",
                family="ordinary_random_subspace",
                x_all=diff,
                basis=random_basis,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 19,
            )
        )
    if pca_diff_basis.shape[1] > 0:
        pca_k = min(64, pca_diff_basis.shape[1])
        results.append(
            _fit_projected_result(
                name=f"pca{pca_k}_diff",
                family="ordinary_pca_subspace",
                x_all=diff,
                basis=pca_diff_basis[:, :pca_k],
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 23,
            )
        )
    if pca_img_basis.shape[1] > 0:
        pca_k = min(64, pca_img_basis.shape[1])
        results.append(
            _fit_projected_result(
                name=f"pca{pca_k}_img",
                family="ordinary_pca_raw_img",
                x_all=z_img,
                basis=pca_img_basis[:, :pca_k],
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 29,
                extra_blind_forward=0,
            )
        )

    top_results: list[ScoreResult] = []
    for k in top_k_grid:
        k_eff = min(k, svd_basis.shape[1])
        if k_eff <= 0:
            continue
        top_result = _fit_projected_result(
            name=f"top{k_eff}_svd_diff",
            family="correction_top_svd",
            x_all=diff,
            basis=svd_basis[:, :k_eff],
            split=split,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 31 + k_eff,
        )
        results.append(top_result)
        top_results.append(top_result)

    tail_result = _tail_result(
        diff=diff,
        svd_basis=svd_basis,
        split=split,
        tail_band=tail_band,
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed + 41,
    )
    if tail_result is not None:
        results.append(tail_result)

    pls_basis = _pls_basis(diff[train_idx], y_train, pls_k)
    pls_result: ScoreResult | None = None
    if pls_basis.shape[1] > 0:
        pls_result = _fit_projected_result(
            name=f"pls{pls_basis.shape[1]}_diff",
            family="supervised_pls_correction_subspace",
            x_all=diff,
            basis=pls_basis,
            split=split,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 43,
        )
        results.append(pls_result)

    result_by_name = {result.name: result for result in results}
    full = result_by_name.get("raw_diff")
    for offset, top_result in enumerate(top_results):
        results.append(
            _fit_combined_score(
                name=f"margin_plus_top{top_result.feature_dim}_svd_diff",
                family="combined_margin_top_svd",
                margin_scores=margin_scores,
                geometry_scores=top_result.scores,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 45 + offset,
                geometry_dim=top_result.feature_dim,
            )
        )
    if full is not None:
        results.append(
            _fit_combined_score(
                name="margin_plus_full_diff",
                family="combined_margin_geometry",
                margin_scores=margin_scores,
                geometry_scores=full.scores,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 47,
                geometry_dim=full.feature_dim,
            )
        )
    if pls_result is not None:
        results.append(
            _fit_combined_score(
                name="margin_plus_pls32_diff",
                family="combined_margin_pls",
                margin_scores=margin_scores,
                geometry_scores=pls_result.scores,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 53,
                geometry_dim=pls_result.feature_dim,
            )
        )
    if tail_result is not None:
        results.append(
            _fit_combined_score(
                name="margin_plus_tail_diff",
                family="combined_margin_tail",
                margin_scores=margin_scores,
                geometry_scores=tail_result.scores,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 59,
                geometry_dim=tail_result.feature_dim,
            )
        )

    return results


def _fit_score_result(
    name: str,
    family: str,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    val_idx: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    extra_blind_forward: int,
) -> ScoreResult:
    start = time.perf_counter()
    scorer, notes = _fit_logistic_grid(
        x_train=x_all[train_idx],
        y_train=y_train,
        x_val=x_all[val_idx],
        y_val=y_val,
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed,
    )
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = scorer(x_all)
    score_seconds = time.perf_counter() - start
    return ScoreResult(
        name=name,
        family=family,
        feature_dim=int(x_all.shape[1]),
        trainable=True,
        scores=scores,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        extra_blind_forward=extra_blind_forward,
        notes=notes,
    )


def _fit_projected_result(
    name: str,
    family: str,
    x_all: np.ndarray,
    basis: np.ndarray,
    split: dict[str, Any],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    extra_blind_forward: int = 1,
) -> ScoreResult:
    start_project = time.perf_counter()
    projected = x_all @ basis
    project_seconds = time.perf_counter() - start_project
    result = _fit_score_result(
        name=name,
        family=family,
        x_all=projected,
        train_idx=split["train_idx"],
        y_train=split["y_train"],
        val_idx=split["val_idx"],
        y_val=split["y_val"],
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed,
        extra_blind_forward=extra_blind_forward,
    )
    return ScoreResult(
        name=result.name,
        family=result.family,
        feature_dim=int(basis.shape[1]),
        trainable=result.trainable,
        scores=result.scores,
        fit_seconds=result.fit_seconds,
        score_seconds=result.score_seconds + project_seconds,
        extra_blind_forward=result.extra_blind_forward,
        notes=result.notes,
    )


def _fit_combined_score(
    name: str,
    family: str,
    margin_scores: np.ndarray,
    geometry_scores: np.ndarray,
    split: dict[str, Any],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    geometry_dim: int,
) -> ScoreResult:
    x_all = np.column_stack([margin_scores, geometry_scores])
    result = _fit_score_result(
        name=name,
        family=family,
        x_all=x_all,
        train_idx=split["train_idx"],
        y_train=split["y_train"],
        val_idx=split["val_idx"],
        y_val=split["y_val"],
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed,
        extra_blind_forward=1,
    )
    return ScoreResult(
        name=result.name,
        family=result.family,
        feature_dim=geometry_dim + 1,
        trainable=True,
        scores=result.scores,
        fit_seconds=result.fit_seconds,
        score_seconds=result.score_seconds,
        extra_blind_forward=1,
        notes=result.notes,
    )


def _fit_logistic_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    finite_cols = np.isfinite(x_train).all(axis=0)
    if not finite_cols.any():
        raise ValueError("All feature columns contain non-finite values.")
    x_train = x_train[:, finite_cols]
    x_val = x_val[:, finite_cols] if len(x_val) else x_val[:, finite_cols]

    best: tuple[float, float, str, StandardScaler, LogisticRegression] | None = None
    for c_value in c_grid:
        for weight in class_weights:
            class_weight: str | None = None if weight == "none" else weight
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            clf = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=seed,
                solver="liblinear",
            )
            clf.fit(x_train_scaled, y_train)
            if len(x_val) and _has_two_classes(y_val):
                val_scores = clf.predict_proba(scaler.transform(x_val))[:, 1]
                primary = _safe_metric(y_val, val_scores, roc_auc_score)
                secondary = _safe_metric(y_val, val_scores, average_precision_score)
            else:
                train_scores = clf.predict_proba(x_train_scaled)[:, 1]
                primary = _safe_metric(y_train, train_scores, roc_auc_score)
                secondary = _safe_metric(y_train, train_scores, average_precision_score)
            candidate = (
                _nan_to_sortable(primary),
                _nan_to_sortable(secondary),
                f"C={c_value};class_weight={weight};kept_cols={int(finite_cols.sum())}",
                scaler,
                clf,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise RuntimeError("No logistic model was fitted.")
    _, _, notes, scaler, clf = best

    def scorer(x: np.ndarray, mask: np.ndarray = finite_cols, fitted_scaler: StandardScaler = scaler, fitted_clf: LogisticRegression = clf) -> np.ndarray:
        return fitted_clf.predict_proba(fitted_scaler.transform(x[:, mask]))[:, 1]

    return scorer, notes


def _baseline_metric_row(
    layer: int,
    task_name: str,
    task: dict[str, Any],
    result: ScoreResult,
    split: dict[str, Any],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold, threshold_info = _best_f1_threshold(result.scores[split["val_idx"]], split["y_val"])
    train_metrics = _ranking_metrics(split["y_train"], result.scores[split["train_idx"]])
    val_metrics = _ranking_metrics(split["y_val"], result.scores[split["val_idx"]])
    test_scores = result.scores[split["test_idx"]]
    y_test = split["y_test"]
    cls_metrics = _classification_metrics(y_test, test_scores, threshold)
    calibration = _calibration_metrics(y_test, test_scores)
    row = {
        "layer": layer,
        "task": task_name,
        "task_display": task["display"],
        "train_subset": train_subset,
        "calibration_subset": calibration_subset,
        "test_subset": test_subset,
        "method": result.name,
        "family": result.family,
        "feature_dim": result.feature_dim,
        "trainable": result.trainable,
        "extra_blind_forward": result.extra_blind_forward,
        "train_n": int(len(split["y_train"])),
        "train_positive_n": int(np.sum(split["y_train"] == 1)),
        "calibration_n": int(len(split["y_val"])),
        "calibration_positive_n": int(np.sum(split["y_val"] == 1)),
        "test_n": int(len(y_test)),
        "test_positive_n": int(np.sum(y_test == 1)),
        "test_base_rate": float(np.mean(y_test)) if len(y_test) else math.nan,
        "train_auroc": train_metrics["auroc"],
        "train_auprc": train_metrics["auprc"],
        "calibration_auroc": val_metrics["auroc"],
        "calibration_auprc": val_metrics["auprc"],
        "test_auroc": _ranking_metrics(y_test, test_scores)["auroc"],
        "test_auprc": _ranking_metrics(y_test, test_scores)["auprc"],
        "threshold_policy": "calibration_f1_optimal",
        "threshold": threshold,
        **cls_metrics,
        **calibration,
        "fit_seconds": result.fit_seconds,
        "detector_score_ms_per_sample": result.score_ms_per_sample,
        "notes": result.notes,
    }
    threshold_row = {
        "layer": layer,
        "task": task_name,
        "method": result.name,
        "threshold_policy": "calibration_f1_optimal",
        "threshold": threshold,
        **threshold_info,
    }
    return row, threshold_row


def _warning_rows(
    layer: int,
    result: ScoreResult,
    split: dict[str, Any],
    trigger_rates: list[float],
    random_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    val_scores = result.scores[split["val_idx"]]
    test_scores = result.scores[split["test_idx"]]
    y_test = split["y_test"]
    base_rate = float(np.mean(y_test)) if len(y_test) else math.nan
    for rate in trigger_rates:
        threshold = _fixed_rate_threshold(val_scores, rate)
        trigger = test_scores >= threshold
        rows.append(
            {
                "layer": layer,
                "method": result.name,
                "family": result.family,
                "target_trigger_rate": rate,
                "gate": "score_top_rate",
                "threshold": threshold,
                **_warning_metrics(y_test, trigger, base_rate),
            }
        )
        random_stats = _random_warning_metrics(y_test, int(np.sum(trigger)), random_repeats, seed + int(rate * 1000))
        rows.append(
            {
                "layer": layer,
                "method": result.name,
                "family": result.family,
                "target_trigger_rate": rate,
                "gate": "same_trigger_random_mean",
                "threshold": math.nan,
                **random_stats,
            }
        )
    return rows


def _dimension_curve_rows(
    layer_context: dict[str, Any],
    layer: int,
    task_name: str,
    task: dict[str, Any],
    split: dict[str, Any],
    dim_k_grid: list[int],
    tail_band: tuple[int, int],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    diff = layer_context["diff"]
    svd_basis = layer_context["svd_basis"]
    pca_diff_basis = layer_context["pca_diff_basis"]
    random_dim = min(max(dim_k_grid), diff.shape[1])
    rows: list[dict[str, Any]] = []
    full = _fit_score_result(
        name="raw_full_diff_reference",
        family="raw_correction_difference",
        x_all=diff,
        train_idx=split["train_idx"],
        y_train=split["y_train"],
        val_idx=split["val_idx"],
        y_val=split["y_val"],
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed + 101,
        extra_blind_forward=1,
    )
    rows.append(_dimension_metric_row(layer, task_name, full, split, "raw_full_diff_reference"))

    for k in dim_k_grid:
        k_eff = min(k, diff.shape[1], svd_basis.shape[1])
        if k_eff <= 0:
            continue
        random_basis = _random_basis(diff.shape[1], min(k, random_dim), seed + 200 + k)
        specs = [
            ("random", random_basis),
            ("pca_diff", pca_diff_basis[:, : min(k, pca_diff_basis.shape[1])]),
            ("top_svd", svd_basis[:, :k_eff]),
        ]
        pls_basis = _pls_basis(diff[split["train_idx"]], split["y_train"], k)
        if pls_basis.shape[1] > 0:
            specs.append(("pls", pls_basis))
        for method, basis in specs:
            if basis.shape[1] <= 0:
                continue
            result = _fit_projected_result(
                name=f"{method}_{basis.shape[1]}",
                family=f"dimension_curve_{method}",
                x_all=diff,
                basis=basis,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 300 + k,
            )
            rows.append(_dimension_metric_row(layer, task_name, result, split, method))

    tail_start, tail_end = tail_band
    tail_start_idx = max(0, tail_start - 1)
    tail_end_idx = min(tail_end, svd_basis.shape[1])
    if tail_end_idx > tail_start_idx:
        tail_basis = svd_basis[:, tail_start_idx:tail_end_idx]
        result = _fit_projected_result(
            name=f"tail_{tail_start}_{tail_end}",
            family="dimension_curve_tail",
            x_all=diff,
            basis=tail_basis,
            split=split,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 503,
        )
        rows.append(_dimension_metric_row(layer, task_name, result, split, "tail"))
    return rows


def _spectral_band_rows(
    layer_context: dict[str, Any],
    layer: int,
    task_name: str,
    task: dict[str, Any],
    split: dict[str, Any],
    bands: list[tuple[str, int, int, str]],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
    warning_target_rate: float,
) -> list[dict[str, Any]]:
    diff = layer_context["diff"]
    svd_basis = layer_context["svd_basis"]
    margin_scores = _task_margin_scores(layer_context["metadata"], task["margin_risk"])
    rows: list[dict[str, Any]] = []
    for offset, (band_name, start, end, role) in enumerate(bands):
        band_basis = _basis_slice(svd_basis, start, end)
        cumulative_basis = _basis_slice(svd_basis, 1, end)
        specs = [
            ("band_only", band_name, start, end, band_basis),
            ("cumulative_top_k", f"top_1_{end}", 1, end, cumulative_basis),
        ]
        for mode, spectral_feature, spec_start, spec_end, basis in specs:
            if basis.shape[1] <= 0:
                continue
            result = _fit_projected_result(
                name=f"{mode}_{spectral_feature}",
                family=f"spectral_{mode}",
                x_all=diff,
                basis=basis,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 700 + offset + basis.shape[1],
            )
            rows.append(
                _spectral_metric_row(
                    layer=layer,
                    task_name=task_name,
                    result=result,
                    split=split,
                    mode=mode,
                    spectral_feature=spectral_feature,
                    spectral_role=role,
                    start=spec_start,
                    end=spec_end,
                    warning_target_rate=warning_target_rate,
                )
            )
            combined = _fit_combined_score(
                name=f"margin_plus_{mode}_{spectral_feature}",
                family=f"margin_plus_spectral_{mode}",
                margin_scores=margin_scores,
                geometry_scores=result.scores,
                split=split,
                c_grid=c_grid,
                class_weights=class_weights,
                max_iter=max_iter,
                seed=seed + 900 + offset + basis.shape[1],
                geometry_dim=result.feature_dim,
            )
            rows.append(
                _spectral_metric_row(
                    layer=layer,
                    task_name=task_name,
                    result=combined,
                    split=split,
                    mode=f"margin_plus_{mode}",
                    spectral_feature=spectral_feature,
                    spectral_role=role,
                    start=spec_start,
                    end=spec_end,
                    warning_target_rate=warning_target_rate,
                )
            )
    return rows


def _spectral_metric_row(
    layer: int,
    task_name: str,
    result: ScoreResult,
    split: dict[str, Any],
    mode: str,
    spectral_feature: str,
    spectral_role: str,
    start: int,
    end: int,
    warning_target_rate: float,
) -> dict[str, Any]:
    train = _ranking_metrics(split["y_train"], result.scores[split["train_idx"]])
    val = _ranking_metrics(split["y_val"], result.scores[split["val_idx"]])
    test_scores = result.scores[split["test_idx"]]
    test = _ranking_metrics(split["y_test"], test_scores)
    threshold = _fixed_rate_threshold(result.scores[split["val_idx"]], warning_target_rate)
    warning = _warning_metrics(
        split["y_test"],
        test_scores >= threshold,
        float(np.mean(split["y_test"])) if len(split["y_test"]) else math.nan,
    )
    return {
        "layer": layer,
        "task": task_name,
        "mode": mode,
        "spectral_feature": spectral_feature,
        "spectral_role": spectral_role,
        "start": start,
        "end": end,
        "feature_dim": result.feature_dim,
        "score_name": result.name,
        "train_auroc": train["auroc"],
        "train_auprc": train["auprc"],
        "calibration_auroc": val["auroc"],
        "calibration_auprc": val["auprc"],
        "test_auroc": test["auroc"],
        "test_auprc": test["auprc"],
        "warning_target_rate": warning_target_rate,
        "warning_trigger_rate": warning["trigger_rate"],
        "warning_precision": warning["warning_precision"],
        "fp_recall": warning["fp_recall"],
        "tp_damage": warning["tp_damage"],
        "fit_seconds": result.fit_seconds,
        "detector_score_ms_per_sample": result.score_ms_per_sample,
    }


def _pls_diagnostic_rows(
    layer_context: dict[str, Any],
    layer: int,
    task_name: str,
    split: dict[str, Any],
    pls_k_grid: list[int],
    bands: list[tuple[str, int, int, str]],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    diff = layer_context["diff"]
    svd_basis = layer_context["svd_basis"]
    train_idx = split["train_idx"]
    y_train = split["y_train"]
    half_a, half_b = _stratified_halves(train_idx, y_train, seed + layer)
    rows: list[dict[str, Any]] = []
    for k in pls_k_grid:
        basis = _pls_basis(diff[train_idx], y_train, k)
        if basis.shape[1] <= 0:
            continue
        result = _fit_projected_result(
            name=f"pls{k}_diagnostic",
            family="pls_diagnostic",
            x_all=diff,
            basis=basis,
            split=split,
            c_grid=c_grid,
            class_weights=class_weights,
            max_iter=max_iter,
            seed=seed + 1100 + k,
        )
        train = _ranking_metrics(split["y_train"], result.scores[split["train_idx"]])
        val = _ranking_metrics(split["y_val"], result.scores[split["val_idx"]])
        test = _ranking_metrics(split["y_test"], result.scores[split["test_idx"]])
        left = _pls_basis(diff[half_a], _labels_for_indices(train_idx, y_train, half_a), k)
        right = _pls_basis(diff[half_b], _labels_for_indices(train_idx, y_train, half_b), k)
        row = {
            "layer": layer,
            "task": task_name,
            "k": k,
            "effective_k": basis.shape[1],
            "train_auroc": train["auroc"],
            "train_auprc": train["auprc"],
            "calibration_auroc": val["auroc"],
            "calibration_auprc": val["auprc"],
            "test_auroc": test["auroc"],
            "test_auprc": test["auprc"],
            "split_half_overlap": _subspace_overlap(left, right),
            "overlap_top4": _subspace_overlap(basis, _basis_slice(svd_basis, 1, 4)),
            "overlap_top16": _subspace_overlap(basis, _basis_slice(svd_basis, 1, 16)),
            "overlap_top64": _subspace_overlap(basis, _basis_slice(svd_basis, 1, 64)),
            "overlap_tail_257_1024": _subspace_overlap(basis, _basis_slice(svd_basis, 257, 1024)),
            "fit_seconds": result.fit_seconds,
        }
        for band_name, start, end, _role in bands:
            row[f"overlap_{band_name}"] = _subspace_overlap(basis, _basis_slice(svd_basis, start, end))
        rows.append(row)
    return rows


def _bootstrap_comparison_rows(
    layer: int,
    task_name: str,
    split: dict[str, Any],
    score_results: list[ScoreResult],
    target_rate: float,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_name = {result.name: result for result in score_results}
    pairs = [
        ("margin_plus_tail_diff", "yes_no_margin"),
        ("margin_plus_full_diff", "yes_no_margin"),
        ("margin_plus_tail_diff", "margin_plus_full_diff"),
        ("margin_plus_top16_svd_diff", "yes_no_margin"),
        ("margin_plus_tail_diff", "raw_diff"),
    ]
    rows: list[dict[str, Any]] = []
    y = split["y_test"]
    if len(y) == 0:
        return rows
    rng = np.random.default_rng(seed + 2027)
    bootstrap_indices = rng.integers(0, len(y), size=(repeats, len(y)))
    for method_a, method_b in pairs:
        if method_a not in by_name or method_b not in by_name:
            continue
        scores_a = by_name[method_a].scores
        scores_b = by_name[method_b].scores
        test_a = scores_a[split["test_idx"]]
        test_b = scores_b[split["test_idx"]]
        threshold_a = _fixed_rate_threshold(scores_a[split["val_idx"]], target_rate)
        threshold_b = _fixed_rate_threshold(scores_b[split["val_idx"]], target_rate)
        trigger_a = test_a >= threshold_a
        trigger_b = test_b >= threshold_b
        metric_specs = [
            ("auroc", True, lambda yy, sa, sb, ta, tb: _safe_metric(yy, sa, roc_auc_score) - _safe_metric(yy, sb, roc_auc_score)),
            ("auprc", True, lambda yy, sa, sb, ta, tb: _safe_metric(yy, sa, average_precision_score) - _safe_metric(yy, sb, average_precision_score)),
            ("warning_precision", True, lambda yy, sa, sb, ta, tb: _warning_metric_value(yy, ta, "warning_precision") - _warning_metric_value(yy, tb, "warning_precision")),
            ("fp_recall", True, lambda yy, sa, sb, ta, tb: _warning_metric_value(yy, ta, "fp_recall") - _warning_metric_value(yy, tb, "fp_recall")),
            ("tp_damage", False, lambda yy, sa, sb, ta, tb: _warning_metric_value(yy, ta, "tp_damage") - _warning_metric_value(yy, tb, "tp_damage")),
        ]
        for metric, higher_is_better, metric_fn in metric_specs:
            deltas = []
            for sample_idx in bootstrap_indices:
                yy = y[sample_idx]
                if metric in {"auroc", "auprc"} and not _has_two_classes(yy):
                    continue
                delta = metric_fn(
                    yy,
                    test_a[sample_idx],
                    test_b[sample_idx],
                    trigger_a[sample_idx],
                    trigger_b[sample_idx],
                )
                if math.isfinite(delta):
                    deltas.append(delta)
            delta_arr = np.array(deltas, dtype=float)
            if len(delta_arr) == 0:
                continue
            rows.append(
                {
                    "layer": layer,
                    "task": task_name,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "target_trigger_rate": target_rate if metric not in {"auroc", "auprc"} else "",
                    "higher_is_better": higher_is_better,
                    "delta_mean": float(np.mean(delta_arr)),
                    "delta_ci95_low": float(np.percentile(delta_arr, 2.5)),
                    "delta_ci95_high": float(np.percentile(delta_arr, 97.5)),
                    "p_delta_le_0": float(np.mean(delta_arr <= 0.0)),
                    "p_delta_ge_0": float(np.mean(delta_arr >= 0.0)),
                    "bootstrap_repeats": int(len(delta_arr)),
                }
            )
    return rows


def _dimension_metric_row(
    layer: int,
    task_name: str,
    result: ScoreResult,
    split: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    metrics = _ranking_metrics(split["y_test"], result.scores[split["test_idx"]])
    return {
        "layer": layer,
        "task": task_name,
        "method": method,
        "score_name": result.name,
        "feature_dim": result.feature_dim,
        "test_auroc": metrics["auroc"],
        "test_auprc": metrics["auprc"],
        "fit_seconds": result.fit_seconds,
        "detector_score_ms_per_sample": result.score_ms_per_sample,
    }


def _tail_result(
    diff: np.ndarray,
    svd_basis: np.ndarray,
    split: dict[str, Any],
    tail_band: tuple[int, int],
    c_grid: list[float],
    class_weights: list[str],
    max_iter: int,
    seed: int,
) -> ScoreResult | None:
    tail_start, tail_end = tail_band
    tail_start_idx = max(0, tail_start - 1)
    tail_end_idx = min(tail_end, svd_basis.shape[1])
    if tail_end_idx <= tail_start_idx:
        return None
    return _fit_projected_result(
        name=f"tail_{tail_start}_{tail_end}_diff",
        family="correction_tail_svd",
        x_all=diff,
        basis=svd_basis[:, tail_start_idx:tail_end_idx],
        split=split,
        c_grid=c_grid,
        class_weights=class_weights,
        max_iter=max_iter,
        seed=seed,
    )


def _basis_slice(basis: np.ndarray, start: int, end: int) -> np.ndarray:
    start_idx = max(0, start - 1)
    end_idx = min(end, basis.shape[1])
    if end_idx <= start_idx:
        return np.zeros((basis.shape[0], 0), dtype=np.float32)
    return basis[:, start_idx:end_idx]


def _subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.shape[0] != right.shape[0]:
        return math.nan
    q_left = _orthonormal_columns(left)
    q_right = _orthonormal_columns(right)
    if q_left.size == 0 or q_right.size == 0:
        return math.nan
    singular_values = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return float(np.sum(singular_values**2) / max(1, q_left.shape[1]))


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float32)
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float32, copy=False)


def _stratified_halves(train_idx: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    for label in sorted(set(y_train.tolist())):
        indices = train_idx[y_train == label].copy()
        rng.shuffle(indices)
        split_at = max(1, len(indices) // 2)
        left.extend(indices[:split_at].tolist())
        right.extend(indices[split_at:].tolist())
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def _labels_for_indices(train_idx: np.ndarray, y_train: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    label_by_idx = {int(idx): int(label) for idx, label in zip(train_idx, y_train)}
    return np.array([label_by_idx[int(idx)] for idx in selected_idx], dtype=np.int64)


def _warning_metric_value(y: np.ndarray, trigger: np.ndarray, metric: str) -> float:
    base_rate = float(np.mean(y)) if len(y) else math.nan
    return float(_warning_metrics(y, trigger, base_rate)[metric])


def _task_split_indices(
    metadata: list[dict[str, Any]],
    labels: dict[str, int],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
) -> dict[str, Any]:
    train_idx, y_train = _label_indices(metadata, train_subset, labels)
    val_idx, y_val = _label_indices(metadata, calibration_subset, labels)
    test_idx, y_test = _label_indices(metadata, test_subset, labels)
    return {
        "train_idx": train_idx,
        "y_train": y_train,
        "val_idx": val_idx,
        "y_val": y_val,
        "test_idx": test_idx,
        "y_test": y_test,
    }


def _label_indices(
    metadata: list[dict[str, Any]],
    subset: str,
    labels: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    y: list[int] = []
    for idx, row in enumerate(metadata):
        if row["subset"] != subset or row["outcome"] not in labels:
            continue
        indices.append(idx)
        y.append(labels[row["outcome"]])
    return np.array(indices, dtype=np.int64), np.array(y, dtype=np.int64)


def _subset_indices(metadata: list[dict[str, Any]], subset: str) -> np.ndarray:
    return np.array([idx for idx, row in enumerate(metadata) if row["subset"] == subset], dtype=np.int64)


def _metadata_row(row: dict[str, Any], margin: dict[str, Any]) -> dict[str, Any]:
    yes_logit = _maybe_float(margin.get("yes_logit"))
    no_logit = _maybe_float(margin.get("no_logit"))
    yes_minus_no = _maybe_float(margin.get("yes_minus_no_logit"))
    binary_entropy = _maybe_float(margin.get("binary_entropy"))
    yes_prob = _binary_yes_probability(yes_logit, no_logit)
    return {
        "sample_id": str(row.get("sample_id", "")),
        "subset": str(row.get("subset", "")),
        "label": str(row.get("label", "")),
        "outcome": str(row.get("outcome", "")),
        "parsed_prediction": str(row.get("parsed_prediction", "")),
        "question": str(row.get("question", "")),
        "image": str(row.get("image", "")),
        "image_path": str(row.get("image_path", "")),
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_minus_no_logit": yes_minus_no,
        "low_yes_margin": -yes_minus_no if math.isfinite(yes_minus_no) else math.nan,
        "abs_yes_no_margin": abs(yes_minus_no) if math.isfinite(yes_minus_no) else math.nan,
        "binary_entropy": binary_entropy,
        "yes_probability": yes_prob,
        "max_binary_probability": max(yes_prob, 1.0 - yes_prob) if math.isfinite(yes_prob) else math.nan,
    }


def _output_feature_matrix(metadata: list[dict[str, Any]]) -> np.ndarray:
    columns = [
        "yes_logit",
        "no_logit",
        "yes_minus_no_logit",
        "low_yes_margin",
        "abs_yes_no_margin",
        "binary_entropy",
        "yes_probability",
        "max_binary_probability",
    ]
    return np.array([[row[col] for col in columns] for row in metadata], dtype=float)


def _task_margin_scores(metadata: list[dict[str, Any]], margin_key: str) -> np.ndarray:
    return np.array([row[margin_key] for row in metadata], dtype=float)


def _fit_svd_basis(matrix: np.ndarray, k: int, seed: int, centered: bool) -> np.ndarray:
    if k <= 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float32)
    x = matrix.astype(np.float32, copy=False)
    if centered:
        x = x - np.nanmean(x, axis=0, keepdims=True)
    k_eff = min(k, x.shape[0] - 1, x.shape[1])
    if k_eff <= 0:
        return np.zeros((x.shape[1], 0), dtype=np.float32)
    _, _, vt = randomized_svd(x, n_components=k_eff, n_iter=4, random_state=seed)
    return vt.T.astype(np.float32, copy=False)


def _pls_basis(matrix: np.ndarray, y: np.ndarray, max_k: int) -> np.ndarray:
    if max_k <= 0 or not _has_two_classes(y):
        return np.zeros((matrix.shape[1], 0), dtype=np.float32)
    n_components = min(max_k, matrix.shape[1], max(1, matrix.shape[0] - 1))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x_scaled, y.astype(np.float64))
    basis = pls.x_weights_ / np.maximum(scaler.scale_[:, None], 1e-12)
    q, _ = np.linalg.qr(basis)
    return q[:, :n_components].astype(np.float32, copy=False)


def _random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((dim, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q[:, :k].astype(np.float32, copy=False)


def _max_basis_k(requested: list[int], matrix: np.ndarray) -> int:
    return min(max(requested), matrix.shape[0] - 1, matrix.shape[1])


def _best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, Any]]:
    finite = np.isfinite(scores)
    scores = scores[finite]
    y = y[finite]
    if len(y) == 0:
        return math.nan, {"calibration_f1": math.nan, "calibration_precision": math.nan, "calibration_recall": math.nan}
    candidates = np.unique(scores)
    if len(candidates) > 512:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 512))
        candidates = np.unique(candidates)
    best = (-1.0, -1.0, -1.0, float(candidates[0]))
    for threshold in candidates:
        pred = scores >= threshold
        f1 = f1_score(y, pred, zero_division=0)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        candidate = (f1, precision, recall, float(threshold))
        if candidate[:3] > best[:3]:
            best = candidate
    return best[3], {
        "calibration_f1": best[0],
        "calibration_precision": best[1],
        "calibration_recall": best[2],
    }


def _fixed_rate_threshold(scores: np.ndarray, rate: float) -> float:
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        return math.nan
    n_trigger = max(1, min(len(finite_scores), int(math.ceil(rate * len(finite_scores)))))
    return float(np.sort(finite_scores)[-n_trigger])


def _ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": _safe_metric(y, scores, roc_auc_score),
        "auprc": _safe_metric(y, scores, average_precision_score),
    }


def _classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    finite = np.isfinite(scores)
    y = y[finite]
    pred = scores[finite] >= threshold
    if len(y) == 0:
        return {}
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(y.tolist())) > 1 else math.nan,
        "detector_tn": int(tn),
        "detector_fp": int(fp),
        "detector_fn": int(fn),
        "detector_tp": int(tp),
    }


def _calibration_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or len(set(y.tolist())) < 2 or np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0:
        return {"brier": math.nan, "nll": math.nan, "ece_10": math.nan}
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, clipped)),
        "nll": float(log_loss(y, clipped, labels=[0, 1])),
        "ece_10": _ece_10(y, clipped),
    }


def _warning_metrics(y: np.ndarray, trigger: np.ndarray, base_rate: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    trigger = np.asarray(trigger, dtype=bool)
    fp_total = int(np.sum(y == 1))
    tp_total = int(np.sum(y == 0))
    triggered_fp = int(np.sum((y == 1) & trigger))
    triggered_tp = int(np.sum((y == 0) & trigger))
    trigger_n = int(np.sum(trigger))
    warning_precision = triggered_fp / trigger_n if trigger_n else math.nan
    return {
        "predicted_yes_n": int(len(y)),
        "fp_n": fp_total,
        "tp_n": tp_total,
        "base_fp_rate": base_rate,
        "trigger_n": trigger_n,
        "trigger_rate": trigger_n / max(1, len(y)),
        "triggered_fp": triggered_fp,
        "triggered_tp": triggered_tp,
        "warning_precision": warning_precision,
        "relative_precision_gain": warning_precision / base_rate if base_rate and math.isfinite(base_rate) else math.nan,
        "fp_recall": triggered_fp / fp_total if fp_total else math.nan,
        "tp_damage": triggered_tp / tp_total if tp_total else math.nan,
    }


def _random_warning_metrics(y: np.ndarray, n_trigger: int, repeats: int, seed: int) -> dict[str, Any]:
    if n_trigger <= 0 or len(y) == 0:
        return _warning_metrics(y, np.zeros(len(y), dtype=bool), float(np.mean(y)) if len(y) else math.nan)
    rng = np.random.default_rng(seed)
    metrics = []
    n_trigger = min(n_trigger, len(y))
    for _ in range(repeats):
        trigger = np.zeros(len(y), dtype=bool)
        trigger[rng.choice(len(y), size=n_trigger, replace=False)] = True
        metrics.append(_warning_metrics(y, trigger, float(np.mean(y))))
    keys = [
        "trigger_rate",
        "warning_precision",
        "relative_precision_gain",
        "fp_recall",
        "tp_damage",
    ]
    row = {
        "predicted_yes_n": int(len(y)),
        "fp_n": int(np.sum(y == 1)),
        "tp_n": int(np.sum(y == 0)),
        "base_fp_rate": float(np.mean(y)) if len(y) else math.nan,
        "trigger_n": n_trigger,
        "triggered_fp": math.nan,
        "triggered_tp": math.nan,
    }
    for key in keys:
        values = np.array([metric[key] for metric in metrics], dtype=float)
        row[key] = float(np.nanmean(values))
    return row


def _ece_10(y: np.ndarray, probs: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            keep = (probs >= lo) & (probs <= hi)
        else:
            keep = (probs >= lo) & (probs < hi)
        if keep.any():
            ece += float(np.mean(keep) * abs(np.mean(probs[keep]) - np.mean(y[keep])))
    return ece


def _feature_audit_rows(
    layer: int,
    metadata: list[dict[str, Any]],
    diff: np.ndarray,
    svd_basis: np.ndarray,
    pca_diff_basis: np.ndarray,
    tail_band: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = [
        {
            "layer": layer,
            "artifact": "hidden_states",
            "n_samples": len(metadata),
            "feature_dim": diff.shape[1],
            "available": True,
            "notes": "z_img, z_blind, and diff are available from cached hidden-state tensors",
        },
        {
            "layer": layer,
            "artifact": "train_svd_basis",
            "n_samples": len(metadata),
            "feature_dim": svd_basis.shape[1],
            "available": svd_basis.shape[1] > 0,
            "notes": "basis fitted on train subset only",
        },
        {
            "layer": layer,
            "artifact": "train_pca_diff_basis",
            "n_samples": len(metadata),
            "feature_dim": pca_diff_basis.shape[1],
            "available": pca_diff_basis.shape[1] > 0,
            "notes": "centered PCA basis fitted on train subset only",
        },
    ]
    tail_start, tail_end = tail_band
    rows.append(
        {
            "layer": layer,
            "artifact": f"tail_{tail_start}_{tail_end}",
            "n_samples": len(metadata),
            "feature_dim": max(0, min(tail_end, svd_basis.shape[1]) - max(0, tail_start - 1)),
            "available": min(tail_end, svd_basis.shape[1]) > max(0, tail_start - 1),
            "notes": "tail is unavailable when train SVD rank is smaller than tail_start",
        }
    )
    return rows


def _bootstrap_main_table(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_df.empty:
        return pd.DataFrame()
    comparison_labels = {
        ("margin_plus_tail_diff", "yes_no_margin"): "margin+tail - margin",
        ("margin_plus_full_diff", "yes_no_margin"): "margin+full - margin",
        ("margin_plus_tail_diff", "margin_plus_full_diff"): "margin+tail - margin+full",
        ("margin_plus_top16_svd_diff", "yes_no_margin"): "margin+top16 - margin",
        ("margin_plus_tail_diff", "raw_diff"): "margin+tail - raw diff",
    }
    metric_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "warning_precision": "Warning Precision",
        "fp_recall": "FP Recall",
        "tp_damage": "TP Damage",
    }
    rows: list[dict[str, Any]] = []
    for row in bootstrap_df.itertuples(index=False):
        key = (str(row.method_a), str(row.method_b))
        comparison = comparison_labels.get(key, f"{row.method_a} - {row.method_b}")
        ci_low = float(row.delta_ci95_low)
        ci_high = float(row.delta_ci95_high)
        delta = float(row.delta_mean)
        excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
        higher_is_better = bool(row.higher_is_better)
        if not excludes_zero:
            significant = "no"
        elif higher_is_better and delta > 0:
            significant = "yes"
        elif (not higher_is_better) and delta < 0:
            significant = "better"
        else:
            significant = "worse"
        rows.append(
            {
                "comparison": comparison,
                "metric": metric_labels.get(str(row.metric), str(row.metric)),
                "delta": delta,
                "ci95": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "significant": significant,
                "higher_is_better": higher_is_better,
            }
        )
    order = {
        "margin+tail - margin": 0,
        "margin+full - margin": 1,
        "margin+tail - margin+full": 2,
        "margin+top16 - margin": 3,
        "margin+tail - raw diff": 4,
    }
    metric_order = {"AUROC": 0, "AUPRC": 1, "Warning Precision": 2, "FP Recall": 3, "TP Damage": 4}
    df = pd.DataFrame(rows)
    df["_comparison_order"] = df["comparison"].map(lambda value: order.get(value, 999))
    df["_metric_order"] = df["metric"].map(lambda value: metric_order.get(value, 999))
    return df.sort_values(["_comparison_order", "_metric_order"]).drop(columns=["_comparison_order", "_metric_order"])


def _trigger_curve_table(warning_df: pd.DataFrame) -> pd.DataFrame:
    if warning_df.empty:
        return pd.DataFrame()
    selected = {
        "yes_no_margin": "margin-only",
        "margin_plus_top16_svd_diff": "margin+top16",
        "margin_plus_tail_diff": "margin+tail",
        "margin_plus_full_diff": "margin+full",
        "tail_257_1024_diff": "tail-only",
    }
    rows: list[dict[str, Any]] = []
    score_rows = warning_df[
        (warning_df["gate"] == "score_top_rate")
        & (warning_df["method"].isin(selected))
    ].copy()
    for row in score_rows.itertuples(index=False):
        rows.append(
            {
                "method": selected[str(row.method)],
                "target_trigger_rate": float(row.target_trigger_rate),
                "actual_trigger_rate": float(row.trigger_rate),
                "warning_precision": float(row.warning_precision),
                "fp_recall": float(row.fp_recall),
                "tp_damage": float(row.tp_damage),
                "source_method": str(row.method),
            }
        )
    random_match = warning_df[
        (warning_df["gate"] == "same_trigger_random_mean")
        & (warning_df["method"] == "margin_plus_tail_diff")
    ].copy()
    for row in random_match.itertuples(index=False):
        rows.append(
            {
                "method": "random",
                "target_trigger_rate": float(row.target_trigger_rate),
                "actual_trigger_rate": float(row.trigger_rate),
                "warning_precision": float(row.warning_precision),
                "fp_recall": float(row.fp_recall),
                "tp_damage": float(row.tp_damage),
                "source_method": "same_trigger_as_margin_plus_tail",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    method_order = {
        "random": 0,
        "margin-only": 1,
        "margin+top16": 2,
        "margin+tail": 3,
        "margin+full": 4,
        "tail-only": 5,
    }
    df["_method_order"] = df["method"].map(lambda value: method_order.get(value, 999))
    return df.sort_values(["target_trigger_rate", "_method_order"]).drop(columns=["_method_order"])


def _speed_cost_table(baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = [
        ("yes_no_margin", "margin-only", "lowest", "No extra model pass; uses first-token logits."),
        ("margin_plus_top16_svd_diff", "margin+top16", "medium", "Requires cached or online blind-reference hidden state plus tiny projection/probe."),
        ("margin_plus_tail_diff", "margin+tail", "medium", "Requires blind-reference hidden state and 768D tail projection/probe."),
        ("margin_plus_full_diff", "margin+full", "medium", "Requires blind-reference hidden state and full 4096D diff probe."),
        ("tail_257_1024_diff", "tail-only", "medium", "Geometry-only tail detector; no output margin."),
    ]
    task_df = baseline_df[baseline_df.get("task", "") == "task_b_pred_yes_fp_vs_tp"].copy() if not baseline_df.empty else pd.DataFrame()
    for method, label, total_cost, note in selected:
        match = task_df[task_df["method"] == method] if not task_df.empty else pd.DataFrame()
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "method": label,
                "extra_forward": int(row.get("extra_blind_forward", 0)),
                "feature_dim": int(row.get("feature_dim", 0)),
                "projection_probe_ms_per_sample": float(row.get("detector_score_ms_per_sample", math.nan)),
                "probe_fit_seconds": float(row.get("fit_seconds", math.nan)),
                "total_cost_class": total_cost,
                "notes": note,
            }
        )
    rows.append(
        {
            "method": "VCD/ICD",
            "extra_forward": "extra distorted/blind decoding",
            "feature_dim": "",
            "projection_probe_ms_per_sample": "",
            "probe_fit_seconds": "",
            "total_cost_class": "high",
            "notes": "Downstream correction operator; substantially more expensive than a linear detector, so selective routing can be worthwhile.",
        }
    )
    return pd.DataFrame(rows)


def _render_summary_note(
    baseline_df: pd.DataFrame,
    warning_df: pd.DataFrame,
    dim_curve_df: pd.DataFrame,
    spectral_band_df: pd.DataFrame,
    pls_diagnostic_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    bootstrap_main_df: pd.DataFrame,
    trigger_curve_df: pd.DataFrame,
    speed_cost_df: pd.DataFrame,
    feature_audit_df: pd.DataFrame,
    paths: dict[str, Path],
    train_subset: str,
    calibration_subset: str,
    test_subset: str,
) -> str:
    lines = [
        "# Detector Experiment Prep",
        "",
        "## Protocol",
        "",
        f"- Strict subset-transfer: train `{train_subset}`, calibrate `{calibration_subset}`, test `{test_subset}`.",
        "- Task A: FP vs TN on ground-truth No samples.",
        "- Task B: predicted-Yes FP vs TP deployment risk.",
        "- Threshold policy for classification metrics: F1-optimal threshold selected on calibration only.",
        "",
        "## Files",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(["", "## Best Test Rows", ""])
    if not baseline_df.empty and "test_auroc" in baseline_df:
        cols = [
            "task",
            "layer",
            "method",
            "feature_dim",
            "test_auroc",
            "test_auprc",
            "f1",
            "mcc",
            "detector_score_ms_per_sample",
        ]
        best = (
            baseline_df.dropna(subset=["test_auroc"])
            .sort_values(["task", "test_auroc", "test_auprc"], ascending=[True, False, False])
            .groupby("task", as_index=False)
            .head(8)
        )
        lines.append(_markdown_table(best[cols]))
    else:
        lines.append("_No baseline rows were generated._")

    lines.extend(["", "## Deployment Warning Snapshot", ""])
    if not warning_df.empty:
        snapshot = warning_df[
            (warning_df["gate"] == "score_top_rate")
            & (np.isclose(warning_df["target_trigger_rate"], 0.2))
        ].copy()
        if not snapshot.empty:
            cols = [
                "layer",
                "method",
                "trigger_rate",
                "warning_precision",
                "relative_precision_gain",
                "fp_recall",
                "tp_damage",
            ]
            snapshot = snapshot.sort_values("warning_precision", ascending=False).head(12)
            lines.append(_markdown_table(snapshot[cols]))
        else:
            lines.append("_No 20% trigger rows were generated._")
    else:
        lines.append("_No deployment warning rows were generated._")

    lines.extend(["", "## Subspace Dimension Curve", ""])
    if not dim_curve_df.empty:
        cols = ["task", "layer", "method", "feature_dim", "test_auroc", "test_auprc"]
        preview = dim_curve_df.sort_values(["task", "test_auroc"], ascending=[True, False]).head(20)
        lines.append(_markdown_table(preview[cols]))
    else:
        lines.append("_Dimension curve was skipped or unavailable._")

    lines.extend(["", "## Spectral Band Curve", ""])
    if not spectral_band_df.empty:
        cols = [
            "task",
            "layer",
            "mode",
            "spectral_feature",
            "feature_dim",
            "test_auroc",
            "test_auprc",
            "warning_precision",
            "fp_recall",
            "tp_damage",
        ]
        preview = (
            spectral_band_df[
                (spectral_band_df["task"] == "task_b_pred_yes_fp_vs_tp")
                & (spectral_band_df["mode"].isin(["band_only", "cumulative_top_k", "margin_plus_band_only", "margin_plus_cumulative_top_k"]))
            ]
            .sort_values(["mode", "test_auroc"], ascending=[True, False])
            .head(24)
        )
        lines.append(_markdown_table(preview[cols]))
    else:
        lines.append("_No spectral band rows were generated._")

    lines.extend(["", "## PLS Diagnostics", ""])
    if not pls_diagnostic_df.empty:
        cols = [
            "task",
            "layer",
            "k",
            "train_auroc",
            "calibration_auroc",
            "test_auroc",
            "split_half_overlap",
            "overlap_top16",
            "overlap_tail_257_1024",
        ]
        preview = pls_diagnostic_df.sort_values(["task", "k"]).head(12)
        lines.append(_markdown_table(preview[cols]))
    else:
        lines.append("_No PLS diagnostic rows were generated._")

    lines.extend(["", "## Bootstrap Comparisons", ""])
    if not bootstrap_main_df.empty:
        cols = [
            "comparison",
            "metric",
            "delta",
            "ci95",
            "significant",
        ]
        preview = bootstrap_main_df.head(40)
        lines.append(_markdown_table(preview[cols]))
    else:
        lines.append("_No bootstrap comparison rows were generated._")

    lines.extend(["", "## Trigger Curve Table", ""])
    if not trigger_curve_df.empty:
        cols = [
            "method",
            "target_trigger_rate",
            "actual_trigger_rate",
            "warning_precision",
            "fp_recall",
            "tp_damage",
        ]
        lines.append(_markdown_table(trigger_curve_df[cols]))
    else:
        lines.append("_No trigger curve rows were generated._")

    lines.extend(["", "## Speed And Cost", ""])
    if not speed_cost_df.empty:
        lines.append(_markdown_table(speed_cost_df))
    else:
        lines.append("_No speed/cost rows were generated._")

    lines.extend(["", "## Sanity Checks And Interpretation", ""])
    lines.extend(_sanity_check_lines(baseline_df, warning_df, dim_curve_df))

    lines.extend(["", "## Artifact Audit", ""])
    if not feature_audit_df.empty:
        lines.append(_markdown_table(feature_audit_df))

    return "\n".join(lines) + "\n"


def _sanity_check_lines(
    baseline_df: pd.DataFrame,
    warning_df: pd.DataFrame,
    dim_curve_df: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    for task in ["task_a_fp_vs_tn", "task_b_pred_yes_fp_vs_tp"]:
        raw_base = _first_row(baseline_df, task=task, method="raw_diff")
        raw_dim = _first_row(dim_curve_df, task=task, method="raw_full_diff_reference")
        if raw_base is not None and raw_dim is not None:
            lines.append(
                "- "
                f"`{task}` raw check: baseline `raw_diff` AUROC/AUPRC = "
                f"{_fmt(raw_base.get('test_auroc'))}/{_fmt(raw_base.get('test_auprc'))}; "
                f"dimension `raw_full_diff_reference` = "
                f"{_fmt(raw_dim.get('test_auroc'))}/{_fmt(raw_dim.get('test_auprc'))}. "
                "They use the same StandardScaler + logistic grid protocol."
            )

    pls_a = _first_row(baseline_df, task="task_a_fp_vs_tn", method="pls32_diff")
    pls_b = _first_row(baseline_df, task="task_b_pred_yes_fp_vs_tp", method="pls32_diff")
    if pls_a is not None:
        lines.append(
            "- "
            "PLS transfer check: Task A `pls32_diff` train/calibration/test AUROC = "
            f"{_fmt(pls_a.get('train_auroc'))}/{_fmt(pls_a.get('calibration_auroc'))}/{_fmt(pls_a.get('test_auroc'))}; "
            "this indicates substantial strict-split domain shift."
        )
    if pls_b is not None:
        lines.append(
            "- "
            "PLS deployment check: Task B `pls32_diff` train/calibration/test AUROC = "
            f"{_fmt(pls_b.get('train_auroc'))}/{_fmt(pls_b.get('calibration_auroc'))}/{_fmt(pls_b.get('test_auroc'))}; "
            "it transfers modestly but is not a stable strongest detector."
        )

    for task in ["task_a_fp_vs_tn", "task_b_pred_yes_fp_vs_tp"]:
        top4 = _first_dim_row(dim_curve_df, task, "top_svd", 4)
        top16 = _first_dim_row(dim_curve_df, task, "top_svd", 16)
        if top4 is not None and top16 is not None:
            lines.append(
                "- "
                f"`{task}` top-SVD check: top-4 AUROC = {_fmt(top4.get('test_auroc'))}, "
                f"top-16 AUROC = {_fmt(top16.get('test_auroc'))}. "
                "The precise claim should be that the dominant top-4 directions are weak, while useful signal can appear in slightly deeper early spectral coordinates."
            )

    raw_warning = _first_warning_row(warning_df, "raw_diff", 0.2)
    tail_warning = _first_warning_row(warning_df, "tail_257_1024_diff", 0.2)
    if raw_warning is not None and tail_warning is not None:
        lines.append(
            "- "
            "Warning-vs-AUROC check: at the 20% predicted-Yes trigger target, "
            f"`raw_diff` precision/FP recall = {_fmt(raw_warning.get('warning_precision'))}/{_fmt(raw_warning.get('fp_recall'))}, "
            f"while tail-only = {_fmt(tail_warning.get('warning_precision'))}/{_fmt(tail_warning.get('fp_recall'))}. "
            "Fixed-trigger warning can look better than global AUROC because it evaluates only the top-risk slice."
        )

    if not lines:
        return ["_No sanity-check rows were available._"]
    lines.append(
        "- "
        "Recommended wording: margin/output confidence remains the strongest simple baseline; geometry-only is strict-transfer fragile, but selected spectral coordinates and margin+geometry provide complementary predicted-Yes warning signal."
    )
    return lines


def _first_row(df: pd.DataFrame, task: str, method: str) -> pd.Series | None:
    if df.empty or "task" not in df or "method" not in df:
        return None
    match = df[(df["task"] == task) & (df["method"] == method)]
    return None if match.empty else match.iloc[0]


def _first_dim_row(df: pd.DataFrame, task: str, method: str, feature_dim: int) -> pd.Series | None:
    if df.empty or "task" not in df or "method" not in df or "feature_dim" not in df:
        return None
    match = df[(df["task"] == task) & (df["method"] == method) & (df["feature_dim"] == feature_dim)]
    return None if match.empty else match.iloc[0]


def _first_warning_row(df: pd.DataFrame, method: str, target_rate: float) -> pd.Series | None:
    if df.empty or "method" not in df or "target_trigger_rate" not in df or "gate" not in df:
        return None
    match = df[
        (df["method"] == method)
        & (df["gate"] == "score_top_rate")
        & np.isclose(df["target_trigger_rate"], target_rate)
    ]
    return None if match.empty else match.iloc[0]


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(numeric) else f"{numeric:.3f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Empty._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    headers = [str(col) for col in formatted.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in formatted.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _ordered_dataframe(rows: list[dict[str, Any]], method_order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "method" not in df:
        return df
    order = {method: idx for idx, method in enumerate(method_order)}
    df["_method_order"] = df["method"].map(lambda method: order.get(method, 10_000))
    df = df.sort_values(["task", "layer", "_method_order", "method"]).drop(columns=["_method_order"])
    return df


def _load_margin_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        return {}
    return {str(row.sample_id): row._asdict() for row in df.itertuples(index=False)}


def _require_alignment(sample_ids: list[str], rows_by_id: dict[str, Any], layer: int) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise ValueError(f"Layer {layer} hidden states contain unknown sample_id(s): {missing[:5]}")


def _binary_yes_probability(yes_logit: float, no_logit: float) -> float:
    if not math.isfinite(yes_logit) or not math.isfinite(no_logit):
        return math.nan
    max_logit = max(yes_logit, no_logit)
    yes_exp = math.exp(yes_logit - max_logit)
    no_exp = math.exp(no_logit - max_logit)
    return yes_exp / (yes_exp + no_exp)


def _safe_metric(y: np.ndarray, scores: np.ndarray, metric: Callable[[Any, Any], float]) -> float:
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return math.nan
    return float(metric(y, scores))


def _nan_to_sortable(value: float) -> float:
    return -math.inf if not math.isfinite(value) else value


def _has_two_classes(y: np.ndarray) -> bool:
    return len(y) > 0 and len(set(y.tolist())) >= 2


def _maybe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
