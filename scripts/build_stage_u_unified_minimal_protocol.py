#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_json


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    display_name: str
    root: Path
    readout: str
    include_in_main: bool = True
    diagnostic_only: bool = False

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions" / "pope_predictions.jsonl"

    @property
    def hidden_dir(self) -> Path:
        return self.root / "hidden_states"

    @property
    def svd_dir(self) -> Path:
        return self.root / "svd"

    @property
    def margins_path(self) -> Path:
        return self.root / "margins" / "pope_margin_scores.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the unified cross-model minimal protocol tables."
    )
    parser.add_argument("--output-dir", default="outputs/stage_u_unified_minimal_protocol")
    parser.add_argument("--notes-path", default="notes/stage_u_unified_minimal_protocol.md")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--tail-start", type=int, default=257)
    parser.add_argument("--tail-end", type=int, default=1024)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "output_dir": args.output_dir,
        "notes_path": args.notes_path,
        "split_dir": args.split_dir,
        "target_rates": args.target_rates,
        "tail_band": [args.tail_start, args.tail_end],
    }
    if not args.dry_run:
        payload.update(
            build_unified_protocol(
                output_dir=Path(args.output_dir),
                notes_path=Path(args.notes_path),
                split_dir=Path(args.split_dir),
                target_rates=args.target_rates,
                tail_band=(args.tail_start, args.tail_end),
                random_repeats=args.random_repeats,
                seed=args.seed,
                max_iter=args.max_iter,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "build_stage_u_unified_minimal_protocol_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "build_stage_u_unified_minimal_protocol",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


def build_unified_protocol(
    output_dir: Path,
    notes_path: Path,
    split_dir: Path,
    target_rates: list[float],
    tail_band: tuple[int, int],
    random_repeats: int,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    split_by_id = _load_split_map(split_dir)
    specs = _default_model_specs()
    available_specs = [spec for spec in specs if _spec_available(spec)]

    mechanism_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    layer_sweep_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for spec in specs:
        if spec not in available_specs:
            missing_rows.append(_missing_row(spec))
            continue

        rows = _read_jsonl(spec.predictions_path)
        rows_by_id = {str(row["sample_id"]): row for row in rows}
        margins = _load_margins(spec.margins_path)
        if spec.include_in_main and not margins:
            missing_rows.append(
                {
                    "alias": spec.alias,
                    "display_name": spec.display_name,
                    "readout": spec.readout,
                    "status": "optional_missing",
                    "reason": f"margin logits unavailable: {spec.margins_path}",
                }
            )
        layers = _available_layers(spec.hidden_dir)
        layer_cache: dict[int, dict[str, Any]] = {}

        for layer in layers:
            try:
                layer_result = _analyze_layer(
                    spec=spec,
                    layer=layer,
                    rows_by_id=rows_by_id,
                    margins=margins,
                    split_by_id=split_by_id,
                    tail_band=tail_band,
                    seed=seed + layer,
                    max_iter=max_iter,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                missing_rows.append(
                    {
                        "alias": spec.alias,
                        "display_name": spec.display_name,
                        "readout": spec.readout,
                        "layer": layer,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                continue
            layer_cache[layer] = layer_result
            mechanism_rows.append(layer_result["mechanism_row"])
            layer_sweep_rows.append(layer_result["layer_sweep_row"])

        if not layer_cache:
            continue
        selected_layer = _select_layer(layer_cache)
        selected = layer_cache[selected_layer]
        deployment_rows.append(selected["deployment_row"])
        gate_rows.extend(
            _gate_rows_for_layer(
                spec=spec,
                layer=selected_layer,
                sample_ids=selected["sample_ids"],
                metadata=selected["metadata"],
                split_by_id=split_by_id,
                scores=selected["deployment_scores"],
                target_rates=target_rates,
                random_repeats=random_repeats,
                seed=seed + selected_layer,
            )
        )
        distribution_rows.extend(
            _predicted_yes_distribution_rows(
                spec=spec,
                layer=selected_layer,
                metadata=selected["metadata"],
                scores=selected["deployment_scores"],
            )
        )
        failure_rows.append(
            _failure_row(
                spec=spec,
                layer=selected_layer,
                layer_result=selected,
            )
        )

        if _needs_failure_controls(spec):
            for layer, result in layer_cache.items():
                shuffle_rows.extend(
                    _shuffle_control_rows(
                        spec=spec,
                        layer=layer,
                        rows_by_id=rows_by_id,
                        margins=margins,
                        split_by_id=split_by_id,
                        tail_band=tail_band,
                        seed=seed + 1000 + layer,
                        max_iter=max_iter,
                    )
                )

    mechanism_df = pd.DataFrame(mechanism_rows)
    deployment_df = pd.DataFrame(deployment_rows)
    gate_df = pd.DataFrame(gate_rows)
    layer_sweep_df = pd.DataFrame(layer_sweep_rows)
    failure_df = pd.DataFrame(failure_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    shuffle_df = pd.DataFrame(shuffle_rows)
    missing_df = pd.DataFrame(missing_rows)

    mechanism_summary_df = _model_mechanism_summary(mechanism_df)
    paths = {
        "mechanism_layer_metrics": out / "mechanism_layer_metrics.csv",
        "mechanism_model_summary": out / "mechanism_model_summary.csv",
        "deployment_model_summary": out / "deployment_model_summary.csv",
        "deployment_gate_metrics": out / "deployment_gate_metrics.csv",
        "layer_deployment_sweep": out / "layer_deployment_sweep.csv",
        "failure_diagnosis": out / "failure_diagnosis.csv",
        "predicted_yes_score_distributions": out / "predicted_yes_score_distributions.csv",
        "shuffle_controls": out / "shuffle_controls.csv",
        "missing_artifacts": out / "missing_artifacts.csv",
    }
    _write_df(mechanism_df, paths["mechanism_layer_metrics"])
    _write_df(mechanism_summary_df, paths["mechanism_model_summary"])
    _write_df(deployment_df, paths["deployment_model_summary"])
    _write_df(gate_df, paths["deployment_gate_metrics"])
    _write_df(layer_sweep_df, paths["layer_deployment_sweep"])
    _write_df(failure_df, paths["failure_diagnosis"])
    _write_df(distribution_df, paths["predicted_yes_score_distributions"])
    _write_df(shuffle_df, paths["shuffle_controls"])
    _write_df(missing_df, paths["missing_artifacts"])

    note = _render_note(
        mechanism_summary_df=mechanism_summary_df,
        deployment_df=deployment_df,
        gate_df=gate_df,
        failure_df=failure_df,
        distribution_df=distribution_df,
        shuffle_df=shuffle_df,
        missing_df=missing_df,
        paths=paths,
    )
    ensure_dir(notes_path.parent)
    notes_path.write_text(note, encoding="utf-8")
    summary_md = out / "unified_minimal_protocol.md"
    summary_md.write_text(note, encoding="utf-8")

    return {
        "num_specs": len(specs),
        "num_available_specs": len(available_specs),
        "num_mechanism_rows": len(mechanism_df),
        "num_gate_rows": len(gate_df),
        "paths": {name: str(path) for name, path in paths.items()},
        "notes_path": str(notes_path),
        "summary_markdown_path": str(summary_md),
    }


def _default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            alias="llava_1_5_7b",
            display_name="LLaVA-1.5-7B",
            root=Path("outputs"),
            readout="last_prompt_token",
        ),
        ModelSpec(
            alias="llava_13b",
            display_name="LLaVA-1.5-13B",
            root=Path("outputs/stage_o_cross_model/llava_13b"),
            readout="last_prompt_token",
        ),
        ModelSpec(
            alias="qwen2_vl_7b",
            display_name="Qwen2-VL-7B",
            root=Path("outputs/stage_o_cross_model_user_readout/qwen2_vl_7b"),
            readout="last_user_content_token",
        ),
        ModelSpec(
            alias="qwen2_5_vl_7b",
            display_name="Qwen2.5-VL-7B",
            root=Path("outputs/stage_o_cross_model_user_readout/qwen2_5_vl_7b"),
            readout="last_user_content_token",
        ),
        ModelSpec(
            alias="internvl2_8b",
            display_name="InternVL2-8B",
            root=Path("outputs/stage_o_cross_model_user_readout/internvl2_8b"),
            readout="last_user_content_token",
        ),
        ModelSpec(
            alias="internvl2_5_8b",
            display_name="InternVL2.5-8B",
            root=Path("outputs/stage_o_cross_model_user_readout/internvl2_5_8b"),
            readout="last_user_content_token",
        ),
        ModelSpec(
            alias="internvl2_8b_question",
            display_name="InternVL2-8B question readout",
            root=Path("outputs/stage_o_cross_model_question_readout/internvl2_8b"),
            readout="last_question_token",
            include_in_main=False,
            diagnostic_only=True,
        ),
        ModelSpec(
            alias="internvl2_5_8b_question",
            display_name="InternVL2.5-8B question readout",
            root=Path("outputs/stage_o_cross_model_question_readout/internvl2_5_8b"),
            readout="last_question_token",
            include_in_main=False,
            diagnostic_only=True,
        ),
    ]


def _spec_available(spec: ModelSpec) -> bool:
    return spec.predictions_path.exists() and spec.hidden_dir.exists() and spec.svd_dir.exists()


def _missing_row(spec: ModelSpec) -> dict[str, Any]:
    missing = []
    for name, path in [
        ("predictions", spec.predictions_path),
        ("hidden_states", spec.hidden_dir),
        ("svd", spec.svd_dir),
    ]:
        if not path.exists():
            missing.append(f"{name}:{path}")
    return {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "status": "missing",
        "reason": "; ".join(missing),
    }


def _analyze_layer(
    spec: ModelSpec,
    layer: int,
    rows_by_id: dict[str, dict[str, Any]],
    margins: dict[str, dict[str, float]],
    split_by_id: dict[str, str],
    tail_band: tuple[int, int],
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        hidden = torch.load(
            spec.hidden_dir / f"layer_{layer}.pt",
            map_location="cpu",
            weights_only=False,
        )
        svd = torch.load(
            spec.svd_dir / f"svd_layer_{layer}.pt",
            map_location="cpu",
            weights_only=False,
        )
    sample_ids = [str(item) for item in hidden["sample_ids"]]
    metadata = [
        _metadata_row(rows_by_id[sample_id], margins.get(sample_id, {}), split_by_id)
        for sample_id in sample_ids
    ]
    z_img = hidden["z_img"].float().numpy()
    z_blind = hidden["z_blind"].float().numpy()
    diff = z_blind - z_img
    vh = svd["Vh"].float().numpy()
    singular_values = svd["singular_values"].float().numpy()

    train_idx, y_train = _label_indices(metadata, "train", {"FP": 1, "TN": 0})
    val_idx, y_val = _label_indices(metadata, "val", {"FP": 1, "TN": 0})
    test_idx, y_test = _label_indices(metadata, "test", {"FP": 1, "TN": 0})
    if len(train_idx) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError(f"{spec.alias} L{layer} has insufficient FP/TN train labels.")

    full_scores, full_train = _fit_scores(diff, train_idx, y_train, seed, max_iter)
    raw_img_scores, _ = _fit_scores(z_img, train_idx, y_train, seed + 1, max_iter)
    raw_blind_scores, _ = _fit_scores(z_blind, train_idx, y_train, seed + 2, max_iter)

    top_metrics: dict[int, dict[str, Any]] = {}
    for k in [4, 64, 256]:
        effective_k = min(k, vh.shape[0])
        x = diff @ vh[:effective_k].T
        scores, _ = _fit_scores(x, train_idx, y_train, seed + k, max_iter)
        top_metrics[k] = {
            "effective_k": effective_k,
            "explained_variance": _explained_variance(singular_values, effective_k),
            "scores": scores,
            "fp_tn_test_auroc": _auroc(y_test, scores[test_idx]),
        }

    tail_start, tail_end = tail_band
    tail_start_idx = max(0, tail_start - 1)
    tail_end_idx = min(tail_end, vh.shape[0])
    tail_effective_dim = max(0, tail_end_idx - tail_start_idx)
    if tail_effective_dim:
        tail_x = diff @ vh[tail_start_idx:tail_end_idx].T
        tail_scores, _ = _fit_scores(tail_x, train_idx, y_train, seed + 17, max_iter)
        tail_auroc = _auroc(y_test, tail_scores[test_idx])
    else:
        tail_scores = np.full(len(sample_ids), np.nan)
        tail_auroc = math.nan

    low_margin = _score_array(metadata, "low_yes_margin")
    entropy = _score_array(metadata, "binary_entropy")

    py_test_idx, y_py_test = _label_indices(metadata, "test", {"FP": 1, "TP": 0})
    py_val_idx, y_py_val = _label_indices(metadata, "val", {"FP": 1, "TP": 0})
    pred_yes_geometry_auroc = _auroc(y_py_test, full_scores[py_test_idx])
    low_margin_auroc = _auroc(y_py_test, low_margin[py_test_idx])
    entropy_auroc = _auroc(y_py_test, entropy[py_test_idx])
    low_margin_plus = _combine_scores(full_scores, low_margin, py_val_idx)
    entropy_plus = _combine_scores(full_scores, entropy, py_val_idx)

    mechanism_row = {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "include_in_main": spec.include_in_main,
        "diagnostic_only": spec.diagnostic_only,
        "layer": layer,
        "selection_val_full_diff_auroc": _auroc(y_val, full_scores[val_idx]),
        "top4_explained_variance": top_metrics[4]["explained_variance"],
        "top4_auroc": top_metrics[4]["fp_tn_test_auroc"],
        "top64_auroc": top_metrics[64]["fp_tn_test_auroc"],
        "top256_auroc": top_metrics[256]["fp_tn_test_auroc"],
        "full_diff_auroc": _auroc(y_test, full_scores[test_idx]),
        "tail_257_1024_auroc": tail_auroc,
        "tail_effective_dim": tail_effective_dim,
        "fp_tn_test_n": int(len(y_test)),
        "fp_tn_test_positive_n": int(np.sum(y_test)),
        "source_root": str(spec.root),
    }
    deployment_row = {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "include_in_main": spec.include_in_main,
        "diagnostic_only": spec.diagnostic_only,
        "selected_layer": layer,
        "selection_val_full_diff_auroc": mechanism_row["selection_val_full_diff_auroc"],
        "predicted_yes_test_n": int(len(y_py_test)),
        "predicted_yes_test_fp_n": int(np.sum(y_py_test)),
        "predicted_yes_base_fp_rate": float(np.mean(y_py_test)) if len(y_py_test) else math.nan,
        "geometry_full_auroc": pred_yes_geometry_auroc,
        "low_margin_auroc": low_margin_auroc,
        "entropy_auroc": entropy_auroc,
        "best_margin_entropy_auroc": _nanmax([low_margin_auroc, entropy_auroc]),
        "low_margin_plus_geometry_auroc": _auroc(y_py_test, low_margin_plus[py_test_idx]),
        "entropy_plus_geometry_auroc": _auroc(y_py_test, entropy_plus[py_test_idx]),
        "margin_available": bool(margins),
        "source_root": str(spec.root),
    }
    layer_sweep_row = {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "layer": layer,
        "fp_tn_full_auroc": mechanism_row["full_diff_auroc"],
        "fp_tn_raw_img_auroc": _auroc(y_test, raw_img_scores[test_idx]),
        "fp_tn_raw_blind_auroc": _auroc(y_test, raw_blind_scores[test_idx]),
        "predicted_yes_full_auroc": pred_yes_geometry_auroc,
        "predicted_yes_low_margin_auroc": low_margin_auroc,
        "predicted_yes_entropy_auroc": entropy_auroc,
        "predicted_yes_n": int(len(y_py_test)),
        "predicted_yes_fp_n": int(np.sum(y_py_test)),
    }

    return {
        "sample_ids": sample_ids,
        "metadata": metadata,
        "z_img": z_img,
        "z_blind": z_blind,
        "diff": diff,
        "vh": vh,
        "train_idx": train_idx,
        "y_train": y_train,
        "val_idx": val_idx,
        "y_val": y_val,
        "test_idx": test_idx,
        "y_test": y_test,
        "py_val_idx": py_val_idx,
        "y_py_val": y_py_val,
        "py_test_idx": py_test_idx,
        "y_py_test": y_py_test,
        "full_scores": full_scores,
        "raw_img_scores": raw_img_scores,
        "raw_blind_scores": raw_blind_scores,
        "tail_scores": tail_scores,
        "low_margin": low_margin,
        "entropy": entropy,
        "deployment_scores": {
            "geometry_full": full_scores,
            "low_margin": low_margin,
            "entropy": entropy,
            "low_margin_plus_geometry": low_margin_plus,
            "entropy_plus_geometry": entropy_plus,
        },
        "mechanism_row": mechanism_row,
        "deployment_row": deployment_row,
        "layer_sweep_row": layer_sweep_row,
    }


def _select_layer(layer_cache: dict[int, dict[str, Any]]) -> int:
    candidates = []
    for layer, result in layer_cache.items():
        value = result["mechanism_row"]["selection_val_full_diff_auroc"]
        candidates.append((float(value) if pd.notna(value) else -math.inf, layer))
    return max(candidates)[1]


def _model_mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for alias, group in df.groupby("alias", sort=False):
        ranked = group.copy()
        ranked["_select"] = ranked["selection_val_full_diff_auroc"].fillna(-math.inf)
        best = ranked.sort_values(["_select", "layer"], ascending=[False, True]).iloc[0].drop(labels=["_select"])
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def _gate_rows_for_layer(
    spec: ModelSpec,
    layer: int,
    sample_ids: list[str],
    metadata: list[dict[str, Any]],
    split_by_id: dict[str, str],
    scores: dict[str, np.ndarray],
    target_rates: list[float],
    random_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    del sample_ids, split_by_id
    rows: list[dict[str, Any]] = []
    for score_name in [
        "geometry_full",
        "low_margin",
        "entropy",
        "low_margin_plus_geometry",
        "entropy_plus_geometry",
    ]:
        values = scores[score_name]
        if not np.isfinite(values).any():
            continue
        for target_rate in target_rates:
            threshold = _calibrate_threshold(values, metadata, "val", target_rate)
            gate_mask = _gate_mask(values, metadata, "test", threshold)
            row = _gate_metric_row(
                spec=spec,
                layer=layer,
                gate=score_name,
                matched_gate=score_name,
                target_rate=target_rate,
                values=values,
                metadata=metadata,
                threshold=threshold,
                gate_mask=gate_mask,
            )
            rows.append(row)
            rows.append(
                _random_gate_metric_row(
                    spec=spec,
                    layer=layer,
                    matched_gate=score_name,
                    target_rate=target_rate,
                    metadata=metadata,
                    n_trigger=int(row["trigger_n"]),
                    repeats=random_repeats,
                    seed=seed + int(round(target_rate * 1000)) + len(rows),
                )
            )
    return rows


def _gate_metric_row(
    spec: ModelSpec,
    layer: int,
    gate: str,
    matched_gate: str,
    target_rate: float,
    values: np.ndarray,
    metadata: list[dict[str, Any]],
    threshold: float,
    gate_mask: np.ndarray,
) -> dict[str, Any]:
    del values
    pred_yes_mask = _predicted_yes_mask(metadata, "test")
    pred_yes_y = np.array(
        [1 if row["outcome"] == "FP" else 0 for row, keep in zip(metadata, pred_yes_mask) if keep],
        dtype=np.int64,
    )
    trigger_y = np.array(
        [1 if row["outcome"] == "FP" else 0 for row, keep in zip(metadata, gate_mask) if keep],
        dtype=np.int64,
    )
    trigger_n = int(len(trigger_y))
    fp_total = int(np.sum(pred_yes_y))
    tp_total = int(len(pred_yes_y) - fp_total)
    triggered_fp = int(np.sum(trigger_y))
    triggered_tp = int(trigger_n - triggered_fp)
    return {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "include_in_main": spec.include_in_main,
        "diagnostic_only": spec.diagnostic_only,
        "layer": layer,
        "gate": gate,
        "matched_gate": matched_gate,
        "gate_family": "score_gate",
        "target_trigger_rate_predicted_yes": target_rate,
        "threshold": threshold,
        "predicted_yes_n": int(len(pred_yes_y)),
        "predicted_yes_fp_n": fp_total,
        "predicted_yes_tp_n": tp_total,
        "trigger_n": trigger_n,
        "trigger_rate_predicted_yes": trigger_n / len(pred_yes_y) if len(pred_yes_y) else math.nan,
        "triggered_fp": triggered_fp,
        "triggered_tp": triggered_tp,
        "precision_fp": triggered_fp / trigger_n if trigger_n else math.nan,
        "fp_recall": triggered_fp / fp_total if fp_total else math.nan,
        "tp_damage": triggered_tp / tp_total if tp_total else math.nan,
    }


def _random_gate_metric_row(
    spec: ModelSpec,
    layer: int,
    matched_gate: str,
    target_rate: float,
    metadata: list[dict[str, Any]],
    n_trigger: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    pred_yes_indices = np.flatnonzero(_predicted_yes_mask(metadata, "test"))
    pred_yes_outcomes = [metadata[idx]["outcome"] for idx in pred_yes_indices]
    fp_total = sum(outcome == "FP" for outcome in pred_yes_outcomes)
    tp_total = sum(outcome == "TP" for outcome in pred_yes_outcomes)
    if len(pred_yes_indices) == 0 or n_trigger <= 0:
        return {
            "alias": spec.alias,
            "display_name": spec.display_name,
            "readout": spec.readout,
            "include_in_main": spec.include_in_main,
            "diagnostic_only": spec.diagnostic_only,
            "layer": layer,
            "gate": "same_trigger_random",
            "matched_gate": matched_gate,
            "gate_family": "same_trigger_random",
            "target_trigger_rate_predicted_yes": target_rate,
            "threshold": math.nan,
            "predicted_yes_n": int(len(pred_yes_indices)),
            "predicted_yes_fp_n": fp_total,
            "predicted_yes_tp_n": tp_total,
            "trigger_n": 0,
            "trigger_rate_predicted_yes": math.nan,
            "triggered_fp": math.nan,
            "triggered_tp": math.nan,
            "precision_fp": math.nan,
            "fp_recall": math.nan,
            "tp_damage": math.nan,
        }
    rng = np.random.default_rng(seed)
    n_trigger = min(n_trigger, len(pred_yes_indices))
    metrics = []
    for _ in range(repeats):
        chosen = rng.choice(pred_yes_indices, size=n_trigger, replace=False)
        outcomes = [metadata[idx]["outcome"] for idx in chosen]
        triggered_fp = sum(outcome == "FP" for outcome in outcomes)
        triggered_tp = sum(outcome == "TP" for outcome in outcomes)
        metrics.append(
            {
                "precision_fp": triggered_fp / n_trigger,
                "fp_recall": triggered_fp / fp_total if fp_total else math.nan,
                "tp_damage": triggered_tp / tp_total if tp_total else math.nan,
                "triggered_fp": triggered_fp,
                "triggered_tp": triggered_tp,
            }
        )
    return {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "include_in_main": spec.include_in_main,
        "diagnostic_only": spec.diagnostic_only,
        "layer": layer,
        "gate": "same_trigger_random",
        "matched_gate": matched_gate,
        "gate_family": "same_trigger_random",
        "target_trigger_rate_predicted_yes": target_rate,
        "threshold": math.nan,
        "predicted_yes_n": int(len(pred_yes_indices)),
        "predicted_yes_fp_n": fp_total,
        "predicted_yes_tp_n": tp_total,
        "trigger_n": n_trigger,
        "trigger_rate_predicted_yes": n_trigger / len(pred_yes_indices),
        "triggered_fp": _nanmean([row["triggered_fp"] for row in metrics]),
        "triggered_tp": _nanmean([row["triggered_tp"] for row in metrics]),
        "precision_fp": _nanmean([row["precision_fp"] for row in metrics]),
        "fp_recall": _nanmean([row["fp_recall"] for row in metrics]),
        "tp_damage": _nanmean([row["tp_damage"] for row in metrics]),
    }


def _predicted_yes_distribution_rows(
    spec: ModelSpec,
    layer: int,
    metadata: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for score_name, values in scores.items():
        for outcome in ["FP", "TP"]:
            mask = np.array(
                [row["protocol_split"] == "test" and row["outcome"] == outcome for row in metadata],
                dtype=bool,
            )
            vals = values[mask]
            vals = vals[np.isfinite(vals)]
            rows.append(
                {
                    "alias": spec.alias,
                    "display_name": spec.display_name,
                    "readout": spec.readout,
                    "layer": layer,
                    "score": score_name,
                    "outcome": outcome,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else math.nan,
                    "std": float(np.std(vals)) if len(vals) else math.nan,
                    "q10": _quantile(vals, 0.10),
                    "q25": _quantile(vals, 0.25),
                    "median": _quantile(vals, 0.50),
                    "q75": _quantile(vals, 0.75),
                    "q90": _quantile(vals, 0.90),
                }
            )
    return rows


def _failure_row(
    spec: ModelSpec,
    layer: int,
    layer_result: dict[str, Any],
) -> dict[str, Any]:
    metadata = layer_result["metadata"]
    test_idx = layer_result["test_idx"]
    y_test = layer_result["y_test"]
    py_test_idx = layer_result["py_test_idx"]
    y_py_test = layer_result["y_py_test"]
    full = layer_result["full_scores"]
    raw_img = layer_result["raw_img_scores"]
    raw_blind = layer_result["raw_blind_scores"]
    margin = _score_array(metadata, "yes_minus_no_logit")
    low_margin = layer_result["low_margin"]
    entropy = layer_result["entropy"]
    fp_tn_full = _auroc(y_test, full[test_idx])
    pred_yes_full = _auroc(y_py_test, full[py_test_idx])
    return {
        "alias": spec.alias,
        "display_name": spec.display_name,
        "readout": spec.readout,
        "diagnostic_only": spec.diagnostic_only,
        "selected_layer": layer,
        "fp_tn_full_auroc": fp_tn_full,
        "fp_tn_raw_img_auroc": _auroc(y_test, raw_img[test_idx]),
        "fp_tn_raw_blind_auroc": _auroc(y_test, raw_blind[test_idx]),
        "fp_tn_yes_margin_auroc": _auroc(y_test, margin[test_idx]),
        "fp_tn_low_margin_auroc": _auroc(y_test, low_margin[test_idx]),
        "fp_tn_entropy_auroc": _auroc(y_test, entropy[test_idx]),
        "predicted_yes_full_auroc": pred_yes_full,
        "predicted_yes_low_margin_auroc": _auroc(y_py_test, low_margin[py_test_idx]),
        "predicted_yes_entropy_auroc": _auroc(y_py_test, entropy[py_test_idx]),
        "fp_tn_full_vs_yes_margin_pearson": _corr(full[test_idx], margin[test_idx]),
        "fp_tn_full_vs_yes_margin_spearman": _rank_corr(full[test_idx], margin[test_idx]),
        "predicted_yes_full_vs_yes_margin_pearson": _corr(full[py_test_idx], margin[py_test_idx]),
        "predicted_yes_full_vs_yes_margin_spearman": _rank_corr(full[py_test_idx], margin[py_test_idx]),
        "gap_fp_tn_minus_predicted_yes_auroc": (
            fp_tn_full - pred_yes_full
            if pd.notna(fp_tn_full) and pd.notna(pred_yes_full)
            else math.nan
        ),
        "interpretation_flag": _interpretation_flag(fp_tn_full, pred_yes_full),
    }


def _needs_failure_controls(spec: ModelSpec) -> bool:
    return spec.alias.startswith("internvl")


def _shuffle_control_rows(
    spec: ModelSpec,
    layer: int,
    rows_by_id: dict[str, dict[str, Any]],
    margins: dict[str, dict[str, float]],
    split_by_id: dict[str, str],
    tail_band: tuple[int, int],
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    del tail_band
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        hidden = torch.load(
            spec.hidden_dir / f"layer_{layer}.pt",
            map_location="cpu",
            weights_only=False,
        )
    sample_ids = [str(item) for item in hidden["sample_ids"]]
    metadata = [
        _metadata_row(rows_by_id[sample_id], margins.get(sample_id, {}), split_by_id)
        for sample_id in sample_ids
    ]
    z_img = hidden["z_img"].float().numpy()
    z_blind = hidden["z_blind"].float().numpy()
    matrices = {
        "paired_difference": z_blind - z_img,
        "image_shuffle_difference": z_blind - _permuted_by_split(z_img, metadata, seed),
        "blind_shuffle_difference": _permuted_by_split(z_blind, metadata, seed + 1) - z_img,
        "raw_img": z_img,
        "raw_blind": z_blind,
    }
    train_idx, y_train = _label_indices(metadata, "train", {"FP": 1, "TN": 0})
    test_idx, y_test = _label_indices(metadata, "test", {"FP": 1, "TN": 0})
    py_test_idx, y_py_test = _label_indices(metadata, "test", {"FP": 1, "TP": 0})
    rows = []
    for feature, matrix in matrices.items():
        scores, _ = _fit_scores(matrix, train_idx, y_train, seed + len(feature), max_iter)
        rows.append(
            {
                "alias": spec.alias,
                "display_name": spec.display_name,
                "readout": spec.readout,
                "layer": layer,
                "feature": feature,
                "fp_tn_test_auroc": _auroc(y_test, scores[test_idx]),
                "predicted_yes_test_auroc": _auroc(y_py_test, scores[py_test_idx]),
                "fp_tn_test_n": int(len(y_test)),
                "predicted_yes_test_n": int(len(y_py_test)),
            }
        )
    return rows


def _fit_scores(
    x: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, float]]:
    scores = np.full(x.shape[0], np.nan, dtype=np.float64)
    if len(train_idx) == 0 or len(np.unique(y_train)) < 2:
        return scores, {"train_auroc": math.nan, "train_auprc": math.nan}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(scaler.transform(x))[:, 1]
    train_scores = scores[train_idx]
    return scores, {
        "train_auroc": _auroc(y_train, train_scores),
        "train_auprc": _auprc(y_train, train_scores),
    }


def _combine_scores(primary: np.ndarray, secondary: np.ndarray, calibration_idx: np.ndarray) -> np.ndarray:
    if len(calibration_idx) == 0:
        return np.full_like(primary, np.nan, dtype=np.float64)
    p_center, p_scale = _center_scale(primary[calibration_idx])
    s_center, s_scale = _center_scale(secondary[calibration_idx])
    return ((primary - p_center) / p_scale) + ((secondary - s_center) / s_scale)


def _center_scale(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


def _metadata_row(
    row: dict[str, Any],
    margin: dict[str, float],
    split_by_id: dict[str, str],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    yes_minus_no = _float_or_nan(margin.get("yes_minus_no_logit"))
    entropy = _float_or_nan(margin.get("binary_entropy"))
    return {
        "sample_id": sample_id,
        "subset": str(row.get("subset", "")),
        "protocol_split": split_by_id.get(sample_id, ""),
        "label": str(row.get("label", "")),
        "outcome": str(row.get("outcome", "")),
        "parsed_prediction": str(row.get("parsed_prediction", "")),
        "yes_minus_no_logit": yes_minus_no,
        "low_yes_margin": -yes_minus_no if np.isfinite(yes_minus_no) else math.nan,
        "binary_entropy": entropy,
    }


def _label_indices(
    metadata: list[dict[str, Any]],
    split: str,
    label_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    idx = []
    y = []
    for i, row in enumerate(metadata):
        if row["protocol_split"] != split:
            continue
        if row["outcome"] not in label_map:
            continue
        idx.append(i)
        y.append(label_map[row["outcome"]])
    return np.array(idx, dtype=np.int64), np.array(y, dtype=np.int64)


def _predicted_yes_mask(metadata: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.array(
        [
            row["protocol_split"] == split
            and row["parsed_prediction"] == "yes"
            and row["outcome"] in {"FP", "TP"}
            for row in metadata
        ],
        dtype=bool,
    )


def _calibrate_threshold(
    values: np.ndarray,
    metadata: list[dict[str, Any]],
    split: str,
    target_rate: float,
) -> float:
    mask = _predicted_yes_mask(metadata, split)
    calibration_values = values[mask]
    calibration_values = calibration_values[np.isfinite(calibration_values)]
    if len(calibration_values) == 0:
        return math.nan
    n_trigger = max(1, int(math.ceil(target_rate * len(calibration_values))))
    n_trigger = min(n_trigger, len(calibration_values))
    return float(np.sort(calibration_values)[-n_trigger])


def _gate_mask(values: np.ndarray, metadata: list[dict[str, Any]], split: str, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold):
        return np.zeros(len(metadata), dtype=bool)
    return _predicted_yes_mask(metadata, split) & (values >= threshold)


def _available_layers(hidden_dir: Path) -> list[int]:
    layers = []
    summary = hidden_dir / "dump_hidden_states_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        layers = [int(layer) for layer in payload.get("layers", [])]
    if not layers:
        for path in hidden_dir.glob("layer_*.pt"):
            layers.append(int(path.stem.split("_", 1)[1]))
    return sorted(set(layers))


def _load_split_map(split_dir: Path) -> dict[str, str]:
    split_by_id: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        path = split_dir / f"pope_{split}_ids.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample_id in payload["sample_ids"]:
            split_by_id[str(sample_id)] = split
    return split_by_id


def _load_margins(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    margins = {}
    for row in df.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        margins[sample_id] = {
            "yes_minus_no_logit": _float_or_nan(row.get("yes_minus_no_logit")),
            "binary_entropy": _float_or_nan(row.get("binary_entropy")),
        }
    return margins


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _score_array(metadata: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([_float_or_nan(row.get(key)) for row in metadata], dtype=np.float64)


def _explained_variance(singular_values: np.ndarray, k: int) -> float:
    if len(singular_values) == 0:
        return math.nan
    power = singular_values.astype(np.float64) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return math.nan
    return float(np.sum(power[: min(k, len(power))]) / total)


def _permuted_by_split(matrix: np.ndarray, metadata: list[dict[str, Any]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = matrix.copy()
    for split in ["train", "val", "test", ""]:
        indices = np.array(
            [i for i, row in enumerate(metadata) if row["protocol_split"] == split],
            dtype=np.int64,
        )
        if len(indices) <= 1:
            continue
        permuted = indices.copy()
        rng.shuffle(permuted)
        out[indices] = matrix[permuted]
    return out


def _interpretation_flag(fp_tn_auroc: float, pred_yes_auroc: float) -> str:
    if pd.notna(fp_tn_auroc) and pd.notna(pred_yes_auroc):
        if fp_tn_auroc >= 0.95 and pred_yes_auroc <= 0.35:
            return "near_perfect_fp_tn_but_non_deployable"
        if fp_tn_auroc >= 0.70 and pred_yes_auroc >= 0.60:
            return "mechanism_and_deployment_partly_align"
        if pred_yes_auroc > fp_tn_auroc:
            return "deployment_stronger_than_fp_tn_probe"
    return "mixed_or_insufficient"


def _write_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _render_note(
    mechanism_summary_df: pd.DataFrame,
    deployment_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    shuffle_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = [
        "# Stage U: Unified Cross-Model Minimal Protocol",
        "",
        "This note consolidates the required third experiment into one reusable protocol.",
        "",
        "## Files",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(["", "## Mechanism Task: Variance vs Discrimination", ""])
    main_mech = mechanism_summary_df[
        mechanism_summary_df.get("include_in_main", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    if not main_mech.empty:
        lines.extend(
            [
                "| Model | Readout | Layer | Top-4 Var | Top-4 AUROC | Top-64 AUROC | Top-256 AUROC | Full Diff AUROC | Tail AUROC |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in main_mech.itertuples(index=False):
            lines.append(
                f"| {row.display_name} | `{row.readout}` | {int(row.layer)} | "
                f"{_fmt(row.top4_explained_variance)} | {_fmt(row.top4_auroc)} | "
                f"{_fmt(row.top64_auroc)} | {_fmt(row.top256_auroc)} | "
                f"{_fmt(row.full_diff_auroc)} | {_fmt(row.tail_257_1024_auroc)} |"
            )
    lines.extend(
        [
            "",
            "Reading: LLaVA/Qwen retain the variance-discrimination decoupling pattern: top-4 directions explain large variance but are not the best discriminators. InternVL is the exception: FP/TN separability is already near-perfect in top coordinates, which is exactly why it needs the deployment diagnosis below.",
            "",
            "## Deployment Task: Predicted-Yes FP vs TP",
            "",
        ]
    )
    main_deploy = deployment_df[
        deployment_df.get("include_in_main", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    if not main_deploy.empty:
        lines.extend(
            [
                "| Model | Layer | FP Base | Geometry AUROC | Low-Margin AUROC | Entropy AUROC | Low-Margin+Geometry AUROC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in main_deploy.itertuples(index=False):
            lines.append(
                f"| {row.display_name} | {int(row.selected_layer)} | "
                f"{_fmt(row.predicted_yes_base_fp_rate)} | {_fmt(row.geometry_full_auroc)} | "
                f"{_fmt(row.low_margin_auroc)} | {_fmt(row.entropy_auroc)} | "
                f"{_fmt(row.low_margin_plus_geometry_auroc)} |"
            )
    gate20 = gate_df[
        (gate_df.get("include_in_main", pd.Series(dtype=bool)).astype(bool))
        & (gate_df.get("target_trigger_rate_predicted_yes", pd.Series(dtype=float)).round(3) == 0.2)
        & (gate_df.get("matched_gate", pd.Series(dtype=str)).isin(["low_margin_plus_geometry"]))
        & (gate_df.get("gate", pd.Series(dtype=str)).isin(["low_margin_plus_geometry", "same_trigger_random"]))
        & (gate_df.get("trigger_n", pd.Series(dtype=float)) > 0)
    ].copy()
    if not gate20.empty:
        lines.extend(
            [
                "",
                "At the calibrated 20% predicted-Yes target rate:",
                "",
                "| Model | Gate | Trigger | Precision FP | FP Recall | TP Damage |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in gate20.sort_values(["display_name", "gate"]).itertuples(index=False):
            lines.append(
                f"| {row.display_name} | `{row.gate}` | {_fmt(row.trigger_rate_predicted_yes)} | "
                f"{_fmt(row.precision_fp)} | {_fmt(row.fp_recall)} | {_fmt(row.tp_damage)} |"
            )

    lines.extend(["", "## Failure-Mode Diagnosis", ""])
    if not failure_df.empty:
        focus = failure_df[failure_df["alias"].str.startswith("internvl")].copy()
        lines.extend(
            [
                "| Model | Readout | Layer | FP/TN Full | Pred-Yes Full | Pred-Yes Low-Margin | Corr(score, margin) | Flag |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in focus.sort_values(["display_name", "readout"]).itertuples(index=False):
            lines.append(
                f"| {row.display_name} | `{row.readout}` | {int(row.selected_layer)} | "
                f"{_fmt(row.fp_tn_full_auroc)} | {_fmt(row.predicted_yes_full_auroc)} | "
                f"{_fmt(row.predicted_yes_low_margin_auroc)} | "
                f"{_fmt(row.fp_tn_full_vs_yes_margin_spearman)} | `{row.interpretation_flag}` |"
            )
        lines.extend(
            [
                "",
                "Key boundary finding: Some architectures can exhibit near-perfect FP/TN internal separability that does not translate into deployable FP/TP risk detection.",
            ]
        )
    if not distribution_df.empty:
        intern_dist = distribution_df[
            distribution_df["alias"].str.startswith("internvl")
            & (distribution_df["score"] == "geometry_full")
            & (distribution_df["outcome"].isin(["FP", "TP"]))
        ].copy()
        if not intern_dist.empty:
            lines.extend(
                [
                    "",
                    "InternVL predicted-Yes score distribution snapshot:",
                    "",
                    "| Model | Readout | Outcome | N | Mean | Median | Q25 | Q75 |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in intern_dist.sort_values(["display_name", "readout", "outcome"]).itertuples(index=False):
                lines.append(
                    f"| {row.display_name} | `{row.readout}` | {row.outcome} | {int(row.n)} | "
                    f"{_fmt(row.mean)} | {_fmt(row.median)} | {_fmt(row.q25)} | {_fmt(row.q75)} |"
                )
    if not shuffle_df.empty:
        paired = shuffle_df[
            shuffle_df["alias"].str.startswith("internvl")
            & shuffle_df["feature"].isin(["paired_difference", "image_shuffle_difference", "blind_shuffle_difference"])
        ].copy()
        lines.extend(
            [
                "",
                "Shuffle controls show whether the signal requires matched blind/image pairing:",
                "",
                "| Model | Readout | Layer | Feature | FP/TN AUROC | Pred-Yes AUROC |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in paired.sort_values(["display_name", "readout", "layer", "feature"]).itertuples(index=False):
            lines.append(
                f"| {row.display_name} | `{row.readout}` | {int(row.layer)} | `{row.feature}` | "
                f"{_fmt(row.fp_tn_test_auroc)} | {_fmt(row.predicted_yes_test_auroc)} |"
            )
    if not missing_df.empty:
        lines.extend(["", "## Missing Artifacts", ""])
        for row in missing_df.itertuples(index=False):
            lines.append(f"- `{row.alias}`: {row.reason}")
    lines.extend(
        [
            "",
            "## Recommended Paper Framing",
            "",
            "- Use the mechanism table to claim cross-model recurrence of variance/discrimination decoupling where it actually holds.",
            "- Use the deployment table to keep the practical claim honest: geometry is a complementary risk signal, not a universal replacement for output confidence.",
            "- Use InternVL as a boundary discovery rather than a failed replication: near-perfect FP/TN separability can be real internally yet non-deployable for predicted-Yes FP/TP routing.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.3f}"


def _float_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if np.isfinite(value) else math.nan


def _auroc(y: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    y = y[mask]
    values = values[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, values))


def _auprc(y: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    y = y[mask]
    values = values[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    return float(average_precision_score(y, values))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(pd.Series(a).rank(method="average").to_numpy(), pd.Series(b).rank(method="average").to_numpy())


def _quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(np.quantile(values, q))


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(~np.isfinite(arr)):
        return math.nan
    return float(np.nanmean(arr))


def _nanmax(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(~np.isfinite(arr)):
        return math.nan
    return float(np.nanmax(arr))


if __name__ == "__main__":
    main()
