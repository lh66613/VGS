"""Stage M local rescue memory-bank preparation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vgs.artifacts import load_hidden_layer, load_svd, read_jsonl
from vgs.io import ensure_dir, write_csv, write_jsonl
from vgs.pope import classify_outcome, parse_yes_no
from vgs.stage_e import (
    _candidate_token_ids,
    _generate_with_optional_intervention,
    _make_intervention,
    _max_token_logit,
    _next_token_logits_with_optional_intervention,
    _to_device_payload,
    _top_decoded_token,
)


def build_stage_m_memory_bank(
    layers: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    tail_band: tuple[int, int],
    max_svd_coords: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    prediction_rows = read_jsonl(predictions_path)
    rows_by_id = {str(row["sample_id"]): row for row in prediction_rows}
    train_ids = _load_split_ids(split_dir)["train"]
    bank: dict[str, Any] = {
        "metadata": {
            "predictions": str(predictions_path),
            "hidden_states_dir": str(hidden_states_dir),
            "svd_dir": str(svd_dir),
            "split_dir": str(split_dir),
            "split": "train",
            "tail_band": tail_band,
            "max_svd_coords": max_svd_coords,
        },
        "layers": {},
    }
    audit_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage M memory layers", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids_all = [str(sample_id) for sample_id in hidden["sample_ids"]]
        keep = [idx for idx, sample_id in enumerate(sample_ids_all) if sample_id in train_ids]
        sample_ids = [sample_ids_all[idx] for idx in keep]
        z_img = hidden["z_img"][keep].float()
        z_blind = hidden["z_blind"][keep].float()
        correction = z_blind - z_img
        basis = load_svd(svd_dir, layer)["Vh"].float().T
        coord_dim = min(max_svd_coords, basis.shape[1])
        svd_coords = correction @ basis[:, :coord_dim]
        tail_start, tail_end = tail_band
        tail_start = max(1, tail_start)
        tail_end = min(tail_end, basis.shape[1])
        tail_coords = correction @ basis[:, tail_start - 1 : tail_end]
        metadata_rows = [_metadata_row(rows_by_id[sample_id]) for sample_id in sample_ids]
        objects = [row["queried_object"] for row in metadata_rows]
        images = [row["image"] for row in metadata_rows]
        bank["layers"][layer] = {
            "sample_ids": sample_ids,
            "metadata_rows": metadata_rows,
            "correction": correction.cpu(),
            "svd_coords": svd_coords.cpu(),
            "tail_coords": tail_coords.cpu(),
            "basis_source": str(Path(svd_dir) / f"svd_layer_{layer}.pt"),
            "tail_band": (tail_start, tail_end),
        }
        outcome_counts = Counter(row["outcome"] for row in metadata_rows)
        label_counts = Counter(row["label"] for row in metadata_rows)
        object_counts = Counter(objects)
        image_counts = Counter(images)
        audit_rows.append(
            {
                "layer": layer,
                "num_entries": len(sample_ids),
                "correction_dim": int(correction.shape[1]),
                "svd_coord_dim": int(svd_coords.shape[1]),
                "tail_coord_dim": int(tail_coords.shape[1]),
                "tail_band": f"{tail_start}-{tail_end}",
                "num_unique_objects": len(object_counts),
                "num_unique_images": len(image_counts),
                "max_same_object_count": max(object_counts.values()) if object_counts else 0,
                "max_same_image_count": max(image_counts.values()) if image_counts else 0,
                "num_fp": outcome_counts.get("FP", 0),
                "num_tn": outcome_counts.get("TN", 0),
                "num_tp": outcome_counts.get("TP", 0),
                "num_fn": outcome_counts.get("FN", 0),
                "num_yes_label": label_counts.get("yes", 0),
                "num_no_label": label_counts.get("no", 0),
            }
        )
        for obj, count in object_counts.most_common(50):
            object_rows.append({"layer": layer, "queried_object": obj, "count": count})

    bank_path = ensure_dir(output_dir) / "memory_bank_train.pt"
    torch.save(bank, bank_path)
    audit_path = write_csv(
        Path(output_dir) / "retrieval_audit.csv",
        audit_rows,
        list(audit_rows[0].keys()) if audit_rows else [],
    )
    object_path = write_csv(
        Path(output_dir) / "memory_bank_object_counts.csv",
        object_rows,
        list(object_rows[0].keys()) if object_rows else [],
    )
    return {
        "memory_bank_path": str(bank_path),
        "retrieval_audit_path": str(audit_path),
        "object_counts_path": str(object_path),
        "num_layers": len(layers),
        "layers": layers,
    }


def prepare_stage_m_retrieval_plan(
    layers: list[int],
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    split_dir: str | Path,
    memory_bank_path: str | Path,
    output_dir: str | Path,
    target_split: str,
    outcomes: list[str],
    max_targets_per_outcome: int,
    k_neighbors: int,
    tail_band: tuple[int, int],
    max_svd_coords: int,
    exclude_same_image: bool,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    prediction_rows = read_jsonl(predictions_path)
    rows_by_id = {str(row["sample_id"]): row for row in prediction_rows}
    split_ids = _load_split_ids(split_dir)
    target_ids = split_ids[target_split]
    bank = torch.load(memory_bank_path, map_location="cpu")
    plan_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage M retrieval layers", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids_all = [str(sample_id) for sample_id in hidden["sample_ids"]]
        target_indices = [
            idx
            for idx, sample_id in enumerate(sample_ids_all)
            if sample_id in target_ids and rows_by_id[sample_id].get("outcome") in outcomes
        ]
        selected_indices = []
        for outcome in outcomes:
            group = [idx for idx in target_indices if rows_by_id[sample_ids_all[idx]].get("outcome") == outcome]
            rng.shuffle(group)
            selected_indices.extend(group[:max_targets_per_outcome])
        selected_indices = sorted(selected_indices, key=lambda idx: sample_ids_all[idx])

        z_img = hidden["z_img"][selected_indices].float()
        z_blind = hidden["z_blind"][selected_indices].float()
        correction = z_blind - z_img
        basis = load_svd(svd_dir, layer)["Vh"].float().T
        coord_dim = min(max_svd_coords, basis.shape[1])
        svd_coords = correction @ basis[:, :coord_dim]
        tail_start, tail_end = tail_band
        tail_start = max(1, tail_start)
        tail_end = min(tail_end, basis.shape[1])
        tail_coords = correction @ basis[:, tail_start - 1 : tail_end]

        layer_bank = bank["layers"][layer]
        bank_meta = layer_bank["metadata_rows"]
        bank_sample_ids = [str(sample_id) for sample_id in layer_bank["sample_ids"]]
        bank_svd = layer_bank["svd_coords"].float()
        bank_tail = layer_bank["tail_coords"].float()
        tn_indices = [idx for idx, row in enumerate(bank_meta) if row["outcome"] == "TN"]
        fp_indices = [idx for idx, row in enumerate(bank_meta) if row["outcome"] == "FP"]
        if not tn_indices:
            raise ValueError(f"No TN entries in memory bank for layer {layer}.")

        for local_idx, hidden_idx in enumerate(selected_indices):
            sample_id = sample_ids_all[hidden_idx]
            target_row = rows_by_id[sample_id]
            queried_object = _extract_object(target_row["question"])
            target_image = target_row.get("image", "")
            tn_candidate_indices = _filter_same_image(bank_meta, tn_indices, target_image) if exclude_same_image else tn_indices
            fp_candidate_indices = _filter_same_image(bank_meta, fp_indices, target_image) if exclude_same_image else fp_indices
            retrievals = {
                "same_object_tn": _same_object_indices(bank_meta, tn_candidate_indices, queried_object),
                "svd_knn_tn": _nearest_indices(svd_coords[local_idx], bank_svd, tn_candidate_indices, k_neighbors),
                "tail_knn_tn": _nearest_indices(tail_coords[local_idx], bank_tail, tn_candidate_indices, k_neighbors),
                "random_tn": _random_indices(tn_candidate_indices, k_neighbors, rng),
                "same_object_fp": _same_object_indices(bank_meta, fp_candidate_indices, queried_object),
            }
            for mode, indices in retrievals.items():
                indices = indices[:k_neighbors]
                retrieved_ids = [bank_sample_ids[idx] for idx in indices]
                retrieved_images = [bank_meta[idx]["image"] for idx in indices]
                same_image_count = sum(1 for image in retrieved_images if image == target_image)
                plan_rows.append(
                    {
                        "layer": layer,
                        "target_split": target_split,
                        "target_sample_id": sample_id,
                        "target_outcome": target_row.get("outcome", ""),
                        "target_label": target_row.get("label", ""),
                        "target_object": queried_object,
                        "target_image": target_image,
                        "retrieval_mode": mode,
                        "k_neighbors": len(indices),
                        "retrieved_sample_ids": "|".join(retrieved_ids),
                        "retrieved_images": "|".join(retrieved_images),
                        "same_image_count": same_image_count,
                    }
                )

        audit_rows.append(
            {
                "layer": layer,
                "target_split": target_split,
                "num_targets": len(selected_indices),
                "outcomes": "|".join(outcomes),
                "max_targets_per_outcome": max_targets_per_outcome,
                "k_neighbors": k_neighbors,
                "num_train_tn_candidates": len(tn_indices),
                "num_train_fp_candidates": len(fp_indices),
                "exclude_same_image": exclude_same_image,
            }
        )

    plan_jsonl = write_jsonl(Path(output_dir) / "retrieval_plan.jsonl", plan_rows)
    plan_csv = write_csv(Path(output_dir) / "retrieval_plan.csv", plan_rows, list(plan_rows[0].keys()) if plan_rows else [])
    audit_path = write_csv(Path(output_dir) / "retrieval_plan_audit.csv", audit_rows, list(audit_rows[0].keys()) if audit_rows else [])
    return {
        "retrieval_plan_jsonl": str(plan_jsonl),
        "retrieval_plan_csv": str(plan_csv),
        "retrieval_plan_audit": str(audit_path),
        "num_plan_rows": len(plan_rows),
        "num_audit_rows": len(audit_rows),
    }


def run_stage_m_local_rescue(
    model: Any,
    processor: Any,
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    memory_bank_path: str | Path,
    retrieval_plan_path: str | Path,
    output_dir: str | Path,
    layers: list[int],
    device: str,
    alpha_grid: list[float],
    gates: list[str],
    retrieval_modes: list[str],
    target_outcomes: list[str],
    max_targets_per_outcome: int,
    margin_threshold: float,
    entropy_threshold: float,
    fp_risk_threshold: float,
    max_new_tokens: int,
    granularities: list[str],
    logits_only: bool,
    seed: int,
) -> dict[str, Any]:
    """Run Stage M gated local rescue interventions from a retrieval plan."""
    ensure_dir(output_dir)
    prediction_rows = read_jsonl(predictions_path)
    rows_by_id = {str(row["sample_id"]): row for row in prediction_rows}
    bank = torch.load(memory_bank_path, map_location="cpu")
    plan_rows = _read_csv_rows(retrieval_plan_path)
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage M local-rescue layers", unit="layer"):
        layer_plan = [
            row
            for row in plan_rows
            if int(row["layer"]) == layer
            and row["retrieval_mode"] in retrieval_modes
            and row["target_outcome"] in target_outcomes
        ]
        layer_plan = _cap_plan_targets(layer_plan, target_outcomes, max_targets_per_outcome, rng)
        if not layer_plan:
            continue
        layer_bank = bank["layers"][layer]
        bank_lookup = {str(sample_id): idx for idx, sample_id in enumerate(layer_bank["sample_ids"])}
        bank_meta = layer_bank["metadata_rows"]
        correction = layer_bank["correction"].float()
        fp_scores, tail_norms = _stage_m_target_scores(
            layer=layer,
            layer_bank=layer_bank,
            hidden_states_dir=hidden_states_dir,
            target_ids=sorted({row["target_sample_id"] for row in layer_plan}),
        )
        global_tn = _mean_bank_correction(correction, bank_meta, [idx for idx, row in enumerate(bank_meta) if row["outcome"] == "TN"])
        sample_plan = _group_plan_by_target(layer_plan)

        for sample_id, target_plan_rows in tqdm(
            sample_plan.items(),
            desc=f"L{layer} local-rescue samples",
            unit="sample",
            leave=False,
        ):
            sample = rows_by_id[sample_id]
            baseline_logits = _next_token_logits_with_optional_intervention(
                model,
                processor,
                sample,
                device,
                layer=None,
                intervention=None,
            )
            baseline_text = (
                ""
                if logits_only
                else _generate_with_optional_intervention(
                    model,
                    processor,
                    sample,
                    device,
                    max_new_tokens=max_new_tokens,
                    layer=None,
                    intervention=None,
                )
            )
            baseline_prediction = parse_yes_no(baseline_text) if baseline_text else sample.get("parsed_prediction")
            baseline = _stage_m_logit_row(
                layer=layer,
                sample=sample,
                gate="baseline",
                intervention="baseline",
                retrieval_mode="none",
                alpha=0.0,
                granularity="none",
                retrieved_ids=[],
                k_neighbors=0,
                text=baseline_text,
                prediction=baseline_prediction,
                logits=baseline_logits,
                processor=processor,
                baseline_margin=None,
                fp_risk_score=fp_scores.get(sample_id),
                tail_norm=tail_norms.get(sample_id),
                gate_pass=True,
                logits_only=logits_only,
            )
            baseline_margin = baseline["yes_minus_no_logit"]
            baseline_entropy = _binary_entropy(baseline["yes_logit"], baseline["no_logit"])
            baseline["binary_entropy"] = baseline_entropy
            results.append(baseline)

            direction_specs = _stage_m_direction_specs(
                plan_rows=target_plan_rows,
                correction=correction,
                bank_lookup=bank_lookup,
                global_tn=global_tn,
            )
            for gate in gates:
                gate_pass = _stage_m_gate_pass(
                    gate=gate,
                    margin=baseline_margin,
                    entropy=baseline_entropy,
                    fp_risk_score=fp_scores.get(sample_id),
                    tail_norm=tail_norms.get(sample_id),
                    margin_threshold=margin_threshold,
                    entropy_threshold=entropy_threshold,
                    fp_risk_threshold=fp_risk_threshold,
                )
                if not gate_pass:
                    continue
                for spec in direction_specs:
                    payload = {
                        "mode": "signed_vector",
                        "direction": spec["direction"],
                        "sign": spec["sign"],
                    }
                    for alpha in alpha_grid:
                        for granularity in granularities:
                            intervention = _make_intervention(spec["intervention"], _to_device_payload(payload, device), alpha)
                            logits = _next_token_logits_with_optional_intervention(
                                model,
                                processor,
                                sample,
                                device,
                                layer=layer,
                                intervention=intervention,
                                granularity=granularity,
                            )
                            text = (
                                ""
                                if logits_only
                                else _generate_with_optional_intervention(
                                    model,
                                    processor,
                                    sample,
                                    device,
                                    max_new_tokens=max_new_tokens,
                                    layer=layer,
                                    intervention=intervention,
                                    granularity=granularity,
                                )
                            )
                            prediction = parse_yes_no(text) if text else None
                            row = _stage_m_logit_row(
                                layer=layer,
                                sample=sample,
                                gate=gate,
                                intervention=spec["intervention"],
                                retrieval_mode=spec["retrieval_mode"],
                                alpha=float(alpha),
                                granularity=granularity,
                                retrieved_ids=spec["retrieved_ids"],
                                k_neighbors=len(spec["retrieved_ids"]),
                                text=text,
                                prediction=prediction,
                                logits=logits,
                                processor=processor,
                                baseline_margin=baseline_margin,
                                fp_risk_score=fp_scores.get(sample_id),
                                tail_norm=tail_norms.get(sample_id),
                                gate_pass=gate_pass,
                                logits_only=logits_only,
                            )
                            row["binary_entropy"] = _binary_entropy(row["yes_logit"], row["no_logit"])
                            results.append(row)

    results_path = write_csv(
        Path(output_dir) / "local_rescue_results.csv",
        results,
        list(results[0].keys()) if results else [],
    )
    return {
        "results_path": str(results_path),
        "num_rows": len(results),
        "logits_only": logits_only,
    }


def analyze_stage_m_local_rescue(
    results_path: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import binomtest

    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    df = pd.read_csv(results_path)
    if df.empty:
        summary_path = write_csv(Path(output_dir) / "local_rescue_summary.csv", [], [])
        return {"summary_path": str(summary_path), "num_rows": 0}

    intervention_df = df[df["intervention"] != "baseline"].copy()
    keys = ["layer", "gate", "intervention", "retrieval_mode", "granularity", "alpha", "outcome_before"]
    summary_rows: list[dict[str, Any]] = []
    for key, group in intervention_df.groupby(keys, dropna=False):
        baseline_lookup = (
            df[df["intervention"] == "baseline"]
            .set_index(["layer", "sample_id"])["outcome_after"]
            .to_dict()
        )
        baseline_correct = []
        intervention_correct = []
        for _, row in group.iterrows():
            base_outcome = baseline_lookup.get((row["layer"], row["sample_id"]), "")
            baseline_correct.append(base_outcome in {"TP", "TN"})
            intervention_correct.append(row["outcome_after"] in {"TP", "TN"})
        b = sum(base and not inter for base, inter in zip(baseline_correct, intervention_correct))
        c = sum((not base) and inter for base, inter in zip(baseline_correct, intervention_correct))
        p_value = float(binomtest(c, b + c, 0.5).pvalue) if (b + c) else 1.0
        summary_rows.append(
            {
                "layer": key[0],
                "gate": key[1],
                "intervention": key[2],
                "retrieval_mode": key[3],
                "granularity": key[4],
                "alpha": key[5],
                "outcome_before": key[6],
                "n": len(group),
                "accuracy": float(np.mean([row in {"TP", "TN"} for row in group["outcome_after"]])),
                "fp_rescue_rate": float(np.mean(group["outcome_after"] == "TN")) if key[6] == "FP" else math.nan,
                "tn_damage_rate": float(np.mean(group["outcome_after"] != "TN")) if key[6] == "TN" else math.nan,
                "tp_damage_rate": float(np.mean(group["outcome_after"] != "TP")) if key[6] == "TP" else math.nan,
                "unknown_rate": float(np.mean(~group["outcome_after"].isin(["TP", "TN", "FP", "FN"]))),
                "median_margin_delta_vs_baseline": float(group["margin_delta_vs_baseline"].median()),
                "median_no_minus_yes_gain": float(group["no_minus_yes_gain"].median()),
                "mean_no_minus_yes_gain": float(group["no_minus_yes_gain"].mean()),
                "mcnemar_b": int(b),
                "mcnemar_c": int(c),
                "mcnemar_p_value": p_value,
            }
        )
    summary_path = write_csv(
        Path(output_dir) / "local_rescue_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    _plot_stage_m_fp_margin(summary_rows, plot_dir)
    _plot_stage_m_gate_tradeoff(summary_rows, plot_dir)
    return {
        "summary_path": str(summary_path),
        "num_rows": len(df),
        "num_summary_rows": len(summary_rows),
        "fp_margin_plot": str(Path(plot_dir) / "stage_m_fp_margin_shift.png"),
        "gate_tradeoff_plot": str(Path(plot_dir) / "stage_m_gate_tradeoff_curve.png"),
    }


def analyze_stage_m_rescue_failures(
    results_path: str | Path,
    predictions_path: str | Path,
    retrieval_plan_path: str | Path,
    memory_bank_path: str | Path,
    hidden_states_dir: str | Path,
    output_dir: str | Path,
    notes_path: str | Path,
) -> dict[str, Any]:
    import pandas as pd

    ensure_dir(output_dir)
    results = pd.read_csv(results_path)
    predictions = pd.DataFrame(read_jsonl(predictions_path))
    plan = pd.read_csv(retrieval_plan_path)
    bank = torch.load(memory_bank_path, map_location="cpu")
    baseline = results[(results["intervention"] == "baseline") & (results["outcome_before"] == "FP")].copy()
    pred_cols = ["sample_id", "subset", "question", "image", "image_path", "raw_generation"]
    baseline = baseline.merge(predictions[pred_cols], how="left", on="sample_id", suffixes=("", "_prediction"))
    taxonomy_rows: list[dict[str, Any]] = []

    for _, base in baseline.iterrows():
        layer = int(base["layer"])
        sample_id = str(base["sample_id"])
        group = results[
            (results["layer"] == layer)
            & (results["sample_id"].astype(str) == sample_id)
            & (results["intervention"] != "baseline")
        ].copy()
        row = _stage_m_failure_taxonomy_row(base, group)
        row.update(_stage_m_retrieval_diagnostics(layer, sample_id, plan, bank, hidden_states_dir))
        taxonomy_rows.append(row)

    taxonomy = pd.DataFrame(taxonomy_rows)
    taxonomy_path = write_csv(
        Path(output_dir) / "rescue_failure_taxonomy.csv",
        taxonomy_rows,
        list(taxonomy_rows[0].keys()) if taxonomy_rows else [],
    )
    summary_rows = _stage_m_failure_summary_rows(taxonomy)
    summary_path = write_csv(
        Path(output_dir) / "rescue_failure_group_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    notes = _stage_m_failure_notes(taxonomy, summary_rows, taxonomy_path, summary_path)
    notes_target = Path(notes_path)
    ensure_dir(notes_target.parent)
    notes_target.write_text(notes, encoding="utf-8")
    return {
        "taxonomy_path": str(taxonomy_path),
        "summary_path": str(summary_path),
        "notes_path": str(notes_target),
        "num_fp_samples": int(len(taxonomy)),
    }


def _metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "question": row["question"],
        "queried_object": _extract_object(row["question"]),
        "subset": row.get("subset", ""),
        "label": row.get("label", ""),
        "outcome": row.get("outcome", ""),
        "image": row.get("image", ""),
        "image_path": row.get("image_path", ""),
        "parsed_prediction": row.get("parsed_prediction", ""),
        "raw_generation": row.get("raw_generation", ""),
        "yes_no_margin": None,
    }


def _stage_m_failure_taxonomy_row(base: Any, group: Any) -> dict[str, Any]:
    baseline_margin = float(base["yes_minus_no_logit"])
    best_gain = float(group["no_minus_yes_gain"].max()) if len(group) else 0.0
    worst_gain = float(group["no_minus_yes_gain"].min()) if len(group) else 0.0
    rescued = group[group["outcome_after"] == "TN"].copy()
    malformed = group[group["outcome_after"] == "unknown"].copy()
    best = group.sort_values("no_minus_yes_gain", ascending=False).head(1)
    best_row = best.iloc[0] if len(best) else None
    if len(rescued):
        label = "rescued_to_correct_no"
    elif len(malformed):
        label = "damaged_or_malformed"
    elif best_gain > 0:
        label = "margin_improved_answer_unchanged"
    elif worst_gain < 0:
        label = "moved_in_wrong_direction"
    else:
        label = "no_effect"
    return {
        "layer": int(base["layer"]),
        "sample_id": str(base["sample_id"]),
        "subset": base.get("subset", ""),
        "target_object": base.get("target_object", ""),
        "question": base.get("question", ""),
        "image": base.get("image", ""),
        "baseline_margin_yes_minus_no": baseline_margin,
        "baseline_margin_bin": _stage_m_margin_bin(baseline_margin),
        "fp_risk_score": float(base["fp_risk_score"]),
        "tail_norm": float(base["tail_norm"]),
        "failure_label": label,
        "rescued": bool(len(rescued)),
        "num_rescue_rows": int(len(rescued)),
        "num_malformed_rows": int(len(malformed)),
        "max_no_minus_yes_gain": best_gain,
        "min_no_minus_yes_gain": worst_gain,
        "best_gate": best_row["gate"] if best_row is not None else "",
        "best_intervention": best_row["intervention"] if best_row is not None else "",
        "best_retrieval_mode": best_row["retrieval_mode"] if best_row is not None else "",
        "best_alpha": float(best_row["alpha"]) if best_row is not None else math.nan,
        "best_margin_yes_minus_no": float(best_row["yes_minus_no_logit"]) if best_row is not None else math.nan,
        "best_prediction": best_row["parsed_prediction"] if best_row is not None else "",
        "best_outcome_after": best_row["outcome_after"] if best_row is not None else "",
    }


def _stage_m_margin_bin(margin: float) -> str:
    abs_margin = abs(margin)
    if abs_margin <= 0.25:
        return "borderline_abs_le_0.25"
    if abs_margin <= 1.0:
        return "medium_abs_0.25_1.0"
    return "high_abs_gt_1.0"


def _stage_m_retrieval_diagnostics(
    layer: int,
    sample_id: str,
    plan: Any,
    bank: dict[str, Any],
    hidden_states_dir: str | Path,
) -> dict[str, Any]:
    layer_bank = bank["layers"][layer]
    bank_ids = [str(item) for item in layer_bank["sample_ids"]]
    bank_index = {sample_id: idx for idx, sample_id in enumerate(bank_ids)}
    hidden = load_hidden_layer(hidden_states_dir, layer)
    hidden_ids = [str(item) for item in hidden["sample_ids"]]
    hidden_index = {sample_id: idx for idx, sample_id in enumerate(hidden_ids)}
    idx = hidden_index.get(sample_id)
    if idx is None:
        return {}
    basis_payload = torch.load(layer_bank["basis_source"], map_location="cpu")
    basis = basis_payload["Vh"].float().T
    diff = hidden["z_blind"][idx].float() - hidden["z_img"][idx].float()
    svd_dim = layer_bank["svd_coords"].shape[1]
    tail_start, tail_end = layer_bank["tail_band"]
    target_svd = diff @ basis[:, :svd_dim]
    target_tail = diff @ basis[:, tail_start - 1 : tail_end]
    rows = plan[(plan["layer"] == layer) & (plan["target_sample_id"].astype(str) == sample_id)]
    diagnostics: dict[str, Any] = {}
    for _, row in rows.iterrows():
        mode = row["retrieval_mode"]
        retrieved_ids = [item for item in str(row.get("retrieved_sample_ids", "")).split("|") if item and item != "nan"]
        indices = [bank_index[item] for item in retrieved_ids if item in bank_index]
        diagnostics[f"{mode}_k"] = int(len(indices))
        if not indices:
            diagnostics[f"{mode}_mean_svd_distance"] = math.nan
            diagnostics[f"{mode}_mean_tail_distance"] = math.nan
            continue
        bank_svd = layer_bank["svd_coords"][indices].float()
        bank_tail = layer_bank["tail_coords"][indices].float()
        diagnostics[f"{mode}_mean_svd_distance"] = float(torch.linalg.norm(bank_svd - target_svd[None, :], dim=1).mean().item())
        diagnostics[f"{mode}_mean_tail_distance"] = float(torch.linalg.norm(bank_tail - target_tail[None, :], dim=1).mean().item())
    return diagnostics


def _stage_m_failure_summary_rows(taxonomy: Any) -> list[dict[str, Any]]:
    if taxonomy.empty:
        return []
    rows: list[dict[str, Any]] = []
    for field in ["failure_label", "subset", "baseline_margin_bin", "target_object"]:
        grouped = taxonomy.groupby(field, dropna=False)
        for value, group in grouped:
            rows.append(
                {
                    "group_type": field,
                    "group": value,
                    "n": int(len(group)),
                    "rescued_rate": float(group["rescued"].mean()),
                    "median_baseline_margin": float(group["baseline_margin_yes_minus_no"].median()),
                    "median_max_no_minus_yes_gain": float(group["max_no_minus_yes_gain"].median()),
                    "median_fp_risk_score": float(group["fp_risk_score"].median()),
                    "median_tail_norm": float(group["tail_norm"].median()),
                }
            )
    return sorted(rows, key=lambda row: (row["group_type"], -row["n"], str(row["group"])))


def _stage_m_failure_notes(taxonomy: Any, summary_rows: list[dict[str, Any]], taxonomy_path: Path, summary_path: Path) -> str:
    if taxonomy.empty:
        return "# Stage M Rescue Failure Analysis\n\nNo FP samples were available in the Stage M result file.\n"
    counts = taxonomy["failure_label"].value_counts().to_dict()
    margin_counts = taxonomy["baseline_margin_bin"].value_counts().to_dict()
    rescued = taxonomy[taxonomy["rescued"]].copy()
    improved = taxonomy[taxonomy["failure_label"] == "margin_improved_answer_unchanged"].copy()
    stubborn = taxonomy[taxonomy["failure_label"].isin(["no_effect", "moved_in_wrong_direction"])].copy()

    def fmt_counts(values: dict[str, Any]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in values.items())

    lines = [
        "# Stage M Rescue Failure Analysis",
        "",
        "Artifacts:",
        "",
        f"- Taxonomy: `{taxonomy_path}`",
        f"- Group summary: `{summary_path}`",
        "",
        "## Outcome Taxonomy",
        "",
        f"Total FP samples analyzed: {len(taxonomy)}",
        "",
        fmt_counts(counts),
        "",
        "## Margin Bins",
        "",
        fmt_counts(margin_counts),
        "",
        "## Rescued Samples",
        "",
    ]
    if rescued.empty:
        lines.append("- No FP samples were rescued to correct `No`.")
    else:
        for row in rescued.sort_values("baseline_margin_yes_minus_no").itertuples(index=False):
            lines.append(
                f"- `{row.sample_id}` / object `{row.target_object}` / subset `{row.subset}`: "
                f"baseline yes-no margin `{row.baseline_margin_yes_minus_no:.6f}`, "
                f"best gain `{row.max_no_minus_yes_gain:.6f}`, "
                f"best `{row.best_gate}` + `{row.best_intervention}` at alpha `{row.best_alpha}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rescue is concentrated in extremely low-margin FP samples.",
            "- Most FP samples receive a positive margin nudge but do not cross the yes/no boundary.",
            "- High-margin FP cases remain failures under this first-token steering setup, consistent with stronger language-prior or visual-evidence-insensitive hallucinations.",
            "- Current retrieval diagnostics should be treated cautiously because this run did not include the `same_object_fp` retrieval mode during intervention.",
            "",
        ]
    )
    if not improved.empty:
        lines.append(
            f"Median baseline margin for margin-improved but unrescued FP samples: "
            f"`{improved['baseline_margin_yes_minus_no'].median():.6f}`."
        )
    if not stubborn.empty:
        lines.append(
            f"Median baseline margin for no-effect / wrong-direction FP samples: "
            f"`{stubborn['baseline_margin_yes_minus_no'].median():.6f}`."
        )
    return "\n".join(lines) + "\n"


def _plot_stage_m_fp_margin(summary_rows: list[dict[str, Any]], plot_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    fp = df[df["outcome_before"] == "FP"].copy()
    if fp.empty:
        return
    for layer, layer_df in fp.groupby("layer"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for (gate, intervention, granularity), group in layer_df.groupby(["gate", "intervention", "granularity"]):
            ordered = group.sort_values("alpha")
            label = f"{gate}/{intervention}/{granularity}"
            ax.plot(ordered["alpha"], ordered["median_no_minus_yes_gain"], marker="o", label=label)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Stage M FP margin shift, L{layer}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("median gain in logit(No)-logit(Yes)")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(Path(plot_dir) / f"stage_m_fp_margin_shift_layer_{layer}.png", dpi=200)
        plt.close(fig)
    best = (
        fp.sort_values("median_no_minus_yes_gain", ascending=False)
        .groupby(["layer", "gate", "intervention", "granularity"], as_index=False)
        .head(1)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [
        f"L{row.layer} {row.gate}\n{row.intervention}"
        for row in best.itertuples(index=False)
    ]
    ax.bar(range(len(best)), best["median_no_minus_yes_gain"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("best median gain in logit(No)-logit(Yes)")
    ax.set_title("Stage M FP margin shift")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / "stage_m_fp_margin_shift.png", dpi=200)
    plt.close(fig)


def _plot_stage_m_gate_tradeoff(summary_rows: list[dict[str, Any]], plot_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    if df.empty:
        return
    fp = df[df["outcome_before"] == "FP"]
    tn = df[df["outcome_before"] == "TN"]
    tp = df[df["outcome_before"] == "TP"]
    if fp.empty:
        return
    keys = ["layer", "gate", "intervention", "retrieval_mode", "granularity", "alpha"]
    fp_gain = fp.set_index(keys)["median_no_minus_yes_gain"]
    tn_damage = tn.set_index(keys)["tn_damage_rate"] if not tn.empty else {}
    tp_damage = tp.set_index(keys)["tp_damage_rate"] if not tp.empty else {}
    rows = []
    for key, gain in fp_gain.items():
        rows.append(
            {
                "key": key,
                "fp_gain": gain,
                "tn_damage": float(tn_damage.get(key, 0.0)) if hasattr(tn_damage, "get") else 0.0,
                "tp_damage": float(tp_damage.get(key, 0.0)) if hasattr(tp_damage, "get") else 0.0,
            }
        )
    tradeoff = pd.DataFrame(rows)
    if tradeoff.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    damage = tradeoff["tn_damage"].fillna(0.0) + tradeoff["tp_damage"].fillna(0.0)
    ax.scatter(damage, tradeoff["fp_gain"], alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("TN damage rate + TP damage rate")
    ax.set_ylabel("FP median gain in logit(No)-logit(Yes)")
    ax.set_title("Stage M gate tradeoff")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / "stage_m_gate_tradeoff_curve.png", dpi=200)
    plt.close(fig)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cap_plan_targets(
    plan_rows: list[dict[str, str]],
    target_outcomes: list[str],
    max_targets_per_outcome: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    if max_targets_per_outcome <= 0:
        return plan_rows
    keep_ids: set[str] = set()
    by_outcome: dict[str, list[str]] = {}
    for row in plan_rows:
        by_outcome.setdefault(row["target_outcome"], []).append(row["target_sample_id"])
    for outcome in target_outcomes:
        ids = sorted(set(by_outcome.get(outcome, [])))
        rng.shuffle(ids)
        keep_ids.update(ids[:max_targets_per_outcome])
    return [row for row in plan_rows if row["target_sample_id"] in keep_ids]


def _group_plan_by_target(plan_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in plan_rows:
        grouped.setdefault(row["target_sample_id"], []).append(row)
    return grouped


def _stage_m_target_scores(
    layer: int,
    layer_bank: dict[str, Any],
    hidden_states_dir: str | Path,
    target_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    metadata_rows = layer_bank["metadata_rows"]
    labels = []
    keep = []
    for idx, row in enumerate(metadata_rows):
        if row["outcome"] == "FP":
            labels.append(1)
            keep.append(idx)
        elif row["outcome"] == "TN":
            labels.append(0)
            keep.append(idx)
    tail_scores: dict[str, float] = {}
    fp_scores: dict[str, float] = {}
    if not keep:
        return fp_scores, tail_scores

    train_x = layer_bank["tail_coords"][keep].float().numpy()
    train_y = np.array(labels, dtype=np.int64)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=layer)
    clf.fit(train_scaled, train_y)

    hidden = load_hidden_layer(hidden_states_dir, layer)
    all_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
    index_by_id = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
    tail_dim = layer_bank["tail_coords"].shape[1]
    basis_payload = torch.load(layer_bank["basis_source"], map_location="cpu")
    basis = basis_payload["Vh"].float().T
    tail_start, tail_end = layer_bank["tail_band"]
    tail_basis = basis[:, tail_start - 1 : tail_end]
    for sample_id in target_ids:
        idx = index_by_id.get(sample_id)
        if idx is None:
            continue
        diff = hidden["z_blind"][idx].float() - hidden["z_img"][idx].float()
        tail_coord = (diff @ tail_basis).reshape(1, tail_dim).numpy()
        tail_scores[sample_id] = float(np.linalg.norm(tail_coord))
        fp_scores[sample_id] = float(clf.predict_proba(scaler.transform(tail_coord))[0, 1])
    return fp_scores, tail_scores


def _mean_bank_correction(
    correction: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
    indices: list[int],
) -> torch.Tensor | None:
    if not indices:
        return None
    return correction[indices].mean(dim=0)


def _stage_m_direction_specs(
    plan_rows: list[dict[str, str]],
    correction: torch.Tensor,
    bank_lookup: dict[str, int],
    global_tn: torch.Tensor | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if global_tn is not None:
        specs.append(
            {
                "intervention": "global_tn_mean_correction",
                "retrieval_mode": "global_tn",
                "direction": global_tn,
                "sign": -1.0,
                "retrieved_ids": [],
            }
        )
    mean_by_mode: dict[str, torch.Tensor] = {}
    ids_by_mode: dict[str, list[str]] = {}
    for row in plan_rows:
        retrieved_ids = [item for item in row.get("retrieved_sample_ids", "").split("|") if item]
        indices = [bank_lookup[item] for item in retrieved_ids if item in bank_lookup]
        if not indices:
            continue
        direction = correction[indices].mean(dim=0)
        mode = row["retrieval_mode"]
        mean_by_mode[mode] = direction
        ids_by_mode[mode] = retrieved_ids
        if mode.endswith("_tn"):
            specs.append(
                {
                    "intervention": f"{mode}_mean_correction",
                    "retrieval_mode": mode,
                    "direction": direction,
                    "sign": -1.0,
                    "retrieved_ids": retrieved_ids,
                }
            )
    fp_mean = mean_by_mode.get("same_object_fp")
    if fp_mean is not None:
        for mode, tn_mean in mean_by_mode.items():
            if not mode.endswith("_tn"):
                continue
            specs.append(
                {
                    "intervention": f"{mode}_minus_same_object_fp",
                    "retrieval_mode": f"{mode}+same_object_fp",
                    "direction": tn_mean - fp_mean,
                    "sign": -1.0,
                    "retrieved_ids": ids_by_mode.get(mode, []) + ids_by_mode.get("same_object_fp", []),
                }
            )
    return specs


def _stage_m_gate_pass(
    gate: str,
    margin: float,
    entropy: float,
    fp_risk_score: float | None,
    tail_norm: float | None,
    margin_threshold: float,
    entropy_threshold: float,
    fp_risk_threshold: float,
) -> bool:
    if gate == "always":
        return True
    if gate == "low_abs_margin":
        return abs(margin) <= margin_threshold
    if gate == "high_entropy":
        return entropy >= entropy_threshold
    if gate == "high_fp_risk":
        return fp_risk_score is not None and fp_risk_score >= fp_risk_threshold
    if gate == "margin_and_fp_risk":
        return (
            abs(margin) <= margin_threshold
            and fp_risk_score is not None
            and fp_risk_score >= fp_risk_threshold
        )
    if gate == "tail_norm_available":
        return tail_norm is not None
    raise ValueError(f"Unknown Stage M gate: {gate}")


def _binary_entropy(yes_logit: float, no_logit: float) -> float:
    values = np.array([yes_logit, no_logit], dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _stage_m_logit_row(
    layer: int,
    sample: dict[str, Any],
    gate: str,
    intervention: str,
    retrieval_mode: str,
    alpha: float,
    granularity: str,
    retrieved_ids: list[str],
    k_neighbors: int,
    text: str,
    prediction: str | None,
    logits: torch.Tensor,
    processor: Any,
    baseline_margin: float | None,
    fp_risk_score: float | None,
    tail_norm: float | None,
    gate_pass: bool,
    logits_only: bool,
) -> dict[str, Any]:
    yes_ids = _candidate_token_ids(processor.tokenizer, ["Yes", " yes", "yes"])
    no_ids = _candidate_token_ids(processor.tokenizer, ["No", " no", "no"])
    yes_logit = _max_token_logit(logits, yes_ids)
    no_logit = _max_token_logit(logits, no_ids)
    margin = yes_logit - no_logit
    outcome_after = "logits_only" if logits_only and intervention != "baseline" else classify_outcome(prediction, str(sample.get("label")).lower())
    return {
        "layer": layer,
        "sample_id": sample.get("sample_id"),
        "outcome_before": sample.get("outcome"),
        "label": sample.get("label"),
        "target_object": _extract_object(str(sample.get("question", ""))),
        "gate": gate,
        "gate_pass": gate_pass,
        "intervention": intervention,
        "retrieval_mode": retrieval_mode,
        "alpha": alpha,
        "granularity": granularity,
        "k_neighbors": k_neighbors,
        "retrieved_sample_ids": "|".join(retrieved_ids),
        "raw_generation": text,
        "parsed_prediction": prediction,
        "outcome_after": outcome_after,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_minus_no_logit": margin,
        "baseline_yes_minus_no_logit": baseline_margin if baseline_margin is not None else margin,
        "margin_delta_vs_baseline": margin - baseline_margin if baseline_margin is not None else 0.0,
        "no_minus_yes_gain": baseline_margin - margin if baseline_margin is not None else 0.0,
        "fp_risk_score": fp_risk_score,
        "tail_norm": tail_norm,
        "top_token": _top_decoded_token(processor, logits),
    }


def _extract_object(question: str) -> str:
    text = question.strip().rstrip("?")
    match = re.search(r"Is there (?:a|an|the|any) (.+?) in the image", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    match = re.search(r"Is there (.+?) in the image", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return text.lower()


def _same_object_indices(
    metadata_rows: list[dict[str, Any]],
    candidate_indices: list[int],
    queried_object: str,
) -> list[int]:
    return [idx for idx in candidate_indices if metadata_rows[idx]["queried_object"] == queried_object]


def _filter_same_image(
    metadata_rows: list[dict[str, Any]],
    candidate_indices: list[int],
    target_image: str,
) -> list[int]:
    return [idx for idx in candidate_indices if metadata_rows[idx]["image"] != target_image]


def _nearest_indices(
    query: torch.Tensor,
    bank_coords: torch.Tensor,
    candidate_indices: list[int],
    k: int,
) -> list[int]:
    candidates = torch.tensor(candidate_indices, dtype=torch.long)
    candidate_coords = bank_coords[candidates]
    distances = torch.sum((candidate_coords - query[None, :]) ** 2, dim=1)
    top_k = min(k, len(candidate_indices))
    order = torch.topk(distances, k=top_k, largest=False).indices
    return [int(candidates[idx]) for idx in order]


def _random_indices(candidate_indices: list[int], k: int, rng: np.random.Generator) -> list[int]:
    if len(candidate_indices) <= k:
        return list(candidate_indices)
    chosen = rng.choice(candidate_indices, size=k, replace=False)
    return [int(idx) for idx in chosen]


def _load_split_ids(split_dir: str | Path) -> dict[str, set[str]]:
    root = Path(split_dir)
    splits = {}
    for split in ["train", "val", "test"]:
        with (root / f"pope_{split}_ids.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        splits[split] = {str(sample_id) for sample_id in payload["sample_ids"]}
    return splits
