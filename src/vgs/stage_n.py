"""Stage N external benchmark preparation and transfer analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vgs.artifacts import load_condition_hidden_layer, load_hidden_layer, load_svd, read_jsonl
from vgs.io import ensure_dir, write_csv, write_jsonl
from vgs.stage_l import (
    _condition_logistic_basis,
    _contrastive_basis,
    _generalized_basis,
    _indices_for_ids,
    _orthonormal_columns,
    _pls_basis,
    _supervised_vector_plus_pca,
    _svd_basis,
)


def write_external_benchmark_choice(path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        "# External Benchmark Choice\n\n"
        "## Decision\n\n"
        "Use **AMBER discriminative** as the first external-validity sanity check.\n\n"
        "## Rationale\n\n"
        "- It is closest to the current POPE yes/no setup, so the first transfer test isolates benchmark shift rather than task-format shift.\n"
        "- It includes discriminative existence, attribute, and relation queries, which lets us test whether POPE-derived correction geometry transfers beyond object existence.\n"
        "- The official AMBER repository exposes separate query files for all discriminative queries and for existence / attribute / relation subsets.\n"
        "- The official evaluation script uses an annotation file with `truth` labels and reports discriminative accuracy / precision / recall / F1.\n\n"
        "## Primary Protocol\n\n"
        "1. Prepare AMBER discriminative rows without refitting any POPE subspace.\n"
        "2. Run LLaVA-1.5-7B on the AMBER rows and save yes/no predictions.\n"
        "3. Dump paired image/blind hidden states using the same `last_prompt_token` readout.\n"
        "4. Apply POPE SVD bases directly to AMBER differences.\n"
        "5. Evaluate transfer by correctness / hallucination outcome and by AMBER dimension.\n\n"
        "## Source Notes\n\n"
        "- Paper / HF page: https://huggingface.co/papers/2311.07397\n"
        "- Official repository: https://github.com/junyangwang0410/AMBER\n"
        "- Repository README lists discriminative query files including `query_discriminative.json`, `query_discriminative-existence.json`, `query_discriminative-attribute.json`, and `query_discriminative-relation.json`.\n\n"
        "## Local Layout\n\n"
        "- Queries: `data/amber/data/query/query_discriminative.json`\n"
        "- Annotations: `data/amber/data/annotations.json`\n"
        "- Images: `data/amber/image/AMBER_*.jpg`\n",
        encoding="utf-8",
    )
    return target


def prepare_amber_discriminative_plan(
    query_path: str | Path,
    annotation_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    dimension: str,
    max_samples: int | None = None,
    max_per_dimension_label: int | None = None,
    dimensions: list[str] | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    queries = _read_json_list(query_path)
    annotations = _read_json_list(annotation_path)
    rows = []
    for row in queries:
        amber_id = int(row["id"])
        annotation = annotations[amber_id - 1]
        label = str(annotation.get("truth", "")).strip().lower()
        image = str(row["image"])
        row_dimension = _amber_dimension(annotation.get("type", ""), fallback=dimension)
        if dimensions and row_dimension not in dimensions:
            continue
        rows.append(
            {
                "sample_id": f"amber:{amber_id}",
                "benchmark": "AMBER",
                "question_id": amber_id,
                "subset": dimension,
                "dimension": row_dimension,
                "image": image,
                "image_path": str(Path(images_dir) / image),
                "question": str(row["query"]),
                "label": label,
                "annotation_type": annotation.get("type", ""),
            }
        )
    if max_per_dimension_label is not None:
        rows = _stratified_limit(rows, ["dimension", "label"], max_per_dimension_label, seed)
    if max_samples is not None:
        rows = rows[:max_samples]
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    invalid_labels = sorted({row["label"] for row in rows if row["label"] not in {"yes", "no"}})
    plan_jsonl = write_jsonl(Path(output_dir) / "amber_discriminative_plan.jsonl", rows)
    plan_csv = write_csv(Path(output_dir) / "amber_discriminative_plan.csv", rows, list(rows[0].keys()) if rows else [])
    return {
        "plan_jsonl": str(plan_jsonl),
        "plan_csv": str(plan_csv),
        "num_rows": len(rows),
        "num_missing_images": len(missing),
        "missing_images_preview": missing[:20],
        "invalid_labels": invalid_labels,
        "dimensions": _counts([row["dimension"] for row in rows]),
        "labels": _counts([row["label"] for row in rows]),
        "ok": not missing and not invalid_labels,
    }


def analyze_external_transfer(
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    svd_dir: str | Path,
    pope_predictions_path: str | Path,
    pope_hidden_states_dir: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    layers: list[int],
    k_grid: list[int],
    tail_band: tuple[int, int],
    condition_hidden_dir: str | Path | None = None,
    condition_plan_path: str | Path | None = None,
    evidence_methods: list[str] | None = None,
    evidence_k_grid: list[int] | None = None,
    seed: int = 13,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    pope_predictions = {str(row["sample_id"]): row for row in read_jsonl(pope_predictions_path)}
    train_ids = _load_split_ids(split_dir, "train")
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    evidence_basis_dir = Path(output_dir) / "evidence_transfer_bases"
    evidence_basis_paths: list[str] = []
    for layer in layers:
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(item) for item in hidden["sample_ids"]]
        diff = hidden["z_blind"].float() - hidden["z_img"].float()
        basis = load_svd(svd_dir, layer)["Vh"].float().T
        tail_start, tail_end = tail_band
        tail_basis = basis[:, tail_start - 1 : tail_end]
        tail_energy = torch.linalg.norm(diff @ tail_basis, dim=1).numpy()
        full_norm = torch.linalg.norm(diff, dim=1).numpy()
        top_energy_by_k = {
            k: torch.linalg.norm(diff @ basis[:, : min(k, basis.shape[1])], dim=1).numpy()
            for k in k_grid
        }
        probe_scores = _pope_probe_transfer_scores(
            layer=layer,
            pope_predictions=pope_predictions,
            train_ids=train_ids,
            pope_hidden_states_dir=pope_hidden_states_dir,
            amber_diff=diff,
            basis=basis,
            k_grid=k_grid,
            tail_band=tail_band,
        )
        evidence_scores, evidence_path = _evidence_specific_transfer_scores(
            layer=layer,
            pope_predictions=pope_predictions,
            train_ids=train_ids,
            pope_hidden_states_dir=pope_hidden_states_dir,
            condition_hidden_dir=condition_hidden_dir,
            amber_diff=diff,
            methods=evidence_methods or ["pls_fp_tn", "fisher_fp_tn"],
            k_grid=evidence_k_grid or [4, 8, 16, 32, 64],
            seed=seed,
            ridge=ridge,
            output_dir=evidence_basis_dir,
        )
        if evidence_path is not None:
            evidence_basis_paths.append(str(evidence_path))
        for idx, sample_id in enumerate(sample_ids):
            pred = predictions.get(sample_id, {})
            row = {
                "layer": layer,
                "sample_id": sample_id,
                "dimension": pred.get("dimension", ""),
                "label": pred.get("label", ""),
                "parsed_prediction": pred.get("parsed_prediction", ""),
                "outcome": pred.get("outcome", ""),
                "is_correct": int(pred.get("outcome") in {"TP", "TN"}),
                "is_fp": int(pred.get("outcome") == "FP"),
                "full_diff_norm": float(full_norm[idx]),
                f"tail_{tail_start}_{tail_end}_energy": float(tail_energy[idx]),
            }
            for k, values in top_energy_by_k.items():
                row[f"top_{k}_energy"] = float(values[idx])
            for feature, values in probe_scores.items():
                row[feature] = float(values[idx])
            for feature, values in evidence_scores.items():
                row[feature] = float(values[idx])
            score_rows.append(row)
    scores_path = write_csv(
        Path(output_dir) / "external_transfer_scores.csv",
        score_rows,
        list(score_rows[0].keys()) if score_rows else [],
    )
    scores = pd.DataFrame(score_rows)
    feature_cols = (
        ["full_diff_norm", f"tail_{tail_band[0]}_{tail_band[1]}_energy"]
        + [f"top_{k}_energy" for k in k_grid]
        + sorted(
            [
                col
                for col in scores.columns
                if col.startswith("pope_probe_") or col.startswith("evidence_")
            ]
        )
    )
    for (layer, dimension), group in scores.groupby(["layer", "dimension"], dropna=False):
        y_fp = group["is_fp"].to_numpy()
        y_error = 1 - group["is_correct"].to_numpy()
        for feature in feature_cols:
            values = group[feature].to_numpy()
            summary_rows.append(
                {
                    "layer": layer,
                    "dimension": dimension,
                    "feature": feature,
                    "n": int(len(group)),
                    "fp_rate": float(np.mean(y_fp)) if len(group) else np.nan,
                    "error_rate": float(np.mean(y_error)) if len(group) else np.nan,
                    "fp_auroc": _safe_auc(y_fp, values, roc_auc_score),
                    "fp_auprc": _safe_auc(y_fp, values, average_precision_score),
                    "error_auroc": _safe_auc(y_error, values, roc_auc_score),
                }
            )
    summary_path = write_csv(
        Path(output_dir) / "external_category_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    _plot_external_transfer(summary_rows, Path(plot_dir) / "stage_n_external_transfer.png")
    return {
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
        "plot_path": str(Path(plot_dir) / "stage_n_external_transfer.png"),
        "num_rows": len(score_rows),
        "num_summary_rows": len(summary_rows),
        "evidence_basis_paths": evidence_basis_paths,
    }


def _read_json_list(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _amber_dimension(annotation_type: str, fallback: str) -> str:
    if annotation_type == "discriminative-hallucination":
        return "existence"
    if annotation_type.startswith("discriminative-attribute"):
        return "attribute"
    if annotation_type in {"discriminative-relation", "relation"}:
        return "relation"
    if annotation_type.startswith("discriminative"):
        return "discriminative"
    return fallback


def _stratified_limit(
    rows: list[dict[str, Any]],
    keys: list[str],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item, "") for item in keys)
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        selected.extend(group[:limit])
    return sorted(selected, key=lambda row: int(row["question_id"]))


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _load_split_ids(split_dir: str | Path, split: str) -> set[str]:
    path = Path(split_dir) / f"pope_{split}_ids.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(item) for item in payload["sample_ids"]}


def _pope_probe_transfer_scores(
    layer: int,
    pope_predictions: dict[str, dict[str, Any]],
    train_ids: set[str],
    pope_hidden_states_dir: str | Path,
    amber_diff: torch.Tensor,
    basis: torch.Tensor,
    k_grid: list[int],
    tail_band: tuple[int, int],
) -> dict[str, np.ndarray]:
    pope_hidden = load_hidden_layer(pope_hidden_states_dir, layer)
    pope_sample_ids = [str(item) for item in pope_hidden["sample_ids"]]
    train_indices = []
    labels = []
    for idx, sample_id in enumerate(pope_sample_ids):
        row = pope_predictions.get(sample_id, {})
        if sample_id not in train_ids:
            continue
        if row.get("outcome") == "FP":
            train_indices.append(idx)
            labels.append(1)
        elif row.get("outcome") == "TN":
            train_indices.append(idx)
            labels.append(0)
    if not train_indices or len(set(labels)) < 2:
        return {}
    pope_diff = pope_hidden["z_blind"][train_indices].float() - pope_hidden["z_img"][train_indices].float()
    y = np.array(labels, dtype=np.int64)
    tail_start, tail_end = tail_band
    feature_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in k_grid:
        dim = min(k, basis.shape[1])
        feature_pairs.append(
            (
                f"pope_probe_top_{k}_fp_risk",
                (pope_diff @ basis[:, :dim]).numpy(),
                (amber_diff @ basis[:, :dim]).numpy(),
            )
        )
    tail_basis = basis[:, tail_start - 1 : tail_end]
    feature_pairs.append(
        (
            f"pope_probe_tail_{tail_start}_{tail_end}_fp_risk",
            (pope_diff @ tail_basis).numpy(),
            (amber_diff @ tail_basis).numpy(),
        )
    )
    scores: dict[str, np.ndarray] = {}
    for name, train_x, test_x in feature_pairs:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=layer)
        clf.fit(train_scaled, y)
        scores[name] = clf.predict_proba(test_scaled)[:, 1]
    return scores


def _evidence_specific_transfer_scores(
    layer: int,
    pope_predictions: dict[str, dict[str, Any]],
    train_ids: set[str],
    pope_hidden_states_dir: str | Path,
    condition_hidden_dir: str | Path | None,
    amber_diff: torch.Tensor,
    methods: list[str],
    k_grid: list[int],
    seed: int,
    ridge: float,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], Path | None]:
    methods = [method for method in methods if method]
    if not methods:
        return {}, None
    pope_hidden = load_hidden_layer(pope_hidden_states_dir, layer)
    pope_sample_ids = [str(item) for item in pope_hidden["sample_ids"]]
    pope_diff_all = (pope_hidden["z_blind"].float() - pope_hidden["z_img"].float()).numpy()
    train_all_idx = _indices_for_ids(pope_sample_ids, train_ids)

    train_label_idx = []
    labels = []
    for idx, sample_id in enumerate(pope_sample_ids):
        row = pope_predictions.get(sample_id, {})
        if sample_id not in train_ids:
            continue
        if row.get("outcome") == "FP":
            train_label_idx.append(idx)
            labels.append(1)
        elif row.get("outcome") == "TN":
            train_label_idx.append(idx)
            labels.append(0)
    if not train_label_idx or len(set(labels)) < 2:
        return {}, None
    y = np.array(labels, dtype=np.int64)
    diff_train_labeled = pope_diff_all[train_label_idx]
    max_k = min(max(k_grid), pope_diff_all.shape[1], max(1, len(train_label_idx) - 1))
    condition_arrays = _load_condition_arrays(condition_hidden_dir, layer, train_ids)
    bases = _build_evidence_transfer_bases(
        methods=methods,
        diff_train_all=pope_diff_all[train_all_idx],
        diff_train_labeled=diff_train_labeled,
        y_train=y,
        condition_arrays=condition_arrays,
        max_k=max_k,
        seed=seed + layer,
        ridge=ridge,
    )
    if not bases:
        return {}, None
    ensure_dir(output_dir)
    basis_path = output_dir / f"layer_{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "methods": sorted(bases),
            "k_grid": k_grid,
            "basis": {name: torch.from_numpy(basis.astype(np.float32, copy=False)) for name, basis in bases.items()},
        },
        basis_path,
    )
    scores: dict[str, np.ndarray] = {}
    amber_np = amber_diff.numpy()
    for method, basis in bases.items():
        for k in k_grid:
            dim = min(k, basis.shape[1])
            if dim <= 0:
                continue
            train_x = diff_train_labeled @ basis[:, :dim]
            test_x = amber_np @ basis[:, :dim]
            scores[f"evidence_{method}_k{k}_energy"] = np.linalg.norm(test_x, axis=1)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_x)
            test_scaled = scaler.transform(test_x)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed + layer)
            clf.fit(train_scaled, y)
            scores[f"evidence_{method}_k{k}_fp_risk"] = clf.predict_proba(test_scaled)[:, 1]
    return scores, basis_path


def _load_condition_arrays(
    condition_hidden_dir: str | Path | None,
    layer: int,
    train_ids: set[str],
) -> dict[str, np.ndarray] | None:
    if condition_hidden_dir is None:
        return None
    path = Path(condition_hidden_dir)
    if not path.exists():
        return None
    payload = load_condition_hidden_layer(path, layer)
    sample_ids = [str(item) for item in payload["sample_ids"]]
    train_idx = _indices_for_ids(sample_ids, train_ids)
    if len(train_idx) == 0:
        train_idx = np.arange(len(sample_ids))
    conditions = {name: tensor.float().numpy() for name, tensor in payload["conditions"].items()}
    blind = conditions["blind"]
    return {
        "matched": (blind - conditions["matched"])[train_idx],
        "random_mismatch": (blind - conditions["random_mismatch"])[train_idx],
        "adversarial_mismatch": (blind - conditions["adversarial_mismatch"])[train_idx],
    }


def _build_evidence_transfer_bases(
    methods: list[str],
    diff_train_all: np.ndarray,
    diff_train_labeled: np.ndarray,
    y_train: np.ndarray,
    condition_arrays: dict[str, np.ndarray] | None,
    max_k: int,
    seed: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    bases: dict[str, np.ndarray] = {}
    for method in methods:
        if method == "plain_svd":
            basis = _svd_basis(diff_train_all, max_k, seed)
        elif method == "fisher_fp_tn":
            basis = _supervised_vector_plus_pca(diff_train_labeled, y_train, max_k, "fisher", seed)
        elif method == "pls_fp_tn":
            basis = _pls_basis(diff_train_labeled, y_train, max_k)
        elif method in {"contrastive_pca", "generalized_matched_vs_mismatch", "matched_vs_adversarial_logistic"}:
            if condition_arrays is None:
                continue
            matched = condition_arrays["matched"]
            random_mismatch = condition_arrays["random_mismatch"]
            adversarial_mismatch = condition_arrays["adversarial_mismatch"]
            if method == "contrastive_pca":
                basis = _contrastive_basis(
                    matched,
                    np.concatenate([random_mismatch, adversarial_mismatch], axis=0),
                    max_k,
                )
            elif method == "generalized_matched_vs_mismatch":
                basis = _generalized_basis(
                    matched,
                    np.concatenate([random_mismatch, adversarial_mismatch], axis=0),
                    max_k,
                    ridge,
                )
            else:
                basis = _condition_logistic_basis(matched, adversarial_mismatch, max_k, seed)
        else:
            continue
        bases[method] = _orthonormal_columns(basis)[:, :max_k]
    return bases


def _safe_auc(y_true: np.ndarray, scores: np.ndarray, metric: Any) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(metric(y_true, scores))


def _plot_external_transfer(summary_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    fp = df[df["dimension"].isin(["existence", "attribute", "relation"])].copy()
    if fp.empty:
        fp = df.copy()
    best = fp.sort_values("fp_auroc", ascending=False).groupby(["layer", "feature"], as_index=False).head(1)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"L{row.layer}\n{row.feature}" for row in best.itertuples(index=False)]
    ax.bar(range(len(best)), best["fp_auroc"])
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("FP AUROC")
    ax.set_title("Stage N external transfer")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
