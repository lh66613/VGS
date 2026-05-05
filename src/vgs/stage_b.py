"""Stage B condition planning and geometry analysis."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from vgs.artifacts import (
    load_condition_hidden_layer,
    load_hidden_layer,
    load_svd,
    read_jsonl,
)
from vgs.geometry import projection_similarity
from vgs.io import ensure_dir, write_csv, write_jsonl


pd = None
plt = None
LinearDiscriminantAnalysis = None
LogisticRegression = None
StandardScaler = None


def prepare_stage_b_condition_plan(
    predictions_path: str | Path,
    output_dir: str | Path,
    seed: int,
    outcomes: list[str],
    max_samples: int | None = None,
    max_samples_per_outcome: int | None = None,
    require_adversarial: bool = True,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    rng = np.random.default_rng(seed)
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_question.setdefault(_normalize_question(row["question"]), []).append(row)

    candidates = [row for row in rows if not outcomes or row.get("outcome") in outcomes]
    if max_samples_per_outcome is not None:
        balanced = []
        for outcome in outcomes:
            group = [row for row in candidates if row.get("outcome") == outcome]
            rng.shuffle(group)
            balanced.extend(group[:max_samples_per_outcome])
        candidates = balanced
    else:
        rng.shuffle(candidates)
    if max_samples is not None:
        candidates = candidates[:max_samples]

    plan_rows: list[dict[str, Any]] = []
    skipped_no_adversarial = 0
    for row in candidates:
        adversarial = _choose_adversarial(row, by_question, rng)
        if adversarial is None and require_adversarial:
            skipped_no_adversarial += 1
            continue
        random_mismatch = _choose_random_mismatch(row, rows, rng)
        if random_mismatch is None:
            continue
        plan_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "question": row["question"],
                "label": row["label"],
                "outcome": row.get("outcome"),
                "subset": row.get("subset"),
                "matched_image_path": row["image_path"],
                "matched_image": row.get("image"),
                "random_mismatch_sample_id": str(random_mismatch["sample_id"]),
                "random_mismatch_label": random_mismatch["label"],
                "random_mismatch_outcome": random_mismatch.get("outcome"),
                "random_mismatch_image_path": random_mismatch["image_path"],
                "random_mismatch_image": random_mismatch.get("image"),
                "adversarial_mismatch_sample_id": str(adversarial["sample_id"]) if adversarial else "",
                "adversarial_mismatch_label": adversarial["label"] if adversarial else "",
                "adversarial_mismatch_outcome": adversarial.get("outcome") if adversarial else "",
                "adversarial_mismatch_image_path": adversarial["image_path"] if adversarial else "",
                "adversarial_mismatch_image": adversarial.get("image") if adversarial else "",
                "adversarial_available": adversarial is not None,
            }
        )

    output_path = write_jsonl(Path(output_dir) / "stage_b_condition_plan.jsonl", plan_rows)
    return {
        "condition_plan_path": str(output_path),
        "num_input_rows": len(rows),
        "num_candidate_rows": len(candidates),
        "num_plan_rows": len(plan_rows),
        "skipped_no_adversarial": skipped_no_adversarial,
        "outcomes": outcomes,
        "max_samples": max_samples,
        "max_samples_per_outcome": max_samples_per_outcome,
        "require_adversarial": require_adversarial,
    }


def analyze_stage_b_geometry(
    layers: list[int],
    top_k_grid: list[int],
    tail_bands: list[tuple[int, int]],
    condition_plan_path: str | Path,
    condition_hidden_dir: str | Path,
    svd_dir: str | Path,
    reference_predictions_path: str | Path,
    reference_hidden_states_dir: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    _ensure_analysis_dependencies()
    ensure_dir(output_dir)
    ensure_dir(plot_dir)
    plan_rows = read_jsonl(condition_plan_path)
    plan_by_id = {str(row["sample_id"]): row for row in plan_rows}
    sample_score_rows: list[dict[str, Any]] = []
    condition_summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage B geometry layers", unit="layer"):
        condition_payload = load_condition_hidden_layer(condition_hidden_dir, layer)
        sample_ids = [str(sample_id) for sample_id in condition_payload["sample_ids"]]
        conditions = {
            condition: tensor.float().numpy()
            for condition, tensor in condition_payload["conditions"].items()
        }
        if "blind" not in conditions:
            raise KeyError(f"Layer {layer} condition artifact must include a blind condition.")
        blind = conditions["blind"]
        basis = load_svd(svd_dir, layer)["Vh"].float().numpy().T
        supervised_vectors = _reference_supervised_vectors(
            layer,
            reference_predictions_path,
            reference_hidden_states_dir,
            seed,
        )
        layer_score_rows = []
        for condition, states in conditions.items():
            diff = blind - states
            for row_idx, sample_id in enumerate(sample_ids):
                plan = plan_by_id.get(sample_id, {})
                base = {
                    "layer": layer,
                    "sample_id": sample_id,
                    "condition": condition,
                    "outcome": plan.get("outcome"),
                    "label": plan.get("label"),
                    "subset": plan.get("subset"),
                }
                score_rows = _stage_b_score_rows(
                    diff[row_idx],
                    basis,
                    supervised_vectors,
                    top_k_grid,
                    tail_bands,
                    base,
                )
                sample_score_rows.extend(score_rows)
                layer_score_rows.extend(score_rows)

        layer_df = pd.DataFrame(layer_score_rows)
        if not layer_df.empty:
            condition_summary_rows.extend(_condition_summary(layer_df))
            pairwise_rows.extend(_pairwise_condition_rows(layer_df, layer))
            outcome_rows.extend(_outcome_summary(layer_df))
        subspace_rows.extend(_condition_subspace_rows(layer, conditions, blind, basis, top_k_grid))

    sample_scores_path = write_csv(
        Path(output_dir) / "stage_b_sample_scores.csv",
        sample_score_rows,
        _fieldnames(sample_score_rows),
    )
    condition_summary_path = write_csv(
        Path(output_dir) / "stage_b_condition_score_summary.csv",
        condition_summary_rows,
        _fieldnames(condition_summary_rows),
    )
    pairwise_path = write_csv(
        Path(output_dir) / "stage_b_pairwise_condition_deltas.csv",
        pairwise_rows,
        _fieldnames(pairwise_rows),
    )
    outcome_path = write_csv(
        Path(output_dir) / "stage_b_outcome_condition_summary.csv",
        outcome_rows,
        _fieldnames(outcome_rows),
    )
    subspace_path = write_csv(
        Path(output_dir) / "stage_b_condition_subspace_similarity.csv",
        subspace_rows,
        _fieldnames(subspace_rows),
    )
    _plot_stage_b_condition_summary(condition_summary_rows, Path(plot_dir))
    _plot_stage_b_outcome_summary(outcome_rows, Path(plot_dir))
    return {
        "num_sample_score_rows": len(sample_score_rows),
        "num_condition_summary_rows": len(condition_summary_rows),
        "num_pairwise_rows": len(pairwise_rows),
        "num_outcome_rows": len(outcome_rows),
        "num_subspace_rows": len(subspace_rows),
        "sample_scores_path": str(sample_scores_path),
        "condition_summary_path": str(condition_summary_path),
        "pairwise_path": str(pairwise_path),
        "outcome_path": str(outcome_path),
        "subspace_path": str(subspace_path),
    }


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _choose_adversarial(
    row: dict[str, Any],
    by_question: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    question_key = _normalize_question(row["question"])
    candidates = [
        candidate
        for candidate in by_question.get(question_key, [])
        if candidate["label"] != row["label"]
        and candidate["sample_id"] != row["sample_id"]
        and candidate["image_path"] != row["image_path"]
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _choose_random_mismatch(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in rows
        if candidate["sample_id"] != row["sample_id"]
        and candidate["image_path"] != row["image_path"]
        and _normalize_question(candidate["question"]) != _normalize_question(row["question"])
    ]
    if not candidates:
        candidates = [
            candidate
            for candidate in rows
            if candidate["sample_id"] != row["sample_id"]
            and candidate["image_path"] != row["image_path"]
        ]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _stage_b_score_rows(
    diff: np.ndarray,
    basis: np.ndarray,
    supervised_vectors: dict[str, np.ndarray],
    top_k_grid: list[int],
    tail_bands: list[tuple[int, int]],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            **base,
            "view": "full",
            "score": "full_l2_sq",
            "start": 1,
            "end": int(diff.shape[0]),
            "value": float(np.dot(diff, diff)),
        }
    ]
    for k in top_k_grid:
        end = min(k, basis.shape[1])
        coords = diff @ basis[:, :end]
        rows.append(
            {
                **base,
                "view": "top_backbone",
                "score": f"top_1_{end}_l2_sq",
                "start": 1,
                "end": end,
                "value": float(np.dot(coords, coords)),
            }
        )
    for start, end in tail_bands:
        band_start = max(start, 1)
        band_end = min(end, basis.shape[1])
        if band_end < band_start:
            continue
        coords = diff @ basis[:, band_start - 1 : band_end]
        rows.append(
            {
                **base,
                "view": "residual_tail",
                "score": f"band_{band_start}_{band_end}_l2_sq",
                "start": band_start,
                "end": band_end,
                "value": float(np.dot(coords, coords)),
            }
        )
    for name, vector in supervised_vectors.items():
        rows.append(
            {
                **base,
                "view": "supervised_decision",
                "score": name,
                "start": math.nan,
                "end": math.nan,
                "value": float(diff @ vector),
            }
        )
    return rows


def _reference_supervised_vectors(
    layer: int,
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    seed: int,
) -> dict[str, np.ndarray]:
    _ensure_analysis_dependencies()
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id = {}
    for row in prediction_rows:
        if row.get("outcome") == "FP":
            labels_by_id[str(row["sample_id"])] = 1
        elif row.get("outcome") == "TN":
            labels_by_id[str(row["sample_id"])] = 0
    hidden = load_hidden_layer(hidden_states_dir, layer)
    sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
    keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
    y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
    diff = hidden["z_blind"][keep].float().numpy() - hidden["z_img"][keep].float().numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(diff)
    vectors = {}
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    logistic.fit(x_scaled, y)
    vectors["logistic_decision"] = _unit_vector(logistic.coef_[0] / np.maximum(scaler.scale_, 1e-12))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(x_scaled, y)
    vectors["lda_fisher_decision"] = _unit_vector(lda.coef_[0] / np.maximum(scaler.scale_, 1e-12))
    return vectors


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _condition_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for keys, group in df.groupby(["layer", "view", "score", "condition"], dropna=False):
        layer, view, score, condition = keys
        rows.append(
            {
                "layer": layer,
                "view": view,
                "score": score,
                "condition": condition,
                "n": int(len(group)),
                "mean": float(group["value"].mean()),
                "std": float(group["value"].std(ddof=1)) if len(group) > 1 else math.nan,
                "median": float(group["value"].median()),
                "q25": float(group["value"].quantile(0.25)),
                "q75": float(group["value"].quantile(0.75)),
            }
        )
    return rows


def _outcome_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for keys, group in df.groupby(["layer", "view", "score", "condition", "outcome"], dropna=False):
        layer, view, score, condition, outcome = keys
        rows.append(
            {
                "layer": layer,
                "view": view,
                "score": score,
                "condition": condition,
                "outcome": outcome,
                "n": int(len(group)),
                "mean": float(group["value"].mean()),
                "median": float(group["value"].median()),
                "q25": float(group["value"].quantile(0.25)),
                "q75": float(group["value"].quantile(0.75)),
            }
        )
    return rows


def _pairwise_condition_rows(df: pd.DataFrame, layer: int) -> list[dict[str, Any]]:
    rows = []
    for keys, group in df.groupby(["view", "score"], dropna=False):
        view, score = keys
        pivot = group.pivot_table(index="sample_id", columns="condition", values="value", aggfunc="first")
        for other in ["random_mismatch", "adversarial_mismatch", "blind"]:
            if "matched" not in pivot or other not in pivot:
                continue
            delta = pivot["matched"] - pivot[other]
            rows.append(
                {
                    "layer": layer,
                    "view": view,
                    "score": score,
                    "comparison": f"matched_minus_{other}",
                    "n": int(delta.dropna().shape[0]),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "q25_delta": float(delta.quantile(0.25)),
                    "q75_delta": float(delta.quantile(0.75)),
                }
            )
    return rows


def _condition_subspace_rows(
    layer: int,
    conditions: dict[str, np.ndarray],
    blind: np.ndarray,
    reference_basis: np.ndarray,
    k_grid: list[int],
) -> list[dict[str, Any]]:
    image_conditions = [
        condition
        for condition in ["matched", "random_mismatch", "adversarial_mismatch"]
        if condition in conditions
    ]
    bases = {}
    for condition in image_conditions:
        diff = blind - conditions[condition]
        _, _, vt = np.linalg.svd(diff, full_matrices=False)
        bases[condition] = vt.T
    rows = []
    for k in k_grid:
        for condition, basis in bases.items():
            k_eff = min(k, basis.shape[1], reference_basis.shape[1])
            rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "condition_a": condition,
                    "condition_b": "reference_matched_svd",
                    "projection_similarity": projection_similarity(
                        basis[:, :k_eff],
                        reference_basis[:, :k_eff],
                    ),
                }
            )
        for idx, condition_a in enumerate(image_conditions):
            for condition_b in image_conditions[idx + 1 :]:
                if condition_a not in bases or condition_b not in bases:
                    continue
                k_eff = min(k, bases[condition_a].shape[1], bases[condition_b].shape[1])
                rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "projection_similarity": projection_similarity(
                            bases[condition_a][:, :k_eff],
                            bases[condition_b][:, :k_eff],
                        ),
                    }
                )
    return rows


def _plot_stage_b_condition_summary(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    _ensure_analysis_dependencies()
    df = pd.DataFrame(rows)
    if df.empty:
        return
    ensure_dir(plot_dir)
    selected = df[
        df["score"].isin(
            [
                "top_1_4_l2_sq",
                "top_1_256_l2_sq",
                "band_257_1024_l2_sq",
                "logistic_decision",
            ]
        )
    ].copy()
    if selected.empty:
        return
    for layer, layer_df in selected.groupby("layer"):
        fig, axes = plt.subplots(1, len(layer_df["score"].unique()), figsize=(14, 4), squeeze=False)
        for ax, (score, group) in zip(axes[0], layer_df.groupby("score")):
            group = group.sort_values("condition")
            ax.bar(group["condition"], group["mean"])
            ax.set_title(score)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Stage B Condition Scores L{layer}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"stage_b_condition_scores_layer_{layer}.png", dpi=180)
        plt.close(fig)


def _plot_stage_b_outcome_summary(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    _ensure_analysis_dependencies()
    df = pd.DataFrame(rows)
    if df.empty:
        return
    selected = df[
        (df["score"].isin(["logistic_decision", "top_1_256_l2_sq", "band_257_1024_l2_sq"]))
        & (df["outcome"].isin(["FP", "TN"]))
    ].copy()
    if selected.empty:
        return
    for layer, layer_df in selected.groupby("layer"):
        fig, axes = plt.subplots(1, len(layer_df["score"].unique()), figsize=(13, 4), squeeze=False)
        for ax, (score, group) in zip(axes[0], layer_df.groupby("score")):
            labels = [f"{row.condition}/{row.outcome}" for row in group.itertuples()]
            ax.bar(labels, group["mean"])
            ax.set_title(score)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Stage B FP/TN Scores L{layer}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"stage_b_outcome_scores_layer_{layer}.png", dpi=180)
        plt.close(fig)


def _ensure_analysis_dependencies() -> None:
    global LinearDiscriminantAnalysis, LogisticRegression, StandardScaler, pd, plt
    if pd is not None:
        return
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as matplotlib_pyplot
    import pandas as pandas_module
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

    pd = pandas_module
    plt = matplotlib_pyplot
    LinearDiscriminantAnalysis = SklearnLDA
    LogisticRegression = SklearnLogisticRegression
    StandardScaler = SklearnStandardScaler


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys
