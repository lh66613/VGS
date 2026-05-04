#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage Q paper-ready tables and figures.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--tables-dir", default="outputs/paper_tables")
    parser.add_argument("--figures-dir", default="outputs/paper_figures")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {
        "predictions": args.predictions,
        "tables_dir": args.tables_dir,
        "figures_dir": args.figures_dir,
    }
    if not args.dry_run:
        payload.update(build_stage_q_assets(args.predictions, args.tables_dir, args.figures_dir))
    summary_path = write_json(Path(args.tables_dir) / "build_stage_q_paper_assets_summary.json", payload)
    append_experiment_log(args.log_path, "build_stage_q_paper_assets", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


def build_stage_q_assets(predictions_path: str | Path, tables_dir: str | Path, figures_dir: str | Path) -> dict[str, Any]:
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    table_paths = {
        "table1": _build_table1(predictions_path, tables_dir),
        "table2": _build_table2(tables_dir),
        "table3": _build_table3(tables_dir),
        "table4": _build_table4(tables_dir),
    }
    figure_paths = {
        "fig1": _plot_fig1(figures_dir),
        "fig2": _plot_fig2(figures_dir),
        "fig3": _plot_fig3(figures_dir),
        "fig4": _plot_fig4(figures_dir),
        "fig5": _plot_fig5(figures_dir),
    }
    _write_asset_index(tables_dir, table_paths, figure_paths)
    return {
        "table_paths": {key: str(path) for key, path in table_paths.items()},
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }


def _build_table1(predictions_path: str | Path, tables_dir: str | Path) -> Path:
    rows = [json.loads(line) for line in Path(predictions_path).open("r", encoding="utf-8")]
    result = []
    for subset, group in _group_rows(rows, "subset"):
        counts = _counts(row.get("outcome", "") for row in group)
        labels = _counts(row.get("parsed_prediction", "") for row in group)
        n = len(group)
        result.append(
            {
                "subset": subset,
                "n": n,
                "accuracy": (counts.get("TP", 0) + counts.get("TN", 0)) / n,
                "TP": counts.get("TP", 0),
                "TN": counts.get("TN", 0),
                "FP": counts.get("FP", 0),
                "FN": counts.get("FN", 0),
                "yes_rate": labels.get("yes", 0) / n,
                "no_rate": labels.get("no", 0) / n,
                "unknown_rate": labels.get("", 0) / n + labels.get("unknown", 0) / n,
            }
        )
    counts = _counts(row.get("outcome", "") for row in rows)
    labels = _counts(row.get("parsed_prediction", "") for row in rows)
    n = len(rows)
    result.append(
        {
            "subset": "overall",
            "n": n,
            "accuracy": (counts.get("TP", 0) + counts.get("TN", 0)) / n,
            "TP": counts.get("TP", 0),
            "TN": counts.get("TN", 0),
            "FP": counts.get("FP", 0),
            "FN": counts.get("FN", 0),
            "yes_rate": labels.get("yes", 0) / n,
            "no_rate": labels.get("no", 0) / n,
            "unknown_rate": labels.get("", 0) / n + labels.get("unknown", 0) / n,
        }
    )
    return write_csv(Path(tables_dir) / "table1_pope_summary.csv", result, list(result[0].keys()))


def _build_table2(tables_dir: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    probe_path = Path("outputs/probes/feature_comparison.csv")
    if probe_path.exists():
        df = pd.read_csv(probe_path)
        for feature, label in [
            ("raw_img", "raw z_img"),
            ("raw_blind", "raw z_blind"),
            ("difference", "full difference"),
            ("projected_difference", "top-K SVD coordinates"),
            ("random_difference", "random projection"),
            ("pca_img", "image-state PCA"),
        ]:
            group = df[df["feature_family"] == feature]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(_feature_row("initial_probe", label, best))
    coord_path = Path("outputs/stage_c_coordinate_control/stage_c_coordinate_control.csv")
    if coord_path.exists():
        df = pd.read_csv(coord_path)
        for feature, label in [
            ("full_svd_coordinates", "full SVD coordinates"),
            ("raw_full_difference", "same-split raw difference"),
            ("random_orthogonal_rotation", "same-split random rotation"),
            ("pca_whitened_difference", "same-split PCA whitening"),
        ]:
            group = df[df["feature"] == feature]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(_feature_row("coordinate_control", label, best))
    stage_l_path = Path("outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv")
    if stage_l_path.exists():
        df = pd.read_csv(stage_l_path)
        for method, label in [
            ("pls_fp_tn", "evidence-specific PLS FP/TN"),
            ("fisher_fp_tn", "evidence-specific Fisher FP/TN"),
            ("contrastive_pca", "evidence-specific contrastive PCA"),
        ]:
            group = df[df["method"] == method]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(_feature_row("stage_l", label, best))
    return write_csv(Path(tables_dir) / "table2_feature_comparison.csv", rows, _fieldnames(rows))


def _feature_row(source: str, feature: str, row: pd.Series) -> dict[str, Any]:
    return {
        "source": source,
        "feature": feature,
        "layer": int(row.get("layer", -1)),
        "k": row.get("k", "full"),
        "feature_dim": row.get("feature_dim", ""),
        "auroc": float(row.get("auroc", np.nan)),
        "auprc": float(row.get("auprc", np.nan)),
        "accuracy": float(row.get("accuracy", np.nan)),
        "f1": float(row.get("f1", np.nan)),
    }


def _build_table3(tables_dir: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    path = Path("outputs/stage_j_controls/shuffle_probe_summary.csv")
    if path.exists():
        df = pd.read_csv(path)
        keep_features = ["full_difference", "top_k_svd_256", "top_k_svd_64", "top_k_svd_4"]
        for control in ["real_matched", "image_shuffled", "blind_shuffled", "label_shuffled", "gaussian_matched"]:
            group = df[(df["control"] == control) & (df["feature"].isin(keep_features))]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(
                {
                    "control": control,
                    "best_layer": int(best["layer"]),
                    "best_feature": best["feature"],
                    "feature_dim": int(best["feature_dim"]),
                    "auroc": float(best["auroc"]),
                    "auprc": float(best["auprc"]),
                    "accuracy": float(best["accuracy"]),
                    "f1": float(best["f1"]),
                }
            )
    return write_csv(Path(tables_dir) / "table3_controls.csv", rows, _fieldnames(rows))


def _build_table4(tables_dir: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    path = Path("outputs/interventions/stage_e_first_token_check_dose_curve.csv")
    if path.exists():
        df = pd.read_csv(path)
        keep = df[
            (df["layer"] == 24)
            & (df["outcome_before"] == "TN")
            & (df["intervention"].isin(["baseline", "ablate_tail_257_1024", "norm_matched_random_tail_control"]))
            & (df["granularity"].isin(["none", "last_token", "full_sequence"]))
        ].copy()
        for _, row in keep.iterrows():
            rows.append(
                {
                    "source": "stage_e_tail_ablation",
                    "layer": int(row["layer"]),
                    "direction_family": row["intervention"],
                    "granularity": row["granularity"],
                    "alpha": float(row["alpha"]),
                    "n": int(row["n"]),
                    "margin_shift_mean": float(row["mean_margin_delta_vs_baseline"]),
                    "yes_rate": float(row["yes_rate_all"]),
                    "no_rate": float(row["no_rate_all"]),
                    "flip_or_damage_rate": float(1.0 - row["accuracy_over_valid"]),
                    "unknown_rate": float(row["unknown_rate"]),
                }
            )
    rescue_path = Path("outputs/stage_m_local_rescue/local_rescue_summary.csv")
    if rescue_path.exists():
        df = pd.read_csv(rescue_path)
        candidates = df[(df["outcome_before"] == "FP") & (df["alpha"] == df["alpha"].max())].copy()
        for _, row in candidates.head(12).iterrows():
            rows.append(
                {
                    "source": "stage_m_local_rescue",
                    "layer": int(row["layer"]),
                    "direction_family": row["intervention"],
                    "granularity": "first_token",
                    "alpha": float(row["alpha"]),
                    "n": int(row["n"]),
                    "margin_shift_mean": float(row.get("mean_no_minus_yes_gain", np.nan)),
                    "yes_rate": np.nan,
                    "no_rate": np.nan,
                    "flip_or_damage_rate": float(row.get("fp_rescue_rate", np.nan)),
                    "unknown_rate": float(row.get("unknown_rate", np.nan)),
                }
            )
    return write_csv(Path(tables_dir) / "table4_intervention.csv", rows, _fieldnames(rows))


def _plot_fig1(figures_dir: str | Path) -> Path:
    path = Path(figures_dir) / "fig1_method_overview.pdf"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    boxes = [
        ("Image + question", (0.08, 0.66)),
        ("Blind question", (0.08, 0.25)),
        ("LVLM forward", (0.32, 0.66)),
        ("LVLM forward", (0.32, 0.25)),
        ("z_img", (0.53, 0.66)),
        ("z_blind", (0.53, 0.25)),
        ("D = z_blind - z_img", (0.72, 0.46)),
        ("SVD bands\nprobes\ninterventions", (0.90, 0.46)),
    ]
    for text, (x, y) in boxes:
        _box(ax, x, y, text)
    arrows = [
        ((0.17, 0.66), (0.27, 0.66)),
        ((0.17, 0.25), (0.27, 0.25)),
        ((0.41, 0.66), (0.48, 0.66)),
        ((0.41, 0.25), (0.48, 0.25)),
        ((0.58, 0.61), (0.67, 0.50)),
        ((0.58, 0.30), (0.67, 0.42)),
        ((0.79, 0.46), (0.84, 0.46)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2))
    ax.text(0.72, 0.14, "paired blind-reference correction geometry", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_fig2(figures_dir: str | Path) -> Path:
    path = Path(figures_dir) / "fig2_variance_vs_auroc.pdf"
    df = pd.read_csv("outputs/stage_c_deep/stage_c_topk_curve.csv")
    layers = [16, 20, 24, 32]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for layer in layers:
        group = df[df["layer"] == layer].sort_values("k")
        axes[0].plot(group["k"], group["explained_variance"], marker="o", label=f"L{layer}")
        axes[1].plot(group["k"], group["auroc"], marker="o", label=f"L{layer}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Top-K SVD coordinates")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[1].set_ylabel("FP/TN AUROC")
    axes[0].set_title("Variance concentrates early")
    axes[1].set_title("Discrimination needs broader coordinates")
    axes[1].axhline(0.5, color="black", linewidth=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_fig3(figures_dir: str | Path) -> Path:
    path = Path(figures_dir) / "fig3_condition_geometry.pdf"
    df = pd.read_csv("outputs/stage_b/stage_b_sample_scores.csv")
    keep = df[
        (df["layer"].isin([20, 24, 32]))
        & (
            ((df["view"] == "top_backbone") & (df["score"] == "top_1_4_l2_sq"))
            | ((df["view"] == "residual_tail") & (df["score"] == "band_257_1024_l2_sq"))
        )
        & (df["condition"].isin(["matched", "random_mismatch", "adversarial_mismatch"]))
    ].copy()
    keep["panel"] = np.where(keep["view"] == "top_backbone", "Top-4 backbone", "Tail 257-1024")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, panel in zip(axes, ["Top-4 backbone", "Tail 257-1024"], strict=True):
        data = keep[keep["panel"] == panel]
        labels = []
        values = []
        for condition in ["matched", "random_mismatch", "adversarial_mismatch"]:
            labels.append(condition.replace("_", "\n"))
            values.append(data[data["condition"] == condition]["value"].to_numpy())
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(panel)
        ax.set_ylabel("Squared coordinate norm")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_fig4(figures_dir: str | Path) -> Path:
    path = Path(figures_dir) / "fig4_intervention_dose.pdf"
    df = pd.read_csv("outputs/interventions/stage_e_first_token_check_dose_curve.csv")
    keep = df[
        (df["layer"] == 24)
        & (df["outcome_before"] == "TN")
        & (df["granularity"].isin(["last_token", "full_sequence"]))
        & (df["intervention"].isin(["ablate_tail_257_1024", "norm_matched_random_tail_control"]))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for (intervention, granularity), group in keep.groupby(["intervention", "granularity"]):
        label = f"{intervention.replace('_', ' ')} / {granularity}"
        group = group.sort_values("alpha")
        axes[0].plot(group["alpha"], group["mean_margin_delta_vs_baseline"], marker="o", label=label)
        axes[1].plot(group["alpha"], group["yes_rate_all"], marker="o", label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean yes-minus-no margin shift")
    axes[1].set_ylabel("TN flipped-to-Yes rate")
    for ax in axes:
        ax.set_xlabel("Alpha")
        ax.grid(alpha=0.25)
    axes[0].set_title("Tail ablation shifts margins")
    axes[1].set_title("Tail ablation damages TN decisions")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_fig5(figures_dir: str | Path) -> Path:
    path = Path(figures_dir) / "fig5_layered_geometry.pdf"
    spectrum = pd.read_csv("outputs/svd/effective_rank_summary.csv")
    probes = pd.read_csv("outputs/stage_p_stats/multiseed_probe_summary.csv")
    top4 = spectrum[spectrum["layer"].isin([16, 20, 24, 32])][["layer", "explained_variance_k4"]]
    full = probes[(probes["feature"] == "full_diff")][["layer", "auroc_mean"]]
    tail = probes[(probes["feature"] == "tail_257_1024")][["layer", "auroc_mean"]].rename(columns={"auroc_mean": "tail_auroc"})
    merged = top4.merge(full, on="layer").merge(tail, on="layer")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(merged["layer"], merged["explained_variance_k4"], marker="o", color="#2f6b9a", label="Top-4 explained variance")
    ax2.plot(merged["layer"], merged["auroc_mean"], marker="s", color="#b2453a", label="Full diff AUROC")
    ax2.plot(merged["layer"], merged["tail_auroc"], marker="^", color="#5f8f3f", label="Tail AUROC")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Top-4 explained variance")
    ax2.set_ylabel("FP/TN AUROC")
    ax1.grid(alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="lower left")
    ax1.set_title("Spectrum vs detection")

    sweep_path = Path("outputs/interventions/stage_e_fp_rescue_layer_sweep_64samples_rescue_margin_summary.csv")
    ax3 = axes[1]
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        keep = sweep[(sweep["granularity"] == "last_token") & (sweep["alpha"] == sweep["alpha"].max())].copy()
        if not keep.empty:
            summary = (
                keep.groupby("layer", as_index=False)
                .agg(max_gain=("mean_no_minus_yes_gain", "max"), max_rescue=("no_rate_all", "max"))
                .sort_values("layer")
            )
            ax3.plot(summary["layer"], summary["max_gain"], marker="o", label="Max no-minus-yes gain")
            ax3b = ax3.twinx()
            ax3b.plot(summary["layer"], summary["max_rescue"], marker="s", color="#8a5a9e", label="Max rescue rate")
            ax3.set_ylabel("Logit gain")
            ax3b.set_ylabel("Rescue rate")
            lines = ax3.get_lines() + ax3b.get_lines()
            ax3.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="upper left")
    ax3.set_xlabel("Layer")
    ax3.set_title("Intervention sensitivity")
    ax3.grid(alpha=0.25)
    fig.suptitle("Layered correction geometry summary", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_asset_index(tables_dir: str | Path, table_paths: dict[str, Path], figure_paths: dict[str, Path]) -> None:
    target = Path(tables_dir) / "stage_q_asset_index.md"
    lines = ["# Stage Q Paper Asset Index", "", "## Tables", ""]
    for key, path in table_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(["", "## Figures", ""])
    for key, path in figure_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _box(ax: plt.Axes, x: float, y: float, text: str) -> None:
    width, height = 0.14, 0.14
    rect = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="#f4f6f8",
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return sorted(groups.items())


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


if __name__ == "__main__":
    main()
