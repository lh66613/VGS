#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _rate(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return float("nan")
    return float((series == value).mean())


def _summarize_dose(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["layer", "outcome_before", "intervention", "granularity", "alpha"]
    for key, group in df.groupby(keys, dropna=False):
        parsed = group["parsed_prediction"].astype(str)
        valid = group[group["outcome_after"].isin(["TP", "TN", "FP", "FN"])]
        correct = valid[valid["outcome_after"].isin(["TP", "TN"])]
        margins = pd.to_numeric(group["yes_minus_no_logit"], errors="coerce").dropna()
        rows.append(
            {
                "layer": key[0],
                "outcome_before": key[1],
                "intervention": key[2],
                "granularity": key[3],
                "alpha": key[4],
                "n": len(group),
                "yes_count": int((parsed == "yes").sum()),
                "no_count": int((parsed == "no").sum()),
                "unknown_count": int((parsed == "unknown").sum()),
                "yes_rate_all": _rate(parsed, "yes"),
                "no_rate_all": _rate(parsed, "no"),
                "unknown_rate": _rate(parsed, "unknown"),
                "accuracy_over_valid": float(len(correct) / len(valid)) if len(valid) else np.nan,
                "mean_yes_minus_no_logit": float(margins.mean()) if len(margins) else np.nan,
                "median_yes_minus_no_logit": float(margins.median()) if len(margins) else np.nan,
                "std_yes_minus_no_logit": float(margins.std()) if len(margins) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _first_flip_thresholds(df: pd.DataFrame, target_prediction: str) -> pd.DataFrame:
    rows = []
    grouped = df.sort_values("alpha").groupby(["layer", "granularity", "sample_id"], dropna=False)
    for (layer, granularity, sample_id), group in grouped:
        target_rows = group[group["parsed_prediction"].astype(str) == target_prediction]
        rows.append(
            {
                "layer": int(layer),
                "granularity": granularity,
                "sample_id": sample_id,
                "target_prediction": target_prediction,
                "first_target_alpha": float(target_rows["alpha"].min()) if len(target_rows) else np.nan,
                "ever_target": bool(len(target_rows)),
                "ever_unknown": bool((group["parsed_prediction"].astype(str) == "unknown").any()),
                "baseline_margin": float(group["baseline_margin"].iloc[0]),
                "last_alpha_margin": float(group.sort_values("alpha")["yes_minus_no_logit"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "granularity", "sample_id"]).reset_index(drop=True)


def _add_baseline_margin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baseline = (
        out[out["intervention"] == "baseline"]
        .set_index(["layer", "sample_id"])["yes_minus_no_logit"]
        .to_dict()
    )
    out["baseline_margin"] = [
        baseline.get((row.layer, row.sample_id), np.nan) for row in out.itertuples(index=False)
    ]
    out["margin_delta_vs_baseline"] = out["yes_minus_no_logit"] - out["baseline_margin"]
    return out


def _plot_ablation_dose(dose: pd.DataFrame, plot_dir: Path, artifact_prefix: str) -> str | None:
    ablation = dose[dose["intervention"].str.startswith("ablate_tail")]
    if ablation.empty:
        return None
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for (layer, granularity), group in ablation.groupby(["layer", "granularity"]):
        ordered = group.sort_values("alpha")
        label = f"L{layer} {granularity}"
        axes[0].plot(ordered["alpha"], ordered["yes_rate_all"], marker="o", label=label)
        axes[1].plot(ordered["alpha"], ordered["median_yes_minus_no_logit"], marker="o", label=label)
    axes[0].set_title("Tail ablation answer flips")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Yes rate over all samples")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("Tail ablation yes-no margin")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("median logit(Yes) - logit(No)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    ensure_dir(plot_dir)
    path = plot_dir / f"{artifact_prefix}_tail_ablation_dose.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_control_unknown(dose: pd.DataFrame, plot_dir: Path, artifact_prefix: str) -> str | None:
    controls = dose[dose["intervention"].str.contains("control", regex=False)]
    if controls.empty:
        return None
    import matplotlib.pyplot as plt

    layers = sorted(controls["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), constrained_layout=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        layer_df = controls[controls["layer"] == layer]
        for intervention, group in layer_df.groupby("intervention"):
            ordered = group.sort_values("alpha")
            ax.plot(ordered["alpha"], ordered["unknown_rate"], marker="o", label=intervention)
        ax.set_title(f"L{layer} control format failures")
        ax.set_xlabel("alpha")
        ax.set_ylabel("unknown rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    ensure_dir(plot_dir)
    path = plot_dir / f"{artifact_prefix}_control_unknown_rate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def analyze_stage_e_results(
    results_path: str | Path,
    output_dir: str | Path,
    plot_dir: str | Path,
    target_prediction: str,
    artifact_prefix: str,
) -> dict[str, object]:
    ensure_dir(output_dir)
    df = pd.read_csv(results_path)
    df = _add_baseline_margin(df)
    dose = _summarize_dose(df)
    dose_path = Path(output_dir) / f"{artifact_prefix}_dose_curve.csv"
    dose.to_csv(dose_path, index=False)

    ablation = df[df["intervention"].str.startswith("ablate_tail")].copy()
    thresholds = _first_flip_thresholds(ablation, target_prediction) if len(ablation) else pd.DataFrame()
    threshold_path = Path(output_dir) / f"{artifact_prefix}_flip_thresholds.csv"
    thresholds.to_csv(threshold_path, index=False)

    threshold_summary = []
    if len(thresholds):
        for (layer, granularity), group in thresholds.groupby(["layer", "granularity"]):
            target = group["first_target_alpha"].dropna()
            threshold_summary.append(
                {
                    "layer": int(layer),
                    "granularity": granularity,
                    "n": int(len(group)),
                    "ever_target_count": int(group["ever_target"].sum()),
                    "ever_unknown_count": int(group["ever_unknown"].sum()),
                    "median_first_target_alpha": float(target.median()) if len(target) else np.nan,
                    "min_first_target_alpha": float(target.min()) if len(target) else np.nan,
                    "max_first_target_alpha": float(target.max()) if len(target) else np.nan,
                }
            )
    threshold_summary_df = pd.DataFrame(threshold_summary)
    threshold_summary_path = Path(output_dir) / f"{artifact_prefix}_flip_threshold_summary.csv"
    threshold_summary_df.to_csv(threshold_summary_path, index=False)

    ablation_plot = _plot_ablation_dose(dose, Path(plot_dir), artifact_prefix)
    control_plot = _plot_control_unknown(dose, Path(plot_dir), artifact_prefix)
    return {
        "results_path": str(results_path),
        "dose_curve_path": str(dose_path),
        "flip_thresholds_path": str(threshold_path),
        "flip_threshold_summary_path": str(threshold_summary_path),
        "ablation_plot": ablation_plot,
        "control_unknown_plot": control_plot,
        "num_rows": int(len(df)),
        "target_prediction": target_prediction,
        "artifact_prefix": artifact_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved Stage E intervention pilot results.")
    parser.add_argument("--results", default="outputs/interventions/intervention_pilot_results.csv")
    parser.add_argument("--output-dir", default="outputs/interventions")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--target-prediction", default="yes", choices=["yes", "no", "unknown"])
    parser.add_argument("--artifact-prefix", default="stage_e")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()
    payload = analyze_stage_e_results(
        args.results,
        args.output_dir,
        args.plot_dir,
        args.target_prediction,
        args.artifact_prefix,
    )
    summary_path = write_json(Path(args.output_dir) / "analyze_stage_e_results_summary.json", payload)
    append_experiment_log(args.log_path, "analyze_stage_e_results", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
