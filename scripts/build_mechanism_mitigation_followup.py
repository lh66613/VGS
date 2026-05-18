#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json


METHOD_SPECS = {
    "Full ICD": ("icd_blind", "full"),
    "Band5-16 ICD": ("icd_blind", "band5_16"),
    "Top4-complement ICD": ("icd_blind", "top4_complement"),
    "Random12 ICD": ("icd_blind", "random12"),
    "Random4-complement ICD": ("icd_blind", "random4_complement"),
    "Random-tail ICD": ("icd_blind", "random_tail_dim"),
    "Full VCD-diffusion": ("vcd_diffusion", "full"),
    "Tail VCD-diffusion": ("vcd_diffusion", "tail257_1024"),
}
COMPARISONS = [
    ("Band5-16 ICD", "Always ICD", "Band5-16 ICD vs Always ICD"),
    ("Top4-complement ICD", "Always ICD", "Top4-complement ICD vs Always ICD"),
    ("Band5-16 ICD", "Random12 ICD", "Band5-16 ICD vs Random12 ICD"),
    ("Top4-complement ICD", "Random4-complement ICD", "Top4-complement ICD vs Random4-complement ICD"),
    ("Tail VCD-diffusion", "Full VCD-diffusion", "Tail VCD vs Full VCD"),
    ("Gated ICD", "Always ICD", "Gated ICD vs Always ICD"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build follow-up mitigation experiments: Pareto, CI, reverse split, and cases.")
    parser.add_argument("--stage2-dir", default="outputs/mechanism_mitigation/stage2_subspace_vcd")
    parser.add_argument("--reverse-stage2-dir", default="outputs/mechanism_mitigation/stage2_reverse_subspace_vcd")
    parser.add_argument("--stage3-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/followup")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_followup(
        stage2_dir=args.stage2_dir,
        reverse_stage2_dir=args.reverse_stage2_dir,
        stage3_dir=args.stage3_dir,
        predictions_path=args.predictions,
        split=args.split,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_mitigation_followup_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_mitigation_followup", summary_path, "ok")
    print(summary_path)


def build_followup(
    stage2_dir: str | Path,
    reverse_stage2_dir: str | Path,
    stage3_dir: str | Path,
    predictions_path: str | Path,
    split: str,
    n_bootstrap: int,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    stage2_root = Path(stage2_dir)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}

    pareto_rows = _build_pareto(stage2_root / "alpha_sweep.csv", split)
    pareto_path = write_csv(output_root / "pareto_curve_points.csv", pareto_rows, _fieldnames(pareto_rows))
    pareto_fig = _plot_pareto(output_root / "pareto_curve.png", pareto_rows)

    sample_predictions = _load_sample_predictions(stage2_root / "sample_predictions.csv", split)
    calibrated = pd.read_csv(stage2_root / "subspace_vcd_results.csv")
    matched_rows = _matched_tp_safe_rows(calibrated)
    matched_path = write_csv(output_root / "matched_tp_safe_operating_points.csv", matched_rows, _fieldnames(matched_rows))
    random_rows = _random_distribution_rows(calibrated)
    random_path = write_csv(output_root / "random_control_distribution.csv", random_rows, _fieldnames(random_rows))
    method_outcomes = _stage2_method_outcomes(sample_predictions, calibrated)
    stage3_predictions = _stage3_method_predictions(stage3_dir, predictions, sample_predictions)
    method_outcomes.update({name: {sid: row["outcome"] for sid, row in values.items()} for name, values in stage3_predictions.items()})
    ci_rows = _bootstrap_comparisons(method_outcomes, predictions, n_bootstrap, seed)
    ci_path = write_csv(output_root / "bootstrap_comparisons.csv", ci_rows, _fieldnames(ci_rows))

    yes_rows = _yes_rate_audit(sample_predictions, calibrated, predictions, stage3_predictions)
    yes_path = write_csv(output_root / "yes_rate_no_bias_audit.csv", yes_rows, _fieldnames(yes_rows))
    case_rows, case_md = _case_studies(sample_predictions, calibrated, predictions)
    case_csv_path = write_csv(output_root / "case_studies.csv", case_rows, _fieldnames(case_rows))
    case_md_path = output_root / "case_studies.md"
    case_md_path.parent.mkdir(parents=True, exist_ok=True)
    case_md_path.write_text(case_md, encoding="utf-8")

    reverse_rows = _reverse_rows(Path(reverse_stage2_dir) / "subspace_vcd_results.csv")
    reverse_path = write_csv(output_root / "reverse_split_results.csv", reverse_rows, _fieldnames(reverse_rows))
    report_path = _write_report(
        output_root / "followup_summary.md",
        pareto_rows,
        ci_rows,
        reverse_rows,
        case_rows,
        matched_rows,
        random_rows,
        yes_rows,
    )
    return {
        "stage2_dir": str(stage2_dir),
        "reverse_stage2_dir": str(reverse_stage2_dir),
        "stage3_dir": str(stage3_dir),
        "pareto_points_path": str(pareto_path),
        "pareto_figure_path": pareto_fig,
        "matched_tp_safe_operating_points_path": str(matched_path),
        "random_control_distribution_path": str(random_path),
        "bootstrap_comparisons_path": str(ci_path),
        "yes_rate_no_bias_audit_path": str(yes_path),
        "reverse_split_results_path": str(reverse_path),
        "case_studies_csv_path": str(case_csv_path),
        "case_studies_markdown_path": str(case_md_path),
        "report_path": str(report_path),
        "num_pareto_rows": len(pareto_rows),
        "num_ci_rows": len(ci_rows),
        "num_matched_rows": len(matched_rows),
        "num_random_rows": len(random_rows),
        "num_yes_audit_rows": len(yes_rows),
        "num_case_rows": len(case_rows),
    }


def _build_pareto(alpha_sweep_path: Path, split: str) -> list[dict[str, Any]]:
    df = pd.read_csv(alpha_sweep_path)
    df = df[df["split"] == split].copy()
    rows: list[dict[str, Any]] = []
    for label, (operator, subspace) in METHOD_SPECS.items():
        group = df[(df["operator"] == operator) & (df["subspace"] == subspace)]
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "method_label": label,
                    "operator": operator,
                    "subspace": subspace,
                    "alpha": row.alpha,
                    "fp_reduction": row.fp_reduction,
                    "tp_preserved": row.tp_preserved,
                    "tp_damage": 1 - row.tp_preserved,
                    "accuracy_delta": row.accuracy_delta,
                    "yes_rate_after": row.yes_rate_after,
                }
            )
    return rows


def _load_sample_predictions(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sample-level Stage 2 predictions: {path}. "
            "Rerun scripts/analyze_mechanism_mitigation_stage2.py after the latest patch."
        )
    df = pd.read_csv(path)
    return df[df["split"] == split].copy()


def _stage2_method_outcomes(sample_predictions: pd.DataFrame, calibrated: pd.DataFrame) -> dict[str, dict[str, str]]:
    outcomes: dict[str, dict[str, str]] = {}
    for label, (operator, subspace) in METHOD_SPECS.items():
        selected = calibrated[(calibrated["operator"] == operator) & (calibrated["subspace"] == subspace)]
        if selected.empty:
            continue
        alpha = float(selected.iloc[0]["alpha"])
        rows = sample_predictions[
            (sample_predictions["operator"] == operator)
            & (sample_predictions["subspace"] == subspace)
            & (np.isclose(sample_predictions["alpha"], alpha))
        ]
        if rows.empty:
            continue
        outcomes[label] = {str(row.sample_id): str(row.final_outcome) for row in rows.itertuples(index=False)}
    return outcomes


def _stage3_method_predictions(
    stage3_dir: str | Path,
    predictions: dict[str, dict[str, Any]],
    sample_predictions: pd.DataFrame,
) -> dict[str, dict[str, dict[str, str]]]:
    sample_ids = sorted(set(sample_predictions["sample_id"].astype(str)))
    original = {
        sample_id: {
            "prediction": str(predictions[sample_id].get("parsed_prediction", "")),
            "outcome": str(predictions[sample_id]["outcome"]),
        }
        for sample_id in sample_ids
        if sample_id in predictions
    }
    icd_rows = _read_jsonl_map(Path(stage3_dir) / "stage_t_vcd_predictions_icd_blind.jsonl")
    always = dict(original)
    for sample_id, row in icd_rows.items():
        if sample_id in always and str(predictions[sample_id].get("outcome", "")) in {"FP", "TP"}:
            always[sample_id] = {
                "prediction": str(row.get("vcd_parsed_prediction", original[sample_id]["prediction"])),
                "outcome": str(row.get("vcd_outcome", original[sample_id]["outcome"])),
            }
    gated = dict(original)
    assignments_path = Path(stage3_dir) / "stage_t_verification_gate_assignments.csv"
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)
        gate = assignments[
            (assignments["score"] == "low_margin_plus_tail_257_1024_probe")
            & (np.isclose(assignments["target_trigger_rate_predicted_yes"], 0.3))
        ]
        for sample_id in gate["sample_id"].astype(str):
            if sample_id in gated and sample_id in icd_rows:
                gated[sample_id] = {
                    "prediction": str(icd_rows[sample_id].get("vcd_parsed_prediction", gated[sample_id]["prediction"])),
                    "outcome": str(icd_rows[sample_id].get("vcd_outcome", gated[sample_id]["outcome"])),
                }
    return {"Always ICD": always, "Gated ICD": gated}


def _matched_tp_safe_rows(calibrated: pd.DataFrame) -> list[dict[str, Any]]:
    labels = {
        ("icd_blind", "full"): "Full ICD",
        ("icd_blind", "band5_16"): "Band5-16 ICD",
        ("icd_blind", "random12"): "Random12 ICD",
        ("icd_blind", "top4_complement"): "Top4-complement ICD",
        ("icd_blind", "random4_complement"): "Random4-complement ICD",
        ("icd_blind", "random_tail_dim"): "Random-tail ICD",
        ("vcd_diffusion", "full"): "Full VCD-diffusion",
        ("vcd_diffusion", "tail257_1024"): "Tail VCD-diffusion",
    }
    rows: list[dict[str, Any]] = []
    for (operator, subspace), label in labels.items():
        selected = calibrated[(calibrated["operator"] == operator) & (calibrated["subspace"] == subspace)]
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "method_label": label,
                "operator": operator,
                "subspace": subspace,
                "selected_alpha": float(row["alpha"]),
                "calibration_constraint": "TP preserved >= 0.95",
                "calibration_constraint_satisfied": bool(row["calibration_tp_preserved"] >= 0.95),
                "calibration_fp_reduction": float(row["calibration_fp_reduction"]),
                "calibration_tp_preserved": float(row["calibration_tp_preserved"]),
                "test_fp_reduction": float(row["fp_reduction"]),
                "test_tp_preserved": float(row["tp_preserved"]),
                "test_accuracy_delta": float(row["accuracy_delta"]),
                "test_yes_rate_after": float(row["yes_rate_after"]),
            }
        )
    return rows


def _random_distribution_rows(calibrated: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("Band5-16 ICD", "icd_blind", "band5_16", "random12"),
        ("Top4-complement ICD", "icd_blind", "top4_complement", "random4_complement"),
        ("Full ICD", "icd_blind", "full", "random_tail_dim"),
        ("Tail VCD-diffusion", "vcd_diffusion", "tail257_1024", "random_tail_dim"),
    ]
    rows: list[dict[str, Any]] = []
    for target_label, operator, target_subspace, random_prefix in specs:
        target = calibrated[(calibrated["operator"] == operator) & (calibrated["subspace"] == target_subspace)]
        if target.empty:
            continue
        target_row = target.iloc[0]
        random = calibrated[
            (calibrated["operator"] == operator)
            & (calibrated["subspace"].astype(str).str.startswith(f"{random_prefix}_s"))
        ].copy()
        if random.empty:
            random = calibrated[
                (calibrated["operator"] == operator)
                & (calibrated["subspace"].astype(str) == random_prefix)
            ].copy()
        fp_values = random["fp_reduction"].to_numpy(dtype=float)
        tp_values = random["tp_preserved"].to_numpy(dtype=float)
        acc_values = random["accuracy_delta"].to_numpy(dtype=float)
        target_fp = float(target_row["fp_reduction"])
        target_tp = float(target_row["tp_preserved"])
        target_acc = float(target_row["accuracy_delta"])
        rows.append(
            {
                "target_method": target_label,
                "operator": operator,
                "target_subspace": target_subspace,
                "random_family": random_prefix,
                "n_random": int(len(random)),
                "target_alpha": float(target_row["alpha"]),
                "target_fp_reduction": target_fp,
                "target_tp_preserved": target_tp,
                "target_accuracy_delta": target_acc,
                "random_fp_mean": _nanmean(fp_values),
                "random_fp_std": _nanstd(fp_values),
                "random_fp_min": _nanmin(fp_values),
                "random_fp_max": _nanmax(fp_values),
                "random_tp_mean": _nanmean(tp_values),
                "random_acc_delta_mean": _nanmean(acc_values),
                "target_fp_percentile": _percentile(fp_values, target_fp),
                "target_outperforms_random_n": int(np.sum(target_fp > fp_values[np.isfinite(fp_values)])),
                "target_ties_or_outperforms_random_n": int(np.sum(target_fp >= fp_values[np.isfinite(fp_values)])),
            }
        )
    return rows


def _bootstrap_comparisons(
    method_outcomes: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_ids = sorted(
        set.intersection(
            *[set(outcomes) for outcomes in method_outcomes.values()]
        )
    ) if method_outcomes else []
    original = {sample_id: str(predictions[sample_id]["outcome"]) for sample_id in sample_ids if sample_id in predictions}
    sample_ids = [sample_id for sample_id in sample_ids if sample_id in original]
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(sample_ids), size=(n_bootstrap, len(sample_ids))) if sample_ids else np.empty((0, 0))
    rows: list[dict[str, Any]] = []
    for method_a, method_b, comparison in COMPARISONS:
        if method_a not in method_outcomes or method_b not in method_outcomes:
            continue
        for metric in ["fp_reduction", "tp_preserved", "accuracy_delta"]:
            point_a = _metric(original, method_outcomes[method_a], sample_ids, metric)
            point_b = _metric(original, method_outcomes[method_b], sample_ids, metric)
            boot = []
            for indices in boot_indices:
                ids = [sample_ids[int(idx)] for idx in indices]
                boot.append(
                    _metric(original, method_outcomes[method_a], ids, metric)
                    - _metric(original, method_outcomes[method_b], ids, metric)
                )
            rows.append(
                {
                    "comparison": comparison,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "point_a": point_a,
                    "point_b": point_b,
                    "diff_a_minus_b": point_a - point_b,
                    "ci_low": _nanquantile(boot, 0.025),
                    "ci_high": _nanquantile(boot, 0.975),
                    "n_bootstrap": n_bootstrap,
                    "n_samples": len(sample_ids),
                }
            )
    return rows


def _metric(
    original: dict[str, str],
    final: dict[str, str],
    sample_ids: list[str],
    metric: str,
) -> float:
    original_outcomes = [original[sample_id] for sample_id in sample_ids]
    final_outcomes = [final[sample_id] for sample_id in sample_ids]
    if metric == "fp_reduction":
        fp = sum(1 for item in original_outcomes if item == "FP")
        fixed = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "FP" and after == "TN")
        return fixed / fp if fp else math.nan
    if metric == "tp_preserved":
        tp = sum(1 for item in original_outcomes if item == "TP")
        kept = sum(1 for before, after in zip(original_outcomes, final_outcomes) if before == "TP" and after == "TP")
        return kept / tp if tp else math.nan
    if metric == "accuracy_delta":
        return _accuracy(final_outcomes) - _accuracy(original_outcomes)
    raise ValueError(f"Unknown metric: {metric}")


def _case_studies(
    sample_predictions: pd.DataFrame,
    calibrated: pd.DataFrame,
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    outcome_maps = _stage2_method_outcomes(sample_predictions, calibrated)
    method_rows = {
        label: _stage2_method_rows(sample_predictions, calibrated, *METHOD_SPECS[label])
        for label in METHOD_SPECS
        if label in outcome_maps
    }
    categories = [
        ("full_icd_hurts_tp_band_keeps", lambda sid: _orig(sid, predictions) == "TP" and outcome_maps.get("Full ICD", {}).get(sid) != "TP" and outcome_maps.get("Band5-16 ICD", {}).get(sid) == "TP"),
        ("band5_16_icd_fixes_fp", lambda sid: _orig(sid, predictions) == "FP" and outcome_maps.get("Band5-16 ICD", {}).get(sid) == "TN"),
        ("top4_complement_icd_fixes_fp", lambda sid: _orig(sid, predictions) == "FP" and outcome_maps.get("Top4-complement ICD", {}).get(sid) == "TN"),
        ("tail_vcd_fixes_full_vcd_fails", lambda sid: _orig(sid, predictions) == "FP" and outcome_maps.get("Tail VCD-diffusion", {}).get(sid) == "TN" and outcome_maps.get("Full VCD-diffusion", {}).get(sid) != "TN"),
        ("failure_fp_not_fixed", lambda sid: _orig(sid, predictions) == "FP" and all(outcome_maps.get(label, {}).get(sid) != "TN" for label in ["Band5-16 ICD", "Top4-complement ICD", "Tail VCD-diffusion"])),
    ]
    sample_ids = sorted({str(item) for item in sample_predictions["sample_id"].unique()})
    rows: list[dict[str, Any]] = []
    for category, predicate in categories:
        chosen = [sample_id for sample_id in sample_ids if sample_id in predictions and predicate(sample_id)][:8]
        for sample_id in chosen:
            pred = predictions[sample_id]
            row = {
                "category": category,
                "sample_id": sample_id,
                "label": pred.get("label", ""),
                "original_prediction": pred.get("parsed_prediction", ""),
                "original_outcome": pred.get("outcome", ""),
                "question": pred.get("question", ""),
                "image": pred.get("image", ""),
                "image_path": pred.get("image_path", ""),
            }
            for label in ["Full ICD", "Band5-16 ICD", "Top4-complement ICD", "Full VCD-diffusion", "Tail VCD-diffusion"]:
                method_row = method_rows.get(label, {}).get(sample_id, {})
                row[f"{label}_outcome"] = outcome_maps.get(label, {}).get(sample_id, "")
                row[f"{label}_prediction"] = method_row.get("final_prediction", "")
                row[f"{label}_base_margin"] = method_row.get("base_no_minus_yes_logit", "")
                row[f"{label}_dmargin"] = method_row.get("dmargin_no_minus_yes", "")
                row[f"{label}_adjusted_margin"] = method_row.get("adjusted_no_minus_yes_logit", "")
            row["explanation"] = _case_explanation(category)
            rows.append(row)
    return rows, _case_markdown(rows)


def _stage2_method_rows(
    sample_predictions: pd.DataFrame,
    calibrated: pd.DataFrame,
    operator: str,
    subspace: str,
) -> dict[str, dict[str, Any]]:
    selected = calibrated[(calibrated["operator"] == operator) & (calibrated["subspace"] == subspace)]
    if selected.empty:
        return {}
    alpha = float(selected.iloc[0]["alpha"])
    rows = sample_predictions[
        (sample_predictions["operator"] == operator)
        & (sample_predictions["subspace"] == subspace)
        & (np.isclose(sample_predictions["alpha"], alpha))
    ]
    return {str(row.sample_id): row._asdict() for row in rows.itertuples(index=False)}


def _reverse_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df["tp_safe"] = df["tp_preserved"] >= 0.95
    keep = df[df["method"].isin([_method_name(*spec) for spec in METHOD_SPECS.values()])].copy()
    keep = keep.sort_values(["tp_safe", "fp_reduction", "tp_preserved"], ascending=[False, False, False])
    return keep.head(20).to_dict(orient="records")


def _plot_pareto(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, group in df.groupby("method_label"):
        ordered = group.sort_values("tp_damage")
        ax.plot(ordered["tp_damage"], ordered["fp_reduction"], marker="o", label=label)
    ax.set_xlabel("TP damage = 1 - TP preserved")
    ax.set_ylabel("FP reduction")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def _yes_rate_audit(
    sample_predictions: pd.DataFrame,
    calibrated: pd.DataFrame,
    predictions: dict[str, dict[str, Any]],
    stage3_predictions: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    sample_ids = sorted(set(sample_predictions["sample_id"].astype(str)))
    method_prediction_maps: dict[str, dict[str, dict[str, str]]] = {
        "Base": {
            sample_id: {
                "prediction": str(predictions[sample_id].get("parsed_prediction", "")),
                "outcome": str(predictions[sample_id].get("outcome", "")),
            }
            for sample_id in sample_ids
            if sample_id in predictions
        }
    }
    for label, (operator, subspace) in {
        "Full ICD": ("icd_blind", "full"),
        "Band5-16 ICD": ("icd_blind", "band5_16"),
        "Random12 ICD": ("icd_blind", "random12"),
        "Top4-complement ICD": ("icd_blind", "top4_complement"),
        "Random4-complement ICD": ("icd_blind", "random4_complement"),
        "Tail VCD-diffusion": ("vcd_diffusion", "tail257_1024"),
    }.items():
        rows = _stage2_method_rows(sample_predictions, calibrated, operator, subspace)
        if not rows:
            continue
        method_prediction_maps[label] = {
            sample_id: {
                "prediction": str(row.get("final_prediction", "")),
                "outcome": str(row.get("final_outcome", "")),
            }
            for sample_id, row in rows.items()
        }
    method_prediction_maps.update(stage3_predictions)

    rows: list[dict[str, Any]] = []
    for method_label, final_map in method_prediction_maps.items():
        ids = [sample_id for sample_id in sample_ids if sample_id in predictions and sample_id in final_map]
        if not ids:
            continue
        final_predictions = [final_map[sample_id]["prediction"] for sample_id in ids]
        final_outcomes = [final_map[sample_id]["outcome"] for sample_id in ids]
        original_outcomes = [str(predictions[sample_id].get("outcome", "")) for sample_id in ids]
        labels = [str(predictions[sample_id].get("label", "")) for sample_id in ids]
        rows.append(
            {
                "method_label": method_label,
                "n": len(ids),
                "overall_yes_rate": _yes_rate(final_predictions),
                "tp_yes_rate": _conditional_yes_rate(final_predictions, original_outcomes, "TP"),
                "fp_yes_rate": _conditional_yes_rate(final_predictions, original_outcomes, "FP"),
                "tn_yes_rate": _conditional_yes_rate(final_predictions, original_outcomes, "TN"),
                "fn_rate_after": _fn_rate_after(final_predictions, labels),
                "unknown_rate": sum(1 for item in final_predictions if item not in {"yes", "no"}) / len(ids),
                "accuracy_after": _accuracy(final_outcomes),
                "fp_after": sum(1 for item in final_outcomes if item == "FP"),
                "tp_after": sum(1 for item in final_outcomes if item == "TP"),
            }
        )
    return rows


def _write_report(
    path: Path,
    pareto_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
    reverse_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    yes_rows: list[dict[str, Any]],
) -> Path:
    lines = ["# Mechanism Mitigation Follow-Up", ""]
    if matched_rows:
        lines.extend(
            [
                "## Matched TP-Safe Operating Points",
                "",
                "| Method | Alpha | Calib TP Safe | Calib FP Reduction | Calib TP Preserved | Test FP Reduction | Test TP Preserved | Test Acc Delta |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in matched_rows:
            safe = "yes" if row["calibration_constraint_satisfied"] else "no"
            lines.append(
                f"| {row['method_label']} | {row['selected_alpha']:.3g} | {safe} | "
                f"{_fmt(row['calibration_fp_reduction'])} | {_fmt(row['calibration_tp_preserved'])} | "
                f"{_fmt(row['test_fp_reduction'])} | {_fmt(row['test_tp_preserved'])} | {_fmt(row['test_accuracy_delta'])} |"
            )
        lines.append("")
    if random_rows:
        lines.extend(
            [
                "## Random Subspace Distribution",
                "",
                "| Target | Random Family | N | Target FP | Random FP Mean | Random FP Range | Target Percentile | Outperforms |",
                "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in random_rows:
            lines.append(
                f"| {row['target_method']} | `{row['random_family']}` | {row['n_random']} | "
                f"{_fmt(row['target_fp_reduction'])} | {_fmt(row['random_fp_mean'])} | "
                f"[{_fmt(row['random_fp_min'])}, {_fmt(row['random_fp_max'])}] | "
                f"{_fmt(row['target_fp_percentile'])} | {row['target_outperforms_random_n']}/{row['n_random']} |"
            )
        lines.append("")
    pareto = pd.DataFrame(pareto_rows)
    if not pareto.empty:
        safe = pareto[pareto["tp_preserved"] >= 0.95].sort_values("fp_reduction", ascending=False)
        lines.extend(["## Pareto TP-Safe Points", "", "| Method | Alpha | FP Reduction | TP Preserved | TP Damage | Acc Delta |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in safe.head(12).itertuples(index=False):
            lines.append(f"| {row.method_label} | {row.alpha:.2g} | {row.fp_reduction:.3f} | {row.tp_preserved:.3f} | {row.tp_damage:.3f} | {row.accuracy_delta:.3f} |")
        lines.append("")
    if ci_rows:
        lines.extend(["## Bootstrap Comparisons", "", "| Comparison | Metric | A | B | Diff | 95% CI |", "| --- | --- | ---: | ---: | ---: | --- |"])
        for row in ci_rows:
            lines.append(f"| {row['comparison']} | {row['metric']} | {_fmt(row['point_a'])} | {_fmt(row['point_b'])} | {_fmt(row['diff_a_minus_b'])} | [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] |")
        lines.append("")
    if yes_rows:
        lines.extend(
            [
                "## Yes-Rate / No-Bias Audit",
                "",
                "| Method | Overall Yes | TP Yes | FP Yes | TN Yes | FN Rate After | Accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in yes_rows:
            lines.append(
                f"| {row['method_label']} | {_fmt(row['overall_yes_rate'])} | {_fmt(row['tp_yes_rate'])} | "
                f"{_fmt(row['fp_yes_rate'])} | {_fmt(row['tn_yes_rate'])} | {_fmt(row['fn_rate_after'])} | "
                f"{_fmt(row['accuracy_after'])} |"
            )
        lines.append("")
    if reverse_rows:
        lines.extend(["## Reverse Split Top Rows", "", "| Method | FP Reduction | TP Preserved | Acc Delta |", "| --- | ---: | ---: | ---: |"])
        for row in reverse_rows[:10]:
            lines.append(f"| {row.get('method','')} | {_fmt(row.get('fp_reduction'))} | {_fmt(row.get('tp_preserved'))} | {_fmt(row.get('accuracy_delta'))} |")
        lines.append("")
    if case_rows:
        counts = pd.Series([row["category"] for row in case_rows]).value_counts()
        lines.extend(["## Case Study Counts", ""])
        for category, count in counts.items():
            lines.append(f"- `{category}`: {count}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _case_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Mitigation Case Studies", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['category']} - {row['sample_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Image: `{row['image_path']}`",
                f"- Label/base: `{row['label']}` / `{row['original_prediction']}` / `{row['original_outcome']}`",
                f"- Full ICD: `{row.get('Full ICD_prediction','')}` / `{row.get('Full ICD_outcome','')}`",
                f"  - margin: base `{_fmt(row.get('Full ICD_base_margin'))}`, delta `{_fmt(row.get('Full ICD_dmargin'))}`, adjusted `{_fmt(row.get('Full ICD_adjusted_margin'))}`",
                f"- Band5-16 ICD: `{row.get('Band5-16 ICD_prediction','')}` / `{row.get('Band5-16 ICD_outcome','')}`",
                f"  - margin: base `{_fmt(row.get('Band5-16 ICD_base_margin'))}`, delta `{_fmt(row.get('Band5-16 ICD_dmargin'))}`, adjusted `{_fmt(row.get('Band5-16 ICD_adjusted_margin'))}`",
                f"- Top4-complement ICD: `{row.get('Top4-complement ICD_prediction','')}` / `{row.get('Top4-complement ICD_outcome','')}`",
                f"  - margin: base `{_fmt(row.get('Top4-complement ICD_base_margin'))}`, delta `{_fmt(row.get('Top4-complement ICD_dmargin'))}`, adjusted `{_fmt(row.get('Top4-complement ICD_adjusted_margin'))}`",
                f"- Full VCD-diffusion: `{row.get('Full VCD-diffusion_prediction','')}` / `{row.get('Full VCD-diffusion_outcome','')}`",
                f"  - margin: base `{_fmt(row.get('Full VCD-diffusion_base_margin'))}`, delta `{_fmt(row.get('Full VCD-diffusion_dmargin'))}`, adjusted `{_fmt(row.get('Full VCD-diffusion_adjusted_margin'))}`",
                f"- Tail VCD-diffusion: `{row.get('Tail VCD-diffusion_prediction','')}` / `{row.get('Tail VCD-diffusion_outcome','')}`",
                f"  - margin: base `{_fmt(row.get('Tail VCD-diffusion_base_margin'))}`, delta `{_fmt(row.get('Tail VCD-diffusion_dmargin'))}`, adjusted `{_fmt(row.get('Tail VCD-diffusion_adjusted_margin'))}`",
                f"- Interpretation: {row.get('explanation', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def _case_explanation(category: str) -> str:
    return {
        "full_icd_hurts_tp_band_keeps": "Full ICD moves the Yes/No margin past the No boundary, while filtered ICD leaves the true positive on the Yes side.",
        "band5_16_icd_fixes_fp": "Band5-16 supplies enough No-directed correction to fix the false positive without relying on the full, TP-damaging ICD vector.",
        "top4_complement_icd_fixes_fp": "Removing the dominant top-4 backbone keeps a No-directed correction that repairs this false positive.",
        "tail_vcd_fixes_full_vcd_fails": "The tail-filtered VCD correction catches a false positive that the full diffusion contrast misses.",
        "failure_fp_not_fixed": "None of the tested filtered corrections moves the sample across the decision boundary.",
    }.get(category, "")


def _orig(sample_id: str, predictions: dict[str, dict[str, Any]]) -> str:
    return str(predictions[sample_id].get("outcome", ""))


def _accuracy(outcomes: list[str]) -> float:
    valid = [item for item in outcomes if item in {"TP", "TN", "FP", "FN"}]
    if not valid:
        return math.nan
    return sum(1 for item in valid if item in {"TP", "TN"}) / len(valid)


def _yes_rate(predictions: list[str]) -> float:
    return sum(1 for item in predictions if item == "yes") / len(predictions) if predictions else math.nan


def _conditional_yes_rate(predictions: list[str], original_outcomes: list[str], outcome: str) -> float:
    kept = [pred for pred, original in zip(predictions, original_outcomes) if original == outcome]
    return _yes_rate(kept)


def _fn_rate_after(predictions: list[str], labels: list[str]) -> float:
    gold_yes = [pred for pred, label in zip(predictions, labels) if label == "yes"]
    return sum(1 for pred in gold_yes if pred != "yes") / len(gold_yes) if gold_yes else math.nan


def _read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def _method_name(operator: str, subspace: str) -> str:
    names = {
        "full": "Full",
        "top4": "Top4",
        "top16": "Top16",
        "band5_16": "Band5-16",
        "tail257_1024": "Tail257-1024",
        "top4_complement": "Top4-Complement",
        "random12": "Random12",
        "random4_complement": "Random4-Complement",
        "random_tail_dim": "RandomTailDim",
    }
    return f"{names.get(subspace, subspace)}-{operator}"


def _case_safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nanquantile(values: list[float], q: float) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if len(arr) else math.nan


def _nanmean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else math.nan


def _nanstd(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else math.nan


def _nanmin(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.min(values)) if len(values) else math.nan


def _nanmax(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else math.nan


def _percentile(values: np.ndarray, target: float) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    return float(100 * np.mean(values <= target))


def _fmt(value: Any) -> str:
    number = _case_safe_float(value)
    return "" if not np.isfinite(number) else f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
