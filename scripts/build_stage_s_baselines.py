#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage S baseline positioning tables.")
    parser.add_argument("--output-dir", default="outputs/stage_s_baselines")
    parser.add_argument("--note-path", default="notes/baseline_positioning.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {"output_dir": args.output_dir, "note_path": args.note_path}
    if not args.dry_run:
        payload.update(build_stage_s_baselines(args.output_dir, args.note_path))
    summary_path = write_json(Path(args.output_dir) / "build_stage_s_baselines_summary.json", payload)
    append_experiment_log(args.log_path, "build_stage_s_baselines", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


def build_stage_s_baselines(output_dir: str | Path, note_path: str | Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    detection_rows = _detection_rows()
    mitigation_rows = _mitigation_rows()
    detection_path = write_csv(
        Path(output_dir) / "detection_baseline_comparison.csv",
        detection_rows,
        _fieldnames(detection_rows),
    )
    mitigation_path = write_csv(
        Path(output_dir) / "mitigation_baseline_comparison.csv",
        mitigation_rows,
        _fieldnames(mitigation_rows),
    )
    _write_note(note_path, detection_rows, mitigation_rows)
    return {
        "detection_path": str(detection_path),
        "mitigation_path": str(mitigation_path),
        "note_path": str(note_path),
        "num_detection_rows": len(detection_rows),
        "num_mitigation_rows": len(mitigation_rows),
    }


def _detection_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_path = Path("outputs/probes/feature_comparison.csv")
    if feature_path.exists():
        df = pd.read_csv(feature_path)
        for family, baseline, baseline_type, note in [
            ("raw_img", "raw image-state hidden probe", "single_forward_hidden_state", "Proxy for a single-forward hidden-state detector."),
            ("raw_blind", "raw blind-state hidden probe", "text_only_hidden_state", "Text-only hidden-state baseline."),
            ("difference", "paired full blind-image difference", "ours_paired_difference", "Primary paired-difference detector."),
            ("projected_difference", "top-K SVD paired difference", "ours_svd_coordinates", "Top-SVD coordinate detector."),
            ("random_difference", "random projected difference", "random_projection_control", "Random projection control."),
            ("pca_img", "image-state PCA hidden probe", "single_forward_hidden_state", "Single-forward PCA baseline."),
        ]:
            group = df[df["feature_family"] == family]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(_metric_row("pope_full_probe", baseline, baseline_type, best, note))
    stage_p_path = Path("outputs/stage_p_stats/multiseed_probe_summary.csv")
    if stage_p_path.exists():
        df = pd.read_csv(stage_p_path)
        for feature, baseline in [
            ("full_diff", "paired full difference, 5-seed"),
            ("top_256", "top-256 SVD coordinates, 5-seed"),
            ("tail_257_1024", "tail 257-1024 SVD coordinates, 5-seed"),
            ("top_4", "top-4 SVD coordinates, 5-seed"),
        ]:
            group = df[df["feature"] == feature]
            if group.empty:
                continue
            best = group.sort_values("auroc_mean", ascending=False).iloc[0]
            rows.append(
                {
                    "source": "stage_p_multiseed",
                    "baseline": baseline,
                    "baseline_type": "robustness_summary",
                    "target": "POPE FP-vs-TN",
                    "availability": "available",
                    "layer": int(best["layer"]),
                    "k": feature,
                    "n": "",
                    "auroc": float(best["auroc_mean"]),
                    "auprc": float(best["auprc_mean"]),
                    "accuracy": float(best["accuracy_mean"]),
                    "f1": float(best["f1_mean"]),
                    "ci95_low": float(best["auroc_ci95_low"]),
                    "ci95_high": float(best["auroc_ci95_high"]),
                    "notes": "Mean over 5 stratified split/logistic seeds.",
                }
            )
    stage_l_path = Path("outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv")
    if stage_l_path.exists():
        df = pd.read_csv(stage_l_path)
        for method in ["pls_fp_tn", "fisher_fp_tn", "contrastive_pca"]:
            group = df[df["method"] == method]
            if group.empty:
                continue
            best = group.sort_values("auroc", ascending=False).iloc[0]
            rows.append(_metric_row("stage_l_evidence_subspace", f"evidence-specific {method}", "ours_evidence_specific", best, "Compact Stage L evidence-specific subspace."))
    rows.extend(_logit_subset_detection_rows())
    rows.extend(
        [
            _unavailable_row("image-text similarity baseline", "image_text_similarity", "No CLIP/image-text similarity artifact is available in this run."),
            _unavailable_row("HALP-style detector", "external_detection_baseline", "No reproduced HALP implementation/artifact is available; raw image-state hidden probe is included as a single-forward proxy."),
        ]
    )
    return rows


def _logit_subset_detection_rows() -> list[dict[str, Any]]:
    path = Path("outputs/stage_m_local_rescue/local_rescue_results.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    base = df[(df["intervention"] == "baseline") & (df["outcome_before"].isin(["FP", "TN"]))].copy()
    if base.empty:
        return []
    y = (base["outcome_before"] == "FP").astype(int).to_numpy()
    specs = [
        ("yes/no margin baseline", "yes_no_margin", base["yes_minus_no_logit"].to_numpy(), "Higher yes-minus-no logit indicates FP risk."),
        ("binary entropy baseline", "entropy", base["binary_entropy"].to_numpy(), "Higher entropy indicates uncertainty."),
        ("Stage M FP-risk score", "ours_fp_risk_subset", base["fp_risk_score"].to_numpy(), "Train-bank FP-risk score on the Stage M first-token subset."),
        ("tail norm score", "ours_tail_energy_subset", base["tail_norm"].to_numpy(), "Tail norm score on the Stage M first-token subset."),
    ]
    rows = []
    for baseline, baseline_type, score, note in specs:
        rows.append(
            {
                "source": "stage_m_first_token_subset",
                "baseline": baseline,
                "baseline_type": baseline_type,
                "target": "Stage M subset FP-vs-TN",
                "availability": "available_subset_only",
                "layer": 32,
                "k": "",
                "n": int(len(y)),
                "auroc": _safe_metric(y, score, roc_auc_score),
                "auprc": _safe_metric(y, score, average_precision_score),
                "accuracy": "",
                "f1": "",
                "ci95_low": "",
                "ci95_high": "",
                "notes": note,
            }
        )
    return rows


def _mitigation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "source": "stage_m_local_rescue",
            "baseline": "no intervention",
            "baseline_type": "none",
            "availability": "available_subset_only",
            "layer": 32,
            "gate": "baseline",
            "intervention": "baseline",
            "retrieval_mode": "none",
            "alpha": 0.0,
            "fp_n": 32,
            "fp_reduction_or_rescue_rate": 0.0,
            "tn_preservation": 1.0,
            "tp_preservation": 1.0,
            "unknown_rate": 0.0,
            "mean_no_minus_yes_gain": 0.0,
            "notes": "Baseline rows from Stage M first-token subset.",
        }
    )
    summary_path = Path("outputs/stage_m_local_rescue/local_rescue_summary.csv")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for label, interventions in [
            ("random steering control", ["random_tn_mean_correction"]),
            ("global mean correction", ["global_tn_mean_correction"]),
            ("local TN correction", ["same_object_tn_mean_correction", "svd_knn_tn_mean_correction", "tail_knn_tn_mean_correction"]),
        ]:
            setting = _best_mitigation_setting(df, interventions)
            if setting:
                setting["baseline"] = label
                rows.append(setting)
    rows.extend(
        [
            {
                "source": "not_run",
                "baseline": "VCD or ICD baseline",
                "baseline_type": "external_mitigation_baseline",
                "availability": "unavailable",
                "layer": "",
                "gate": "",
                "intervention": "",
                "retrieval_mode": "",
                "alpha": "",
                "fp_n": "",
                "fp_reduction_or_rescue_rate": "",
                "tn_preservation": "",
                "tp_preservation": "",
                "unknown_rate": "",
                "mean_no_minus_yes_gain": "",
                "notes": "No VCD/ICD run or implementation artifact is available in this repository state.",
            },
            {
                "source": "not_run",
                "baseline": "evidence-specific correction",
                "baseline_type": "ours_evidence_specific_mitigation",
                "availability": "not_wired",
                "layer": "",
                "gate": "",
                "intervention": "",
                "retrieval_mode": "",
                "alpha": "",
                "fp_n": "",
                "fp_reduction_or_rescue_rate": "",
                "tn_preservation": "",
                "tp_preservation": "",
                "unknown_rate": "",
                "mean_no_minus_yes_gain": "",
                "notes": "Stage L evidence-specific subspaces are detection/geometry artifacts; not wired into mitigation steering yet.",
            },
        ]
    )
    return rows


def _best_mitigation_setting(df: pd.DataFrame, interventions: list[str]) -> dict[str, Any] | None:
    fp = df[(df["outcome_before"] == "FP") & (df["intervention"].isin(interventions))].copy()
    if fp.empty:
        return None
    fp = fp.sort_values(["fp_rescue_rate", "mean_no_minus_yes_gain"], ascending=False)
    best = fp.iloc[0]
    same = df[
        (df["gate"] == best["gate"])
        & (df["intervention"] == best["intervention"])
        & (df["retrieval_mode"] == best["retrieval_mode"])
        & (df["alpha"] == best["alpha"])
    ]
    tn = same[same["outcome_before"] == "TN"]
    tp = same[same["outcome_before"] == "TP"]
    tn_preservation = 1.0 - float(tn.iloc[0]["tn_damage_rate"]) if not tn.empty and not pd.isna(tn.iloc[0]["tn_damage_rate"]) else ""
    tp_preservation = 1.0 - float(tp.iloc[0]["tp_damage_rate"]) if not tp.empty and not pd.isna(tp.iloc[0]["tp_damage_rate"]) else ""
    control_note = ""
    if tn_preservation == "" or tp_preservation == "":
        control_note = " TN/TP preservation is undefined for this gated setting because no TN/TP samples passed the gate."
    return {
        "source": "stage_m_local_rescue",
        "baseline": "",
        "baseline_type": "mitigation_or_rescue",
        "availability": "available_subset_only",
        "layer": int(best["layer"]),
        "gate": best["gate"],
        "intervention": best["intervention"],
        "retrieval_mode": best["retrieval_mode"],
        "alpha": float(best["alpha"]),
        "fp_n": int(best["n"]),
        "fp_reduction_or_rescue_rate": float(best["fp_rescue_rate"]),
        "tn_preservation": tn_preservation,
        "tp_preservation": tp_preservation,
        "unknown_rate": float(best["unknown_rate"]),
        "mean_no_minus_yes_gain": float(best["mean_no_minus_yes_gain"]),
        "notes": "Best setting by FP rescue rate, tie-broken by no-minus-yes gain; Stage M first-token subset." + control_note,
    }


def _metric_row(source: str, baseline: str, baseline_type: str, row: pd.Series, notes: str) -> dict[str, Any]:
    return {
        "source": source,
        "baseline": baseline,
        "baseline_type": baseline_type,
        "target": "POPE FP-vs-TN",
        "availability": "available",
        "layer": int(row.get("layer", -1)),
        "k": row.get("k", ""),
        "n": int(row.get("num_samples", row.get("test_size", 0))) if not pd.isna(row.get("num_samples", row.get("test_size", 0))) else "",
        "auroc": float(row.get("auroc", math.nan)),
        "auprc": float(row.get("auprc", math.nan)),
        "accuracy": float(row.get("accuracy", math.nan)),
        "f1": float(row.get("f1", math.nan)),
        "ci95_low": "",
        "ci95_high": "",
        "notes": notes,
    }


def _unavailable_row(baseline: str, baseline_type: str, notes: str) -> dict[str, Any]:
    return {
        "source": "not_run",
        "baseline": baseline,
        "baseline_type": baseline_type,
        "target": "",
        "availability": "unavailable",
        "layer": "",
        "k": "",
        "n": "",
        "auroc": "",
        "auprc": "",
        "accuracy": "",
        "f1": "",
        "ci95_low": "",
        "ci95_high": "",
        "notes": notes,
    }


def _safe_metric(y: np.ndarray, score: np.ndarray, metric: Any) -> float:
    if len(set(y.tolist())) < 2:
        return math.nan
    return float(metric(y, score))


def _write_note(note_path: str | Path, detection_rows: list[dict[str, Any]], mitigation_rows: list[dict[str, Any]]) -> None:
    target = Path(note_path)
    ensure_dir(target.parent)
    det = pd.DataFrame(detection_rows)
    mit = pd.DataFrame(mitigation_rows)
    det_available = det[det["availability"].astype(str).str.startswith("available")].copy()
    det_available["auroc_num"] = pd.to_numeric(det_available["auroc"], errors="coerce")
    mit_available = mit[mit["availability"].astype(str).str.startswith("available")].copy()
    mit_available["rescue_num"] = pd.to_numeric(mit_available["fp_reduction_or_rescue_rate"], errors="coerce")
    lines = [
        "# Baseline Positioning",
        "",
        "## Detection",
        "",
        "Available detection baselines include raw image/blind hidden-state probes, paired difference probes, SVD-coordinate probes, Stage L evidence-specific subspaces, and a small Stage M first-token margin/entropy subset.",
        "",
        "Top available AUROC rows:",
        "",
    ]
    for _, row in det_available.sort_values("auroc_num", ascending=False).head(8).iterrows():
        lines.append(f"- `{row['baseline']}` ({row['source']}): AUROC `{_fmt(row['auroc'])}`, notes: {row['notes']}")
    lines.extend(
        [
            "",
            "Interpretation: the paired-difference family is competitive, but it should not be framed only as a leaderboard win. Its value is mechanistic: it explains where hallucination-related signal appears and why top-variance directions are misleading.",
            "",
            "## Mitigation / Rescue",
            "",
            "Available mitigation comparisons are Stage M first-token rescue rows. VCD/ICD and evidence-specific steering were not run in the current artifact set.",
            "",
            "Top available rescue rows:",
            "",
        ]
    )
    for _, row in mit_available.sort_values("rescue_num", ascending=False).head(8).iterrows():
        preservation_note = ""
        if pd.isna(row["tn_preservation"]) or pd.isna(row["tp_preservation"]):
            preservation_note = " (TN/TP preservation undefined because the gate selected no TN/TP controls.)"
        lines.append(
            f"- `{row['baseline']}` / `{row['intervention']}`: rescue `{_fmt(row['fp_reduction_or_rescue_rate'])}`, "
            f"TN preservation `{_fmt(row['tn_preservation'])}`, TP preservation `{_fmt(row['tp_preservation'])}`.{preservation_note}"
        )
    lines.extend(
        [
            "",
            "Interpretation: current rescue is boundary-local and weak. Random/global TN-like controls remain competitive, so this is not yet a strong mitigation method.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


if __name__ == "__main__":
    main()
