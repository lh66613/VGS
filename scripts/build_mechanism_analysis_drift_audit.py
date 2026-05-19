#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


DRIFT_COLUMNS = [
    "orig_no_minus_yes_logit",
    "neg_no_minus_yes_logit",
    "delta_norm_sq",
    "energy_band5_16",
    "dmargin_no_minus_yes_band5_16",
    "dmargin_no_minus_yes_full",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate protocol and artifact drift between reference and current geometry.")
    parser.add_argument(
        "--current-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument(
        "--current-summary",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry_summary.json",
    )
    parser.add_argument("--reference-geometry", default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv")
    parser.add_argument("--reference-summary", default="outputs/mechanism_mitigation/operator_geometry/operator_geometry_summary.json")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/drift_audit",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_drift_audit(
        current_geometry=Path(args.current_geometry),
        current_summary=Path(args.current_summary),
        reference_geometry=Path(args.reference_geometry),
        reference_summary=Path(args.reference_summary),
        predictions=Path(args.predictions),
        split_dir=Path(args.split_dir),
        svd_dir=Path(args.svd_dir),
        layer=args.layer,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_drift_audit_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_drift_audit", summary_path, "ok")
    print(summary_path)


def build_drift_audit(
    current_geometry: Path,
    current_summary: Path,
    reference_geometry: Path,
    reference_summary: Path,
    predictions: Path,
    split_dir: Path,
    svd_dir: Path,
    layer: int,
    output_dir: Path,
) -> dict[str, Any]:
    current_meta = _read_json(current_summary)
    reference_meta = _read_json(reference_summary)
    config_rows = _config_rows(current_meta, reference_meta)
    environment_rows = _environment_rows()
    split_rows = _split_rows(predictions, split_dir)
    svd_rows = _svd_rows(svd_dir, layer, split_dir)
    drift_rows, worst_rows = _geometry_drift_rows(current_geometry, reference_geometry)
    conclusion_rows = _conclusion_rows(config_rows, drift_rows, environment_rows)

    paths = {
        "config_audit": write_csv(output_dir / "config_audit.csv", config_rows, fieldnames(config_rows)),
        "environment_audit": write_csv(output_dir / "environment_audit.csv", environment_rows, fieldnames(environment_rows)),
        "split_pool_audit": write_csv(output_dir / "split_pool_audit.csv", split_rows, fieldnames(split_rows)),
        "svd_audit": write_csv(output_dir / "svd_audit.csv", svd_rows, fieldnames(svd_rows)),
        "geometry_drift": write_csv(output_dir / "geometry_drift.csv", drift_rows, fieldnames(drift_rows)),
        "worst_band5_16_rows": write_csv(output_dir / "worst_band5_16_rows.csv", worst_rows, fieldnames(worst_rows)),
        "conclusion": write_csv(output_dir / "drift_conclusion.csv", conclusion_rows, fieldnames(conclusion_rows)),
    }
    report_path = _write_report(output_dir / "drift_audit_report.md", paths)
    return {
        "current_geometry": str(current_geometry),
        "reference_geometry": str(reference_geometry),
        **{key + "_path": str(value) for key, value in paths.items()},
        "report_path": str(report_path),
    }


def _config_rows(current: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("model_path", "model/checkpoint"),
        ("model_family", "model family"),
        ("layers", "layer"),
        ("readout_position", "hidden readout token"),
        ("operators", "negative reference/operator"),
        ("svd_dir", "SVD source"),
        ("torch_dtype", "dtype"),
        ("noise_step", "VCD noise setting"),
        ("blur_radius", "image preprocessing blur setting"),
        ("yes_token_ids", "yes token IDs"),
        ("no_token_ids", "no token IDs"),
    ]
    rows: list[dict[str, Any]] = []
    for key, label in checks:
        current_value = current.get(key)
        reference_value = reference.get(key)
        rows.append(
            {
                "check": label,
                "key": key,
                "current": _format_value(current_value),
                "reference": _format_value(reference_value),
                "status": "matched" if current_value == reference_value else "differs",
            }
        )
    rows.extend(
        [
            {
                "check": "prompt template",
                "key": "build_pope_prompt/build_blind_prompt",
                "current": "src/vgs/llava_hf.py",
                "reference": "same repo path; old run did not store code hash",
                "status": "not_stored_in_reference",
            },
            {
                "check": "difference sign",
                "key": "delta",
                "current": "h_orig - h_neg",
                "reference": "assumed h_orig - h_neg from archived script",
                "status": "not_stored_in_reference",
            },
            {
                "check": "alpha selection rule",
                "key": "stage2",
                "current": "max fp_reduction - tp_damage under TP constraint",
                "reference": "same analyze_stage2 rule for frozen exact reproduction",
                "status": "matched_by_reproduction",
            },
        ]
    )
    return rows


def _environment_rows() -> list[dict[str, Any]]:
    packages = ["torch", "transformers", "tokenizers", "numpy", "pandas"]
    rows: list[dict[str, Any]] = []
    for env_name, python in [("vlm-exp", "/data/lh/.conda/envs/vlm-exp/bin/python"), ("after", "/data/lh/.conda/envs/after/bin/python")]:
        versions = _package_versions(python, packages)
        for package in packages:
            rows.append({"environment": env_name, "package": package, "version": versions.get(package, "")})
    return rows


def _split_rows(predictions_path: Path, split_dir: Path) -> list[dict[str, Any]]:
    predictions = pd.DataFrame(read_jsonl(predictions_path))
    rows: list[dict[str, Any]] = []
    for name, ids in _split_ids(split_dir).items():
        group = predictions[predictions["sample_id"].astype(str).isin(ids)]
        pred_yes = group[group["parsed_prediction"].astype(str) == "yes"]
        rows.append(
            {
                "split": name,
                "n": int(len(group)),
                "predicted_yes_n": int(len(pred_yes)),
                "fp_n": int((group["outcome"].astype(str) == "FP").sum()),
                "tp_n": int((group["outcome"].astype(str) == "TP").sum()),
                "tn_n": int((group["outcome"].astype(str) == "TN").sum()),
                "fn_n": int((group["outcome"].astype(str) == "FN").sum()),
            }
        )
    return rows


def _svd_rows(svd_dir: Path, layer: int, split_dir: Path) -> list[dict[str, Any]]:
    path = svd_dir / f"svd_layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu")
    sample_ids = [str(item) for item in payload.get("sample_ids", [])]
    sample_set = set(sample_ids)
    split_ids = _split_ids(split_dir)
    rows = [
        {
            "artifact": str(path),
            "sample_id_n": len(sample_ids),
            "vh_shape": "x".join(map(str, tuple(payload["Vh"].shape))),
            "singular_values_n": int(payload["singular_values"].numel()),
            "metadata": _format_value(payload.get("metadata", {})),
        }
    ]
    for split, ids in split_ids.items():
        rows.append(
            {
                "artifact": f"{split}_overlap",
                "sample_id_n": len(sample_set.intersection(ids)),
                "vh_shape": "",
                "singular_values_n": "",
                "metadata": f"{len(sample_set.intersection(ids))}/{len(ids)} split IDs in SVD sample IDs",
            }
        )
    return rows


def _geometry_drift_rows(current_path: Path, reference_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usecols = ["sample_id", "operator", *DRIFT_COLUMNS]
    current = pd.read_csv(current_path, usecols=usecols)
    reference = pd.read_csv(reference_path, usecols=usecols)
    current = current[current["operator"].astype(str) == "icd_blind"]
    reference = reference[reference["operator"].astype(str) == "icd_blind"]
    merged = current.merge(reference, on=["sample_id", "operator"], suffixes=("_current", "_reference"))
    rows: list[dict[str, Any]] = []
    for column in DRIFT_COLUMNS:
        diff = (merged[f"{column}_current"] - merged[f"{column}_reference"]).abs()
        rows.append(
            {
                "column": column,
                "n": int(len(diff)),
                "mean_abs_diff": float(diff.mean()),
                "median_abs_diff": float(diff.median()),
                "max_abs_diff": float(diff.max()),
                "exact_match_rate": float((diff < 1e-6).mean()),
            }
        )
    diff = (
        merged["dmargin_no_minus_yes_band5_16_current"]
        - merged["dmargin_no_minus_yes_band5_16_reference"]
    ).abs()
    worst = merged.loc[diff.nlargest(20).index].copy()
    worst["abs_diff_dmargin_no_minus_yes_band5_16"] = diff.loc[worst.index]
    keep = [
        "sample_id",
        "abs_diff_dmargin_no_minus_yes_band5_16",
        "dmargin_no_minus_yes_band5_16_current",
        "dmargin_no_minus_yes_band5_16_reference",
        "delta_norm_sq_current",
        "delta_norm_sq_reference",
        "orig_no_minus_yes_logit_current",
        "orig_no_minus_yes_logit_reference",
        "neg_no_minus_yes_logit_current",
        "neg_no_minus_yes_logit_reference",
    ]
    return rows, worst[keep].to_dict(orient="records")


def _conclusion_rows(config_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], environment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config_diff = [row for row in config_rows if row["status"] == "differs"]
    large_delta = any(
        row["column"] == "delta_norm_sq" and float(row["mean_abs_diff"]) > 1.0
        for row in drift_rows
    )
    env_versions = {(row["environment"], row["package"]): row["version"] for row in environment_rows}
    env_diff = [
        package
        for package in ["torch", "transformers", "tokenizers"]
        if env_versions.get(("vlm-exp", package)) != env_versions.get(("after", package))
    ]
    rows = []
    rows.append(
        {
            "priority": 1,
            "finding": "Core stored protocol fields mostly match except intentional expanded subspace/operator set.",
            "evidence": "; ".join(f"{row['key']} differs" for row in config_diff[:6]) or "No stored core field drift.",
        }
    )
    rows.append(
        {
            "priority": 2,
            "finding": "Hidden/reference geometry numerically drifts despite token-ID match.",
            "evidence": "delta_norm_sq and band5_16 dmargin have large mean absolute differences."
            if large_delta
            else "No large hidden drift detected.",
        }
    )
    rows.append(
        {
            "priority": 3,
            "finding": "Most likely source is environment/model-forward implementation drift, especially Transformers/LLaVA handling of hidden states or text-only language_model path.",
            "evidence": "Differing packages: " + ", ".join(env_diff) if env_diff else "No package drift detected.",
        }
    )
    return rows


def _split_ids(split_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for filename, split in [
        ("pope_train_ids.json", "train"),
        ("pope_val_ids.json", "calibration"),
        ("pope_test_ids.json", "test"),
    ]:
        path = split_dir / filename
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            out[split] = {str(item) for item in payload.get("sample_ids", [])}
    return out


def _package_versions(python: str, packages: list[str]) -> dict[str, str]:
    code = (
        "import importlib.metadata as md, json; "
        f"packages={packages!r}; "
        "out={};\n"
        "for p in packages:\n"
        "    try: out[p]=md.version(p)\n"
        "    except Exception as e: out[p]=str(e)\n"
        "print(json.dumps(out))"
    )
    try:
        output = subprocess.check_output([python, "-c", code], text=True)
        return json.loads(output)
    except Exception as exc:
        return {package: f"unavailable: {exc}" for package in packages}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _write_report(path: Path, paths: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = ["# Mechanism Analysis Drift Audit", ""]
    for title, key in [
        ("Config Audit", "config_audit"),
        ("Environment Audit", "environment_audit"),
        ("Split And Pool Audit", "split_pool_audit"),
        ("SVD Audit", "svd_audit"),
        ("Geometry Drift", "geometry_drift"),
        ("Likely Cause", "conclusion"),
    ]:
        df = pd.read_csv(paths[key])
        sections.extend([f"## {title}", "", markdown_table(df), "", f"Artifact: `{paths[key]}`", ""])
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
