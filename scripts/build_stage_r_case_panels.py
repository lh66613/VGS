#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage R human-readable case panels.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--output-dir", default="outputs/case_studies")
    parser.add_argument("--notes-path", default="notes/case_studies.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {
        "predictions": args.predictions,
        "output_dir": args.output_dir,
        "notes_path": args.notes_path,
        "per_category": args.per_category,
    }
    if not args.dry_run:
        payload.update(build_case_panels(args.predictions, args.output_dir, args.notes_path, args.per_category))
    summary_path = write_json(Path(args.output_dir) / "build_stage_r_case_panels_summary.json", payload)
    append_experiment_log(args.log_path, "build_stage_r_case_panels", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


def build_case_panels(
    predictions_path: str | Path,
    output_dir: str | Path,
    notes_path: str | Path,
    per_category: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    predictions = [json.loads(line) for line in Path(predictions_path).open("r", encoding="utf-8")]
    pred_by_id = {str(row["sample_id"]): row for row in predictions}
    geometry = _load_geometry_scores()
    cases: list[dict[str, Any]] = []
    cases.extend(_select_successful_tn(pred_by_id, geometry, per_category))
    cases.extend(_select_fp_weak_correction(pred_by_id, geometry, per_category))
    cases.extend(_select_rescue_cases(pred_by_id, geometry, per_category))
    cases.extend(_select_unrescued_high_risk(pred_by_id, geometry, per_category))
    cases.extend(_select_adversarial_mismatch(pred_by_id, geometry, per_category))
    cases.extend(_select_semantic_extremes(pred_by_id, geometry, per_category))
    cases = _dedupe_cases(cases)
    for idx, row in enumerate(cases, start=1):
        row["case_id"] = f"case_{idx:02d}"
    metadata_path = write_csv(Path(output_dir) / "case_panel_metadata.csv", cases, _fieldnames(cases))
    _write_case_notes(notes_path, cases)
    return {
        "metadata_path": str(metadata_path),
        "notes_path": str(notes_path),
        "num_cases": len(cases),
        "categories": dict(pd.Series([row["case_category"] for row in cases]).value_counts().sort_index()),
    }


def _load_geometry_scores() -> pd.DataFrame:
    path = Path("outputs/stage_b/stage_b_sample_scores.csv")
    df = pd.read_csv(path)
    keep = df[
        (df["layer"].isin([24, 32]))
        & (df["condition"].isin(["matched", "random_mismatch", "adversarial_mismatch"]))
        & (
            ((df["view"] == "full") & (df["score"] == "full_l2_sq"))
            | ((df["view"] == "top_backbone") & (df["score"] == "top_1_4_l2_sq"))
            | ((df["view"] == "residual_tail") & (df["score"] == "band_257_1024_l2_sq"))
        )
    ].copy()
    keep["score_key"] = (
        "L"
        + keep["layer"].astype(str)
        + "_"
        + keep["condition"].astype(str)
        + "_"
        + keep["score"].astype(str)
    )
    pivot = keep.pivot_table(index="sample_id", columns="score_key", values="value", aggfunc="first").reset_index()
    return pivot


def _select_successful_tn(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    key = "L24_matched_band_257_1024_l2_sq"
    rows = geometry[["sample_id", key]].dropna().sort_values(key, ascending=False)
    result = []
    for _, row in rows.iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred and pred.get("outcome") == "TN":
            extra = row.to_dict()
            extra.update(_geometry_dict(geometry, row["sample_id"]))
            result.append(_case_row("successful_tn_strong_tail", pred, extra, "Correct No with a strong matched tail-correction score."))
        if len(result) >= n:
            break
    return result


def _select_fp_weak_correction(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    key = "L24_matched_band_257_1024_l2_sq"
    rows = geometry[["sample_id", key]].dropna().sort_values(key, ascending=True)
    result = []
    for _, row in rows.iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred and pred.get("outcome") == "FP":
            extra = row.to_dict()
            extra.update(_geometry_dict(geometry, row["sample_id"]))
            result.append(_case_row("fp_weak_matched_correction", pred, extra, "False Positive with weak matched tail-correction score."))
        if len(result) >= n:
            break
    return result


def _select_rescue_cases(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    path = Path("outputs/stage_m_local_rescue/rescue_failure_taxonomy.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rescued = df[df["rescued"] == True].sort_values("max_no_minus_yes_gain", ascending=False)
    result = []
    for _, row in rescued.head(n).iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred is None:
            continue
        extra = row.to_dict()
        extra.update(_geometry_dict(geometry, row["sample_id"]))
        result.append(_case_row("fp_rescued_by_local_steering", pred, extra, "Borderline False Positive rescued by a Stage M steering setting."))
    return result


def _select_unrescued_high_risk(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    path = Path("outputs/stage_m_local_rescue/rescue_failure_taxonomy.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    keep = df[df["rescued"] == False].sort_values(["fp_risk_score", "baseline_margin_yes_minus_no"], ascending=False)
    result = []
    for _, row in keep.head(n).iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred is None:
            continue
        extra = row.to_dict()
        extra.update(_geometry_dict(geometry, row["sample_id"]))
        result.append(_case_row("fp_not_rescued_high_score", pred, extra, "High-risk False Positive remains unrescued, useful for failure analysis."))
    return result


def _select_adversarial_mismatch(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    key_matched = "L24_matched_band_257_1024_l2_sq"
    key_adv = "L24_adversarial_mismatch_band_257_1024_l2_sq"
    available = geometry.dropna(subset=[key_matched, key_adv]).copy()
    available["matched_minus_adv_tail"] = available[key_matched] - available[key_adv]
    rows = available.reindex(available["matched_minus_adv_tail"].abs().sort_values(ascending=False).index)
    result = []
    for _, row in rows.iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred and pred.get("subset") == "adversarial":
            result.append(_case_row("adversarial_mismatch_example", pred, row.to_dict(), "Adversarial subset case with a large matched-vs-adversarial tail-score difference."))
        if len(result) >= n:
            break
    return result


def _select_semantic_extremes(pred_by_id: dict[str, dict[str, Any]], geometry: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    path = Path("outputs/semantics/semantic_sample_extremes.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    priority = df[
        (df["rank"] <= 3)
        & (df["object"].isin(["L24_tail_257_1024", "L32_local_knn_tn_correction", "L32_svd_5", "L24_svd_5"]))
    ].copy()
    priority = priority.sort_values(["object", "side", "rank"])
    result = []
    seen_objects: set[str] = set()
    for _, row in priority.iterrows():
        pred = pred_by_id.get(str(row["sample_id"]))
        if pred is None:
            continue
        object_side = f"{row['object']}:{row['side']}"
        if object_side in seen_objects:
            continue
        seen_objects.add(object_side)
        extra = row.to_dict()
        extra.update(_geometry_dict(geometry, row["sample_id"]))
        result.append(_case_row("semantic_direction_extreme", pred, extra, "Extreme sample for an interpreted semantic geometry object."))
        if len(result) >= n:
            break
    return result


def _case_row(category: str, pred: dict[str, Any], extra: dict[str, Any], note: str) -> dict[str, Any]:
    sample_id = str(pred["sample_id"])
    row = {
        "case_id": "",
        "case_category": category,
        "sample_id": sample_id,
        "subset": pred.get("subset", ""),
        "image": pred.get("image", ""),
        "image_path": pred.get("image_path", ""),
        "question": pred.get("question", ""),
        "target_object": extra.get("target_object", _target_object(pred.get("question", ""))),
        "ground_truth": pred.get("label", ""),
        "model_answer": pred.get("parsed_prediction", ""),
        "outcome": pred.get("outcome", ""),
        "raw_generation": pred.get("raw_generation", ""),
        "short_human_explanation": note,
        "intervention_result": _intervention_summary(extra),
        "semantic_object": extra.get("object", ""),
        "semantic_side": extra.get("side", ""),
        "semantic_rank": extra.get("rank", ""),
        "baseline_margin_yes_minus_no": extra.get("baseline_margin_yes_minus_no", ""),
        "best_no_minus_yes_gain": extra.get("max_no_minus_yes_gain", ""),
        "best_gate": extra.get("best_gate", ""),
        "best_intervention": extra.get("best_intervention", ""),
        "best_alpha": extra.get("best_alpha", ""),
    }
    for key in [
        "L24_matched_band_257_1024_l2_sq",
        "L24_matched_top_1_4_l2_sq",
        "L24_matched_full_l2_sq",
        "L24_random_mismatch_band_257_1024_l2_sq",
        "L24_adversarial_mismatch_band_257_1024_l2_sq",
        "matched_minus_adv_tail",
        "fp_risk_score",
        "tail_norm",
        "score",
        "z_score",
    ]:
        row[key] = extra.get(key, "")
    return row


def _intervention_summary(extra: dict[str, Any]) -> str:
    if "best_intervention" not in extra or pd.isna(extra.get("best_intervention")):
        return ""
    return (
        f"{extra.get('best_gate', '')}/"
        f"{extra.get('best_intervention', '')}/"
        f"alpha={extra.get('best_alpha', '')}/"
        f"after={extra.get('best_prediction', '')}:{extra.get('best_outcome_after', '')}"
    )


def _geometry_dict(geometry: pd.DataFrame, sample_id: str) -> dict[str, Any]:
    match = geometry[geometry["sample_id"] == sample_id]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _dedupe_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for case in cases:
        key = (case["case_category"], case["sample_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(case)
    return result


def _write_case_notes(path: str | Path, cases: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    lines = ["# Case Studies", "", "Human-readable case panels generated from Stage R.", ""]
    for category, group in pd.DataFrame(cases).groupby("case_category", sort=False):
        lines.extend([f"## {category}", ""])
        for row in group.to_dict(orient="records"):
            lines.extend(
                [
                    f"### {row['case_id']} — {row['sample_id']}",
                    "",
                    f"- Image: `{row['image_path']}`",
                    f"- Question: {row['question']}",
                    f"- Ground truth / model answer / outcome: `{row['ground_truth']}` / `{row['model_answer']}` / `{row['outcome']}`",
                    f"- Target object: `{row['target_object']}`",
                    f"- Geometry: L24 tail `{_fmt(row.get('L24_matched_band_257_1024_l2_sq'))}`, L24 top-4 `{_fmt(row.get('L24_matched_top_1_4_l2_sq'))}`, FP-risk `{_fmt(row.get('fp_risk_score'))}`",
                    f"- Intervention: {row['intervention_result'] or 'none recorded'}",
                    f"- Note: {row['short_human_explanation']}",
                    "",
                ]
            )
    target.write_text("\n".join(lines), encoding="utf-8")


def _target_object(question: str) -> str:
    match = re.search(r"Is there an? (.+?) in the image\\?", question)
    return match.group(1) if match else ""


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
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
