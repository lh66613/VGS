#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage R semantic fingerprint summary.")
    parser.add_argument("--semantics-dir", default="outputs/semantics")
    parser.add_argument("--stage-l-dir", default="outputs/stage_l_evidence_subspace")
    parser.add_argument("--output-dir", default="outputs/stage_r_semantics")
    parser.add_argument("--note-path", default="notes/semantic_interpretation_conclusion.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {
        "semantics_dir": args.semantics_dir,
        "stage_l_dir": args.stage_l_dir,
        "output_dir": args.output_dir,
        "note_path": args.note_path,
    }
    if not args.dry_run:
        payload.update(
            build_semantic_fingerprints(
                semantics_dir=args.semantics_dir,
                stage_l_dir=args.stage_l_dir,
                output_dir=args.output_dir,
                note_path=args.note_path,
            )
        )
    summary_path = write_json(Path(args.output_dir) / "build_stage_r_semantic_fingerprints_summary.json", payload)
    append_experiment_log(args.log_path, "build_stage_r_semantic_fingerprints", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


def build_semantic_fingerprints(
    semantics_dir: str | Path,
    stage_l_dir: str | Path,
    output_dir: str | Path,
    note_path: str | Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    panels_dir = Path(output_dir) / "semantic_sample_panels"
    ensure_dir(panels_dir)
    object_summary = pd.read_csv(Path(semantics_dir) / "semantic_object_summary.csv")
    cluster_summary = pd.read_csv(Path(semantics_dir) / "semantic_cluster_summary.csv")
    contrasts = pd.read_csv(Path(semantics_dir) / "semantic_outcome_contrasts.csv")
    extremes = pd.read_csv(Path(semantics_dir) / "semantic_sample_extremes.csv")

    rows = []
    for _, row in object_summary.iterrows():
        object_name = str(row["object"])
        object_contrasts = contrasts[contrasts["object"] == object_name]
        fp_tn = object_contrasts[object_contrasts["contrast"] == "FP_vs_TN"]
        fn_tp = object_contrasts[object_contrasts["contrast"] == "FN_vs_TP"]
        categories = _dominant_categories(cluster_summary[cluster_summary["object"] == object_name])
        reading = _semantic_reading(row, categories)
        rows.append(
            {
                "object": object_name,
                "family": row.get("family", ""),
                "layer": row.get("layer", ""),
                "projection_status": "projected",
                "top_positive_or_energy_tokens": row.get("top_positive", "") or row.get("top_energy", ""),
                "top_negative_tokens": row.get("top_negative", ""),
                "dominant_semantic_categories": categories,
                "fp_vs_tn_auc": _first(fp_tn, "auc"),
                "fp_vs_tn_cohen_d": _first(fp_tn, "cohen_d"),
                "fn_vs_tp_auc": _first(fn_tp, "auc"),
                "fn_vs_tp_cohen_d": _first(fn_tp, "cohen_d"),
                "sample_extreme_count": int(len(extremes[extremes["object"] == object_name])),
                "interpretation": reading,
                "claim_status": _claim_status(row.get("family", ""), _first(fp_tn, "auc"), categories),
            }
        )
        _write_object_panel(panels_dir, object_name, row, object_contrasts, extremes[extremes["object"] == object_name])
    rows.extend(_stage_l_rows(stage_l_dir))

    fingerprint_path = write_csv(
        Path(output_dir) / "semantic_fingerprint_summary.csv",
        rows,
        _fieldnames(rows),
    )
    _write_conclusion(note_path, rows)
    return {
        "fingerprint_path": str(fingerprint_path),
        "sample_panels_dir": str(panels_dir),
        "note_path": str(note_path),
        "num_rows": len(rows),
        "num_projected_objects": int(sum(row["projection_status"] == "projected" for row in rows)),
        "num_stage_l_rows": int(sum(row["projection_status"] == "not_projected_stage_l_summary" for row in rows)),
    }


def _stage_l_rows(stage_l_dir: str | Path) -> list[dict[str, Any]]:
    probe_path = Path(stage_l_dir) / "evidence_subspace_probe.csv"
    gap_path = Path(stage_l_dir) / "evidence_subspace_condition_gap.csv"
    if not probe_path.exists():
        return []
    probe = pd.read_csv(probe_path)
    gaps = pd.read_csv(gap_path) if gap_path.exists() else pd.DataFrame()
    result = []
    for method in ["pls_fp_tn", "fisher_fp_tn", "contrastive_pca", "matched_vs_adversarial_logistic"]:
        group = probe[probe["method"] == method]
        if group.empty:
            continue
        best = group.sort_values("auroc", ascending=False).iloc[0]
        gap_text = ""
        if not gaps.empty:
            gap_group = gaps[(gaps["method"] == method) & (gaps["row_type"] == "condition_delta")]
            if not gap_group.empty:
                best_gap = gap_group.reindex(gap_group["mean"].abs().sort_values(ascending=False).index).iloc[0]
                gap_text = f"{best_gap['comparison']} mean={best_gap['mean']:.3f} at L{int(best_gap['layer'])} K={int(best_gap['k'])}"
        result.append(
            {
                "object": f"stage_l_{method}",
                "family": "evidence_specific_subspace",
                "layer": int(best["layer"]),
                "projection_status": "not_projected_stage_l_summary",
                "top_positive_or_energy_tokens": "",
                "top_negative_tokens": "",
                "dominant_semantic_categories": "",
                "fp_vs_tn_auc": float(best["auroc"]),
                "fp_vs_tn_cohen_d": "",
                "fn_vs_tp_auc": "",
                "fn_vs_tp_cohen_d": "",
                "sample_extreme_count": "",
                "interpretation": f"Stage L evidence-specific subspace; best FP/TN AUROC={best['auroc']:.3f} at L{int(best['layer'])} K={int(best['k'])}. {gap_text}",
                "claim_status": "use as evidence-specific quantitative result; no vocabulary fingerprint yet",
            }
        )
    return result


def _semantic_reading(row: pd.Series, categories: str) -> str:
    family = str(row.get("family", ""))
    object_name = str(row.get("object", ""))
    pos = str(row.get("top_positive", "") or row.get("top_energy", ""))
    neg = str(row.get("top_negative", ""))
    if object_name == "L24_tail_257_1024":
        return "Object-heavy tail slice; useful as a concrete visual-evidence fingerprint, but weak as a standalone FP/TN separator."
    if family == "local_tn_rescue":
        return "Late local TN/rescue direction with relational/contextual and yes-no arbitration vocabulary; not a simple object detector."
    if family == "top_svd_backbone":
        if any(token in (pos + " " + neg) for token in ["sky", "tree", "window", "door", "cloud", "person", "holding"]):
            return "Broad visual-scene/action backbone axis with interpretable but mixed vocabulary."
        return "Top-SVD backbone axis with weak/noisy vocabulary fingerprint."
    return f"Semantic categories: {categories}"


def _claim_status(family: str, fp_auc: Any, categories: str) -> str:
    auc = _float_or_none(fp_auc)
    if auc is not None and (auc < 0.43 or auc > 0.57):
        classifier_note = "has a small sample-level FP/TN contrast"
    else:
        classifier_note = "not a strong standalone FP/TN classifier"
    if family == "tail_slice":
        return f"supports object-evidence interpretation; {classifier_note}"
    if family == "local_tn_rescue":
        return f"supports late arbitration interpretation; {classifier_note}"
    if family == "top_svd_backbone":
        return f"supports broad visual-semantic backbone interpretation; {classifier_note}"
    return classifier_note


def _dominant_categories(cluster_rows: pd.DataFrame) -> str:
    if cluster_rows.empty:
        return ""
    grouped = cluster_rows.groupby("semantic_category", as_index=False)["count"].sum()
    grouped = grouped[grouped["semantic_category"] != "other"].sort_values("count", ascending=False)
    if grouped.empty:
        return "other"
    return "; ".join(
        f"{row.semantic_category}:{int(row.count)}" for row in grouped.head(4).itertuples(index=False)
    )


def _write_object_panel(
    panels_dir: Path,
    object_name: str,
    summary_row: pd.Series,
    contrasts: pd.DataFrame,
    extremes: pd.DataFrame,
) -> None:
    safe_name = object_name.replace("/", "_")
    path = panels_dir / f"{safe_name}.md"
    lines = [
        f"# {object_name}",
        "",
        f"- Family: `{summary_row.get('family', '')}`",
        f"- Layer: `{summary_row.get('layer', '')}`",
        f"- Positive / energy tokens: {summary_row.get('top_positive', '') or summary_row.get('top_energy', '')}",
        f"- Negative tokens: {summary_row.get('top_negative', '')}",
        "",
        "## Outcome Contrasts",
        "",
    ]
    if contrasts.empty:
        lines.append("No contrast rows available.")
    else:
        keep = contrasts[contrasts["contrast"].isin(["FP_vs_TN", "FN_vs_TP", "TN_vs_TP"])]
        for _, row in keep.iterrows():
            lines.append(
                f"- `{row['contrast']}`: mean `{_fmt(row.get('mean'))}`, Cohen d `{_fmt(row.get('cohen_d'))}`, AUC `{_fmt(row.get('auc'))}`"
            )
    lines.extend(["", "## Representative Extremes", ""])
    if extremes.empty:
        lines.append("No sample extremes available.")
    else:
        for _, row in extremes.sort_values(["side", "rank"]).head(12).iterrows():
            lines.append(
                f"- `{row['side']}` rank {int(row['rank'])}: `{row['sample_id']}` / `{row['outcome']}` / {row['question']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_conclusion(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    df = pd.DataFrame(rows)
    projected = df[df["projection_status"] == "projected"]
    stage_l = df[df["projection_status"] != "projected"]
    lines = [
        "# Semantic Interpretation Conclusion",
        "",
        "## What The Semantic Fingerprints Support",
        "",
        "- Top-SVD backbone directions often show broad visual-scene, attribute, count, action, or spatial vocabulary fingerprints.",
        "- The L24 tail 257-1024 slice is object-heavy in vocabulary projection and aligns with the causal/tail-ablation story.",
        "- L32 local TN rescue directions look more like relational/contextual or yes-no arbitration directions than object-presence detectors.",
        "- Stage L evidence-specific subspaces are quantitatively useful, but they have not yet been vocabulary-projected in this artifact bundle.",
        "",
        "## What They Do Not Support",
        "",
        "- Do not claim that any single semantic direction is a strong hallucination detector.",
        "- Do not claim that the token projection proves a full mechanistic circuit.",
        "- Do not call the result a universal visual grounding subspace.",
        "",
        "## Summary Counts",
        "",
        f"- Projected geometry objects: {len(projected)}",
        f"- Stage L quantitative-only rows: {len(stage_l)}",
        "",
        "## Strongest Sample-Level FP/TN Contrasts",
        "",
    ]
    tmp = projected.copy()
    tmp["fp_vs_tn_auc_num"] = pd.to_numeric(tmp["fp_vs_tn_auc"], errors="coerce")
    tmp["distance_from_chance"] = (tmp["fp_vs_tn_auc_num"] - 0.5).abs()
    for _, row in tmp.sort_values("distance_from_chance", ascending=False).head(8).iterrows():
        lines.append(
            f"- `{row['object']}` ({row['family']}): FP/TN AUC `{_fmt(row['fp_vs_tn_auc'])}`, interpretation: {row['interpretation']}"
        )
    lines.extend(["", "## Recommended Paper Wording", ""])
    lines.append(
        "Use: **partially interpretable grounding-related correction geometry**. "
        "Avoid: **semantic hallucination coordinate**, **universal grounding subspace**, or claims that vocabulary projection alone establishes causality."
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df:
        return ""
    return df.iloc[0][column]


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
