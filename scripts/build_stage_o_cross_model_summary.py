#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact Stage O cross-model replication summary.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--stage-o-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--note-path", default="notes/cross_model_replication.md")
    args = parser.parse_args()

    root = Path(args.stage_o_dir or f"outputs/stage_o_cross_model/{args.model_alias}")
    output_dir = Path(args.output_dir or root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(_prediction_rows(args.model_alias, root))
    rows.extend(_spectrum_rows(args.model_alias, root))
    rows.extend(_probe_rows(args.model_alias, root))
    rows.extend(_curve_rows(args.model_alias, root))
    rows.extend(_margin_rows(args.model_alias, root))
    rows.extend(_condition_rows(args.model_alias, root))

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "minimal_replication_summary.csv"
    summary.to_csv(summary_path, index=False)

    note_path = Path(args.note_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(_render_note(args.model_alias, root, summary), encoding="utf-8")

    payload = {
        "model_alias": args.model_alias,
        "stage_o_dir": str(root),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "note_path": str(note_path),
        "num_rows": int(len(summary)),
    }
    (output_dir / "build_stage_o_cross_model_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _prediction_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "predictions" / "run_pope_eval_summary.json"
    if not path.exists():
        return [_missing_row(model_alias, "pope_prediction_summary", path)]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "model_alias": model_alias,
            "section": "pope_prediction_summary",
            "metric": "accuracy",
            "layer": "",
            "feature": "",
            "comparison": "",
            "value": data.get("accuracy"),
            "status": "available",
            "source": str(path),
        }
    ]
    counts = data.get("counts", {})
    for key, value in counts.items():
        rows.append(
            {
                "model_alias": model_alias,
                "section": "pope_prediction_summary",
                "metric": f"count_{key}",
                "layer": "",
                "feature": "",
                "comparison": "",
                "value": value,
                "status": "available",
                "source": str(path),
            }
        )
    return rows


def _spectrum_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "svd" / "effective_rank_summary.csv"
    if not path.exists():
        return [_missing_row(model_alias, "blind_image_svd_spectrum", path)]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        for metric in ["effective_rank", "explained_variance_k4", "explained_variance_k32"]:
            rows.append(
                {
                    "model_alias": model_alias,
                    "section": "blind_image_svd_spectrum",
                    "metric": metric,
                    "layer": int(row["layer"]),
                    "feature": "blind_minus_image_difference",
                    "comparison": "",
                    "value": row.get(metric),
                    "status": "available",
                    "source": str(path),
                }
            )
    return rows


def _probe_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "probes" / "probe_results.csv"
    if not path.exists():
        return [_missing_row(model_alias, "fp_tn_probe", path)]
    df = pd.read_csv(path)
    keep_features = {"raw_img", "raw_blind", "difference", "projected_difference", "random_difference", "pca_img"}
    df = df[df["feature_family"].isin(keep_features)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model_alias": model_alias,
                "section": "fp_tn_probe",
                "metric": "auroc",
                "layer": int(row["layer"]),
                "feature": row["feature_family"],
                "comparison": f"k={row['k']}",
                "value": row.get("auroc"),
                "status": "available",
                "source": str(path),
            }
        )
    return rows


def _curve_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "stage_c_deep" / "stage_c_topk_curve.csv"
    if not path.exists():
        return [_missing_row(model_alias, "explained_variance_vs_auroc_curve", path)]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model_alias": model_alias,
                "section": "explained_variance_vs_auroc_curve",
                "metric": "auroc",
                "layer": int(row["layer"]),
                "feature": "projected_difference",
                "comparison": f"k={row['k']}; explained_variance={row['explained_variance']}",
                "value": row.get("auroc"),
                "status": "available",
                "source": str(path),
            }
        )
    return rows


def _margin_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "margins" / "margin_baseline_metrics.csv"
    if not path.exists():
        return [_missing_row(model_alias, "first_token_margin_baseline", path)]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model_alias": model_alias,
                "section": "first_token_margin_baseline",
                "metric": "auroc",
                "layer": "",
                "feature": row["baseline"],
                "comparison": row["direction"],
                "value": row.get("auroc"),
                "status": "available",
                "source": str(path),
            }
        )
    return rows


def _condition_rows(model_alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "stage_b" / "stage_b_pairwise_condition_deltas.csv"
    if not path.exists():
        return [_missing_row(model_alias, "matched_mismatch_condition_geometry", path)]
    df = pd.read_csv(path)
    wanted = df[
        (df["comparison"].isin(["matched_minus_random_mismatch", "matched_minus_adversarial_mismatch"]))
        & (
            df["score"].str.contains("band_257_1024", na=False)
            | df["score"].eq("full_l2_sq")
        )
    ].copy()
    rows = []
    for _, row in wanted.iterrows():
        rows.append(
            {
                "model_alias": model_alias,
                "section": "matched_mismatch_condition_geometry",
                "metric": "mean_delta",
                "layer": int(row["layer"]),
                "feature": row["score"],
                "comparison": row["comparison"],
                "value": row.get("mean_delta"),
                "status": "available",
                "source": str(path),
            }
        )
    return rows


def _missing_row(model_alias: str, section: str, path: Path) -> dict[str, Any]:
    return {
        "model_alias": model_alias,
        "section": section,
        "metric": "missing_artifact",
        "layer": "",
        "feature": "",
        "comparison": "",
        "value": "",
        "status": "missing",
        "source": str(path),
    }


def _render_note(model_alias: str, root: Path, summary: pd.DataFrame) -> str:
    available = summary[summary["status"] == "available"].copy()
    missing = summary[summary["status"] == "missing"].copy()
    lines = [
        "# Cross-Model Replication",
        "",
        f"Model alias: `{model_alias}`",
        f"Artifact root: `{root}`",
        "",
    ]
    if missing.empty:
        lines.extend(["## Status", "", "Stage O minimal replication artifacts are available.", ""])
    else:
        lines.extend(["## Status", "", "Stage O is prepared but some artifacts are still missing:", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['section']}`: `{row['source']}`")
        lines.append("")

    if not available.empty:
        lines.extend(["## Best Available Probe Rows", ""])
        probes = available[(available["section"] == "fp_tn_probe") & (available["metric"] == "auroc")].copy()
        if not probes.empty:
            probes["value"] = pd.to_numeric(probes["value"], errors="coerce")
            probes = probes.sort_values("value", ascending=False).head(8)
            for _, row in probes.iterrows():
                lines.append(
                    f"- L{row['layer']} `{row['feature']}` {row['comparison']}: AUROC `{row['value']:.3f}`"
                )
        else:
            lines.append("- No probe rows available yet.")
        lines.append("")

        lines.extend(["## Margin Baseline", ""])
        margins = available[
            (available["section"] == "first_token_margin_baseline")
            & (available["metric"] == "auroc")
        ].copy()
        if not margins.empty:
            margins["value"] = pd.to_numeric(margins["value"], errors="coerce")
            for _, row in margins.sort_values("value", ascending=False).iterrows():
                lines.append(
                    f"- `{row['feature']}` `{row['comparison']}`: AUROC `{row['value']:.3f}`"
                )
        else:
            lines.append("- No margin-baseline rows available yet.")
        lines.append("")

        lines.extend(["## Variance/AUROC Curve", ""])
        curve = available[available["section"] == "explained_variance_vs_auroc_curve"].copy()
        if not curve.empty:
            lines.append(f"- Available rows: `{len(curve)}` from `stage_c_deep/stage_c_topk_curve.csv`.")
            lines.append("- Plot: `plots/stage_c_topk_auroc_explained_variance.png`.")
        else:
            lines.append("- No explained-variance/AUROC curve rows available yet.")
        lines.append("")

        lines.extend(["## Condition Geometry Snapshot", ""])
        condition = available[available["section"] == "matched_mismatch_condition_geometry"].copy()
        if not condition.empty:
            condition["value"] = pd.to_numeric(condition["value"], errors="coerce")
            for _, row in condition.head(12).iterrows():
                lines.append(
                    f"- L{row['layer']} `{row['feature']}` `{row['comparison']}` mean delta `{row['value']:.3f}`"
                )
        else:
            lines.append("- No condition-geometry rows available yet.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation Template",
            "",
            "- Strong replication if variance/discrimination mismatch, mid-layer residual/tail strength, and matched-vs-mismatch gaps all reappear.",
            "- Acceptable replication if only two of the three qualitative patterns reappear.",
            "- If the pattern fails, report Stage O as a limitation rather than expanding the paper's generality claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
