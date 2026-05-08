#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage O cross-architecture result consistency.")
    parser.add_argument("--root", default="outputs/stage_o_cross_model")
    parser.add_argument("--output-dir", default="outputs/stage_o_cross_model/audit")
    parser.add_argument("--notes-path", default="notes/stage_o_cross_arch_audit.md")
    args = parser.parse_args()

    root = Path(args.root)
    aliases = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "audit")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for alias in aliases:
        model_root = root / alias
        run_summary = _read_json(model_root / "predictions" / "run_pope_eval_summary.json")
        hidden_summary = _read_json(model_root / "hidden_states" / "dump_hidden_states_summary.json")
        condition_summary = _read_json(model_root / "condition_hidden" / "dump_stage_b_condition_hidden_states_summary.json")
        margin_summary = _read_json(model_root / "margins" / "dump_pope_margins_summary.json")

        counts = run_summary.get("counts", {}) if run_summary else {}
        num_samples = int(run_summary.get("num_samples") or 0) if run_summary else 0
        fp = int(counts.get("FP", 0) or 0)
        tn = int(counts.get("TN", 0) or 0)
        tp = int(counts.get("TP", 0) or 0)
        fn = int(counts.get("FN", 0) or 0)
        unknown = int(counts.get("unknown", 0) or 0)
        no_count = fp + tn
        yes_count = tp + fn

        summary_rows.append(
            {
                "alias": alias,
                "model_path": run_summary.get("model_path") if run_summary else "",
                "model_family": run_summary.get("model_family") if run_summary else "",
                "accuracy": run_summary.get("accuracy") if run_summary else None,
                "num_samples": num_samples,
                "FP": fp,
                "TN": tn,
                "TP": tp,
                "FN": fn,
                "unknown": unknown,
                "no_label_error_rate": fp / no_count if no_count else None,
                "yes_label_error_rate": fn / yes_count if yes_count else None,
                "layers": " ".join(str(item) for item in hidden_summary.get("layers", [])) if hidden_summary else "",
                "readout_position": hidden_summary.get("readout_position") if hidden_summary else "",
                "hidden_num_samples": hidden_summary.get("num_samples") if hidden_summary else None,
                "condition_num_samples": condition_summary.get("num_samples") if condition_summary else None,
                "margin_num_rows": margin_summary.get("num_rows") if margin_summary else None,
            }
        )

        probe_rows.extend(_best_probe_rows(alias, model_root))
        margin_rows.extend(_margin_rows(alias, model_root))
        condition_rows.extend(_condition_rows(alias, model_root))
        spectrum_rows.extend(_spectrum_rows(alias, model_root))
        diagnostics_rows.extend(_diagnostics(alias, model_root, run_summary, hidden_summary, condition_summary))

    paths = {
        "model_summary": output_dir / "model_summary.csv",
        "probe_summary": output_dir / "probe_summary.csv",
        "margin_summary": output_dir / "margin_summary.csv",
        "condition_summary": output_dir / "condition_summary.csv",
        "spectrum_summary": output_dir / "spectrum_summary.csv",
        "diagnostics": output_dir / "diagnostics.csv",
    }
    pd.DataFrame(summary_rows).to_csv(paths["model_summary"], index=False)
    pd.DataFrame(probe_rows).to_csv(paths["probe_summary"], index=False)
    pd.DataFrame(margin_rows).to_csv(paths["margin_summary"], index=False)
    pd.DataFrame(condition_rows).to_csv(paths["condition_summary"], index=False)
    pd.DataFrame(spectrum_rows).to_csv(paths["spectrum_summary"], index=False)
    pd.DataFrame(diagnostics_rows).to_csv(paths["diagnostics"], index=False)

    note = _render_note(
        pd.DataFrame(summary_rows),
        pd.DataFrame(probe_rows),
        pd.DataFrame(margin_rows),
        pd.DataFrame(condition_rows),
        pd.DataFrame(spectrum_rows),
        pd.DataFrame(diagnostics_rows),
        paths,
    )
    note_path = Path(args.notes_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")

    payload = {name: str(path) for name, path in paths.items()}
    payload["notes_path"] = str(note_path)
    payload["num_aliases"] = len(aliases)
    (output_dir / "audit_stage_o_cross_arch_results_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _best_probe_rows(alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "probes" / "probe_results.csv"
    if not path.exists():
        return [{"alias": alias, "status": "missing", "source": str(path)}]
    df = pd.read_csv(path)
    rows = []
    for feature in ["raw_img", "raw_blind", "difference", "projected_difference"]:
        sub = df[df["feature_family"] == feature].copy()
        if sub.empty:
            continue
        sub["auroc"] = pd.to_numeric(sub["auroc"], errors="coerce")
        best = sub.sort_values("auroc", ascending=False).iloc[0]
        rows.append(
            {
                "alias": alias,
                "feature": feature,
                "best_layer": int(best["layer"]),
                "best_k": best["k"],
                "best_auroc": float(best["auroc"]),
                "num_positive": int(best["num_positive"]),
                "num_samples": int(best["num_samples"]),
                "source": str(path),
                "status": "available",
            }
        )
    return rows


def _margin_rows(alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "margins" / "margin_baseline_metrics.csv"
    if not path.exists():
        return [{"alias": alias, "status": "missing", "source": str(path)}]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "alias": alias,
                "baseline": row["baseline"],
                "direction": row["direction"],
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
                "num_positive": int(row["num_positive"]),
                "num_samples": int(row.get("num_samples", row.get("n", 0))),
                "source": str(path),
                "status": "available",
            }
        )
    return rows


def _condition_rows(alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "stage_b" / "stage_b_pairwise_condition_deltas.csv"
    if not path.exists():
        return [{"alias": alias, "status": "missing", "source": str(path)}]
    df = pd.read_csv(path)
    keep = df[
        df["comparison"].isin(["matched_minus_random_mismatch", "matched_minus_adversarial_mismatch"])
        & (df["score"].eq("full_l2_sq") | df["score"].str.contains("band_257_1024", na=False))
    ].copy()
    rows = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "alias": alias,
                "layer": int(row["layer"]),
                "view": row["view"],
                "score": row["score"],
                "comparison": row["comparison"],
                "n": int(row["n"]),
                "mean_delta": float(row["mean_delta"]),
                "median_delta": float(row["median_delta"]),
                "source": str(path),
                "status": "available",
            }
        )
    return rows


def _spectrum_rows(alias: str, root: Path) -> list[dict[str, Any]]:
    path = root / "svd" / "effective_rank_summary.csv"
    if not path.exists():
        return [{"alias": alias, "status": "missing", "source": str(path)}]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "alias": alias,
                "layer": int(row["layer"]),
                "num_samples": int(row["num_samples"]),
                "hidden_dim": int(row["hidden_dim"]),
                "effective_rank": float(row["effective_rank"]),
                "explained_variance_k4": float(row["explained_variance_k4"]),
                "explained_variance_k32": float(row["explained_variance_k32"]),
                "source": str(path),
                "status": "available",
            }
        )
    return rows


def _diagnostics(alias: str, root: Path, run_summary: dict[str, Any], hidden_summary: dict[str, Any], condition_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    counts = run_summary.get("counts", {}) if run_summary else {}
    num_samples = int(run_summary.get("num_samples") or 0) if run_summary else 0
    known = sum(int(counts.get(key, 0) or 0) for key in ["TP", "TN", "FP", "FN"])
    unknown = int(counts.get("unknown", 0) or 0)
    if num_samples and known + unknown != num_samples:
        rows.append(_diag(alias, "prediction_count_mismatch", f"counts sum {known + unknown} != num_samples {num_samples}", "high"))
    if num_samples and unknown / num_samples > 0.05:
        rows.append(_diag(alias, "high_unknown_rate", f"unknown rate {unknown / num_samples:.3f}", "high"))
    if hidden_summary and hidden_summary.get("num_samples") != num_samples:
        rows.append(_diag(alias, "hidden_prediction_count_mismatch", f"hidden {hidden_summary.get('num_samples')} vs predictions {num_samples}", "high"))
    if condition_summary and int(condition_summary.get("num_samples") or 0) <= 0:
        rows.append(_diag(alias, "empty_condition_hidden", "condition hidden summary has no samples", "high"))
    readout = hidden_summary.get("readout_position") if hidden_summary else None
    if readout != "last_prompt_token":
        rows.append(_diag(alias, "nonstandard_readout", f"readout_position={readout}", "medium"))
    return rows or [_diag(alias, "basic_metadata_checks", "no basic metadata mismatch detected", "info")]


def _diag(alias: str, check: str, message: str, severity: str) -> dict[str, Any]:
    return {"alias": alias, "check": check, "severity": severity, "message": message}


def _render_note(
    model_summary: pd.DataFrame,
    probes: pd.DataFrame,
    margins: pd.DataFrame,
    conditions: pd.DataFrame,
    spectra: pd.DataFrame,
    diagnostics: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = [
        "# Stage O Cross-Architecture Audit",
        "",
        "## Files",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Model Summary", ""])
    if not model_summary.empty:
        for _, row in model_summary.sort_values("alias").iterrows():
            lines.append(
                f"- `{row['alias']}`: family `{row.get('model_family', '')}`, "
                f"accuracy `{_fmt(row['accuracy'])}`, FP/TN/TP/FN/unk "
                f"`{int(row['FP'])}/{int(row['TN'])}/{int(row['TP'])}/{int(row['FN'])}/{int(row['unknown'])}`, "
                f"layers `{row['layers']}`, readout `{row['readout_position']}`"
            )
    lines.extend(["", "## Probe Snapshot", ""])
    if not probes.empty and "best_auroc" in probes:
        for alias, group in probes[probes["status"] == "available"].groupby("alias"):
            diff = _feature_row(group, "difference")
            raw_blind = _feature_row(group, "raw_blind")
            projected = _feature_row(group, "projected_difference")
            lines.append(
                f"- `{alias}`: difference `{_row_metric(diff)}`, raw_blind `{_row_metric(raw_blind)}`, projected `{_row_metric(projected)}`"
            )
    lines.extend(["", "## Margin Baseline Snapshot", ""])
    if not margins.empty and "auroc" in margins:
        for alias, group in margins[margins["status"] == "available"].groupby("alias"):
            best = group.sort_values("auroc", ascending=False).iloc[0]
            lines.append(f"- `{alias}`: best `{best['baseline']}` AUROC `{best['auroc']:.3f}` ({best['direction']})")
    lines.extend(["", "## Condition Geometry Snapshot", ""])
    if not conditions.empty and "mean_delta" in conditions:
        for alias, group in conditions[conditions["status"] == "available"].groupby("alias"):
            adv_tail = group[
                (group["score"].str.contains("band_257_1024", na=False))
                & (group["comparison"] == "matched_minus_adversarial_mismatch")
            ]
            values = ", ".join(f"L{int(row.layer)} {row.mean_delta:.2f}" for row in adv_tail.itertuples(index=False))
            lines.append(f"- `{alias}` adversarial tail deltas: {values or 'none'}")
    lines.extend(["", "## Diagnostics", ""])
    for _, row in diagnostics.iterrows():
        lines.append(f"- `{row['alias']}` `{row['severity']}` `{row['check']}`: {row['message']}")
    lines.extend(
        [
            "",
            "## Initial Reading",
            "",
            "- If the Qwen/InternVL rows have strong margin baselines but weak hidden probes, the model output is probably fine while the hidden readout is not architecture-equivalent.",
            "- If both margin baselines and hidden probes collapse, first inspect yes/no tokenization and prompt formatting.",
            "- If condition deltas flip sign only in full space but not in tail bands, treat the condition result as geometry-specific rather than a pipeline failure.",
            "",
        ]
    )
    return "\n".join(lines)


def _feature_row(group: pd.DataFrame, feature: str) -> pd.Series | None:
    sub = group[group["feature"] == feature]
    if sub.empty:
        return None
    return sub.iloc[0]


def _row_metric(row: pd.Series | None) -> str:
    if row is None:
        return "missing"
    return f"L{int(row['best_layer'])} k={row['best_k']} AUROC {float(row['best_auroc']):.3f}"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
