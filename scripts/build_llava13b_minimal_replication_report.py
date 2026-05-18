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


DETECTOR_FEATURES = {
    "margin-only": ["base_no_minus_yes_logit"],
    "margin+tail": ["base_no_minus_yes_logit", "dmargin_no_minus_yes_tail257_1024"],
    "margin+full": ["base_no_minus_yes_logit", "dmargin_no_minus_yes_full"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the LLaVA-1.5-13B minimal detector/mitigation replication report."
    )
    parser.add_argument("--operator-geometry", default="outputs/mechanism_mitigation/llava13b_minimal/operator_geometry/operator_geometry.csv")
    parser.add_argument("--stage2-dir", default="outputs/mechanism_mitigation/llava13b_minimal/stage2_subspace_icd")
    parser.add_argument("--predictions", default="outputs/stage_o_cross_model/llava_13b/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/stage_o_cross_model/llava_13b/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--always-alpha", type=float, default=1.0)
    parser.add_argument("--gate-score", choices=sorted(DETECTOR_FEATURES), default="margin+tail")
    parser.add_argument("--target-trigger-rate", type=float, default=0.3)
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/llava13b_minimal/report")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_report(
        operator_geometry_path=args.operator_geometry,
        stage2_dir=args.stage2_dir,
        predictions_path=args.predictions,
        margin_scores_path=args.margin_scores,
        split_dir=args.split_dir,
        layer=args.layer,
        always_alpha=args.always_alpha,
        gate_score=args.gate_score,
        target_trigger_rate=args.target_trigger_rate,
        output_dir=args.output_dir,
    )
    summary_path = write_json(Path(args.output_dir) / "build_llava13b_minimal_replication_report_summary.json", result)
    append_experiment_log(args.log_path, "build_llava13b_minimal_replication_report", summary_path, "ok")
    print(summary_path)


def build_report(
    operator_geometry_path: str | Path,
    stage2_dir: str | Path,
    predictions_path: str | Path,
    margin_scores_path: str | Path,
    split_dir: str | Path,
    layer: int,
    always_alpha: float,
    gate_score: str,
    target_trigger_rate: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    split_map = _load_split_map(split_dir)
    margins = _load_margins(margin_scores_path)
    geometry = _load_geometry(operator_geometry_path, predictions, split_map, margins, layer)

    detector_rows, detector_scores, gate_thresholds = _detector_rows(
        geometry,
        target_trigger_rate=target_trigger_rate,
    )
    detector_path = write_csv(output_root / "detector_comparison.csv", detector_rows, _fieldnames(detector_rows))

    mitigation_rows, method_predictions, trigger_ids = _mitigation_rows(
        stage2_dir=Path(stage2_dir),
        predictions=predictions,
        split_map=split_map,
        detector_scores=detector_scores,
        gate_thresholds=gate_thresholds,
        gate_score=gate_score,
        target_trigger_rate=target_trigger_rate,
        layer=layer,
        always_alpha=always_alpha,
    )
    mitigation_path = write_csv(output_root / "mitigation_comparison.csv", mitigation_rows, _fieldnames(mitigation_rows))

    yes_rows = _yes_rate_rows(predictions, method_predictions)
    yes_path = write_csv(output_root / "yes_rate_audit.csv", yes_rows, _fieldnames(yes_rows))

    pareto_rows = _pareto_rows(Path(stage2_dir) / "alpha_sweep.csv", layer)
    pareto_path = write_csv(output_root / "pareto_curve_points.csv", pareto_rows, _fieldnames(pareto_rows))
    pareto_fig = _plot_pareto(output_root / "pareto_curve.png", pareto_rows)

    success_rows = _success_rows(detector_rows, mitigation_rows)
    success_path = write_csv(output_root / "success_criteria.csv", success_rows, _fieldnames(success_rows))

    report_path = _write_report(
        output_root / "llava13b_minimal_replication_summary.md",
        detector_rows,
        mitigation_rows,
        yes_rows,
        success_rows,
        pareto_fig,
        gate_score,
        target_trigger_rate,
        len(trigger_ids),
    )
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "stage2_dir": str(stage2_dir),
        "predictions_path": str(predictions_path),
        "margin_scores_path": str(margin_scores_path),
        "layer": layer,
        "always_alpha": always_alpha,
        "gate_score": gate_score,
        "target_trigger_rate": target_trigger_rate,
        "detector_comparison_path": str(detector_path),
        "mitigation_comparison_path": str(mitigation_path),
        "yes_rate_audit_path": str(yes_path),
        "pareto_points_path": str(pareto_path),
        "pareto_figure_path": pareto_fig,
        "success_criteria_path": str(success_path),
        "report_path": str(report_path),
    }


def _load_geometry(
    path: str | Path,
    predictions: dict[str, dict[str, Any]],
    split_map: dict[str, str],
    margins: dict[str, float],
    layer: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["operator"] == "icd_blind") & (df["layer"] == layer)].copy()
    if df.empty:
        raise ValueError(f"No icd_blind rows for layer {layer} in {path}.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["sample_id"].map(split_map).fillna(df.get("source_subset", ""))
    df["outcome"] = df["sample_id"].map(lambda sample_id: predictions[sample_id]["outcome"])
    df["label"] = df["sample_id"].map(lambda sample_id: predictions[sample_id]["label"])
    df["parsed_prediction"] = df["sample_id"].map(lambda sample_id: predictions[sample_id].get("parsed_prediction", ""))
    df["base_no_minus_yes_logit"] = [
        margins.get(str(row.sample_id), float(row.orig_no_minus_yes_logit))
        for row in df.itertuples(index=False)
    ]
    return df


def _detector_rows(
    geometry: pd.DataFrame,
    target_trigger_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows: list[dict[str, Any]] = []
    scores_by_name: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}
    pred_yes = geometry[geometry["outcome"].isin(["FP", "TP"])].copy()
    train = pred_yes[pred_yes["split"] == "train"].copy()
    calibration = pred_yes[pred_yes["split"] == "calibration"].copy()
    test = pred_yes[pred_yes["split"] == "test"].copy()
    y_train = (train["outcome"] == "FP").astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Detector training split needs both FP and TP predicted-yes samples.")

    for name, columns in DETECTOR_FEATURES.items():
        if any(column not in pred_yes.columns for column in columns):
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[columns].to_numpy(dtype=float))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=13)
        clf.fit(x_train, y_train)
        scores = clf.predict_proba(scaler.transform(pred_yes[columns].to_numpy(dtype=float)))[:, 1]
        scores_by_id = dict(zip(pred_yes["sample_id"].astype(str), scores.astype(float)))
        scores_by_name[name] = scores_by_id
        calibration_scores = np.array([scores_by_id[str(sample_id)] for sample_id in calibration["sample_id"]], dtype=float)
        threshold = _top_rate_threshold(calibration_scores, target_trigger_rate)
        thresholds[name] = threshold
        for split_name, split_df in [("calibration", calibration), ("test", test)]:
            y = (split_df["outcome"] == "FP").astype(int).to_numpy()
            split_scores = np.array([scores_by_id[str(sample_id)] for sample_id in split_df["sample_id"]], dtype=float)
            rows.append(
                _detector_metric_row(
                    method=name,
                    split=split_name,
                    y=y,
                    scores=split_scores,
                    threshold=threshold,
                    target_trigger_rate=target_trigger_rate,
                )
            )
    return rows, scores_by_name, thresholds


def _detector_metric_row(
    method: str,
    split: str,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    target_trigger_rate: float,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    triggered = scores >= threshold
    triggered_fp = int(np.sum(y[triggered])) if len(y) else 0
    trigger_n = int(np.sum(triggered))
    fp_n = int(np.sum(y))
    return {
        "method": method,
        "split": split,
        "n": int(len(y)),
        "fp_n": fp_n,
        "auroc": _safe_metric(lambda: roc_auc_score(y, scores)),
        "auprc": _safe_metric(lambda: average_precision_score(y, scores)),
        "target_trigger_rate": target_trigger_rate,
        "threshold": threshold,
        "trigger_n": trigger_n,
        "trigger_rate": trigger_n / len(y) if len(y) else math.nan,
        "warning_precision": triggered_fp / trigger_n if trigger_n else math.nan,
        "fp_recall": triggered_fp / fp_n if fp_n else math.nan,
    }


def _mitigation_rows(
    stage2_dir: Path,
    predictions: dict[str, dict[str, Any]],
    split_map: dict[str, str],
    detector_scores: dict[str, dict[str, float]],
    gate_thresholds: dict[str, float],
    gate_score: str,
    target_trigger_rate: float,
    layer: int,
    always_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, str]]], set[str]]:
    calibrated = pd.read_csv(stage2_dir / "subspace_vcd_results.csv")
    sample_predictions = pd.read_csv(stage2_dir / "sample_predictions.csv")
    sample_predictions["sample_id"] = sample_predictions["sample_id"].astype(str)
    test_ids = {sample_id for sample_id, split in split_map.items() if split == "test" and sample_id in predictions}
    method_predictions: dict[str, dict[str, dict[str, str]]] = {
        "Base": {
            sample_id: {
                "prediction": str(predictions[sample_id].get("parsed_prediction", "")),
                "outcome": str(predictions[sample_id]["outcome"]),
            }
            for sample_id in test_ids
        }
    }

    selected_specs = [
        ("Full ICD TP-safe", "full", _selected_row(calibrated, layer, "full")),
        ("Band5-16 ICD", "band5_16", _selected_row(calibrated, layer, "band5_16")),
    ]
    for label, subspace, selected in selected_specs:
        if selected is None:
            continue
        method_predictions[label] = _prediction_map(sample_predictions, layer, subspace, float(selected["alpha"]))

    random_rows = _random12_rows(calibrated, layer)
    if random_rows:
        random_mean = _mean_metric_row("Random12 mean (x10)", random_rows)
        random_best = max(random_rows, key=lambda row: (row["fp_reduction"], row["tp_preserved"], row["accuracy_delta"]))
    else:
        random_mean = None
        random_best = None

    always_map = _prediction_map(sample_predictions, layer, "full", _closest_alpha(sample_predictions, layer, "full", always_alpha))
    method_predictions["Always ICD"] = always_map

    trigger_ids = _trigger_ids(predictions, test_ids, detector_scores.get(gate_score, {}), gate_thresholds.get(gate_score, math.inf))
    gated = dict(method_predictions["Base"])
    for sample_id in trigger_ids:
        if sample_id in always_map:
            gated[sample_id] = always_map[sample_id]
    method_predictions[f"Gated ICD ({gate_score}@{target_trigger_rate:.0%})"] = gated

    rows = [_metric_row_from_map("Base", "baseline", method_predictions["Base"], predictions, test_ids, "No correction.")]
    for label in ["Full ICD TP-safe", "Band5-16 ICD", "Always ICD", f"Gated ICD ({gate_score}@{target_trigger_rate:.0%})"]:
        if label in method_predictions:
            rows.append(_metric_row_from_map(label, "mitigation", method_predictions[label], predictions, test_ids, ""))
    if random_mean:
        rows.append(random_mean)
    if random_best is not None:
        out = dict(random_best)
        out["method"] = "Random12 best (x10)"
        out["family"] = "random_control"
        out["notes"] = "Best random12_sXX TP-safe row."
        rows.append(out)
    return rows, method_predictions, trigger_ids


def _selected_row(calibrated: pd.DataFrame, layer: int, subspace: str) -> pd.Series | None:
    rows = calibrated[
        (calibrated["operator"] == "icd_blind")
        & (calibrated["layer"] == layer)
        & (calibrated["subspace"] == subspace)
    ].copy()
    if rows.empty:
        return None
    rows["_score"] = rows["fp_reduction"].fillna(-1) - (1 - rows["tp_preserved"].fillna(0))
    return rows.sort_values(["_score", "fp_reduction", "tp_preserved"], ascending=[False, False, False]).iloc[0]


def _random12_rows(calibrated: pd.DataFrame, layer: int) -> list[dict[str, Any]]:
    rows = calibrated[
        (calibrated["operator"] == "icd_blind")
        & (calibrated["layer"] == layer)
        & (calibrated["subspace"].astype(str).str.match(r"random12_s\d+"))
    ]
    return [_stage2_metric_row(f"Random12 {row.subspace}", "random_control", row._asdict(), "") for row in rows.itertuples(index=False)]


def _mean_metric_row(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ["fp_reduction", "tp_preserved", "accuracy_delta", "overall_yes_rate", "fp_yes_rate"]
    out = {"method": method, "family": "random_control", "setting": "mean over random12_s00-s09", "notes": ""}
    for key in numeric:
        out[key] = _fmt_float(float(np.nanmean([float(row[key]) for row in rows])))
    return out


def _stage2_metric_row(method: str, family: str, row: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "method": method,
        "family": family,
        "setting": f"alpha={_fmt_float(row.get('alpha'))}",
        "fp_reduction": _fmt_float(row.get("fp_reduction")),
        "tp_preserved": _fmt_float(row.get("tp_preserved")),
        "accuracy_delta": _fmt_float(row.get("accuracy_delta")),
        "overall_yes_rate": _fmt_float(row.get("yes_rate_after")),
        "fp_yes_rate": _fmt_float(1.0 - float(row["fp_reduction"])) if _is_number(row.get("fp_reduction")) else "",
        "notes": notes,
    }


def _prediction_map(sample_predictions: pd.DataFrame, layer: int, subspace: str, alpha: float) -> dict[str, dict[str, str]]:
    rows = sample_predictions[
        (sample_predictions["operator"] == "icd_blind")
        & (sample_predictions["layer"] == layer)
        & (sample_predictions["split"] == "test")
        & (sample_predictions["subspace"] == subspace)
        & (np.isclose(sample_predictions["alpha"], alpha))
    ]
    return {
        str(row.sample_id): {
            "prediction": str(row.final_prediction),
            "outcome": str(row.final_outcome),
        }
        for row in rows.itertuples(index=False)
    }


def _closest_alpha(sample_predictions: pd.DataFrame, layer: int, subspace: str, target: float) -> float:
    rows = sample_predictions[
        (sample_predictions["operator"] == "icd_blind")
        & (sample_predictions["layer"] == layer)
        & (sample_predictions["split"] == "test")
        & (sample_predictions["subspace"] == subspace)
    ]
    if rows.empty:
        raise ValueError(f"No sample predictions for layer={layer}, subspace={subspace}.")
    alphas = sorted(float(value) for value in rows["alpha"].dropna().unique())
    return min(alphas, key=lambda value: abs(value - target))


def _trigger_ids(
    predictions: dict[str, dict[str, Any]],
    test_ids: set[str],
    scores: dict[str, float],
    threshold: float,
) -> set[str]:
    return {
        sample_id
        for sample_id in test_ids
        if sample_id in scores
        and scores[sample_id] >= threshold
        and str(predictions[sample_id].get("outcome", "")) in {"FP", "TP"}
    }


def _metric_row_from_map(
    method: str,
    family: str,
    after: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, Any]],
    test_ids: set[str],
    notes: str,
) -> dict[str, Any]:
    original_outcomes = [str(predictions[sample_id]["outcome"]) for sample_id in sorted(test_ids)]
    final_outcomes = [after.get(sample_id, {"outcome": predictions[sample_id]["outcome"]})["outcome"] for sample_id in sorted(test_ids)]
    final_predictions = [after.get(sample_id, {"prediction": predictions[sample_id].get("parsed_prediction", "")})["prediction"] for sample_id in sorted(test_ids)]
    original_fp = sum(1 for item in original_outcomes if item == "FP")
    original_tp = sum(1 for item in original_outcomes if item == "TP")
    fp_fixed = sum(1 for before, after_value in zip(original_outcomes, final_outcomes) if before == "FP" and after_value == "TN")
    tp_kept = sum(1 for before, after_value in zip(original_outcomes, final_outcomes) if before == "TP" and after_value == "TP")
    fp_yes = sum(
        1
        for before, pred in zip(original_outcomes, final_predictions)
        if before == "FP" and pred == "yes"
    )
    counts = _counts(final_outcomes)
    return {
        "method": method,
        "family": family,
        "setting": "",
        "fp_reduction": _fmt_float(fp_fixed / original_fp if original_fp else math.nan),
        "tp_preserved": _fmt_float(tp_kept / original_tp if original_tp else math.nan),
        "accuracy_delta": _fmt_float(_accuracy(counts) - _accuracy(_counts(original_outcomes))),
        "overall_yes_rate": _fmt_float(sum(1 for item in final_predictions if item == "yes") / len(final_predictions)),
        "fp_yes_rate": _fmt_float(fp_yes / original_fp if original_fp else math.nan),
        "notes": notes,
    }


def _yes_rate_rows(
    predictions: dict[str, dict[str, Any]],
    method_predictions: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in method_predictions.items():
        sample_ids = sorted(values)
        original = [str(predictions[sample_id]["outcome"]) for sample_id in sample_ids]
        final_predictions = [values[sample_id]["prediction"] for sample_id in sample_ids]
        final_outcomes = [values[sample_id]["outcome"] for sample_id in sample_ids]
        rows.append(
            {
                "method": method,
                "n": len(sample_ids),
                "overall_yes_rate": _fmt_float(np.mean([pred == "yes" for pred in final_predictions])),
                "tp_yes_rate": _fmt_float(_conditional_rate(original, final_predictions, "TP", "yes")),
                "fp_yes_rate": _fmt_float(_conditional_rate(original, final_predictions, "FP", "yes")),
                "tn_yes_rate": _fmt_float(_conditional_rate(original, final_predictions, "TN", "yes")),
                "fn_rate_after": _fmt_float(sum(1 for item in final_outcomes if item == "FN") / len(final_outcomes) if final_outcomes else math.nan),
                "accuracy": _fmt_float(_accuracy(_counts(final_outcomes))),
            }
        )
    return rows


def _pareto_rows(stage2_alpha_sweep_path: Path, layer: int) -> list[dict[str, Any]]:
    df = pd.read_csv(stage2_alpha_sweep_path)
    rows = df[
        (df["operator"] == "icd_blind")
        & (df["layer"] == layer)
        & (df["split"] == "test")
        & (
            df["subspace"].isin(["full", "band5_16"])
            | df["subspace"].astype(str).str.match(r"random12_s\d+")
        )
    ].copy()
    rows["tp_damage"] = 1 - rows["tp_preserved"]
    rows["method_label"] = rows["subspace"].map({"full": "Full ICD", "band5_16": "Band5-16 ICD"}).fillna("Random12")
    return [
        {
            "method": row.method_label,
            "subspace": row.subspace,
            "alpha": row.alpha,
            "fp_reduction": row.fp_reduction,
            "tp_damage": row.tp_damage,
            "tp_preserved": row.tp_preserved,
            "accuracy_delta": row.accuracy_delta,
        }
        for row in rows.itertuples(index=False)
    ]


def _success_rows(detector_rows: list[dict[str, Any]], mitigation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detector = pd.DataFrame(detector_rows)
    mitigation = {row["method"]: row for row in mitigation_rows}
    test_detector = detector[detector["split"] == "test"] if not detector.empty else pd.DataFrame()
    margin = _detector_value(test_detector, "margin-only", "auprc")
    tail = _detector_value(test_detector, "margin+tail", "auprc")
    full = _detector_value(test_detector, "margin+full", "auprc")
    band = mitigation.get("Band5-16 ICD", {})
    full_icd = mitigation.get("Full ICD TP-safe", {})
    always = mitigation.get("Always ICD", {})
    gated = next((row for row in mitigation_rows if row["method"].startswith("Gated ICD")), {})
    base = mitigation.get("Base", {})
    return [
        {
            "criterion": "Detector margin+tail/full beats margin-only",
            "status": _pass(max(tail, full) > margin),
            "value": f"margin={_fmt_float(margin)}, tail={_fmt_float(tail)}, full={_fmt_float(full)}",
        },
        {
            "criterion": "Band5-16 TP-safe beats Full ICD TP-safe in FP reduction",
            "status": _pass(_float(band.get("fp_reduction")) > _float(full_icd.get("fp_reduction"))),
            "value": f"band={band.get('fp_reduction', '')}, full={full_icd.get('fp_reduction', '')}",
        },
        {
            "criterion": "Gated ICD keeps most Always ICD FP reduction with higher TP preserved",
            "status": _pass(
                _float(gated.get("fp_reduction")) >= 0.8 * _float(always.get("fp_reduction"))
                and _float(gated.get("tp_preserved")) > _float(always.get("tp_preserved"))
            ),
            "value": f"gated fp/tp={gated.get('fp_reduction', '')}/{gated.get('tp_preserved', '')}; always fp/tp={always.get('fp_reduction', '')}/{always.get('tp_preserved', '')}",
        },
        {
            "criterion": "Always ICD shows stronger conservative bias than Base",
            "status": _pass(_float(always.get("overall_yes_rate")) < _float(base.get("overall_yes_rate"))),
            "value": f"always yes={always.get('overall_yes_rate', '')}, base yes={base.get('overall_yes_rate', '')}",
        },
    ]


def _plot_pareto(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/vgs_mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, group in df.groupby("method", sort=False):
        alpha = 0.25 if method == "Random12" else 0.9
        ax.scatter(group["tp_damage"], group["fp_reduction"], label=method, alpha=alpha, s=24)
    ax.set_xlabel("TP damage (1 - TP preserved)")
    ax.set_ylabel("FP reduction")
    ax.set_title("LLaVA-13B minimal ICD Pareto")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_report(
    path: Path,
    detector_rows: list[dict[str, Any]],
    mitigation_rows: list[dict[str, Any]],
    yes_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    pareto_fig: str,
    gate_score: str,
    target_trigger_rate: float,
    trigger_n: int,
) -> Path:
    lines = [
        "# LLaVA-13B Minimal Replication",
        "",
        f"- Gated ICD score: `{gate_score}` at target trigger rate {target_trigger_rate:.0%}.",
        f"- Triggered predicted-yes test samples: {trigger_n}.",
        f"- Pareto figure: `{pareto_fig}`" if pareto_fig else "- Pareto figure: unavailable.",
        "",
        "## Success Criteria",
        "",
    ]
    lines.extend(_markdown_table(success_rows))
    lines.extend(["", "## Detector", ""])
    lines.extend(_markdown_table([row for row in detector_rows if row["split"] == "test"]))
    lines.extend(["", "## Mitigation", ""])
    lines.extend(_markdown_table(mitigation_rows))
    lines.extend(["", "## Yes-Rate Audit", ""])
    lines.extend(_markdown_table(yes_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_split_map(split_dir: str | Path) -> dict[str, str]:
    root = Path(split_dir)
    mapping: dict[str, str] = {}
    for filename, split in [
        ("pope_train_ids.json", "train"),
        ("pope_val_ids.json", "calibration"),
        ("pope_test_ids.json", "test"),
    ]:
        path = root / filename
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for sample_id in payload.get("sample_ids", []):
            mapping[str(sample_id)] = split
    return mapping


def _load_margins(path: str | Path) -> dict[str, float]:
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "no_minus_yes_logit" not in df.columns:
        return {}
    return {str(row.sample_id): float(row.no_minus_yes_logit) for row in df.itertuples(index=False)}


def _top_rate_threshold(scores: np.ndarray, target_rate: float) -> float:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return math.inf
    keep = max(1, int(math.ceil(len(finite) * target_rate)))
    return float(np.sort(finite)[-keep])


def _safe_metric(fn: Any) -> float:
    try:
        return float(fn())
    except ValueError:
        return math.nan


def _detector_value(df: pd.DataFrame, method: str, metric: str) -> float:
    if df.empty:
        return math.nan
    row = df[df["method"] == method]
    if row.empty:
        return math.nan
    return float(row.iloc[0][metric])


def _counts(outcomes: list[str]) -> dict[str, int]:
    return {key: sum(1 for item in outcomes if item == key) for key in ["TP", "TN", "FP", "FN", "unknown"]}


def _accuracy(counts: dict[str, int]) -> float:
    denom = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    return (counts["TP"] + counts["TN"]) / denom if denom else math.nan


def _conditional_rate(original_outcomes: list[str], predictions: list[str], outcome: str, prediction: str) -> float:
    denom = sum(1 for item in original_outcomes if item == outcome)
    if not denom:
        return math.nan
    return sum(1 for before, after in zip(original_outcomes, predictions) if before == outcome and after == prediction) / denom


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    fields = _fieldnames(rows)
    out = ["| " + " | ".join(_title(field) for field in fields) + " |"]
    out.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return out


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def _pass(value: bool) -> str:
    return "pass" if bool(value) else "fail"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.3f}"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


if __name__ == "__main__":
    main()
