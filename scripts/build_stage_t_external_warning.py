#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json, write_jsonl


DEFAULT_SCORES = [
    "pls32_probe",
    "tail_257_1024_probe",
    "full_probe",
    "top_4_probe",
    "random64_probe",
    "tail_257_1024_energy",
    "margin_probe",
    "low_margin_probe",
    "margin_plus_pls32_probe",
    "margin_plus_tail_257_1024_probe",
    "margin_plus_full_probe",
    "low_margin_plus_pls32_probe",
    "low_margin_plus_tail_257_1024_probe",
    "low_margin_plus_full_probe",
]
DEFAULT_POLICIES = ["pope_calibrated_threshold", "external_top_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Stage T external warning/abstention transfer metrics and VCD gate assignments."
    )
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--external-scores", default=None)
    parser.add_argument("--gate-metrics", default=None)
    parser.add_argument("--output-dir", default="outputs/stage_t_external_amber_fixed_ids")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument("--scores", nargs="*", default=DEFAULT_SCORES)
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_external_warning(
        stage_t_dir=args.stage_t_dir,
        external_scores_path=args.external_scores,
        gate_metrics_path=args.gate_metrics,
        output_dir=args.output_dir,
        target_rates=args.target_rates,
        selected_scores=args.scores,
        policies=args.policies,
        random_repeats=args.random_repeats,
        seed=args.seed,
    )
    summary_path = write_json(Path(args.output_dir) / "build_stage_t_external_warning_summary.json", result)
    append_experiment_log(args.log_path, "build_stage_t_external_warning", summary_path, "ok")
    print(summary_path)


def build_external_warning(
    stage_t_dir: str | Path,
    external_scores_path: str | Path | None,
    gate_metrics_path: str | Path | None,
    output_dir: str | Path,
    target_rates: list[float],
    selected_scores: list[str],
    policies: list[str],
    random_repeats: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(stage_t_dir)
    output_root = ensure_dir(output_dir)
    external_path = Path(external_scores_path) if external_scores_path else root / "stage_t_external_scores.csv"
    gate_path = Path(gate_metrics_path) if gate_metrics_path else root / "stage_t_gate_metrics.csv"
    external = pd.read_csv(external_path)
    gate = pd.read_csv(gate_path)
    target_keys = {_rate_key(rate) for rate in target_rates}
    gate = gate[gate["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_keys)].copy()

    available_scores = [score for score in selected_scores if score in external.columns and score in set(gate["score"].astype(str))]
    missing_scores = [score for score in selected_scores if score not in available_scores]
    external["parsed_prediction"] = external["parsed_prediction"].astype(str).str.lower()
    pred_yes = external[
        (external["parsed_prediction"] == "yes")
        & (external["outcome"].astype(str).isin(["FP", "TP"]))
    ].copy()
    if pred_yes.empty:
        raise ValueError(f"No external predicted-Yes FP/TP rows in {external_path}")

    rows: list[dict[str, Any]] = []
    assignment_rows: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    pool_rows = _external_pool_rows(pred_yes)
    rng = np.random.default_rng(seed)
    for layer, layer_pred_yes in pred_yes.groupby("layer", dropna=False):
        rows.extend(_original_and_always_rows(layer, external[external["layer"] == layer], layer_pred_yes))
        layer_gate = gate[gate["layer"].astype(str) == str(layer)]
        for _, gate_row in layer_gate[layer_gate["score"].isin(available_scores)].iterrows():
            score = str(gate_row["score"])
            target_rate = float(gate_row["target_trigger_rate_predicted_yes"])
            score_values = pd.to_numeric(layer_pred_yes[score], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(score_values)
            for policy in policies:
                mask = _policy_mask(
                    policy=policy,
                    scores=score_values,
                    finite=finite,
                    threshold=float(gate_row["threshold"]),
                    target_rate=target_rate,
                )
                row = _warning_row(
                    layer=layer,
                    policy=policy,
                    score=score,
                    target_rate=target_rate,
                    threshold=float(gate_row["threshold"]),
                    pred_yes=layer_pred_yes,
                    mask=mask,
                    aggregation="deterministic",
                    matched_score="",
                    random_repeats="",
                )
                rows.append(row)
                assignment_rows.setdefault(policy, []).extend(
                    _assignment_rows(
                        layer=layer,
                        policy=policy,
                        score=score,
                        target_rate=target_rate,
                        threshold=float(gate_row["threshold"]),
                        pred_yes=layer_pred_yes,
                        mask=mask,
                    )
                )
                rows.append(
                    _random_warning_row(
                        layer=layer,
                        policy=f"same_trigger_random_{policy}",
                        target_rate=target_rate,
                        pred_yes=layer_pred_yes,
                        n_trigger=int(mask.sum()),
                        matched_score=score,
                        random_repeats=random_repeats,
                        rng=rng,
                    )
                )

    metrics_path = write_csv(
        output_root / "stage_t_external_warning_metrics.csv",
        rows,
        _fieldnames(rows),
    )
    assignment_paths: dict[str, str] = {}
    for policy, policy_rows in assignment_rows.items():
        path = output_root / f"stage_t_external_gate_assignments_{policy}.csv"
        assignment_paths[policy] = str(write_csv(path, policy_rows, _fieldnames(policy_rows)))
    pool_path = write_jsonl(output_root / "stage_t_external_vcd_pool.jsonl", pool_rows)
    md_path = _write_markdown(
        output_root / "stage_t_external_warning_metrics.md",
        rows,
        missing_scores=missing_scores,
    )
    return {
        "stage_t_dir": str(root),
        "external_scores_path": str(external_path),
        "gate_metrics_path": str(gate_path),
        "output_dir": str(output_root),
        "target_rates": target_rates,
        "policies": policies,
        "available_selected_scores": available_scores,
        "missing_selected_scores": missing_scores,
        "metrics_path": str(metrics_path),
        "markdown_path": str(md_path),
        "assignment_paths": assignment_paths,
        "vcd_pool_path": str(pool_path),
        "num_rows": len(rows),
        "num_external_predicted_yes": len(pool_rows),
    }


def _policy_mask(
    policy: str,
    scores: np.ndarray,
    finite: np.ndarray,
    threshold: float,
    target_rate: float,
) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    if not finite.any():
        return mask
    if policy == "pope_calibrated_threshold":
        mask[finite] = scores[finite] >= threshold
        return mask
    if policy == "external_top_rate":
        finite_idx = np.flatnonzero(finite)
        n_trigger = max(1, int(math.ceil(target_rate * len(finite_idx))))
        order = finite_idx[np.argsort(scores[finite_idx])[::-1]]
        mask[order[:n_trigger]] = True
        return mask
    raise ValueError(f"Unknown external warning policy: {policy}")


def _original_and_always_rows(layer: Any, external_layer: pd.DataFrame, pred_yes: pd.DataFrame) -> list[dict[str, Any]]:
    total_fp = int((pred_yes["outcome"].astype(str) == "FP").sum())
    total_tp = int((pred_yes["outcome"].astype(str) == "TP").sum())
    predicted_yes_n = len(pred_yes)
    quality = _quality(external_layer["outcome"].astype(str).tolist())
    return [
        {
            "layer": layer,
            "selection_policy": "none",
            "method": "Original",
            "gate_family": "none",
            "score": "",
            "matched_score": "",
            "target_trigger_rate_predicted_yes": 0.0,
            "threshold": "",
            "aggregation": "deterministic",
            "random_repeats": "",
            "trigger_n": 0,
            "predicted_yes_n": predicted_yes_n,
            "trigger_rate_predicted_yes": 0.0,
            "fp_captured": 0,
            "fp_capture_rate": 0.0,
            "tp_damaged": 0,
            "tp_damage": 0.0,
            "tp_preserved": 1.0,
            "warning_precision": math.nan,
            "compute_saved_vs_always": 1.0,
            "original_accuracy": quality["accuracy"],
            "original_f1": quality["f1"],
            "original_fp_rate": quality["fp_rate"],
        },
        {
            "layer": layer,
            "selection_policy": "always",
            "method": "Always external warning",
            "gate_family": "always",
            "score": "always_predicted_yes",
            "matched_score": "",
            "target_trigger_rate_predicted_yes": 1.0,
            "threshold": "",
            "aggregation": "deterministic",
            "random_repeats": "",
            "trigger_n": predicted_yes_n,
            "predicted_yes_n": predicted_yes_n,
            "trigger_rate_predicted_yes": 1.0,
            "fp_captured": total_fp,
            "fp_capture_rate": 1.0,
            "tp_damaged": total_tp,
            "tp_damage": 1.0,
            "tp_preserved": 0.0,
            "warning_precision": total_fp / predicted_yes_n if predicted_yes_n else math.nan,
            "compute_saved_vs_always": 0.0,
            "original_accuracy": quality["accuracy"],
            "original_f1": quality["f1"],
            "original_fp_rate": quality["fp_rate"],
        },
    ]


def _warning_row(
    layer: Any,
    policy: str,
    score: str,
    target_rate: float,
    threshold: float,
    pred_yes: pd.DataFrame,
    mask: np.ndarray,
    aggregation: str,
    matched_score: str,
    random_repeats: int | str,
) -> dict[str, Any]:
    total_fp = int((pred_yes["outcome"].astype(str) == "FP").sum())
    total_tp = int((pred_yes["outcome"].astype(str) == "TP").sum())
    triggered = pred_yes.iloc[np.flatnonzero(mask)]
    triggered_fp = int((triggered["outcome"].astype(str) == "FP").sum())
    triggered_tp = int((triggered["outcome"].astype(str) == "TP").sum())
    trigger_n = len(triggered)
    return {
        "layer": layer,
        "selection_policy": policy,
        "method": _method_name(score),
        "gate_family": _gate_family(score),
        "score": score,
        "matched_score": matched_score,
        "target_trigger_rate_predicted_yes": target_rate,
        "threshold": threshold,
        "aggregation": aggregation,
        "random_repeats": random_repeats,
        "trigger_n": trigger_n,
        "predicted_yes_n": len(pred_yes),
        "trigger_rate_predicted_yes": trigger_n / len(pred_yes) if len(pred_yes) else math.nan,
        "fp_captured": triggered_fp,
        "fp_capture_rate": triggered_fp / total_fp if total_fp else math.nan,
        "tp_damaged": triggered_tp,
        "tp_damage": triggered_tp / total_tp if total_tp else math.nan,
        "tp_preserved": 1.0 - triggered_tp / total_tp if total_tp else math.nan,
        "warning_precision": triggered_fp / trigger_n if trigger_n else math.nan,
        "compute_saved_vs_always": 1.0 - trigger_n / len(pred_yes) if len(pred_yes) else math.nan,
        "original_accuracy": "",
        "original_f1": "",
        "original_fp_rate": "",
    }


def _random_warning_row(
    layer: Any,
    policy: str,
    target_rate: float,
    pred_yes: pd.DataFrame,
    n_trigger: int,
    matched_score: str,
    random_repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if n_trigger <= 0:
        return _warning_row(layer, policy, "same_trigger_random", target_rate, math.nan, pred_yes, np.zeros(len(pred_yes), dtype=bool), "random_mean", matched_score, random_repeats)
    n_trigger = min(n_trigger, len(pred_yes))
    metrics = []
    for _ in range(random_repeats):
        mask = np.zeros(len(pred_yes), dtype=bool)
        mask[rng.choice(len(pred_yes), size=n_trigger, replace=False)] = True
        metrics.append(
            _warning_row(
                layer=layer,
                policy=policy,
                score="same_trigger_random",
                target_rate=target_rate,
                threshold=math.nan,
                pred_yes=pred_yes,
                mask=mask,
                aggregation="random_sample",
                matched_score=matched_score,
                random_repeats=random_repeats,
            )
        )
    row = metrics[0].copy()
    row["method"] = "Random external warning"
    row["gate_family"] = "same_trigger_random"
    row["aggregation"] = "random_mean"
    row["matched_score"] = matched_score
    for key in [
        "fp_captured",
        "fp_capture_rate",
        "tp_damaged",
        "tp_damage",
        "tp_preserved",
        "warning_precision",
    ]:
        row[key] = float(np.mean([float(item[key]) for item in metrics]))
    return row


def _assignment_rows(
    layer: Any,
    policy: str,
    score: str,
    target_rate: float,
    threshold: float,
    pred_yes: pd.DataFrame,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "selection_policy": policy,
            "score": score,
            "target_trigger_rate_predicted_yes": target_rate,
            "threshold": threshold,
            "sample_id": str(row.sample_id),
        }
        for row in pred_yes.iloc[np.flatnonzero(mask)].itertuples(index=False)
    ]


def _external_pool_rows(pred_yes: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for row in pred_yes.sort_values(["sample_id", "layer"]).itertuples(index=False):
        sample_id = str(row.sample_id)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "subset": str(getattr(row, "subset", "")),
                "dimension": str(getattr(row, "dimension", "")),
                "image": str(getattr(row, "image", "")),
                "image_path": str(getattr(row, "image_path", "")),
                "question": str(getattr(row, "question", "")),
                "label": str(getattr(row, "label", "")),
                "original_outcome": str(getattr(row, "outcome", "")),
                "original_parsed_prediction": str(getattr(row, "parsed_prediction", "")),
            }
        )
    return rows


def _quality(outcomes: list[str]) -> dict[str, float]:
    tp = sum(outcome == "TP" for outcome in outcomes)
    tn = sum(outcome == "TN" for outcome in outcomes)
    fp = sum(outcome == "FP" for outcome in outcomes)
    fn = sum(outcome == "FN" for outcome in outcomes)
    total = len(outcomes)
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "accuracy": (tp + tn) / total if total else math.nan,
        "f1": f1,
        "fp_rate": fp / (fp + tn) if fp + tn else math.nan,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]], missing_scores: list[str]) -> Path:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage T External Warning Transfer",
        "",
        "External warning applies POPE-trained Stage T gates to AMBER predicted-Yes samples.",
        "",
    ]
    if missing_scores:
        lines.extend(["Missing requested scores: " + ", ".join(f"`{score}`" for score in missing_scores), ""])
    for policy in [item for item in DEFAULT_POLICIES if item in set(df["selection_policy"])]:
        lines.extend(
            [
                f"## {policy}",
                "",
                "| Target | Method | Score | Trigger Rate | FP Recall | TP Damage | Warning Precision | Compute Saved |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        subset = df[
            (df["selection_policy"] == policy)
            & (df["aggregation"] == "deterministic")
            & (~df["method"].isin(["Original", "Always external warning"]))
        ].copy()
        subset = subset.sort_values(
            ["target_trigger_rate_predicted_yes", "warning_precision", "fp_capture_rate"],
            ascending=[True, False, False],
        )
        for row in subset.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row.target_trigger_rate_predicted_yes),
                        str(row.method),
                        f"`{row.score}`",
                        _fmt(row.trigger_rate_predicted_yes),
                        _fmt(row.fp_capture_rate),
                        _fmt(row.tp_damage),
                        _fmt(row.warning_precision),
                        _fmt(row.compute_saved_vs_always),
                    ]
                )
                + " |"
            )
        lines.append("")
        random_subset = df[
            (df["selection_policy"] == f"same_trigger_random_{policy}")
            & (df["aggregation"] == "random_mean")
        ].copy()
        if not random_subset.empty:
            lines.extend(
                [
                    f"### Random baseline for {policy}",
                    "",
                    "| Target | Matched Score | Trigger Rate | FP Recall | TP Damage | Warning Precision |",
                    "| ---: | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            random_subset = random_subset.sort_values(
                ["target_trigger_rate_predicted_yes", "warning_precision"],
                ascending=[True, False],
            )
            for row in random_subset.itertuples(index=False):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _fmt(row.target_trigger_rate_predicted_yes),
                            f"`{row.matched_score}`",
                            _fmt(row.trigger_rate_predicted_yes),
                            _fmt(row.fp_capture_rate),
                            _fmt(row.tp_damage),
                            _fmt(row.warning_precision),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _method_name(score: str) -> str:
    if score == "pls32_probe":
        return "PLS external warning"
    if score == "tail_257_1024_probe":
        return "Tail external warning"
    if score == "tail_257_1024_energy":
        return "Tail-energy external warning"
    if score == "full_probe":
        return "FullD external warning"
    if score == "top_4_probe":
        return "Top-4 external warning"
    if score == "random64_probe":
        return "Random-subspace external warning"
    if score == "same_trigger_random":
        return "Random external warning"
    if score == "low_margin_probe":
        return "Low-margin external warning"
    if score.startswith("low_margin_plus_"):
        return "Low-margin+Geometry external warning"
    if score == "margin_probe":
        return "Margin external warning"
    if score.startswith("margin_plus_"):
        return "Margin+Geometry external warning"
    return f"{score} external warning"


def _gate_family(score: str) -> str:
    if score == "same_trigger_random":
        return "same_trigger_random"
    if score.startswith("low_margin_plus_"):
        return "low_margin_plus_geometry"
    if score == "low_margin_probe":
        return "low_margin"
    if score.startswith("margin_plus_"):
        return "margin_plus_geometry"
    if score == "margin_probe":
        return "margin"
    if score.startswith("top_"):
        return "top_variance"
    if score.startswith("tail_") or score in {"pls32_probe", "full_probe"}:
        return "geometry"
    if score == "random64_probe":
        return "random_subspace"
    return "score_gate"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
