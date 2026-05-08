#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json


DEFAULT_SCORES = [
    "pls32_probe",
    "tail_257_1024_probe",
    "tail_257_1024_energy",
    "full_probe",
    "top_4_probe",
    "random64_probe",
    "margin_probe",
    "low_margin_probe",
    "margin_plus_pls32_probe",
    "margin_plus_tail_257_1024_probe",
    "margin_plus_full_probe",
    "low_margin_plus_pls32_probe",
    "low_margin_plus_tail_257_1024_probe",
    "low_margin_plus_full_probe",
]

OUTCOME_CODES = {"TP": 0, "TN": 1, "FP": 2, "FN": 3, "unknown": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for Stage T gated VCD/ICD metrics.")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument(
        "--gate-assignments",
        default="outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_gate_assignments.csv",
    )
    parser.add_argument(
        "--vcd-predictions",
        default="outputs/stage_t_selective_correction_fixed_ids/stage_t_vcd_predictions_icd_blind.jsonl",
    )
    parser.add_argument("--operator", default="icd_blind")
    parser.add_argument("--test-subset", default="test")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.2, 0.3])
    parser.add_argument("--scores", nargs="*", default=DEFAULT_SCORES)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = bootstrap_vcd_results(
        predictions_path=args.predictions,
        gate_assignments_path=args.gate_assignments,
        vcd_predictions_path=args.vcd_predictions,
        operator=args.operator,
        test_subset=args.test_subset,
        split_dir=args.split_dir,
        target_rates=args.target_rates,
        selected_scores=args.scores,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary_path = write_json(
        Path(args.output_dir) / f"bootstrap_stage_t_vcd_results_{args.operator}_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "bootstrap_stage_t_vcd_results", summary_path, "ok")
    print(summary_path)


def bootstrap_vcd_results(
    predictions_path: str | Path,
    gate_assignments_path: str | Path,
    vcd_predictions_path: str | Path,
    operator: str,
    test_subset: str,
    split_dir: str | Path | None,
    target_rates: list[float],
    selected_scores: list[str],
    n_bootstrap: int,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    all_predictions = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    test_ids = _test_ids(all_predictions, test_subset, split_dir)
    predictions = {sample_id: all_predictions[sample_id] for sample_id in test_ids if sample_id in all_predictions}
    sample_ids = list(predictions)
    original_codes = np.array([_code(predictions[sample_id].get("outcome", "unknown")) for sample_id in sample_ids])
    predicted_yes_mask = np.array(
        [
            str(predictions[sample_id].get("parsed_prediction", "")).lower() == "yes"
            and str(predictions[sample_id].get("outcome", "")) in {"FP", "TP"}
            for sample_id in sample_ids
        ],
        dtype=bool,
    )
    vcd_rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(vcd_predictions_path)
        if row.get("vcd_parsed_prediction") in {"yes", "no"}
    }
    vcd_codes = np.array(
        [
            _code(vcd_rows[sample_id].get("vcd_outcome", "unknown"))
            if sample_id in vcd_rows
            else original_codes[idx]
            for idx, sample_id in enumerate(sample_ids)
        ],
        dtype=np.int64,
    )
    assignments = pd.read_csv(gate_assignments_path)
    target_keys = {_rate_key(rate) for rate in target_rates}
    assignments = assignments[
        assignments["target_trigger_rate_predicted_yes"].map(_rate_key).isin(target_keys)
    ].copy()
    available_scores = set(assignments["score"].dropna().astype(str))
    wanted_scores = [score for score in selected_scores if score in available_scores]
    missing_scores = [score for score in selected_scores if score not in available_scores]

    methods: list[dict[str, Any]] = [
        {
            "method": "Always VCD/ICD",
            "score": "always_predicted_yes",
            "target_trigger_rate_predicted_yes": 1.0,
            "trigger_mask": predicted_yes_mask,
        }
    ]
    for (layer, score, target_rate), group in assignments.groupby(
        ["layer", "score", "target_trigger_rate_predicted_yes"],
        dropna=False,
    ):
        score = str(score)
        if score not in wanted_scores:
            continue
        ids = {str(item) for item in group["sample_id"].tolist()}
        trigger_mask = np.array([sample_id in ids and sample_id in vcd_rows for sample_id in sample_ids], dtype=bool)
        methods.append(
            {
                "layer": layer,
                "method": _method_name(score),
                "score": score,
                "target_trigger_rate_predicted_yes": float(target_rate),
                "trigger_mask": trigger_mask,
            }
        )

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(sample_ids), size=(n_bootstrap, len(sample_ids)))
    rows: list[dict[str, Any]] = []
    for method in methods:
        trigger_mask = method["trigger_mask"]
        after_codes = np.where(trigger_mask, vcd_codes, original_codes)
        point = _metric_values(original_codes, after_codes, trigger_mask)
        boot = _bootstrap_metrics(original_codes, after_codes, trigger_mask, boot_indices)
        for metric_name, point_value in point.items():
            values = boot[metric_name]
            rows.append(
                {
                    "operator": operator,
                    "layer": method.get("layer", ""),
                    "method": method["method"],
                    "score": method["score"],
                    "target_trigger_rate_predicted_yes": method["target_trigger_rate_predicted_yes"],
                    "metric": metric_name,
                    "point": point_value,
                    "ci_low": _nanquantile(values, 0.025),
                    "ci_high": _nanquantile(values, 0.975),
                    "n_bootstrap": n_bootstrap,
                    "trigger_n": int(trigger_mask.sum()),
                }
            )

    output_root = Path(output_dir)
    ci_path = write_csv(
        output_root / f"stage_t_vcd_bootstrap_ci_{operator}.csv",
        rows,
        _fieldnames(rows),
    )
    md_path = _write_markdown(output_root / f"stage_t_vcd_bootstrap_ci_{operator}.md", rows)
    return {
        "operator": operator,
        "ci_path": str(ci_path),
        "markdown_path": str(md_path),
        "num_rows": len(rows),
        "num_test_predictions": len(sample_ids),
        "n_bootstrap": n_bootstrap,
        "available_selected_scores": wanted_scores,
        "missing_selected_scores": missing_scores,
    }


def _metric_values(
    original_codes: np.ndarray,
    after_codes: np.ndarray,
    trigger_mask: np.ndarray,
) -> dict[str, float]:
    original = _counts(original_codes)
    after = _counts(after_codes)
    original_acc = (original["TP"] + original["TN"]) / len(original_codes)
    after_acc = (after["TP"] + after["TN"]) / len(after_codes)
    trigger_n = int(trigger_mask.sum())
    fp_reduced = original["FP"] - after["FP"]
    return {
        "fp_reduction": fp_reduced / original["FP"] if original["FP"] else math.nan,
        "tp_preserved": after["TP"] / original["TP"] if original["TP"] else math.nan,
        "accuracy_delta": after_acc - original_acc,
        "fp_reduction_per_trigger": fp_reduced / trigger_n if trigger_n else math.nan,
    }


def _bootstrap_metrics(
    original_codes: np.ndarray,
    after_codes: np.ndarray,
    trigger_mask: np.ndarray,
    boot_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    original_boot = original_codes[boot_indices]
    after_boot = after_codes[boot_indices]
    trigger_boot = trigger_mask[boot_indices]
    original_fp = (original_boot == OUTCOME_CODES["FP"]).sum(axis=1)
    after_fp = (after_boot == OUTCOME_CODES["FP"]).sum(axis=1)
    original_tp = (original_boot == OUTCOME_CODES["TP"]).sum(axis=1)
    after_tp = (after_boot == OUTCOME_CODES["TP"]).sum(axis=1)
    original_acc = ((original_boot == OUTCOME_CODES["TP"]) | (original_boot == OUTCOME_CODES["TN"])).mean(axis=1)
    after_acc = ((after_boot == OUTCOME_CODES["TP"]) | (after_boot == OUTCOME_CODES["TN"])).mean(axis=1)
    trigger_n = trigger_boot.sum(axis=1)
    fp_reduced = original_fp - after_fp
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "fp_reduction": fp_reduced / original_fp,
            "tp_preserved": after_tp / original_tp,
            "accuracy_delta": after_acc - original_acc,
            "fp_reduction_per_trigger": fp_reduced / trigger_n,
        }


def _counts(codes: np.ndarray) -> dict[str, int]:
    return {
        name: int((codes == code).sum())
        for name, code in OUTCOME_CODES.items()
    }


def _test_ids(
    predictions: dict[str, dict[str, Any]],
    test_subset: str,
    split_dir: str | Path | None,
) -> list[str]:
    if split_dir:
        split_name = {"train": "train", "calibration": "val", "test": "test"}.get(test_subset)
        if split_name:
            path = Path(split_dir) / f"pope_{split_name}_ids.json"
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return [str(sample_id) for sample_id in payload.get("sample_ids", [])]
    return [
        sample_id
        for sample_id, row in predictions.items()
        if str(row.get("subset", "")) == test_subset
    ]


def _method_name(score: str) -> str:
    if score == "pls32_probe":
        return "PLS-gated VCD/ICD"
    if score == "tail_257_1024_probe":
        return "Tail-gated VCD/ICD"
    if score == "tail_257_1024_energy":
        return "Tail-energy-gated VCD/ICD"
    if score == "full_probe":
        return "FullD-gated VCD/ICD"
    if score == "top_4_probe":
        return "Top-4-gated VCD/ICD"
    if score == "top_64_probe":
        return "Top-64-gated VCD/ICD"
    if score == "random64_probe":
        return "Random-subspace-gated VCD/ICD"
    if score == "margin_probe":
        return "Margin-gated VCD/ICD"
    if score.startswith("margin_plus_"):
        return "Margin+Geometry-gated VCD/ICD"
    if score == "low_margin_probe":
        return "Low-margin-gated VCD/ICD"
    if score.startswith("low_margin_plus_"):
        return "Low-margin+Geometry-gated VCD/ICD"
    return f"{score}-gated VCD/ICD"


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> Path:
    df = pd.DataFrame(rows)
    keep_metrics = ["fp_reduction", "tp_preserved", "accuracy_delta", "fp_reduction_per_trigger"]
    lines = [
        f"# Stage T Bootstrap CI: {df['operator'].iloc[0] if not df.empty else ''}",
        "",
        "| Method | Score | Target | Metric | Point | 95% CI |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    view = df[df["metric"].isin(keep_metrics)].copy()
    for row in view.sort_values(["target_trigger_rate_predicted_yes", "method", "metric"]).itertuples(index=False):
        lines.append(
            f"| {row.method} | `{row.score}` | {_fmt(row.target_trigger_rate_predicted_yes)} | "
            f"{row.metric} | {_fmt(row.point)} | [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _code(outcome: Any) -> int:
    return OUTCOME_CODES.get(str(outcome), OUTCOME_CODES["unknown"])


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _nanquantile(values: np.ndarray, q: float) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return math.nan
    return float(np.quantile(valid, q))


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
