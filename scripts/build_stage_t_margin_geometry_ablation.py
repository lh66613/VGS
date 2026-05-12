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

from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json


GATE_SPECS = [
    ("Random", "same_trigger_random", "same_trigger_random", ""),
    ("Margin-only", "low_margin", "margin", "low_margin_probe"),
    ("Geometry-only full", "geometry_full", "geometry", "full_probe"),
    ("Geometry-only PLS", "geometry_pls", "geometry", "pls32_probe"),
    ("Geometry-only tail", "geometry_tail", "geometry", "tail_257_1024_probe"),
    ("Margin + full", "low_margin_plus_full", "margin_plus_geometry", "low_margin_plus_full_probe"),
    ("Margin + PLS", "low_margin_plus_pls", "margin_plus_geometry", "low_margin_plus_pls32_probe"),
    (
        "Margin + tail",
        "low_margin_plus_tail",
        "margin_plus_geometry",
        "low_margin_plus_tail_257_1024_probe",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed-trigger-rate Stage T margin-vs-geometry ablation tables."
    )
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--scores-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--test-subset", default="test")
    parser.add_argument("--target-rates", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument(
        "--operators",
        nargs="*",
        default=["icd_blind"],
        help="VCD/ICD operators to include. Use 'all' to discover every prediction file.",
    )
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_margin_geometry_ablation(
        stage_t_dir=args.stage_t_dir,
        scores_path=args.scores_path,
        output_dir=args.output_dir,
        layer=args.layer,
        test_subset=args.test_subset,
        target_rates=args.target_rates,
        operators=args.operators,
        random_repeats=args.random_repeats,
        seed=args.seed,
    )
    summary_path = write_json(
        Path(result["output_dir"]) / "build_stage_t_margin_geometry_ablation_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "build_stage_t_margin_geometry_ablation", summary_path, "ok")
    print(summary_path)


def build_margin_geometry_ablation(
    stage_t_dir: str | Path,
    scores_path: str | Path | None,
    output_dir: str | Path | None,
    layer: int,
    test_subset: str,
    target_rates: list[float],
    operators: list[str],
    random_repeats: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(stage_t_dir)
    output_root = ensure_dir(output_dir or root)
    scores = pd.read_csv(scores_path or root / "stage_t_scores.csv")
    split = scores[(scores["layer"] == layer) & (scores["subset"] == test_subset)].copy()
    if split.empty:
        raise ValueError(f"No rows found for layer={layer}, subset={test_subset}.")

    predicted_yes = split[
        (split["parsed_prediction"].astype(str).str.lower() == "yes")
        & split["outcome"].astype(str).isin(["FP", "TP"])
    ].copy()
    if predicted_yes.empty:
        raise ValueError(f"No predicted-Yes FP/TP rows found for layer={layer}, subset={test_subset}.")

    missing_scores = sorted(
        {
            score
            for _, _, _, score in GATE_SPECS
            if score and score not in predicted_yes.columns
        }
    )
    if missing_scores:
        raise ValueError(f"Missing required score columns: {missing_scores}")

    operator_predictions = _load_operator_predictions(root, operators)
    if not operator_predictions:
        raise ValueError(f"No VCD/ICD prediction files found in {root}.")

    original_quality = _quality(split["outcome"].astype(str).tolist())
    candidate_ids = predicted_yes["sample_id"].astype(str).tolist()
    candidate_outcomes = dict(
        zip(predicted_yes["sample_id"].astype(str), predicted_yes["outcome"].astype(str))
    )

    rows: list[dict[str, Any]] = []
    for rate in sorted(target_rates):
        n_trigger = max(1, int(math.ceil(float(rate) * len(predicted_yes))))
        n_trigger = min(n_trigger, len(predicted_yes))
        random_sets = _random_trigger_sets(candidate_ids, n_trigger, random_repeats, seed, rate)

        for gate, gate_key, gate_family, score in GATE_SPECS:
            if score:
                triggered_ids = _top_score_ids(predicted_yes, score, n_trigger)
                score_threshold = _score_threshold(predicted_yes, score, triggered_ids)
                for operator, vcd in operator_predictions.items():
                    rows.append(
                        _deterministic_row(
                            split=split,
                            candidate_outcomes=candidate_outcomes,
                            original_quality=original_quality,
                            vcd=vcd,
                            operator=operator,
                            layer=layer,
                            test_subset=test_subset,
                            target_rate=rate,
                            n_trigger=n_trigger,
                            gate=gate,
                            gate_key=gate_key,
                            gate_family=gate_family,
                            score=score,
                            score_threshold=score_threshold,
                            triggered_ids=triggered_ids,
                        )
                    )
                continue

            random_warning = [
                _warning_metrics(triggered, candidate_outcomes)
                for triggered in random_sets
            ]
            for operator, vcd in operator_predictions.items():
                random_actual = [
                    _actual_metrics(split, original_quality, vcd, triggered)
                    for triggered in random_sets
                ]
                rows.append(
                    _random_row(
                        warning_rows=random_warning,
                        actual_rows=random_actual,
                        operator=operator,
                        layer=layer,
                        test_subset=test_subset,
                        target_rate=rate,
                        n_trigger=n_trigger,
                        predicted_yes_n=len(predicted_yes),
                        gate=gate,
                        gate_key=gate_key,
                        gate_family=gate_family,
                        random_repeats=random_repeats,
                    )
                )

    csv_path = write_csv(
        output_root / "stage_t_margin_geometry_fixed_trigger_ablation.csv",
        rows,
        _fieldnames(rows),
    )
    md_path = _write_markdown(
        output_root / "stage_t_margin_geometry_fixed_trigger_ablation.md",
        rows,
        original_quality,
        len(predicted_yes),
    )
    return {
        "stage_t_dir": str(root),
        "output_dir": str(output_root),
        "scores_path": str(scores_path or root / "stage_t_scores.csv"),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "layer": layer,
        "test_subset": test_subset,
        "target_rates": sorted(target_rates),
        "operators": sorted(operator_predictions),
        "random_repeats": random_repeats,
        "predicted_yes_n": len(predicted_yes),
        "test_n": len(split),
        "original_accuracy": original_quality["accuracy"],
        "original_fp": original_quality["FP"],
        "original_tp": original_quality["TP"],
        "num_rows": len(rows),
    }


def _deterministic_row(
    split: pd.DataFrame,
    candidate_outcomes: dict[str, str],
    original_quality: dict[str, Any],
    vcd: dict[str, str],
    operator: str,
    layer: int,
    test_subset: str,
    target_rate: float,
    n_trigger: int,
    gate: str,
    gate_key: str,
    gate_family: str,
    score: str,
    score_threshold: float,
    triggered_ids: set[str],
) -> dict[str, Any]:
    warning = _warning_metrics(triggered_ids, candidate_outcomes)
    actual = _actual_metrics(split, original_quality, vcd, triggered_ids)
    return {
        "layer": layer,
        "split": test_subset,
        "operator": operator,
        "target_trigger_rate_predicted_yes": target_rate,
        "gate": gate,
        "gate_key": gate_key,
        "gate_family": gate_family,
        "score": score,
        "aggregation": "deterministic",
        "random_repeats": "",
        "score_threshold": score_threshold,
        "trigger_n": n_trigger,
        "predicted_yes_n": len(candidate_outcomes),
        "trigger_rate_predicted_yes": n_trigger / len(candidate_outcomes),
        **warning,
        **actual,
        **_empty_std_columns(),
    }


def _random_row(
    warning_rows: list[dict[str, float]],
    actual_rows: list[dict[str, float]],
    operator: str,
    layer: int,
    test_subset: str,
    target_rate: float,
    n_trigger: int,
    predicted_yes_n: int,
    gate: str,
    gate_key: str,
    gate_family: str,
    random_repeats: int,
) -> dict[str, Any]:
    mean_warning = _mean_metrics(warning_rows)
    mean_actual = _mean_metrics(actual_rows)
    return {
        "layer": layer,
        "split": test_subset,
        "operator": operator,
        "target_trigger_rate_predicted_yes": target_rate,
        "gate": gate,
        "gate_key": gate_key,
        "gate_family": gate_family,
        "score": "same_trigger_random",
        "aggregation": "mean_random",
        "random_repeats": random_repeats,
        "score_threshold": "",
        "trigger_n": n_trigger,
        "predicted_yes_n": predicted_yes_n,
        "trigger_rate_predicted_yes": n_trigger / predicted_yes_n,
        **mean_warning,
        **mean_actual,
        **_std_columns(warning_rows, actual_rows),
    }


def _top_score_ids(df: pd.DataFrame, score: str, n_trigger: int) -> set[str]:
    ranked = df.copy()
    ranked["_score_for_rank"] = pd.to_numeric(ranked[score], errors="coerce").fillna(-np.inf)
    ranked["sample_id"] = ranked["sample_id"].astype(str)
    ranked = ranked.sort_values(["_score_for_rank", "sample_id"], ascending=[False, True])
    return set(ranked.head(n_trigger)["sample_id"].astype(str).tolist())


def _score_threshold(df: pd.DataFrame, score: str, triggered_ids: set[str]) -> float:
    selected = df[df["sample_id"].astype(str).isin(triggered_ids)]
    if selected.empty:
        return math.nan
    return float(pd.to_numeric(selected[score], errors="coerce").min())


def _random_trigger_sets(
    candidate_ids: list[str],
    n_trigger: int,
    repeats: int,
    seed: int,
    rate: float,
) -> list[set[str]]:
    rng = np.random.default_rng(seed + _rate_key(rate))
    ids = np.array(candidate_ids, dtype=object)
    return [set(rng.choice(ids, size=n_trigger, replace=False).tolist()) for _ in range(repeats)]


def _warning_metrics(triggered_ids: set[str], candidate_outcomes: dict[str, str]) -> dict[str, float]:
    triggered = [candidate_outcomes[sample_id] for sample_id in triggered_ids]
    counts = _counts(triggered)
    all_counts = _counts(list(candidate_outcomes.values()))
    trigger_n = len(triggered_ids)
    return {
        "triggered_fp": counts["FP"],
        "triggered_tp": counts["TP"],
        "fp_recall": counts["FP"] / all_counts["FP"] if all_counts["FP"] else math.nan,
        "tp_damage": counts["TP"] / all_counts["TP"] if all_counts["TP"] else math.nan,
        "warning_precision": counts["FP"] / trigger_n if trigger_n else math.nan,
    }


def _actual_metrics(
    split: pd.DataFrame,
    original_quality: dict[str, Any],
    vcd: dict[str, str],
    triggered_ids: set[str],
) -> dict[str, float]:
    final_outcomes: list[str] = []
    missing_vcd_n = 0
    for row in split.itertuples(index=False):
        sample_id = str(getattr(row, "sample_id"))
        original = str(getattr(row, "outcome"))
        if sample_id in triggered_ids:
            if sample_id in vcd:
                final_outcomes.append(vcd[sample_id])
            else:
                missing_vcd_n += 1
                final_outcomes.append(original)
        else:
            final_outcomes.append(original)
    after = _quality(final_outcomes)
    fp_reduced_n = original_quality["FP"] - after["FP"]
    return {
        "icd_vcd_fp_reduced_n": fp_reduced_n,
        "icd_vcd_fp_reduction": fp_reduced_n / original_quality["FP"]
        if original_quality["FP"]
        else math.nan,
        "tp_preserved": after["TP"] / original_quality["TP"] if original_quality["TP"] else math.nan,
        "accuracy_before": original_quality["accuracy"],
        "accuracy_after": after["accuracy"],
        "accuracy_delta": after["accuracy"] - original_quality["accuracy"],
        "f1_before": original_quality["f1"],
        "f1_after": after["f1"],
        "f1_delta": after["f1"] - original_quality["f1"],
        "fp_rate_before": original_quality["fp_rate"],
        "fp_rate_after": after["fp_rate"],
        "missing_vcd_n": missing_vcd_n,
    }


def _quality(outcomes: list[str]) -> dict[str, Any]:
    counts = _counts(outcomes)
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    unknown = counts["unknown"]
    n = tp + tn + fp + fn + unknown
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    return {
        **counts,
        "accuracy": (tp + tn) / n if n else math.nan,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else math.nan,
        "fp_rate": fp / (fp + tn) if fp + tn else math.nan,
    }


def _counts(outcomes: list[str]) -> dict[str, int]:
    names = ["TP", "TN", "FP", "FN", "unknown"]
    return {name: int(sum(outcome == name for outcome in outcomes)) for name in names}


def _load_operator_predictions(root: Path, operators: list[str]) -> dict[str, dict[str, str]]:
    if len(operators) == 1 and operators[0] == "all":
        paths = sorted(root.glob("stage_t_vcd_predictions_*.jsonl"))
    else:
        paths = [root / f"stage_t_vcd_predictions_{operator}.jsonl" for operator in operators]
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        operator = path.name.removeprefix("stage_t_vcd_predictions_").removesuffix(".jsonl")
        out[operator] = _read_vcd_prediction_map(path)
    return out


def _read_vcd_prediction_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("vcd_parsed_prediction") in {"yes", "no"}:
                out[str(row["sample_id"])] = str(row["vcd_outcome"])
    return out


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


def _std_columns(
    warning_rows: list[dict[str, float]],
    actual_rows: list[dict[str, float]],
) -> dict[str, float]:
    rows = [{**warning, **actual} for warning, actual in zip(warning_rows, actual_rows)]
    tracked = [
        "fp_recall",
        "tp_damage",
        "warning_precision",
        "icd_vcd_fp_reduction",
        "tp_preserved",
        "accuracy_delta",
    ]
    return {
        f"std_{key}": float(np.nanstd([float(row[key]) for row in rows]))
        for key in tracked
    }


def _empty_std_columns() -> dict[str, str]:
    return {
        "std_fp_recall": "",
        "std_tp_damage": "",
        "std_warning_precision": "",
        "std_icd_vcd_fp_reduction": "",
        "std_tp_preserved": "",
        "std_accuracy_delta": "",
    }


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    original_quality: dict[str, Any],
    predicted_yes_n: int,
) -> Path:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage T Margin-vs-Geometry Fixed-Trigger Ablation",
        "",
        "Held-out fixed split; gates are evaluated on the predicted-Yes FP/TP subset with exact top-rate trigger budgets.",
        "The margin rows use the low-margin risk direction (`low_margin_probe`) because high Yes-No margin is not a hallucination-risk signal inside predicted-Yes samples.",
        "",
        f"Original accuracy: `{original_quality['accuracy']:.3f}`. Predicted-Yes pool: `{predicted_yes_n}`.",
        "",
    ]
    order = {gate: idx for idx, (gate, _, _, _) in enumerate(GATE_SPECS)}
    for operator in sorted(df["operator"].unique()):
        op_df = df[df["operator"] == operator].copy()
        lines.extend([f"## Operator `{operator}`", ""])
        for rate in sorted(op_df["target_trigger_rate_predicted_yes"].unique()):
            subset = op_df[op_df["target_trigger_rate_predicted_yes"] == rate].copy()
            subset["_order"] = subset["gate"].map(order)
            subset = subset.sort_values("_order")
            lines.extend(
                [
                    f"### Target Trigger Rate {float(rate):.2f}",
                    "",
                    "| Gate | FP Recall | TP Damage | Warning Precision | ICD/VCD FP Reduction | TP Preserved | Accuracy Delta |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in subset.itertuples(index=False):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.gate),
                            _fmt(row.fp_recall),
                            _fmt(row.tp_damage),
                            _fmt(row.warning_precision),
                            _fmt(row.icd_vcd_fp_reduction),
                            _fmt(row.tp_preserved),
                            _fmt(row.accuracy_delta, signed=True),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _rate_key(rate: Any) -> int:
    return int(round(float(rate) * 10000))


def _fmt(value: Any, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    if signed:
        return f"{number:+.3f}"
    return f"{number:.3f}"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
