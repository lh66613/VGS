#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage T operator upper-bound gap table.")
    parser.add_argument("--stage-t-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    output_dir = args.output_dir or args.stage_t_dir
    result = build_operator_upper_bound(args.stage_t_dir, output_dir)
    summary_path = write_json(
        Path(output_dir) / "build_stage_t_operator_upper_bound_summary.json",
        result,
    )
    append_experiment_log(args.log_path, "build_stage_t_operator_upper_bound", summary_path, "ok")
    print(summary_path)


def build_operator_upper_bound(stage_t_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    stage_root = Path(stage_t_dir)
    gate_path = stage_root / "stage_t_gate_metrics.csv"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing gate metrics: {gate_path}")
    gate = pd.read_csv(gate_path)
    actual_paths = sorted(stage_root.glob("stage_t_actual_verification_metrics*.csv"))
    if not actual_paths:
        raise FileNotFoundError(f"No actual verification metrics found in {stage_root}")

    frames = []
    for path in actual_paths:
        df = pd.read_csv(path)
        if "prompt_variant" not in df.columns:
            df["prompt_variant"] = _variant_from_path(path)
        frames.append(df)
    actual = pd.concat(frames, ignore_index=True)
    merged = actual.merge(
        gate[
            [
                "layer",
                "score",
                "target_trigger_rate_predicted_yes",
                "trigger_n",
                "trigger_rate_predicted_yes",
                "triggered_fp_ratio",
                "fp_recall_among_predicted_yes",
                "tp_damage",
                "oracle_fp_reduction",
                "oracle_tp_preserved",
                "oracle_flip_accuracy",
                "oracle_flip_f1",
            ]
        ],
        on=["layer", "score", "target_trigger_rate_predicted_yes"],
        how="left",
    )
    merged["operator_realization_ratio"] = merged["actual_fp_reduction"] / merged[
        "oracle_fp_reduction"
    ].replace(0, pd.NA)
    keep_cols = [
        "prompt_variant",
        "layer",
        "score",
        "target_trigger_rate_predicted_yes",
        "assigned_trigger_n",
        "verified_trigger_n",
        "triggered_fp_ratio",
        "fp_recall_among_predicted_yes",
        "tp_damage",
        "oracle_fp_reduction",
        "actual_fp_reduction",
        "operator_realization_ratio",
        "oracle_tp_preserved",
        "actual_tp_preserved",
        "oracle_flip_accuracy",
        "accuracy_after",
        "oracle_flip_f1",
        "f1_after",
    ]
    rows = merged[keep_cols].sort_values(
        ["prompt_variant", "target_trigger_rate_predicted_yes", "actual_fp_reduction"],
        ascending=[True, True, False],
    )
    table_path = write_csv(
        Path(output_dir) / "stage_t_operator_upper_bound_gap.csv",
        rows.to_dict(orient="records"),
        keep_cols,
    )
    return {
        "stage_t_dir": str(stage_t_dir),
        "gate_metrics_path": str(gate_path),
        "actual_metric_paths": [str(path) for path in actual_paths],
        "operator_upper_bound_gap_path": str(table_path),
        "num_rows": int(len(rows)),
        "prompt_variants": sorted(str(item) for item in rows["prompt_variant"].dropna().unique()),
    }


def _variant_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "stage_t_actual_verification_metrics"
    suffix = stem[len(prefix) :]
    if not suffix:
        return "legacy"
    return suffix.lstrip("_")


if __name__ == "__main__":
    main()
