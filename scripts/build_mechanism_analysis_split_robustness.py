#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from analyze_mechanism_mitigation_stage2 import analyze_stage2
from mechanism_analysis_common import add_metric_rates, fieldnames, markdown_table, require_present_subspaces


DEFAULT_PAIRS = ["random:popular", "random:adversarial", "popular:adversarial", "adversarial:random"]
DEFAULT_CANDIDATES = [f"band{start}_{start + 11}" for start in range(1, 54, 4)]
DEFAULT_ALPHAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subset-transfer robustness for fixed and selected bands.")
    parser.add_argument(
        "--operator-geometry",
        default="outputs/mechanism_mitigation/mechanism_analysis/operator_geometry_7b_icd/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--fixed-subspace", default="band5_16")
    parser.add_argument("--candidate-subspaces", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--min-tp-preserved", type=float, default=0.95)
    parser.add_argument("--min-calibration-accuracy-delta", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/split_robustness_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_split_robustness(
        operator_geometry_path=Path(args.operator_geometry),
        predictions_path=Path(args.predictions),
        margin_scores_path=Path(args.margin_scores),
        pairs=args.pairs,
        fixed_subspace=args.fixed_subspace,
        candidate_subspaces=args.candidate_subspaces,
        alphas=args.alphas,
        min_tp_preserved=args.min_tp_preserved,
        min_calibration_accuracy_delta=args.min_calibration_accuracy_delta,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_split_robustness_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_split_robustness", summary_path, "ok")
    print(summary_path)


def build_split_robustness(
    operator_geometry_path: Path,
    predictions_path: Path,
    margin_scores_path: Path,
    pairs: list[str],
    fixed_subspace: str,
    candidate_subspaces: list[str],
    alphas: list[float],
    min_tp_preserved: float,
    min_calibration_accuracy_delta: float,
    output_dir: Path,
) -> dict[str, Any]:
    geometry = pd.read_csv(operator_geometry_path)
    subspaces = list(dict.fromkeys([fixed_subspace, *candidate_subspaces]))
    present = require_present_subspaces(geometry, subspaces, operator_geometry_path)
    usable_candidates = [name for name in candidate_subspaces if name in present]
    if fixed_subspace not in present:
        raise ValueError(f"Fixed subspace {fixed_subspace} is not present in {operator_geometry_path}")

    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage2_dirs: list[str] = []
    for pair in pairs:
        calibration_subset, test_subset = _parse_pair(pair)
        pair_dir = output_dir / f"{calibration_subset}_to_{test_subset}" / "stage2"
        stage2_dirs.append(str(pair_dir))
        analyze_stage2(
            operator_geometry_path=operator_geometry_path,
            predictions_path=predictions_path,
            margin_scores_path=margin_scores_path,
            subspaces=present,
            alphas=alphas,
            split_policy="subset_transfer",
            split_dir=None,
            calibration_subset=calibration_subset,
            test_subset=test_subset,
            min_tp_preserved=min_tp_preserved,
            output_dir=pair_dir,
        )
        selected = pd.read_csv(pair_dir / "subspace_vcd_results.csv")
        samples = pd.read_csv(pair_dir / "sample_predictions.csv")
        selected = add_metric_rates(selected, samples, split=test_subset)
        selected["calibration_subset"] = calibration_subset
        selected["test_subset"] = test_subset
        selected["pair"] = f"{calibration_subset}->{test_subset}"
        candidate_view = selected[selected["subspace"].isin(usable_candidates)].copy()
        candidate_rows.extend(candidate_view.to_dict(orient="records"))

        fixed = selected[selected["subspace"].astype(str) == fixed_subspace]
        if not fixed.empty:
            row = fixed.iloc[0].to_dict()
            row["setting"] = "fixed_band5_16"
            summary_rows.append(row)

        best = _best_calibrated_candidate(
            candidate_view,
            min_tp_preserved=min_tp_preserved,
            min_calibration_accuracy_delta=min_calibration_accuracy_delta,
        )
        if best is not None:
            best["setting"] = "calibration_selected_window"
            summary_rows.append(best)

    summary_path = write_csv(output_dir / "split_robustness_summary.csv", summary_rows, fieldnames(summary_rows))
    candidate_path = write_csv(output_dir / "split_robustness_candidates.csv", candidate_rows, fieldnames(candidate_rows))
    report_path = _write_report(output_dir / "split_robustness_report.md", pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows))
    frozen_report_path = _write_report(
        output_dir / "frozen_split_spectral_selection_report.md",
        pd.DataFrame(summary_rows),
        pd.DataFrame(candidate_rows),
    )
    return {
        "operator_geometry_path": str(operator_geometry_path),
        "present_subspaces": present,
        "pairs": pairs,
        "stage2_dirs": stage2_dirs,
        "split_robustness_summary_path": str(summary_path),
        "split_robustness_candidates_path": str(candidate_path),
        "report_path": str(report_path),
        "frozen_split_spectral_selection_report_path": str(frozen_report_path),
        "num_summary_rows": len(summary_rows),
        "num_candidate_rows": len(candidate_rows),
        "min_calibration_accuracy_delta": min_calibration_accuracy_delta,
    }


def _best_calibrated_candidate(
    candidate_view: pd.DataFrame,
    min_tp_preserved: float,
    min_calibration_accuracy_delta: float,
) -> dict[str, Any] | None:
    if candidate_view.empty:
        return None
    view = candidate_view.copy()
    constrained = view[
        (view["calibration_tp_preserved"].fillna(0) >= min_tp_preserved)
        & (view["calibration_accuracy_delta"].fillna(-999) >= min_calibration_accuracy_delta)
    ].copy()
    if constrained.empty:
        constrained = view[view["calibration_tp_preserved"].fillna(0) >= min_tp_preserved].copy()
    if constrained.empty:
        constrained = view.copy()
    constrained["selection_score"] = constrained["calibration_fp_reduction"].fillna(-1) - (
        1 - constrained["calibration_tp_preserved"].fillna(0)
    )
    constrained = constrained.sort_values(
        ["calibration_fp_reduction", "calibration_tp_preserved", "calibration_accuracy_delta", "selection_score"],
        ascending=[False, False, False, False],
    )
    out = constrained.iloc[0].to_dict()
    out["selection_constraint"] = (
        f"calibration_tp_preserved>={min_tp_preserved} and "
        f"calibration_accuracy_delta>={min_calibration_accuracy_delta}; fallback relaxes accuracy then TP"
    )
    return out


def _parse_pair(pair: str) -> tuple[str, str]:
    if ":" in pair:
        left, right = pair.split(":", 1)
    elif "->" in pair:
        left, right = pair.split("->", 1)
    else:
        raise ValueError(f"Pair must use ':' or '->': {pair}")
    return left.strip(), right.strip()


def _write_report(path: Path, summary: pd.DataFrame, candidates: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pair",
        "setting",
        "subspace",
        "alpha",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
        "calibration_fp_reduction",
        "calibration_tp_preserved",
    ]
    cand_cols = [
        "pair",
        "subspace",
        "alpha",
        "fp_reduction",
        "tp_preserved",
        "accuracy_delta",
        "calibration_fp_reduction",
        "calibration_tp_preserved",
    ]
    lines = [
        "# Frozen Split Spectral Selection",
        "",
        "Rows compare fixed Band5-16 transfer with calibration-selected stride-4 spectral windows.",
        "",
        "## Summary",
        "",
        markdown_table(summary[[col for col in cols if col in summary.columns]] if not summary.empty else summary),
        "",
        "## Candidate Rows",
        "",
        markdown_table(candidates[[col for col in cand_cols if col in candidates.columns]] if not candidates.empty else candidates),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
