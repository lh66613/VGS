import pandas as pd

from scripts.analyze_stage_t_geometry_complementarity import build_geometry_complementarity


def test_geometry_complementarity_outputs_margin_bins_and_residuals(tmp_path) -> None:
    stage_t_dir = tmp_path / "stage_t"
    stage_t_dir.mkdir()
    rows = [
        _row("c1", "calibration", "FP", 0.1, 0.90),
        _row("c2", "calibration", "TP", 0.2, 0.10),
        _row("c3", "calibration", "FP", 1.0, 0.80),
        _row("c4", "calibration", "TP", 1.2, 0.20),
        _row("t1", "test", "FP", 0.2, 0.95),
        _row("t2", "test", "TP", 0.3, 0.05),
        _row("t3", "test", "FP", 1.0, 0.85),
        _row("t4", "test", "TP", 1.1, 0.15),
        _row("t5", "test", "TP", 3.5, 0.10),
    ]
    pd.DataFrame(rows).to_csv(stage_t_dir / "stage_t_scores.csv", index=False)

    result = build_geometry_complementarity(
        stage_t_dir=stage_t_dir,
        scores_path=None,
        output_dir=tmp_path / "out",
        layer=24,
        split="test",
        calibration_split="calibration",
        scores=["geo_probe"],
        primary_score="geo_probe",
        margin_score="low_margin_probe",
        target_rates=[0.5],
        bin_policy="fixed",
        margin_bin_edges=[0.5, 1.5, 3.0],
        max_pairs=2,
        pair_max_margin_delta=0.2,
    )

    bins = pd.read_csv(result["margin_bin_path"])
    very_low = bins[(bins["margin_bin"] == "very_low") & (bins["score"] == "geo_probe")].iloc[0]
    assert very_low["auroc_fp_vs_tp"] == 1.0
    assert very_low["warning_precision"] == 1.0

    residual = pd.read_csv(result["residual_prediction_path"]).iloc[0]
    assert residual["margin_missed_fp_n"] >= 1
    assert residual["additional_fp_caught"] >= 0

    correlations = pd.read_csv(result["correlation_path"])
    assert set(correlations["reference"]) == {"yes_minus_no_logit", "binary_entropy"}

    pairs = pd.read_csv(result["same_margin_pair_path"])
    assert pairs.iloc[0]["score_delta_fp_minus_tp"] > 0


def _row(sample_id: str, subset: str, outcome: str, margin: float, geo: float) -> dict[str, object]:
    return {
        "layer": 24,
        "sample_id": sample_id,
        "subset": subset,
        "source_subset": subset,
        "dimension": "",
        "label": "no" if outcome == "FP" else "yes",
        "outcome": outcome,
        "parsed_prediction": "yes",
        "question": f"Question {sample_id}?",
        "image": f"{sample_id}.jpg",
        "image_path": f"{sample_id}.jpg",
        "yes_minus_no_logit": margin,
        "binary_entropy": 1.0 / (1.0 + margin),
        "low_margin_probe": -margin,
        "geo_probe": geo,
    }
