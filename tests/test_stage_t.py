import numpy as np
import pandas as pd
import torch

from scripts.analyze_stage_t_vcd_results import analyze_vcd_results
from scripts.build_stage_s_baselines import _stage_t_vcd_mitigation_rows
from scripts.build_stage_t_margin_geometry_ablation import build_margin_geometry_ablation
from scripts.build_stage_t_external_warning import build_external_warning
from scripts.build_stage_t_selective_warning import build_selective_warning_metrics
from vgs.stage_t import (
    _calibrate_threshold,
    _capture_damage_at_rate,
    _projected_outcome,
    _quality_metrics,
)
from vgs.vcd import add_diffusion_noise, contrastive_logits


def test_projected_verification_flip_changes_only_predicted_yes_errors() -> None:
    assert _projected_outcome("FP", True) == "TN"
    assert _projected_outcome("TP", True) == "FN"
    assert _projected_outcome("TN", True) == "TN"
    assert _projected_outcome("FP", False) == "FP"


def test_quality_metrics_tracks_fp_rate_and_f1() -> None:
    metrics = _quality_metrics(["TP", "TP", "TN", "FP", "FN"])
    assert metrics["accuracy"] == 0.6
    assert metrics["fp_rate"] == 0.5
    assert round(metrics["f1"], 6) == round(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)), 6)


def test_calibration_threshold_uses_predicted_yes_only() -> None:
    metadata = [
        {"subset": "cal", "parsed_prediction": "yes", "outcome": "TP"},
        {"subset": "cal", "parsed_prediction": "yes", "outcome": "FP"},
        {"subset": "cal", "parsed_prediction": "no", "outcome": "TN"},
        {"subset": "test", "parsed_prediction": "yes", "outcome": "FP"},
    ]
    threshold, info = _calibrate_threshold(np.array([0.1, 0.9, 1.0, 0.8]), metadata, "cal", 0.5)
    assert threshold == 0.9
    assert info["calibration_predicted_yes_n"] == 2
    assert info["calibration_trigger_n"] == 1


def test_capture_damage_at_rate_sorts_high_risk_first() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    capture, damage, precision = _capture_damage_at_rate(y, scores, 0.5)
    assert capture == 0.5
    assert damage == 0.5
    assert precision == 0.5


def test_contrastive_logits_prefers_visual_specific_token() -> None:
    visual = torch.tensor([0.0, 2.0, 0.1])
    contrast = torch.tensor([0.0, 0.5, 0.1])
    logits = contrastive_logits(visual, contrast, alpha=1.0, beta=0.0)
    assert int(torch.argmax(logits).item()) == 1
    assert torch.isclose(logits[1], torch.tensor(3.5))


def test_contrastive_logits_uses_official_plausibility_cutoff() -> None:
    visual = torch.tensor([0.0, 2.0, -3.0])
    contrast = torch.zeros_like(visual)
    logits = contrastive_logits(visual, contrast, alpha=1.0, beta=0.1)
    assert torch.isfinite(logits[0])
    assert torch.isfinite(logits[1])
    assert torch.isneginf(logits[2])


def test_diffusion_noise_preserves_image_tensor_shape() -> None:
    torch.manual_seed(0)
    image = torch.zeros(1, 3, 4, 4)
    noisy = add_diffusion_noise(image, noise_step=500)
    assert noisy.shape == image.shape
    assert noisy.dtype == image.dtype
    assert not torch.equal(noisy, image)


def test_stage_s_loads_official_vcd_baseline_rows(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "layer": "",
                "operator": "vcd_diffusion",
                "method": "Original",
                "gate_family": "none",
                "score": "",
                "target_trigger_rate_predicted_yes": 0.0,
                "aggregation": "deterministic",
                "triggered_fp_before": 0,
                "fp_reduced_n": 0,
                "fp_reduction": 0.0,
                "tp_preserved": 1.0,
                "trigger_rate_predicted_yes": 0.0,
                "accuracy_before": 0.8,
                "accuracy_after": 0.8,
            },
            {
                "layer": "",
                "operator": "vcd_diffusion",
                "method": "Always VCD/ICD",
                "gate_family": "always",
                "score": "always_predicted_yes",
                "target_trigger_rate_predicted_yes": 1.0,
                "aggregation": "deterministic",
                "triggered_fp_before": 10,
                "fp_reduced_n": 3,
                "fp_reduction": 0.3,
                "tp_preserved": 0.9,
                "trigger_rate_predicted_yes": 1.0,
                "accuracy_before": 0.8,
                "accuracy_after": 0.78,
            },
            {
                "layer": 24,
                "operator": "vcd_diffusion",
                "method": "Low-margin+Geometry-gated VCD/ICD",
                "gate_family": "low_margin_plus_geometry",
                "score": "low_margin_plus_pls32_probe",
                "target_trigger_rate_predicted_yes": 0.2,
                "aggregation": "deterministic",
                "triggered_fp_before": 6,
                "fp_reduced_n": 4,
                "fp_reduction": 0.4,
                "tp_preserved": 0.95,
                "trigger_rate_predicted_yes": 0.2,
                "accuracy_before": 0.8,
                "accuracy_after": 0.82,
            },
        ]
    ).to_csv(tmp_path / "stage_t_vcd_metrics_vcd_diffusion.csv", index=False)
    (tmp_path / "run_stage_t_vcd_eval_vcd_diffusion_summary.json").write_text(
        '{"alpha":1.0,"beta":0.1,"noise_step":500,"decode_strategy":"sample"}',
        encoding="utf-8",
    )

    rows = _stage_t_vcd_mitigation_rows(tmp_path)

    assert any(row["baseline"] == "official VCD baseline (diffusion, always-on)" for row in rows)
    best = [row for row in rows if row["baseline"] == "official VCD + best FP-reduction gate"][0]
    assert best["fp_reduction_or_rescue_rate"] == 0.4
    assert "DAMO-NLP-SG/VCD" in best["notes"]


def test_stage_t_vcd_analysis_scores_geometry_gate(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                '{"sample_id":"s1","subset":"test","label":"no","outcome":"FP","parsed_prediction":"yes"}',
                '{"sample_id":"s2","subset":"test","label":"yes","outcome":"TP","parsed_prediction":"yes"}',
                '{"sample_id":"s3","subset":"test","label":"no","outcome":"TN","parsed_prediction":"no"}',
                '{"sample_id":"s4","subset":"test","label":"yes","outcome":"FN","parsed_prediction":"no"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gates_path = tmp_path / "gates.csv"
    pd.DataFrame(
        [
            {
                "layer": 24,
                "score": "pls32_probe",
                "target_trigger_rate_predicted_yes": 0.5,
                "sample_id": "s1",
            },
            {
                "layer": 24,
                "score": "top_4_probe",
                "target_trigger_rate_predicted_yes": 0.5,
                "sample_id": "s2",
            },
        ]
    ).to_csv(gates_path, index=False)
    vcd_path = tmp_path / "vcd.jsonl"
    vcd_path.write_text(
        "\n".join(
            [
                '{"sample_id":"s1","vcd_parsed_prediction":"no","vcd_outcome":"TN"}',
                '{"sample_id":"s2","vcd_parsed_prediction":"no","vcd_outcome":"FN"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_vcd_results(
        predictions_path=predictions_path,
        gate_assignments_path=gates_path,
        vcd_predictions_path=vcd_path,
        operator="vcd_blur",
        test_subset="test",
        split_dir=None,
        target_rates=[0.5],
        selected_scores=["pls32_probe", "top_4_probe"],
        random_repeats=2,
        seed=1,
        output_dir=tmp_path,
    )

    df = pd.read_csv(result["metrics_path"])
    pls = df[df["method"] == "PLS-gated VCD/ICD"].iloc[0]
    assert pls["fp_reduction"] == 1.0
    assert pls["tp_preserved"] == 1.0
    assert pls["fp_reduction_per_trigger"] == 1.0


def test_margin_geometry_ablation_uses_fixed_predicted_yes_budget(tmp_path) -> None:
    stage_t_dir = tmp_path / "stage_t"
    stage_t_dir.mkdir()
    required_scores = {
        "low_margin_probe": [0.9, 0.1, 0.0, 0.0],
        "full_probe": [0.1, 0.9, 0.0, 0.0],
        "pls32_probe": [0.8, 0.2, 0.0, 0.0],
        "tail_257_1024_probe": [0.7, 0.3, 0.0, 0.0],
        "low_margin_plus_full_probe": [0.9, 0.1, 0.0, 0.0],
        "low_margin_plus_pls32_probe": [0.8, 0.2, 0.0, 0.0],
        "low_margin_plus_tail_257_1024_probe": [0.7, 0.3, 0.0, 0.0],
    }
    pd.DataFrame(
        {
            "layer": [24, 24, 24, 24],
            "sample_id": ["s1", "s2", "s3", "s4"],
            "subset": ["test", "test", "test", "test"],
            "outcome": ["FP", "TP", "TN", "FN"],
            "parsed_prediction": ["yes", "yes", "no", "no"],
            **required_scores,
        }
    ).to_csv(stage_t_dir / "stage_t_scores.csv", index=False)
    (stage_t_dir / "stage_t_vcd_predictions_icd_blind.jsonl").write_text(
        "\n".join(
            [
                '{"sample_id":"s1","vcd_parsed_prediction":"no","vcd_outcome":"TN"}',
                '{"sample_id":"s2","vcd_parsed_prediction":"no","vcd_outcome":"FN"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_margin_geometry_ablation(
        stage_t_dir=stage_t_dir,
        scores_path=None,
        output_dir=tmp_path / "out",
        layer=24,
        test_subset="test",
        target_rates=[0.5],
        operators=["icd_blind"],
        random_repeats=3,
        seed=1,
    )

    df = pd.read_csv(result["csv_path"])
    margin = df[df["gate"] == "Margin-only"].iloc[0]
    full = df[df["gate"] == "Geometry-only full"].iloc[0]
    assert margin["trigger_n"] == 1
    assert margin["fp_recall"] == 1.0
    assert margin["tp_damage"] == 0.0
    assert margin["icd_vcd_fp_reduction"] == 1.0
    assert margin["accuracy_delta"] == 0.25
    assert full["fp_recall"] == 0.0
    assert full["tp_damage"] == 1.0


def test_warning_random_rows_use_group_total_when_matched_gate_has_zero_fp(tmp_path) -> None:
    stage_t_dir = tmp_path / "stage_t"
    stage_t_dir.mkdir()
    gate_rows = [
        {
            "layer": 24,
            "split": "test",
            "score": "pls32_probe",
            "target_trigger_rate_predicted_yes": 0.2,
            "trigger_n": 2,
            "predicted_yes_n": 10,
            "trigger_rate_predicted_yes": 0.2,
            "triggered_fp": 1,
            "triggered_tp": 1,
            "triggered_fp_ratio": 0.5,
            "fp_recall_among_predicted_yes": 0.5,
            "tp_damage": 0.125,
            "original_accuracy": 0.8,
            "original_f1": 0.8,
            "original_fp_rate": 0.2,
            "oracle_flip_accuracy": 0.85,
            "oracle_flip_f1": 0.85,
            "oracle_fp_reduction": 0.5,
            "oracle_tp_preserved": 0.875,
        },
        {
            "layer": 24,
            "split": "test",
            "score": "margin_probe",
            "target_trigger_rate_predicted_yes": 0.2,
            "trigger_n": 2,
            "predicted_yes_n": 10,
            "trigger_rate_predicted_yes": 0.2,
            "triggered_fp": 0,
            "triggered_tp": 2,
            "triggered_fp_ratio": 0.0,
            "fp_recall_among_predicted_yes": 0.0,
            "tp_damage": 0.25,
            "original_accuracy": 0.8,
            "original_f1": 0.8,
            "original_fp_rate": 0.2,
            "oracle_flip_accuracy": 0.7,
            "oracle_flip_f1": 0.7,
            "oracle_fp_reduction": 0.0,
            "oracle_tp_preserved": 0.75,
        },
    ]
    pd.DataFrame(gate_rows).to_csv(stage_t_dir / "stage_t_gate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "layer": 24,
                "matched_score": "margin_probe",
                "split": "test",
                "target_trigger_rate_predicted_yes": 0.2,
                "n_trigger": 2,
                "metric": "fp_recall_among_predicted_yes",
                "mean": 0.5,
                "std": 0.0,
                "p05": 0.5,
                "p95": 0.5,
            },
            {
                "layer": 24,
                "matched_score": "margin_probe",
                "split": "test",
                "target_trigger_rate_predicted_yes": 0.2,
                "n_trigger": 2,
                "metric": "tp_damage",
                "mean": 0.25,
                "std": 0.0,
                "p05": 0.25,
                "p95": 0.25,
            },
            {
                "layer": 24,
                "matched_score": "margin_probe",
                "split": "test",
                "target_trigger_rate_predicted_yes": 0.2,
                "n_trigger": 2,
                "metric": "triggered_fp_ratio",
                "mean": 0.25,
                "std": 0.0,
                "p05": 0.25,
                "p95": 0.25,
            },
        ]
    ).to_csv(stage_t_dir / "stage_t_random_gate_metrics.csv", index=False)

    result = build_selective_warning_metrics(
        stage_t_dir=stage_t_dir,
        target_rates=[0.2],
        selected_scores=["margin_probe"],
        output_dir=tmp_path / "out",
    )

    df = pd.read_csv(result["metrics_path"])
    random_row = df[df["method"] == "Random warning"].iloc[0]
    assert random_row["fp_captured"] == 1.0
    assert random_row["warning_precision"] == 0.25


def test_external_warning_builder_writes_top_rate_assignments(tmp_path) -> None:
    stage_t_dir = tmp_path / "stage_t"
    stage_t_dir.mkdir()
    pd.DataFrame(
        [
            {
                "layer": 24,
                "score": "pls32_probe",
                "target_trigger_rate_predicted_yes": 0.5,
                "threshold": 0.7,
            }
        ]
    ).to_csv(stage_t_dir / "stage_t_gate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "layer": 24,
                "sample_id": "a",
                "subset": "discriminative",
                "dimension": "existence",
                "label": "no",
                "outcome": "FP",
                "parsed_prediction": "yes",
                "question": "Is there a chair?",
                "image": "a.jpg",
                "image_path": "a.jpg",
                "pls32_probe": 0.9,
            },
            {
                "layer": 24,
                "sample_id": "b",
                "subset": "discriminative",
                "dimension": "existence",
                "label": "yes",
                "outcome": "TP",
                "parsed_prediction": "yes",
                "question": "Is there a table?",
                "image": "b.jpg",
                "image_path": "b.jpg",
                "pls32_probe": 0.1,
            },
        ]
    ).to_csv(stage_t_dir / "stage_t_external_scores.csv", index=False)

    result = build_external_warning(
        stage_t_dir=stage_t_dir,
        external_scores_path=None,
        gate_metrics_path=None,
        output_dir=tmp_path / "external",
        target_rates=[0.5],
        selected_scores=["pls32_probe"],
        policies=["external_top_rate"],
        random_repeats=2,
        seed=1,
    )

    assignments = pd.read_csv(result["assignment_paths"]["external_top_rate"])
    metrics = pd.read_csv(result["metrics_path"])
    row = metrics[
        (metrics["selection_policy"] == "external_top_rate")
        & (metrics["score"] == "pls32_probe")
    ].iloc[0]
    assert assignments["sample_id"].tolist() == ["a"]
    assert row["warning_precision"] == 1.0
