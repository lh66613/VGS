#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import load_condition_hidden_layer, load_hidden_layer, read_jsonl
from vgs.stage_l import (
    _condition_logistic_basis,
    _fp_target_labels,
    _indices_for_ids,
    _load_split_ids,
    _orthonormal_columns,
    _pls_basis,
    _supervised_vector_plus_pca,
    _svd_basis,
)


METHOD_BUILDERS = {
    "plain_svd": "plain_svd",
    "fisher_fp_tn": "fisher_fp_tn",
    "pls_fp_tn": "pls_fp_tn",
    "matched_vs_adversarial_logistic": "matched_vs_adversarial_logistic",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare low-rank representation-editing directions from completed geometry artifacts."
    )
    parser.add_argument("--layers", nargs="*", default=["20", "24", "32"])
    parser.add_argument("--k-grid", nargs="*", default=["8", "16", "32", "64"])
    parser.add_argument("--methods", nargs="*", default=list(METHOD_BUILDERS))
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--stage-l-probe", default="outputs/stage_l_evidence_subspace/evidence_subspace_probe.csv")
    parser.add_argument("--output-dir", default="outputs/representation_editing_prep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layers = _parse_ints(args.layers)
    k_grid = _parse_ints(args.k_grid)
    max_k = max(k_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(args.predictions)
    labels_by_id = _fp_target_labels(predictions)
    split_ids = _load_split_ids(args.split_dir)
    probe_lookup = _load_probe_lookup(args.stage_l_probe)

    bank: dict[str, Any] = {
        "metadata": {
            "predictions": args.predictions,
            "hidden_states_dir": args.hidden_states_dir,
            "condition_hidden_dir": args.condition_hidden_dir,
            "split_dir": args.split_dir,
            "stage_l_probe": args.stage_l_probe,
            "layers": layers,
            "k_grid": k_grid,
            "methods": args.methods,
            "split": "train",
            "note": "Directions are CPU-prepared candidates for GPU-side activation editing; they are not LoRA weights.",
        },
        "layers": {},
    }
    summary_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []

    for layer in layers:
        hidden = load_hidden_layer(args.hidden_states_dir, layer)
        sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
        z_img = hidden["z_img"].float().numpy()
        z_blind = hidden["z_blind"].float().numpy()
        diff = z_blind - z_img

        train_label_idx = [
            idx
            for idx, sample_id in enumerate(sample_ids)
            if sample_id in labels_by_id and sample_id in split_ids["train"]
        ]
        y_train = np.array([labels_by_id[sample_ids[idx]] for idx in train_label_idx], dtype=np.int64)
        diff_train = diff[train_label_idx]
        tn_train = diff_train[y_train == 0]
        fp_train = diff_train[y_train == 1]
        if len(tn_train) == 0 or len(fp_train) == 0:
            raise ValueError(f"Layer {layer} needs both train TN and train FP samples.")

        condition_payload = load_condition_hidden_layer(args.condition_hidden_dir, layer)
        condition_sample_ids = [str(sample_id) for sample_id in condition_payload["sample_ids"]]
        condition_train_idx = _indices_for_ids(condition_sample_ids, split_ids["train"])
        if len(condition_train_idx) == 0:
            condition_train_idx = np.arange(len(condition_sample_ids))
        conditions = {
            condition: tensor.float().numpy()
            for condition, tensor in condition_payload["conditions"].items()
        }
        blind = conditions["blind"]
        matched = blind - conditions["matched"]
        adversarial = blind - conditions["adversarial_mismatch"]

        bases = _build_bases(args.methods, diff_train, y_train, matched[condition_train_idx], adversarial[condition_train_idx], max_k, args.seed + layer)

        global_tn = tn_train.mean(axis=0)
        global_fp = fp_train.mean(axis=0)
        global_tn_norm = _safe_normalize(global_tn)
        global_delta = global_tn - global_fp
        global_delta_norm = _safe_normalize(global_delta)
        layer_payload: dict[str, Any] = {
            "sample_counts": {
                "train_fp": int(len(fp_train)),
                "train_tn": int(len(tn_train)),
            },
            "global_tn_mean_correction": torch.from_numpy(global_tn_norm.astype(np.float32)),
            "global_tn_minus_fp_correction": torch.from_numpy(global_delta_norm.astype(np.float32)),
            "directions": {},
            "bases": {},
        }
        _append_direction_row(
            summary_rows,
            layer,
            "global",
            "global_tn_mean_correction",
            "full",
            global_tn_norm,
            global_tn_norm,
            global_delta_norm,
            diff_train,
            y_train,
            None,
        )
        _append_direction_row(
            summary_rows,
            layer,
            "global",
            "global_tn_minus_fp_correction",
            "full",
            global_delta_norm,
            global_tn_norm,
            global_delta_norm,
            diff_train,
            y_train,
            None,
        )

        for method, basis in bases.items():
            layer_payload["bases"][method] = torch.from_numpy(basis.astype(np.float32))
            for k in k_grid:
                k_eff = min(k, basis.shape[1])
                subspace = basis[:, :k_eff]
                tn_coords = tn_train @ subspace
                fp_coords = fp_train @ subspace
                candidates = {
                    "projected_tn_mean": subspace @ tn_coords.mean(axis=0),
                    "projected_tn_minus_fp": subspace @ (tn_coords.mean(axis=0) - fp_coords.mean(axis=0)),
                }
                for direction_type, vector in candidates.items():
                    direction_name = f"{method}_k{k_eff}_{direction_type}"
                    normalized = _safe_normalize(vector)
                    layer_payload["directions"][direction_name] = torch.from_numpy(normalized.astype(np.float32))
                    _append_direction_row(
                        summary_rows,
                        layer,
                        method,
                        direction_type,
                        str(k_eff),
                        normalized,
                        global_tn_norm,
                        global_delta_norm,
                        diff_train,
                        y_train,
                        probe_lookup.get((layer, method, k_eff)),
                    )
                    for alpha in [1.0, 2.0, 4.0, 8.0]:
                        plan_rows.append(
                            {
                                "layer": layer,
                                "method": method,
                                "k": k_eff,
                                "direction_name": direction_name,
                                "direction_type": direction_type,
                                "alpha": alpha,
                                "target_outcomes": "FP|TN|TP",
                                "recommended_gate": "margin_and_fp_risk",
                                "status": "prepared_not_run",
                            }
                        )
        bank["layers"][layer] = layer_payload

    bank_path = output_dir / "editing_direction_bank.pt"
    torch.save(bank, bank_path)
    summary_path = output_dir / "direction_summary.csv"
    plan_path = output_dir / "candidate_eval_plan.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)

    note_path = Path("notes/representation_editing_prep.md")
    note_path.write_text(_render_note(summary_rows, str(bank_path), str(plan_path)), encoding="utf-8")
    payload = {
        "bank_path": str(bank_path),
        "direction_summary_path": str(summary_path),
        "candidate_eval_plan_path": str(plan_path),
        "note_path": str(note_path),
        "num_direction_rows": len(summary_rows),
        "num_eval_plan_rows": len(plan_rows),
    }
    (output_dir / "build_representation_editing_prep_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _build_bases(
    methods: list[str],
    diff_train: np.ndarray,
    y_train: np.ndarray,
    matched_train: np.ndarray,
    adversarial_train: np.ndarray,
    max_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    bases: dict[str, np.ndarray] = {}
    for method in methods:
        if method == "plain_svd":
            basis = _svd_basis(diff_train, max_k, seed)
        elif method == "fisher_fp_tn":
            basis = _supervised_vector_plus_pca(diff_train, y_train, max_k, "fisher", seed)
        elif method == "pls_fp_tn":
            basis = _pls_basis(diff_train, y_train, max_k)
        elif method == "matched_vs_adversarial_logistic":
            basis = _condition_logistic_basis(matched_train, adversarial_train, max_k, seed)
        else:
            raise ValueError(f"Unsupported method for representation-editing prep: {method}")
        bases[method] = _orthonormal_columns(basis)[:, :max_k]
    return bases


def _append_direction_row(
    rows: list[dict[str, Any]],
    layer: int,
    method: str,
    direction_type: str,
    k: str,
    vector: np.ndarray,
    global_tn: np.ndarray,
    global_delta: np.ndarray,
    diff_train: np.ndarray,
    y_train: np.ndarray,
    source_probe_auroc: float | None,
) -> None:
    fp_scores = diff_train[y_train == 1] @ vector
    tn_scores = diff_train[y_train == 0] @ vector
    rows.append(
        {
            "layer": layer,
            "method": method,
            "direction_type": direction_type,
            "k": k,
            "norm": float(np.linalg.norm(vector)),
            "cosine_to_global_tn": _cosine(vector, global_tn),
            "cosine_to_global_tn_minus_fp": _cosine(vector, global_delta),
            "train_fp_projection_mean": float(fp_scores.mean()),
            "train_tn_projection_mean": float(tn_scores.mean()),
            "train_tn_minus_fp_projection": float(tn_scores.mean() - fp_scores.mean()),
            "source_probe_auroc": source_probe_auroc,
        }
    )


def _render_note(summary_rows: list[dict[str, Any]], bank_path: str, plan_path: str) -> str:
    df = pd.DataFrame(summary_rows)
    best = df.sort_values(["source_probe_auroc", "train_tn_minus_fp_projection"], ascending=False, na_position="last").head(12)
    lines = [
        "# Representation Editing Prep",
        "",
        "This is a CPU-prepared direction bank for future GPU-side activation-editing experiments. It is not a trained LoRA adapter.",
        "",
        "Artifacts:",
        "",
        f"- Direction bank: `{bank_path}`",
        f"- Candidate eval plan: `{plan_path}`",
        "- Direction summary: `outputs/representation_editing_prep/direction_summary.csv`",
        "",
        "Recommended first GPU test:",
        "",
        "- Use L24/L32 `pls_fp_tn` or `fisher_fp_tn` projected directions.",
        "- Start with `projected_tn_minus_fp` at alpha 2/4/8.",
        "- Gate with `margin_and_fp_risk` and evaluate FP rescue plus TN/TP damage.",
        "",
        "Top candidate rows:",
        "",
    ]
    for _, row in best.iterrows():
        auroc = row["source_probe_auroc"]
        auroc_text = "" if pd.isna(auroc) else f", source AUROC `{auroc:.3f}`"
        lines.append(
            f"- L{int(row['layer'])} `{row['method']}` `{row['direction_type']}` k={row['k']}: "
            f"TN-FP projection `{row['train_tn_minus_fp_projection']:.3f}`{auroc_text}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The bank turns Stage L subspaces into concrete normalized edit directions.",
            "- It is suitable for a next-step activation-editing pilot.",
            "- It does not by itself establish a mitigation method.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_probe_lookup(path: str | Path) -> dict[tuple[int, str, int], float]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(int(row["layer"]), str(row["method"]), int(row["effective_k"]))] = float(row["auroc"])
    return lookup


def _parse_ints(values: list[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item) for item in value.split(",") if item)
    return parsed


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float64, copy=True)
    return vector.astype(np.float64, copy=True) / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


if __name__ == "__main__":
    main()
