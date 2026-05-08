#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split-locked and label-shuffled probe sanity checks for Stage O.")
    parser.add_argument("--root", default="outputs/stage_o_cross_model")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--aliases", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/stage_o_cross_model/audit")
    parser.add_argument("--notes-path", default="notes/stage_o_probe_sanity.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    aliases = args.aliases or sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "audit")
    split_ids = _load_split_ids(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for alias in aliases:
        model_root = root / alias
        pred_path = model_root / "predictions" / "pope_predictions.jsonl"
        hidden_summary = _read_json(model_root / "hidden_states" / "dump_hidden_states_summary.json")
        layers = [int(layer) for layer in hidden_summary.get("layers", [])]
        labels_by_id = _fp_tn_labels(pred_path)
        for layer in layers:
            hidden_path = model_root / "hidden_states" / f"layer_{layer}.pt"
            svd_path = model_root / "svd" / f"svd_layer_{layer}.pt"
            if not hidden_path.exists() or not svd_path.exists():
                continue
            hidden = torch.load(hidden_path, map_location="cpu")
            svd = torch.load(svd_path, map_location="cpu")
            sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
            z_img = hidden["z_img"].float().numpy()
            z_blind = hidden["z_blind"].float().numpy()
            diff = z_blind - z_img
            basis = svd["Vh"].float().numpy().T
            features = {
                "raw_img": z_img,
                "raw_blind": z_blind,
                "difference": diff,
                "top4_projected_difference": diff @ basis[:, :4],
                "top32_projected_difference": diff @ basis[:, :32],
            }
            train_idx, test_idx, y_train, y_test = _split_indices(sample_ids, labels_by_id, split_ids)
            for feature_name, matrix in features.items():
                x_train = matrix[train_idx]
                x_test = matrix[test_idx]
                rows.append(
                    {
                        "alias": alias,
                        "layer": layer,
                        "feature": feature_name,
                        "control": "real_labels",
                        **_fit_metrics(x_train, x_test, y_train, y_test, args.seed),
                    }
                )
                rng = np.random.default_rng(args.seed + layer)
                shuffled = y_train.copy()
                rng.shuffle(shuffled)
                rows.append(
                    {
                        "alias": alias,
                        "layer": layer,
                        "feature": feature_name,
                        "control": "train_label_shuffled",
                        **_fit_metrics(x_train, x_test, shuffled, y_test, args.seed),
                    }
                )

    path = output_dir / "probe_sanity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    note_path = Path(args.notes_path)
    note_path.write_text(_render_note(pd.DataFrame(rows), path), encoding="utf-8")
    (output_dir / "audit_stage_o_probe_sanity_summary.json").write_text(
        json.dumps({"probe_sanity_path": str(path), "note_path": str(note_path), "num_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )


def _fit_metrics(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> dict[str, Any]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "train_size": len(y_train), "test_size": len(y_test), "num_positive_train": int(y_train.sum()), "num_positive_test": int(y_test.sum())}
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x_train_scaled, y_train)
    probs = clf.predict_proba(x_test_scaled)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, probs)),
        "auprc": float(average_precision_score(y_test, probs)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "num_positive_train": int(y_train.sum()),
        "num_positive_test": int(y_test.sum()),
    }


def _split_indices(sample_ids: list[str], labels_by_id: dict[str, int], split_ids: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = []
    test_idx = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in labels_by_id:
            continue
        if sample_id in split_ids["train"]:
            train_idx.append(idx)
        elif sample_id in split_ids["test"]:
            test_idx.append(idx)
    y_train = np.array([labels_by_id[sample_ids[idx]] for idx in train_idx], dtype=np.int64)
    y_test = np.array([labels_by_id[sample_ids[idx]] for idx in test_idx], dtype=np.int64)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64), y_train, y_test


def _fp_tn_labels(path: Path) -> dict[str, int]:
    labels = {}
    for row in _read_jsonl(path):
        if row.get("outcome") == "FP":
            labels[str(row["sample_id"])] = 1
        elif row.get("outcome") == "TN":
            labels[str(row["sample_id"])] = 0
    return labels


def _load_split_ids(split_dir: str | Path) -> dict[str, set[str]]:
    root = Path(split_dir)
    splits = {}
    for split in ["train", "test"]:
        payload = json.loads((root / f"pope_{split}_ids.json").read_text(encoding="utf-8"))
        splits[split] = {str(sample_id) for sample_id in payload["sample_ids"]}
    return splits


def _render_note(df: pd.DataFrame, path: Path) -> str:
    lines = [
        "# Stage O Probe Sanity",
        "",
        f"CSV: `{path}`",
        "",
        "## Best Real-Label Rows",
        "",
    ]
    real = df[df["control"] == "real_labels"].copy()
    if not real.empty:
        for alias, group in real.groupby("alias"):
            best = group.sort_values("auroc", ascending=False).head(5)
            lines.append(f"### {alias}")
            for _, row in best.iterrows():
                lines.append(f"- L{int(row['layer'])} `{row['feature']}` AUROC `{row['auroc']:.3f}` AUPRC `{row['auprc']:.3f}`")
            lines.append("")
    lines.extend(["## Label-Shuffle Check", ""])
    shuffled = df[df["control"] == "train_label_shuffled"].copy()
    if not shuffled.empty:
        for alias, group in shuffled.groupby("alias"):
            max_auroc = group["auroc"].max()
            lines.append(f"- `{alias}` max shuffled-label AUROC `{max_auroc:.3f}`")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- If real-label AUROC stays near 1.0 under split-locked evaluation while shuffled-label AUROC collapses, the separability is not a train/test leakage bug.",
            "- It can still be a readout-position confound if the representation is taken at the assistant generation prompt and therefore linearly exposes the next-token decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
