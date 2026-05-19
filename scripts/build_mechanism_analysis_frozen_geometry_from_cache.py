#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json


DEFAULT_WINDOWS = [f"band{start}_{start + 11}" for start in range(1, 54, 4)]
DEFAULT_SUBSPACES = [
    "full",
    "top4",
    "top16",
    *DEFAULT_WINDOWS,
    "tail257_1024",
    "random12",
    *[f"random12_s{idx:02d}" for idx in range(20)],
    *[f"randcontig12_s{idx:02d}" for idx in range(20)],
]
YES_TOKEN_IDS = [3869, 4874, 22483]
NO_TOKEN_IDS = [694, 1939, 11698]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build expanded frozen ICD geometry from cached hidden states and SVD."
    )
    parser.add_argument("--hidden-states", default="outputs/hidden_states/layer_24.pt")
    parser.add_argument("--svd", default="outputs/svd/svd_layer_24.pt")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument(
        "--reference-geometry",
        default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv",
    )
    parser.add_argument("--model-path", default="/data/lh/ModelandDataset/llava-1.5-7b-hf")
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--yes-token-ids", nargs="+", type=int, default=YES_TOKEN_IDS)
    parser.add_argument("--no-token-ids", nargs="+", type=int, default=NO_TOKEN_IDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_operator_geometry",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_frozen_geometry(
        hidden_states=Path(args.hidden_states),
        svd=Path(args.svd),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        reference_geometry=Path(args.reference_geometry),
        model_path=Path(args.model_path),
        subspaces=args.subspaces,
        yes_token_ids=sorted(set(args.yes_token_ids)),
        no_token_ids=sorted(set(args.no_token_ids)),
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "operator_geometry_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_frozen_geometry_from_cache", summary_path, "ok")
    print(summary_path)


def build_frozen_geometry(
    hidden_states: Path,
    svd: Path,
    predictions: Path,
    margin_scores: Path,
    reference_geometry: Path,
    model_path: Path,
    subspaces: list[str],
    yes_token_ids: list[int],
    no_token_ids: list[int],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    hidden = torch.load(hidden_states, map_location="cpu")
    svd_payload = torch.load(svd, map_location="cpu")
    layer = int(hidden["layer"])
    sample_ids = [str(item) for item in hidden["sample_ids"]]
    if sample_ids != [str(item) for item in svd_payload["sample_ids"]]:
        raise ValueError("Hidden-state and SVD sample_ids differ; refusing to build frozen geometry.")

    predictions_by_id = {str(row["sample_id"]): row for row in read_jsonl(predictions)}
    margin_by_id = _margin_rows(margin_scores)
    reference_by_id = _reference_rows(reference_geometry, operator="icd_blind", layer=layer)
    yes_weight, no_weight = _load_lm_head_weights(model_path, yes_token_ids, no_token_ids)

    delta = hidden["z_img"].float() - hidden["z_blind"].float()
    basis = svd_payload["Vh"].float().T.contiguous()
    bases = _subspace_bases(basis, subspaces, seed + layer)
    delta_norm_sq = (delta * delta).sum(dim=1).numpy()

    metric_columns: dict[str, dict[str, np.ndarray]] = {}
    for name, subspace_basis in bases.items():
        metric_columns[name] = _projected_metrics(delta, subspace_basis, yes_weight, no_weight, delta_norm_sq)

    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        pred = predictions_by_id.get(sample_id, {})
        margin = margin_by_id.get(sample_id, {})
        ref = reference_by_id.get(sample_id, {})
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_subset": pred.get("subset", margin.get("subset", "")),
            "label": pred.get("label", margin.get("label", "")),
            "outcome": pred.get("outcome", margin.get("outcome", "")),
            "parsed_prediction": pred.get("parsed_prediction", margin.get("parsed_prediction", "")),
            "question": pred.get("question", ""),
            "image": pred.get("image", ""),
            "image_path": pred.get("image_path", ""),
            "operator": "icd_blind",
            "layer": layer,
            "delta_norm_sq": float(delta_norm_sq[idx]),
            "orig_yes_minus_no_logit": _coalesce(ref.get("orig_yes_minus_no_logit"), margin.get("yes_minus_no_logit")),
            "neg_yes_minus_no_logit": _coalesce(ref.get("neg_yes_minus_no_logit"), math.nan),
            "orig_no_minus_yes_logit": _coalesce(ref.get("orig_no_minus_yes_logit"), margin.get("no_minus_yes_logit")),
            "neg_no_minus_yes_logit": _coalesce(ref.get("neg_no_minus_yes_logit"), math.nan),
        }
        for name in bases:
            metrics = metric_columns[name]
            row[f"energy_{name}"] = float(metrics["energy"][idx])
            row[f"energy_frac_{name}"] = float(metrics["energy_frac"][idx])
            row[f"dlogit_yes_{name}"] = float(metrics["dlogit_yes"][idx])
            row[f"dlogit_no_{name}"] = float(metrics["dlogit_no"][idx])
            row[f"dmargin_no_minus_yes_{name}"] = float(metrics["dmargin_no_minus_yes"][idx])
        rows.append(row)

    output_root = Path(output_dir)
    geometry_path = write_csv(output_root / "operator_geometry.csv", rows, _fieldnames(rows))
    validation = _validate_reference(rows, reference_by_id, ["full", "top4", "top16", "band5_16", "tail257_1024", "random12"])
    validation_path = write_csv(output_root / "reference_validation.csv", validation, _fieldnames(validation))
    return {
        "operator_geometry_path": str(geometry_path),
        "reference_validation_path": str(validation_path),
        "hidden_states": str(hidden_states),
        "svd": str(svd),
        "predictions": str(predictions),
        "margin_scores": str(margin_scores),
        "reference_geometry": str(reference_geometry),
        "model_path": str(model_path),
        "layer": layer,
        "operator": "icd_blind",
        "subspaces": list(bases.keys()),
        "yes_token_ids": yes_token_ids,
        "no_token_ids": no_token_ids,
        "num_rows": len(rows),
        "difference_convention": "delta = z_img - z_blind, matching frozen operator_geometry orig - neg",
        "checksums": {
            "hidden_states": _sha256(hidden_states),
            "svd": _sha256(svd),
            "reference_geometry": _sha256(reference_geometry),
            "model_config": _sha256(model_path / "config.json"),
            "tokenizer_json": _sha256(model_path / "tokenizer.json"),
            "tokenizer_model": _sha256(model_path / "tokenizer.model"),
        },
    }


def _projected_metrics(
    delta: torch.Tensor,
    basis: torch.Tensor | None,
    yes_weight: torch.Tensor,
    no_weight: torch.Tensor,
    delta_norm_sq: np.ndarray,
) -> dict[str, np.ndarray]:
    if basis is None:
        energy = delta_norm_sq
        yes_logits = delta @ yes_weight.T
        no_logits = delta @ no_weight.T
    elif basis.numel() == 0:
        n = delta.shape[0]
        energy = np.zeros(n, dtype=np.float32)
        yes_logits = torch.zeros((n, yes_weight.shape[0]), dtype=torch.float32)
        no_logits = torch.zeros((n, no_weight.shape[0]), dtype=torch.float32)
    else:
        basis = basis.float().contiguous()
        coeff = delta @ basis
        energy = (coeff * coeff).sum(dim=1).numpy()
        yes_logits = coeff @ (yes_weight @ basis).T
        no_logits = coeff @ (no_weight @ basis).T
    dlogit_yes = torch.max(yes_logits, dim=1).values.numpy()
    dlogit_no = torch.max(no_logits, dim=1).values.numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        energy_frac = np.where(delta_norm_sq > 0, energy / delta_norm_sq, np.nan)
    return {
        "energy": energy,
        "energy_frac": energy_frac,
        "dlogit_yes": dlogit_yes,
        "dlogit_no": dlogit_no,
        "dmargin_no_minus_yes": dlogit_no - dlogit_yes,
    }


def _subspace_bases(basis: torch.Tensor, names: list[str], seed: int) -> dict[str, torch.Tensor | None]:
    hidden_dim, max_dim = basis.shape
    out: dict[str, torch.Tensor | None] = {}
    for name in dict.fromkeys(names):
        if name == "full":
            out[name] = None
        elif name == "top4":
            out[name] = basis[:, : min(4, max_dim)]
        elif name == "top16":
            out[name] = basis[:, : min(16, max_dim)]
        elif name == "band5_16":
            out[name] = basis[:, 4 : min(16, max_dim)]
        elif name == "tail257_1024":
            out[name] = basis[:, 256 : min(1024, max_dim)]
        elif name.startswith("random12"):
            out[name] = _random_basis(hidden_dim, min(12, hidden_dim), np.random.default_rng(_stable_seed(seed, name)))
        elif match := re.fullmatch(r"band(\d+)_(\d+)", name):
            start, end = _one_indexed_interval(match.group(1), match.group(2), max_dim)
            out[name] = basis[:, start:end]
        elif match := re.fullmatch(r"randcontig(\d+)_s(\d+)", name):
            width = max(1, int(match.group(1)))
            rng = np.random.default_rng(_stable_seed(seed, name))
            max_start = max(0, max_dim - width)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            out[name] = basis[:, start : min(max_dim, start + width)]
        else:
            raise ValueError(f"Unsupported frozen subspace: {name}")
    return out


def _load_lm_head_weights(model_path: Path, yes_ids: list[int], no_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = model_path / index["weight_map"]["language_model.lm_head.weight"]
    weight = load_file(str(shard), device="cpu")["language_model.lm_head.weight"].float()
    return weight[yes_ids].contiguous(), weight[no_ids].contiguous()


def _margin_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(row.sample_id): row._asdict() for row in df.itertuples(index=False)}


def _reference_rows(path: Path, operator: str, layer: int) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    view = df[(df["operator"].astype(str) == operator) & (df["layer"].astype(int) == int(layer))]
    return {str(row.sample_id): row._asdict() for row in view.itertuples(index=False)}


def _validate_reference(rows: list[dict[str, Any]], reference_by_id: dict[str, dict[str, Any]], subspaces: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for subspace in subspaces:
        col = f"dmargin_no_minus_yes_{subspace}"
        diffs = []
        for row in rows:
            ref = reference_by_id.get(str(row["sample_id"]), {})
            if col in row and col in ref and pd.notna(ref[col]):
                diffs.append(abs(float(row[col]) - float(ref[col])))
        if diffs:
            out.append(
                {
                    "subspace": subspace,
                    "n": len(diffs),
                    "mean_abs_diff": float(np.mean(diffs)),
                    "max_abs_diff": float(np.max(diffs)),
                }
            )
    return out


def _random_basis(hidden_dim: int, dim: int, rng: np.random.Generator) -> torch.Tensor:
    random_matrix = rng.normal(size=(hidden_dim, dim)).astype(np.float32)
    random_q, _ = np.linalg.qr(random_matrix)
    return torch.from_numpy(random_q.astype(np.float32, copy=False))


def _one_indexed_interval(start_text: str, end_text: str, max_dim: int) -> tuple[int, int]:
    start = int(start_text)
    end = int(end_text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid 1-indexed subspace interval: {start_text}_{end_text}")
    return min(start - 1, max_dim), min(end, max_dim)


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _coalesce(value: Any, fallback: Any) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    try:
        return float(fallback)
    except Exception:
        return math.nan


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


if __name__ == "__main__":
    main()
