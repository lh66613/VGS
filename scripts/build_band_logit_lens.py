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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


DEFAULT_SUBSPACES = ["band5_16", "top4", "random12", "full"]
DEFAULT_MODEL_PATHS = [
    Path("/data/lh/ModelandDataset/llava-1.5-7b-hf"),
    Path("/home/NCUT/25/25_lh/~/llava-1.5-7b-hf"),
]
YES_TOKEN_IDS = [3869, 4874, 22483]
NO_TOKEN_IDS = [694, 1939, 11698]
COCO_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
ABSENCE_TERMS = [
    "no",
    "not",
    "none",
    "absent",
    "missing",
    "without",
    "cannot",
    "neither",
    "nothing",
]
PRESENCE_TERMS = [
    "yes",
    "present",
    "visible",
    "shown",
    "seen",
    "there",
    "appears",
    "contains",
    "includes",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
    "you",
}
SEMANTIC_KEEP = set(ABSENCE_TERMS + PRESENCE_TERMS + COCO_OBJECTS)
COCO_OBJECT_SET = set(COCO_OBJECTS)
COCO_ALIASES = {
    "people": "person",
    "men": "person",
    "women": "person",
    "children": "person",
    "television": "tv",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "ski": "skis",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a differential Band Logit Lens from cached hidden-state corrections."
    )
    parser.add_argument("--hidden-states", default="outputs/hidden_states/layer_24.pt")
    parser.add_argument("--svd", default="outputs/svd/svd_layer_24.pt")
    parser.add_argument(
        "--sample-predictions",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/stage2/sample_predictions.csv",
    )
    parser.add_argument(
        "--selected-table",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_scan_7b/band_scan_table.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/band_logit_lens_7b",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_band_logit_lens(
        hidden_states=Path(args.hidden_states),
        svd=Path(args.svd),
        sample_predictions=Path(args.sample_predictions),
        selected_table=Path(args.selected_table),
        predictions=Path(args.predictions),
        model_path=_resolve_model_path(args.model_path),
        subspaces=args.subspaces,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    summary_path = write_json(Path(args.output_dir) / "build_band_logit_lens_summary.json", result)
    append_experiment_log(args.log_path, "build_band_logit_lens", summary_path, "ok")
    print(summary_path)


def build_band_logit_lens(
    hidden_states: Path,
    svd: Path,
    sample_predictions: Path,
    selected_table: Path,
    predictions: Path,
    model_path: Path,
    subspaces: list[str],
    top_k: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
    lm_head = _load_lm_head_weight(model_path)
    hidden = _torch_load(hidden_states)
    svd_payload = _torch_load(svd)
    layer = int(hidden["layer"])
    sample_ids = [str(item) for item in hidden["sample_ids"]]
    if sample_ids != [str(item) for item in svd_payload["sample_ids"]]:
        raise ValueError("Hidden-state and SVD sample_ids differ; refusing to build logit lens.")

    delta = hidden["z_img"].float() - hidden["z_blind"].float()
    basis = svd_payload["Vh"].float().T.contiguous()
    bases = _subspace_bases(basis, subspaces, seed + layer)
    selected = _selected_alphas(selected_table, subspaces)
    samples = pd.read_csv(sample_predictions)
    prediction_by_id = {str(row["sample_id"]): row for row in read_jsonl(predictions)}
    index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    group_defs = _token_groups(tokenizer)

    top_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for subspace in subspaces:
        if subspace not in bases or subspace not in selected:
            continue
        alpha = selected[subspace]
        selected_rows.append({"subspace": subspace, "alpha": alpha})
        subspace_samples = _selected_sample_rows(samples, subspace, alpha)
        for transition, transition_samples in _transition_groups(subspace_samples):
            ids = [str(item) for item in transition_samples["sample_id"].tolist()]
            valid_ids = [item for item in ids if item in index_by_id]
            hidden_indices = [index_by_id[item] for item in valid_ids]
            if not hidden_indices:
                transition_rows.append(
                    {
                        "subspace": subspace,
                        "transition": transition,
                        "alpha": alpha,
                        "n": 0,
                    }
                )
                continue

            projected = _project_delta(delta[hidden_indices], bases[subspace], alpha)
            mean_shift = _mean_vocab_shift(projected, lm_head)
            transition_rows.append(
                {
                    "subspace": subspace,
                    "transition": transition,
                    "alpha": alpha,
                    "n": len(hidden_indices),
                    "mean_shift_norm": float(torch.linalg.vector_norm(projected.mean(dim=0)).item()),
                }
            )
            top_rows.extend(
                _top_token_rows(
                    tokenizer=tokenizer,
                    shift=mean_shift,
                    subspace=subspace,
                    transition=transition,
                    alpha=alpha,
                    n=len(hidden_indices),
                    top_k=top_k,
                )
            )
            group_rows.extend(
                _token_group_rows(
                    tokenizer=tokenizer,
                    lm_head=lm_head,
                    projected=projected,
                    sample_ids=valid_ids,
                    prediction_by_id=prediction_by_id,
                    groups=group_defs,
                    subspace=subspace,
                    transition=transition,
                    alpha=alpha,
                )
            )
            object_rows.extend(
                _object_category_rows(
                    tokenizer=tokenizer,
                    lm_head=lm_head,
                    projected=projected,
                    sample_ids=valid_ids,
                    prediction_by_id=prediction_by_id,
                    subspace=subspace,
                    transition=transition,
                    alpha=alpha,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    selectivity_rows = _selectivity_rows(pd.DataFrame(group_rows))
    top_path = write_csv(output_dir / "band_logit_lens_top_tokens.csv", top_rows, fieldnames(top_rows))
    group_path = write_csv(
        output_dir / "band_logit_lens_token_group_shift.csv",
        group_rows,
        fieldnames(group_rows),
    )
    selectivity_path = write_csv(
        output_dir / "band_logit_lens_selectivity_scores.csv",
        selectivity_rows,
        fieldnames(selectivity_rows),
    )
    object_path = write_csv(
        output_dir / "band_logit_lens_object_category_shift.csv",
        object_rows,
        fieldnames(object_rows),
    )
    transitions_path = write_csv(
        output_dir / "band_logit_lens_transition_counts.csv",
        transition_rows,
        fieldnames(transition_rows),
    )
    selected_path = write_csv(
        output_dir / "band_logit_lens_selected_alphas.csv",
        selected_rows,
        fieldnames(selected_rows),
    )
    report_path = _write_report(
        output_dir / "band_logit_lens_report.md",
        selected=pd.DataFrame(selected_rows),
        transitions=pd.DataFrame(transition_rows),
        groups=pd.DataFrame(group_rows),
        selectivity=pd.DataFrame(selectivity_rows),
        object_categories=pd.DataFrame(object_rows),
        top_tokens=pd.DataFrame(top_rows),
        top_k=min(10, top_k),
    )
    return {
        "hidden_states": str(hidden_states),
        "svd": str(svd),
        "sample_predictions": str(sample_predictions),
        "selected_table": str(selected_table),
        "predictions": str(predictions),
        "model_path": str(model_path),
        "layer": layer,
        "subspaces": [row["subspace"] for row in selected_rows],
        "top_tokens_path": str(top_path),
        "token_group_shift_path": str(group_path),
        "selectivity_scores_path": str(selectivity_path),
        "object_category_shift_path": str(object_path),
        "transition_counts_path": str(transitions_path),
        "selected_alphas_path": str(selected_path),
        "report_path": str(report_path),
        "num_top_token_rows": len(top_rows),
        "num_group_rows": len(group_rows),
        "num_object_category_rows": len(object_rows),
        "yes_token_ids": YES_TOKEN_IDS,
        "no_token_ids": NO_TOKEN_IDS,
        "checksums": {
            "hidden_states": _sha256(hidden_states),
            "svd": _sha256(svd),
            "sample_predictions": _sha256(sample_predictions),
            "selected_table": _sha256(selected_table),
            "model_config": _sha256(model_path / "config.json"),
            "tokenizer_json": _sha256(model_path / "tokenizer.json"),
        },
        "logit_lens_definition": "Delta logits are alpha * W_U * P_subspace(z_img - z_blind), averaged by transition group.",
    }


def _resolve_model_path(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Model path does not exist: {path}")
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a local LLaVA model path.")


def _load_lm_head_weight(model_path: Path) -> torch.Tensor:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "language_model.lm_head.weight"
    shard = model_path / index["weight_map"][key]
    return load_file(str(shard), device="cpu")[key].float().contiguous()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _selected_alphas(path: Path, subspaces: list[str]) -> dict[str, float]:
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    for subspace in subspaces:
        rows = df[
            (df["operator"].astype(str) == "icd_blind")
            & (df["subspace"].astype(str) == subspace)
        ]
        if rows.empty:
            continue
        out[subspace] = float(rows.iloc[0]["alpha"])
    return out


def _selected_sample_rows(samples: pd.DataFrame, subspace: str, alpha: float) -> pd.DataFrame:
    rows = samples[
        (samples["operator"].astype(str) == "icd_blind")
        & (samples["subspace"].astype(str) == subspace)
        & (samples["split"].astype(str) == "test")
        & np.isclose(samples["alpha"].astype(float), alpha)
    ].copy()
    return rows


def _transition_groups(samples: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    fp = samples[samples["original_outcome"].astype(str) == "FP"]
    tp = samples[samples["original_outcome"].astype(str) == "TP"]
    return [
        ("FP Yes->No", fp[fp["final_outcome"].astype(str) == "TN"].copy()),
        ("FP Yes->Yes", fp[fp["final_outcome"].astype(str) == "FP"].copy()),
        ("TP Yes->Yes", tp[tp["final_outcome"].astype(str) == "TP"].copy()),
        ("TP Yes->No", tp[tp["final_outcome"].astype(str) == "FN"].copy()),
    ]


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
        elif match := re.fullmatch(r"band(\d+)_(\d+)", name):
            start, end = _one_indexed_interval(match.group(1), match.group(2), max_dim)
            out[name] = basis[:, start:end]
        elif name.startswith("random12"):
            out[name] = _random_basis(hidden_dim, min(12, hidden_dim), np.random.default_rng(_stable_seed(seed, name)))
        else:
            raise ValueError(f"Unsupported subspace: {name}")
    return out


def _project_delta(delta: torch.Tensor, basis: torch.Tensor | None, alpha: float) -> torch.Tensor:
    if basis is None:
        return alpha * delta.float()
    if basis.numel() == 0:
        return torch.zeros_like(delta, dtype=torch.float32)
    basis = basis.float().contiguous()
    coeff = delta.float() @ basis
    return alpha * (coeff @ basis.T)


def _mean_vocab_shift(projected: torch.Tensor, lm_head: torch.Tensor) -> torch.Tensor:
    if projected.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    mean_hidden = projected.mean(dim=0)
    return mean_hidden @ lm_head.T


def _token_groups(tokenizer: Any) -> dict[str, list[int]]:
    return {
        "yes_main": sorted(set(YES_TOKEN_IDS)),
        "no_main": sorted(set(NO_TOKEN_IDS)),
        "absence_vocab": _term_token_ids(tokenizer, ABSENCE_TERMS),
        "presence_vocab": _term_token_ids(tokenizer, PRESENCE_TERMS),
        "coco_object_vocab": _term_token_ids(tokenizer, COCO_OBJECTS),
    }


def _term_token_ids(tokenizer: Any, terms: list[str]) -> list[int]:
    ids: set[int] = set()
    allowed_words = _allowed_token_words(terms)
    for term in terms:
        variants = {f" {term}", f" {term.capitalize()}"}
        for variant in variants:
            ids.update(int(item) for item in tokenizer.encode(variant, add_special_tokens=False))
    special = set(_special_token_ids(tokenizer))
    return sorted(
        item
        for item in ids
        if item not in special and _keep_group_token(tokenizer, item, allowed_words)
    )


def _token_group_rows(
    tokenizer: Any,
    lm_head: torch.Tensor,
    projected: torch.Tensor,
    sample_ids: list[str],
    prediction_by_id: dict[str, dict[str, Any]],
    groups: dict[str, list[int]],
    subspace: str,
    transition: str,
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values_by_group: dict[str, np.ndarray] = {}
    for name, token_ids in groups.items():
        values = _fixed_token_group_values(projected, lm_head, token_ids)
        values_by_group[name] = values
        rows.append(_group_row(subspace, transition, alpha, name, token_ids, tokenizer, values))

    queried_values, queried_token_count = _queried_object_values(
        tokenizer=tokenizer,
        lm_head=lm_head,
        projected=projected,
        sample_ids=sample_ids,
        prediction_by_id=prediction_by_id,
    )
    values_by_group["queried_object"] = queried_values
    rows.append(
        _group_row(
            subspace,
            transition,
            alpha,
            "queried_object",
            [],
            tokenizer,
            queried_values,
            token_count_override=queried_token_count,
        )
    )

    rows.extend(
        [
            _contrast_row(
                subspace,
                transition,
                alpha,
                "contrast:no_main-yes_main",
                values_by_group["no_main"],
                values_by_group["yes_main"],
            ),
            _contrast_row(
                subspace,
                transition,
                alpha,
                "contrast:absence-presence",
                values_by_group["absence_vocab"],
                values_by_group["presence_vocab"],
            ),
            _contrast_row(
                subspace,
                transition,
                alpha,
                "contrast:no_main-queried_object",
                values_by_group["no_main"],
                values_by_group["queried_object"],
            ),
        ]
    )
    return rows


def _fixed_token_group_values(projected: torch.Tensor, lm_head: torch.Tensor, token_ids: list[int]) -> np.ndarray:
    if projected.numel() == 0 or not token_ids:
        return np.array([], dtype=np.float32)
    weights = lm_head[token_ids]
    values = (projected @ weights.T).mean(dim=1)
    return values.detach().cpu().numpy()


def _queried_object_values(
    tokenizer: Any,
    lm_head: torch.Tensor,
    projected: torch.Tensor,
    sample_ids: list[str],
    prediction_by_id: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, int]:
    values: list[float] = [math.nan] * len(sample_ids)
    total_token_count = 0
    for row_idx, sample_id in enumerate(sample_ids):
        row = prediction_by_id.get(str(sample_id), {})
        phrase = _extract_queried_object(str(row.get("question", "")))
        if not phrase:
            continue
        token_ids = _term_token_ids(tokenizer, [phrase])
        if not token_ids:
            continue
        total_token_count += len(token_ids)
        weights = lm_head[token_ids]
        value = (projected[row_idx] @ weights.T).mean()
        values[row_idx] = float(value.item())
    return np.asarray(values, dtype=np.float32), total_token_count


def _object_category_rows(
    tokenizer: Any,
    lm_head: torch.Tensor,
    projected: torch.Tensor,
    sample_ids: list[str],
    prediction_by_id: dict[str, dict[str, Any]],
    subspace: str,
    transition: str,
    alpha: float,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, float]]] = {}
    all_by_presence: dict[str, list[dict[str, float]]] = {}
    for row_idx, sample_id in enumerate(sample_ids):
        row = prediction_by_id.get(str(sample_id), {})
        info = _sample_object_info(row)
        if info is None:
            continue
        category, presence = info
        token_ids = _object_token_ids(tokenizer, category)
        if not token_ids:
            continue
        token_values = projected[row_idx] @ lm_head[token_ids].T
        values = {
            "mean": float(token_values.mean().item()),
            "logmeanexp": float(_logmeanexp(token_values).item()),
            "token_count": float(len(token_ids)),
        }
        rows_by_key.setdefault((category, presence), []).append(values)
        all_by_presence.setdefault(presence, []).append(values)

    rows: list[dict[str, Any]] = []
    for (category, presence), values in sorted(rows_by_key.items()):
        rows.append(
            _object_category_summary_row(
                subspace=subspace,
                transition=transition,
                alpha=alpha,
                category=category,
                presence=presence,
                values=values,
            )
        )
    for presence, values in sorted(all_by_presence.items()):
        rows.append(
            _object_category_summary_row(
                subspace=subspace,
                transition=transition,
                alpha=alpha,
                category="ALL",
                presence=presence,
                values=values,
            )
        )
    return rows


def _object_category_summary_row(
    subspace: str,
    transition: str,
    alpha: float,
    category: str,
    presence: str,
    values: list[dict[str, float]],
) -> dict[str, Any]:
    mean_values = np.asarray([item["mean"] for item in values], dtype=np.float32)
    lse_values = np.asarray([item["logmeanexp"] for item in values], dtype=np.float32)
    token_counts = np.asarray([item["token_count"] for item in values], dtype=np.float32)
    return {
        "subspace": subspace,
        "transition": transition,
        "alpha": alpha,
        "object_category": category,
        "object_presence": presence,
        "n": int(len(values)),
        "mean_token_count": _safe_mean(token_counts),
        "mean_token_delta": _safe_mean(mean_values),
        "median_token_delta": _safe_median(mean_values),
        "positive_rate_mean": _positive_rate(mean_values),
        "mean_logmeanexp_delta": _safe_mean(lse_values),
        "median_logmeanexp_delta": _safe_median(lse_values),
        "positive_rate_logmeanexp": _positive_rate(lse_values),
    }


def _sample_object_info(row: dict[str, Any]) -> tuple[str, str] | None:
    category = _extract_queried_object(str(row.get("question", "")))
    if not category:
        return None
    label = str(row.get("label", "")).lower()
    if label == "yes":
        presence = "present"
    elif label == "no":
        presence = "absent"
    else:
        presence = "unknown"
    return category, presence


def _object_token_ids(tokenizer: Any, category: str) -> list[int]:
    ids = tokenizer.encode(f" {category}", add_special_tokens=False)
    special = _special_token_ids(tokenizer)
    return [int(item) for item in ids if int(item) not in special]


def _logmeanexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values.float(), dim=0) - math.log(max(1, int(values.numel())))


def _group_row(
    subspace: str,
    transition: str,
    alpha: float,
    name: str,
    token_ids: list[int],
    tokenizer: Any,
    values: np.ndarray,
    token_count_override: int | None = None,
) -> dict[str, Any]:
    finite_values = values[np.isfinite(values)]
    return {
        "subspace": subspace,
        "transition": transition,
        "alpha": alpha,
        "token_group": name,
        "n": int(len(finite_values)),
        "token_count": int(token_count_override if token_count_override is not None else len(token_ids)),
        "mean_delta_logit": _safe_mean(values),
        "median_delta_logit": _safe_median(values),
        "positive_rate": _positive_rate(values),
        "token_ids": " ".join(str(item) for item in token_ids[:50]),
        "token_texts": " | ".join(_clean_token_text(tokenizer, item) for item in token_ids[:20]),
    }


def _contrast_row(
    subspace: str,
    transition: str,
    alpha: float,
    name: str,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    n = min(len(left), len(right))
    values = left[:n] - right[:n] if n else np.array([], dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    return {
        "subspace": subspace,
        "transition": transition,
        "alpha": alpha,
        "token_group": name,
        "n": int(len(finite_values)),
        "token_count": "",
        "mean_delta_logit": _safe_mean(values),
        "median_delta_logit": _safe_median(values),
        "positive_rate": _positive_rate(values),
        "token_ids": "",
        "token_texts": "",
    }


def _top_token_rows(
    tokenizer: Any,
    shift: torch.Tensor,
    subspace: str,
    transition: str,
    alpha: float,
    n: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if shift.numel() == 0:
        return rows
    for direction, descending in [("promoted", True), ("suppressed", False)]:
        ranks = torch.argsort(shift, descending=descending)
        kept = 0
        for token_id_tensor in ranks:
            token_id = int(token_id_tensor.item())
            token_text = _clean_token_text(tokenizer, token_id)
            if not _keep_top_token(tokenizer, token_id, token_text):
                continue
            rows.append(
                {
                    "subspace": subspace,
                    "transition": transition,
                    "alpha": alpha,
                    "n": n,
                    "direction": direction,
                    "rank": kept + 1,
                    "token_id": token_id,
                    "token_text": token_text,
                    "delta_logit": float(shift[token_id].item()),
                }
            )
            kept += 1
            if kept >= top_k:
                break
    return rows


def _keep_top_token(tokenizer: Any, token_id: int, text: str) -> bool:
    if token_id in _special_token_ids(tokenizer):
        return False
    clean = text.strip()
    if not clean or not re.search(r"[A-Za-z]", clean):
        return False
    lower = clean.lower()
    if lower in STOPWORDS and lower not in SEMANTIC_KEEP:
        return False
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    has_boundary = str(raw).startswith("▁") or text.startswith(" ")
    if not has_boundary and lower not in SEMANTIC_KEEP:
        return False
    return True


def _keep_group_token(tokenizer: Any, token_id: int, allowed_words: set[str]) -> bool:
    text = _clean_token_text(tokenizer, token_id)
    lower = text.lower().strip()
    if not lower or not re.search(r"[a-z]", lower):
        return False
    return lower in allowed_words


def _allowed_token_words(terms: list[str]) -> set[str]:
    words: set[str] = set()
    for term in terms:
        clean = re.sub(r"[^a-zA-Z ]+", " ", term.lower())
        parts = [part for part in clean.split() if part]
        words.update(parts)
        if len(parts) == 1:
            words.add(clean.strip())
    return words


def _extract_queried_object(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\?$", "", text)
    patterns = [
        r"^is there (?:a |an |any )?(.+?) in (?:the|this) image$",
        r"^are there (?:a |an |any |some )?(.+?) in (?:the|this) image$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            phrase = match.group(1).strip()
            return _canonical_coco_object(phrase)
    return ""


def _canonical_coco_object(phrase: str) -> str:
    phrase = phrase.strip().lower()
    if phrase in COCO_OBJECT_SET:
        return phrase
    if phrase in COCO_ALIASES:
        return COCO_ALIASES[phrase]
    singular = _singularize_query(phrase)
    if singular in COCO_OBJECT_SET:
        return singular
    return COCO_ALIASES.get(singular, "")


def _singularize_query(phrase: str) -> str:
    irregular = {
        "people": "person",
        "men": "person",
        "women": "person",
        "children": "person",
    }
    if phrase in irregular:
        return irregular[phrase]
    if phrase.endswith("ies"):
        return phrase[:-3] + "y"
    if phrase.endswith("es") and not phrase.endswith(("ses", "xes")):
        return phrase[:-2]
    if phrase.endswith("s") and not phrase.endswith("ss"):
        return phrase[:-1]
    return phrase


def _clean_token_text(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return text.replace("\n", "\\n").replace("\t", "\\t").strip()


def _write_report(
    path: Path,
    selected: pd.DataFrame,
    transitions: pd.DataFrame,
    groups: pd.DataFrame,
    selectivity: pd.DataFrame,
    object_categories: pd.DataFrame,
    top_tokens: pd.DataFrame,
    top_k: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_groups = [
        "no_main",
        "yes_main",
        "queried_object",
        "contrast:no_main-yes_main",
        "contrast:absence-presence",
        "contrast:no_main-queried_object",
    ]
    key = groups[groups["token_group"].astype(str).isin(key_groups)].copy() if not groups.empty else groups
    if not key.empty:
        key = key.sort_values(["subspace", "transition", "token_group"])
    band_key = key[
        (key["subspace"].astype(str) == "band5_16")
        & (key["transition"].astype(str).isin(["FP Yes->No", "FP Yes->Yes", "TP Yes->Yes", "TP Yes->No"]))
    ].copy() if not key.empty else key

    lines = [
        "# Band Logit Lens",
        "",
        "Definition: `Delta logits = alpha * W_U * P_subspace(z_img - z_blind)`, grouped by calibrated test-set outcome transition.",
        "",
        "This is a differential lens over the actual intervention component; it does not decode bare SVD directions.",
        "",
        "## Key readout",
        "",
        *_key_readout_lines(groups, selectivity),
        "",
        "## Selected alphas",
        "",
        markdown_table(selected),
        "",
        "## Transition counts",
        "",
        markdown_table(transitions, columns=["subspace", "transition", "alpha", "n", "mean_shift_norm"]),
        "",
        "## Selectivity scores",
        "",
        "Selectivity is `Delta(No-Yes)_{FP Yes->No} - Delta(No-Yes)_{TP Yes->Yes}`.",
        "",
        markdown_table(
            selectivity,
            columns=[
                "subspace",
                "fp_yes_to_no_delta_no_yes",
                "tp_yes_to_yes_delta_no_yes",
                "selectivity_no_yes",
                "fp_yes_to_no_absence_presence",
                "tp_yes_to_yes_absence_presence",
                "selectivity_absence_presence",
            ],
        ),
        "",
        "## Band5-16 token group shifts",
        "",
        markdown_table(
            band_key,
            columns=[
                "transition",
                "token_group",
                "n",
                "token_count",
                "mean_delta_logit",
                "median_delta_logit",
                "positive_rate",
            ],
        ),
        "",
        "## Cross-subspace semantic contrasts",
        "",
        markdown_table(
            key[key["token_group"].astype(str).str.startswith("contrast:")] if not key.empty else key,
            columns=["subspace", "transition", "token_group", "mean_delta_logit", "positive_rate"],
            max_rows=80,
        ),
        "",
        "## Object category shifts",
        "",
        "Object rows aggregate the queried COCO category per sample, split by whether that object is actually absent or present. `logmeanexp` is a normalized log-sum-exp over the category tokenization.",
        "",
        markdown_table(
            _object_report_view(object_categories),
            columns=[
                "subspace",
                "transition",
                "object_category",
                "object_presence",
                "n",
                "mean_token_count",
                "mean_token_delta",
                "positive_rate_mean",
                "mean_logmeanexp_delta",
                "positive_rate_logmeanexp",
            ],
            max_rows=80,
        ),
        "",
        "## Top promoted/suppressed tokens",
        "",
    ]
    for subspace in ["band5_16", "top4", "random12", "full"]:
        for transition in ["FP Yes->No", "FP Yes->Yes", "TP Yes->Yes", "TP Yes->No"]:
            view = top_tokens[
                (top_tokens["subspace"].astype(str) == subspace)
                & (top_tokens["transition"].astype(str) == transition)
                & (top_tokens["rank"].astype(int) <= top_k)
            ].copy() if not top_tokens.empty else top_tokens
            if view.empty:
                continue
            lines.extend(
                [
                    f"### {subspace} / {transition}",
                    "",
                    markdown_table(view, columns=["direction", "rank", "token_text", "delta_logit"]),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _key_readout_lines(groups: pd.DataFrame, selectivity: pd.DataFrame) -> list[str]:
    if groups.empty:
        return ["_Missing._"]
    rows: list[str] = []
    specs = [
        (
            "Band5-16 / FP Yes->No",
            "band5_16",
            "FP Yes->No",
            "successful FP corrections",
        ),
        (
            "Band5-16 / TP Yes->Yes",
            "band5_16",
            "TP Yes->Yes",
            "preserved true positives",
        ),
    ]
    for title, subspace, transition, description in specs:
        no_shift = _lookup_group_shift(groups, subspace, transition, "no_main")
        yes_shift = _lookup_group_shift(groups, subspace, transition, "yes_main")
        object_shift = _lookup_group_shift(groups, subspace, transition, "queried_object")
        no_yes = _lookup_group_shift(groups, subspace, transition, "contrast:no_main-yes_main")
        absence_presence = _lookup_group_shift(groups, subspace, transition, "contrast:absence-presence")
        rows.append(
            f"- {title}: in {description}, No={_fmt(no_shift)}, Yes={_fmt(yes_shift)}, "
            f"queried_object={_fmt(object_shift)}, No-Yes={_fmt(no_yes)}, "
            f"Absence-Presence={_fmt(absence_presence)}."
        )

    for subspace in ["top4", "full", "random12"]:
        no_yes_fp = _lookup_group_shift(groups, subspace, "FP Yes->No", "contrast:no_main-yes_main")
        no_yes_tp = _lookup_group_shift(groups, subspace, "TP Yes->Yes", "contrast:no_main-yes_main")
        rows.append(
            f"- {subspace}: No-Yes contrast is {_fmt(no_yes_fp)} on FP Yes->No and "
            f"{_fmt(no_yes_tp)} on TP Yes->Yes, useful for checking whether the shift is selective."
        )
    if not selectivity.empty:
        best = selectivity.sort_values("selectivity_no_yes", ascending=False).iloc[0]
        rows.append(
            f"- Best selectivity: {best['subspace']} with No-Yes selectivity "
            f"{_fmt(float(best['selectivity_no_yes']))}."
        )
    return rows


def _selectivity_rows(groups: pd.DataFrame) -> list[dict[str, Any]]:
    if groups.empty:
        return []
    rows: list[dict[str, Any]] = []
    for subspace in sorted(groups["subspace"].astype(str).unique()):
        fp_no_yes = _lookup_group_shift(groups, subspace, "FP Yes->No", "contrast:no_main-yes_main")
        tp_no_yes = _lookup_group_shift(groups, subspace, "TP Yes->Yes", "contrast:no_main-yes_main")
        fp_abs = _lookup_group_shift(groups, subspace, "FP Yes->No", "contrast:absence-presence")
        tp_abs = _lookup_group_shift(groups, subspace, "TP Yes->Yes", "contrast:absence-presence")
        fp_no_obj = _lookup_group_shift(groups, subspace, "FP Yes->No", "contrast:no_main-queried_object")
        tp_no_obj = _lookup_group_shift(groups, subspace, "TP Yes->Yes", "contrast:no_main-queried_object")
        rows.append(
            {
                "subspace": subspace,
                "fp_yes_to_no_delta_no_yes": fp_no_yes,
                "tp_yes_to_yes_delta_no_yes": tp_no_yes,
                "selectivity_no_yes": _diff(fp_no_yes, tp_no_yes),
                "fp_yes_to_no_absence_presence": fp_abs,
                "tp_yes_to_yes_absence_presence": tp_abs,
                "selectivity_absence_presence": _diff(fp_abs, tp_abs),
                "fp_yes_to_no_no_minus_queried_object": fp_no_obj,
                "tp_yes_to_yes_no_minus_queried_object": tp_no_obj,
                "selectivity_no_minus_queried_object": _diff(fp_no_obj, tp_no_obj),
            }
        )
    return sorted(rows, key=lambda row: row["selectivity_no_yes"], reverse=True)


def _object_report_view(object_categories: pd.DataFrame) -> pd.DataFrame:
    if object_categories.empty:
        return object_categories
    aggregate = object_categories[
        (object_categories["subspace"].astype(str).isin(["band5_16", "top4", "random12", "full"]))
        & (object_categories["object_category"].astype(str) == "ALL")
    ].copy()
    band_categories = object_categories[
        (object_categories["subspace"].astype(str) == "band5_16")
        & (object_categories["object_category"].astype(str) != "ALL")
        & (object_categories["n"].astype(int) >= 3)
    ].copy()
    view = pd.concat([aggregate, band_categories], ignore_index=True)
    if view.empty:
        view = object_categories.copy()
    rank = {"band5_16": 0, "top4": 1, "full": 2, "random12": 3}
    view["_rank"] = view["subspace"].map(rank).fillna(99).astype(int)
    view["_category_rank"] = (view["object_category"].astype(str) != "ALL").astype(int)
    return (
        view.sort_values(["_rank", "transition", "object_presence", "_category_rank", "n"], ascending=[True, True, True, True, False])
        .drop(columns=["_rank", "_category_rank"])
    )


def _lookup_group_shift(groups: pd.DataFrame, subspace: str, transition: str, token_group: str) -> float:
    row = groups[
        (groups["subspace"].astype(str) == subspace)
        & (groups["transition"].astype(str) == transition)
        & (groups["token_group"].astype(str) == token_group)
    ]
    if row.empty:
        return math.nan
    return float(row.iloc[0]["mean_delta_logit"])


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.3f}"


def _diff(left: float, right: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.nan
    return float(left - right)


def _one_indexed_interval(start_text: str, end_text: str, max_dim: int) -> tuple[int, int]:
    start = int(start_text)
    end = int(end_text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid 1-indexed subspace interval: {start_text}_{end_text}")
    return min(start - 1, max_dim), min(end, max_dim)


def _random_basis(hidden_dim: int, dim: int, rng: np.random.Generator) -> torch.Tensor:
    random_matrix = rng.normal(size=(hidden_dim, dim)).astype(np.float32)
    random_q, _ = np.linalg.qr(random_matrix)
    return torch.from_numpy(random_q.astype(np.float32, copy=False))


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def _safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def _positive_rate(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float((values > 0).mean()) if len(values) else math.nan


def _special_token_ids(tokenizer: Any) -> set[int]:
    return {int(item) for item in tokenizer.all_special_ids if item is not None}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
