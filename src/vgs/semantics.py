"""Vocabulary-space semantic projections for Stage G."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from vgs.artifacts import load_hidden_layer, load_svd, read_jsonl
from vgs.io import ensure_dir, write_csv, write_json
from vgs.stage_e import _local_rescue_vectors, _unit_vector


def run_semantic_interpretation(
    layers: list[int],
    k: int,
    model_path: str | Path,
    svd_dir: str | Path,
    hidden_states_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    tail_layer: int = 24,
    tail_band: tuple[int, int] = (257, 1024),
    rescue_layer: int = 32,
    top_n: int = 30,
    apply_final_norm: bool = True,
    normalize_token_vectors: bool = True,
    natural_token_filter: bool = True,
    condition_plan_path: str | Path = "outputs/stage_b/stage_b_condition_plan.jsonl",
    condition_hidden_dir: str | Path = "outputs/stage_b_hidden",
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    unembedding = _load_unembedding(
        model_path,
        apply_final_norm=apply_final_norm,
        normalize_token_vectors=normalize_token_vectors,
    )
    object_vocab = _object_vocab(read_jsonl(predictions_path))
    projector = VocabularyProjector(tokenizer, unembedding, object_vocab, natural_token_filter)

    token_rows: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    sample_specs: list[dict[str, Any]] = []

    for layer in tqdm(layers, desc="Stage G SVD directions", unit="layer"):
        svd = load_svd(svd_dir, layer)
        vh = svd["Vh"].float()
        for direction_index in range(min(k, vh.shape[0])):
            name = f"L{layer}_svd_{direction_index + 1}"
            rows = projector.project_signed(
                name=name,
                family="top_svd_backbone",
                layer=layer,
                vector=vh[direction_index],
                top_n=top_n,
                metadata={"direction_index": direction_index + 1},
            )
            token_rows.extend(rows)
            object_summaries.append(_summarize_object(name, "top_svd_backbone", layer, rows))
            sample_specs.append(
                {
                    "name": name,
                    "family": "top_svd_backbone",
                    "layer": layer,
                    "kind": "signed",
                    "vector": vh[direction_index].float(),
                    "metadata": {"direction_index": direction_index + 1},
                }
            )

    tail_svd = load_svd(svd_dir, tail_layer)
    start, end = tail_band
    tail_basis = tail_svd["Vh"].float()[start - 1 : end].T
    tail_name = f"L{tail_layer}_tail_{start}_{end}"
    tail_rows = projector.project_subspace_energy(
        name=tail_name,
        family="tail_slice",
        layer=tail_layer,
        basis=tail_basis,
        top_n=top_n,
        metadata={"band_start": start, "band_end": end},
    )
    token_rows.extend(tail_rows)
    object_summaries.append(_summarize_object(tail_name, "tail_slice", tail_layer, tail_rows))
    sample_specs.append(
        {
            "name": tail_name,
            "family": "tail_slice",
            "layer": tail_layer,
            "kind": "subspace_energy",
            "basis": tail_basis.float(),
            "metadata": {"band_start": start, "band_end": end},
        }
    )

    rescue_vectors = _build_local_rescue_mean_vectors(
        layer=rescue_layer,
        hidden_states_dir=hidden_states_dir,
        predictions_path=predictions_path,
        condition_plan_path=condition_plan_path,
        condition_hidden_dir=condition_hidden_dir,
    )
    for vector_name, vector in rescue_vectors.items():
        name = f"L{rescue_layer}_{vector_name}"
        rows = projector.project_signed(
            name=name,
            family="local_tn_rescue",
            layer=rescue_layer,
            vector=torch.as_tensor(vector).float(),
            top_n=top_n,
            metadata={"source": vector_name},
        )
        token_rows.extend(rows)
        object_summaries.append(_summarize_object(name, "local_tn_rescue", rescue_layer, rows))
        sample_specs.append(
            {
                "name": name,
                "family": "local_tn_rescue",
                "layer": rescue_layer,
                "kind": "signed",
                "vector": torch.as_tensor(vector).float(),
                "metadata": {"source": vector_name},
            }
        )

    token_path = output_dir / "semantic_projection_tokens.jsonl"
    with token_path.open("w", encoding="utf-8") as f:
        for row in token_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cluster_rows = _cluster_summary(token_rows)
    write_csv(output_dir / "semantic_cluster_summary.csv", cluster_rows, list(cluster_rows[0].keys()))
    write_csv(output_dir / "semantic_object_summary.csv", object_summaries, list(object_summaries[0].keys()))
    sample_rows, contrast_rows = _sample_level_semantics(
        sample_specs=sample_specs,
        hidden_states_dir=hidden_states_dir,
        prediction_rows=read_jsonl(predictions_path),
        top_n=min(top_n, 20),
    )
    sample_path = write_csv(output_dir / "semantic_sample_extremes.csv", sample_rows, _fieldnames(sample_rows))
    contrast_path = write_csv(
        output_dir / "semantic_outcome_contrasts.csv",
        contrast_rows,
        _fieldnames(contrast_rows),
    )
    markdown_path = _write_markdown_summary(output_dir, object_summaries, token_rows, contrast_rows, sample_rows)
    summary = {
        "layers": layers,
        "k": k,
        "model_path": str(model_path),
        "svd_dir": str(svd_dir),
        "hidden_states_dir": str(hidden_states_dir),
        "predictions_path": str(predictions_path),
        "tail_layer": tail_layer,
        "tail_band": f"{start}-{end}",
        "rescue_layer": rescue_layer,
        "top_n": top_n,
        "apply_final_norm": apply_final_norm,
        "normalize_token_vectors": normalize_token_vectors,
        "natural_token_filter": natural_token_filter,
        "num_token_rows": len(token_rows),
        "token_path": str(token_path),
        "cluster_summary_path": str(output_dir / "semantic_cluster_summary.csv"),
        "object_summary_path": str(output_dir / "semantic_object_summary.csv"),
        "sample_extremes_path": str(sample_path),
        "outcome_contrasts_path": str(contrast_path),
        "markdown_summary_path": str(markdown_path),
        "num_sample_extreme_rows": len(sample_rows),
        "num_outcome_contrast_rows": len(contrast_rows),
    }
    write_json(output_dir / "semantic_interpretation_results_summary.json", summary)
    return summary


class VocabularyProjector:
    def __init__(
        self,
        tokenizer: Any,
        unembedding: torch.Tensor,
        object_vocab: set[str],
        natural_token_filter: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.unembedding = unembedding.float().cpu()
        self.object_vocab = object_vocab
        self.natural_token_filter = natural_token_filter
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def project_signed(
        self,
        name: str,
        family: str,
        layer: int,
        vector: torch.Tensor,
        top_n: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vector = vector.float().cpu()
        vector = vector / torch.clamp(torch.linalg.norm(vector), min=1e-12)
        scores = self.unembedding @ vector
        pos = _top_filtered(
            scores,
            top_n,
            largest=True,
            tokenizer=self.tokenizer,
            special_ids=self.special_ids,
            natural_token_filter=self.natural_token_filter,
        )
        neg = _top_filtered(
            scores,
            top_n,
            largest=False,
            tokenizer=self.tokenizer,
            special_ids=self.special_ids,
            natural_token_filter=self.natural_token_filter,
        )
        return self._rows(name, family, layer, "positive", pos, metadata) + self._rows(
            name, family, layer, "negative", neg, metadata
        )

    def project_subspace_energy(
        self,
        name: str,
        family: str,
        layer: int,
        basis: torch.Tensor,
        top_n: int,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 2048,
    ) -> list[dict[str, Any]]:
        basis = torch.linalg.qr(basis.float().cpu(), mode="reduced").Q
        scores = []
        for start in range(0, self.unembedding.shape[0], chunk_size):
            chunk = self.unembedding[start : start + chunk_size]
            scores.append(torch.linalg.norm(chunk @ basis, dim=1))
        energy = torch.cat(scores, dim=0)
        top = _top_filtered(
            energy,
            top_n,
            largest=True,
            tokenizer=self.tokenizer,
            special_ids=self.special_ids,
            natural_token_filter=self.natural_token_filter,
        )
        return self._rows(name, family, layer, "energy", top, metadata)

    def _rows(
        self,
        name: str,
        family: str,
        layer: int,
        side: str,
        items: list[tuple[int, float, str]],
        metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        rows = []
        for rank, (token_id, score, token) in enumerate(items, start=1):
            clean = _clean_token(token)
            rows.append(
                {
                    "object": name,
                    "family": family,
                    "layer": layer,
                    "side": side,
                    "rank": rank,
                    "token_id": token_id,
                    "token": token,
                    "clean_token": clean,
                    "score": score,
                    "semantic_category": _semantic_category(clean, self.object_vocab),
                    **(metadata or {}),
                }
            )
        return rows


def _load_unembedding(
    model_path: str | Path,
    apply_final_norm: bool,
    normalize_token_vectors: bool,
) -> torch.Tensor:
    model_path = Path(model_path)
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    lm_key = "language_model.lm_head.weight"
    lm_file = model_path / weight_map[lm_key]
    with safe_open(lm_file, framework="pt", device="cpu") as f:
        lm_head = f.get_tensor(lm_key).float()
        if apply_final_norm:
            norm_key = "language_model.model.norm.weight"
            norm_file = model_path / weight_map[norm_key]
            if norm_file == lm_file:
                norm = f.get_tensor(norm_key).float()
            else:
                with safe_open(norm_file, framework="pt", device="cpu") as norm_f:
                    norm = norm_f.get_tensor(norm_key).float()
            lm_head = lm_head * norm.unsqueeze(0)
    if normalize_token_vectors:
        lm_head = lm_head / torch.clamp(torch.linalg.norm(lm_head, dim=1, keepdim=True), min=1e-12)
    return lm_head


def _build_local_rescue_mean_vectors(
    layer: int,
    hidden_states_dir: str | Path,
    predictions_path: str | Path,
    condition_plan_path: str | Path,
    condition_hidden_dir: str | Path,
) -> dict[str, np.ndarray]:
    prediction_rows = read_jsonl(predictions_path)
    labels_by_id: dict[str, int] = {}
    for row in prediction_rows:
        sample_id = str(row["sample_id"])
        if row.get("outcome") == "FP":
            labels_by_id[sample_id] = 1
        elif row.get("outcome") == "TN":
            labels_by_id[sample_id] = 0

    hidden = load_hidden_layer(hidden_states_dir, layer)
    sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
    keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
    kept_sample_ids = [sample_ids[idx] for idx in keep]
    y = np.array([labels_by_id[sample_id] for sample_id in kept_sample_ids], dtype=np.int64)
    diff = hidden["z_blind"][keep].float().numpy() - hidden["z_img"][keep].float().numpy()
    x_scaled = StandardScaler().fit_transform(diff)
    local_payloads = _local_rescue_vectors(
        layer=layer,
        prediction_rows=prediction_rows,
        sample_ids=kept_sample_ids,
        diff=diff,
        x_scaled=x_scaled,
        y=y,
        condition_plan_path=condition_plan_path,
        condition_hidden_dir=condition_hidden_dir,
    )

    vectors: dict[str, np.ndarray] = {}
    for name in ["question_tn_correction", "object_tn_correction", "local_knn_tn_correction"]:
        lookup = local_payloads.get(name, {}).get("vectors", {})
        signed_vectors = []
        for payload in lookup.values():
            direction = np.asarray(payload["direction"], dtype=np.float64)
            sign = float(payload.get("sign", 1.0))
            signed_vectors.append(sign * direction)
        if signed_vectors:
            vectors[name] = _unit_vector(np.mean(signed_vectors, axis=0))
    return vectors


def _top_filtered(
    scores: torch.Tensor,
    top_n: int,
    largest: bool,
    tokenizer: Any,
    special_ids: set[int],
    natural_token_filter: bool,
) -> list[tuple[int, float, str]]:
    multiplier = 1 if largest else -1
    ordered = torch.argsort(multiplier * scores, descending=True)
    items: list[tuple[int, float, str]] = []
    seen: set[str] = set()
    for token_id_tensor in ordered:
        token_id = int(token_id_tensor.item())
        if token_id in special_ids:
            continue
        try:
            token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except (IndexError, ValueError):
            continue
        clean = _clean_token(token)
        if not clean or clean in seen:
            continue
        if natural_token_filter and not _is_interpretable_token(clean):
            continue
        seen.add(clean)
        items.append((token_id, float(scores[token_id].item()), token))
        if len(items) >= top_n:
            break
    return items


def _clean_token(token: str) -> str:
    return token.replace("\n", "\\n").strip().lower()


def _is_interpretable_token(clean: str) -> bool:
    if clean in {"yes", "no", "not", "none", "never", "without", "cannot", "sorry", "sure"}:
        return True
    if clean.isdigit():
        return True
    try:
        clean.encode("ascii")
    except UnicodeEncodeError:
        return False
    if len(clean) < 3:
        return False
    if not any(ch.isalpha() for ch in clean):
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9'-]*", clean))


def _object_vocab(rows: list[dict[str, Any]]) -> set[str]:
    vocab = {
        "person",
        "people",
        "man",
        "woman",
        "child",
        "dog",
        "cat",
        "car",
        "bus",
        "truck",
        "bicycle",
        "bike",
        "motorcycle",
        "horse",
        "bird",
        "chair",
        "table",
        "cup",
        "bottle",
        "book",
        "phone",
        "laptop",
        "bag",
        "ball",
        "food",
        "pizza",
        "cake",
        "bed",
        "clock",
        "umbrella",
    }
    for row in rows:
        match = re.match(r"is there a[n]?\s+(.+?)\s+in the image\??$", str(row.get("question", "")).strip().lower())
        if match:
            phrase = match.group(1).strip()
            vocab.add(phrase)
            for part in re.split(r"[\s_/.-]+", phrase):
                if part:
                    vocab.add(part)
    return vocab


def _semantic_category(clean: str, object_vocab: set[str]) -> str:
    token = clean.strip().lower()
    if not token:
        return "empty"
    if token in {"yes", "no"}:
        return "yes_no"
    if token in {"not", "never", "none", "nothing", "without", "absent", "missing", "cannot", "can't", "n't"}:
        return "negation_absence"
    if token in {"image", "picture", "photo", "scene", "visible", "shown", "see", "seen", "look", "appears", "background"}:
        return "evidence_visual"
    if token in {"left", "right", "front", "behind", "above", "below", "under", "over", "near", "inside", "outside", "between", "around"}:
        return "spatial"
    if token in {"red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey", "orange", "purple", "pink"}:
        return "attribute_color"
    if token in {"small", "large", "big", "little", "tall", "short", "old", "young", "wooden", "metal", "plastic"}:
        return "attribute_other"
    if token.isdigit() or token in {"one", "two", "three", "four", "five", "many", "several", "few"}:
        return "counting"
    if token in object_vocab:
        return "object"
    if not any(ch.isalpha() for ch in token):
        return "punctuation_symbol"
    if len(token) <= 2:
        return "function_short"
    return "other"


def _cluster_summary(token_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in token_rows:
        key = (row["object"], row["side"], row["semantic_category"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"object": key[0], "side": key[1], "semantic_category": key[2], "count": count}
        for key, count in sorted(counts.items())
    ]


def _sample_level_semantics(
    sample_specs: list[dict[str, Any]],
    hidden_states_dir: str | Path,
    prediction_rows: list[dict[str, Any]],
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id = {str(row["sample_id"]): row for row in prediction_rows}
    sample_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    specs_by_layer: dict[int, list[dict[str, Any]]] = {}
    for spec in sample_specs:
        specs_by_layer.setdefault(int(spec["layer"]), []).append(spec)

    for layer, layer_specs in tqdm(sorted(specs_by_layer.items()), desc="Stage G sample projections", unit="layer"):
        hidden = load_hidden_layer(hidden_states_dir, layer)
        sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
        diff = hidden["z_blind"].float() - hidden["z_img"].float()
        sample_metadata = [rows_by_id.get(sample_id, {}) for sample_id in sample_ids]

        for spec in layer_specs:
            if spec["kind"] == "signed":
                vector = spec["vector"].float()
                vector = vector / torch.clamp(torch.linalg.norm(vector), min=1e-12)
                scores = (diff @ vector).cpu().numpy()
                sample_rows.extend(
                    _extreme_sample_rows(
                        spec=spec,
                        sample_ids=sample_ids,
                        sample_metadata=sample_metadata,
                        scores=scores,
                        side="positive",
                        order=np.argsort(-scores),
                        top_n=top_n,
                    )
                )
                sample_rows.extend(
                    _extreme_sample_rows(
                        spec=spec,
                        sample_ids=sample_ids,
                        sample_metadata=sample_metadata,
                        scores=scores,
                        side="negative",
                        order=np.argsort(scores),
                        top_n=top_n,
                    )
                )
            elif spec["kind"] == "subspace_energy":
                basis = torch.linalg.qr(spec["basis"].float(), mode="reduced").Q
                scores = torch.linalg.norm(diff @ basis, dim=1).cpu().numpy()
                sample_rows.extend(
                    _extreme_sample_rows(
                        spec=spec,
                        sample_ids=sample_ids,
                        sample_metadata=sample_metadata,
                        scores=scores,
                        side="energy",
                        order=np.argsort(-scores),
                        top_n=top_n,
                    )
                )
            else:
                raise ValueError(f"Unknown sample semantic spec kind: {spec['kind']}")
            contrast_rows.extend(_outcome_contrast_rows(spec, sample_metadata, scores))
    return sample_rows, contrast_rows


def _extreme_sample_rows(
    spec: dict[str, Any],
    sample_ids: list[str],
    sample_metadata: list[dict[str, Any]],
    scores: np.ndarray,
    side: str,
    order: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    for rank, idx in enumerate(order[:top_n], start=1):
        meta = sample_metadata[int(idx)]
        centered = float(scores[int(idx)] - mean)
        rows.append(
            {
                "object": spec["name"],
                "family": spec["family"],
                "layer": spec["layer"],
                "kind": spec["kind"],
                "side": side,
                "rank": rank,
                "score": float(scores[int(idx)]),
                "centered_score": centered,
                "z_score": centered / std if std > 0 else 0.0,
                "sample_id": sample_ids[int(idx)],
                "subset": meta.get("subset", ""),
                "outcome": meta.get("outcome", ""),
                "label": meta.get("label", ""),
                "parsed_prediction": meta.get("parsed_prediction", ""),
                "question": meta.get("question", ""),
                "image": meta.get("image", ""),
                **spec.get("metadata", {}),
            }
        )
    return rows


def _outcome_contrast_rows(
    spec: dict[str, Any],
    sample_metadata: list[dict[str, Any]],
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = [str(meta.get("outcome", "")) for meta in sample_metadata]
    for outcome in ["TP", "TN", "FP", "FN"]:
        values = np.asarray([score for score, item in zip(scores, outcomes, strict=True) if item == outcome])
        if values.size == 0:
            continue
        rows.append(
            {
                "object": spec["name"],
                "family": spec["family"],
                "layer": spec["layer"],
                "kind": spec["kind"],
                "contrast": f"outcome_{outcome}",
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "cohen_d": "",
                "auc": "",
            }
        )

    pair_rows = []
    pair_rows.append(_pair_contrast(spec, scores, outcomes, "TN", "FP", "FP_vs_TN"))
    pair_rows.append(_pair_contrast(spec, scores, outcomes, "TP", "FN", "FN_vs_TP"))
    pair_rows.append(_pair_contrast(spec, scores, outcomes, "TN", "TP", "TN_vs_TP"))
    rows.extend([row for row in pair_rows if row])
    return rows


def _pair_contrast(
    spec: dict[str, Any],
    scores: np.ndarray,
    outcomes: list[str],
    negative_outcome: str,
    positive_outcome: str,
    contrast: str,
) -> dict[str, Any] | None:
    negative = np.asarray([score for score, item in zip(scores, outcomes, strict=True) if item == negative_outcome])
    positive = np.asarray([score for score, item in zip(scores, outcomes, strict=True) if item == positive_outcome])
    if negative.size == 0 or positive.size == 0:
        return None
    combined = np.concatenate([negative, positive])
    labels = np.concatenate([np.zeros(negative.size), np.ones(positive.size)])
    pooled_std = math.sqrt(
        ((negative.size - 1) * float(np.var(negative, ddof=1)) + (positive.size - 1) * float(np.var(positive, ddof=1)))
        / max(negative.size + positive.size - 2, 1)
    )
    cohen_d = (float(np.mean(positive)) - float(np.mean(negative))) / pooled_std if pooled_std > 0 else 0.0
    try:
        auc = float(roc_auc_score(labels, combined))
    except ValueError:
        auc = float("nan")
    return {
        "object": spec["name"],
        "family": spec["family"],
        "layer": spec["layer"],
        "kind": spec["kind"],
        "contrast": contrast,
        "n": int(combined.size),
        "mean": float(np.mean(positive) - np.mean(negative)),
        "median": float(np.median(positive) - np.median(negative)),
        "std": "",
        "q25": "",
        "q75": "",
        "cohen_d": float(cohen_d),
        "auc": auc,
    }


def _summarize_object(name: str, family: str, layer: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_positive = _tokens_for(rows, "positive", 8)
    top_negative = _tokens_for(rows, "negative", 8)
    top_energy = _tokens_for(rows, "energy", 12)
    categories: dict[str, int] = {}
    for row in rows:
        categories[row["semantic_category"]] = categories.get(row["semantic_category"], 0) + 1
    category_text = "; ".join(f"{key}:{value}" for key, value in sorted(categories.items()))
    return {
        "object": name,
        "family": family,
        "layer": layer,
        "top_positive": ", ".join(top_positive),
        "top_negative": ", ".join(top_negative),
        "top_energy": ", ".join(top_energy),
        "category_counts": category_text,
    }


def _tokens_for(rows: list[dict[str, Any]], side: str, limit: int) -> list[str]:
    return [
        row["clean_token"]
        for row in sorted((r for r in rows if r["side"] == side), key=lambda item: item["rank"])
    ][:limit]


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _write_markdown_summary(
    output_dir: Path,
    object_summaries: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> Path:
    path = output_dir / "semantic_projection_summary.md"
    lines = [
        "# Stage G Semantic Projection Summary",
        "",
        "This is a vocabulary-space projection, not a full mechanistic logit-lens proof.",
        "",
        "| Object | Family | Layer | Top positive / energy tokens | Top negative tokens | Category counts |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in object_summaries:
        top = row["top_positive"] or row["top_energy"]
        lines.append(
            f"| `{row['object']}` | `{row['family']}` | {row['layer']} | {top} | "
            f"{row['top_negative']} | {row['category_counts']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(f"- Token rows: `{len(token_rows)}`")
    lines.append(f"- Sample extreme rows: `{len(sample_rows)}`")
    lines.append("- Signed directions list positive and negative vocabulary projections.")
    lines.append("- Subspace slices list high-energy vocabulary projections.")
    lines.extend(["", "## Strongest TN/FP Sample Contrasts", ""])
    lines.append("| Object | Family | Layer | Kind | Contrast | Mean diff | Cohen d | AUC |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: |")
    pair_rows = [row for row in contrast_rows if row["contrast"] == "FP_vs_TN"]
    pair_rows = sorted(pair_rows, key=lambda row: abs(float(row["cohen_d"])), reverse=True)[:12]
    for row in pair_rows:
        lines.append(
            f"| `{row['object']}` | `{row['family']}` | {row['layer']} | `{row['kind']}` | "
            f"`{row['contrast']}` | {float(row['mean']):.4f} | {float(row['cohen_d']):.3f} | "
            f"{float(row['auc']):.3f} |"
        )
    lines.extend(["", "## Representative Extreme Samples", ""])
    lines.append("| Object | Side | Rank | Score | Sample | Outcome | Question |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
    for row in sample_rows[:40]:
        question = str(row["question"]).replace("|", "\\|")
        if len(question) > 90:
            question = question[:87] + "..."
        display_score = row.get("centered_score", row["score"])
        lines.append(
            f"| `{row['object']}` | `{row['side']}` | {row['rank']} | {float(display_score):.4f} | "
            f"`{row['sample_id']}` | `{row['outcome']}` | {question} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
