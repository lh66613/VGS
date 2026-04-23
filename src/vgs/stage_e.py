"""Stage E causal intervention helpers."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from vgs.artifacts import load_hidden_layer, load_svd, read_jsonl
from vgs.io import ensure_dir, write_csv
from vgs.llava_hf import build_pope_prompt, _move_inputs
from vgs.pope import classify_outcome, parse_yes_no


InterventionFn = Callable[[torch.Tensor], torch.Tensor]


def run_intervention_precheck(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    layers: list[int],
    device: str,
    output_dir: str | Path,
    seed: int,
    max_new_tokens: int,
    random_scale: float,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    torch.manual_seed(seed)
    sample = rows[0]
    baseline_text = _generate_with_optional_intervention(
        model,
        processor,
        sample,
        device,
        max_new_tokens=max_new_tokens,
        layer=None,
        intervention=None,
    )
    precheck_rows = []
    module_names = _layer_module_names(model, layers)
    for layer in tqdm(layers, desc="intervention precheck layers", unit="layer"):
        hidden_dim = int(model.config.text_config.hidden_size)
        random_direction = torch.randn(hidden_dim, device=device)
        random_direction = random_direction / torch.clamp(torch.linalg.norm(random_direction), min=1e-12)

        baseline_logits = _next_token_logits_with_optional_intervention(
            model,
            processor,
            sample,
            device,
            layer=None,
            intervention=None,
        )
        noop_logits = _next_token_logits_with_optional_intervention(
            model,
            processor,
            sample,
            device,
            layer=layer,
            intervention=lambda h: h,
        )
        random_logits = _next_token_logits_with_optional_intervention(
            model,
            processor,
            sample,
            device,
            layer=layer,
            intervention=lambda h, direction=random_direction: h + random_scale * direction.to(dtype=h.dtype),
        )
        noop_text = _generate_with_optional_intervention(
            model,
            processor,
            sample,
            device,
            max_new_tokens=max_new_tokens,
            layer=layer,
            intervention=lambda h: h,
        )
        random_text = _generate_with_optional_intervention(
            model,
            processor,
            sample,
            device,
            max_new_tokens=max_new_tokens,
            layer=layer,
            intervention=lambda h, direction=random_direction: h + random_scale * direction.to(dtype=h.dtype),
        )
        yes_ids = _candidate_token_ids(processor.tokenizer, ["Yes", " yes", "yes"])
        no_ids = _candidate_token_ids(processor.tokenizer, ["No", " no", "no"])
        precheck_rows.append(
            {
                "layer": layer,
                "hook_module": module_names.get(layer, ""),
                "baseline_text": baseline_text,
                "noop_text": noop_text,
                "random_perturb_text": random_text,
                "noop_equal_to_baseline": noop_text == baseline_text,
                "random_changed_text": random_text != baseline_text,
                "random_scale": random_scale,
                "noop_max_abs_logit_delta": _max_abs_delta(baseline_logits, noop_logits),
                "random_max_abs_logit_delta": _max_abs_delta(baseline_logits, random_logits),
                "baseline_top_token": _top_decoded_token(processor, baseline_logits),
                "random_top_token": _top_decoded_token(processor, random_logits),
                "baseline_yes_logit": _max_token_logit(baseline_logits, yes_ids),
                "baseline_no_logit": _max_token_logit(baseline_logits, no_ids),
                "random_yes_logit": _max_token_logit(random_logits, yes_ids),
                "random_no_logit": _max_token_logit(random_logits, no_ids),
            }
        )
    path = write_csv(
        Path(output_dir) / "intervention_precheck.csv",
        precheck_rows,
        list(precheck_rows[0].keys()) if precheck_rows else [],
    )
    return {
        "precheck_path": str(path),
        "num_rows": len(precheck_rows),
        "baseline_sample_id": str(sample.get("sample_id")),
        "baseline_text": baseline_text,
        "hook_strategy": "register_forward_hook on model.language_model.model.layers[layer-1]",
        "random_scale": random_scale,
        "readout_position": "last prompt token during initial full-prompt forward; all positions during decode steps are left untouched",
    }


def run_intervention_pilot(
    model: Any,
    processor: Any,
    prediction_rows: list[dict[str, Any]],
    layers: list[int],
    device: str,
    output_dir: str | Path,
    svd_dir: str | Path,
    hidden_states_dir: str | Path,
    seed: int,
    max_new_tokens: int,
    max_samples_per_outcome: int,
    alpha_grid: list[float],
    tail_band: tuple[int, int],
    outcomes: list[str],
    families: list[str],
    granularities: list[str],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    selected = _select_pilot_rows(prediction_rows, rng, max_samples_per_outcome, outcomes)
    rows = []
    for layer in tqdm(layers, desc="intervention pilot layers", unit="layer"):
        vectors = _intervention_vectors(layer, svd_dir, hidden_states_dir, prediction_rows, tail_band, seed)
        for sample in tqdm(selected, desc=f"L{layer} intervention samples", unit="sample", leave=False):
            baseline_text = _generate_with_optional_intervention(
                model,
                processor,
                sample,
                device,
                max_new_tokens=max_new_tokens,
                layer=None,
                intervention=None,
            )
            baseline_logits = _next_token_logits_with_optional_intervention(
                model,
                processor,
                sample,
                device,
                layer=None,
                intervention=None,
            )
            baseline_pred = parse_yes_no(baseline_text)
            rows.append(
                _pilot_row(
                    layer,
                    sample,
                    "baseline",
                    "none",
                    0.0,
                    baseline_text,
                    baseline_pred,
                    baseline_logits,
                    processor,
                )
            )
            for spec_name, vector in vectors.items():
                for alpha in alpha_grid:
                    if _skip_spec(spec_name, sample.get("outcome"), families):
                        continue
                    payload = _to_device_payload(vector, device)
                    for granularity in granularities:
                        intervention = _make_intervention(spec_name, payload, float(alpha))
                        logits = _next_token_logits_with_optional_intervention(
                            model,
                            processor,
                            sample,
                            device,
                            layer=layer,
                            intervention=intervention,
                            granularity=granularity,
                        )
                        text = _generate_with_optional_intervention(
                            model,
                            processor,
                            sample,
                            device,
                            max_new_tokens=max_new_tokens,
                            layer=layer,
                            intervention=intervention,
                            granularity=granularity,
                        )
                        pred = parse_yes_no(text)
                        rows.append(
                            _pilot_row(
                                layer,
                                sample,
                                spec_name,
                                granularity,
                                alpha,
                                text,
                                pred,
                                logits,
                                processor,
                            )
                        )
    path = write_csv(
        Path(output_dir) / "intervention_pilot_results.csv",
        rows,
        list(rows[0].keys()) if rows else [],
    )
    summary_rows = _intervention_summary(rows)
    summary_path = write_csv(
        Path(output_dir) / "intervention_pilot_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    return {
        "results_path": str(path),
        "summary_path": str(summary_path),
        "num_result_rows": len(rows),
        "num_summary_rows": len(summary_rows),
        "tail_band": f"{tail_band[0]}-{tail_band[1]}",
        "alpha_grid": alpha_grid,
        "outcomes": outcomes,
        "families": families,
        "granularities": granularities,
    }


def _layer_module_names(model: Any, layers: list[int]) -> dict[int, str]:
    modules = dict(model.named_modules())
    names = {}
    for layer in layers:
        name = f"language_model.model.layers.{layer - 1}"
        names[layer] = name if name in modules else ""
    return names


def _layer_module(model: Any, layer: int) -> torch.nn.Module:
    return model.language_model.model.layers[layer - 1]


@contextmanager
def _activation_hook(model: Any, layer: int, intervention: InterventionFn, granularity: str):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            return output
        changed = hidden.clone()
        if granularity == "last_token":
            if hidden.shape[1] <= 1:
                return output
            changed[:, -1, :] = intervention(changed[:, -1, :])
        elif granularity == "full_sequence":
            changed = intervention(changed)
        elif granularity == "generated_token":
            if hidden.shape[1] != 1:
                return output
            changed[:, -1, :] = intervention(changed[:, -1, :])
        else:
            raise ValueError(f"Unknown intervention granularity: {granularity}")
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    handle = _layer_module(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def _generate_with_optional_intervention(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: str,
    max_new_tokens: int,
    layer: int | None,
    intervention: InterventionFn | None,
    granularity: str = "last_token",
) -> str:
    image = Image.open(Path(sample["image_path"])).convert("RGB")
    prompt = build_pope_prompt(processor, sample["question"])
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = _move_inputs(inputs, device, dtype=next(model.parameters()).dtype)
    hook_context = _activation_hook(model, layer, intervention, granularity) if layer and intervention else _nullcontext()
    with hook_context:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def _next_token_logits_with_optional_intervention(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: str,
    layer: int | None,
    intervention: InterventionFn | None,
    granularity: str = "last_token",
) -> torch.Tensor:
    image = Image.open(Path(sample["image_path"])).convert("RGB")
    prompt = build_pope_prompt(processor, sample["question"])
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = _move_inputs(inputs, device, dtype=next(model.parameters()).dtype)
    hook_context = _activation_hook(model, layer, intervention, granularity) if layer and intervention else _nullcontext()
    with hook_context:
        outputs = model(
            **inputs,
            return_dict=True,
            use_cache=False,
        )
    return outputs.logits[0, -1].detach().float().cpu()


def _candidate_token_ids(tokenizer: Any, candidates: list[str]) -> list[int]:
    token_ids = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            token_ids.append(int(ids[-1]))
    return sorted(set(token_ids))


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _max_token_logit(logits: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return math.nan
    return float(torch.max(logits[token_ids]).item())


def _top_decoded_token(processor: Any, logits: torch.Tensor) -> str:
    token_id = int(torch.argmax(logits).item())
    return processor.decode([token_id], skip_special_tokens=False)


@contextmanager
def _nullcontext():
    yield


def _select_pilot_rows(
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    max_samples_per_outcome: int,
    outcomes: list[str],
) -> list[dict[str, Any]]:
    selected = []
    for outcome in outcomes:
        group = [row for row in rows if row.get("outcome") == outcome]
        rng.shuffle(group)
        selected.extend(group[:max_samples_per_outcome])
    return selected


def _intervention_vectors(
    layer: int,
    svd_dir: str | Path,
    hidden_states_dir: str | Path,
    prediction_rows: list[dict[str, Any]],
    tail_band: tuple[int, int],
    seed: int,
) -> dict[str, np.ndarray]:
    basis = load_svd(svd_dir, layer)["Vh"].float().numpy().T
    start, end = tail_band
    band = basis[:, start - 1 : end]
    random_band = _random_basis(basis.shape[0], band.shape[1], seed + layer)
    orthogonal_random = _orthogonal_random_basis(basis.shape[0], band, band.shape[1], seed + layer + 101)
    vectors = {
        "ablate_tail_257_1024": {"mode": "projection", "basis": band},
        "random_tail_control": {"mode": "projection", "basis": random_band},
        "orthogonal_tail_control": {"mode": "projection", "basis": orthogonal_random},
        "norm_matched_random_tail_control": {
            "mode": "norm_matched_projection",
            "basis": orthogonal_random,
            "target_basis": band,
        },
    }
    labels_by_id = {}
    outcomes_by_id = {}
    for row in prediction_rows:
        sample_id = str(row["sample_id"])
        outcomes_by_id[sample_id] = row.get("outcome")
        if row.get("outcome") == "FP":
            labels_by_id[sample_id] = 1
        elif row.get("outcome") == "TN":
            labels_by_id[sample_id] = 0
    hidden = load_hidden_layer(hidden_states_dir, layer)
    sample_ids = [str(sample_id) for sample_id in hidden["sample_ids"]]
    keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels_by_id]
    y = np.array([labels_by_id[sample_ids[idx]] for idx in keep], dtype=np.int64)
    diff = hidden["z_blind"][keep].float().numpy() - hidden["z_img"][keep].float().numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(diff)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    logistic.fit(x_scaled, y)
    vectors["reduce_logistic_fp_score"] = {
        "mode": "signed_vector",
        "direction": _unit_vector(logistic.coef_[0] / np.maximum(scaler.scale_, 1e-12)),
        "sign": 1.0,
    }
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(x_scaled, y)
    vectors["reduce_lda_fp_score"] = {
        "mode": "signed_vector",
        "direction": _unit_vector(lda.coef_[0] / np.maximum(scaler.scale_, 1e-12)),
        "sign": 1.0,
    }
    tn_indices = [idx for idx, sample_id in enumerate(sample_ids) if outcomes_by_id.get(sample_id) == "TN"]
    fp_indices = [idx for idx, sample_id in enumerate(sample_ids) if outcomes_by_id.get(sample_id) == "FP"]
    z_blind = hidden["z_blind"].float().numpy()
    z_img = hidden["z_img"].float().numpy()
    if tn_indices:
        tn_d = (z_blind[tn_indices] - z_img[tn_indices]).mean(axis=0)
        vectors["add_tn_correction"] = {
            "mode": "signed_vector",
            "direction": _unit_vector(tn_d),
            "sign": -1.0,
        }
    if fp_indices:
        fp_d = (z_blind[fp_indices] - z_img[fp_indices]).mean(axis=0)
        vectors["subtract_fp_shift"] = {
            "mode": "signed_vector",
            "direction": _unit_vector(fp_d),
            "sign": 1.0,
        }
    return vectors


def _to_device_payload(payload: Any, device: str) -> Any:
    if isinstance(payload, dict):
        return {
            key: _to_device_payload(value, device)
            for key, value in payload.items()
        }
    if isinstance(payload, np.ndarray):
        return torch.as_tensor(payload, device=device)
    return payload


def _make_intervention(spec_name: str, payload: Any, alpha: float) -> InterventionFn:
    if isinstance(payload, dict) and payload.get("mode") == "projection":
        basis = payload["basis"]

        def intervention(hidden: torch.Tensor) -> torch.Tensor:
            basis_cast = basis.to(device=hidden.device, dtype=hidden.dtype)
            projection = (hidden @ basis_cast) @ basis_cast.T
            return hidden - alpha * projection

        return intervention

    if isinstance(payload, dict) and payload.get("mode") == "norm_matched_projection":
        basis = payload["basis"]
        target_basis = payload["target_basis"]

        def intervention(hidden: torch.Tensor) -> torch.Tensor:
            basis_cast = basis.to(device=hidden.device, dtype=hidden.dtype)
            target_cast = target_basis.to(device=hidden.device, dtype=hidden.dtype)
            control_projection = (hidden @ basis_cast) @ basis_cast.T
            target_projection = (hidden @ target_cast) @ target_cast.T
            control_norm = torch.clamp(torch.linalg.norm(control_projection, dim=-1, keepdim=True), min=1e-6)
            target_norm = torch.linalg.norm(target_projection, dim=-1, keepdim=True)
            scaled_control = control_projection * (target_norm / control_norm)
            return hidden - alpha * scaled_control

        return intervention

    if isinstance(payload, dict) and payload.get("mode") == "signed_vector":
        direction = payload["direction"]
        sign = float(payload.get("sign", 1.0))

        def intervention(hidden: torch.Tensor) -> torch.Tensor:
            vector = direction.to(device=hidden.device, dtype=hidden.dtype)
            vector = vector / torch.clamp(torch.linalg.norm(vector), min=1e-12)
            return hidden + sign * alpha * vector

        return intervention

    direction = payload

    def intervention(hidden: torch.Tensor) -> torch.Tensor:
        vector = direction.to(device=hidden.device, dtype=hidden.dtype)
        vector = vector / torch.clamp(torch.linalg.norm(vector), min=1e-12)
        return hidden - alpha * vector

    return intervention


def _skip_spec(spec_name: str, outcome: str | None, families: list[str]) -> bool:
    tail_specs = {
        "ablate_tail_257_1024",
        "random_tail_control",
        "orthogonal_tail_control",
        "norm_matched_random_tail_control",
    }
    rescue_specs = {
        "reduce_logistic_fp_score",
        "reduce_lda_fp_score",
        "add_tn_correction",
        "subtract_fp_shift",
    }
    if spec_name in tail_specs and "tail" not in families:
        return True
    if spec_name in rescue_specs and "rescue" not in families:
        return True
    if outcome == "TN" and spec_name in rescue_specs:
        return True
    if outcome == "FP" and spec_name in tail_specs:
        return True
    return False


def _pilot_row(
    layer: int,
    sample: dict[str, Any],
    intervention: str,
    granularity: str,
    alpha: float,
    text: str,
    pred: str | None,
    logits: torch.Tensor,
    processor: Any,
) -> dict[str, Any]:
    yes_ids = _candidate_token_ids(processor.tokenizer, ["Yes", " yes", "yes"])
    no_ids = _candidate_token_ids(processor.tokenizer, ["No", " no", "no"])
    yes_logit = _max_token_logit(logits, yes_ids)
    no_logit = _max_token_logit(logits, no_ids)
    return {
        "layer": layer,
        "sample_id": sample.get("sample_id"),
        "outcome_before": sample.get("outcome"),
        "label": sample.get("label"),
        "intervention": intervention,
        "granularity": granularity,
        "alpha": alpha,
        "raw_generation": text,
        "parsed_prediction": pred,
        "outcome_after": classify_outcome(pred, str(sample.get("label")).lower()),
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_minus_no_logit": yes_logit - no_logit,
        "top_token": _top_decoded_token(processor, logits),
    }


def _intervention_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df_rows = []
    baseline_margin = {
        (row["layer"], row["sample_id"]): row.get("yes_minus_no_logit")
        for row in rows
        if row["intervention"] == "baseline"
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["layer"], row["outcome_before"], row["intervention"], row["granularity"], row["alpha"])
        groups.setdefault(key, []).append(row)
    for (layer, outcome_before, intervention, granularity, alpha), group in groups.items():
        valid = [row for row in group if row["outcome_after"] in {"TP", "TN", "FP", "FN"}]
        correct = [row for row in valid if row["outcome_after"] in {"TP", "TN"}]
        fp = [row for row in valid if row["outcome_after"] == "FP"]
        margins = [row.get("yes_minus_no_logit") for row in group if row.get("yes_minus_no_logit") is not None]
        deltas = []
        for row in group:
            key = (row["layer"], row["sample_id"])
            if key in baseline_margin and row.get("yes_minus_no_logit") is not None:
                deltas.append(row["yes_minus_no_logit"] - baseline_margin[key])
        df_rows.append(
            {
                "layer": layer,
                "outcome_before": outcome_before,
                "intervention": intervention,
                "granularity": granularity,
                "alpha": alpha,
                "n": len(group),
                "accuracy": len(correct) / len(valid) if valid else math.nan,
                "fp_rate": len(fp) / len(valid) if valid else math.nan,
                "unknown_rate": (len(group) - len(valid)) / len(group) if group else math.nan,
                "mean_yes_minus_no_logit": float(np.mean(margins)) if margins else math.nan,
                "median_yes_minus_no_logit": float(np.median(margins)) if margins else math.nan,
                "mean_margin_delta_vs_baseline": float(np.mean(deltas)) if deltas else math.nan,
            }
        )
    return df_rows


def _random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q[:, :k]


def _orthogonal_random_basis(dim: int, forbidden_basis: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(dim, k))
    forbidden = np.asarray(forbidden_basis, dtype=np.float64)
    random_matrix = random_matrix - forbidden @ (forbidden.T @ random_matrix)
    q, _ = np.linalg.qr(random_matrix)
    return q[:, :k]


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm
