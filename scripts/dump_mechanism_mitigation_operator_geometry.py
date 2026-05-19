#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import math
import re
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import load_svd, read_jsonl
from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.config import config_get, load_config
from vgs.io import append_experiment_log, write_csv, write_json
from vgs.llava_hf import build_blind_prompt, build_pope_prompt, _move_inputs, _readout_hidden_state
from vgs.vcd import _degrade_image, add_diffusion_noise
from vgs.vlm_hf import MODEL_FAMILIES, candidate_token_ids, load_vlm_hf, max_token_logit


YES_CANDIDATES = ["yes", "Yes", " yes", " Yes", "YES"]
NO_CANDIDATES = ["no", "No", " no", " No", "NO"]
OPERATORS = ["icd_blind", "vcd_diffusion", "vcd_blur", "vcd_gray"]
DEFAULT_SUBSPACES = [
    "full",
    "top4",
    "top16",
    "band5_16",
    "band17_64",
    "band65_256",
    "tail257_1024",
    "top4_complement",
    "random12",
    "random4_complement",
    "random_tail_dim",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dump operator-level correction geometry for the mitigation plan. "
            "The script stores energy fractions and yes/no logit-margin "
            "contributions after projecting h_orig - h_neg into the existing "
            "blind-reference SVD spectrum."
        )
    )
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["24"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-family", choices=sorted(MODEL_FAMILIES), default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=["icd_blind", "vcd_diffusion"])
    parser.add_argument("--subspaces", nargs="+", default=DEFAULT_SUBSPACES)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--readout-position", default="last_prompt_token")
    parser.add_argument("--blur-radius", type=float, default=5.0)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--yes-token-ids", nargs="+", type=int, default=None)
    parser.add_argument("--no-token-ids", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/mechanism_mitigation/operator_geometry")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    payload: dict[str, Any] = {
        "model_path": model_path,
        "model_family": args.model_family,
        "predictions": args.predictions,
        "svd_dir": args.svd_dir,
        "layers": layers,
        "operators": args.operators,
        "subspaces": args.subspaces,
        "max_samples": args.max_samples,
        "readout_position": args.readout_position,
        "blur_radius": args.blur_radius,
        "noise_step": args.noise_step,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        summary_path = write_json(Path(args.output_dir) / "operator_geometry_summary.json", payload)
        append_experiment_log(args.log_path, "dump_mechanism_mitigation_operator_geometry", summary_path, "dry_run")
        print(summary_path)
        return

    bundle = load_vlm_hf(
        model_path,
        model_family=args.model_family,
        device=args.device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
        trust_remote_code=args.trust_remote_code,
    )
    if bundle.family != "llava":
        raise NotImplementedError("Operator hidden-geometry dumping currently supports the HF LLaVA adapter.")
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.predictions)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    yes_ids = sorted(set(args.yes_token_ids or candidate_token_ids(bundle.tokenizer, YES_CANDIDATES)))
    no_ids = sorted(set(args.no_token_ids or candidate_token_ids(bundle.tokenizer, NO_CANDIDATES)))
    if not yes_ids or not no_ids:
        raise ValueError("Could not resolve yes/no candidate token IDs.")
    yes_weight, no_weight = _selected_lm_head_weights(bundle.model, yes_ids, no_ids)

    bases_by_layer = {
        layer: _subspace_bases(load_svd(args.svd_dir, layer)["Vh"].float().numpy(), args.subspaces, args.seed + layer)
        for layer in layers
    }

    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="operator correction geometry", unit="sample"):
        orig = _forward_llava_condition(
            bundle.model,
            bundle.processor,
            row["question"],
            row["image_path"],
            layers,
            bundle.device,
            args.readout_position,
        )
        for operator in args.operators:
            neg = _forward_llava_condition(
                bundle.model,
                bundle.processor,
                row["question"],
                row["image_path"],
                layers,
                bundle.device,
                args.readout_position,
                operator=operator,
                blur_radius=args.blur_radius,
                noise_step=args.noise_step,
            )
            for layer in layers:
                delta = orig["hidden"][layer] - neg["hidden"][layer]
                delta_norm_sq = float(torch.dot(delta, delta).item())
                out_row = {
                    "sample_id": str(row["sample_id"]),
                    "source_subset": row.get("subset", ""),
                    "benchmark": row.get("benchmark", ""),
                    "dimension": row.get("dimension", ""),
                    "annotation_type": row.get("annotation_type", ""),
                    "label": row.get("label", ""),
                    "outcome": row.get("outcome", ""),
                    "parsed_prediction": row.get("parsed_prediction", ""),
                    "question": row.get("question", ""),
                    "image": row.get("image", ""),
                    "image_path": row.get("image_path", ""),
                    "operator": operator,
                    "layer": layer,
                    "delta_norm_sq": delta_norm_sq,
                    "orig_yes_minus_no_logit": _yes_minus_no(orig["logits"], yes_ids, no_ids),
                    "neg_yes_minus_no_logit": _yes_minus_no(neg["logits"], yes_ids, no_ids),
                    "orig_no_minus_yes_logit": _no_minus_yes(orig["logits"], yes_ids, no_ids),
                    "neg_no_minus_yes_logit": _no_minus_yes(neg["logits"], yes_ids, no_ids),
                }
                for name, projector in bases_by_layer[layer].items():
                    component = projector(delta)
                    energy = float(torch.dot(component, component).item())
                    yes_logit, no_logit = _yes_no_logit_contribution(
                        component,
                        yes_weight,
                        no_weight,
                    )
                    out_row[f"energy_{name}"] = energy
                    out_row[f"energy_frac_{name}"] = energy / delta_norm_sq if delta_norm_sq > 0 else math.nan
                    out_row[f"dlogit_yes_{name}"] = yes_logit
                    out_row[f"dlogit_no_{name}"] = no_logit
                    out_row[f"dmargin_no_minus_yes_{name}"] = no_logit - yes_logit
                out_rows.append(out_row)

    output_root = Path(args.output_dir)
    geometry_path = write_csv(output_root / "operator_geometry.csv", out_rows, _fieldnames(out_rows))
    payload.update(
        {
            "resolved_device": bundle.device,
            "resolved_model_family": bundle.family,
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
            "num_rows": len(out_rows),
            "operator_geometry_path": str(geometry_path),
        }
    )
    summary_path = write_json(output_root / "operator_geometry_summary.json", payload)
    append_experiment_log(args.log_path, "dump_mechanism_mitigation_operator_geometry", summary_path, "ok")
    print(summary_path)


def _forward_llava_condition(
    model: Any,
    processor: Any,
    question: str,
    image_path: str,
    layers: list[int],
    device: str,
    readout_position: str,
    operator: str = "orig",
    blur_radius: float = 5.0,
    noise_step: int = 500,
) -> dict[str, Any]:
    if operator == "icd_blind":
        prompt = build_blind_prompt(question)
        inputs = processor.tokenizer(prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, device)
        outputs = model.language_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    else:
        image = Image.open(Path(image_path)).convert("RGB")
        if operator == "vcd_blur":
            image = _degrade_image(image, "blur", blur_radius)
        elif operator == "vcd_gray":
            image = _degrade_image(image, "gray", blur_radius)
        elif operator not in {"orig", "vcd_diffusion"}:
            raise ValueError(f"Unsupported operator: {operator}")
        prompt = build_pope_prompt(processor, question)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, device, dtype=next(model.parameters()).dtype)
        if operator == "vcd_diffusion":
            inputs = dict(inputs)
            inputs["pixel_values"] = add_diffusion_noise(inputs["pixel_values"], noise_step=noise_step)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    index = int(inputs["attention_mask"][0].sum().item()) - 1
    return {
        "hidden": {
            layer: _readout_hidden_state(outputs.hidden_states[layer][0], index, readout_position)
            for layer in layers
        },
        "logits": _next_token_logits(model, outputs),
    }


def _selected_lm_head_weights(model: Any, yes_ids: list[int], no_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    lm_head = _lm_head(model)
    if not hasattr(lm_head, "weight"):
        raise ValueError("Could not locate an lm_head.weight on the LLaVA model.")
    weight = lm_head.weight.detach().float().cpu()
    return weight[yes_ids], weight[no_ids]


def _next_token_logits(model: Any, outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits[0, -1].detach().float().cpu()
    lm_head = _lm_head(model)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states):
        last_hidden = hidden_states[-1][0, -1]
    else:
        last_hidden = outputs.last_hidden_state[0, -1]
    return lm_head(last_hidden).detach().float().cpu()


def _lm_head(model: Any) -> Any:
    lm_head = getattr(getattr(model, "language_model", model), "lm_head", None)
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            lm_head = get_output_embeddings()
    if lm_head is None:
        raise ValueError("Could not locate an lm_head/output embeddings module.")
    return lm_head


def _subspace_bases(vh: np.ndarray, names: list[str], seed: int) -> dict[str, Any]:
    basis = torch.from_numpy(vh.T.astype(np.float32, copy=False))
    hidden_dim = basis.shape[0]
    tail_dim = max(1, min(hidden_dim, basis.shape[1]) - 256)
    projectors: dict[str, Any] = {}
    for name in names:
        if name == "full":
            projectors[name] = lambda x: x
        elif name == "top4":
            projectors[name] = _basis_projector(basis[:, : min(4, basis.shape[1])])
        elif name == "top16":
            projectors[name] = _basis_projector(basis[:, : min(16, basis.shape[1])])
        elif name == "band5_16":
            projectors[name] = _basis_projector(basis[:, 4 : min(16, basis.shape[1])])
        elif name == "band17_64":
            projectors[name] = _basis_projector(basis[:, 16 : min(64, basis.shape[1])])
        elif name == "band65_256":
            projectors[name] = _basis_projector(basis[:, 64 : min(256, basis.shape[1])])
        elif name == "tail257_1024":
            projectors[name] = _basis_projector(basis[:, 256 : min(1024, basis.shape[1])])
        elif name == "top4_complement":
            top4 = basis[:, : min(4, basis.shape[1])]
            top4_project = _basis_projector(top4)
            projectors[name] = lambda x, p=top4_project: x - p(x)
        elif name.startswith("random12"):
            random12_basis = _random_basis(
                hidden_dim,
                min(12, hidden_dim),
                np.random.default_rng(_stable_seed(seed, name)),
            )
            projectors[name] = _basis_projector(random12_basis)
        elif name.startswith("random4_complement"):
            random4_basis = _random_basis(
                hidden_dim,
                min(4, hidden_dim),
                np.random.default_rng(_stable_seed(seed, name)),
            )
            random4_project = _basis_projector(random4_basis)
            projectors[name] = lambda x, p=random4_project: x - p(x)
        elif name.startswith("random_tail_dim"):
            random_tail_basis = _random_basis(
                hidden_dim,
                min(tail_dim, hidden_dim),
                np.random.default_rng(_stable_seed(seed, name)),
            )
            projectors[name] = _basis_projector(random_tail_basis)
        elif match := re.fullmatch(r"band(\d+)_(\d+)", name):
            start, end = _one_indexed_interval(match.group(1), match.group(2), basis.shape[1])
            projectors[name] = _basis_projector(basis[:, start:end])
        elif match := re.fullmatch(r"v(\d+)", name):
            start, end = _one_indexed_interval(match.group(1), match.group(1), basis.shape[1])
            projectors[name] = _basis_projector(basis[:, start:end])
        elif match := re.fullmatch(r"band(\d+)_(\d+)_minus_v(\d+)", name):
            start, end = _one_indexed_interval(match.group(1), match.group(2), basis.shape[1])
            remove_start, remove_end = _one_indexed_interval(match.group(3), match.group(3), basis.shape[1])
            keep = [idx for idx in range(start, end) if not remove_start <= idx < remove_end]
            projectors[name] = _basis_projector(basis[:, keep])
        elif match := re.fullmatch(r"randcontig(\d+)_s(\d+)", name):
            width = max(1, int(match.group(1)))
            rng = np.random.default_rng(_stable_seed(seed, name))
            max_start = max(0, basis.shape[1] - width)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            end = min(basis.shape[1], start + width)
            projectors[name] = _basis_projector(basis[:, start:end])
        else:
            raise ValueError(f"Unsupported subspace name: {name}")
    return projectors


def _one_indexed_interval(start_text: str, end_text: str, max_dim: int) -> tuple[int, int]:
    start = int(start_text)
    end = int(end_text)
    if start < 1 or end < start:
        raise ValueError(f"Invalid 1-indexed subspace interval: {start_text}_{end_text}")
    start_idx = min(start - 1, max_dim)
    end_idx = min(end, max_dim)
    return start_idx, end_idx


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _random_basis(hidden_dim: int, dim: int, rng: np.random.Generator) -> torch.Tensor:
    random_matrix = rng.normal(size=(hidden_dim, dim)).astype(np.float32)
    random_q, _ = np.linalg.qr(random_matrix)
    return torch.from_numpy(random_q.astype(np.float32, copy=False))


def _basis_projector(basis: torch.Tensor) -> Any:
    if basis.numel() == 0:
        return lambda x: torch.zeros_like(x)
    basis = basis.float().cpu()
    return lambda x, b=basis: b @ (b.T @ x.float().cpu())


def _yes_minus_no(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    return float(max_token_logit(logits, yes_ids) - max_token_logit(logits, no_ids))


def _no_minus_yes(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    return -_yes_minus_no(logits, yes_ids, no_ids)


def _yes_no_logit_contribution(
    component: torch.Tensor,
    yes_weight: torch.Tensor,
    no_weight: torch.Tensor,
) -> tuple[float, float]:
    component = component.float().cpu()
    yes_value = float(torch.max(yes_weight @ component).item())
    no_value = float(torch.max(no_weight @ component).item())
    return yes_value, no_value


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
