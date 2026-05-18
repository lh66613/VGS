"""Visual Contrastive Decoding operators for Stage T selective correction.

The canonical `vcd_diffusion` path follows DAMO-NLP-SG/VCD: add diffusion
noise to the processed image tensor, contrast clean and distorted logits, apply
the adaptive plausibility cutoff, then decode from the resulting distribution.
The local implementation keeps the same token-level rule but avoids monkey
patching Transformers generation, which makes it stable for the HF LLaVA
adapter used in this project.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter, ImageOps

from vgs.datasets import PopeSample
from vgs.llava_hf import build_blind_prompt, build_pope_prompt, _move_inputs


OFFICIAL_VCD_REPOSITORY = "https://github.com/DAMO-NLP-SG/VCD"
OFFICIAL_VCD_SAMPLE = "vcd_utils/vcd_sample.py"
OFFICIAL_VCD_NOISE = "vcd_utils/vcd_add_noise.py"


def official_vcd_reference() -> dict[str, str]:
    """Return the upstream implementation reference used by this baseline."""

    return {
        "repository": OFFICIAL_VCD_REPOSITORY,
        "sampling_file": OFFICIAL_VCD_SAMPLE,
        "noise_file": OFFICIAL_VCD_NOISE,
        "local_adapter": "manual HF LLaVA decoding loop; no GenerationMixin monkey patch",
    }


@torch.inference_mode()
def generate_llava_contrastive_answer(
    model: Any,
    processor: Any,
    sample: PopeSample,
    device: str,
    max_new_tokens: int = 4,
    alpha: float = 1.0,
    beta: float = 0.1,
    contrast_source: str = "blur",
    blur_radius: float = 5.0,
    noise_step: int = 500,
    decode_strategy: str = "greedy",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Generate a yes/no answer with a VCD/ICD-style logit contrast.

    `contrast_source` controls the weaker reference distribution:
    - `diffusion`: official VCD-style diffusion noise on processed image tensors;
    - `blur`: same image after Gaussian blur, a lightweight VCD proxy;
    - `gray`: same image converted to grayscale, then back to RGB;
    - `blind`: text-only prompt through the LLaVA language model, an ICD proxy.

    The implementation recomputes the full prompt at each decoding step. That is
    slower than cached generation, but keeps the operator robust across
    transformers versions and is acceptable for the small fixed-split Stage T
    predicted-Yes pool.
    """

    image = Image.open(Path(sample.image_path)).convert("RGB")
    prompt = build_pope_prompt(processor, sample.question)
    visual_base = processor(images=image, text=prompt, return_tensors="pt")
    visual_base = _move_inputs(visual_base, device, dtype=next(model.parameters()).dtype)

    if contrast_source == "blind":
        contrast_base = processor.tokenizer(build_blind_prompt(sample.question), return_tensors="pt")
        contrast_base = _move_inputs(contrast_base, device)
        contrast_forward = model.language_model
    elif contrast_source == "diffusion":
        contrast_base = _with_diffusion_noise(visual_base, noise_step=noise_step)
        contrast_forward = model
    else:
        contrast_image = _degrade_image(image, contrast_source, blur_radius)
        contrast_base = processor(images=contrast_image, text=prompt, return_tensors="pt")
        contrast_base = _move_inputs(contrast_base, device, dtype=next(model.parameters()).dtype)
        contrast_forward = model

    generated: list[int] = []
    eos_ids = _eos_token_ids(processor.tokenizer)
    for _ in range(max_new_tokens):
        visual_inputs = _append_generated(visual_base, generated, device)
        contrast_inputs = _append_generated(contrast_base, generated, device)
        visual_outputs = model(**visual_inputs, return_dict=True, use_cache=False)
        contrast_outputs = contrast_forward(**contrast_inputs, return_dict=True, use_cache=False)
        logits = contrastive_logits(
            visual_outputs.logits[0, -1].float(),
            contrast_outputs.logits[0, -1].float(),
            alpha=alpha,
            beta=beta,
        )
        next_token_id = _select_next_token(
            logits,
            decode_strategy=decode_strategy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if next_token_id in eos_ids:
            break
        generated.append(next_token_id)

    return processor.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def contrastive_logits(
    visual_logits: torch.Tensor,
    contrast_logits: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """Combine visual and contrast logits with the official VCD APC mask.

    Official VCD computes `(1 + alpha) * logits_clean - alpha * logits_cd`
    and masks tokens whose clean-image logit falls below
    `max(logits_clean) + log(beta)`. This is equivalent to retaining tokens
    with clean-image probability at least `beta` times the maximum probability,
    but using the logit-space form mirrors the upstream implementation.
    """

    combined = (1.0 + alpha) * visual_logits - alpha * contrast_logits
    if beta > 0:
        beta_tensor = torch.tensor(beta, device=visual_logits.device, dtype=visual_logits.dtype)
        cutoff = torch.log(beta_tensor) + visual_logits.max(dim=-1, keepdim=True).values
        plausible = visual_logits >= cutoff
        if torch.any(plausible):
            combined = combined.masked_fill(~plausible, -torch.inf)
    return combined


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int) -> torch.Tensor:
    """Add the diffusion-style VCD perturbation to preprocessed image tensors."""

    num_steps = 1000
    step = max(0, min(int(noise_step), num_steps - 1))
    betas = torch.linspace(
        -6,
        6,
        num_steps,
        device=image_tensor.device,
        dtype=torch.float32,
    )
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alpha_bar_sqrt = torch.sqrt(alphas_prod[step]).to(dtype=image_tensor.dtype)
    one_minus_alpha_bar_sqrt = torch.sqrt(1 - alphas_prod[step]).to(dtype=image_tensor.dtype)
    noise = torch.randn_like(image_tensor)
    return alpha_bar_sqrt * image_tensor + one_minus_alpha_bar_sqrt * noise


def _append_generated(base_inputs: dict[str, Any], generated: list[int], device: str) -> dict[str, Any]:
    if not generated:
        return dict(base_inputs)
    generated_ids = torch.tensor([generated], dtype=torch.long, device=device)
    out: dict[str, Any] = {}
    for key, value in base_inputs.items():
        if key == "input_ids":
            out[key] = torch.cat([value, generated_ids], dim=1)
        elif key == "attention_mask":
            extra = torch.ones((value.shape[0], len(generated)), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, extra], dim=1)
        elif key == "token_type_ids" and getattr(value, "ndim", 0) == 2:
            extra = torch.zeros((value.shape[0], len(generated)), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, extra], dim=1)
        else:
            out[key] = value
    return out


def _with_diffusion_noise(base_inputs: dict[str, Any], noise_step: int) -> dict[str, Any]:
    out = copy.copy(base_inputs)
    if "pixel_values" not in out:
        raise ValueError("Diffusion VCD requires `pixel_values` in processor inputs.")
    out["pixel_values"] = add_diffusion_noise(out["pixel_values"], noise_step=noise_step)
    return out


def _select_next_token(
    logits: torch.Tensor,
    decode_strategy: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> int:
    if decode_strategy == "greedy":
        return int(torch.argmax(logits).item())
    if decode_strategy != "sample":
        raise ValueError(f"Unsupported decode strategy: {decode_strategy}")
    scores = logits.clone()
    if temperature <= 0:
        raise ValueError("temperature must be positive for sample decoding.")
    scores = scores / temperature
    scores = _top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float,
) -> torch.Tensor:
    filtered = logits
    if top_k is not None and top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        if top_p <= 0:
            raise ValueError("top_p must be in (0, 1] when sampling.")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[1:] = sorted_remove[:-1].clone()
        sorted_remove[0] = False
        remove = torch.zeros_like(sorted_remove).scatter(0, sorted_indices, sorted_remove)
        filtered = filtered.masked_fill(remove, -torch.inf)
    return filtered


def _degrade_image(image: Image.Image, source: str, blur_radius: float) -> Image.Image:
    if source == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if source == "gray":
        return ImageOps.grayscale(image).convert("RGB")
    raise ValueError(f"Unsupported contrast source: {source}")


def _eos_token_ids(tokenizer: Any) -> set[int]:
    ids = set()
    for value in [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)]:
        if isinstance(value, list):
            ids.update(int(item) for item in value if item is not None)
        elif value is not None and not (isinstance(value, float) and math.isnan(value)):
            ids.add(int(value))
    return ids
