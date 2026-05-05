"""Hugging Face VLM adapters for cross-architecture POPE runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from vgs.datasets import PopeSample
from vgs.llava_hf import (
    build_pope_prompt as build_llava_pope_prompt,
    extract_condition_hidden_states as extract_llava_condition_hidden_states,
    extract_hidden_pair as extract_llava_hidden_pair,
    generate_pope_answer as generate_llava_pope_answer,
    load_llava_hf,
)


MODEL_FAMILIES = {"auto", "llava", "qwen2_vl", "qwen2_5_vl", "internvl2"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass
class VLMHFBundle:
    model: Any
    processor: Any
    tokenizer: Any
    family: str
    model_path: str
    device: str
    torch_dtype: torch.dtype
    internvl_max_tiles: int = 12
    internvl_image_size: int = 448


def resolve_device(device: str, allow_cpu: bool = False) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not visible. Refusing to load a large VLM on CPU; pass --allow-cpu "
            "only for tiny smoke tests or use a CUDA-visible shell."
        )
    return device


def infer_model_family(model_path: str | Path, requested: str = "auto") -> str:
    if requested not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {requested}")
    if requested != "auto":
        return requested

    path = Path(model_path)
    config_path = path / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(str(item) for item in config.get("architectures", [])).lower()
    name = str(model_path).lower()
    if "qwen2_5_vl" in model_type or "qwen2_5_vl" in architectures or "qwen2.5-vl" in name:
        return "qwen2_5_vl"
    if "qwen2_vl" in model_type or "qwen2vl" in architectures or "qwen2-vl" in name:
        return "qwen2_vl"
    if "internvl" in model_type or "internvl" in architectures or "internvl" in name:
        return "internvl2"
    return "llava"


def load_vlm_hf(
    model_path: str,
    model_family: str = "auto",
    device: str = "auto",
    torch_dtype: str = "float16",
    allow_cpu: bool = False,
    trust_remote_code: bool = True,
    qwen_min_pixels: int | None = None,
    qwen_max_pixels: int | None = None,
    internvl_max_tiles: int = 12,
) -> VLMHFBundle:
    resolved_device = resolve_device(device, allow_cpu=allow_cpu)
    dtype = _resolve_dtype(torch_dtype, resolved_device)
    family = infer_model_family(model_path, model_family)
    if family == "llava":
        model, processor, resolved_device = load_llava_hf(
            model_path,
            device=resolved_device,
            torch_dtype=torch_dtype,
            allow_cpu=allow_cpu,
        )
        return VLMHFBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            family=family,
            model_path=model_path,
            device=resolved_device,
            torch_dtype=dtype,
            internvl_max_tiles=internvl_max_tiles,
        )
    if family in {"qwen2_vl", "qwen2_5_vl"}:
        model = _load_qwen_model(model_path, family, dtype, trust_remote_code)
        model.to(resolved_device)
        model.eval()
        processor_kwargs = {}
        if qwen_min_pixels is not None:
            processor_kwargs["min_pixels"] = qwen_min_pixels
        if qwen_max_pixels is not None:
            processor_kwargs["max_pixels"] = qwen_max_pixels
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )
        return VLMHFBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            family=family,
            model_path=model_path,
            device=resolved_device,
            torch_dtype=dtype,
            internvl_max_tiles=internvl_max_tiles,
        )
    if family == "internvl2":
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            use_flash_attn=True,
        )
        model.to(resolved_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        return VLMHFBundle(
            model=model,
            processor=tokenizer,
            tokenizer=tokenizer,
            family=family,
            model_path=model_path,
            device=resolved_device,
            torch_dtype=dtype,
            internvl_max_tiles=internvl_max_tiles,
            internvl_image_size=int(getattr(model.config, "force_image_size", 448) or 448),
        )
    raise ValueError(f"Unsupported model family: {family}")


@torch.inference_mode()
def generate_pope_answer(
    bundle: VLMHFBundle,
    sample: PopeSample,
    max_new_tokens: int = 8,
) -> str:
    if bundle.family == "llava":
        return generate_llava_pope_answer(
            bundle.model,
            bundle.processor,
            sample,
            bundle.device,
            max_new_tokens=max_new_tokens,
        )
    if bundle.family in {"qwen2_vl", "qwen2_5_vl"}:
        image = Image.open(Path(sample.image_path)).convert("RGB")
        inputs = _qwen_inputs(bundle, sample.question, image)
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_pad_token_id(bundle.tokenizer),
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_length:]
        return bundle.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if bundle.family == "internvl2":
        pixel_values = _internvl_load_image(
            sample.image_path,
            input_size=bundle.internvl_image_size,
            max_tiles=bundle.internvl_max_tiles,
        ).to(device=bundle.device, dtype=bundle.torch_dtype)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        return bundle.model.chat(
            bundle.tokenizer,
            pixel_values,
            _pope_instruction(sample.question),
            generation_config,
        ).strip()
    raise ValueError(f"Unsupported model family: {bundle.family}")


@torch.inference_mode()
def extract_hidden_pair(
    bundle: VLMHFBundle,
    row: dict[str, Any],
    layers: list[int],
    readout_position: str = "last_prompt_token",
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if bundle.family == "llava":
        return extract_llava_hidden_pair(
            bundle.model,
            bundle.processor,
            row,
            layers,
            bundle.device,
            readout_position=readout_position,
        )
    image_states = extract_condition_hidden_states(
        bundle,
        row["question"],
        row["image_path"],
        layers,
        readout_position=readout_position,
    )
    blind_states = extract_condition_hidden_states(
        bundle,
        row["question"],
        None,
        layers,
        readout_position=readout_position,
    )
    return {layer: (image_states[layer], blind_states[layer]) for layer in layers}


@torch.inference_mode()
def extract_condition_hidden_states(
    bundle: VLMHFBundle,
    question: str,
    image_path: str | None,
    layers: list[int],
    readout_position: str = "last_prompt_token",
) -> dict[int, torch.Tensor]:
    if bundle.family == "llava":
        return extract_llava_condition_hidden_states(
            bundle.model,
            bundle.processor,
            question,
            image_path,
            layers,
            bundle.device,
            readout_position=readout_position,
        )
    if bundle.family in {"qwen2_vl", "qwen2_5_vl"}:
        image = Image.open(Path(image_path)).convert("RGB") if image_path else None
        inputs = _qwen_inputs(bundle, question, image)
        outputs = bundle.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        index = int(inputs["attention_mask"][0].sum().item()) - 1
        return {
            layer: _readout_hidden_state(outputs.hidden_states[layer][0], index, readout_position)
            for layer in layers
        }
    if bundle.family == "internvl2":
        if image_path:
            inputs = _internvl_image_forward_inputs(bundle, question, image_path)
            outputs = bundle.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        else:
            inputs = _internvl_blind_inputs(bundle, question)
            outputs = bundle.model.language_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        index = int(inputs["attention_mask"][0].sum().item()) - 1
        return {
            layer: _readout_hidden_state(outputs.hidden_states[layer][0], index, readout_position)
            for layer in layers
        }
    raise ValueError(f"Unsupported model family: {bundle.family}")


@torch.inference_mode()
def next_token_logits(bundle: VLMHFBundle, row: dict[str, Any]) -> torch.Tensor:
    if bundle.family == "llava":
        image = Image.open(Path(row["image_path"])).convert("RGB")
        prompt = build_llava_pope_prompt(bundle.processor, row["question"])
        inputs = bundle.processor(images=image, text=prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, bundle.device, dtype=bundle.torch_dtype)
        outputs = bundle.model(
            **inputs,
            return_dict=True,
            use_cache=False,
        )
        return outputs.logits[0, -1].detach().float().cpu()
    if bundle.family in {"qwen2_vl", "qwen2_5_vl"}:
        image = Image.open(Path(row["image_path"])).convert("RGB")
        inputs = _qwen_inputs(bundle, row["question"], image)
        outputs = bundle.model(
            **inputs,
            return_dict=True,
            use_cache=False,
        )
        return outputs.logits[0, -1].detach().float().cpu()
    if bundle.family == "internvl2":
        inputs = _internvl_image_forward_inputs(bundle, row["question"], row["image_path"])
        outputs = bundle.model(
            **inputs,
            return_dict=True,
            use_cache=False,
        )
        return outputs.logits[0, -1].detach().float().cpu()
    raise ValueError(f"Unsupported model family: {bundle.family}")


def candidate_token_ids(tokenizer: Any, candidates: list[str]) -> list[int]:
    token_ids = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            token_ids.append(int(ids[-1]))
    return sorted(set(token_ids))


def max_token_logit(logits: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return math.nan
    return float(torch.max(logits[token_ids]).item())


def binary_entropy(logit_a: float, logit_b: float) -> float:
    values = np.array([logit_a, logit_b], dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values) / np.exp(values).sum()
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def _load_qwen_model(
    model_path: str,
    family: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    try:
        if family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            cls = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration

            cls = Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "This environment's transformers build does not expose the Qwen2-VL classes. "
            "Use an environment with transformers support for qwen2_vl/qwen2_5_vl "
            "(Qwen's model card recommends a recent transformers build)."
        ) from exc
    return cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )


def _qwen_inputs(bundle: VLMHFBundle, question: str, image: Image.Image | None) -> Any:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": _pope_instruction(question)})
    messages = [{"role": "user", "content": content}]
    prompt = bundle.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if image is None:
        inputs = bundle.tokenizer(prompt, return_tensors="pt")
    else:
        inputs = bundle.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    return _move_inputs(inputs, bundle.device, dtype=bundle.torch_dtype)


def _internvl_image_forward_inputs(
    bundle: VLMHFBundle,
    question: str,
    image_path: str,
) -> dict[str, torch.Tensor]:
    pixel_values = _internvl_load_image(
        image_path,
        input_size=bundle.internvl_image_size,
        max_tiles=bundle.internvl_max_tiles,
    ).to(device=bundle.device, dtype=bundle.torch_dtype)
    query = _internvl_query(bundle, question, num_patches=pixel_values.shape[0])
    model_inputs = bundle.tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(bundle.device)
    attention_mask = model_inputs["attention_mask"].to(bundle.device)
    bundle.model.img_context_token_id = bundle.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=bundle.device)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_flags": image_flags,
    }


def _internvl_blind_inputs(bundle: VLMHFBundle, question: str) -> dict[str, torch.Tensor]:
    query = _internvl_blind_query(bundle, question)
    model_inputs = bundle.tokenizer(query, return_tensors="pt")
    return {
        "input_ids": model_inputs["input_ids"].to(bundle.device),
        "attention_mask": model_inputs["attention_mask"].to(bundle.device),
    }


def _internvl_query(bundle: VLMHFBundle, question: str, num_patches: int) -> str:
    query = _internvl_blind_query(bundle, f"<image>\n{_pope_instruction(question)}", already_wrapped=True)
    image_tokens = "<img>" + "<IMG_CONTEXT>" * bundle.model.num_image_token * num_patches + "</img>"
    return query.replace("<image>", image_tokens, 1)


def _internvl_blind_query(
    bundle: VLMHFBundle,
    question: str,
    already_wrapped: bool = False,
) -> str:
    template = bundle.model.conv_template.copy()
    template.system_message = bundle.model.system_message
    user_text = question if already_wrapped else _pope_instruction(question)
    template.append_message(template.roles[0], user_text)
    template.append_message(template.roles[1], None)
    return template.get_prompt()


def _internvl_load_image(image_path: str | Path, input_size: int, max_tiles: int) -> torch.Tensor:
    image = Image.open(Path(image_path)).convert("RGB")
    tiles = _internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        max_tiles=max_tiles,
        use_thumbnail=True,
    )
    return torch.stack([_internvl_transform(tile, input_size) for tile in tiles])


def _internvl_transform(image: Image.Image, input_size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.convert("RGB").resize((input_size, input_size), resampling)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _internvl_dynamic_preprocess(
    image: Image.Image,
    image_size: int,
    max_tiles: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        },
        key=lambda item: item[0] * item[1],
    )
    target_ratio = _internvl_closest_ratio(aspect_ratio, target_ratios, width, height, image_size)
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = image.resize((target_width, target_height), resampling)
    tiles = []
    tiles_per_row = target_width // image_size
    for idx in range(blocks):
        left = (idx % tiles_per_row) * image_size
        upper = (idx // tiles_per_row) * image_size
        tiles.append(resized.crop((left, upper, left + image_size, upper + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), resampling))
    return tiles


def _internvl_closest_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        candidate = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - candidate)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _pope_instruction(question: str) -> str:
    return f"{question} Answer with yes or no only."


def _readout_hidden_state(
    sequence_states: torch.Tensor,
    last_index: int,
    readout_position: str,
) -> torch.Tensor:
    if readout_position in {"last_prompt_token", "first_answer_prefill"}:
        return sequence_states[last_index].detach().cpu().float()
    if readout_position == "last_4_prompt_mean":
        start = max(0, last_index - 3)
        return sequence_states[start : last_index + 1].mean(dim=0).detach().cpu().float()
    if readout_position == "last_8_prompt_mean":
        start = max(0, last_index - 7)
        return sequence_states[start : last_index + 1].mean(dim=0).detach().cpu().float()
    raise NotImplementedError(f"Unsupported readout position: {readout_position}")


def _move_inputs(inputs: Any, device: str, dtype: torch.dtype | None = None) -> Any:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_dtype(torch_dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float32":
        return torch.float32
    return torch.float16


def _pad_token_id(tokenizer: Any) -> int | None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, list):
        return int(eos_token_id[0]) if eos_token_id else None
    return int(eos_token_id) if eos_token_id is not None else None
