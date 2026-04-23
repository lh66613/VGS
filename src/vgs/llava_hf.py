"""Hugging Face LLaVA inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration
from transformers.models.llava.processing_llava import LlavaProcessor

from vgs.datasets import PopeSample


def resolve_device(device: str, allow_cpu: bool = False) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not visible. Refusing to load LLaVA-1.5-7B on CPU; pass --allow-cpu "
            "only for tiny smoke tests or use a CUDA-visible shell."
        )
    return device


def load_llava_hf(
    model_path: str,
    device: str = "auto",
    torch_dtype: str = "float16",
    allow_cpu: bool = False,
) -> tuple[LlavaForConditionalGeneration, AutoProcessor, str]:
    resolved_device = resolve_device(device, allow_cpu=allow_cpu)
    dtype = _resolve_dtype(torch_dtype, resolved_device)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)
    model.eval()
    processor = load_llava_processor(model_path)
    return model, processor, resolved_device


def load_llava_processor(model_path: str) -> LlavaProcessor:
    try:
        return AutoProcessor.from_pretrained(model_path, use_fast=False)
    except TypeError as exc:
        if "image_token" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
        return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def build_pope_prompt(processor: AutoProcessor, question: str) -> str:
    text = f"{question} Answer with yes or no only."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return f"USER: <image>\n{text} ASSISTANT:"


def build_blind_prompt(question: str) -> str:
    text = f"{question} Answer with yes or no only."
    return f"USER: {text} ASSISTANT:"


@torch.inference_mode()
def generate_pope_answer(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    sample: PopeSample,
    device: str,
    max_new_tokens: int = 8,
) -> str:
    image = Image.open(Path(sample.image_path)).convert("RGB")
    prompt = build_pope_prompt(processor, sample.question)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = _move_inputs(inputs, device, dtype=next(model.parameters()).dtype)
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
def extract_hidden_pair(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    row: dict[str, Any],
    layers: list[int],
    device: str,
    readout_position: str = "last_prompt_token",
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if readout_position != "last_prompt_token":
        raise NotImplementedError(
            "Only last_prompt_token is implemented for the first HF hidden-state dump."
        )
    image = Image.open(Path(row["image_path"])).convert("RGB")
    image_prompt = build_pope_prompt(processor, row["question"])
    image_inputs = processor(images=image, text=image_prompt, return_tensors="pt")
    image_inputs = _move_inputs(image_inputs, device, dtype=next(model.parameters()).dtype)
    image_outputs = model(
        **image_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    blind_prompt = build_blind_prompt(row["question"])
    blind_inputs = processor.tokenizer(blind_prompt, return_tensors="pt")
    blind_inputs = _move_inputs(blind_inputs, device)
    blind_outputs = model.language_model(
        **blind_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    image_index = int(image_inputs["attention_mask"][0].sum().item()) - 1
    blind_index = int(blind_inputs["attention_mask"][0].sum().item()) - 1
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        image_state = image_outputs.hidden_states[layer][0, image_index].detach().cpu().float()
        blind_state = blind_outputs.hidden_states[layer][0, blind_index].detach().cpu().float()
        pairs[layer] = (image_state, blind_state)
    return pairs


@torch.inference_mode()
def extract_condition_hidden_states(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    question: str,
    image_path: str | None,
    layers: list[int],
    device: str,
    readout_position: str = "last_prompt_token",
) -> dict[int, torch.Tensor]:
    if readout_position != "last_prompt_token":
        raise NotImplementedError(
            "Only last_prompt_token is implemented for condition hidden-state dumps."
        )
    if image_path:
        image = Image.open(Path(image_path)).convert("RGB")
        prompt = build_pope_prompt(processor, question)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, device, dtype=next(model.parameters()).dtype)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        index = int(inputs["attention_mask"][0].sum().item()) - 1
    else:
        prompt = build_blind_prompt(question)
        inputs = processor.tokenizer(prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, device)
        outputs = model.language_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        index = int(inputs["attention_mask"][0].sum().item()) - 1

    return {
        layer: outputs.hidden_states[layer][0, index].detach().cpu().float()
        for layer in layers
    }


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
