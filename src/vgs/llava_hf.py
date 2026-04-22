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
    inputs = _move_inputs(inputs, device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def _move_inputs(inputs: Any, device: str) -> Any:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
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
