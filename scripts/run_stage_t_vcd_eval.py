#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.cli import add_common_args
from vgs.config import config_get, load_config
from vgs.datasets import PopeSample
from vgs.io import append_experiment_log, write_json, write_jsonl
from vgs.pope import classify_outcome, parse_yes_no
from vgs.vcd import generate_llava_contrastive_answer, official_vcd_reference
from vgs.vlm_hf import MODEL_FAMILIES, load_vlm_hf


OPERATORS = {
    "vcd_diffusion": (
        "Official VCD baseline: contrast original image logits against "
        "diffusion-noised image tensors."
    ),
    "vcd_blur": "VCD-style ablation: contrast original image logits against a Gaussian-blurred image.",
    "vcd_gray": "VCD-style ablation: contrast original image logits against a grayscale image.",
    "icd_blind": "Contrast original image logits against a text-only blind prompt.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage T VCD/ICD operator on predicted-Yes samples.")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-family", choices=sorted(MODEL_FAMILIES), default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI symmetry; Stage T VCD currently supports LLaVA HF only.",
    )
    parser.add_argument(
        "--vcd-samples",
        default="outputs/stage_t_selective_correction_fixed_ids/stage_t_verification_pool.jsonl",
        help="Predicted-Yes pool to run the decoding operator on.",
    )
    parser.add_argument("--operator", choices=sorted(OPERATORS), default="vcd_diffusion")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--blur-radius", type=float, default=5.0)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--decode-strategy", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction_fixed_ids")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    rows = [] if args.dry_run else read_jsonl(args.vcd_samples)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    payload: dict[str, Any] = {
        "model_path": model_path,
        "model_family": args.model_family,
        "vcd_samples": args.vcd_samples,
        "operator": args.operator,
        "operator_description": OPERATORS[args.operator],
        "implementation_reference": official_vcd_reference() if args.operator == "vcd_diffusion" else None,
        "alpha": args.alpha,
        "beta": args.beta,
        "blur_radius": args.blur_radius,
        "noise_step": args.noise_step,
        "decode_strategy": args.decode_strategy,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "num_samples": len(rows) if rows else None,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    summary_name = _summary_filename(args.operator)
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        summary_path = write_json(Path(args.output_dir) / summary_name, payload)
        append_experiment_log(args.log_path, "run_stage_t_vcd_eval", summary_path, "dry_run")
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
        raise NotImplementedError(
            "Stage T VCD/ICD generation currently supports the LLaVA HF adapter only. "
            "Use --model-family llava for the base experiment."
        )

    contrast_source = _contrast_source(args.operator)
    generator = None
    if args.decode_strategy == "sample":
        import torch

        torch.manual_seed(args.seed)
        generator = torch.Generator(device=bundle.device)
        generator.manual_seed(args.seed)
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"Stage T {args.operator}", unit="sample"):
        sample = PopeSample(
            sample_id=str(row["sample_id"]),
            question_id=str(row["sample_id"]),
            family="stage_t",
            subset=str(row.get("subset", "")),
            image=str(row.get("image", "")),
            image_path=str(row["image_path"]),
            question=str(row["question"]),
            label=str(row["label"]),
        )
        raw_generation = generate_llava_contrastive_answer(
            bundle.model,
            bundle.processor,
            sample,
            bundle.device,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            beta=args.beta,
            contrast_source=contrast_source,
            blur_radius=args.blur_radius,
            noise_step=args.noise_step,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generator=generator,
        )
        parsed_prediction = parse_yes_no(raw_generation)
        out_rows.append(
            {
                **row,
                "vcd_operator": args.operator,
                "vcd_alpha": args.alpha,
                "vcd_beta": args.beta,
                "vcd_blur_radius": args.blur_radius,
                "vcd_noise_step": args.noise_step,
                "vcd_decode_strategy": args.decode_strategy,
                "vcd_temperature": args.temperature,
                "vcd_top_p": args.top_p,
                "vcd_top_k": args.top_k,
                "vcd_raw_generation": raw_generation,
                "vcd_parsed_prediction": parsed_prediction,
                "vcd_outcome": classify_outcome(parsed_prediction, sample.label),
            }
        )

    predictions_path = write_jsonl(Path(args.output_dir) / f"{_prediction_stem(args.operator)}.jsonl", out_rows)
    payload.update(
        {
            "resolved_device": bundle.device,
            "resolved_model_family": bundle.family,
            "vcd_predictions_path": str(predictions_path),
        }
    )
    summary_path = write_json(Path(args.output_dir) / summary_name, payload)
    append_experiment_log(args.log_path, "run_stage_t_vcd_eval", summary_path, "ok")
    print(summary_path)


def _contrast_source(operator: str) -> str:
    if operator == "vcd_diffusion":
        return "diffusion"
    if operator == "vcd_blur":
        return "blur"
    if operator == "vcd_gray":
        return "gray"
    if operator == "icd_blind":
        return "blind"
    raise ValueError(f"Unsupported operator: {operator}")


def _prediction_stem(operator: str) -> str:
    return f"stage_t_vcd_predictions_{operator}"


def _summary_filename(operator: str) -> str:
    return f"run_stage_t_vcd_eval_{operator}_summary.json"


if __name__ == "__main__":
    main()
