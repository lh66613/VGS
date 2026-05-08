#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args
from vgs.config import config_get, load_config
from vgs.datasets import PopeSample
from vgs.io import append_experiment_log, write_json, write_jsonl
from vgs.pope import classify_outcome, parse_yes_no
from vgs.artifacts import read_jsonl
from vgs.vlm_hf import MODEL_FAMILIES, generate_pope_answer, load_vlm_hf


PROMPT_VARIANTS = {
    "legacy": "Use the verification_question stored in the Stage T sample file.",
    "forced_evidence": "Forced image-evidence check with common-sense warning.",
    "conservative": "Negative-biased conservative visual evidence check.",
    "internal_rationale": "Ask for internal two-step verification but final yes/no only.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage T verification prompts for gated samples.")
    add_common_args(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-family", choices=sorted(MODEL_FAMILIES), default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow HF remote-code models such as InternVL.",
    )
    parser.add_argument("--qwen-min-pixels", type=int, default=None)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)
    parser.add_argument("--internvl-max-tiles", type=int, default=12)
    parser.add_argument(
        "--verification-samples",
        default="outputs/stage_t_selective_correction/stage_t_verification_samples.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPT_VARIANTS),
        default="legacy",
        help="Verification prompt template to apply.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="outputs/stage_t_selective_correction")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    rows = [] if args.dry_run else read_jsonl(args.verification_samples)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    payload: dict[str, Any] = {
        "model_path": model_path,
        "model_family": args.model_family,
        "verification_samples": args.verification_samples,
        "max_samples": args.max_samples,
        "num_samples": len(rows) if rows else None,
        "max_new_tokens": args.max_new_tokens,
        "prompt_variant": args.prompt_variant,
        "prompt_variant_description": PROMPT_VARIANTS[args.prompt_variant],
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        summary_path = write_json(
            Path(args.output_dir) / _summary_filename(args.prompt_variant),
            payload,
        )
        append_experiment_log(args.log_path, "run_stage_t_verification_eval", summary_path, "dry_run")
        print(summary_path)
        return

    bundle = load_vlm_hf(
        model_path,
        model_family=args.model_family,
        device=args.device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
        trust_remote_code=args.trust_remote_code,
        qwen_min_pixels=args.qwen_min_pixels,
        qwen_max_pixels=args.qwen_max_pixels,
        internvl_max_tiles=args.internvl_max_tiles,
    )
    out_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Stage T verification", unit="sample"):
        verification_question = _verification_prompt(row, args.prompt_variant)
        sample = PopeSample(
            sample_id=str(row["sample_id"]),
            question_id=str(row["sample_id"]),
            family="stage_t",
            subset=str(row.get("subset", "")),
            image=str(row.get("image", "")),
            image_path=str(row["image_path"]),
            question=verification_question,
            label=str(row["label"]),
        )
        raw_generation = generate_pope_answer(bundle, sample, max_new_tokens=args.max_new_tokens)
        parsed_prediction = parse_yes_no(raw_generation)
        out_rows.append(
            {
                **row,
                "verification_prompt_variant": args.prompt_variant,
                "verification_question_used": verification_question,
                "verification_raw_generation": raw_generation,
                "verification_parsed_prediction": parsed_prediction,
                "verification_outcome": classify_outcome(parsed_prediction, sample.label),
            }
        )

    output_stem = _prediction_stem(args.prompt_variant)
    predictions_path = write_jsonl(Path(args.output_dir) / f"{output_stem}.jsonl", out_rows)
    payload.update(
        {
            "resolved_device": bundle.device,
            "resolved_model_family": bundle.family,
            "verification_predictions_path": str(predictions_path),
        }
    )
    summary_path = write_json(Path(args.output_dir) / _summary_filename(args.prompt_variant), payload)
    append_experiment_log(args.log_path, "run_stage_t_verification_eval", summary_path, "ok")
    print(summary_path)


def _verification_prompt(row: dict[str, Any], variant: str) -> str:
    question = str(row.get("question", ""))
    if variant == "legacy":
        return str(row.get("verification_question", question))
    if variant == "forced_evidence":
        return (
            "Look at the image carefully. Is the object mentioned in the question visibly present?\n"
            "Answer with only one word: Yes or No.\n"
            "Do not rely on common sense or prior likelihood.\n"
            f"Question: {question}"
        )
    if variant == "conservative":
        return (
            'Answer "Yes" only if the object is clearly visible in the image.\n'
            'If the object is partially visible, uncertain, or not directly supported by the image, answer "No".\n'
            f"Question: {question}\n"
            "Answer only Yes or No."
        )
    if variant == "internal_rationale":
        return (
            "First internally verify whether the queried object is directly visible.\n"
            'Then answer only "Yes" or "No".\n'
            'If visual evidence is insufficient, answer "No".\n'
            f"Question: {question}"
        )
    raise ValueError(f"Unsupported prompt variant: {variant}")


def _prediction_stem(prompt_variant: str) -> str:
    if prompt_variant == "legacy":
        return "stage_t_verification_predictions"
    return f"stage_t_verification_predictions_{prompt_variant}"


def _summary_filename(prompt_variant: str) -> str:
    if prompt_variant == "legacy":
        return "run_stage_t_verification_eval_summary.json"
    return f"run_stage_t_verification_eval_{prompt_variant}_summary.json"


if __name__ == "__main__":
    main()
