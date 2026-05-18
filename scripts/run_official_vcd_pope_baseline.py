#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from typing import Any

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.cli import add_common_args
from vgs.config import config_get, load_config
from vgs.constants import DEFAULT_POPE_SUBSETS
from vgs.datasets import PopeSample, load_pope_subset, validate_pope_samples
from vgs.io import append_experiment_log, ensure_dir, write_csv, write_json
from vgs.pope import classify_outcome, parse_yes_no
from vgs.vcd import generate_llava_contrastive_answer, official_vcd_reference
from vgs.vlm_hf import load_vlm_hf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official VCD-style diffusion baseline on POPE with "
            "LLaVA-1.5-7B and save predictions/metrics."
        )
    )
    add_common_args(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--decode-strategy", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing predictions JSONL and skip completed sample_ids.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baselines/official_vcd_pope_llava15_7b",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    family = args.family or config_get(config, "dataset.pope_family", "coco")
    questions_dir = args.questions_dir or config_get(config, "dataset.questions_dir", "data/pope/questions")
    images_dir = args.images_dir or config_get(config, "dataset.images_dir", "data/pope/images")
    subsets = args.subsets or config_get(config, "dataset.subsets", DEFAULT_POPE_SUBSETS)
    pattern = config_get(config, "dataset.question_file_pattern", "{family}_pope_{subset}.json")

    samples = load_pope_samples(
        questions_dir=questions_dir,
        images_dir=images_dir,
        family=family,
        subsets=subsets,
        pattern=pattern,
        max_samples=args.max_samples,
    )
    validation = validate_pope_samples(samples)
    output_dir = ensure_dir(args.output_dir)
    predictions_path = output_dir / "pope_vcd_predictions.jsonl"
    metrics_path = output_dir / "pope_vcd_metrics.csv"
    summary_path = output_dir / "run_official_vcd_pope_baseline_summary.json"

    payload: dict[str, Any] = {
        "baseline": "official_vcd_diffusion",
        "implementation_reference": official_vcd_reference(),
        "local_adapter_note": (
            "Uses the DAMO-NLP-SG/VCD token-level diffusion/APC/logit rule in "
            "the local HF LLaVA decoding loop instead of monkey-patching Transformers."
        ),
        "model_path": model_path,
        "model_family": "llava",
        "family": family,
        "questions_dir": questions_dir,
        "images_dir": images_dir,
        "subsets": subsets,
        "max_samples": args.max_samples,
        "num_samples": len(samples),
        "validation": validation,
        "alpha": args.alpha,
        "beta": args.beta,
        "noise_step": args.noise_step,
        "decode_strategy": args.decode_strategy,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "torch_dtype": torch_dtype,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        write_json(summary_path, payload)
        append_experiment_log(args.log_path, "run_official_vcd_pope_baseline", summary_path, "dry_run")
        print(summary_path)
        return

    if not validation["ok"]:
        raise RuntimeError(f"POPE validation failed: {validation}")

    completed = _load_completed(predictions_path) if args.resume else {}
    pending = [sample for sample in samples if sample.sample_id not in completed]

    bundle = load_vlm_hf(
        model_path,
        model_family="llava",
        device=args.device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
    )
    if bundle.family != "llava":
        raise RuntimeError(f"Expected LLaVA adapter, got {bundle.family}.")

    generator = None
    if args.decode_strategy == "sample":
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=bundle.device)
        generator.manual_seed(args.seed)

    mode = "a" if args.resume and completed else "w"
    with predictions_path.open(mode, encoding="utf-8") as f:
        for sample in tqdm(pending, desc="official VCD POPE", unit="sample"):
            row = run_one_sample(
                model=bundle.model,
                processor=bundle.processor,
                sample=sample,
                device=bundle.device,
                max_new_tokens=args.max_new_tokens,
                alpha=args.alpha,
                beta=args.beta,
                noise_step=args.noise_step,
                decode_strategy=args.decode_strategy,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                generator=generator,
            )
            json.dump(row, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            completed[row["sample_id"]] = row

    rows = [completed[sample.sample_id] for sample in samples if sample.sample_id in completed]
    metrics_rows = summarize_pope_rows(rows)
    write_csv(metrics_path, metrics_rows, _fieldnames(metrics_rows))
    payload.update(
        {
            "resolved_device": bundle.device,
            "resolved_model_family": bundle.family,
            "num_completed": len(rows),
            "num_pending": len(samples) - len(rows),
            "metrics": metrics_rows,
        }
    )
    write_json(summary_path, payload)
    append_experiment_log(args.log_path, "run_official_vcd_pope_baseline", summary_path, "ok")
    print(summary_path)


def load_pope_samples(
    questions_dir: str | Path,
    images_dir: str | Path,
    family: str,
    subsets: list[str],
    pattern: str,
    max_samples: int | None,
) -> list[PopeSample]:
    samples: list[PopeSample] = []
    for subset in subsets:
        samples.extend(load_pope_subset(questions_dir, images_dir, family, subset, pattern))
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def run_one_sample(
    model: Any,
    processor: Any,
    sample: PopeSample,
    device: str,
    max_new_tokens: int,
    alpha: float,
    beta: float,
    noise_step: int,
    decode_strategy: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> dict[str, Any]:
    raw_generation = generate_llava_contrastive_answer(
        model,
        processor,
        sample,
        device,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
        beta=beta,
        contrast_source="diffusion",
        noise_step=noise_step,
        decode_strategy=decode_strategy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )
    parsed_prediction = parse_yes_no(raw_generation)
    outcome = classify_outcome(parsed_prediction, sample.label)
    return {
        **sample.to_json(),
        "baseline": "official_vcd_diffusion",
        "vcd_alpha": alpha,
        "vcd_beta": beta,
        "vcd_noise_step": noise_step,
        "vcd_decode_strategy": decode_strategy,
        "vcd_temperature": temperature,
        "vcd_top_p": top_p,
        "vcd_top_k": top_k,
        "raw_generation": raw_generation,
        "parsed_prediction": parsed_prediction,
        "outcome": outcome,
    }


def summarize_pope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subsets = ["overall"] + sorted({str(row.get("subset", "")) for row in rows})
    return [_metric_row(rows, subset) for subset in subsets]


def _metric_row(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    group = rows if subset == "overall" else [row for row in rows if str(row.get("subset", "")) == subset]
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "unknown": 0}
    for row in group:
        outcome = str(row.get("outcome", "unknown"))
        counts[outcome if outcome in counts else "unknown"] += 1

    known_n = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    predicted_yes = counts["TP"] + counts["FP"]
    gold_yes = counts["TP"] + counts["FN"]
    gold_no = counts["TN"] + counts["FP"]
    precision = _safe_div(counts["TP"], predicted_yes)
    recall = _safe_div(counts["TP"], gold_yes)
    return {
        "subset": subset,
        "n": len(group),
        "known_n": known_n,
        "unknown": counts["unknown"],
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "accuracy": _safe_div(counts["TP"] + counts["TN"], known_n),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "fp_rate": _safe_div(counts["FP"], gold_no),
        "yes_rate": _safe_div(predicted_yes, known_n),
    }


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


if __name__ == "__main__":
    main()
