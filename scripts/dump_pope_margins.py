#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.cli import add_common_args
from vgs.config import config_get, load_config
from vgs.io import append_experiment_log, write_csv, write_json
from vgs.vlm_hf import (
    MODEL_FAMILIES,
    binary_entropy,
    candidate_token_ids,
    load_vlm_hf,
    max_token_logit,
    next_token_logits,
)


YES_CANDIDATES = ["yes", "Yes", " yes", " Yes", "YES"]
NO_CANDIDATES = ["no", "No", " no", " No", "NO"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump first-token yes/no margin scores for POPE rows.")
    add_common_args(parser)
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
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
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="outputs/margins")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    rows = [] if args.dry_run else read_jsonl(args.predictions)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    payload: dict[str, Any] = {
        "model_source": args.model_source,
        "model_path": model_path,
        "model_family": args.model_family,
        "predictions": args.predictions,
        "max_samples": args.max_samples,
        "num_samples": len(rows) if rows else None,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        summary_path = write_json(Path(args.output_dir) / "dump_pope_margins_summary.json", payload)
        append_experiment_log(args.log_path, "dump_pope_margins", summary_path, "dry_run")
        print(summary_path)
        return
    if args.model_source != "hf":
        raise NotImplementedError("Only the Hugging Face path is implemented at this stage.")

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
    yes_ids = candidate_token_ids(bundle.tokenizer, YES_CANDIDATES)
    no_ids = candidate_token_ids(bundle.tokenizer, NO_CANDIDATES)
    margin_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="POPE first-token margins", unit="sample"):
        logits = next_token_logits(bundle, row)
        yes_logit = max_token_logit(logits, yes_ids)
        no_logit = max_token_logit(logits, no_ids)
        top_token_id = int(torch.argmax(logits).item())
        margin_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "subset": row.get("subset", ""),
                "label": row.get("label", ""),
                "outcome": row.get("outcome", ""),
                "parsed_prediction": row.get("parsed_prediction", ""),
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "yes_minus_no_logit": yes_logit - no_logit,
                "no_minus_yes_logit": no_logit - yes_logit,
                "binary_entropy": binary_entropy(yes_logit, no_logit),
                "top_token_id": top_token_id,
                "top_token": bundle.tokenizer.decode([top_token_id], skip_special_tokens=False),
            }
        )

    output_dir = Path(args.output_dir)
    path = write_csv(output_dir / "pope_margin_scores.csv", margin_rows, _fieldnames(margin_rows))
    payload.update(
        {
            "resolved_device": bundle.device,
            "resolved_model_family": bundle.family,
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
            "margin_scores_path": str(path),
        }
    )
    summary_path = write_json(output_dir / "dump_pope_margins_summary.json", payload)
    append_experiment_log(args.log_path, "dump_pope_margins", summary_path, "ok")
    print(summary_path)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


if __name__ == "__main__":
    main()
