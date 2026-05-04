#!/usr/bin/env python
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl
from vgs.config import config_get, load_config
from vgs.io import append_experiment_log, ensure_dir, write_json, write_jsonl
from vgs.llava_hf import generate_pope_answer, load_llava_hf, resolve_device
from vgs.pope import classify_outcome, parse_yes_no


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaVA on AMBER discriminative rows.")
    parser.add_argument("--plan", default="outputs/stage_n_external/amber_discriminative_plan.jsonl")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/stage_n_external")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    rows = [] if args.dry_run else read_jsonl(args.plan)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    payload = {
        "plan": args.plan,
        "model_path": model_path,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if not args.dry_run:
        resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
        model, processor, resolved_device = load_llava_hf(
            model_path,
            device=resolved_device,
            torch_dtype=torch_dtype,
            allow_cpu=args.allow_cpu,
        )
        out_rows = []
        amber_responses = []
        for row in rows:
            sample = SimpleNamespace(image_path=row["image_path"], question=row["question"])
            raw = generate_pope_answer(model, processor, sample, device=resolved_device, max_new_tokens=args.max_new_tokens)
            parsed = parse_yes_no(raw)
            outcome = classify_outcome(parsed, row["label"])
            response = "Yes" if parsed == "yes" else "No" if parsed == "no" else raw
            out_rows.append({**row, "raw_generation": raw, "parsed_prediction": parsed, "outcome": outcome})
            amber_responses.append({"id": int(row["question_id"]), "response": response})
        predictions_path = write_jsonl(Path(args.output_dir) / "amber_predictions.jsonl", out_rows)
        amber_response_path = Path(args.output_dir) / "amber_responses_for_official_eval.json"
        ensure_dir(amber_response_path.parent)
        amber_response_path.write_text(json.dumps(amber_responses, indent=2) + "\n", encoding="utf-8")
        payload.update(
            {
                "resolved_device": resolved_device,
                "num_rows": len(out_rows),
                "predictions_path": str(predictions_path),
                "amber_response_path": str(amber_response_path),
            }
        )
    summary_path = write_json(Path(args.output_dir) / "run_stage_n_amber_eval_summary.json", payload)
    append_experiment_log(args.log_path, "run_stage_n_amber_eval", summary_path, "dry_run" if args.dry_run else "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
