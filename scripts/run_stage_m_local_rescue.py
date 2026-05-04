#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.config import config_get, load_config
from vgs.io import append_experiment_log, write_json
from vgs.llava_hf import load_llava_hf, resolve_device
from vgs.stage_m import run_stage_m_local_rescue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage M gated local rescue interventions.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["32"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--memory-bank", default="outputs/stage_m_local_rescue/memory_bank_train.pt")
    parser.add_argument("--retrieval-plan", default="outputs/stage_m_local_rescue/retrieval_plan.csv")
    parser.add_argument("--alpha-grid", nargs="*", default=["2", "4", "8"])
    parser.add_argument(
        "--gates",
        nargs="*",
        default=["always", "low_abs_margin", "high_fp_risk", "margin_and_fp_risk"],
        choices=["always", "low_abs_margin", "high_entropy", "high_fp_risk", "margin_and_fp_risk", "tail_norm_available"],
    )
    parser.add_argument(
        "--retrieval-modes",
        nargs="*",
        default=["same_object_tn", "svd_knn_tn", "tail_knn_tn", "random_tn", "same_object_fp"],
    )
    parser.add_argument("--target-outcomes", nargs="*", default=["FP", "TN", "TP"])
    parser.add_argument("--max-targets-per-outcome", type=int, default=32)
    parser.add_argument("--margin-threshold", type=float, default=0.25)
    parser.add_argument("--entropy-threshold", type=float, default=0.65)
    parser.add_argument("--fp-risk-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--granularities", nargs="*", default=["last_token"], choices=["last_token", "full_sequence", "generated_token"])
    parser.add_argument("--logits-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/stage_m_local_rescue")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    alpha_grid = [float(item) for value in args.alpha_grid for item in value.split(",") if item]
    payload = {
        "layers": layers,
        "model_path": model_path,
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "memory_bank": args.memory_bank,
        "retrieval_plan": args.retrieval_plan,
        "alpha_grid": alpha_grid,
        "gates": args.gates,
        "retrieval_modes": args.retrieval_modes,
        "target_outcomes": args.target_outcomes,
        "max_targets_per_outcome": args.max_targets_per_outcome,
        "margin_threshold": args.margin_threshold,
        "entropy_threshold": args.entropy_threshold,
        "fp_risk_threshold": args.fp_risk_threshold,
        "max_new_tokens": args.max_new_tokens,
        "granularities": args.granularities,
        "logits_only": args.logits_only,
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
        payload["resolved_device"] = resolved_device
        payload.update(
            run_stage_m_local_rescue(
                model=model,
                processor=processor,
                predictions_path=args.predictions,
                hidden_states_dir=args.hidden_states_dir,
                memory_bank_path=args.memory_bank,
                retrieval_plan_path=args.retrieval_plan,
                output_dir=args.output_dir,
                layers=layers,
                device=resolved_device,
                alpha_grid=alpha_grid,
                gates=args.gates,
                retrieval_modes=args.retrieval_modes,
                target_outcomes=args.target_outcomes,
                max_targets_per_outcome=args.max_targets_per_outcome,
                margin_threshold=args.margin_threshold,
                entropy_threshold=args.entropy_threshold,
                fp_risk_threshold=args.fp_risk_threshold,
                max_new_tokens=args.max_new_tokens,
                granularities=args.granularities,
                logits_only=args.logits_only,
                seed=args.seed,
            )
        )

    summary_path = write_json(Path(args.output_dir) / "run_stage_m_local_rescue_summary.json", payload)
    append_experiment_log(
        args.log_path,
        "run_stage_m_local_rescue",
        summary_path,
        "dry_run" if args.dry_run else "ok",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
