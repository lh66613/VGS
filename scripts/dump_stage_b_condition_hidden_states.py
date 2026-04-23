#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.artifacts import read_jsonl, save_condition_hidden_layer
from vgs.cli import add_common_args, add_layer_args, resolve_layers
from vgs.config import config_get, load_config
from vgs.io import append_experiment_log, write_json
from vgs.llava_hf import extract_condition_hidden_states, load_llava_hf, resolve_device


CONDITION_IMAGE_KEYS = {
    "matched": "matched_image_path",
    "random_mismatch": "random_mismatch_image_path",
    "adversarial_mismatch": "adversarial_mismatch_image_path",
    "blind": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump hidden states for Stage B matched/mismatched/blind conditions."
    )
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["matched", "random_mismatch", "adversarial_mismatch", "blind"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--readout-position", default="last_prompt_token")
    parser.add_argument("--output-dir", default="outputs/stage_b_hidden")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    rows = []
    if not args.dry_run:
        rows = read_jsonl(args.condition_plan)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    payload = {
        "layers": layers,
        "model_source": args.model_source,
        "model_path": model_path,
        "condition_plan": args.condition_plan,
        "conditions": args.conditions,
        "max_samples": args.max_samples,
        "num_samples": len(rows) if rows else None,
        "readout_position": args.readout_position,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        summary_path = write_json(Path(args.output_dir) / "dump_stage_b_condition_hidden_states_summary.json", payload)
        append_experiment_log(args.log_path, "dump_stage_b_condition_hidden_states", summary_path, "dry_run")
        print(summary_path)
        return

    if args.model_source != "hf":
        raise NotImplementedError("Only the Hugging Face LLaVA path is implemented at this stage.")
    for condition in args.conditions:
        if condition not in CONDITION_IMAGE_KEYS:
            raise ValueError(f"Unknown Stage B condition: {condition}")

    resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    model, processor, resolved_device = load_llava_hf(
        model_path,
        device=resolved_device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
    )

    layer_condition_states = {
        layer: {condition: [] for condition in args.conditions}
        for layer in layers
    }
    sample_ids = []
    for row in tqdm(rows, desc="Stage B condition hidden states", unit="sample"):
        sample_ids.append(str(row["sample_id"]))
        for condition in args.conditions:
            image_key = CONDITION_IMAGE_KEYS[condition]
            image_path = row.get(image_key) if image_key else None
            if image_key and not image_path:
                raise ValueError(f"Missing image path for condition {condition}, sample {row['sample_id']}")
            states = extract_condition_hidden_states(
                model,
                processor,
                row["question"],
                image_path,
                layers,
                resolved_device,
                readout_position=args.readout_position,
            )
            for layer, state in states.items():
                layer_condition_states[layer][condition].append(state)

    artifacts = []
    for layer in tqdm(layers, desc="save Stage B hidden layers", unit="layer"):
        condition_tensors = {
            condition: torch.stack(states)
            for condition, states in layer_condition_states[layer].items()
        }
        path = save_condition_hidden_layer(
            args.output_dir,
            layer,
            sample_ids,
            condition_tensors,
            args.condition_plan,
            metadata={
                "model_path": model_path,
                "model_source": args.model_source,
                "readout_position": args.readout_position,
                "condition_plan": args.condition_plan,
            },
        )
        artifacts.append(str(path))
    payload.update({"resolved_device": resolved_device, "artifacts": artifacts})
    summary_path = write_json(Path(args.output_dir) / "dump_stage_b_condition_hidden_states_summary.json", payload)
    append_experiment_log(args.log_path, "dump_stage_b_condition_hidden_states", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
