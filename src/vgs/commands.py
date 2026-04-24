"""CLI command implementations.

These commands are intentionally light scaffolds. They define stable argument
contracts, create reproducible run summaries, and provide TODO anchors for the
model-specific implementation that will be added as each validation stage lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from vgs.analysis import (
    analyze_k_sensitivity,
    analyze_spectra,
    build_difference_matrices,
    compare_probe_features,
    layerwise_summary,
    train_probe_models,
)
from vgs.cli import add_common_args, add_k_args, add_layer_args, resolve_k_grid, resolve_layers
from vgs.config import config_get, load_config
from vgs.constants import DEFAULT_POPE_SUBSETS
from vgs.datasets import load_pope_subset, validate_pope_samples
from vgs.io import append_experiment_log, write_json, write_jsonl
from vgs.llava_hf import extract_hidden_pair, generate_pope_answer, load_llava_hf, resolve_device
from vgs.pope import classify_outcome, parse_yes_no
from vgs.artifacts import read_jsonl, save_hidden_layer
from vgs.stage_e import run_intervention_pilot, run_intervention_precheck
from vgs.semantics import run_semantic_interpretation


def _finish(args: argparse.Namespace, stage: str, output_dir: str | Path, payload: dict[str, Any]) -> Path:
    out_dir = Path(output_dir)
    status = "dry_run" if args.dry_run else "scaffold_ready"
    summary = {
        "stage": stage,
        "status": status,
        "config": args.config,
        "seed": args.seed,
        **payload,
    }
    summary_path = write_json(out_dir / f"{stage}_summary.json", summary)
    append_experiment_log(args.log_path, stage=stage, summary_path=summary_path, status=status)
    print(summary_path)
    return summary_path


def run_pope_eval_main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaVA-1.5-7B on POPE and save predictions.")
    add_common_args(parser)
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    family = args.family or config_get(config, "dataset.pope_family", "coco")
    questions_dir = args.questions_dir or config_get(config, "dataset.questions_dir", "data/pope/questions")
    images_dir = args.images_dir or config_get(config, "dataset.images_dir", "data/pope/images")
    subsets = args.subsets or config_get(config, "dataset.subsets", DEFAULT_POPE_SUBSETS)
    pattern = config_get(config, "dataset.question_file_pattern", "{family}_pope_{subset}.json")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")

    samples = []
    for subset in tqdm(subsets, desc="load POPE subsets", unit="subset"):
        samples.extend(load_pope_subset(questions_dir, images_dir, family, subset, pattern))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    payload = {
        "model_source": args.model_source,
        "model_path": model_path,
        "family": family,
        "questions_dir": questions_dir,
        "images_dir": images_dir,
        "subsets": subsets,
        "split": args.split,
        "max_samples": args.max_samples,
        "num_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        _finish(args, "run_pope_eval", args.output_dir, payload)
        return

    if args.model_source != "hf":
        raise NotImplementedError("Only the Hugging Face LLaVA path is implemented at this stage.")
    resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    model, processor, resolved_device = load_llava_hf(
        model_path,
        device=resolved_device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
    )

    rows = []
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "unknown": 0}
    for sample in tqdm(samples, desc="POPE generation", unit="sample"):
        raw_generation = generate_pope_answer(
            model,
            processor,
            sample,
            device=resolved_device,
            max_new_tokens=args.max_new_tokens,
        )
        parsed_prediction = parse_yes_no(raw_generation)
        outcome = classify_outcome(parsed_prediction, sample.label)
        counts[outcome] += 1
        rows.append(
            {
                **sample.to_json(),
                "raw_generation": raw_generation,
                "parsed_prediction": parsed_prediction,
                "outcome": outcome,
            }
        )

    output_dir = Path(args.output_dir)
    predictions_path = write_jsonl(output_dir / "pope_predictions.jsonl", rows)
    accuracy_denominator = sum(counts[key] for key in ["TP", "TN", "FP", "FN"])
    accuracy = (
        (counts["TP"] + counts["TN"]) / accuracy_denominator if accuracy_denominator else None
    )
    payload.update(
        {
            "resolved_device": resolved_device,
            "predictions_path": str(predictions_path),
            "counts": counts,
            "accuracy": accuracy,
        }
    )
    _finish(args, "run_pope_eval", args.output_dir, payload)


def validate_pope_data_main() -> None:
    parser = argparse.ArgumentParser(description="Validate local POPE question files and image paths.")
    add_common_args(parser)
    parser.add_argument("--family", default=None, help="POPE family prefix, e.g. coco, aokvqa, gqa.")
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    family = args.family or config_get(config, "dataset.pope_family", "coco")
    questions_dir = args.questions_dir or config_get(config, "dataset.questions_dir", "data/pope/questions")
    images_dir = args.images_dir or config_get(config, "dataset.images_dir", "data/pope/images")
    subsets = args.subsets or config_get(config, "dataset.subsets", DEFAULT_POPE_SUBSETS)
    pattern = args.pattern or config_get(config, "dataset.question_file_pattern", "{family}_pope_{subset}.json")

    samples = []
    per_subset = {}
    for subset in tqdm(subsets, desc="validate POPE subsets", unit="subset"):
        subset_samples = load_pope_subset(questions_dir, images_dir, family, subset, pattern)
        samples.extend(subset_samples)
        per_subset[subset] = validate_pope_samples(subset_samples)

    validation = validate_pope_samples(samples)
    payload = {
        "family": family,
        "questions_dir": questions_dir,
        "images_dir": images_dir,
        "subsets": subsets,
        "pattern": pattern,
        "overall": validation,
        "per_subset": per_subset,
    }
    status = "ok" if validation["ok"] else "failed"
    summary_path = write_json(Path(args.output_dir) / "validate_pope_data_summary.json", payload)
    append_experiment_log(args.log_path, "validate_pope_data", summary_path, status)
    print(summary_path)


def dump_hidden_states_main() -> None:
    parser = argparse.ArgumentParser(description="Dump paired image/blind hidden states.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--readout-position", default="last_prompt_token")
    parser.add_argument("--hidden-stream", default="post_block")
    parser.add_argument("--output-dir", default="outputs/hidden_states")
    args = parser.parse_args()
    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    payload = {
        "layers": layers,
        "model_source": args.model_source,
        "model_path": model_path,
        "predictions": args.predictions,
        "max_samples": args.max_samples,
        "readout_position": args.readout_position,
        "hidden_stream": args.hidden_stream,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if args.dry_run:
        payload["todo"] = "Dry run only; no model loaded."
        _finish(args, "dump_hidden_states", args.output_dir, payload)
        return

    if args.model_source != "hf":
        raise NotImplementedError("Only the Hugging Face LLaVA path is implemented at this stage.")
    resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    model, processor, resolved_device = load_llava_hf(
        model_path,
        device=resolved_device,
        torch_dtype=torch_dtype,
        allow_cpu=args.allow_cpu,
    )
    rows = read_jsonl(args.predictions)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    layer_img = {layer: [] for layer in layers}
    layer_blind = {layer: [] for layer in layers}
    sample_ids = []
    for row in tqdm(rows, desc="dump hidden states", unit="sample"):
        sample_ids.append(str(row["sample_id"]))
        pairs = extract_hidden_pair(
            model,
            processor,
            row,
            layers,
            resolved_device,
            readout_position=args.readout_position,
        )
        for layer, (z_img, z_blind) in pairs.items():
            layer_img[layer].append(z_img)
            layer_blind[layer].append(z_blind)

    artifacts = []
    for layer in tqdm(layers, desc="save hidden layers", unit="layer"):
        path = save_hidden_layer(
            args.output_dir,
            layer,
            sample_ids,
            z_img=torch.stack(layer_img[layer]),
            z_blind=torch.stack(layer_blind[layer]),
            metadata={
                "model_path": model_path,
                "model_source": args.model_source,
                "readout_position": args.readout_position,
                "hidden_stream": args.hidden_stream,
                "predictions": args.predictions,
            },
        )
        artifacts.append(str(path))
    payload.update(
        {
            "resolved_device": resolved_device,
            "num_samples": len(sample_ids),
            "artifacts": artifacts,
        }
    )
    _finish(
        args,
        "dump_hidden_states",
        args.output_dir,
        payload,
    )


def build_difference_matrix_main() -> None:
    parser = argparse.ArgumentParser(description="Build D_layer_l = z_blind - z_img matrices.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument(
        "--control",
        choices=["none", "shuffle_image_question", "shuffle_blind_image", "gaussian"],
        default="none",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/svd")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "hidden_states_dir": args.hidden_states_dir,
        "control": args.control,
        "split": args.split,
    }
    if not args.dry_run:
        rows = build_difference_matrices(
            resolve_layers(args),
            args.hidden_states_dir,
            args.output_dir,
            args.control,
            args.seed,
        )
        payload["artifacts"] = rows
    _finish(
        args,
        "build_difference_matrix",
        args.output_dir,
        payload,
    )


def analyze_spectrum_main() -> None:
    parser = argparse.ArgumentParser(description="Analyze singular spectra and effective ranks.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--matrix-dir", default="outputs/svd")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--output-dir", default="outputs/svd")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "matrix_dir": args.matrix_dir,
        "plot_dir": args.plot_dir,
    }
    if not args.dry_run:
        payload["summary_rows"] = analyze_spectra(
            resolve_layers(args), args.matrix_dir, args.output_dir, args.plot_dir
        )
    _finish(
        args,
        "analyze_spectrum",
        args.output_dir,
        payload,
    )


def analyze_k_sensitivity_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stability and explained variance over K.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--matrix-dir", default="outputs/svd")
    parser.add_argument("--probe-dir", default="outputs/probes")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--stability-sample-size",
        type=int,
        default=1024,
        help="Rows used for split-half stability estimation; <=0 means all rows.",
    )
    parser.add_argument(
        "--stability-method",
        choices=["randomized", "exact"],
        default="randomized",
        help="Method for split-half top-K subspace estimation.",
    )
    parser.add_argument("--output-dir", default="outputs/svd")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "svd_dir": args.svd_dir,
        "matrix_dir": args.matrix_dir,
        "probe_dir": args.probe_dir,
        "repeats": args.repeats,
        "stability_method": args.stability_method,
        "stability_sample_size": args.stability_sample_size,
    }
    if not args.dry_run:
        payload["summary_rows"] = analyze_k_sensitivity(
            resolve_layers(args),
            resolve_k_grid(args),
            args.svd_dir,
            args.matrix_dir,
            args.output_dir,
            args.plot_dir,
            args.seed,
            args.repeats,
            args.stability_method,
            None if args.stability_sample_size <= 0 else args.stability_sample_size,
        )
    _finish(
        args,
        "analyze_k_sensitivity",
        args.output_dir,
        payload,
    )


def train_probe_main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight probes for hallucination detection.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.add_argument("--feature-family", nargs="+", default=["projected_difference"])
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--output-dir", default="outputs/probes")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "feature_family": args.feature_family,
        "predictions": args.predictions,
        "hidden_states_dir": args.hidden_states_dir,
        "svd_dir": args.svd_dir,
    }
    if not args.dry_run:
        payload["summary_rows"] = train_probe_models(
            resolve_layers(args),
            resolve_k_grid(args),
            args.feature_family,
            args.predictions,
            args.hidden_states_dir,
            args.svd_dir,
            args.output_dir,
            args.seed,
        )
    _finish(
        args,
        "train_probe",
        args.output_dir,
        payload,
    )


def compare_features_main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw, difference, projected, and control features.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.add_argument("--probe-dir", default="outputs/probes")
    parser.add_argument("--output-dir", default="outputs/probes")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "probe_dir": args.probe_dir,
    }
    if not args.dry_run:
        payload["summary_rows"] = compare_probe_features(args.probe_dir, args.output_dir)
    _finish(
        args,
        "compare_features",
        args.output_dir,
        payload,
    )


def layerwise_analysis_main() -> None:
    parser = argparse.ArgumentParser(description="Create layerwise geometry and information-flow summaries.")
    add_common_args(parser)
    add_layer_args(parser)
    add_k_args(parser)
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--probe-dir", default="outputs/probes")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--output-dir", default="outputs/svd")
    args = parser.parse_args()
    payload = {
        "layers": resolve_layers(args),
        "k_grid": resolve_k_grid(args),
        "svd_dir": args.svd_dir,
        "probe_dir": args.probe_dir,
        "plot_dir": args.plot_dir,
    }
    if not args.dry_run:
        payload.update(
            layerwise_summary(
                resolve_layers(args),
                resolve_k_grid(args),
                args.svd_dir,
                args.probe_dir,
                args.output_dir,
                args.plot_dir,
            )
        )
    _finish(
        args,
        "layerwise_analysis",
        args.output_dir,
        payload,
    )


def intervention_precheck_main() -> None:
    parser = argparse.ArgumentParser(description="Check whether activation interventions are feasible.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--random-scale", type=float, default=5.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--readout-position", default="last_prompt_token")
    parser.add_argument("--output-dir", default="outputs/interventions")
    args = parser.parse_args()
    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    payload = {
        "layers": layers,
        "model_source": args.model_source,
        "model_path": model_path,
        "predictions": args.predictions,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "random_scale": args.random_scale,
        "device": args.device,
        "torch_dtype": torch_dtype,
        "readout_position": args.readout_position,
    }
    if not args.dry_run:
        if args.model_source != "hf":
            raise NotImplementedError("Only the Hugging Face LLaVA path is implemented for Stage E.")
        resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
        model, processor, resolved_device = load_llava_hf(
            model_path,
            device=resolved_device,
            torch_dtype=torch_dtype,
            allow_cpu=args.allow_cpu,
        )
        rows = read_jsonl(args.predictions)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        payload.update({"resolved_device": resolved_device})
        payload.update(
            run_intervention_precheck(
                model,
                processor,
                rows,
                layers,
                resolved_device,
                args.output_dir,
                args.seed,
                args.max_new_tokens,
                args.random_scale,
            )
        )
    _finish(
        args,
        "intervention_precheck",
        args.output_dir,
        payload,
    )


def intervention_pilot_main() -> None:
    parser = argparse.ArgumentParser(description="Run causal ablation/rescue intervention pilot.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.set_defaults(layers=["24"])
    parser.add_argument("--model-source", choices=["hf", "official"], default="hf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-samples-per-outcome", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha-grid", nargs="*", default=["4.0", "5.0", "6.0", "7.0", "8.0"])
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--outcomes", nargs="*", default=["TN", "FP"])
    parser.add_argument("--families", nargs="*", default=["tail", "rescue"], choices=["tail", "rescue"])
    parser.add_argument("--granularities", nargs="*", default=["last_token"], choices=["last_token", "full_sequence", "generated_token"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--output-dir", default="outputs/interventions")
    args = parser.parse_args()
    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    torch_dtype = args.torch_dtype or config_get(config, "model.torch_dtype", "float16")
    layers = resolve_layers(args)
    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    alpha_grid = [float(item) for value in args.alpha_grid for item in value.split(",") if item]
    payload = {
        "layers": layers,
        "model_source": args.model_source,
        "model_path": model_path,
        "max_samples_per_outcome": args.max_samples_per_outcome,
        "max_new_tokens": args.max_new_tokens,
        "alpha_grid": alpha_grid,
        "tail_band": args.tail_band,
        "outcomes": args.outcomes,
        "families": args.families,
        "granularities": args.granularities,
        "svd_dir": args.svd_dir,
        "hidden_states_dir": args.hidden_states_dir,
        "condition_plan": args.condition_plan,
        "condition_hidden_dir": args.condition_hidden_dir,
        "predictions": args.predictions,
        "device": args.device,
        "torch_dtype": torch_dtype,
    }
    if not args.dry_run:
        if args.model_source != "hf":
            raise NotImplementedError("Only the Hugging Face LLaVA path is implemented for Stage E.")
        resolved_device = resolve_device(args.device, allow_cpu=args.allow_cpu)
        model, processor, resolved_device = load_llava_hf(
            model_path,
            device=resolved_device,
            torch_dtype=torch_dtype,
            allow_cpu=args.allow_cpu,
        )
        rows = read_jsonl(args.predictions)
        payload.update({"resolved_device": resolved_device})
        payload.update(
            run_intervention_pilot(
                model,
                processor,
                rows,
                layers,
                resolved_device,
                args.output_dir,
                args.svd_dir,
                args.hidden_states_dir,
                args.seed,
                args.max_new_tokens,
                args.max_samples_per_outcome,
                alpha_grid,
                (tail_start, tail_end),
                args.outcomes,
                args.families,
                args.granularities,
                args.condition_plan,
                args.condition_hidden_dir,
            )
        )
    _finish(
        args,
        "intervention_pilot",
        args.output_dir,
        payload,
    )


def semantic_interpretation_main() -> None:
    parser = argparse.ArgumentParser(description="Interpret top singular directions semantically.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--tail-layer", type=int, default=24)
    parser.add_argument("--tail-band", default="257-1024")
    parser.add_argument("--rescue-layer", type=int, default=32)
    parser.add_argument("--svd-dir", default="outputs/svd")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--condition-plan", default="outputs/stage_b/stage_b_condition_plan.jsonl")
    parser.add_argument("--condition-hidden-dir", default="outputs/stage_b_hidden")
    parser.add_argument("--apply-final-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-token-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--natural-token-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/semantics")
    args = parser.parse_args()
    config = load_config(args.config)
    model_path = args.model_path or config_get(config, "model.checkpoint_path")
    tail_start, tail_end = [int(item) for item in args.tail_band.split("-", 1)]
    payload = {
        "layers": resolve_layers(args),
        "k": args.k,
        "top_n": args.top_n,
        "model_path": model_path,
        "svd_dir": args.svd_dir,
        "hidden_states_dir": args.hidden_states_dir,
        "predictions": args.predictions,
        "tail_layer": args.tail_layer,
        "tail_band": args.tail_band,
        "rescue_layer": args.rescue_layer,
        "condition_plan": args.condition_plan,
        "condition_hidden_dir": args.condition_hidden_dir,
        "apply_final_norm": args.apply_final_norm,
        "normalize_token_vectors": args.normalize_token_vectors,
        "natural_token_filter": args.natural_token_filter,
    }
    if not args.dry_run:
        payload.update(
            run_semantic_interpretation(
                layers=resolve_layers(args),
                k=args.k,
                model_path=model_path,
                svd_dir=args.svd_dir,
                hidden_states_dir=args.hidden_states_dir,
                predictions_path=args.predictions,
                output_dir=args.output_dir,
                tail_layer=args.tail_layer,
                tail_band=(tail_start, tail_end),
                rescue_layer=args.rescue_layer,
                top_n=args.top_n,
                apply_final_norm=args.apply_final_norm,
                normalize_token_vectors=args.normalize_token_vectors,
                natural_token_filter=args.natural_token_filter,
                condition_plan_path=args.condition_plan,
                condition_hidden_dir=args.condition_hidden_dir,
            )
        )
    _finish(
        args,
        "semantic_interpretation",
        args.output_dir,
        payload,
    )


def chair_sanity_check_main() -> None:
    parser = argparse.ArgumentParser(description="Optional caption benchmark sanity check.")
    add_common_args(parser)
    add_layer_args(parser)
    parser.add_argument("--chair-dir", default="data/chair_optional")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/sanity_checks")
    args = parser.parse_args()
    _finish(
        args,
        "chair_sanity_check",
        args.output_dir,
        {
            "layers": resolve_layers(args),
            "chair_dir": args.chair_dir,
            "k": args.k,
            "todo": "Run a small caption sanity check after the POPE chain is stable.",
        },
    )
