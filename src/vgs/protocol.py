"""Protocol-locking utilities for paper-upgrade experiments."""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vgs.artifacts import load_hidden_layer, read_jsonl
from vgs.io import ensure_dir, write_csv, write_json


def prepare_stage_i_protocol(
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    output_dir: str | Path,
    notes_dir: str | Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    layers: list[int],
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    split_rows = _make_stratified_splits(rows, seed, train_frac, val_frac)
    output_root = ensure_dir(output_dir)
    notes_root = ensure_dir(notes_dir)

    split_ids = {split: [row["sample_id"] for row in split_rows if row["split"] == split] for split in ["train", "val", "test"]}
    split_paths = {}
    for split, sample_ids in split_ids.items():
        path = write_json(output_root / f"pope_{split}_ids.json", {"split": split, "sample_ids": sample_ids})
        split_paths[split] = str(path)

    summary_rows = _split_summary(split_rows)
    summary_path = write_csv(output_root / "split_summary.csv", summary_rows, _fieldnames(summary_rows))

    protocol_path = notes_root / "protocol_lock.md"
    protocol_path.write_text(
        _protocol_lock_markdown(
            predictions_path,
            hidden_states_dir,
            split_paths,
            summary_rows,
            seed,
            train_frac,
            val_frac,
        ),
        encoding="utf-8",
    )

    prompt_path = notes_root / "prompt_templates.md"
    image_prompt, blind_prompt = _prompt_templates()
    prompt_path.write_text(_prompt_templates_markdown(image_prompt, blind_prompt), encoding="utf-8")

    diff_path = ensure_dir("outputs/sanity_checks") / "prompt_template_diff.txt"
    diff_path.write_text(_prompt_diff(image_prompt, blind_prompt), encoding="utf-8")

    readout_path = notes_root / "hidden_readout_protocol.md"
    readout_path.write_text(
        _hidden_readout_markdown(hidden_states_dir, layers),
        encoding="utf-8",
    )

    return {
        "num_predictions": len(rows),
        "split_paths": split_paths,
        "split_summary_path": str(summary_path),
        "protocol_lock_path": str(protocol_path),
        "prompt_templates_path": str(prompt_path),
        "prompt_template_diff_path": str(diff_path),
        "hidden_readout_protocol_path": str(readout_path),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
    }


def _make_stratified_splits(
    rows: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> list[dict[str, Any]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be smaller than 1.")

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("subset", "")),
            str(row.get("label", "")),
            str(row.get("outcome", "")),
        )
        groups[key].append(row)

    split_rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        n = len(group)
        train_n = int(round(n * train_frac))
        val_n = int(round(n * val_frac))
        if n >= 3:
            train_n = min(max(train_n, 1), n - 2)
            val_n = min(max(val_n, 1), n - train_n - 1)
        else:
            train_n = min(train_n, n)
            val_n = min(val_n, n - train_n)
        boundaries = {
            "train": train_n,
            "val": train_n + val_n,
            "test": n,
        }
        for idx, row in enumerate(group):
            if idx < boundaries["train"]:
                split = "train"
            elif idx < boundaries["val"]:
                split = "val"
            else:
                split = "test"
            split_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "split": split,
                    "subset": row.get("subset", ""),
                    "label": row.get("label", ""),
                    "outcome": row.get("outcome", ""),
                }
            )
    return sorted(split_rows, key=lambda row: row["sample_id"])


def _split_summary(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            row["split"],
            row["subset"],
            row["label"],
            row["outcome"],
        )
        for row in split_rows
    )
    rows = []
    for (split, subset, label, outcome), count in sorted(counts.items()):
        rows.append(
            {
                "split": split,
                "subset": subset,
                "label": label,
                "outcome": outcome,
                "count": count,
            }
        )
    for split in ["train", "val", "test"]:
        rows.append(
            {
                "split": split,
                "subset": "ALL",
                "label": "ALL",
                "outcome": "ALL",
                "count": sum(1 for row in split_rows if row["split"] == split),
            }
        )
    return rows


def _protocol_lock_markdown(
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    split_paths: dict[str, str],
    summary_rows: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> str:
    totals = {row["split"]: row["count"] for row in summary_rows if row["subset"] == "ALL"}
    return "\n".join(
        [
            "# Protocol Lock",
            "",
            "## Fixed Artifacts",
            "",
            f"- Predictions: `{predictions_path}`",
            f"- Hidden states: `{hidden_states_dir}`",
            f"- Train IDs: `{split_paths['train']}`",
            f"- Validation IDs: `{split_paths['val']}`",
            f"- Test IDs: `{split_paths['test']}`",
            "",
            "## Split Policy",
            "",
            f"- Seed: `{seed}`",
            f"- Train / validation / test fractions: `{train_frac:.2f}` / `{val_frac:.2f}` / `{1.0 - train_frac - val_frac:.2f}`",
            "- Stratification keys: `subset`, `label`, `outcome`.",
            "- Test labels must not be used for subspace extraction, classifier fitting, or intervention memory-bank construction.",
            "",
            "## Split Counts",
            "",
            "| Split | Count |",
            "| --- | ---: |",
            f"| train | {totals.get('train', 0)} |",
            f"| val | {totals.get('val', 0)} |",
            f"| test | {totals.get('test', 0)} |",
            "",
        ]
    )


def _prompt_templates() -> tuple[str, str]:
    image_prompt = "USER: <image>\n{question} Answer with yes or no only. ASSISTANT:"
    blind_prompt = "USER: {question} Answer with yes or no only. ASSISTANT:"
    return image_prompt, blind_prompt


def _prompt_templates_markdown(image_prompt: str, blind_prompt: str) -> str:
    return "\n".join(
        [
            "# Prompt Templates",
            "",
            "## Image + Question",
            "",
            "```text",
            image_prompt,
            "```",
            "",
            "## Blind / Text Only",
            "",
            "```text",
            blind_prompt,
            "```",
            "",
            "## Current Check",
            "",
            "- Text instruction is identical: `Answer with yes or no only.`",
            "- Difference intended by the current HF path: image prompt includes `<image>` and image pixels; blind prompt omits both.",
            "- The exact image prompt may be expanded by `processor.apply_chat_template` at runtime; keep the processor/model version fixed in reported runs.",
            "",
        ]
    )


def _prompt_diff(image_prompt: str, blind_prompt: str) -> str:
    diff = difflib.unified_diff(
        blind_prompt.splitlines(),
        image_prompt.splitlines(),
        fromfile="blind_prompt",
        tofile="image_prompt",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def _hidden_readout_markdown(hidden_states_dir: str | Path, layers: list[int]) -> str:
    lines = [
        "# Hidden Readout Protocol",
        "",
        "## Current Primary Readout",
        "",
        "- Hidden stream: `post_block` transformer hidden-state tuple from HuggingFace `LlavaForConditionalGeneration`.",
        "- Current token position: `last_prompt_token`.",
        "- Image path: full multimodal model forward with image tokens and text prompt.",
        "- Blind path: language model forward on the text-only prompt.",
        "- Difference convention: `D = z_blind - z_img`.",
        "",
        "## Observed Artifact Metadata",
        "",
        "| Layer | Readout position | Hidden stream | Samples | Hidden dim |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for layer in layers:
        try:
            payload = load_hidden_layer(hidden_states_dir, layer)
        except FileNotFoundError:
            lines.append(f"| {layer} | missing | missing | 0 | 0 |")
            continue
        metadata = payload.get("metadata", {})
        z_img = payload["z_img"]
        if isinstance(z_img, torch.Tensor):
            shape = z_img.shape
            n, dim = int(shape[0]), int(shape[1])
        else:
            n, dim = 0, 0
        lines.append(
            f"| {layer} | {metadata.get('readout_position', 'unknown')} | {metadata.get('hidden_stream', 'unknown')} | {n} | {dim} |"
        )
    lines.extend(
        [
            "",
            "## Pending Robustness Positions",
            "",
            "- `first_answer_prefill`",
            "- `last_4_prompt_mean`",
            "- `last_8_prompt_mean`",
            "- `question_object_token_mean` if object-token localization is implemented.",
            "- `image_adjacent_text_token` if accessible in the chosen implementation.",
            "",
        ]
    )
    return "\n".join(lines)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []
