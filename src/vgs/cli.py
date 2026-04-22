"""Shared CLI construction for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from vgs.constants import DEFAULT_K_GRID, DEFAULT_LAYERS, EXPERIMENT_LOG


def comma_or_space_ints(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item) for item in value.split(",") if item)
    return parsed


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-path", default=EXPERIMENT_LOG, help="Experiment log markdown path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate args without heavy compute.")


def add_layer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Layer list, either space separated or comma separated.",
    )


def add_k_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--k-grid",
        nargs="*",
        default=None,
        help="K values, either space separated or comma separated.",
    )


def resolve_layers(args: argparse.Namespace) -> list[int]:
    return comma_or_space_ints(args.layers, DEFAULT_LAYERS)


def resolve_k_grid(args: argparse.Namespace) -> list[int]:
    return comma_or_space_ints(args.k_grid, DEFAULT_K_GRID)


def path_arg(value: str) -> Path:
    return Path(value).expanduser()
