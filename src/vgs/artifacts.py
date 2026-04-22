"""Artifact readers and writers for CPU-side analysis stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vgs.io import ensure_dir


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def save_hidden_layer(
    output_dir: str | Path,
    layer: int,
    sample_ids: list[str],
    z_img: torch.Tensor,
    z_blind: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if z_img.shape != z_blind.shape:
        raise ValueError(f"z_img and z_blind shape mismatch: {z_img.shape} vs {z_blind.shape}")
    if z_img.shape[0] != len(sample_ids):
        raise ValueError("Number of sample_ids must match hidden-state rows.")
    target = ensure_dir(output_dir) / f"layer_{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "sample_ids": sample_ids,
            "z_img": z_img.detach().cpu().float(),
            "z_blind": z_blind.detach().cpu().float(),
            "metadata": metadata or {},
        },
        target,
    )
    return target


def load_hidden_layer(hidden_states_dir: str | Path, layer: int) -> dict[str, Any]:
    path = Path(hidden_states_dir) / f"layer_{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden-state artifact: {path}")
    payload = torch.load(path, map_location="cpu")
    for key in ["sample_ids", "z_img", "z_blind"]:
        if key not in payload:
            raise KeyError(f"{path} is missing required key: {key}")
    return payload


def save_difference_matrix(
    output_dir: str | Path,
    layer: int,
    sample_ids: list[str],
    matrix: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = ensure_dir(output_dir) / f"D_layer_{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "sample_ids": sample_ids,
            "D": matrix.detach().cpu().float(),
            "metadata": metadata or {},
        },
        target,
    )
    return target


def load_difference_matrix(matrix_dir: str | Path, layer: int) -> dict[str, Any]:
    path = Path(matrix_dir) / f"D_layer_{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing difference matrix: {path}")
    payload = torch.load(path, map_location="cpu")
    if "D" not in payload:
        raise KeyError(f"{path} is missing required key: D")
    return payload


def save_svd(
    output_dir: str | Path,
    layer: int,
    sample_ids: list[str],
    singular_values: torch.Tensor,
    vh: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = ensure_dir(output_dir) / f"svd_layer_{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "sample_ids": sample_ids,
            "singular_values": singular_values.detach().cpu().float(),
            "Vh": vh.detach().cpu().float(),
            "metadata": metadata or {},
        },
        target,
    )
    return target


def load_svd(svd_dir: str | Path, layer: int) -> dict[str, Any]:
    path = Path(svd_dir) / f"svd_layer_{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing SVD artifact: {path}")
    payload = torch.load(path, map_location="cpu")
    for key in ["singular_values", "Vh"]:
        if key not in payload:
            raise KeyError(f"{path} is missing required key: {key}")
    return payload
