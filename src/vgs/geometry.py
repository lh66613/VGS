"""Numerical geometry utilities for subspace validation."""

from __future__ import annotations

import numpy as np


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    total = values.sum()
    if total <= eps:
        return 0.0
    probs = values / total
    entropy = -(probs * np.log(probs + eps)).sum()
    return float(np.exp(entropy))


def cumulative_explained_variance(singular_values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(singular_values, dtype=np.float64) ** 2
    total = values.sum()
    if total <= eps:
        return np.zeros_like(values)
    return np.cumsum(values) / total


def projection_similarity(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    a = np.asarray(basis_a, dtype=np.float64)
    b = np.asarray(basis_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Subspace bases must be 2D arrays.")
    k = min(a.shape[1], b.shape[1])
    if k == 0:
        return 0.0
    overlap = a[:, :k].T @ b[:, :k]
    return float(np.sum(overlap * overlap) / k)
