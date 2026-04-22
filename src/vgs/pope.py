"""POPE-specific parsing and label helpers."""

from __future__ import annotations

from vgs.schema import PredictionOutcome


YES_TOKENS = {"yes", "yeah", "yep", "true"}
NO_TOKENS = {"no", "nope", "false"}


def parse_yes_no(text: str) -> str:
    """Return a normalized yes/no label from model output, or unknown."""
    normalized = text.strip().lower()
    if not normalized:
        return "unknown"
    first = normalized.replace(".", " ").replace(",", " ").split()[0]
    if first in YES_TOKENS:
        return "yes"
    if first in NO_TOKENS:
        return "no"
    return "unknown"


def classify_outcome(prediction: str, ground_truth: str) -> PredictionOutcome:
    pred = prediction.strip().lower()
    gold = ground_truth.strip().lower()
    if pred not in {"yes", "no"} or gold not in {"yes", "no"}:
        return "unknown"
    if pred == "yes" and gold == "yes":
        return "TP"
    if pred == "no" and gold == "no":
        return "TN"
    if pred == "yes" and gold == "no":
        return "FP"
    if pred == "no" and gold == "yes":
        return "FN"
    return "unknown"
