"""Typed records shared across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


PredictionOutcome = Literal["TP", "TN", "FP", "FN", "unknown"]


@dataclass(frozen=True)
class PopePrediction:
    sample_id: str
    subset: str
    question: str
    image_path: str
    ground_truth: str
    raw_generation: str
    parsed_prediction: str
    outcome: PredictionOutcome

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HiddenStateRecord:
    sample_id: str
    subset: str
    layer: int
    readout_position: str
    image_tensor_path: str
    blind_tensor_path: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
