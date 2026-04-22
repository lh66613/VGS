"""Dataset loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PopeSample:
    sample_id: str
    question_id: int | str
    family: str
    subset: str
    image: str
    image_path: str
    question: str
    label: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {target}")
        return data
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {target}:{line_no}") from exc
    return rows


def load_pope_subset(
    questions_dir: str | Path,
    images_dir: str | Path,
    family: str,
    subset: str,
    pattern: str = "{family}_pope_{subset}.json",
) -> list[PopeSample]:
    question_path = Path(questions_dir) / pattern.format(family=family, subset=subset)
    records = read_json_or_jsonl(question_path)
    samples: list[PopeSample] = []
    for idx, row in enumerate(records):
        question_id = row.get("question_id", idx)
        image = str(row["image"])
        label = str(row["label"]).strip().lower()
        sample_id = f"{family}:{subset}:{question_id}"
        samples.append(
            PopeSample(
                sample_id=sample_id,
                question_id=question_id,
                family=family,
                subset=subset,
                image=image,
                image_path=str(Path(images_dir) / image),
                question=str(row["text"]),
                label=label,
            )
        )
    return samples


def validate_pope_samples(samples: list[PopeSample]) -> dict[str, Any]:
    missing_images = [sample.image_path for sample in samples if not Path(sample.image_path).exists()]
    invalid_labels = sorted({sample.label for sample in samples if sample.label not in {"yes", "no"}})
    duplicate_ids = sorted(
        sample_id for sample_id, count in _counts([sample.sample_id for sample in samples]).items() if count > 1
    )
    labels = _counts([sample.label for sample in samples])
    subsets = _counts([sample.subset for sample in samples])
    return {
        "num_samples": len(samples),
        "num_missing_images": len(missing_images),
        "missing_images_preview": missing_images[:20],
        "invalid_labels": invalid_labels,
        "duplicate_sample_ids": duplicate_ids[:20],
        "labels": labels,
        "subsets": subsets,
        "ok": not missing_images and not invalid_labels and not duplicate_ids,
    }


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
