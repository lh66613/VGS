"""Generate tiny deterministic artifacts for validating the CPU pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from vgs.artifacts import save_hidden_layer
from vgs.io import write_jsonl


def create_smoke_artifacts(
    layers: list[int],
    num_samples: int,
    hidden_dim: int,
    predictions_path: str | Path,
    hidden_states_dir: str | Path,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    sample_ids = [f"smoke:{idx:04d}" for idx in range(num_samples)]
    rows = []
    labels = torch.zeros(num_samples)
    for idx, sample_id in enumerate(tqdm(sample_ids, desc="create smoke predictions", unit="sample")):
        is_fp = idx % 4 == 0
        is_tn = not is_fp
        labels[idx] = 1.0 if is_fp else 0.0
        rows.append(
            {
                "sample_id": sample_id,
                "family": "smoke",
                "subset": "random",
                "question_id": idx,
                "image": f"smoke_{idx:04d}.jpg",
                "image_path": f"data/pope/images/smoke_{idx:04d}.jpg",
                "question": "Is this a smoke-test sample?",
                "label": "no",
                "raw_generation": "yes" if is_fp else "no",
                "parsed_prediction": "yes" if is_fp else "no",
                "outcome": "FP" if is_fp else "TN",
            }
        )
    write_jsonl(predictions_path, rows)

    rank = min(4, hidden_dim)
    basis = torch.linalg.qr(torch.randn(hidden_dim, rank, generator=generator)).Q
    hidden_paths = []
    for layer in tqdm(layers, desc="create smoke hidden states", unit="layer"):
        z_img = torch.randn(num_samples, hidden_dim, generator=generator)
        coefficients = torch.randn(num_samples, rank, generator=generator)
        coefficients[:, 0] += labels * (2.0 + layer / 100.0)
        noise = 0.05 * torch.randn(num_samples, hidden_dim, generator=generator)
        z_blind = z_img + coefficients @ basis.T + noise
        path = save_hidden_layer(
            hidden_states_dir,
            layer,
            sample_ids,
            z_img,
            z_blind,
            metadata={"source": "smoke", "hidden_dim": hidden_dim},
        )
        hidden_paths.append(str(path))
    return {
        "predictions_path": str(predictions_path),
        "hidden_paths": hidden_paths,
        "num_samples": num_samples,
        "hidden_dim": hidden_dim,
        "layers": layers,
    }
