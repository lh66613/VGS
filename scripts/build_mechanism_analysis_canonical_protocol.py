#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_json


CANONICAL_PACKAGES = ["torch", "transformers", "tokenizers", "numpy", "pandas", "safetensors"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the canonical protocol for frozen mechanism analysis.")
    parser.add_argument("--model-path", default="/data/lh/ModelandDataset/llava-1.5-7b-hf")
    parser.add_argument("--canonical-python", default="/data/lh/.conda/envs/after/bin/python")
    parser.add_argument("--hidden-cache", default="outputs/hidden_states/layer_24.pt")
    parser.add_argument("--svd-input", default="outputs/svd/D_layer_24.pt")
    parser.add_argument("--svd", default="outputs/svd/svd_layer_24.pt")
    parser.add_argument(
        "--reference-geometry",
        default="outputs/mechanism_mitigation/operator_geometry/operator_geometry.csv",
    )
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--margin-scores", default="outputs/margins/pope_margin_scores.csv")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument(
        "--output-path",
        default="outputs/mechanism_mitigation/mechanism_analysis/canonical_protocol.md",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_protocol(
        model_path=Path(args.model_path),
        canonical_python=Path(args.canonical_python),
        hidden_cache=Path(args.hidden_cache),
        svd_input=Path(args.svd_input),
        svd=Path(args.svd),
        reference_geometry=Path(args.reference_geometry),
        predictions=Path(args.predictions),
        margin_scores=Path(args.margin_scores),
        split_dir=Path(args.split_dir),
        output_path=Path(args.output_path),
    )
    summary_path = write_json(Path(args.output_path).with_suffix(".summary.json"), result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_canonical_protocol", summary_path, "ok")
    print(args.output_path)


def build_protocol(
    model_path: Path,
    canonical_python: Path,
    hidden_cache: Path,
    svd_input: Path,
    svd: Path,
    reference_geometry: Path,
    predictions: Path,
    margin_scores: Path,
    split_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    current_env = _package_versions(sys.executable)
    canonical_env = _package_versions(str(canonical_python)) if canonical_python.exists() else {}
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    checksums = {
        "model_config": _sha256(model_path / "config.json"),
        "tokenizer_json_vocab_hash": _sha256(model_path / "tokenizer.json"),
        "tokenizer_model_hash": _sha256(model_path / "tokenizer.model"),
        "hidden_cache_layer24": _sha256(hidden_cache),
        "svd_input_D_layer24": _sha256(svd_input),
        "svd_layer24": _sha256(svd),
        "reference_geometry": _sha256(reference_geometry),
        "predictions": _sha256(predictions),
        "margin_scores": _sha256(margin_scores),
        "split_train_ids": _sha256(split_dir / "pope_train_ids.json"),
        "split_val_ids": _sha256(split_dir / "pope_val_ids.json"),
        "split_test_ids": _sha256(split_dir / "pope_test_ids.json"),
    }
    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "model_revision": _model_revision(model_path),
        "model_config_name": config.get("model_type", ""),
        "text_backbone": config.get("text_config", {}).get("_name_or_path", ""),
        "canonical_python": str(canonical_python),
        "canonical_environment": canonical_env,
        "current_environment": current_env,
        "checksums": checksums,
        "yes_token_ids": [3869, 4874, 22483],
        "no_token_ids": [694, 1939, 11698],
        "readout_token": "last_prompt_token",
        "difference_convention": "operator geometry uses delta = z_img - z_blind; SVD input was D = z_blind - z_img",
        "alpha_selection_rule": "On fixed IDs, choose alpha per subspace on validation/calibration by max(fp_reduction - tp_damage) among TP preserved >= 0.95; evaluate once on held-out test IDs.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown(payload), encoding="utf-8")
    return payload | {"output_path": str(output_path)}


def _markdown(payload: dict[str, Any]) -> str:
    canonical = payload["canonical_environment"]
    current = payload["current_environment"]
    checksums = payload["checksums"]
    lines = [
        "# Canonical Mechanism-Analysis Protocol",
        "",
        "Unless otherwise stated, all main results use the frozen exact-reproduction pipeline.",
        "",
        "The frozen reference is the `after` conda environment with cached hidden states and reference geometry that exactly reproduces the paper-ready Band5-16 ICD result. Results from `vlm-exp` / Transformers 4.52.4 are retained only as environment-sensitivity and implementation-drift audits.",
        "",
        "## Fixed Protocol",
        "",
        "| Item | Frozen value |",
        "| --- | --- |",
        f"| model checkpoint | `{payload['model_path']}` |",
        f"| checkpoint revision | `{payload['model_revision']}` |",
        f"| model config | `{payload['model_config_name']}` / text backbone `{payload['text_backbone']}` |",
        f"| canonical python | `{payload['canonical_python']}` |",
        f"| transformers | `{canonical.get('transformers', 'unknown')}` |",
        f"| torch | `{canonical.get('torch', 'unknown')}` |",
        f"| tokenizers | `{canonical.get('tokenizers', 'unknown')}` |",
        "| image prompt template | `USER: <image>\\n{question} Answer with yes or no only. ASSISTANT:` via the fixed LLaVA processor/chat template |",
        "| blind prompt template | `USER: {question} Answer with yes or no only. ASSISTANT:` |",
        f"| readout token | `{payload['readout_token']}` |",
        f"| yes token ids | `{' '.join(map(str, payload['yes_token_ids']))}` |",
        f"| no token ids | `{' '.join(map(str, payload['no_token_ids']))}` |",
        f"| hidden difference convention | `{payload['difference_convention']}` |",
        f"| alpha selection rule | `{payload['alpha_selection_rule']}` |",
        "",
        "## Frozen Artifact Checksums",
        "",
        "| Artifact | SHA256 |",
        "| --- | --- |",
    ]
    for key, value in checksums.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Environment Boundary",
            "",
            "| Package | Canonical `after` | Current `vlm-exp` |",
            "| --- | --- | --- |",
        ]
    )
    for package in CANONICAL_PACKAGES:
        lines.append(f"| {package} | `{canonical.get(package, 'unknown')}` | `{current.get(package, 'unknown')}` |")
    lines.extend(
        [
            "",
            "## Reporting Rule",
            "",
            "- Main tables and mechanism figures should use the frozen exact-reproduction geometry/cache unless a table is explicitly marked as an environment-sensitivity result.",
            "- The `vlm-exp` expanded geometry remains useful for drift diagnosis, but should not be mixed into the main frozen tables.",
            "",
        ]
    )
    return "\n".join(lines)


def _package_versions(python_bin: str) -> dict[str, str]:
    if python_bin == sys.executable:
        out: dict[str, str] = {}
        for package in CANONICAL_PACKAGES:
            try:
                out[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                out[package] = "missing"
        return out
    code = (
        "import importlib.metadata as m, json; "
        f"pkgs={CANONICAL_PACKAGES!r}; "
        "out={};\n"
        "for p in pkgs:\n"
        "    try:\n"
        "        out[p]=m.version(p)\n"
        "    except Exception:\n"
        "        out[p]='missing'\n"
        "print(json.dumps(out))"
    )
    try:
        completed = subprocess.run(
            [python_bin, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
    except Exception:
        return {}


def _model_revision(model_path: Path) -> str:
    for candidate in [model_path / "refs" / "main", model_path / ".git" / "HEAD"]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return "local snapshot; no revision metadata found"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
