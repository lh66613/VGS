"""Small IO helpers shared by pipeline entry points."""

from __future__ import annotations

import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return target


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    return target


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def append_experiment_log(
    log_path: str | Path,
    stage: str,
    summary_path: str | Path,
    status: str,
) -> None:
    path = Path(log_path)
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text(
            "# Experiment Log\n\n"
            "| Time UTC | Stage | Status | Command | Summary |\n"
            "| --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )

    command = current_command().replace("|", "\\|")
    line = f"| {now_utc()} | {stage} | {status} | `{command}` | {summary_path} |\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
