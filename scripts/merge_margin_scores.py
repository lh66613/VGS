#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge first-token margin score CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input margin score CSV files.")
    parser.add_argument("--output", required=True, help="Merged output CSV path.")
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="last",
        help="Duplicate sample_id policy.",
    )
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    merged = pd.concat(frames, ignore_index=True)
    if "sample_id" not in merged.columns:
        raise ValueError("Merged margin table must contain a sample_id column.")
    merged = merged.drop_duplicates(subset=["sample_id"], keep=args.keep)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
