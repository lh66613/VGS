#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_csv, write_json

from mechanism_analysis_common import fieldnames, markdown_table


METHOD_ORDER = [
    "Base",
    "Always ICD",
    "Full ICD TP-safe",
    "Band5-16 ICD",
    "Random12 ICD",
    "Tail VCD-diffusion",
    "Full VCD-diffusion",
    "Gated ICD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze paper-ready mitigation baselines.")
    parser.add_argument(
        "--paper-tables-dir",
        default="outputs/mechanism_mitigation/paper_tables",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mechanism_mitigation/mechanism_analysis/frozen_baseline",
    )
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    result = build_freeze_table(Path(args.paper_tables_dir), Path(args.output_dir))
    summary_path = write_json(Path(args.output_dir) / "build_mechanism_analysis_freeze_table_summary.json", result)
    append_experiment_log(args.log_path, "build_mechanism_analysis_freeze_table", summary_path, "ok")
    print(summary_path)


def build_freeze_table(paper_dir: Path, output_dir: Path) -> dict[str, Any]:
    table_a = pd.read_csv(paper_dir / "table_a_tp_safe_mitigation.csv")
    table_c = pd.read_csv(paper_dir / "table_c_no_bias_audit.csv")
    table_e = pd.read_csv(paper_dir / "table_e_best_vs_vcd_baseline.csv")

    by_method = {str(row.method): row._asdict() for row in table_e.itertuples(index=False)}
    by_a = {str(row.method): row._asdict() for row in table_a.itertuples(index=False)}
    by_c = {str(row.method): row._asdict() for row in table_c.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        source_method = method
        if method == "Random12 ICD":
            source_method = "Random12 mean"
        metric_source = by_method.get(method) or by_a.get(source_method) or {}
        audit_source = by_c.get(method) or by_c.get(source_method) or {}
        rows.append(
            {
                "method": method,
                "fp_reduction": _value(metric_source, "fp_reduction"),
                "tp_preserved": _value(metric_source, "tp_preserved"),
                "accuracy_delta": _value(metric_source, "accuracy_delta"),
                "overall_yes_rate": _value(metric_source, "overall_yes_rate", audit_source),
                "fp_yes_rate": _value(metric_source, "fp_yes_rate", audit_source),
                "tp_yes_rate": _value(audit_source, "tp_yes_rate"),
                "tn_yes_rate": _value(audit_source, "tn_yes_rate"),
                "notes": _value(metric_source, "takeaway")
                or _value(metric_source, "notes")
                or _value(by_a.get(source_method, {}), "notes"),
            }
        )

    csv_path = write_csv(output_dir / "frozen_main_table.csv", rows, fieldnames(rows))
    note = [
        "# Frozen LLaVA-7B Main Table",
        "",
        "Source: existing paper-ready outputs under `outputs/mechanism_mitigation/paper_tables/`.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
    ]
    md_path = output_dir / "frozen_main_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(note), encoding="utf-8")
    return {
        "paper_tables_dir": str(paper_dir),
        "frozen_main_table_path": str(csv_path),
        "frozen_main_table_markdown_path": str(md_path),
        "num_rows": len(rows),
    }


def _value(primary: dict[str, Any], key: str, fallback: dict[str, Any] | None = None) -> Any:
    value = primary.get(key, None)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = (fallback or {}).get(key, None)
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else value


if __name__ == "__main__":
    main()
