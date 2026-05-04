#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_json
from vgs.stage_m import analyze_stage_m_local_rescue


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage M gated local rescue results.")
    parser.add_argument("--results", default="outputs/stage_m_local_rescue/local_rescue_results.csv")
    parser.add_argument("--output-dir", default="outputs/stage_m_local_rescue")
    parser.add_argument("--plot-dir", default="outputs/plots")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    payload = {
        "results": args.results,
        "output_dir": args.output_dir,
        "plot_dir": args.plot_dir,
    }
    payload.update(analyze_stage_m_local_rescue(args.results, args.output_dir, args.plot_dir))
    summary_path = write_json(Path(args.output_dir) / "analyze_stage_m_local_rescue_summary.json", payload)
    append_experiment_log(args.log_path, "analyze_stage_m_local_rescue", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
