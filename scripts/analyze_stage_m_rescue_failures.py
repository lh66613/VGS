#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgs.io import append_experiment_log, write_json
from vgs.stage_m import analyze_stage_m_rescue_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage M FP rescue failure taxonomy.")
    parser.add_argument("--results", default="outputs/stage_m_local_rescue/local_rescue_results.csv")
    parser.add_argument("--predictions", default="outputs/predictions/pope_predictions.jsonl")
    parser.add_argument("--retrieval-plan", default="outputs/stage_m_local_rescue/retrieval_plan.csv")
    parser.add_argument("--memory-bank", default="outputs/stage_m_local_rescue/memory_bank_train.pt")
    parser.add_argument("--hidden-states-dir", default="outputs/hidden_states")
    parser.add_argument("--output-dir", default="outputs/stage_m_local_rescue")
    parser.add_argument("--notes-path", default="notes/rescue_failure_analysis.md")
    parser.add_argument("--log-path", default="notes/experiment_log.md")
    args = parser.parse_args()

    payload = {
        "results": args.results,
        "predictions": args.predictions,
        "retrieval_plan": args.retrieval_plan,
        "memory_bank": args.memory_bank,
        "hidden_states_dir": args.hidden_states_dir,
        "output_dir": args.output_dir,
        "notes_path": args.notes_path,
    }
    payload.update(
        analyze_stage_m_rescue_failures(
            results_path=args.results,
            predictions_path=args.predictions,
            retrieval_plan_path=args.retrieval_plan,
            memory_bank_path=args.memory_bank,
            hidden_states_dir=args.hidden_states_dir,
            output_dir=args.output_dir,
            notes_path=args.notes_path,
        )
    )
    summary_path = write_json(Path(args.output_dir) / "analyze_stage_m_rescue_failures_summary.json", payload)
    append_experiment_log(args.log_path, "analyze_stage_m_rescue_failures", summary_path, "ok")
    print(summary_path)


if __name__ == "__main__":
    main()
