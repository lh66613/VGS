from scripts.run_official_vcd_pope_baseline import summarize_pope_rows
from vgs.pope import classify_outcome, parse_yes_no


def test_parse_yes_no_uses_first_answer_token():
    assert parse_yes_no("Yes, there is a dog.") == "yes"
    assert parse_yes_no("No. The image does not show that.") == "no"


def test_parse_yes_no_handles_empty_or_punctuation_only_output():
    assert parse_yes_no("") == "unknown"
    assert parse_yes_no("   ") == "unknown"
    assert parse_yes_no(".") == "unknown"
    assert parse_yes_no(",,,") == "unknown"


def test_classify_outcome_marks_false_positive():
    assert classify_outcome("yes", "no") == "FP"


def test_official_vcd_baseline_summary_metrics():
    rows = [
        {"subset": "random", "outcome": "TP"},
        {"subset": "random", "outcome": "TN"},
        {"subset": "random", "outcome": "FP"},
        {"subset": "popular", "outcome": "FN"},
        {"subset": "popular", "outcome": "unknown"},
    ]

    metrics = {row["subset"]: row for row in summarize_pope_rows(rows)}

    assert metrics["overall"]["n"] == 5
    assert metrics["overall"]["known_n"] == 4
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["precision"] == 0.5
    assert metrics["overall"]["recall"] == 0.5
    assert metrics["random"]["fp_rate"] == 0.5
