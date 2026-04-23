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
