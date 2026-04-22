import json

from vgs.datasets import read_json_or_jsonl


def test_read_json_array(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
    assert read_json_or_jsonl(path) == [{"question_id": 1}]


def test_read_jsonl(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"question_id": 1}\n{"question_id": 2}\n', encoding="utf-8")
    assert len(read_json_or_jsonl(path)) == 2
