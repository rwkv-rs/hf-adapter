from __future__ import annotations

from bench.extract_jsonl_from_log import extract_json_rows


def test_extract_json_rows_ignores_progress_and_python_dicts():
    text = """
progress 50%\n
{'axis': 'not_json', 'status': 'pass'}
{"axis":"one","status":"pass","value":1}
not a row
{"axis":"two","status":"info"}
"""
    rows = extract_json_rows(text)
    assert [row["axis"] for row in rows] == ["one", "two"]
    assert [row["source_log_line"] for row in rows] == [4, 6]
