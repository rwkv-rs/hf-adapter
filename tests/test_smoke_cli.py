from __future__ import annotations

import json

from rwkv7_hf import smoke


def test_smoke_defaults_to_small_public_model() -> None:
    args = smoke.parse_args([])
    assert args.model == "wangyue114514/rwkv7-g1d-0.1b-hf"
    assert args.revision == "v0.7.0"
    assert args.prompt == "User: Hello! Assistant:"
    assert args.max_new_tokens == 4


def test_smoke_json_cli_writes_same_report(monkeypatch, tmp_path, capsys) -> None:
    expected = {
        "status": "passed",
        "timing": {},
        "runtime": {},
    }
    monkeypatch.setattr(smoke, "run_smoke", lambda _args: expected)
    output = tmp_path / "smoke.json"

    assert smoke.main(["--json", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
