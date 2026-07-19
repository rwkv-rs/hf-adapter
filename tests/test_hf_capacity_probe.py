from __future__ import annotations

import json
from pathlib import Path

from bench.bench_hf_capacity_probe import checkpoint_payload_bytes, classify_result


def test_checkpoint_payload_uses_safetensors_index_metadata(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123456}, "weight_map": {}}),
        encoding="utf-8",
    )
    assert checkpoint_payload_bytes(tmp_path) == 123456


def test_checkpoint_payload_falls_back_to_safetensor_files(tmp_path: Path) -> None:
    (tmp_path / "model-1.safetensors").write_bytes(b"abc")
    (tmp_path / "model-2.safetensors").write_bytes(b"defgh")
    assert checkpoint_payload_bytes(tmp_path) == 8


def test_capacity_expectation_is_fail_closed() -> None:
    assert classify_result("fit", "fit") == ("pass", None)
    assert classify_result("capacity-limit", "capacity-limit") == ("pass", None)
    status, error = classify_result("fit", "capacity-limit")
    assert status == "fail"
    assert error == "expected fit, observed capacity-limit"
