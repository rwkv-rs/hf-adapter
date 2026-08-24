from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "run_4080_dynamic_prefill_matrix.sh"
).read_text(encoding="utf-8")


def test_dynamic_prefill_runner_covers_continuous_batches_and_tile_boundaries() -> None:
    for text in (
        "1,2,3,4,5,6,7,8",
        "127,128,129,511,512,513",
        "31,32,33,63,64,65",
        "2047,2048,2049",
        "--reference-backend native-direct",
        "check_dynamic_prefill_matrix.py",
        "--require-safe-fusions",
        "--max-padding-latency-ratio 1.5",
        "--max-boundary-throughput-ratio 1.35",
        "--max-cross-route-boundary-ratio 3.0",
    ):
        assert text in SCRIPT


def test_dynamic_prefill_runner_is_exact_card_and_append_never() -> None:
    assert "OUT_DIR must not already exist" in SCRIPT
    assert "--model 4080 --name" in SCRIPT
    assert "system.json" in SCRIPT
    assert "dynamic_prefill.jsonl" in SCRIPT
