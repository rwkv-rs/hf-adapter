import json
from pathlib import Path

from bench.summarize_native_quant_e2e_matrix import load_rows, summarize


def row(quantization: str, mode: str, tokps: float) -> dict:
    return {
        "axis": "native_quant_e2e_decode",
        "status": "pass",
        "model_size_label": "1.5b",
        "batch_size": 1,
        "prompt_tokens": 128,
        "decode_tokens": 128,
        "quantization": quantization,
        "fused_quant_ffn": mode in {"up", "deep"},
        "fused_quant_ffn_down_add": mode == "deep",
        "decode_tokps_total": tokps,
        "decode_speed_ratio_vs_fp16": tokps / 100.0,
        "footprint_ratio_vs_fp16": 0.5,
        "prompt_logits_cos_vs_fp16": 0.999,
        "final_logits_cos_vs_fp16": 0.998,
        "same_next_token_as_fp16": True,
        "peak_vram_mb": 1000,
    }


def test_summary_pairs_fusion_modes():
    rows = [
        row("none", "off", 100),
        row("mm8", "off", 80),
        row("mm8", "up", 88),
        row("mm8", "deep", 84),
        row("mm4", "off", 110),
        row("mm4", "up", 121),
    ]
    summary = summarize(rows, [], expected_models=1)
    assert summary["completed_rows"] == 6
    assert summary["unresolved_failures"] == 0
    assert summary["paired"]["mm4_up_vs_off"]["ratio_median"] == 1.1
    assert summary["paired"]["mm8_up_vs_off"]["right_wins"] == 1
    assert summary["paired"]["mm8_deep_vs_up"]["right_wins"] == 0


def test_load_rows_deduplicates_success_and_keeps_failure(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    success = row("mm4", "up", 120)
    newer = row("mm4", "up", 121)
    failure = {"axis": "native_quant_e2e_matrix_attempt", "status": "fail"}
    path.write_text("\n".join(json.dumps(item) for item in (success, newer, failure)), encoding="utf-8")
    rows, failures = load_rows([path])
    assert len(rows) == 1
    assert rows[0]["decode_tokps_total"] == 121
    assert failures == [failure]
