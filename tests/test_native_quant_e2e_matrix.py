from argparse import Namespace
from pathlib import Path

from bench.run_native_quant_e2e_matrix import (
    BaseCase,
    baseline_key,
    expanded_cases,
    read_completed,
)


def test_expanded_profile_has_seven_shapes_per_model():
    cases = expanded_cases([("1.5b", "/models/1.5b"), ("2.9b", "/models/2.9b")])
    assert len(cases) == 14
    assert {(case.batch_size, case.prompt_tokens, case.decode_tokens) for case in cases[:7]} == {
        (1, 128, 128),
        (2, 128, 128),
        (4, 128, 128),
        (8, 128, 128),
        (1, 512, 128),
        (1, 2048, 128),
        (1, 128, 512),
    }


def test_baseline_key_is_shared_by_fused_modes():
    case = BaseCase("1.5b", "/models/1.5b", 1, 128, 128)
    args = Namespace(
        dtype="fp16",
        fast_token_backend="native_graph",
        policy="memory",
        min_params=8_000_000,
        warmup=1,
        timing_repeats=3,
    )
    assert "fused" not in baseline_key(case, args)


def test_read_completed_ignores_failure_rows(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"axis":"native_quant_e2e_decode","status":"pass","model_size_label":"1.5b","batch_size":1,"prompt_tokens":128,"decode_tokens":128,"quantization":"mm8","fused_quant_ffn":true,"fused_quant_ffn_down_add":true}',
                '{"axis":"native_quant_e2e_matrix_attempt","status":"fail","model_size_label":"1.5b"}',
            ]
        ),
        encoding="utf-8",
    )
    assert read_completed(path) == {("1.5b", 1, 128, 128, "mm8", "deep")}
