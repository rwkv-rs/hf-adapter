from argparse import Namespace
from pathlib import Path

import pytest

from bench.run_native_quant_e2e_matrix import (
    BaseCase,
    baseline_key,
    expanded_cases,
    read_completed,
)
from bench.bench_native_quant_e2e_decode import prepare_model_dir, validate_quantize_before_device


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


def test_repo_code_staging_uses_requested_volume(tmp_path: Path):
    source = tmp_path / "model"
    staging = tmp_path / "staging"
    source.mkdir()
    weight = source / "model.safetensors"
    weight.write_bytes(b"weights")
    effective, temporary = prepare_model_dir(str(source), "repo", str(staging))
    try:
        staged = Path(effective) / weight.name
        assert staged.read_bytes() == b"weights"
        assert Path(effective).parent == staging.resolve()
    finally:
        temporary.cleanup()


def test_quantize_before_device_requires_explicit_quant_only_mode():
    args = Namespace(
        quantize_before_device=True,
        device="cuda",
        single_quantization="mm4",
        paired_baseline=False,
        allow_missing_baseline=True,
    )
    validate_quantize_before_device(args)

    args.single_quantization = None
    with pytest.raises(ValueError, match="single-quantization"):
        validate_quantize_before_device(args)

    args.single_quantization = "mm4"
    args.paired_baseline = True
    with pytest.raises(ValueError, match="fp16 baseline"):
        validate_quantize_before_device(args)
