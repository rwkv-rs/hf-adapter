from __future__ import annotations

from itertools import product
from pathlib import Path

from bench.capture_hf_fast_path_v1_environment import compare_runtime_lock
from bench.validate_hf_fast_path_v1 import PAIRS, validate_matrix

ROOT = Path(__file__).resolve().parents[1]


def row(role: str, pair: str, batch: int, prompt: int, decode: int) -> dict:
    candidate_size, reference_size = PAIRS[pair]
    value = {
        "_source": f"{role}:{pair}:{batch}:{prompt}:{decode}",
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "hf_fast_path_v1",
        "model_pair": pair,
        "model_role": role,
        "model_kind": "rwkv" if role == "candidate" else "qwen35",
        "model_size_label": candidate_size if role == "candidate" else reference_size,
        "dtype": "fp16",
        "quantization": "none",
        "batch_size": batch,
        "prompt_tokens": prompt,
        "decode_tokens": decode,
        "prefill_chunk_size": 512,
        "warmup": 3,
        "runs": 7,
        "timing_statistic": "median",
        "mtp_enabled": False,
        "speculative_decoding_enabled": False,
        "resident_sweep": True,
        "status": "pass",
        "device": "NVIDIA GeForce RTX 4090",
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
        "triton_version": "3.4.0",
        "transformers_version": "5.12.1",
        "fla_version": "0.5.1",
        "causal_conv1d_version": "1.5.3",
    }
    if role == "candidate":
        value.update(
            {
                "rwkv_fast_token_backend_requested": "native_jit",
                "rwkv_prefill_graph_requested": "0",
                "effective_backend": "native_jit",
                "step_backend": "rwkv_fast_token",
                "prefill_backend_effective": "native_prefill",
            }
        )
    else:
        value.update(
            {
                "qwen_backend_requested": "fla",
                "qwen_conv_backend_requested": "causal_conv1d",
                "qwen_fast_path_required": True,
                "qwen_fast_path_available": True,
                "qwen_fast_path_verified": True,
                "qwen_full_fused_contract_pass": True,
                "qwen_causal_conv1d_importable": True,
                "qwen_conv_backend_effective": "causal_conv1d",
                "qwen_force_torch": False,
            }
        )
    return value


def complete_rows(role: str) -> list[dict]:
    return [
        row(role, pair, batch, prompt, decode)
        for pair in PAIRS
        for batch, prompt, decode in product((1, 8), (128, 512, 2048), (128, 512))
    ]


def test_complete_hf_fast_path_v1_matrix_passes() -> None:
    summary = validate_matrix(
        complete_rows("candidate"),
        complete_rows("reference"),
        expected_device="NVIDIA GeForce RTX 4090",
    )
    assert summary["status"] == "pass"
    assert summary["candidate_rows"] == 48
    assert summary["reference_rows"] == 48
    assert summary["runtime_signature_count"] == 1


def test_qwen_official_contract_failure_blocks_the_main_table() -> None:
    references = complete_rows("reference")
    references[0]["qwen_conv_backend_effective"] = "fla_triton"
    summary = validate_matrix(complete_rows("candidate"), references)
    assert summary["status"] == "fail"
    assert any("qwen_conv_backend_effective='fla_triton'" in error for error in summary["errors"])


def test_rwkv_cuda_graph_or_missing_cell_blocks_the_main_table() -> None:
    candidates = complete_rows("candidate")
    candidates[0]["effective_backend"] = "native_graph"
    references = complete_rows("reference")[:-1]
    summary = validate_matrix(candidates, references)
    assert summary["status"] == "fail"
    assert any("CUDA Graph route" in error for error in summary["errors"])
    assert any("reference row count=47" in error for error in summary["errors"])
    assert any("reference missing cells" in error for error in summary["errors"])


def test_runtime_lock_comparison_is_exact() -> None:
    expected = {"python_version": "3.10.16", "torch_version": "2.8.0+cu128"}
    assert compare_runtime_lock(dict(expected), expected) == []
    assert compare_runtime_lock(
        {**expected, "torch_version": "2.9.0+cu128"}, expected
    ) == ["torch_version: actual='2.9.0+cu128', expected='2.8.0+cu128'"]


def test_single_card_script_is_official_and_fail_closed() -> None:
    text = (ROOT / "bench" / "run_hf_fast_path_v1.sh").read_text(encoding="utf-8")
    assert "--qwen-conv-backend causal_conv1d" in text
    assert "--require-qwen-fast-path" in text
    assert "--qwen-conv-backend fla_triton" not in text
    assert "RWKV7_FAST_TOKEN_BACKEND=native_jit" in text
    assert "RWKV7_NATIVE_PREFILL_GRAPH=0" in text
    assert "SM120 official HF fast path unverified" in text
    assert "--batch-sizes 1 8" in text
    assert "--prompt-tokens 128 512 2048" in text
    assert "--decode-tokens 128 512" in text
    assert "--warmup 3" in text
    assert "--runs 7" in text
    assert "FLA_SOURCE_COMMIT" in text
    assert "CAUSAL_CONV1D_SOURCE_COMMIT" in text


def test_extension_build_script_pins_sources_and_all_card_arches() -> None:
    text = (ROOT / "bench" / "build_hf_fast_path_v1_extensions.sh").read_text(
        encoding="utf-8"
    )
    assert 'TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"' in text
    assert "2e38c1fab332174d056928feaf29f8c5fd5ac550" in text
    assert "4f6ae4e26ae5fe8af9372f8d312ab25cc4595223" in text
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in text
    assert "--force-reinstall" in text
