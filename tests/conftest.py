"""Central pytest marker policy for the mixed CPU/hardware test tree."""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest


REGISTERED_MARKERS = frozenset(
    {"cpu", "cuda", "sm70", "ada", "blackwell", "apple", "musa", "slow", "model_required"}
)

MUSA_PATTERNS = (
    "test_musa_*.py",
)
APPLE_PATTERNS = (
    "test_apple_*.py",
    "test_mlx_*.py",
    "test_coreml_*.py",
    "test_qwen35_apple_*.py",
)
CUDA_PATTERNS = (
    "test_*cuda*.py",
    "test_*triton*.py",
    "test_*fused*.py",
    "test_native_graph*.py",
    "test_native_prefill_scan.py",
    "test_native_quant_a8w8.py",
    "test_native_quant_bnb8.py",
    "test_deepspeed_*.py",
    "test_device_map_generate.py",
    "test_tensor_parallel_generate.py",
    "test_extension_build_env.py",
    "test_official_alignment.py",
    "test_self_chunk_rwkv7.py",
    "test_sm70_*.py",
    "test_t4_*.py",
    "test_v100_*.py",
    "test_ada_*.py",
    "test_4080_*.py",
    "test_blackwell_*.py",
    "test_5090_*.py",
)
MODEL_REQUIRED_PATTERNS = (
    "test_batch_cache.py",
    "test_chunked_prefill.py",
    "test_deepspeed_resume_smoke.py",
    "test_deepspeed_training_smoke.py",
    "test_device_map_generate.py",
    "test_tensor_parallel_generate.py",
    "test_dynamic_batch_cache.py",
    "test_fast_cache.py",
    "test_fast_decode_api.py",
    "test_hf_api_contract.py",
    "test_hf_rl_training_smoke.py",
    "test_hf_trainer_resume_smoke.py",
    "test_hf_training_smoke.py",
    "test_native_bnb_quant_smoke.py",
    "test_native_dpo_smoke.py",
    "test_native_grpo_smoke.py",
    "test_native_mm8_persist.py",
    "test_native_model.py",
    "test_native_peft_save_load_merge.py",
    "test_native_quant_mm4.py",
    "test_native_quant_mm8.py",
    "test_native_quant_torchao.py",
    "test_native_sft_smoke.py",
    "test_native_trainer_resume_smoke.py",
    "test_native_trainer_smoke.py",
    "test_peft_lora.py",
    "test_quantized_inference.py",
    "test_reload_roundtrip.py",
    "test_speculative_decode.py",
)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def classify_test_path(path: str | Path) -> frozenset[str]:
    """Return additive execution/domain markers for one test module path."""

    name = Path(path).name
    markers = {"cpu"}
    if _matches(name, MUSA_PATTERNS):
        markers.add("musa")
    if _matches(name, APPLE_PATTERNS):
        markers.add("apple")
    if _matches(name, CUDA_PATTERNS):
        markers.add("cuda")
    if fnmatch(name, "test_sm70_*.py") or fnmatch(name, "test_v100_*.py"):
        markers.update({"cuda", "sm70"})
    if fnmatch(name, "test_ada_*.py") or fnmatch(name, "test_4080_*.py"):
        markers.update({"cuda", "ada"})
    if fnmatch(name, "test_blackwell_*.py") or fnmatch(name, "test_5090_*.py"):
        markers.update({"cuda", "blackwell"})
    if _matches(name, MODEL_REQUIRED_PATTERNS):
        markers.update({"model_required", "slow"})
    return frozenset(markers)


def validate_marker_set(markers: set[str] | frozenset[str]) -> None:
    unknown = set(markers) - REGISTERED_MARKERS
    if unknown:
        raise pytest.UsageError(f"unknown RWKV7 pytest markers: {sorted(unknown)}")
    if "cpu" not in markers:
        raise pytest.UsageError("every collected test must be classified for CPU/offline collection")
    if set(markers) & {"sm70", "ada", "blackwell"} and "cuda" not in markers:
        raise pytest.UsageError("GPU-family markers require the cuda marker")
    if "model_required" in markers and "slow" not in markers:
        raise pytest.UsageError("model_required tests must also be marked slow")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        markers = classify_test_path(item.path)
        validate_marker_set(markers)
        for marker in sorted(markers):
            item.add_marker(getattr(pytest.mark, marker))


def pytest_report_header() -> str:
    return "rwkv7 marker policy: cpu + additive hardware/slow/model domains"
