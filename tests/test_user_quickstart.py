from __future__ import annotations

import torch
import pytest

from examples.generate import (
    build_parser,
    resolve_device,
    resolve_dtype,
    select_native_backend,
)


def test_generate_example_defaults_are_beginner_safe() -> None:
    args = build_parser().parse_args(["--model", "model", "--prompt", "hello"])
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.backend == "auto"
    assert args.temperature == 0.0
    assert args.max_new_tokens == 64


def test_generate_example_cpu_defaults_to_fp32() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert resolve_dtype("auto", device) == torch.float32
    assert resolve_dtype("bf16", device) == torch.bfloat16


def test_backend_auto_falls_back_without_fla() -> None:
    assert select_native_backend(
        "auto", device_type="cpu", fla_available=False, native_env_enabled=False
    )
    assert select_native_backend(
        "auto", device_type="cuda", fla_available=False, native_env_enabled=False
    )
    assert not select_native_backend(
        "auto", device_type="cuda", fla_available=True, native_env_enabled=False
    )
    assert select_native_backend(
        "auto", device_type="cuda", fla_available=True, native_env_enabled=True
    )
    assert select_native_backend(
        "native", device_type="cuda", fla_available=True, native_env_enabled=False
    )
    assert not select_native_backend(
        "fla", device_type="cuda", fla_available=True, native_env_enabled=True
    )
    with pytest.raises(RuntimeError, match="requires flash-linear-attention"):
        select_native_backend(
            "fla", device_type="cuda", fla_available=False, native_env_enabled=False
        )
