from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rwkv7_hf import native_jit, native_jit_bnb8


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _fake_bnb8_operand(*, threshold: float = 0.0, out_features: int = 3):
    fake_type = type(
        "Linear8bitLt",
        (),
        {"__module__": "bitsandbytes.nn.modules"},
    )
    operand = fake_type()
    operand.state = SimpleNamespace(
        threshold=threshold,
        CB=torch.ones(out_features, 4, dtype=torch.int8),
        SCB=torch.ones(out_features),
    )
    operand.weight = SimpleNamespace(CB=None)
    operand.training = False
    operand.bias = None
    operand.out_features = out_features
    return operand


def test_bnb8_ownership_moves_without_hot_path_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_bnb8.py")
    moved = {
        "_bnb8_direct_linear",
        "_bnb8_direct_relu_square_linear",
        "_bnb8_ffn_mix_quant_enabled",
        "_bnb8_prequant_linear",
        "_bnb8_rkv_mix_quant_enabled",
        "_is_bnb8_linear",
        "_native_bnb8_direct_enabled",
        "_native_bnb8_policy_block",
        "_native_bnb8_policy_flag",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        assert getattr(native_jit, name) is getattr(native_jit_bnb8, name)


def test_bnb8_policy_environment_precedence(monkeypatch) -> None:
    policy = SimpleNamespace(
        native_bnb8_direct=True,
        native_bnb8_attn_mix_block=2048,
    )
    monkeypatch.setattr(
        native_jit_bnb8,
        "current_kernel_policy",
        lambda **_kwargs: policy,
    )
    monkeypatch.delenv("RWKV7_NATIVE_BNB8_DIRECT", raising=False)
    monkeypatch.delenv("RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", raising=False)

    assert native_jit._native_bnb8_direct_enabled()
    assert native_jit._native_bnb8_policy_block(
        "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK",
        "native_bnb8_attn_mix_block",
        1024,
    ) == 2048

    monkeypatch.setenv("RWKV7_NATIVE_BNB8_DIRECT", "0")
    monkeypatch.setenv("RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", "4096")
    assert not native_jit._native_bnb8_direct_enabled()
    assert native_jit._native_bnb8_policy_block(
        "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK",
        "native_bnb8_attn_mix_block",
        1024,
    ) == 4096

    monkeypatch.setenv("RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", "128")
    with pytest.raises(ValueError, match="must be 256"):
        native_jit._native_bnb8_policy_block(
            "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK",
            "native_bnb8_attn_mix_block",
            1024,
        )


def test_bnb8_detection_and_safe_fallbacks(monkeypatch) -> None:
    operand = _fake_bnb8_operand(threshold=1.0)

    assert native_jit._is_bnb8_linear(operand)
    monkeypatch.setenv("RWKV7_NATIVE_BNB8_DIRECT", "1")
    with torch.inference_mode():
        assert native_jit._bnb8_direct_linear(torch.randn(2, 4), operand) is None
    assert native_jit._bnb8_direct_linear(torch.randn(2, 4), object()) is None


def test_bnb8_direct_operator_wiring_and_rank_restore(monkeypatch) -> None:
    calls = {}

    def quantize(rows, threshold):
        calls["quant_shape"] = tuple(rows.shape)
        calls["threshold"] = threshold
        return torch.ones_like(rows, dtype=torch.int8), torch.ones(rows.shape[0]), None

    def scaled_mm(quantized, cb, scales, scb, *, bias, dtype):
        calls["mm_shape"] = tuple(quantized.shape)
        calls["dtype"] = dtype
        return torch.full(
            (quantized.shape[0], cb.shape[0]),
            2,
            dtype=dtype,
        )

    fake_ops = SimpleNamespace(
        int8_vectorwise_quant=SimpleNamespace(default=quantize),
        int8_scaled_mm=SimpleNamespace(default=scaled_mm),
    )
    monkeypatch.setattr(torch.ops, "bitsandbytes", fake_ops)
    monkeypatch.setenv("RWKV7_NATIVE_BNB8_DIRECT", "1")
    operand = _fake_bnb8_operand()
    x = torch.randn(2, 5, 4, dtype=torch.float32)

    with torch.inference_mode():
        out = native_jit._bnb8_direct_linear(x, operand)

    assert out is not None
    assert out.shape == (2, 5, 3)
    assert torch.equal(out, torch.full_like(out, 2))
    assert calls == {
        "quant_shape": (10, 4),
        "threshold": 0.0,
        "mm_shape": (10, 4),
        "dtype": torch.float32,
    }


def test_bnb8_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_bnb8.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_bnb8 import" in entrypoint
