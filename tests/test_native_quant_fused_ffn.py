#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rwkv7_hf.native_jit import _native_graph_ffn_up_relu2_dispatch
from rwkv7_hf.native_quant_mm4 import MM4Linear
from rwkv7_hf.native_quant_mm8 import MM8Linear


def _assert_module_fallback(module_type) -> None:
    torch.manual_seed(7)
    dense = torch.nn.Linear(16, 32, bias=False, dtype=torch.float32)
    module = module_type(dense, fused=True)
    x = torch.randn(3, 16)
    expected = torch.relu(module(x)) ** 2
    actual = module.rwkv7_forward_relu2(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mm8_relu2_cpu_fallback() -> None:
    _assert_module_fallback(MM8Linear)


def test_mm4_relu2_cpu_fallback() -> None:
    _assert_module_fallback(MM4Linear)


class ProbeQuantLinear(torch.nn.Module):
    in_features = 4
    out_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.fused_calls = 0
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return torch.ones((*x.shape[:-1], self.out_features), dtype=x.dtype)

    def rwkv7_forward_relu2(self, x):
        self.fused_calls += 1
        return torch.full((*x.shape[:-1], self.out_features), 4.0, dtype=x.dtype)


def test_native_graph_quant_ffn_route_is_opt_in() -> None:
    old = os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN")
    try:
        x = torch.ones(2, 4)
        probe = ProbeQuantLinear()

        os.environ["RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN"] = "0"
        disabled = _native_graph_ffn_up_relu2_dispatch(x, probe)
        assert probe.fused_calls == 0
        assert probe.forward_calls == 1
        torch.testing.assert_close(disabled, torch.ones(2, 8))

        os.environ["RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN"] = "1"
        enabled = _native_graph_ffn_up_relu2_dispatch(x, probe)
        assert probe.fused_calls == 1
        assert probe.forward_calls == 1
        torch.testing.assert_close(enabled, torch.full((2, 8), 4.0))
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN", None)
        else:
            os.environ["RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN"] = old


def main() -> int:
    test_mm8_relu2_cpu_fallback()
    test_mm4_relu2_cpu_fallback()
    test_native_graph_quant_ffn_route_is_opt_in()
    print("NATIVE QUANT FUSED FFN TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
