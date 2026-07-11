#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv7_hf.ada_sparse_ffn import (
    ada_ffn_up,
    ada_ffn_up_should_use,
    ada_linear,
    ada_linear_should_use,
    ada_sparse_ffn_down_add,
    ada_sparse_ffn_should_use,
)


def test_shape_policy_is_narrow() -> None:
    assert ada_sparse_ffn_should_use(1, 768, 3072)
    assert ada_sparse_ffn_should_use(8, 1024, 4096)
    assert ada_sparse_ffn_should_use(19, 2048, 8192)
    assert not ada_sparse_ffn_should_use(20, 1024, 4096)
    assert not ada_sparse_ffn_should_use(8, 1000, 4000)
    assert not ada_sparse_ffn_should_use(8, 1024, 2048)
    assert ada_ffn_up_should_use(1, 4096, 1024)
    assert ada_ffn_up_should_use(2, 3072, 768)
    assert not ada_ffn_up_should_use(4, 3072, 768)
    assert not ada_ffn_up_should_use(8, 4096, 1024)
    assert ada_linear_should_use(2, 65536, 1024)
    assert ada_linear_should_use(1, 1024, 1024)
    assert ada_linear_should_use(4, 1024, 1024)
    assert ada_linear_should_use(4, 4096, 1024)
    assert not ada_linear_should_use(4, 1024, 4096)


def test_cpu_fallback_matches_torch() -> None:
    torch.manual_seed(123)
    preact = torch.randn(3, 1024, dtype=torch.float32)
    weight = torch.randn(256, 1024, dtype=torch.float32)
    residual = torch.randn(3, 256, dtype=torch.float32)
    expected = residual + F.linear(torch.relu(preact) ** 2, weight)
    actual = ada_sparse_ffn_down_add(preact, weight, residual, force_fallback=True)
    torch.testing.assert_close(actual, expected)


def test_scalar_fallback_preserves_shape() -> None:
    preact = torch.randn(1024)
    weight = torch.randn(256, 1024)
    residual = torch.randn(256)
    actual = ada_sparse_ffn_down_add(preact, weight, residual, force_fallback=True)
    assert actual.shape == residual.shape


def test_fallback_out_buffer_is_reused() -> None:
    preact = torch.randn(3, 1024)
    weight = torch.randn(256, 1024)
    residual = torch.randn(3, 256)
    out = torch.empty_like(residual)
    expected = residual + F.linear(torch.relu(preact) ** 2, weight)
    actual = ada_sparse_ffn_down_add(
        preact, weight, residual, out=out, force_fallback=True
    )
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_ffn_up_cpu_fallback_matches_torch() -> None:
    x = torch.randn(3, 256)
    weight = torch.randn(1024, 256)
    expected = F.linear(x, weight)
    actual = ada_ffn_up(x, weight, force_fallback=True)
    torch.testing.assert_close(actual, expected)


def test_ada_linear_cpu_fallback_matches_torch() -> None:
    x = torch.randn(2, 256)
    weight = torch.randn(768, 256)
    expected = F.linear(x, weight)
    actual = ada_linear(x, weight, force_fallback=True)
    torch.testing.assert_close(actual, expected)
