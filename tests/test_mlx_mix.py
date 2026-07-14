#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX/Metal attention mix fusion seam."""
from __future__ import annotations

import importlib.util


def test_mlx_mix_import_safe():
    import rwkv7_hf.mlx_mix as mm

    assert hasattr(mm, "attn_mix")
    assert hasattr(mm, "metal_attn_mix_available")


def test_mlx_attn_mix_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_mix import attn_mix, attn_mix_reference, metal_attn_mix_available

    mx.random.seed(20260707)
    x = mx.random.normal((3, 8)).astype(mx.float16)
    x_prev = mx.random.normal((3, 8)).astype(mx.float16)
    mixes = [mx.random.normal((8,)).astype(mx.float16) for _ in range(6)]
    ref = attn_mix_reference(x, x_prev, *mixes)
    auto, backend = attn_mix(x, x_prev, *mixes, backend="auto")
    mx.eval(*ref, *auto)
    expected_backend = "metal" if metal_attn_mix_available() else "reference"
    assert backend == expected_backend
    assert len(auto) == 6
    for got, exp in zip(auto, ref, strict=True):
        assert tuple(int(v) for v in got.shape) == (3, 8)
        assert float(mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32)))) < 5e-3

    portable, portable_backend = attn_mix(x, x_prev, *mixes, backend="reference")
    mx.eval(*portable)
    assert portable_backend == "reference"
    for got, exp in zip(portable, ref, strict=True):
        assert float(mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32)))) == 0.0


def test_mlx_sequence_mix_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_mix import attn_sequence_mix_metal, ffn_sequence_mix_metal

    mx.random.seed(20260714)
    x = mx.random.normal((2, 5, 64)).astype(mx.float16)
    x_prev = mx.random.normal((2, 64)).astype(mx.float16)
    mixes = [mx.random.normal((64,)).astype(mx.float16) for _ in range(4)]
    previous = mx.concatenate([x_prev[:, None, :], x[:, :-1, :]], axis=1)
    xx_ref = previous - x
    refs = [xx_ref, *(x + xx_ref * mix.reshape(1, 1, 64) for mix in mixes)]
    outputs = attn_sequence_mix_metal(x, x_prev, *mixes)
    ffn = ffn_sequence_mix_metal(x, x_prev, mixes[0])
    mx.eval(*refs, *outputs, ffn)
    for got, expected in zip(outputs, refs, strict=True):
        assert float(mx.max(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)))) <= 8e-3
    assert float(mx.max(mx.abs(ffn.astype(mx.float32) - refs[1].astype(mx.float32)))) <= 8e-3
