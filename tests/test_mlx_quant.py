#!/usr/bin/env python3
# coding=utf-8
"""Tests for the optional MLX packed W8/W4 quantized projection path."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mlx_quant_import_safe():
    import rwkv7_hf.mlx_quant as mq

    assert hasattr(mq, "MLXQuantizedLinear")
    assert hasattr(mq, "quantize_mlx_mm8")
    assert hasattr(mq, "quantize_mlx_mm4")


def test_mlx_groupwise_w4_nax_relu2_matches_public_qmm_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import (
        groupwise_w4_matmul_relu2_metal,
        groupwise_w4_relu2_metal_available,
        quantize_mlx_groupwise_linear,
    )

    if not groupwise_w4_relu2_metal_available():
        return
    mx.random.seed(20260715)
    dense = mx.random.normal((64, 128)).astype(mx.float16)
    weight = quantize_mlx_groupwise_linear(dense, bits=4, group_size=128)
    x = mx.random.normal((2, 33, 128)).astype(mx.float16)
    expected = mx.quantized_matmul(
        x,
        weight.w_q,
        scales=weight.scales,
        biases=weight.biases,
        transpose=True,
        group_size=128,
        bits=4,
        mode="affine",
    )
    expected = mx.maximum(expected, 0)
    expected = expected * expected
    actual = groupwise_w4_matmul_relu2_metal(x, weight)
    mx.eval(expected, actual)
    assert tuple(int(dim) for dim in actual.shape) == (2, 33, 64)
    assert float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))) == 0.0


def test_mlx_groupwise_w4_nax_relu2_decode_tile_matches_public_qmm_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import (
        groupwise_w4_matmul_relu2_metal,
        groupwise_w4_relu2_metal_available,
        quantize_mlx_groupwise_linear,
    )

    if not groupwise_w4_relu2_metal_available():
        return
    mx.random.seed(20260717)
    dense = mx.random.uniform(low=-0.25, high=0.25, shape=(8192, 2048)).astype(mx.float16)
    weight = quantize_mlx_groupwise_linear(dense, bits=4, group_size=128)
    x = (mx.random.normal((8, 2048)) * 0.1).astype(mx.float16)
    expected = mx.quantized_matmul(
        x,
        weight.w_q,
        scales=weight.scales,
        biases=weight.biases,
        transpose=True,
        group_size=128,
        bits=4,
        mode="affine",
    )
    expected = mx.maximum(expected, 0)
    expected = expected * expected
    actual = groupwise_w4_matmul_relu2_metal(x, weight)
    mx.eval(expected, actual)
    assert tuple(int(dim) for dim in actual.shape) == (8, 8192)
    assert bool(mx.allclose(expected, actual, rtol=0.01, atol=0.01))


def test_mlx_groupwise_w4_nax_square_matches_public_qmm_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import (
        groupwise_w4_square_matmul_metal,
        groupwise_w4_square_metal_available,
        quantize_mlx_groupwise_linear,
    )

    if not groupwise_w4_square_metal_available():
        return

    mx.random.seed(20260716)
    dense = mx.random.normal((128, 128)).astype(mx.float16)
    weight = quantize_mlx_groupwise_linear(dense, bits=4, group_size=128)
    x = mx.random.normal((2, 33, 128)).astype(mx.float16)
    expected = mx.quantized_matmul(
        x,
        weight.w_q,
        scales=weight.scales,
        biases=weight.biases,
        transpose=True,
        group_size=128,
        bits=4,
        mode="affine",
    )
    actual = groupwise_w4_square_matmul_metal(x, weight)
    mx.eval(expected, actual)
    assert tuple(int(dim) for dim in actual.shape) == (2, 33, 128)
    assert float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))) == 0.0


def test_mlx_q4_k_m_profile_policy():
    from rwkv7_hf.mlx_model import mlx_quant_bits_for_weight

    assert mlx_quant_bits_for_weight("lm_head.weight", bits=4, profile="q4_k_m") == 8
    assert mlx_quant_bits_for_weight(
        "model.layers.3.ffn.value.weight", bits=4, profile="q4_k_m"
    ) == 8
    assert mlx_quant_bits_for_weight(
        "model.layers.3.attn.r_proj.weight", bits=4, profile="q4_k_m"
    ) == 8
    assert mlx_quant_bits_for_weight(
        "model.layers.3.attn.v_proj.weight", bits=4, profile="q4_k_m"
    ) == 8
    assert mlx_quant_bits_for_weight(
        "model.layers.3.ffn.key.weight", bits=4, profile="q4_k_m"
    ) == 4
    assert mlx_quant_bits_for_weight(
        "model.layers.3.attn.k_proj.weight", bits=4, profile="q4_k_m"
    ) == 4
    assert mlx_quant_bits_for_weight("lm_head.weight", bits=8, profile="q4_k_m") == 8

    try:
        mlx_quant_bits_for_weight("lm_head.weight", bits=4, profile="unknown")
    except ValueError as exc:
        assert "quant profile" in str(exc)
    else:
        raise AssertionError("unknown MLX quant profile must fail")


def test_mlx_quant_formula_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import (
        MLXQuantizedLinear,
        groupwise_embedding,
        metal_quant_available,
        mm4_group_matmul_metal,
        mm4_group_matmul_metal_inputs,
        mm4_matmul_mlx,
        mm4_matmul_relu2_metal,
        mm4_triple_matmul_metal_inputs,
        mm8_group_matmul_metal,
        mm8_group_matmul_metal_inputs,
        mm8_matmul_mlx,
        mm8_triple_matmul_metal_inputs,
        pack_mlx_mm4_group,
        pack_mlx_mm8_group,
        quantize_mlx_mm4,
        quantize_mlx_mm8,
    )

    mx.random.seed(20260704)
    x = mx.random.normal((2, 5)).astype(mx.float16)
    weight = mx.random.normal((5, 7)).astype(mx.float16)  # W used as x @ W

    q8 = quantize_mlx_mm8(weight)
    y8_ref = mm8_matmul_mlx(x, q8, backend="reference")
    y8_affine = mm8_matmul_mlx(x, q8, backend="affine")
    mx.eval(y8_ref, y8_affine)
    assert float(mx.max(mx.abs(y8_ref - y8_affine))) < 1e-2
    if metal_quant_available():
        q8_metal = quantize_mlx_mm8(weight, layout="metal")
        y8_metal = mm8_matmul_mlx(x, q8_metal, backend="metal")
        mx.eval(y8_metal)
        assert float(mx.max(mx.abs(y8_ref - y8_metal))) < 5e-2
        q8_group = [
            quantize_mlx_mm8(weight, layout="metal"),
            quantize_mlx_mm8((weight * 0.5).astype(mx.float16), layout="metal"),
        ]
        y8_group = mm8_group_matmul_metal(x, pack_mlx_mm8_group(q8_group))
        y8_group_expected = mx.stack([mm8_matmul_mlx(x, q, backend="metal") for q in q8_group], axis=0)
        mx.eval(y8_group, y8_group_expected)
        assert tuple(int(v) for v in y8_group.shape) == (2, 2, 7)
        assert float(mx.max(mx.abs(y8_group - y8_group_expected))) < 5e-2
        x8_group = mx.stack([x, (x * 0.25).astype(mx.float16)], axis=0)
        y8_group_inputs = mm8_group_matmul_metal_inputs(x8_group, pack_mlx_mm8_group(q8_group))
        y8_group_inputs_expected = mx.stack(
            [mm8_matmul_mlx(x8_group[i], q, backend="metal") for i, q in enumerate(q8_group)],
            axis=0,
        )
        mx.eval(y8_group_inputs, y8_group_inputs_expected)
        assert tuple(int(v) for v in y8_group_inputs.shape) == (2, 2, 7)
        assert float(mx.max(mx.abs(y8_group_inputs - y8_group_inputs_expected))) < 5e-2
        q8_triple = q8_group + [quantize_mlx_mm8((weight * 0.25).astype(mx.float16), layout="metal")]
        x8_triple = [
            x,
            (x * 0.25).astype(mx.float16),
            (x * -0.5).astype(mx.float16),
        ]
        y8_triple = mm8_triple_matmul_metal_inputs(*x8_triple, q8_triple)
        y8_triple_expected = mx.stack(
            [mm8_matmul_mlx(xi, q, backend="metal") for xi, q in zip(x8_triple, q8_triple, strict=True)],
            axis=0,
        )
        mx.eval(y8_triple, y8_triple_expected)
        assert tuple(int(v) for v in y8_triple.shape) == (3, 2, 7)
        assert float(mx.max(mx.abs(y8_triple - y8_triple_expected))) < 5e-2

    q4 = quantize_mlx_mm4(weight)
    y4_ref = mm4_matmul_mlx(x, q4, backend="reference")
    y4_affine = mm4_matmul_mlx(x, q4, backend="affine")
    mx.eval(y4_ref, y4_affine)
    assert float(mx.max(mx.abs(y4_ref - y4_affine))) < 1e-2
    if metal_quant_available():
        q4_metal = quantize_mlx_mm4(weight, layout="metal")
        y4_metal = mm4_matmul_mlx(x, q4_metal, backend="metal")
        mx.eval(y4_metal)
        assert float(mx.max(mx.abs(y4_ref - y4_metal))) < 5e-2
        y4_relu2 = mm4_matmul_relu2_metal(x, q4_metal)
        y4_relu2_expected = mx.maximum(y4_metal, 0)
        y4_relu2_expected = y4_relu2_expected * y4_relu2_expected
        mx.eval(y4_relu2, y4_relu2_expected)
        assert float(mx.max(mx.abs(y4_relu2 - y4_relu2_expected))) < 5e-2
        q4_group = [
            quantize_mlx_mm4(weight, layout="metal"),
            quantize_mlx_mm4((weight * 0.5).astype(mx.float16), layout="metal"),
        ]
        y4_group = mm4_group_matmul_metal(x, pack_mlx_mm4_group(q4_group))
        y4_group_expected = mx.stack([mm4_matmul_mlx(x, q, backend="metal") for q in q4_group], axis=0)
        mx.eval(y4_group, y4_group_expected)
        assert tuple(int(v) for v in y4_group.shape) == (2, 2, 7)
        assert float(mx.max(mx.abs(y4_group - y4_group_expected))) < 5e-2
        x4_group = mx.stack([x, (x * 0.25).astype(mx.float16)], axis=0)
        y4_group_inputs = mm4_group_matmul_metal_inputs(x4_group, pack_mlx_mm4_group(q4_group))
        y4_group_inputs_expected = mx.stack(
            [mm4_matmul_mlx(x4_group[i], q, backend="metal") for i, q in enumerate(q4_group)],
            axis=0,
        )
        mx.eval(y4_group_inputs, y4_group_inputs_expected)
        assert tuple(int(v) for v in y4_group_inputs.shape) == (2, 2, 7)
        assert float(mx.max(mx.abs(y4_group_inputs - y4_group_inputs_expected))) < 5e-2
        q4_triple = q4_group + [quantize_mlx_mm4((weight * 0.25).astype(mx.float16), layout="metal")]
        x4_triple = [
            x,
            (x * 0.25).astype(mx.float16),
            (x * -0.5).astype(mx.float16),
        ]
        y4_triple = mm4_triple_matmul_metal_inputs(*x4_triple, q4_triple)
        y4_triple_expected = mx.stack(
            [mm4_matmul_mlx(xi, q, backend="metal") for xi, q in zip(x4_triple, q4_triple, strict=True)],
            axis=0,
        )
        mx.eval(y4_triple, y4_triple_expected)
        assert tuple(int(v) for v in y4_triple.shape) == (3, 2, 7)
        assert float(mx.max(mx.abs(y4_triple - y4_triple_expected))) < 5e-2

    # Linear weights are stored [out, in]; the helper quantizes weight.T.
    linear_weight = weight.T
    lin = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=8, backend="affine")
    y = lin(x)
    mx.eval(y)
    assert tuple(int(v) for v in y.shape) == (2, 7)
    assert lin.telemetry()["bits"] == 8
    assert lin.telemetry()["backend_counts"]["affine"] == 1

    lin8_auto = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=8, backend="auto")
    y8_auto = lin8_auto(x)
    mx.eval(y8_auto)
    assert tuple(int(v) for v in y8_auto.shape) == (2, 7)
    assert lin8_auto.telemetry()["last_backend"] == "affine"

    lin4_auto = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=4, backend="auto")
    y4_auto = lin4_auto(x)
    mx.eval(y4_auto)
    assert tuple(int(v) for v in y4_auto.shape) == (2, 7)
    expected_auto_backend = "metal" if metal_quant_available() else "affine"
    assert lin4_auto.telemetry()["last_backend"] == expected_auto_backend
    if metal_quant_available():
        assert lin4_auto.telemetry()["auto_metal_max_rows"] == 4096

    old_limit = os.environ.get("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS")
    os.environ["RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS"] = "1"
    try:
        lin4_auto_limited = MLXQuantizedLinear.from_linear_weight(linear_weight, bits=4, backend="auto")
        y4_auto_limited = lin4_auto_limited(x)
        mx.eval(y4_auto_limited)
        assert lin4_auto_limited.telemetry()["last_backend"] == "affine"
    finally:
        if old_limit is None:
            os.environ.pop("RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS", None)
        else:
            os.environ["RWKV7_MLX_QUANT_AUTO_W4_METAL_MAX_ROWS"] = old_limit

    # MLX's native groupwise layout is the production Apple W8/W4 path.  Use
    # dimensions divisible by the native group size and verify both execution
    # and the actual packed-storage reduction rather than only the CLI seam.
    groupwise_weight = mx.arange(128 * 64, dtype=mx.float32).reshape(128, 64)
    groupwise_weight = ((groupwise_weight % 31) - 15).astype(mx.float16) / 16
    groupwise_x = mx.ones((3, 64), dtype=mx.float16)
    groupwise_dense_y = groupwise_x @ groupwise_weight.T
    for bits in (8, 4):
        groupwise = MLXQuantizedLinear.from_linear_weight(
            groupwise_weight,
            bits=bits,
            backend="groupwise",
        )
        groupwise_y = groupwise(groupwise_x)
        mx.eval(groupwise_y)
        assert tuple(int(v) for v in groupwise_y.shape) == (3, 128)
        assert bool(mx.all(mx.isfinite(groupwise_y)))
        max_abs = float(mx.max(mx.abs(groupwise_y - groupwise_dense_y)))
        assert max_abs <= (0.03 if bits == 8 else 0.45)
        assert groupwise.storage_bytes < int(groupwise_weight.size * groupwise_weight.itemsize)
        assert groupwise.telemetry()["last_backend"] == "groupwise"
        assert groupwise.telemetry()["backend_counts"]["groupwise"] == 1

        # Wide-to-narrow sequence projections may flatten leading dimensions
        # for the faster Apple prefill route without changing values or shape.
        wide_weight = mx.arange(64 * 128, dtype=mx.float32).reshape(64, 128)
        wide_weight = ((wide_weight % 29) - 14).astype(mx.float16) / 16
        wide = MLXQuantizedLinear.from_linear_weight(
            wide_weight,
            bits=bits,
            backend="groupwise",
            group_size=32,
        )
        wide_x = mx.ones((2, 3, 128), dtype=mx.float16)
        rank_preserving = wide(wide_x)
        flattened = wide(wide_x, flatten_wide=True)
        mx.eval(rank_preserving, flattened)
        assert tuple(int(v) for v in flattened.shape) == (2, 3, 64)
        assert float(mx.max(mx.abs(rank_preserving - flattened))) == 0.0
        embedding_ids = mx.array([[0, 3], [7, 11]], dtype=mx.int32)
        embedding_metal, embedding_backend = groupwise_embedding(
            embedding_ids,
            groupwise.weight,
            backend="auto",
        )
        embedding_reference, _ = groupwise_embedding(
            embedding_ids,
            groupwise.weight,
            backend="reference",
        )
        mx.eval(embedding_metal, embedding_reference)
        assert tuple(int(v) for v in embedding_metal.shape) == (2, 2, 64)
        assert float(mx.max(mx.abs(embedding_metal - embedding_reference))) == 0.0
        assert embedding_backend == ("metal" if metal_quant_available() else "reference")


def test_mlx_model_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    dense_logits, _ = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(dense_logits)

    replaced = model.quantize_linears("mm8", min_params=1, backend="affine")
    assert replaced > 0
    assert "model.embeddings.weight" in model.arrays
    assert "lm_head.weight" in model.quantized_linears
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_count"] == replaced
    assert telemetry["quantized_linear_bits"] == 8
    assert telemetry["quantized_linear_backend"] == "affine"
    assert telemetry["quantized_linear_last_backend_counts"] == {
        "reference": 0,
        "affine": 0,
        "metal": 0,
        "groupwise": 0,
    }
    assert telemetry["quantized_linear_bytes"] > 0
    assert telemetry["quantized_dense_equivalent_bytes"] > 0

    q_logits, q_state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(q_logits)
    assert tuple(int(v) for v in q_logits.shape) == tuple(int(v) for v in dense_logits.shape)
    assert int(q_state.seen_tokens) == 3
    assert model.telemetry()["quantized_linear_last_backend_counts"]["affine"] > 0


def test_mlx_groupwise_batched_rkv_matches_three_calls_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    mx.random.seed(20260714)
    weights = [mx.random.normal((96, 64)).astype(mx.float16) for _ in range(3)]
    packed = [mx.quantize(weight, group_size=32, bits=4, mode="affine") for weight in weights]
    inputs = [mx.random.normal((2, 5, 64)).astype(mx.float16) for _ in range(3)]
    separate = [
        mx.quantized_matmul(
            value,
            q,
            scales=scale,
            biases=bias,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
        )
        for value, (q, scale, bias) in zip(inputs, packed, strict=True)
    ]
    grouped = mx.quantized_matmul(
        mx.stack(inputs, axis=0),
        mx.stack([value[0] for value in packed], axis=0)[:, None],
        scales=mx.stack([value[1] for value in packed], axis=0)[:, None],
        biases=mx.stack([value[2] for value in packed], axis=0)[:, None],
        transpose=True,
        group_size=32,
        bits=4,
        mode="affine",
    )
    mx.eval(grouped, *separate)
    assert tuple(int(dim) for dim in grouped.shape) == (3, 2, 5, 96)
    for index, expected in enumerate(separate):
        assert float(mx.max(mx.abs(grouped[index] - expected))) == 0.0


def test_mlx_model_q4_k_m_mixed_precision_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return

    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears(
        "mm4",
        min_params=1,
        backend="affine",
        profile="q4_k_m",
    )
    assert replaced > 0
    assert model.quantized_linears["lm_head.weight"].bits == 8
    assert model.quantized_linears["model.layers.0.ffn.value.weight"].bits == 8
    assert model.quantized_linears["model.layers.0.attn.r_proj.weight"].bits == 8
    assert model.quantized_linears["model.layers.0.attn.v_proj.weight"].bits == 8
    assert model.quantized_linears["model.layers.0.ffn.key.weight"].bits == 4
    assert model.quantized_linears["model.layers.0.attn.k_proj.weight"].bits == 4
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_profile"] == "q4_k_m"
    assert telemetry["quantized_linear_bits_histogram"][4] > 0
    assert telemetry["quantized_linear_bits_histogram"][8] > 0


def test_mlx_model_metal_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_quant_available():
        return

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears("mm8", min_params=1, backend="metal")
    assert replaced > 0
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_backend"] == "metal"
    logits, state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(logits)
    assert tuple(int(v) for v in logits.shape[:2]) == (1, 1)
    assert int(state.seen_tokens) == 3


def test_mlx_model_rkv_quant_min_params_if_available():
    if importlib.util.find_spec("mlx") is None:
        return

    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears("mm4", min_params=10**9, rkv_min_params=1, backend="auto")
    assert replaced > 0
    keys = set(model.quantized_linears)
    assert "model.layers.0.attn.r_proj.weight" in keys
    assert "model.layers.0.attn.k_proj.weight" in keys
    assert "model.layers.0.attn.v_proj.weight" in keys
    assert "model.layers.0.attn.o_proj.weight" not in keys
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_min_params"] == 10**9
    assert telemetry["quantized_linear_rkv_min_params"] == 1


def test_mlx_model_step_eval_interval_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    old_interval = os.environ.get("RWKV7_MLX_STEP_EVAL_INTERVAL")
    try:
        os.environ["RWKV7_MLX_STEP_EVAL_INTERVAL"] = "1"
        _, baseline, _ = tiny_torch_model_to_mlx()
        os.environ["RWKV7_MLX_STEP_EVAL_INTERVAL"] = "4"
        _, delayed, _ = tiny_torch_model_to_mlx()

        baseline_logits, baseline_state = baseline.forward([[1, 2, 3, 4]], collect_all=False)
        delayed_logits, delayed_state = delayed.forward([[1, 2, 3, 4]], collect_all=False)
        mx.eval(baseline_logits, delayed_logits)

        assert baseline.telemetry()["step_eval_interval"] == 1
        assert delayed.telemetry()["step_eval_interval"] == 4
        assert int(baseline_state.seen_tokens) == int(delayed_state.seen_tokens) == 4
        assert float(mx.max(mx.abs(baseline_logits - delayed_logits))) < 1e-5
    finally:
        if old_interval is None:
            os.environ.pop("RWKV7_MLX_STEP_EVAL_INTERVAL", None)
        else:
            os.environ["RWKV7_MLX_STEP_EVAL_INTERVAL"] = old_interval


def test_mlx_model_grouped_rkv_quant_projection_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_quant_available():
        return

    for quantization in ("mm8", "mm4"):
        _, separate_model, _ = tiny_torch_model_to_mlx()
        _, grouped_model, _ = tiny_torch_model_to_mlx()
        assert separate_model.quantize_linears(quantization, min_params=1, backend="metal") > 0
        assert grouped_model.quantize_linears(quantization, min_params=1, backend="metal") > 0

        separate_logits, separate_state = separate_model.forward([[1, 2, 3]], collect_all=False)
        grouped_model.group_rkv_quant_projection = True
        grouped_logits, grouped_state = grouped_model.forward([[1, 2, 3]], collect_all=False)
        mx.eval(separate_logits, grouped_logits)

        assert tuple(int(v) for v in grouped_logits.shape) == tuple(int(v) for v in separate_logits.shape)
        assert int(separate_state.seen_tokens) == int(grouped_state.seen_tokens) == 3
        assert float(mx.max(mx.abs(separate_logits - grouped_logits))) < 5e-2

        separate_telemetry = separate_model.telemetry()
        grouped_telemetry = grouped_model.telemetry()
        assert separate_telemetry["group_rkv_quant_projection"] is False
        assert separate_telemetry["group_rkv_quant_projection_counts"]["metal"] == 0
        assert grouped_telemetry["group_rkv_quant_projection"] is True
        assert grouped_telemetry["group_rkv_quant_projection_mode"] == "direct"
        assert grouped_telemetry["group_rkv_quant_projection_counts"]["metal"] > 0
        assert grouped_telemetry["group_rkv_quant_projection_counts"]["fallback"] == 0
        assert grouped_model._rkv_group_quant_cache == {}


def test_mlx_model_fused_ffn_key_relu2_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not metal_quant_available():
        return

    _, baseline_model, _ = tiny_torch_model_to_mlx()
    _, fused_model, _ = tiny_torch_model_to_mlx()
    assert baseline_model.quantize_linears("mm4", min_params=1, backend="metal") > 0
    assert fused_model.quantize_linears("mm4", min_params=1, backend="metal") > 0
    fused_model.fused_ffn_key_relu2 = True

    baseline_logits, baseline_state = baseline_model.forward([[1, 2, 3]], collect_all=False)
    fused_logits, fused_state = fused_model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(baseline_logits, fused_logits)

    assert tuple(int(v) for v in fused_logits.shape) == tuple(int(v) for v in baseline_logits.shape)
    assert int(baseline_state.seen_tokens) == int(fused_state.seen_tokens) == 3
    assert float(mx.max(mx.abs(baseline_logits - fused_logits))) < 5e-2

    telemetry = fused_model.telemetry()
    assert telemetry["fused_ffn_key_relu2"] is True
    assert telemetry["fused_ffn_key_relu2_counts"]["metal"] > 0
    assert telemetry["fused_ffn_key_relu2_counts"]["fallback"] == 0


def test_mlx_model_auto_quantized_linear_hook_if_available():
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_quant import metal_quant_available
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    replaced = model.quantize_linears("mm4", min_params=1, backend="auto")
    assert replaced > 0
    logits, state = model.forward([[1, 2, 3]], collect_all=False)
    mx.eval(logits)
    telemetry = model.telemetry()
    assert telemetry["quantized_linear_backend"] == "auto"
    if metal_quant_available():
        assert next(iter(model.quantized_linears.values())).telemetry()["auto_metal_max_rows"] == 4096
    expected = "metal" if metal_quant_available() else "affine"
    assert telemetry["quantized_linear_last_backend_counts"][expected] > 0
    assert int(state.seen_tokens) == 3


if __name__ == "__main__":
    test_mlx_quant_import_safe()
    test_mlx_q4_k_m_profile_policy()
    test_mlx_quant_formula_if_available()
    test_mlx_model_quantized_linear_hook_if_available()
    test_mlx_model_q4_k_m_mixed_precision_hook_if_available()
    test_mlx_model_metal_quantized_linear_hook_if_available()
    test_mlx_model_rkv_quant_min_params_if_available()
    test_mlx_model_step_eval_interval_if_available()
    test_mlx_model_grouped_rkv_quant_projection_if_available()
    test_mlx_model_fused_ffn_key_relu2_if_available()
    test_mlx_model_auto_quantized_linear_hook_if_available()
    print("MLX QUANT TESTS PASS")
