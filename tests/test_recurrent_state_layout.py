from __future__ import annotations

import pytest
import torch

from rwkv7_hf import native_graph_runtime as native_graph_runtime_module
from rwkv7_hf.model_cache import (
    NativeRWKV7Cache,
    _copy_native_cache_tuple,
    _native_cache_tuple_or_none,
)
from rwkv7_hf.native_graph_runtime import native_graph_recurrent_decode_route
from rwkv7_hf.recurrent_state import (
    RecurrentStateLayout,
    convert_recurrent_state_tensor,
    normalize_recurrent_state_layout,
    resolve_recurrent_decode_route,
)


def _build_cache(layout: str = "vk_v1") -> NativeRWKV7Cache:
    state_vk = torch.tensor([[[[1.0, 2.0, 3.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]]]])
    state = state_vk if layout == "vk_v1" else state_vk.transpose(-1, -2).contiguous()
    return NativeRWKV7Cache(
        [state],
        [torch.zeros(1, 3)],
        [torch.zeros(1, 3)],
        torch.zeros(1, 3),
        seen_tokens=7,
        state_layout=layout,
    )


def test_layout_names_are_explicit_and_fail_closed() -> None:
    assert normalize_recurrent_state_layout(None) == RecurrentStateLayout.VK_V1
    assert normalize_recurrent_state_layout("v1") == RecurrentStateLayout.VK_V1
    assert normalize_recurrent_state_layout("kv") == RecurrentStateLayout.KV_V2
    with pytest.raises(ValueError, match="recurrent-state layout"):
        normalize_recurrent_state_layout("shape_is_square_so_guess")


def test_vk_v1_and_kv_v2_math_are_transposes() -> None:
    state_vk = torch.tensor([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]])
    r = torch.tensor([0.2, -0.3, 0.5])
    w = torch.tensor([0.7, 0.8, 0.9])
    k = torch.tensor([0.4, -0.6, 0.1])
    v = torch.tensor([0.9, 0.3, -0.2])
    kk = torch.tensor([0.5, -0.25, 0.75])
    a = torch.tensor([0.6, 0.4, 0.2])

    state_dot_kk = state_vk @ kk
    new_vk = (
        state_vk * w[None, :]
        + v[:, None] * k[None, :]
        - state_dot_kk[:, None] * (kk * a)[None, :]
    )
    out_vk = new_vk @ r

    state_kv = convert_recurrent_state_tensor(
        state_vk,
        RecurrentStateLayout.VK_V1,
        RecurrentStateLayout.KV_V2,
    )
    kk_dot_state = kk @ state_kv
    new_kv = (
        w[:, None] * state_kv
        + k[:, None] * v[None, :]
        - (kk * a)[:, None] * kk_dot_state[None, :]
    )
    out_kv = new_kv.transpose(-1, -2) @ r

    torch.testing.assert_close(new_kv, new_vk.transpose(-1, -2))
    torch.testing.assert_close(out_kv, out_vk)
    round_trip = convert_recurrent_state_tensor(
        state_kv,
        RecurrentStateLayout.KV_V2,
        RecurrentStateLayout.VK_V1,
    )
    assert round_trip.is_contiguous()
    torch.testing.assert_close(round_trip, state_vk)


def test_cache_lifecycle_and_legacy_tuple_preserve_layout() -> None:
    cache = _build_cache("kv_v2")
    assert cache.state_layout == "kv_v2"
    assert cache.rwkv7_cache_metrics()["state_layout"] == "kv_v2"

    legacy = cache.to_legacy_cache()
    assert isinstance(legacy, tuple) and len(legacy) == 4
    assert legacy.state_layout == "kv_v2"
    assert _native_cache_tuple_or_none(cache).state_layout == "kv_v2"

    restored = NativeRWKV7Cache.from_legacy_cache(legacy)
    assert restored.state_layout == "kv_v2"
    assert restored.get_seq_length() == 7
    assert restored.clone().state_layout == "kv_v2"
    assert restored.detach(inplace=False).state_layout == "kv_v2"
    assert restored.to(copy=True, inplace=False).state_layout == "kv_v2"
    assert (
        restored.select_batch(torch.tensor([0]), inplace=False).state_layout == "kv_v2"
    )
    assert restored.batch_repeat_interleave(2).state_layout == "kv_v2"

    assert _native_cache_tuple_or_none(NativeRWKV7Cache()) is None


def test_existing_paths_copy_kv_v2_cache_to_canonical_vk_v1_once() -> None:
    cache = _build_cache("kv_v2")
    physical_kv = cache._state[0].clone()
    state, _, _, _ = _copy_native_cache_tuple(cache)
    assert state[0].is_contiguous()
    torch.testing.assert_close(state[0], physical_kv.transpose(-1, -2))
    # Conversion returns independent physical storage; the tagged source cache
    # remains kv_v2 until a runner explicitly binds it to another layout.
    assert state[0].data_ptr() != cache._state[0].data_ptr()
    assert cache.state_layout == "kv_v2"


def test_decode_route_is_opt_in_and_fail_closed(monkeypatch) -> None:
    route = resolve_recurrent_decode_route(state_dtype=torch.float32)
    assert route.state_layout == RecurrentStateLayout.VK_V1
    assert route.implementation == "existing_vk_v1"
    assert route.source == "conservative_fallback"

    explicit_v1 = resolve_recurrent_decode_route(
        state_dtype=torch.float16,
        requested_layout="vk_v1",
    )
    assert explicit_v1.state_layout == RecurrentStateLayout.VK_V1
    assert explicit_v1.source == "explicit"

    with pytest.raises(RuntimeError, match="no kv_v2 decode kernel"):
        resolve_recurrent_decode_route(
            state_dtype=torch.float32,
            requested_layout="kv_v2",
        )

    enabled = resolve_recurrent_decode_route(
        state_dtype=torch.float32,
        requested_layout="kv_v2",
        kv_v2_kernel_available=True,
    )
    assert enabled.state_layout == RecurrentStateLayout.KV_V2
    assert enabled.signature()[0] == "kv_v2"

    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_STATE_LAYOUT", raising=False)
    assert (
        native_graph_recurrent_decode_route(torch.float32).state_layout
        == RecurrentStateLayout.VK_V1
    )
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_LAYOUT", "kv_v2")
    monkeypatch.setattr(
        native_graph_runtime_module,
        "fused_recurrent_output_prepare_raw_kv_v2_available",
        lambda **_: False,
    )
    with pytest.raises(RuntimeError, match="no kv_v2 decode kernel"):
        native_graph_recurrent_decode_route(torch.float32, head_dim=64)

    monkeypatch.setattr(
        native_graph_runtime_module,
        "fused_recurrent_output_prepare_raw_kv_v2_available",
        lambda **kwargs: (
            kwargs["state_dtype"] == torch.float32
            and kwargs["head_dim"] == 64
        ),
    )
    route = native_graph_recurrent_decode_route(torch.float32, head_dim=64)
    assert route.state_layout == RecurrentStateLayout.KV_V2
    assert route.source == "explicit"
    with pytest.raises(RuntimeError, match="no kv_v2 decode kernel"):
        native_graph_recurrent_decode_route(torch.float16, head_dim=64)
    with pytest.raises(RuntimeError, match="no kv_v2 decode kernel"):
        native_graph_recurrent_decode_route(torch.float32, head_dim=32)
