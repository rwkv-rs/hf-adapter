from __future__ import annotations

import pytest

from bench.bench_decode_components import cache_layer_state, store_native_cache_layer


class _NativeCacheStub:
    def __init__(self) -> None:
        self._state = ["state-0"]
        self._xpa = ["attn-shift-0"]
        self._xpf = ["ffn-shift-0"]


class _WrapperCacheStub:
    def __init__(self) -> None:
        self.states = []

    def _ensure_layer(self, layer_idx: int):
        while len(self.states) <= layer_idx:
            self.states.append(
                {
                    "recurrent_state": None,
                    "attn_state": None,
                    "conv_state": None,
                    "ffn_state": None,
                }
            )
        return self.states[layer_idx]


def test_decode_component_cache_adapter_supports_native_layout() -> None:
    cache = _NativeCacheStub()
    state, native_layout = cache_layer_state(cache, 0)

    assert native_layout is True
    assert state == {
        "recurrent_state": "state-0",
        "attn_state": None,
        "conv_state": "attn-shift-0",
        "ffn_state": "ffn-shift-0",
    }

    state.update(
        recurrent_state="state-1",
        conv_state="attn-shift-1",
        ffn_state="ffn-shift-1",
    )
    store_native_cache_layer(cache, 0, state)
    assert cache._state == ["state-1"]
    assert cache._xpa == ["attn-shift-1"]
    assert cache._xpf == ["ffn-shift-1"]


def test_decode_component_cache_adapter_preserves_wrapper_layout() -> None:
    cache = _WrapperCacheStub()
    state, native_layout = cache_layer_state(cache, 2)

    assert native_layout is False
    assert state is cache.states[2]


def test_decode_component_cache_adapter_rejects_unknown_layout() -> None:
    with pytest.raises(TypeError, match="Unsupported recurrent cache type"):
        cache_layer_state(object(), 0)
