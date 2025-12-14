from __future__ import annotations

import torch

from rwkv7_hf import blackwell_norm_mix, native_jit


def test_blackwell_norm_mix_rejects_cpu_without_building() -> None:
    value = torch.zeros(1, 2048, dtype=torch.float16)
    assert not blackwell_norm_mix.blackwell_norm_mix_should_use(value, value, value)


def test_blackwell_norm_mix_keeps_official_reduction_contract() -> None:
    source = blackwell_norm_mix._CUDA_SOURCE
    assert "constexpr int THREADS = 256;" in source
    assert "block_sum<BlockThreads>(sum)" in source
    assert "__floats2half2_rn" in source
    assert "previous)[pair_base + p] = normalized" in source


def test_blackwell_norm_mix_route_is_explicit(monkeypatch) -> None:
    value = torch.zeros(1, 8, dtype=torch.float16)
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", raising=False)
    assert not native_jit._native_graph_blackwell_norm_mix_enabled(
        value, value, value, layer_index=23
    )

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", "1")
    monkeypatch.setattr(native_jit, "blackwell_ffn_add_norm_mix", object())
    monkeypatch.setattr(native_jit, "blackwell_norm_mix_should_use", lambda *_args: True)
    assert native_jit._native_graph_blackwell_norm_mix_enabled(
        value, value, value, layer_index=23
    )

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS", "23")
    assert native_jit._native_graph_blackwell_norm_mix_enabled(
        value, value, value, layer_index=23
    )
    assert not native_jit._native_graph_blackwell_norm_mix_enabled(
        value, value, value, layer_index=22
    )


def test_blackwell_norm_mix_layers_can_be_selected_per_batch(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", "1")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS", "9")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS_B1", "7")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS_B8", "8")
    monkeypatch.setattr(native_jit, "blackwell_ffn_add_norm_mix", object())
    monkeypatch.setattr(native_jit, "blackwell_norm_mix_should_use", lambda *_args: True)

    single = torch.zeros(16)
    batched = torch.zeros(8, 16)
    other_batch = torch.zeros(4, 16)
    assert native_jit._native_graph_blackwell_norm_mix_enabled(
        single, single, single, layer_index=7
    )
    assert native_jit._native_graph_blackwell_norm_mix_enabled(
        batched, batched, batched, layer_index=8
    )
    assert native_jit._native_graph_blackwell_norm_mix_enabled(
        other_batch, other_batch, other_batch, layer_index=9
    )
