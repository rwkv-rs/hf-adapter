from __future__ import annotations

from types import SimpleNamespace

import torch

from rwkv7_hf.native import _cuda_peer_copy_usable, _eager_model_is_multi_device
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


def _owner(device_map):
    return SimpleNamespace(
        hf_device_map=device_map,
        parameters=lambda: iter(()),
        _rwkv7_has_multi_cuda_device_map=lambda: True,
    )


def test_native_model_detects_multi_cuda_device_map() -> None:
    detect = NativeRWKV7ForCausalLM._rwkv7_has_multi_cuda_device_map
    assert detect(_owner({"model.embeddings": 0, "model.layers.0": 1}))
    owner = _owner({"": 0})
    owner._rwkv7_has_multi_cuda_device_map = None
    assert not detect(owner)


def test_cuda_peer_copy_fails_closed_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_CUDA_PEER_COPY", raising=False)
    assert not _cuda_peer_copy_usable(torch.device("cuda:0"), torch.device("cuda:1"))


def test_eager_multi_device_detection_is_cached() -> None:
    calls = []
    owner = SimpleNamespace(
        _rwkv7_has_multi_cuda_device_map=lambda: calls.append(True) or True,
    )
    assert _eager_model_is_multi_device(owner)
    assert _eager_model_is_multi_device(owner)
    assert len(calls) == 1


def test_multi_cuda_device_map_disables_every_packed_route(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    owner = _owner({"model.embeddings": 0, "model.layers.0": 1})
    token_ids = torch.ones(1, 2, dtype=torch.long)

    assert not NativeRWKV7ForCausalLM._native_prefill_can_run(
        owner,
        token_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=True,
        logits_to_keep=1,
    )
    assert not NativeRWKV7ForCausalLM._native_graph_can_run(
        owner,
        token_ids[:, :1],
        SimpleNamespace(),
        attention_mask=None,
        output_hidden_states=False,
    )
    assert NativeRWKV7ForCausalLM._native_jit_packs(owner) is None
