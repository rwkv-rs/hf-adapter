from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from rwkv7_hf import native_model


def test_native_prefill_graph_is_explicit_and_signature_tracks_flags(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native_model, "_native_jit_prefill", object())
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_GRAPH", raising=False)
    assert not native_model._native_prefill_graph_enabled()

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH", "1")
    assert native_model._native_prefill_graph_enabled()
    before = native_model._native_prefill_graph_signature()
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT", "1")
    after = native_model._native_prefill_graph_signature()
    assert before != after
    assert ("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT", "1") in after


def test_native_prefill_graph_signature_tracks_shift_mix_precision(monkeypatch) -> None:
    from rwkv7_hf import modeling_rwkv7

    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16", raising=False)
    before = modeling_rwkv7._native_prefill_graph_signature()
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16", "1")
    after = modeling_rwkv7._native_prefill_graph_signature()

    assert before != after
    assert ("RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16", "1") in after

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SHIFT_MIX_STRICT_FP16", "1")
    generic = modeling_rwkv7._native_prefill_graph_signature()
    assert generic != after
    assert ("RWKV7_NATIVE_PREFILL_SHIFT_MIX_STRICT_FP16", "1") in generic

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16", "1")
    split = modeling_rwkv7._native_prefill_graph_signature()
    assert split != generic
    assert ("RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16", "1") in split


def test_native_prefill_graph_policy_is_exact_shape_allowlisted(monkeypatch) -> None:
    policy = SimpleNamespace(
        prefill_graph=True,
        prefill_graph_model_shapes=((4096, 61, 8, 512),),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native_model, "_native_jit_prefill", object())
    monkeypatch.setattr(native_model, "current_kernel_policy", lambda **_kwargs: policy)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_GRAPH", raising=False)

    assert native_model._native_prefill_graph_enabled(8, 512, 4096, 61)
    assert not native_model._native_prefill_graph_enabled(8, 2048, 4096, 61)
    assert not native_model._native_prefill_graph_enabled()

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH", "1")
    assert native_model._native_prefill_graph_enabled(8, 2048, 4096, 61)


def test_native_prefill_graph_cache_size_uses_policy_and_env_override(monkeypatch) -> None:
    policy = SimpleNamespace(prefill_graph_cache_size=4)
    monkeypatch.setattr(native_model, "current_kernel_policy", lambda **_kwargs: policy)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", raising=False)

    assert native_model._native_prefill_graph_cache_size() == 4

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", "7")
    assert native_model._native_prefill_graph_cache_size() == 7

    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", "invalid")
    assert native_model._native_prefill_graph_cache_size() == 4


def test_native_prefill_graph_runner_cache_is_shape_keyed_lru(monkeypatch) -> None:
    created = []

    class FakeRunner:
        def __init__(self, owner, packs, batch_size, prompt_tokens, logits_to_keep):
            self.batch_size = int(batch_size)
            self.prompt_tokens = int(prompt_tokens)
            self.logits_to_keep = logits_to_keep
            self.detached = 0
            created.append((self.batch_size, self.prompt_tokens, self.logits_to_keep))

        def detach_bound_cache(self):
            self.detached += 1

    class Device:
        type = "cuda"
        index = 0

    owner = SimpleNamespace(
        model=SimpleNamespace(embeddings=SimpleNamespace(weight=SimpleNamespace(device=Device(), dtype=torch.float16))),
        _rwkv7_native_mm_quantization="none",
    )
    owner._native_graph_packs = lambda: [(0, 64, 64)]
    owner.rwkv7_native_prefill_graph_cache_shapes = MethodType(
        native_model.NativeRWKV7ForCausalLM.rwkv7_native_prefill_graph_cache_shapes,
        owner,
    )
    owner.rwkv7_native_prefill_graph_cache_stats = MethodType(
        native_model.NativeRWKV7ForCausalLM.rwkv7_native_prefill_graph_cache_stats,
        owner,
    )

    monkeypatch.setattr(native_model, "_NativePrefillGraphRunner", FakeRunner)
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", "1")
    get_runner = native_model.NativeRWKV7ForCausalLM._native_prefill_graph_runner
    first = get_runner(owner, 1, 128, 1)
    assert first is get_runner(owner, 1, 128, 1)
    second = get_runner(owner, 8, 128, 1)
    assert second is not first
    assert first.detached == 1
    assert created == [(1, 128, 1), (8, 128, 1)]
    stats = owner.rwkv7_native_prefill_graph_cache_stats()
    assert stats["requests"] == 3
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["evictions"] == 1
    assert stats["shapes"] == [(8, 128)]


def test_native_prefill_graph_replay_detaches_previous_cache() -> None:
    class FakeGraph:
        def __init__(self, state):
            self.state = state

        def replay(self):
            self.state.add_(1)

    runner = object.__new__(native_model._NativePrefillGraphRunner)
    runner.batch_size = 1
    runner.prompt_tokens = 2
    runner.device = torch.device("cpu")
    runner.input_ids = torch.zeros(1, 2, dtype=torch.long)
    runner.logits = torch.tensor([[[3.0]]])
    runner.state_outputs = [torch.zeros(1, 1, 1, 1)]
    runner.xpa_outputs = [torch.zeros(1, 1)]
    runner.xpf_outputs = [torch.zeros(1, 1)]
    runner.v_first = torch.zeros(1, 1)
    runner.graph = FakeGraph(runner.state_outputs[0])
    runner._bound_cache_ref = None

    _, first = runner.replay(torch.ones(1, 2, dtype=torch.long), seen_tokens=2)
    assert first._state[0].item() == 1
    _, second = runner.replay(torch.full((1, 2), 2, dtype=torch.long), seen_tokens=2)
    assert first._state[0].item() == 1
    assert second._state[0].item() == 2
    assert not first._native_graph_bound_to(runner)
    assert second._native_graph_bound_to(runner)

    # Decode runners replace the cache tensors with their own stable buffers
    # before rebinding. Prefill can then reuse its graph outputs without a
    # redundant safety clone.
    second._state = [second._state[0].clone()]
    second._xpa = [second._xpa[0].clone()]
    second._xpf = [second._xpf[0].clone()]
    second._v_first = second._v_first.clone()
    second._bind_native_graph_runner(object())
    _, third = runner.replay(torch.full((1, 2), 3, dtype=torch.long), seen_tokens=2)
    assert second._state[0].item() == 2
    assert third._state[0].item() == 3


def test_adapter_scan_caches_negative_but_peft_metadata_invalidates_it() -> None:
    scans = []
    plain = object()
    peft_type = type("LoraLayer", (), {"__module__": "peft.tuners.lora"})
    peft = peft_type()
    peft.base_layer = object()
    peft.lora_A = object()

    class Layers:
        values = [plain]

        def modules(self):
            scans.append(1)
            return iter(self.values)

    owner = SimpleNamespace(model=SimpleNamespace(layers=Layers()))
    check = native_model.NativeRWKV7ForCausalLM._native_model_has_adapter_layers
    assert not check(owner)
    assert not check(owner)
    assert len(scans) == 1

    owner.peft_config = {"default": object()}
    owner.model.layers.values = [plain, peft]
    assert check(owner)
    assert len(scans) == 2
