#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        parent = _ensure_module(parent_name)
        setattr(parent, child, module)
    return module


def _install_runtime_stubs() -> None:
    """Install minimal optional-dependency stubs for local cache-only tests."""

    torch_mod = _ensure_module("torch")

    class Tensor:
        pass

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_mod.Tensor = Tensor
    torch_mod.LongTensor = Tensor
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.float32 = "float32"
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
    _ensure_module("torch.nn")
    _ensure_module("torch.nn.functional")

    transformers_mod = _ensure_module("transformers")
    transformers_mod.PreTrainedTokenizer = object
    outputs_mod = _ensure_module("transformers.modeling_outputs")

    class CausalLMOutputWithPast:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    outputs_mod.CausalLMOutputWithPast = CausalLMOutputWithPast


def _install_fla_stubs() -> None:
    """Install minimal FLA stubs so cache bookkeeping can be unit-tested locally."""

    class DummyConfig:
        model_type = "rwkv7"

        def __init__(self, *args, **kwargs):
            pass

    class DummyCache:
        def __init__(self, *args, **kwargs):
            pass

    class DummyModel:
        pass

    class DummyForCausalLM:
        pass

    for name in [
        "fla",
        "fla.models",
        "fla.models.rwkv7",
        "fla.models.rwkv7.configuration_rwkv7",
        "fla.models.rwkv7.modeling_rwkv7",
        "fla.models.utils",
        "fla.ops",
        "fla.ops.rwkv7",
        "fla.ops.rwkv7.fused_recurrent",
    ]:
        _ensure_module(name)

    sys.modules["fla.models.rwkv7.configuration_rwkv7"].RWKV7Config = DummyConfig
    sys.modules["fla.models.rwkv7.modeling_rwkv7"].RWKV7Model = DummyModel
    sys.modules["fla.models.rwkv7.modeling_rwkv7"].RWKV7ForCausalLM = DummyForCausalLM
    sys.modules["fla.models.utils"].Cache = DummyCache
    sys.modules["fla.ops.rwkv7.fused_recurrent"].fused_mul_recurrent_rwkv7 = lambda *args, **kwargs: None


def main() -> int:
    _install_runtime_stubs()
    _install_fla_stubs()
    for name in list(sys.modules):
        if name == "rwkv7_hf" or name.startswith("rwkv7_hf."):
            del sys.modules[name]
    modeling = importlib.import_module("rwkv7_hf.modeling_rwkv7")

    state_cache = modeling.RWKV7StateCache()
    assert state_cache.rwkv7_cache_metrics()["updates"] == 0
    state_cache.update(recurrent_state="r0", layer_idx=0)
    state_cache.update(recurrent_state="r1", layer_idx=1)
    metrics = state_cache.rwkv7_cache_metrics()
    assert metrics["updates"] == 2, metrics
    assert metrics["new_layers"] == 2, metrics
    assert metrics["layers"] == 2, metrics
    assert metrics["seen_tokens"] == 1, metrics
    binding_cache = modeling.RWKV7StateCache()
    binding_cache.update(recurrent_state="r0", layer_idx=0)
    runner_marker = object()
    binding_cache._bind_native_graph_runner(runner_marker)
    assert binding_cache._native_graph_bound_to(runner_marker)
    assert not binding_cache.clone()._native_graph_bound_to(runner_marker)
    binding_cache.update(recurrent_state="r1", layer_idx=0)
    assert not binding_cache._native_graph_bound_to(runner_marker)
    binding_cache._bind_native_graph_runner(runner_marker)
    binding_cache.detach(inplace=True)
    assert not binding_cache._native_graph_bound_to(runner_marker)
    binding_cache._bind_native_graph_runner(runner_marker)
    binding_cache.to(inplace=True)
    assert not binding_cache._native_graph_bound_to(runner_marker)
    binding_cache._bind_native_graph_runner(runner_marker)
    binding_cache.select_batch(object(), inplace=True)
    assert not binding_cache._native_graph_bound_to(runner_marker)
    binding_cache._bind_native_graph_runner(runner_marker)
    binding_cache.reset()
    assert not binding_cache._native_graph_bound_to(runner_marker)

    cloned = state_cache.clone()
    assert cloned.rwkv7_cache_metrics()["clones"] == 1
    cloned.detach(inplace=True)
    cloned.to(inplace=True)
    cloned.select_batch(object(), inplace=True)
    cloned.batch_select(object(), inplace=True)
    cloned.reorder_cache(object())
    metrics = cloned.rwkv7_cache_metrics()
    assert metrics["detaches"] == 1, metrics
    assert metrics["device_moves"] == 1, metrics
    assert metrics["select_batch_calls"] == 3, metrics
    assert metrics["batch_select_calls"] == 1, metrics
    assert metrics["reorder_calls"] == 1, metrics
    cloned.reset()
    metrics = cloned.rwkv7_cache_metrics()
    assert metrics["resets"] == 1 and metrics["layers"] == 0 and metrics["seen_tokens"] == 0, metrics

    created: list[tuple[str, int]] = []

    class ScalarRunner:
        def __init__(self, owner, packs):
            self.batch_size = 1
            created.append(("scalar", 1))

    class BatchedRunner:
        def __init__(self, owner, packs, batch_size: int):
            self.batch_size = int(batch_size)
            created.append(("batched", int(batch_size)))

    modeling._RWKV7NativeGraphTokenRunner = ScalarRunner
    modeling._RWKV7NativeGraphBatchedTokenRunner = BatchedRunner

    class Device:
        type = "cuda"
        index = None

    class Weight:
        device = Device()
        dtype = "float16"

    class Embeddings:
        weight = Weight()

    class BaseModel:
        embeddings = Embeddings()

    class Owner:
        model = BaseModel()

    owner = Owner()
    packs = [(0, 12, 64)]
    old_limit = os.environ.get("RWKV7_NATIVE_GRAPH_CACHE_SIZE")
    old_recurrent_output = os.environ.get("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT")
    os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = "2"
    try:
        os.environ.pop("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", None)
        assert modeling._native_graph_fused_recurrent_output_requested() is True
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT"] = "0"
        assert modeling._native_graph_fused_recurrent_output_requested() is False
        os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT"] = "1"
        assert modeling._native_graph_fused_recurrent_output_requested() is True

        get_runner = modeling.RWKV7ForCausalLM._rwkv7_native_graph_runner
        clear_cache = modeling.RWKV7ForCausalLM.rwkv7_clear_native_graph_cache
        owner.rwkv7_native_graph_cache_batch_sizes = types.MethodType(
            modeling.RWKV7ForCausalLM.rwkv7_native_graph_cache_batch_sizes, owner
        )
        owner.rwkv7_native_graph_cache_stats = types.MethodType(
            modeling.RWKV7ForCausalLM.rwkv7_native_graph_cache_stats, owner
        )
        owner.rwkv7_reset_native_graph_cache_stats = types.MethodType(
            modeling.RWKV7ForCausalLM.rwkv7_reset_native_graph_cache_stats, owner
        )

        r1 = get_runner(owner, packs, 1)
        r2 = get_runner(owner, packs, 2)
        assert r1 is get_runner(owner, packs, 1), "bsz=1 runner should be reused"
        assert created == [("scalar", 1), ("batched", 2)], created

        cache = owner._rwkv7_native_graph_runner_cache
        assert [key[-1] for key in cache.keys()] == [2, 1], list(cache.keys())

        r4 = get_runner(owner, packs, 4)
        assert r4.batch_size == 4
        assert [key[-1] for key in owner._rwkv7_native_graph_runner_cache.keys()] == [1, 4]

        r2_new = get_runner(owner, packs, 2)
        assert r2_new is not r2, "evicted bsz=2 runner should be rebuilt"
        assert [key[-1] for key in owner._rwkv7_native_graph_runner_cache.keys()] == [4, 2]
        stats = owner.rwkv7_native_graph_cache_stats()
        assert stats["requests"] == 5, stats
        assert stats["hits"] == 1, stats
        assert stats["misses"] == 4, stats
        assert stats["evictions"] == 2, stats
        assert stats["batch_sizes"] == [2, 4], stats
        assert abs(stats["hit_rate"] - 0.2) < 1e-9, stats
        reset_stats = owner.rwkv7_reset_native_graph_cache_stats()
        assert reset_stats["requests"] == 0 and reset_stats["hits"] == 0, reset_stats
        assert reset_stats["batch_sizes"] == [2, 4], reset_stats

        assert clear_cache(owner) == 2
        assert len(owner._rwkv7_native_graph_runner_cache) == 0
        assert clear_cache(owner) == 0
        assert modeling._native_graph_cache_size() == 2
        os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = "not-an-int"
        assert modeling._native_graph_cache_size() == 8
    finally:
        if old_limit is None:
            os.environ.pop("RWKV7_NATIVE_GRAPH_CACHE_SIZE", None)
        else:
            os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = old_limit
        if old_recurrent_output is None:
            os.environ.pop("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", None)
        else:
            os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT"] = old_recurrent_output

    old_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")
    old_fast_forward = os.environ.get("RWKV7_FAST_FORWARD")
    old_jit_block_step = modeling._native_jit_block_step
    old_jit_block_step_batched = modeling._native_jit_block_step_batched
    old_jit_extract = modeling._native_jit_extract
    old_graph_block_ip = modeling._native_graph_block_ip
    old_graph_block_ip_batched = modeling._native_graph_block_ip_batched
    try:
        modeling._native_jit_block_step = object()
        modeling._native_jit_block_step_batched = object()
        modeling._native_jit_extract = lambda owner: (packs, None, None, None)
        modeling._native_graph_block_ip = object()
        modeling._native_graph_block_ip_batched = object()

        owner._rwkv7_native_jit_packs = types.MethodType(modeling.RWKV7ForCausalLM._rwkv7_native_jit_packs, owner)
        owner._rwkv7_has_multi_cuda_device_map = types.MethodType(
            modeling.RWKV7ForCausalLM._rwkv7_has_multi_cuda_device_map, owner
        )
        owner._rwkv7_uses_external_quantization = types.MethodType(
            modeling.RWKV7ForCausalLM._rwkv7_uses_external_quantization, owner
        )
        owner._rwkv7_can_use_native_backend = types.MethodType(
            modeling.RWKV7ForCausalLM._rwkv7_can_use_native_backend, owner
        )
        owner._rwkv7_resolve_fast_token_backend = types.MethodType(
            modeling.RWKV7ForCausalLM._rwkv7_resolve_fast_token_backend, owner
        )
        owner._rwkv7_native_graph_runner = types.MethodType(
            modeling.RWKV7ForCausalLM._rwkv7_native_graph_runner, owner
        )
        owner.rwkv7_warmup_fast_token = types.MethodType(modeling.RWKV7ForCausalLM.rwkv7_warmup_fast_token, owner)
        owner.rwkv7_native_graph_cache_batch_sizes = types.MethodType(
            modeling.RWKV7ForCausalLM.rwkv7_native_graph_cache_batch_sizes, owner
        )
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "auto"
        assert modeling._fast_token_backend() == "auto"
        assert owner._rwkv7_resolve_fast_token_backend(1) == "native_graph"
        assert owner._rwkv7_resolve_fast_token_backend(4) == "native_graph"
        assert owner._rwkv7_can_use_native_backend("native_graph", 4) is True
        warmed = owner.rwkv7_warmup_fast_token([1, 4], backend="auto")
        assert warmed == {1: "native_graph", 4: "native_graph"}, warmed
        assert owner.rwkv7_native_graph_cache_batch_sizes() == [1, 4]
        try:
            owner.rwkv7_warmup_fast_token(0)
        except ValueError:
            pass
        else:
            raise AssertionError("batch_size=0 warmup should fail")

        modeling._native_graph_block_ip_batched = None
        assert owner._rwkv7_resolve_fast_token_backend(4) == "native_jit"
        assert owner.rwkv7_warmup_fast_token([4], backend="auto") == {4: "native_jit"}

        owner.is_loaded_in_4bit = True
        assert owner._rwkv7_resolve_fast_token_backend(1) == "fla"
        owner.is_loaded_in_4bit = False

        owner.hf_device_map = {"model.embeddings": 0, "model.layers.0": 1}
        assert owner._rwkv7_has_multi_cuda_device_map() is True
        assert owner._rwkv7_resolve_fast_token_backend(1) == "fla"
        owner.hf_device_map = {"": 0}
        assert owner._rwkv7_has_multi_cuda_device_map() is False

        modeling._native_jit_extract = lambda owner: (_ for _ in ()).throw(RuntimeError("extract failed"))
        modeling._native_graph_block_ip = None
        owner._rwkv7_native_jit_pack_cache = None
        assert owner._rwkv7_resolve_fast_token_backend(1) == "fla"

        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "graph"
        assert modeling._fast_token_backend() == "native_graph"
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "jit"
        assert modeling._fast_token_backend() == "native_jit"
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "unknown"
        assert modeling._fast_token_backend() == "fla"
        os.environ["RWKV7_FAST_FORWARD"] = "0"
        assert modeling._fast_forward_enabled() is False
        os.environ["RWKV7_FAST_FORWARD"] = "1"
        assert modeling._fast_forward_enabled() is True
    finally:
        modeling._native_jit_block_step = old_jit_block_step
        modeling._native_jit_block_step_batched = old_jit_block_step_batched
        modeling._native_jit_extract = old_jit_extract
        modeling._native_graph_block_ip = old_graph_block_ip
        modeling._native_graph_block_ip_batched = old_graph_block_ip_batched
        if old_backend is None:
            os.environ.pop("RWKV7_FAST_TOKEN_BACKEND", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = old_backend
        if old_fast_forward is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old_fast_forward

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
