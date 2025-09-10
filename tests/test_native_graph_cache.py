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
        type = "cpu"
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
    os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = "2"
    try:
        get_runner = modeling.RWKV7ForCausalLM._rwkv7_native_graph_runner
        clear_cache = modeling.RWKV7ForCausalLM.rwkv7_clear_native_graph_cache

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

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
