from types import SimpleNamespace
import weakref

import torch

from rwkv7_hf import ascend_graph_runtime as graph_runtime
from rwkv7_hf.native import _init_state_batched
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def test_graph_availability_is_capability_based(monkeypatch):
    monkeypatch.setattr(torch, "npu", None, raising=False)
    assert not graph_runtime.ascend_graph_available()

    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        NPUGraph=object,
        graph=lambda graph: graph,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    assert graph_runtime.ascend_graph_available()


def test_graph_cache_size_is_bounded_and_fail_safe(monkeypatch):
    monkeypatch.delenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", raising=False)
    assert graph_runtime.ascend_graph_cache_size() == 3

    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "0")
    assert graph_runtime.ascend_graph_cache_size() == 1

    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "not-an-integer")
    assert graph_runtime.ascend_graph_cache_size() == 3


def test_graph_runtime_signature_tracks_only_ascend_graph_and_quant(monkeypatch):
    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "4")
    monkeypatch.setenv("RWKV7_ASCEND_QUANT_POLICY", "candidate")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH", "1")

    assert graph_runtime.ascend_graph_runtime_signature() == (
        ("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "4"),
        ("RWKV7_ASCEND_QUANT_POLICY", "candidate"),
    )


def test_quant_buffer_replacement_changes_graph_module_signature():
    class PackedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bit = 8
            self.group_size = 0
            self.register_buffer("qweight", torch.ones(2, 3, dtype=torch.int8))
            self.register_buffer("scales", torch.ones(3, dtype=torch.float16))

    layers = torch.nn.ModuleList([PackedProjection()])
    owner = SimpleNamespace(model=SimpleNamespace(layers=layers))

    before = graph_runtime.ascend_graph_module_signature(owner)
    layers[0].qweight = layers[0].qweight.clone()
    after = graph_runtime.ascend_graph_module_signature(owner)

    assert before
    assert before != after


def test_hf_w8_buffer_replacement_changes_graph_module_signature():
    from rwkv7_hf.ascend_quant import AscendW8A16Linear

    layers = torch.nn.ModuleList([AscendW8A16Linear(2, 3, admitted_rows=(1,))])
    layers[0].q_weight = torch.ones(2, 3, dtype=torch.int8)
    layers[0].scale = torch.ones(3, dtype=torch.float16)
    owner = SimpleNamespace(model=SimpleNamespace(layers=layers))

    before = graph_runtime.ascend_graph_module_signature(owner)
    layers[0].q_weight = layers[0].q_weight.clone()
    after = graph_runtime.ascend_graph_module_signature(owner)

    assert before
    assert before != after


def test_graph_one_step_matches_current_modular_eager_model(monkeypatch):
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "eager")
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    torch.manual_seed(20260804)
    model = NativeRWKV7ForCausalLM(config).eval()
    token_ids = torch.tensor([3, 7], dtype=torch.long)
    with torch.inference_mode():
        reference = model(token_ids[:, None], use_cache=True).logits[:, -1]

        runner = object.__new__(graph_runtime.AscendGraphRunner)
        runner.owner_ref = weakref.ref(model)
        runner.batch_size = 2
        runner.device = torch.device("cpu")
        runner.dtype = model.model.embeddings.weight.dtype
        runner.state, runner.xpa, runner.xpf, runner.v_first = _init_state_batched(
            model,
            runner.batch_size,
            runner.device,
            runner.dtype,
        )
        runner.token_ids = token_ids.clone()
        runner.logits = torch.empty(
            runner.batch_size,
            config.vocab_size,
            dtype=runner.dtype,
        )
        runner._one_step()

    torch.testing.assert_close(runner.logits, reference, rtol=1e-5, atol=1e-6)
