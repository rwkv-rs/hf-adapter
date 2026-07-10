from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from rwkv7_hf.cache_rwkv7 import RWKV7Cache
from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model


ROOT = Path(__file__).resolve().parents[1]


def _load_dense_backend(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)
    return importlib.import_module("rwkv7_kernels.model.dense")


def test_migrated_dense_model_math_cache_padding_and_hidden_states(
    tiny_config, monkeypatch
):
    dense = _load_dense_backend(monkeypatch)
    torch.manual_seed(107)
    model = RWKV7Model(tiny_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)

    ids = torch.tensor([[3, 5, 7, 0], [0, 11, 13, 17]])
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    with torch.inference_mode():
        expected = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden = model.embeddings(ids) * mask.unsqueeze(-1)
        actual = dense.run_base_model(
            model,
            {
                "model_kind": "base",
                "hidden_states": hidden,
                "attention_mask": mask,
                "past_key_values": RWKV7Cache(
                    num_layers=tiny_config.num_hidden_layers
                ),
                "training": False,
                "use_cache": True,
                "output_hidden_states": True,
            },
        )

    torch.testing.assert_close(
        actual["last_hidden_state"],
        expected.last_hidden_state,
        rtol=2e-5,
        atol=2e-6,
    )
    assert len(actual["hidden_states"]) == len(expected.hidden_states)
    for migrated, reference in zip(actual["hidden_states"], expected.hidden_states):
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)
    assert actual["past_key_values"].seen_tokens == ids.shape[1]
    for migrated, reference in zip(
        actual["past_key_values"].recurrent_state,
        expected.past_key_values.recurrent_state,
    ):
        # This assertion also catches an accidental [V,K] ABI leak.
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)


def test_migrated_native_prefill_and_cached_decode_preserve_canonical_cache(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    torch.manual_seed(109)
    model = RWKV7ForCausalLM(tiny_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)

    prompt = torch.tensor([[3, 5, 7], [11, 13, 17]])
    with torch.inference_mode():
        expected_prefill = model(input_ids=prompt, use_cache=True)
        migrated_prefill = dispatcher._run_native_prefill(
            model,
            {
                "model_kind": "causal_lm",
                "input_ids": prompt,
                "past_key_values": RWKV7Cache(
                    num_layers=tiny_config.num_hidden_layers
                ),
                "training": False,
                "grad_enabled": False,
                "use_cache": True,
                "logits_to_keep": 0,
            },
        )

    torch.testing.assert_close(
        migrated_prefill["logits"], expected_prefill.logits, rtol=2e-5, atol=2e-6
    )
    migrated_cache = migrated_prefill["past_key_values"]
    assert migrated_cache.seen_tokens == prompt.shape[1]
    for migrated, reference in zip(
        migrated_cache.recurrent_state,
        expected_prefill.past_key_values.recurrent_state,
    ):
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)

    expected_cache = expected_prefill.past_key_values.clone()
    migrated_cache = migrated_cache.clone()
    next_token = torch.tensor([[19], [23]])
    with torch.inference_mode():
        expected_decode = model(
            input_ids=next_token,
            past_key_values=expected_cache,
            use_cache=True,
        )
        migrated_decode = dispatcher._run_native_decode(
            model,
            {
                "model_kind": "causal_lm",
                "input_ids": next_token,
                "past_key_values": migrated_cache,
                "training": False,
                "grad_enabled": False,
                "use_cache": True,
            },
        )

    torch.testing.assert_close(
        migrated_decode["logits"], expected_decode.logits, rtol=2e-5, atol=2e-6
    )
    assert migrated_decode["past_key_values"].seen_tokens == prompt.shape[1] + 1
    for migrated, reference in zip(
        migrated_decode["past_key_values"].recurrent_state,
        expected_decode.past_key_values.recurrent_state,
    ):
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)


def test_graph_runner_binding_exposes_only_canonical_cache_views(monkeypatch):
    _load_dense_backend(monkeypatch)
    graph = importlib.import_module("rwkv7_kernels.nvidia.native_graph_runtime")
    layout = importlib.import_module("rwkv7_kernels.nvidia.recurrent_state")

    runner = graph.NativeGraphRunner.__new__(graph.NativeGraphRunner)
    runner.num_layers = 1
    runner.single = False
    runner.state_layout = layout.RecurrentStateLayout.VK_V1
    runner.state = [torch.zeros(2, 1, 3, 3)]
    runner.xpa = [torch.zeros(2, 4)]
    runner.xpf = [torch.zeros(2, 4)]
    runner.elapsed = None
    runner._bound_cache_ref = None
    runner.copy_from_cache_calls = 0
    runner.copy_from_cache_fast_skips = 0
    runner.bind_cache_calls = 0
    runner.bind_cache_fast_skips = 0

    canonical = torch.arange(18, dtype=torch.float32).view(2, 1, 3, 3)
    attention = torch.randn(2, 4)
    ffn = torch.randn(2, 4)
    cache = RWKV7Cache([canonical.clone()], [attention.clone()], [ffn.clone()])
    runner.copy_from_cache(cache)
    torch.testing.assert_close(runner.state[0], canonical.transpose(-1, -2))
    runner.bind_cache(cache)
    torch.testing.assert_close(cache.recurrent_state[0], canonical)
    assert runner._cache_bound_to_runner(cache)

    runner.state[0].add_(1)
    torch.testing.assert_close(cache.recurrent_state[0], canonical + 1)
    runner.detach_bound_cache()
    detached = cache.recurrent_state[0].clone()
    runner.state[0].add_(1)
    torch.testing.assert_close(cache.recurrent_state[0], detached)
