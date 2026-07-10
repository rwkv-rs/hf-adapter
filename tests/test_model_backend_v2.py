from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

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


def test_native_model_runtime_compacts_left_right_padding_without_state_updates(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    torch.manual_seed(110)
    model = RWKV7ForCausalLM(tiny_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)

    prompt = torch.tensor([[3, 5, 0, 0], [0, 0, 11, 13]])
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    with torch.inference_mode():
        expected = model(input_ids=prompt, attention_mask=mask, use_cache=True)
        actual = dispatcher._run_native_prefill(
            model,
            {
                "model_kind": "causal_lm",
                "input_ids": prompt,
                "attention_mask": mask,
                "past_key_values": RWKV7Cache(
                    num_layers=tiny_config.num_hidden_layers
                ),
                "training": False,
                "grad_enabled": False,
                "use_cache": True,
                "logits_to_keep": 0,
            },
        )
    torch.testing.assert_close(actual["logits"], expected.logits, rtol=2e-5, atol=2e-6)
    assert "masked_compact" in actual["implementation"]
    for migrated, reference in zip(
        actual["past_key_values"].recurrent_state,
        expected.past_key_values.recurrent_state,
    ):
        assert migrated.dtype == torch.float32
        torch.testing.assert_close(migrated, reference, rtol=2e-5, atol=2e-6)

    expected_cache = expected.past_key_values.clone()
    actual_cache = actual["past_key_values"].clone()
    token = torch.tensor([[17], [19]])
    decode_mask = torch.tensor([[1], [0]], dtype=torch.bool)
    with torch.inference_mode():
        expected_decode = model(
            input_ids=token,
            attention_mask=decode_mask,
            past_key_values=expected_cache,
            use_cache=True,
        )
        actual_decode = dispatcher._run_native_decode(
            model,
            {
                "model_kind": "causal_lm",
                "input_ids": token,
                "attention_mask": decode_mask,
                "past_key_values": actual_cache,
                "training": False,
                "grad_enabled": False,
                "use_cache": True,
            },
        )
    torch.testing.assert_close(
        actual_decode["logits"], expected_decode.logits, rtol=2e-5, atol=2e-6
    )
    assert "masked_compact" in actual_decode["implementation"]
    for migrated, reference in zip(
        actual_decode["past_key_values"].recurrent_state,
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


def test_training_runtime_uses_direct_layer_loop_without_monkeypatch(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    runtime = importlib.import_module("rwkv7_kernels.nvidia.training_runtime")
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.train_temp_cuda")
    monkeypatch.setattr(train_temp, "load_train_temp_cuda_extension", lambda: None)

    def attention_forward(module, hidden, v_first, *, native_lora_math):
        del native_lora_math
        batch, tokens, _ = hidden.shape
        state = torch.zeros(
            batch,
            module.num_heads,
            module.head_dim,
            module.head_dim,
            dtype=torch.float32,
            device=hidden.device,
        )
        shift = torch.zeros(
            batch, module.hidden_size, dtype=hidden.dtype, device=hidden.device
        )
        mask = torch.ones(batch, tokens, dtype=torch.bool, device=hidden.device)
        output, _state, _shift, v_first = module(
            hidden, state, shift, v_first, mask
        )
        return output, v_first

    class FakeCMix:
        @staticmethod
        def apply(hidden, x_k, key_weight, value_weight):
            shifted = torch.cat(
                (torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1
            )
            mixed = hidden + (shifted - hidden) * x_k.view(1, 1, -1)
            return F.linear(torch.relu(F.linear(mixed, key_weight)).square(), value_weight)

    def causal_loss(logits, labels):
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
        )

    monkeypatch.setattr(train_temp, "_train_temp_attention_forward", attention_forward)
    monkeypatch.setattr(train_temp, "_CMix", FakeCMix)
    monkeypatch.setattr(train_temp, "train_temp_causal_cross_entropy", causal_loss)

    torch.manual_seed(113)
    reference = RWKV7ForCausalLM(tiny_config).train()
    migrated = RWKV7ForCausalLM(tiny_config).train()
    migrated.load_state_dict(reference.state_dict())
    ids = torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])

    expected = reference(input_ids=ids, labels=ids, use_cache=False)
    actual = runtime.run_training(
        migrated,
        {
            "model_kind": "causal_lm",
            "input_ids": ids,
            "inputs_embeds": None,
            "labels": ids,
            "training": True,
            "gradient_checkpointing": False,
            "grad_enabled": True,
            "use_cache": False,
            "logits_to_keep": 0,
        },
    )
    torch.testing.assert_close(actual["logits"], expected.logits)
    torch.testing.assert_close(actual["loss"], expected.loss)
    expected.loss.backward()
    actual["loss"].backward()
    torch.testing.assert_close(
        migrated.model.layers[0].attn.r_proj.weight.grad,
        reference.model.layers[0].attn.r_proj.weight.grad,
    )
