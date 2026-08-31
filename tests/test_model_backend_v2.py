from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from rwkv7_hf.cache_rwkv7 import RWKV7Cache
from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Linear, RWKV7Model
from rwkv7_hf.ops_rwkv7 import get_last_model_route, rwkv7_recurrent_reference


ROOT = Path(__file__).resolve().parents[1]


def test_training_forward_records_complete_reference_model_route(
    tiny_config, monkeypatch
):
    monkeypatch.setenv("RWKV7_BACKEND", "auto")
    monkeypatch.setenv("RWKV7_MODEL_KERNEL_IMPL", "auto")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "auto")
    model = RWKV7ForCausalLM(tiny_config).train()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    output = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
        logits_to_keep=0,
    )

    assert torch.isfinite(output.loss)
    assert get_last_model_route() == {
        "requested": "auto",
        "selected": "reference",
        "implementation": "torch-reference-model-v1",
        "reason": (
            "readable HF training loop owns structure; optional tensor "
            "leaves dispatch through one explicit execution context"
        ),
        "phase": "training",
    }


def _load_dense_backend(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)
    return importlib.import_module("rwkv7_kernels.model.dense")


def test_native_packer_recognizes_clean_linear_and_preserves_fp32_decay_bias(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    linear = importlib.import_module("rwkv7_kernels.nvidia.native_jit_linear")
    packing = importlib.import_module("rwkv7_kernels.nvidia.native_jit_packing")

    projection = RWKV7Linear(8, 4, bias=False).half()
    assert linear.dense_linear_module(projection)
    assert linear.graph_linear_operand(projection) is projection.weight

    model = RWKV7ForCausalLM(tiny_config).half().eval()
    for layer in model.model.layers:
        decay = layer.attn.w_lora.lora[2]
        decay.bias = torch.nn.Parameter(decay.bias.float())
    assert model.model.layers[0].attn.w_lora.lora[2].bias.dtype == torch.float32

    dense_packs, *_ = packing.extract_dense_packs(model, rkv_policy="linear")
    assert [pack[4] for pack in dense_packs] == [1, 0]
    assert dense_packs[0][26].dtype == torch.float32
    assert model.model.layers[0].attn.w_lora.lora[2].bias.dtype == torch.float32

    graph_packs, *_ = packing.extract_graph_packs(
        model,
        rkv_policy="linear",
        sparse_ffn_low_memory_pack_enabled=lambda: False,
        try_relayout_ffn_value_weight=lambda module: False,
        graph_linear_operand=linear.graph_linear_operand,
        graph_linear_is_dense=linear.graph_linear_is_dense,
    )
    assert [pack[4] for pack in graph_packs] == [1, 0]
    assert isinstance(graph_packs[0][24], torch.Tensor)
    assert isinstance(graph_packs[0][25], torch.Tensor)
    assert graph_packs[0][26].dtype == torch.float32
    assert model.model.layers[0].attn.w_lora.lora[2].bias.dtype == torch.float32

    structural = importlib.import_module("rwkv7_kernels.model.packing")
    structural_packs, *_ = structural.extract_dense_packs(model.model)
    assert [pack[5] for pack in structural_packs] == [1, 0]


def test_structural_dense_packs_follow_module_replacement_and_model_moves(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    structural = importlib.import_module("rwkv7_kernels.model.packing")
    model = RWKV7ForCausalLM(tiny_config).half().eval()

    first_packs, *_ = structural.extract_dense_packs(model.model)
    old_projection = model.model.layers[0].attn.r_proj
    old_weight = old_projection.weight
    assert first_packs[0][21] is old_weight

    replacement = RWKV7Linear(
        tiny_config.hidden_size,
        tiny_config.attention_hidden_size,
        bias=False,
    ).half()
    model.model.layers[0].attn.r_proj = replacement
    second_packs, *_ = structural.extract_dense_packs(model.model)
    assert second_packs is not first_packs
    assert second_packs[0][21] is replacement.weight
    assert second_packs[0][21] is not old_weight

    version = replacement.weight._version
    with torch.no_grad():
        replacement.weight.add_(1)
    third_packs, *_ = structural.extract_dense_packs(model.model)
    assert replacement.weight._version == version + 1
    assert third_packs is not second_packs
    assert third_packs[0][21] is replacement.weight

    # Later blocks use synthetic tensors for their Identity pre-norm. They must
    # be rebuilt on the model's current dtype/device rather than surviving from
    # a previous structural extraction.
    assert third_packs[1][6].dtype == torch.float16
    model.float()
    float_packs, *_ = structural.extract_dense_packs(model.model)
    assert float_packs[1][6].dtype == torch.float32
    model.to(device="meta")
    meta_packs, *_ = structural.extract_dense_packs(model.model)
    assert meta_packs[1][6].device.type == "meta"
    assert meta_packs[0][-1].device.type == "meta"


def test_native_lm_head_preserves_rank3_reference_linear_contract(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    linear = importlib.import_module("rwkv7_kernels.nvidia.native_jit_linear")

    projection = RWKV7Linear(
        tiny_config.hidden_size,
        tiny_config.vocab_size,
        bias=False,
    )
    calls = []
    original_forward = RWKV7Linear.forward

    def recorded_forward(self, value):
        calls.append(tuple(value.shape))
        return original_forward(self, value)

    recorded_forward._rwkv7_dense_linear_contract = True
    monkeypatch.setattr(RWKV7Linear, "forward", recorded_forward)

    rank3 = torch.randn(2, 3, tiny_config.hidden_size)
    rank2 = rank3.reshape(-1, tiny_config.hidden_size)
    torch.testing.assert_close(
        linear.linear_module(projection, rank3),
        projection(rank3),
    )
    assert calls == [tuple(rank3.shape), tuple(rank3.shape)]

    calls.clear()
    torch.testing.assert_close(
        linear.linear_module(projection, rank2),
        F.linear(rank2, projection.weight, projection.bias),
    )
    assert calls == []


def test_native_whole_model_operands_preserve_linear_subclass_forward(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    linear = importlib.import_module("rwkv7_kernels.nvidia.native_jit_linear")
    packing = importlib.import_module("rwkv7_kernels.nvidia.native_jit_packing")
    native_jit = importlib.import_module("rwkv7_kernels.nvidia.native_jit")
    graph_runtime = importlib.import_module("rwkv7_kernels.nvidia.native_graph_runtime")

    class OffsetLinear(torch.nn.Linear):
        def forward(self, value):
            return super().forward(value) + 0.75

    model = RWKV7ForCausalLM(tiny_config).eval()
    projection = OffsetLinear(
        tiny_config.hidden_size,
        tiny_config.attention_hidden_size,
        bias=False,
    )
    model.model.layers[0].attn.r_proj = projection
    assert not linear.dense_linear_module(projection)
    assert linear.graph_linear_operand(projection) is projection

    graph_packs, *_ = packing.extract_graph_packs(
        model,
        rkv_policy="linear",
        sparse_ffn_low_memory_pack_enabled=lambda: False,
        try_relayout_ffn_value_weight=lambda module: False,
        graph_linear_operand=linear.graph_linear_operand,
        graph_linear_is_dense=linear.graph_linear_is_dense,
    )
    assert graph_packs[0][20] is projection
    hidden = torch.randn(2, tiny_config.hidden_size)
    torch.testing.assert_close(
        native_jit._graph_linear_call(hidden, graph_packs[0][20]),
        projection(hidden),
    )

    head = OffsetLinear(
        tiny_config.hidden_size,
        tiny_config.vocab_size,
        bias=False,
    )
    model.set_output_embeddings(head)
    expected = head(hidden)
    torch.testing.assert_close(native_jit._lm_head(model, hidden), expected)
    destination = torch.empty_like(expected)
    graph_runtime._head_linear_into(head, hidden, destination)
    torch.testing.assert_close(destination, expected)

    # A class-level monkeypatch of the adapter's own Linear must also revoke
    # the marker.  Checking only the class name/boolean marker would silently
    # bypass this forward on the whole-model route.
    original_forward = RWKV7Linear.forward

    def overridden_rwkv7_forward(self, value):
        return original_forward(self, value) + 0.25

    monkeypatch.setattr(RWKV7Linear, "forward", overridden_rwkv7_forward)
    internal_projection = model.model.layers[1].attn.r_proj
    assert not linear.dense_linear_module(internal_projection)
    assert linear.graph_linear_operand(internal_projection) is internal_projection
    torch.testing.assert_close(
        native_jit._graph_linear_call(hidden, internal_projection),
        internal_projection(hidden),
    )

    # The tensor-only dense-v2 diagnostic cannot call a custom projection and
    # therefore must decline rather than silently pack its raw weight.
    dispatcher = importlib.import_module("rwkv7_kernels.model_dispatcher")
    assert any(
        not linear.dense_linear_module(module)
        for module in dispatcher._dense_model_linears(model.model)
    )

    class FakeCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda")

    fake_cuda_hidden = torch.zeros(
        1, 2, tiny_config.hidden_size, dtype=torch.float16
    ).as_subclass(FakeCudaTensor)
    support = dispatcher._probe_dense(
        model.model,
        {
            "model_kind": "base",
            "hidden_states": fake_cuda_hidden,
            "training": False,
            "grad_enabled": False,
        },
    )
    assert support["supported"] is False
    assert "custom linear forward" in support["reason"]


def test_native_decay_projection_adds_w0_only_after_fp32_promotion(
    monkeypatch,
):
    _load_dense_backend(monkeypatch)
    native_jit = importlib.import_module("rwkv7_kernels.nvidia.native_jit")
    decode = importlib.import_module("rwkv7_kernels.nvidia.native_jit_decode")
    prefill = importlib.import_module("rwkv7_kernels.nvidia.native_jit_prefill")
    prefill.bind_runtime(vars(native_jit))

    x = torch.tensor([[1.125, -0.375, 0.75, -1.5]], dtype=torch.float16)
    down = torch.tensor(
        [[0.25, -0.5, 0.75, 0.125], [-0.625, 0.5, 0.25, -0.375]],
        dtype=torch.float16,
    )
    up = torch.tensor(
        [[0.25, -0.5], [0.75, 0.125], [-0.625, 0.5]],
        dtype=torch.float16,
    )
    w0 = torch.tensor([1.0003, -0.5002, 0.1251], dtype=torch.float32)
    expected = F.linear(torch.tanh(F.linear(x, down)), up).float() + w0

    decode_actual = decode._native_decay_projection(x, down, up, w0)
    prefill_actual = prefill._native_prefill_decay_projection(x, down, up, w0)
    assert decode_actual.dtype == torch.float32
    assert prefill_actual.dtype == torch.float32
    torch.testing.assert_close(decode_actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prefill_actual, expected, rtol=0, atol=0)


def test_native_prefill_dense_linear_uses_clean_fixed_row_contract(monkeypatch):
    _load_dense_backend(monkeypatch)
    native_jit = importlib.import_module("rwkv7_kernels.nvidia.native_jit")
    prefill = importlib.import_module("rwkv7_kernels.nvidia.native_jit_prefill")
    prefill.bind_runtime(vars(native_jit))

    torch.manual_seed(211)
    value = torch.randn(1, 17, 8, dtype=torch.float16)
    weight = torch.randn(11, 8, dtype=torch.float16)
    expected = RWKV7Linear(8, 11, bias=False).half()
    with torch.no_grad():
        expected.weight.copy_(weight)

    projected = prefill._native_prefill_linear(value, weight)
    repeated = prefill._native_prefill_linear(value.repeat(8, 1, 1), weight)
    torch.testing.assert_close(projected, expected(value), rtol=0, atol=0)
    torch.testing.assert_close(projected, repeated[:1], rtol=0, atol=0)


def test_native_prefill_scan_uses_batch_invariant_row_contract(monkeypatch):
    _load_dense_backend(monkeypatch)
    native_jit = importlib.import_module("rwkv7_kernels.nvidia.native_jit")
    prefill = importlib.import_module("rwkv7_kernels.nvidia.native_jit_prefill")
    prefill.bind_runtime(vars(native_jit))

    monkeypatch.setattr(prefill, "_native_prefill_fused_scan_enabled", lambda *_: False)
    monkeypatch.setattr(prefill, "_native_prefill_dplr_scan_enabled", lambda: False)
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_REFERENCE_BATCH_SCAN", "1")

    torch.manual_seed(223)
    batch, sequence, heads, head_size = 3, 5, 2, 4
    shape = (batch, sequence, heads * head_size)
    r = torch.randn(shape)
    w = torch.sigmoid(torch.randn(shape))
    k = torch.randn(shape)
    v = torch.randn(shape)
    kk = torch.randn(shape)
    a = torch.randn(shape)
    state = torch.randn(batch, heads, head_size, head_size)

    actual_output, actual_state = prefill._native_prefill_scan(
        r,
        w,
        k,
        v,
        kk,
        a,
        state,
        batch,
        sequence,
        heads,
        head_size,
        use_self_chunk=False,
    )
    row_results = [
        prefill._native_prefill_scan(
            r[index : index + 1],
            w[index : index + 1],
            k[index : index + 1],
            v[index : index + 1],
            kk[index : index + 1],
            a[index : index + 1],
            state[index : index + 1],
            1,
            sequence,
            heads,
            head_size,
            use_self_chunk=False,
        )
        for index in range(batch)
    ]
    expected_output = torch.cat([result[0] for result in row_results], dim=0)
    expected_state = torch.cat([result[1] for result in row_results], dim=0)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


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
                "past_key_values": RWKV7Cache(num_layers=tiny_config.num_hidden_layers),
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
                "past_key_values": RWKV7Cache(num_layers=tiny_config.num_hidden_layers),
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
                "past_key_values": RWKV7Cache(num_layers=tiny_config.num_hidden_layers),
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
    # A second decode token must retain the exact canonical views.  The
    # adapter boundary is not allowed to clone/rebind the cache per token.
    runner.copy_from_cache(cache)
    runner.bind_cache(cache)
    assert runner.copy_from_cache_fast_skips == 1
    assert runner.bind_cache_fast_skips == 1

    runner.state[0].add_(1)
    torch.testing.assert_close(cache.recurrent_state[0], canonical + 1)
    runner.detach_bound_cache()
    detached = cache.recurrent_state[0].clone()
    runner.state[0].add_(1)
    torch.testing.assert_close(cache.recurrent_state[0], detached)


def test_private_training_diagnostic_uses_direct_layer_loop_without_monkeypatch(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    runtime = importlib.import_module("rwkv7_kernels.nvidia.training_runtime")
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    monkeypatch.setattr(
        train_temp, "load_training_runtime_cuda_extensions", lambda: None
    )

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
            hidden,
            state,
            shift,
            v_first,
            mask,
            mask_fully_active=True,
        )
        return output, v_first

    monkeypatch.setattr(train_temp, "_train_temp_attention_forward", attention_forward)

    torch.manual_seed(113)
    reference = RWKV7ForCausalLM(tiny_config).train()
    migrated = RWKV7ForCausalLM(tiny_config).train()
    migrated.load_state_dict(reference.state_dict())
    ids = torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]])
    labels = ids.clone()
    labels[0, 2] = -100

    expected = reference(input_ids=ids, labels=labels, use_cache=False)
    actual = runtime._run_training_diagnostic(
        migrated,
        {
            "model_kind": "causal_lm",
            "input_ids": ids,
            "inputs_embeds": None,
            "labels": labels,
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
    source = Path(runtime.__file__).read_text()
    assert "ffn_output = channel_mix(layer.ffn, ffn_input)" in source
    assert "full_logits = module_linear(owner.lm_head, hidden_states)" in source
    assert "layer.ffn(" not in source
    assert "train_temp._CMix.apply(" not in source


def test_training_runtime_loader_compiles_only_accepted_leaf_set(monkeypatch):
    _load_dense_backend(monkeypatch)
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    calls = []
    monkeypatch.setattr(
        train_temp,
        "load_mix6_training_cuda_extension",
        lambda **kwargs: calls.append(("mix6", kwargs)),
    )
    monkeypatch.setattr(
        train_temp,
        "load_recurrent_training_cuda_extension",
        lambda **kwargs: calls.append(("recurrent", kwargs)),
    )

    train_temp.load_training_runtime_cuda_extensions(verbose=True)

    assert calls == [
        ("mix6", {"verbose": True}),
        ("recurrent", {"verbose": True}),
    ]


def test_recurrent_training_loader_uses_isolated_cuda_build_environment(monkeypatch):
    """A thin venv must expose base Ninja/NVCC and the active GPU arch."""

    _load_dense_backend(monkeypatch)
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    state = {"active": False, "built": False}
    calls = []

    @contextmanager
    def build_environment(*, arch_list):
        calls.append(("environment", arch_list))
        state["active"] = True
        try:
            yield None
        finally:
            state["active"] = False

    def build_recurrent(_cpp_extension, cuda_home, *, verbose):
        assert state["active"]
        calls.append(("build", cuda_home, verbose))
        state["built"] = True

    monkeypatch.setattr(train_temp, "_RECURRENT_LOADED", False)
    monkeypatch.setattr(train_temp, "_RECURRENT_LOAD_ERROR", None)
    monkeypatch.setattr(train_temp, "_validate_runtime", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(
        train_temp,
        "cuda_extension_build_environment",
        build_environment,
    )
    monkeypatch.setattr(
        train_temp,
        "_resolve_cuda_home",
        lambda _cpp_extension: Path("/cuda-13.0"),
    )
    monkeypatch.setattr(
        train_temp,
        "_op_registered",
        lambda namespace: namespace == "rwkv7_clampw_v3" and state["built"],
    )
    monkeypatch.setattr(train_temp, "_build_recurrent_operator", build_recurrent)

    train_temp.load_recurrent_training_cuda_extension(verbose=True)

    assert calls == [
        ("environment", "8.9"),
        ("build", Path("/cuda-13.0"), True),
    ]
    assert train_temp._RECURRENT_LOADED is True


def test_native_training_causal_loss_avoids_shifted_logits_copy(monkeypatch):
    _load_dense_backend(monkeypatch)
    runtime = importlib.import_module("rwkv7_kernels.nvidia.training_runtime")
    torch.manual_seed(119)
    expected_logits = torch.randn(3, 7, 19, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 19, (3, 7))
    labels[0, 2] = -100
    labels[2, 5:] = -100

    expected = torch.nn.functional.cross_entropy(
        expected_logits[:, :-1].contiguous().reshape(-1, 19),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=-100,
    )
    actual = runtime.causal_cross_entropy(actual_logits, labels)
    torch.testing.assert_close(actual, expected)

    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)

    ignored_logits = torch.randn(2, 4, 19, requires_grad=True)
    ignored = runtime.causal_cross_entropy(
        ignored_logits,
        torch.full((2, 4), -100, dtype=torch.long),
    )
    assert ignored.item() == 0.0
    ignored.backward()
    assert ignored_logits.grad is not None
    assert torch.count_nonzero(ignored_logits.grad) == 0

    single_logits = torch.randn(2, 1, 19, requires_grad=True)
    single = runtime.causal_cross_entropy(
        single_logits,
        torch.randint(0, 19, (2, 1)),
    )
    assert single.item() == 0.0


def test_native_training_math_matches_clean_fixed_row_contract(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    training_math = importlib.import_module("rwkv7_kernels.nvidia.training_math")

    torch.manual_seed(127)
    model = RWKV7ForCausalLM(tiny_config).train()
    value = torch.randn(3, 17, tiny_config.hidden_size)
    projection = model.model.layers[0].attn.r_proj

    expected = projection(value)
    actual = training_math.fixed_row_linear(value, projection.weight, projection.bias)
    repeated = training_math.fixed_row_linear(
        value.repeat(3, 1, 1), projection.weight, projection.bias
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, repeated[:3], rtol=0, atol=0)


def test_native_training_linear_flattens_only_multiple_reference_tiles(monkeypatch):
    _load_dense_backend(monkeypatch)
    training_math = importlib.import_module("rwkv7_kernels.nvidia.training_math")
    original_linear = training_math.F.linear
    calls = []

    def record_linear(value, weight, bias=None):
        calls.append(tuple(value.shape))
        return original_linear(value, weight, bias)

    monkeypatch.setattr(training_math.F, "linear", record_linear)
    weight = torch.randn(11, 7)

    one_tile = torch.randn(4, 16, 7)
    one_output = training_math.training_linear(one_tile, weight)
    assert one_output.shape == (4, 16, 11)
    assert calls == [(training_math.REFERENCE_LINEAR_ROWS, 7)]

    calls.clear()
    four_tiles = torch.randn(4, 128, 7, requires_grad=True)
    four_output = training_math.training_linear(four_tiles, weight)
    assert four_output.shape == (4, 128, 11)
    assert calls == [(512, 7)]
    four_output.square().mean().backward()
    assert four_tiles.grad is not None
    assert torch.isfinite(four_tiles.grad).all()


def test_native_training_linear_bounds_four_times_wide_ffn_rows(monkeypatch):
    _load_dense_backend(monkeypatch)
    training_math = importlib.import_module("rwkv7_kernels.nvidia.training_math")
    original_linear = training_math.F.linear
    calls = []

    def record_linear(value, weight, bias=None):
        calls.append(tuple(value.shape))
        return original_linear(value, weight, bias)

    monkeypatch.setattr(training_math.F, "linear", record_linear)
    value = torch.randn(4, 128, 8, requires_grad=True)
    weight = torch.randn(32, 8, requires_grad=True)

    output = training_math.training_linear(value, weight)

    assert output.shape == (4, 128, 32)
    assert calls == [(4, 80, 8), (4, 48, 8)]
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_native_training_channel_mix_does_not_reenter_linear_dispatch(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    training_math = importlib.import_module("rwkv7_kernels.nvidia.training_math")

    torch.manual_seed(131)
    channel = RWKV7ForCausalLM(tiny_config).train().model.layers[0].ffn
    value = torch.randn(2, 9, tiny_config.hidden_size)
    mask = torch.ones(2, 9, dtype=torch.bool)
    shift = torch.zeros(2, tiny_config.hidden_size)
    expected, _ = channel(value, shift, mask, mask_fully_active=True)

    def reject_nested_dispatch(*_args, **_kwargs):
        raise AssertionError("native training recursively called RWKV7Linear.forward")

    monkeypatch.setattr(RWKV7Linear, "forward", reject_nested_dispatch)
    actual = training_math.channel_mix(channel, value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_train_temp_include_paths_merge_partial_overlay_and_pip_headers(
    tmp_path, monkeypatch
):
    _load_dense_backend(monkeypatch)
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    overlay = tmp_path / "cuda"
    overlay_include = overlay / "include"
    overlay_include.mkdir(parents=True)
    (overlay_include / "cuda_runtime.h").write_text("// core CUDA header\n")

    site_packages = tmp_path / "site-packages"
    torch_root = site_packages / "torch"
    torch_root.mkdir(parents=True)
    torch_init = torch_root / "__init__.py"
    torch_init.write_text("# synthetic torch package root\n")
    cusparse_include = site_packages / "nvidia" / "cusparse" / "include"
    cusparse_include.mkdir(parents=True)
    (cusparse_include / "cusparse.h").write_text("// split CUDA header\n")
    monkeypatch.setattr(train_temp.torch, "__file__", str(torch_init))

    include_paths = train_temp._cuda_include_paths(overlay)

    assert include_paths == [str(overlay_include), str(cusparse_include)]


def test_train_temp_decay_operand_privately_adds_fp32_public_bias(
    tiny_config, monkeypatch
):
    _load_dense_backend(monkeypatch)
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    attention = RWKV7ForCausalLM(tiny_config).model.layers[0].attn.bfloat16()
    projection = attention.w_lora.lora[2]
    projection.bias = torch.nn.Parameter(projection.bias.float())
    xw = torch.randn(2, 4, tiny_config.hidden_size, dtype=torch.bfloat16)

    raw_decay = attention.w_lora.project_without_bias(xw, torch.tanh)
    actual = torch.exp(
        -0.6065306597 * torch.sigmoid(raw_decay.float() + projection.bias.float())
    )

    assert actual.dtype == torch.float32
    assert projection.bias.dtype == torch.float32
    actual.float().sum().backward()
    assert projection.bias.grad is not None
    assert projection.weight.grad is not None
    # Keep the adapted runtime source contract visible to the unit suite.
    source = Path(train_temp.__file__).read_text()
    assert "raw_decay.float() + decay_bias.float()" in source
    assert "decay.dtype != torch.float32" in source
    assert '"--fmad=false"' in source
    assert "torch.ops.rwkv7_clampw_v3.backward" in source
    assert "_recurrent_decay_reference(" in source
    assert "a = torch.sigmoid(low_rank_projection(self.a_lora, xa))" in source
    assert "normalized_key = F.normalize(" in source
    assert "value_mix = torch.sigmoid(low_rank_projection(self.v_lora, xv))" in source


def test_recurrent_training_replay_matches_reference_full_gradient(monkeypatch):
    _load_dense_backend(monkeypatch)
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    torch.manual_seed(73)
    shape = (2, 3, 2, 4)
    recurrent_inputs = [(torch.randn(shape) * 0.1).requires_grad_() for _ in range(6)]
    recurrent_inputs[1] = torch.rand(shape).requires_grad_()
    initial_state = (torch.randn(2, 2, 4, 4) * 0.01).requires_grad_()

    reference = rwkv7_recurrent_reference(*recurrent_inputs, initial_state)
    replay = train_temp._recurrent_decay_reference(*recurrent_inputs, initial_state)
    for candidate, expected in zip(replay, reference, strict=True):
        torch.testing.assert_close(candidate, expected)

    reference_loss = sum(value.square().mean() for value in reference)
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (*recurrent_inputs, initial_state),
        retain_graph=True,
    )
    replay_loss = sum(value.square().mean() for value in replay)
    replay_gradients = torch.autograd.grad(
        replay_loss, (*recurrent_inputs, initial_state)
    )
    for candidate, expected in zip(replay_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(candidate, expected)


def test_recurrent_training_mask_compaction_preserves_padding_and_gradients(
    monkeypatch,
):
    _load_dense_backend(monkeypatch)
    training = importlib.import_module("rwkv7_kernels.recurrent.training_factorized")
    torch.manual_seed(79)
    shape = (2, 5, 2, 4)
    base = [torch.randn(shape, dtype=torch.float64) * 0.1 for _ in range(6)]
    base[1] = torch.rand(shape, dtype=torch.float64)
    state = torch.zeros(2, 2, 4, 4, dtype=torch.float64)
    mask = torch.tensor(
        [[False, False, True, True, True], [False, False, False, False, False]]
    )
    runner_calls = []

    def packed_reference(*args):
        runner_calls.append(tuple(args[0].shape))
        return rwkv7_recurrent_reference(*args)

    def collect(compacted: bool):
        values = [value.detach().clone().requires_grad_() for value in base]
        initial_state = state.detach().clone().requires_grad_()
        if compacted:
            output, final_state = training._run_masked_training(
                values,
                initial_state,
                mask,
                runner=packed_reference,
            )
        else:
            output, final_state = rwkv7_recurrent_reference(
                *values,
                initial_state,
                mask,
            )
        loss = output.square().mean() + final_state.square().mean()
        gradients = torch.autograd.grad(loss, (*values, initial_state))
        return output, final_state, gradients

    compacted = collect(True)
    reference = collect(False)
    for candidate, expected in zip(compacted[:2], reference[:2], strict=True):
        torch.testing.assert_close(candidate, expected)
    for candidate, expected in zip(compacted[2], reference[2], strict=True):
        torch.testing.assert_close(candidate, expected)
    assert runner_calls == [(2, 16, 2, 4)]


def test_recurrent_training_dense_hints_avoid_mask_scalar_sync(monkeypatch):
    _load_dense_backend(monkeypatch)
    training = importlib.import_module("rwkv7_kernels.recurrent.training_factorized")
    shape = (2, 16, 2, 4)
    recurrent_inputs = [torch.randn(shape) for _ in range(6)]
    state = torch.zeros(2, 2, 4, 4)
    mask = torch.ones(2, 16, dtype=torch.bool)
    runner_calls = []

    def runner(*args):
        runner_calls.append(tuple(args[0].shape))
        return args[3], args[-1]

    def reject_scalar_sync(*_args, **_kwargs):
        raise AssertionError("dense request must not reduce the device mask")

    monkeypatch.setattr(torch.Tensor, "all", reject_scalar_sync)
    output, final_state = training._run_masked_training(
        recurrent_inputs,
        state,
        mask,
        runner=runner,
        fully_active=True,
        token_aligned=True,
    )

    assert runner_calls == [(2, 16, 2, 4)]
    assert output is recurrent_inputs[3]
    assert final_state is state


def test_train_temp_mix6_backward_matches_canonical_token_mix():
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    torch.manual_seed(127)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mixes = [torch.randn(8, requires_grad=True) for _ in range(6)]
    outputs = []
    shifted = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    for mix in mixes:
        outputs.append(x + (shifted - x) * mix.view(1, 1, -1))
    output_grads = [torch.randn_like(output) for output in outputs]
    torch.autograd.backward(outputs, output_grads)
    expected = (x.grad.clone(), *(mix.grad.clone() for mix in mixes))

    class Context:
        saved_tensors = (x.detach(), *(mix.detach() for mix in mixes))

    actual = train_temp._Mix6.backward(Context(), *output_grads)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference)


def test_train_temp_mix6_large_backward_calls_native_operator(monkeypatch):
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    torch.manual_seed(131)
    x = torch.randn(4, 8, 8).transpose(1, 2)
    mixes = [torch.randn(16)[::2] for _ in range(6)]
    output_grads = [torch.randn(4, 8, 8).transpose(1, 2) for _ in range(6)]
    native_results = (torch.randn_like(x), *(torch.randn_like(mix) for mix in mixes))
    calls = []

    def native_backward(*args):
        calls.append(args)
        return native_results

    monkeypatch.setattr(
        torch.ops.rwkv7_tmix_mix6_bf16_v5,
        "backward",
        native_backward,
        raising=False,
    )

    class Context:
        saved_tensors = (x, *mixes)

    with torch.no_grad():
        actual = train_temp._Mix6.backward(Context(), *output_grads)

    assert len(calls) == 1
    assert all(
        candidate is expected for candidate, expected in zip(actual, native_results)
    )
    assert all(value.is_contiguous() for value in calls[0])
    for candidate, expected in zip(calls[0][:6], output_grads, strict=True):
        torch.testing.assert_close(candidate, expected)
    for candidate, expected in zip(calls[0][6:], (x, *mixes), strict=True):
        torch.testing.assert_close(candidate, expected)


def test_train_temp_mix6_small_backward_replays_canonical_math(monkeypatch):
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    torch.manual_seed(133)
    x = torch.randn(1, 16, 8, requires_grad=True)
    mixes = [torch.randn(8, requires_grad=True) for _ in range(6)]
    output_grads = [torch.randn_like(x) for _ in range(6)]

    def reject_native_backward(*_args):
        raise AssertionError("small Mix6 backward must use canonical replay")

    monkeypatch.setattr(
        torch.ops.rwkv7_tmix_mix6_bf16_v5,
        "backward",
        reject_native_backward,
        raising=False,
    )

    shifted = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    canonical_outputs = tuple(x + (shifted - x) * mix.view(1, 1, -1) for mix in mixes)
    expected = torch.autograd.grad(
        canonical_outputs,
        (x, *mixes),
        output_grads,
    )

    class Context:
        saved_tensors = (x.detach(), *(mix.detach() for mix in mixes))

    with torch.no_grad():
        actual = train_temp._Mix6.backward(Context(), *output_grads)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference)


def test_train_temp_mix6_double_backward_retains_canonical_graph(monkeypatch):
    train_temp = importlib.import_module("rwkv7_kernels.nvidia.official_training_cuda")
    torch.manual_seed(137)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mixes = [torch.randn(8, requires_grad=True) for _ in range(6)]
    output_grads = [torch.randn_like(x) for _ in range(6)]

    def canonical_forward(value, *parameters):
        shifted = torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)
        delta = shifted - value
        return tuple(
            value + delta * parameter.view(1, 1, -1) for parameter in parameters
        )

    def reject_native_backward(*_args):
        raise AssertionError("double backward must use the canonical replay")

    monkeypatch.setattr(
        torch.ops.rwkv7_tmix_mix6_bf16_v5,
        "forward",
        canonical_forward,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_tmix_mix6_bf16_v5,
        "backward",
        reject_native_backward,
        raising=False,
    )

    outputs = train_temp._Mix6.apply(x, *mixes)
    first_gradients = torch.autograd.grad(
        outputs,
        (x, *mixes),
        output_grads,
        create_graph=True,
    )
    assert all(gradient.requires_grad for gradient in first_gradients)
    second_gradients = torch.autograd.grad(
        sum(gradient.sum() for gradient in first_gradients),
        (x, *mixes),
    )
    assert all(torch.isfinite(gradient).all() for gradient in second_gradients)


def test_train_temp_mix6_cuda_backward_has_fixed_reduction_order():
    cuda_source = (
        ROOT
        / "kernels"
        / "rwkv7_kernels"
        / "nvidia"
        / "csrc"
        / "training"
        / "rwkv_lm"
        / "rwkv7_tmix_mix6_bf16_v5.cu"
    ).read_text()
    cpp_source = (
        ROOT
        / "kernels"
        / "rwkv7_kernels"
        / "nvidia"
        / "csrc"
        / "training"
        / "rwkv_lm"
        / "rwkv7_tmix_mix6_bf16_v5.cpp"
    ).read_text()

    assert "tmix_mix6_backward_deterministic_kernel_v5" in cuda_source
    assert "for (int64_t bt = 0; bt < bt_size; ++bt)" in cuda_source
    assert "atomicAdd" not in cuda_source
    assert "blockIdx.y" not in cuda_source
    assert "torch::zeros" not in cuda_source
    assert cuda_source.count("at::cuda::getCurrentCUDAStream(x.get_device())") == 2
    assert cuda_source.count("C10_CUDA_KERNEL_LAUNCH_CHECK()") == 2
    assert "grad_x_r.data_ptr<at::BFloat16>()" in cuda_source
    assert cpp_source.count("c10::cuda::CUDAGuard device_guard(x.device())") == 2
    assert cpp_source.count("check_same_device(*item.first, x, item.second)") == 3


def test_train_temp_clampw_uses_checked_tensor_device_and_current_stream():
    root = (
        ROOT / "kernels" / "rwkv7_kernels" / "nvidia" / "csrc" / "training" / "rwkv_lm"
    )
    cpp_source = (root / "rwkv7_clampw_v3.cpp").read_text()
    cuda_source = (root / "rwkv7_clampw_v3_for_h100.cu").read_text()

    assert "check_common_inputs(r, decay, k, v, a, b, s, sa)" in cpp_source
    assert "value.device() == reference.device()" in cpp_source
    assert cpp_source.count("c10::cuda::CUDAGuard device_guard(r.device())") == 2
    assert cpp_source.count("at::cuda::getCurrentCUDAStream(r.get_device())") == 2
    assert cuda_source.count(",0,stream>>>") == 2
    assert cuda_source.count("C10_CUDA_KERNEL_LAUNCH_CHECK()") == 2
