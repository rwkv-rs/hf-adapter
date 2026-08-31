from __future__ import annotations

import torch
import pytest

from rwkv7_hf.cache_rwkv7 import RWKV7Cache
from rwkv7_hf.modeling_rwkv7 import (
    RWKV7Block,
    RWKV7ChannelMix,
    RWKV7ForCausalLM,
    RWKV7Model,
    RWKV7TimeMix,
)


def test_public_structure_and_forward(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).eval()
    assert isinstance(model.model, RWKV7Model)
    assert isinstance(model.model.layers[0], RWKV7Block)
    assert isinstance(model.model.layers[0].attn, RWKV7TimeMix)
    assert isinstance(model.model.layers[0].ffn, RWKV7ChannelMix)
    assert isinstance(model.model.layers[0].pre_norm, torch.nn.LayerNorm)
    assert all(
        isinstance(layer.pre_norm, torch.nn.Identity)
        for layer in model.model.layers[1:]
    )

    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            use_cache=True,
            output_hidden_states=True,
        )
    assert output.logits.shape == (2, 3, tiny_config.vocab_size)
    assert isinstance(output.past_key_values, RWKV7Cache)
    assert output.past_key_values.get_seq_length() == 3
    assert len(output.hidden_states) == tiny_config.num_hidden_layers + 1
    for state in output.past_key_values.recurrent_state:
        assert state.shape == (2, 2, 8, 8)


def test_default_config_has_generation_token_ids():
    from rwkv7_hf.configuration_rwkv7 import RWKV7Config

    config = RWKV7Config(
        vocab_size=64,
        hidden_size=32,
        attention_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_heads=1,
        head_dim=32,
        value_dim=[32, 32],
    )
    assert config.pad_token_id == 0
    assert config.eos_token_id == 0
    assert config.bos_token_id == 1
    RWKV7ForCausalLM(config)


def test_cache_decode_matches_full_forward(tiny_config):
    torch.manual_seed(7)
    model = RWKV7ForCausalLM(tiny_config).eval()
    ids = torch.randint(1, tiny_config.vocab_size, (2, 7))
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=False).logits
        cache = None
        pieces = []
        for token_idx in range(ids.shape[1]):
            output = model(
                input_ids=ids[:, token_idx : token_idx + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            pieces.append(output.logits)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
    assert cache.get_seq_length() == ids.shape[1]


def test_fp16_prefix_is_invariant_to_batch_regrouping(tiny_config):
    """Evaluation batching must not change an existing sequence's scores."""

    torch.manual_seed(19)
    model = RWKV7ForCausalLM(tiny_config)
    # The tiny constructor deliberately leaves checkpoint-specific mixing
    # vectors at zero; populate every tensor as a converted checkpoint would.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    model = model.half().eval()
    sample = torch.tensor([[7, 11, 13, 17, 19]])
    regrouped = torch.randint(1, tiny_config.vocab_size, (8, 11))
    regrouped[5, : sample.shape[1]] = sample[0]
    # lm_eval right-pads requests internally and does not currently forward an
    # attention mask through HFLM._model_call. Later positions are irrelevant
    # to the causal prefix, but their presence changes the outer GEMM shape in
    # an ordinary batched implementation.
    regrouped[5, sample.shape[1] :] = 0

    with torch.inference_mode():
        isolated = model(input_ids=sample, use_cache=False).logits
        batched = model(input_ids=regrouped, use_cache=False).logits

    torch.testing.assert_close(
        batched[5, : sample.shape[1]], isolated[0], rtol=0, atol=0
    )


def test_left_and_right_padding_do_not_update_state(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).eval()
    core = torch.tensor([[11, 12, 13]])
    with torch.inference_mode():
        plain = model(input_ids=core, use_cache=True)
        left = model(
            input_ids=torch.tensor([[0, 0, 11, 12, 13]]),
            attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
            use_cache=True,
        )
        right = model(
            input_ids=torch.tensor([[11, 12, 13, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
            use_cache=True,
        )
    torch.testing.assert_close(left.logits[:, 2:], plain.logits)
    torch.testing.assert_close(right.logits[:, :3], plain.logits)
    for expected, left_state, right_state in zip(
        plain.past_key_values.recurrent_state,
        left.past_key_values.recurrent_state,
        right.past_key_values.recurrent_state,
    ):
        torch.testing.assert_close(left_state, expected)
        torch.testing.assert_close(right_state, expected)


def _parameter_gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_gradients_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if expected[name] is None:
            assert actual[name] is None, name
        else:
            assert actual[name] is not None, name
            torch.testing.assert_close(actual[name], expected[name])


def _assert_caches_close(actual, expected, *, compare_sequence_length=True):
    if compare_sequence_length:
        assert actual.get_seq_length() == expected.get_seq_length()
    for field in ("recurrent_state", "attention_shift", "ffn_shift"):
        for actual_state, expected_state in zip(
            getattr(actual, field),
            getattr(expected, field),
        ):
            torch.testing.assert_close(actual_state, expected_state)


def test_missing_and_all_one_masks_have_identical_forward_and_gradients(tiny_config):
    torch.manual_seed(23)
    model = RWKV7ForCausalLM(tiny_config).train()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    without_mask = model(input_ids=ids, labels=ids, use_cache=False)
    without_mask.loss.backward()
    expected_gradients = _parameter_gradients(model)

    model.zero_grad(set_to_none=True)
    with_ones = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids,
        use_cache=False,
    )
    with_ones.loss.backward()

    torch.testing.assert_close(with_ones.logits, without_mask.logits)
    torch.testing.assert_close(with_ones.loss, without_mask.loss)
    _assert_gradients_close(_parameter_gradients(model), expected_gradients)

    model.eval()
    with torch.inference_mode():
        without_mask_cache = model(input_ids=ids, use_cache=True)
        with_ones_cache = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
        )
    torch.testing.assert_close(with_ones_cache.logits, without_mask_cache.logits)
    _assert_caches_close(
        with_ones_cache.past_key_values,
        without_mask_cache.past_key_values,
    )


@pytest.mark.parametrize(
    ("padded_ids", "mask", "valid_slice"),
    [
        (
            torch.tensor([[0, 0, 11, 12, 13]]),
            torch.tensor([[0, 0, 1, 1, 1]]),
            slice(2, None),
        ),
        (
            torch.tensor([[11, 12, 13, 0, 0]]),
            torch.tensor([[1, 1, 1, 0, 0]]),
            slice(None, 3),
        ),
    ],
    ids=("left", "right"),
)
def test_padding_preserves_selected_forward_gradients_and_cache(
    tiny_config,
    padded_ids,
    mask,
    valid_slice,
):
    torch.manual_seed(29)
    model = RWKV7ForCausalLM(tiny_config).eval()
    core = torch.tensor([[11, 12, 13]])

    plain = model(input_ids=core, use_cache=True)
    plain.logits.square().sum().backward()
    expected_gradients = _parameter_gradients(model)

    model.zero_grad(set_to_none=True)
    padded = model(input_ids=padded_ids, attention_mask=mask, use_cache=True)
    padded.logits[:, valid_slice].square().sum().backward()

    torch.testing.assert_close(padded.logits[:, valid_slice], plain.logits)
    _assert_gradients_close(_parameter_gradients(model), expected_gradients)
    # Cache sequence length records positions consumed, including padding; the
    # recurrent and shift tensors are the state whose masked semantics match.
    _assert_caches_close(
        padded.past_key_values,
        plain.past_key_values,
        compare_sequence_length=False,
    )


def test_loss_gradient_inputs_embeds_and_unsupported_attentions(tiny_config):
    torch.manual_seed(11)
    model = RWKV7ForCausalLM(tiny_config).train()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    labels = ids.clone()
    labels[1, -1] = -100
    output = model(input_ids=ids, labels=labels, use_cache=True)
    assert output.past_key_values is None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradient = model.model.layers[0].attn.r_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0

    model.eval()
    embeds = model.get_input_embeddings()(ids)
    with torch.inference_mode():
        from_ids = model(input_ids=ids, use_cache=False).logits
        from_embeds = model(inputs_embeds=embeds, use_cache=False).logits
    torch.testing.assert_close(from_ids, from_embeds)
    with pytest.raises(NotImplementedError):
        model(input_ids=ids, output_attentions=True)


def test_causal_loss_shifts_targets_without_copying_logits():
    from rwkv7_hf.modeling_rwkv7 import _causal_language_model_loss

    torch.manual_seed(13)
    expected_logits = torch.randn(3, 7, 23, requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 23, (3, 7))
    labels[0, 2] = -100
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1],
        ]
    )
    expected_labels = labels[:, 1:].contiguous().masked_fill(~mask[:, 1:].bool(), -100)
    expected = torch.nn.functional.cross_entropy(
        expected_logits[:, :-1].contiguous().reshape(-1, 23),
        expected_labels.reshape(-1),
        ignore_index=-100,
    )
    actual = _causal_language_model_loss(actual_logits, labels, mask)
    torch.testing.assert_close(actual, expected)

    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)

    ignored_logits = torch.randn(2, 4, 23, requires_grad=True)
    ignored = _causal_language_model_loss(
        ignored_logits,
        torch.full((2, 4), -100, dtype=torch.long),
        None,
    )
    assert ignored.item() == 0.0
    ignored.backward()
    assert ignored_logits.grad is not None
    assert torch.count_nonzero(ignored_logits.grad) == 0

    single = _causal_language_model_loss(
        torch.randn(2, 1, 23, requires_grad=True),
        torch.randint(0, 23, (2, 1)),
        None,
    )
    assert single.item() == 0.0


def test_gradient_checkpointing_disables_cache(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).train()
    model.gradient_checkpointing_enable()
    ids = torch.tensor([[1, 2, 3, 4]])
    output = model(input_ids=ids, labels=ids, use_cache=True)
    assert output.past_key_values is None
    output.loss.backward()


def test_causal_lm_reuses_one_explicit_context_for_base_and_head(
    tiny_config,
    monkeypatch,
):
    import rwkv7_hf.modeling_rwkv7 as modeling

    monkeypatch.setenv("RWKV7_BACKEND", "reference")
    resolved = []
    linear_contexts = []
    original_resolve = modeling.resolve_execution_context
    original_linear = modeling.RWKV7Linear.forward
    import rwkv7_hf.ops_rwkv7 as ops

    def record_resolve(*args, **kwargs):
        context = original_resolve(*args, **kwargs)
        resolved.append(context)
        return context

    def record_linear(self, value):
        linear_contexts.append(ops._linear_execution_context.get())
        return original_linear(self, value)

    monkeypatch.setattr(modeling, "resolve_execution_context", record_resolve)
    monkeypatch.setattr(modeling.RWKV7Linear, "forward", record_linear)
    model = RWKV7ForCausalLM(tiny_config).train()
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()

    assert len(resolved) == 1
    assert linear_contexts
    assert all(context is resolved[0] for context in linear_contexts)


def test_projection_and_output_modules_keep_standard_forward_contract(tiny_config):
    """PEFT and quantization wrappers must only need ``module(value)``."""

    model = RWKV7ForCausalLM(tiny_config).eval()
    model.model.layers[0].attn.r_proj = torch.nn.Linear(
        tiny_config.hidden_size,
        tiny_config.attention_hidden_size,
        bias=False,
    )
    model.set_output_embeddings(
        torch.nn.Linear(tiny_config.hidden_size, tiny_config.vocab_size, bias=False)
    )
    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert output.logits.shape == (1, 3, tiny_config.vocab_size)


def test_rwkv7_linear_forward_signature_matches_nn_linear():
    import inspect
    from rwkv7_hf.modeling_rwkv7 import RWKV7Linear

    parameters = tuple(inspect.signature(RWKV7Linear.forward).parameters)
    assert parameters == ("self", "value")


def test_execution_context_cannot_be_injected_through_public_forward(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).eval()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(TypeError, match="internal"):
        model.model(input_ids=ids, _execution_context=object())
    with pytest.raises(TypeError, match="internal"):
        model(input_ids=ids, _execution_context=object())


def test_causal_lm_calls_decoder_public_boundary_before_embeddings(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).eval()
    events = []
    decoder_hook = model.model.register_forward_pre_hook(
        lambda *_args: events.append("decoder_pre")
    )
    embedding_hook = model.model.embeddings.register_forward_pre_hook(
        lambda *_args: events.append("embedding_pre")
    )
    try:
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        decoder_hook.remove()
        embedding_hook.remove()
    assert events[:2] == ["decoder_pre", "embedding_pre"]


def test_set_decoder_standard_forward_wrapper_remains_supported(tiny_config):
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, *args, **kwargs):
            return self.decoder(*args, **kwargs)

    model = RWKV7ForCausalLM(tiny_config).eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids=ids, use_cache=False).logits
    model.set_decoder(DecoderWrapper(model.get_decoder()))
    actual = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


def test_decoder_context_capture_resets_after_exception(tiny_config):
    import rwkv7_hf.modeling_rwkv7 as modeling

    class FailingDecoder(torch.nn.Module):
        def forward(self, **_kwargs):
            raise RuntimeError("decoder failed")

    model = RWKV7ForCausalLM(tiny_config).eval()
    model.set_decoder(FailingDecoder())
    with pytest.raises(RuntimeError, match="decoder failed"):
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert modeling._execution_context_capture.get() is None


@pytest.mark.parametrize("instrumentation", ["instance_forward", "accelerate_hook"])
def test_whole_model_shortcut_never_bypasses_decoder_instrumentation(
    tiny_config, monkeypatch, instrumentation
):
    import types
    import rwkv7_hf.modeling_rwkv7 as modeling

    model = RWKV7ForCausalLM(tiny_config).eval()
    decoder_calls = []
    backend_calls = []
    original_forward = model.model.forward

    if instrumentation == "instance_forward":
        def wrapped_forward(_self, *args, **kwargs):
            decoder_calls.append("forward")
            return original_forward(*args, **kwargs)

        model.model.forward = types.MethodType(wrapped_forward, model.model)
    else:
        model.model._hf_hook = object()
        model.model._old_forward = original_forward
        decoder_calls.append("accelerate-installed")

    def record_backend(owner, *_args, **_kwargs):
        backend_calls.append(owner)
        return None

    monkeypatch.setattr(modeling, "maybe_model_forward", record_backend)
    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert output.logits.shape == (1, 3, tiny_config.vocab_size)
    assert backend_calls == []
    if instrumentation == "instance_forward":
        assert decoder_calls == ["forward"]


@pytest.mark.parametrize("target", ["embedding", "projection"])
def test_whole_model_shortcut_preserves_child_module_hooks(
    tiny_config, monkeypatch, target
):
    import rwkv7_hf.modeling_rwkv7 as modeling

    model = RWKV7ForCausalLM(tiny_config).eval()
    events = []
    backend_owners = []
    module = (
        model.model.embeddings
        if target == "embedding"
        else model.model.layers[0].attn.r_proj
    )
    handle = module.register_forward_pre_hook(
        lambda *_args: events.append(f"{target}_pre")
    )

    def record_backend(owner, *_args, **_kwargs):
        backend_owners.append(owner)
        return None

    monkeypatch.setattr(modeling, "maybe_model_forward", record_backend)
    try:
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        handle.remove()
    assert events == [f"{target}_pre"]
    assert backend_owners == []


def test_base_model_shortcut_preserves_child_module_hooks(tiny_config, monkeypatch):
    import rwkv7_hf.modeling_rwkv7 as modeling

    model = modeling.RWKV7Model(tiny_config).eval()
    events = []
    handle = model.layers[0].attn.r_proj.register_forward_pre_hook(
        lambda *_args: events.append("projection_pre")
    )
    monkeypatch.setattr(
        modeling,
        "maybe_model_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("instrumented base model must skip whole-model dispatch")
        ),
    )
    try:
        output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        handle.remove()
    assert output.last_hidden_state.shape == (1, 3, tiny_config.hidden_size)
    assert events == ["projection_pre"]


def test_checkpoint_recomputation_reuses_explicit_execution_context(
    tiny_config,
    monkeypatch,
):
    """Checkpoint replay must select the same optional linear program."""

    import rwkv7_hf.modeling_rwkv7 as modeling

    calls = []
    original = modeling.RWKV7Block.forward

    def record_context(self, *args, execution_context=None, **kwargs):
        calls.append(execution_context)
        return original(
            self,
            *args,
            execution_context=execution_context,
            **kwargs,
        )

    monkeypatch.setattr(modeling.RWKV7Block, "forward", record_context)
    model = RWKV7ForCausalLM(tiny_config).train()
    model.gradient_checkpointing_enable()
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()

    # Both the original checkpoint forward and backward recomputation receive
    # the same immutable model-call context by ordinary argument passing.
    assert len(calls) >= 2 * tiny_config.num_hidden_layers
    assert calls[0] is not None
    assert all(context is calls[0] for context in calls)
    assert calls[0].fully_active is True
    assert calls[0].token_aligned is None
    assert calls[0].initial_state_zero is True
    assert calls[0].optimized_program is False


def test_model_computes_mask_activity_once_outside_layer_shift_math():
    import inspect

    from rwkv7_hf.modeling_rwkv7 import _masked_token_shift

    source = inspect.getsource(_masked_token_shift)
    assert ".all(" not in source
    assert "fully_active" in source


def test_all_active_mask_writes_are_guarded_in_readable_model_source():
    import inspect

    time_mix_source = inspect.getsource(RWKV7TimeMix.forward)
    channel_mix_source = inspect.getsource(RWKV7ChannelMix.forward)
    block_source = inspect.getsource(RWKV7Block.forward)
    model_source = inspect.getsource(RWKV7Model.forward)

    assert "if mask_fully_active:\n                v_first = value" in time_mix_source
    assert "if not mask_fully_active:" in time_mix_source
    assert "if not mask_fully_active:" in channel_mix_source
    assert "if not mask_fully_active:" in block_source
    assert model_source.count("if not mask_fully_active:") == 2
    assert "v_first = torch.empty(" in model_source
