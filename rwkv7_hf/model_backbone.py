# coding=utf-8
"""Native RWKV-7 backbone and its recurrent forward helpers."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import (
    _eager_model_is_multi_device,
    _init_state_batched,
    _move_layer_inputs,
    _ordered_to_device,
    _step_token_batched,
)
from .model_cache import (
    NativeRWKV7Cache,
    _cache_seen,
    _copy_native_cache_tuple,
    _native_cache_tuple_or_none,
    _validate_native_cache_batch_size,
)
from .model_config import NativeRWKV7Config
from .model_layers import NativeRWKV7Layer


def _validate_native_attention_mask(
    attention_mask,
    batch_size: int,
    seq_len: int,
    device=None,
    *,
    allow_trailing: bool = False,
):
    """Validate and normalize the native/upstream attention-mask contract.

    RWKV recurrent state is order-sensitive and does not have Transformer-style
    random-access KV masking.  All-ones masks are equivalent to no mask.  Masked
    tokens are handled by skipping recurrent-state updates for those batch rows.
    """

    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("NativeRWKV7 attention_mask must be a torch.Tensor when provided")
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.view(1, -1)
    if attention_mask.dim() != 2:
        raise ValueError("NativeRWKV7 attention_mask must be shaped [batch, seq]")
    if int(attention_mask.shape[0]) != int(batch_size):
        raise ValueError("NativeRWKV7 attention_mask batch size must match inputs")
    if int(attention_mask.shape[1]) != int(seq_len):
        if not allow_trailing or int(attention_mask.shape[1]) < int(seq_len):
            raise ValueError("NativeRWKV7 attention_mask must have the same [batch, seq] shape as inputs")
        attention_mask = attention_mask[:, -seq_len:]
    mask = attention_mask.to(device=device) if device is not None else attention_mask
    mask = mask[:, :seq_len] != 0
    if mask.numel() and bool(torch.all(mask).detach().cpu().item()):
        return None
    return mask


def _blend_native_recurrent_state(mask: torch.Tensor, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first):
    """Keep old recurrent rows where ``mask`` is false."""

    if bool(torch.all(mask).detach().cpu().item()):
        return state, xpa, xpf, v_first
    state_mask = mask.view(-1, 1, 1, 1)
    hidden_mask = mask.view(-1, 1)
    state = [torch.where(state_mask.to(new.device), new, old) for old, new in zip(old_state, state, strict=False)]
    xpa = [torch.where(hidden_mask.to(new.device), new, old) for old, new in zip(old_xpa, xpa, strict=False)]
    xpf = [torch.where(hidden_mask.to(new.device), new, old) for old, new in zip(old_xpf, xpf, strict=False)]
    v_first = torch.where(hidden_mask.to(v_first.device), v_first, old_v_first)
    return state, xpa, xpf, v_first


def _validate_native_output_attentions(output_attentions, config) -> None:
    requested = bool(getattr(config, "output_attentions", False) if output_attentions is None else output_attentions)
    if requested:
        raise NotImplementedError("NativeRWKV7 does not expose Transformer-style attention maps")


def _step_token_batched_with_hidden(model, x, state, xpa, xpf, v_first):
    """Native eager token step that also returns per-layer hidden outputs."""

    layer_hiddens = []
    multi_device = _eager_model_is_multi_device(model)
    for i, layer in enumerate(model.model.layers):
        if multi_device:
            x, state[i], xpa[i], xpf[i], v_first = _move_layer_inputs(
                layer,
                x,
                state[i],
                xpa[i],
                xpf[i],
                v_first,
            )
        attn = layer.attn
        residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
        h = layer.attn_norm(residual)
        a, xpa[i], state[i], v_first = attn(h, xpa[i], v_first, state[i])
        x = residual + a
        residual = x
        h2 = layer.ffn_norm(x)
        f, xpf[i] = layer.ffn(h2, xpf[i])
        x = residual + f
        layer_hiddens.append(x)
    return x, state, xpa, xpf, v_first, layer_hiddens


class NativeRWKV7Model(PreTrainedModel):
    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["NativeRWKV7Layer"]
    _skip_keys_device_placement = ["past_key_values"]
    supports_gradient_checkpointing = True
    _tied_weights_keys = {}

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NativeRWKV7Layer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        # Populates the Transformers-native TP/PP plans from the config.  All
        # official model implementations call post_init after constructing
        # their modules; doing the same is required for ``tp_plan="auto"``.
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def resize_token_embeddings(self, new_num_tokens: int | None = None, *args, **kwargs):
        """RWKV checkpoints use the fixed official trie vocabulary."""

        if new_num_tokens is None or int(new_num_tokens) == int(self.config.vocab_size):
            return self.get_input_embeddings()
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; changing vocab size "
            "with resize_token_embeddings is not supported by this adapter."
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask=None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        position_ids=None,
        cache_position=None,
        token_type_ids=None,
        head_mask=None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("NativeRWKV7Model accepts either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7Model requires input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.view(1, -1)
            if input_ids.dim() != 2:
                raise ValueError("NativeRWKV7Model expects input_ids shaped [batch, seq]")
            batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            device, dtype = input_ids.device, self.embeddings.weight.dtype
        else:
            if inputs_embeds.dim() != 3:
                raise ValueError("NativeRWKV7Model expects inputs_embeds shaped [batch, seq, hidden]")
            if int(inputs_embeds.shape[-1]) != int(self.config.hidden_size):
                raise ValueError("NativeRWKV7Model inputs_embeds last dimension must match hidden_size")
            batch_size, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
            device, dtype = inputs_embeds.device, inputs_embeds.dtype
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("NativeRWKV7Model requires a non-empty batch and sequence")
        native_cache = _native_cache_tuple_or_none(past_key_values)
        _validate_native_cache_batch_size(native_cache, batch_size)
        native_attention_mask = _validate_native_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            device=device,
            allow_trailing=native_cache is not None,
        )
        _validate_native_output_attentions(output_attentions, self.config)
        if return_dict is None:
            return_dict = bool(getattr(self.config, "return_dict", True))
        output_hidden_states = bool(
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)

        class _Runner:
            pass

        runner = _Runner()
        runner.model = self
        if native_cache is None:
            state, xpa, xpf, v_first = _init_state_batched(runner, batch_size, device, dtype)
            seen = seq_len
        else:
            state, xpa, xpf, v_first = _copy_native_cache_tuple(native_cache)
            seen = _cache_seen(past_key_values) + seq_len

        final_hidden = []
        hidden_buckets = [[] for _ in range(self.config.num_hidden_layers + 1)] if output_hidden_states else None
        hidden_size = int(self.config.hidden_size)
        last_normed = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        last_layer_hiddens = (
            [torch.zeros(batch_size, hidden_size, device=device, dtype=dtype) for _ in range(self.config.num_hidden_layers + 1)]
            if hidden_buckets is not None
            else None
        )
        for t in range(seq_len):
            x = inputs_embeds[:, t] if inputs_embeds is not None else self.embeddings(input_ids[:, t])
            token_mask = native_attention_mask[:, t] if native_attention_mask is not None else None
            if token_mask is not None:
                old_state, old_xpa, old_xpf, old_v_first = list(state), list(xpa), list(xpf), v_first
            if hidden_buckets is not None:
                emb_hidden = x
                if token_mask is not None:
                    emb_hidden = torch.where(token_mask.view(batch_size, 1).to(x.device), emb_hidden, last_layer_hiddens[0])
                hidden_buckets[0].append(emb_hidden)
                x, state, xpa, xpf, v_first, layer_hiddens = _step_token_batched_with_hidden(
                    runner, x, state, xpa, xpf, v_first
                )
                normed = self.norm(x)
                if token_mask is not None:
                    state, xpa, xpf, v_first = _blend_native_recurrent_state(
                        token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                    )
                    mask_h = token_mask.view(batch_size, 1).to(normed.device)
                    normed = torch.where(mask_h, normed, last_normed)
                    layer_hiddens = [
                        torch.where(mask_h.to(layer_hidden.device), layer_hidden, last_layer_hiddens[layer_idx + 1])
                        for layer_idx, layer_hidden in enumerate(layer_hiddens)
                    ]
                for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1):
                    hidden_buckets[layer_idx].append(normed if layer_idx == self.config.num_hidden_layers else layer_hidden)
                last_layer_hiddens = [emb_hidden] + [
                    normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1)
                ]
            else:
                x, state, xpa, xpf, v_first = _step_token_batched(runner, x, state, xpa, xpf, v_first)
                normed = self.norm(x)
                if token_mask is not None:
                    state, xpa, xpf, v_first = _blend_native_recurrent_state(
                        token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                    )
                    normed = torch.where(token_mask.view(batch_size, 1).to(normed.device), normed, last_normed)
            final_hidden.append(normed)
            last_normed = normed

        last_hidden_state = _ordered_to_device(torch.stack(final_hidden, dim=1), device)
        new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=seen) if use_cache else None
        hidden_states = None
        if hidden_buckets is not None:
            hidden_states = tuple(
                _ordered_to_device(torch.stack(bucket, dim=1), device)
                for bucket in hidden_buckets
            )
        if not return_dict:
            values = (last_hidden_state, new_cache, hidden_states)
            return tuple(v for v in values if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=new_cache,
            hidden_states=hidden_states,
        )
