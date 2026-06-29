# coding=utf-8
"""Readable, pure-PyTorch Hugging Face implementation of RWKV-7.

The file intentionally shows the full model structure: token mixing, channel
mixing, residuals, normalization, layer iteration, cache handling, language
model loss, and generation integration. The only mathematical operator boundary
is rwkv7_recurrent in ops_rwkv7.py.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.generation import GenerationMixin
except ImportError:  # pragma: no cover - older Transformers
    from transformers.generation.utils import GenerationMixin

from .cache_rwkv7 import RWKV7Cache
from .configuration_rwkv7 import RWKV7Config
from .ops_rwkv7 import rwkv7_recurrent


# Mathematically equivalent form used by the official NumPy reference.
RWKV7_DECAY_BASE = math.exp(-0.5)


def _layer_norm(hidden_size: int, eps: float, bias: bool) -> nn.LayerNorm:
    try:
        return nn.LayerNorm(hidden_size, eps=eps, bias=bias)
    except TypeError:  # pragma: no cover - old PyTorch
        layer = nn.LayerNorm(hidden_size, eps=eps)
        if not bias:
            layer.register_parameter("bias", None)
        return layer


def _normalize_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            batch_size, sequence_length, dtype=torch.bool, device=device
        )
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if attention_mask.ndim != 2 or int(attention_mask.shape[0]) != batch_size:
        raise ValueError("attention_mask must be shaped [batch, sequence]")
    if int(attention_mask.shape[1]) < sequence_length:
        raise ValueError("attention_mask cannot be shorter than the model input")
    # Generation usually supplies the complete prefix mask while input_ids only
    # contains the newly generated suffix.
    return attention_mask[:, -sequence_length:].to(device=device, dtype=torch.bool)


def _masked_token_shift(
    hidden_states: torch.Tensor,
    previous: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the preceding valid token and the final shift state."""

    batch_size, sequence_length, _ = hidden_states.shape
    if tuple(previous.shape) != (batch_size, int(hidden_states.shape[-1])):
        raise ValueError("shift state must be shaped [batch, hidden]")

    if bool(attention_mask.all().detach().cpu()):
        shifted = torch.cat((previous.unsqueeze(1), hidden_states[:, :-1]), dim=1)
        return shifted, hidden_states[:, -1]

    shifted_tokens: list[torch.Tensor] = []
    carry = previous
    for token_idx in range(sequence_length):
        shifted_tokens.append(carry)
        active = attention_mask[:, token_idx].unsqueeze(-1)
        carry = torch.where(active, hidden_states[:, token_idx], carry)
    return torch.stack(shifted_tokens, dim=1), carry


def _group_norm_reference(
    value: torch.Tensor,
    norm: nn.GroupNorm,
) -> torch.Tensor:
    """Apply GroupNorm while retaining a CPU half-precision fallback."""

    if value.device.type == "cpu" and value.dtype in (torch.float16, torch.bfloat16):
        output = F.group_norm(
            value.float(),
            num_groups=norm.num_groups,
            weight=None if norm.weight is None else norm.weight.float(),
            bias=None if norm.bias is None else norm.bias.float(),
            eps=norm.eps,
        )
        return output.to(dtype=value.dtype)
    return F.group_norm(
        value,
        num_groups=norm.num_groups,
        weight=norm.weight,
        bias=norm.bias,
        eps=norm.eps,
    )


class RWKV7LowRank(nn.Module):
    """Two linear layers with checkpoint-compatible lora.0/lora.2 names."""

    def __init__(
        self,
        input_size: int,
        rank: int,
        output_size: int,
        *,
        bias: bool,
    ):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_size, rank, bias=False),
            nn.Identity(),
            nn.Linear(rank, output_size, bias=bias),
        )

    def project(self, value: torch.Tensor, activation=None) -> torch.Tensor:
        value = self.lora[0](value)
        if activation is not None:
            value = activation(value)
        return self.lora[2](value)


class RWKV7TimeMix(nn.Module):
    """RWKV-7 TMix, including projections and recurrent state update."""

    def __init__(self, config: RWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.head_dim)
        self.attention_hidden_size = int(config.attention_hidden_size)

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(
                self,
                name,
                nn.Parameter(torch.zeros(1, 1, self.hidden_size)),
            )

        self.k_k = nn.Parameter(torch.zeros(self.attention_hidden_size))
        self.k_a = nn.Parameter(torch.zeros(self.attention_hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(
            self.hidden_size, self.attention_hidden_size, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.attention_hidden_size, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.attention_hidden_size, bias=False
        )
        self.o_proj = nn.Linear(
            self.attention_hidden_size, self.hidden_size, bias=False
        )

        self.w_lora = RWKV7LowRank(
            self.hidden_size,
            config.decay_low_rank_dim,
            self.attention_hidden_size,
            bias=True,
        )
        self.a_lora = RWKV7LowRank(
            self.hidden_size,
            config.a_low_rank_dim,
            self.attention_hidden_size,
            bias=True,
        )
        self.g_lora = RWKV7LowRank(
            self.hidden_size,
            config.gate_low_rank_dim,
            self.attention_hidden_size,
            bias=False,
        )
        if self.layer_idx != 0:
            self.v_lora = RWKV7LowRank(
                self.hidden_size,
                config.v_low_rank_dim,
                self.attention_hidden_size,
                bias=True,
            )

        self.g_norm = nn.GroupNorm(
            self.num_heads,
            self.attention_hidden_size,
            eps=self.head_dim * config.norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor,
        shift_state: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        shifted, final_shift = _masked_token_shift(
            hidden_states, shift_state, attention_mask
        )
        delta = shifted - hidden_states

        xr = hidden_states + delta * self.x_r
        xw = hidden_states + delta * self.x_w
        xk = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        xa = hidden_states + delta * self.x_a
        xg = hidden_states + delta * self.x_g

        receptance = self.r_proj(xr)
        # ``w0`` is evaluated in FP32 by the official implementation.  Keep
        # the low-rank update in the model dtype, then add the stored bias only
        # after promoting both terms.  This is especially important when a
        # FP16 checkpoint is executed as BF16.
        decay_rank = torch.tanh(self.w_lora.lora[0](xw))
        raw_decay = F.linear(
            decay_rank,
            self.w_lora.lora[2].weight,
            bias=None,
        )
        key = self.k_proj(xk)
        value = self.v_proj(xv)
        in_context_learning = torch.sigmoid(self.a_lora.project(xa))
        gate = self.g_lora.project(xg, torch.sigmoid)

        weighted_key = key * self.k_k.view(1, 1, -1)
        normalized_key = F.normalize(
            weighted_key.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ),
            p=2,
            dim=-1,
        ).view(
            batch_size, sequence_length, self.attention_hidden_size
        )
        key = key * (
            1
            + (in_context_learning - 1)
            * self.k_a.view(1, 1, self.attention_hidden_size)
        )

        if self.layer_idx == 0:
            v_first = torch.where(
                attention_mask.unsqueeze(-1), value, torch.zeros_like(value)
            )
        else:
            value_mix = torch.sigmoid(self.v_lora.project(xv))
            value = value + (v_first - value) * value_mix

        decay_bias = self.w_lora.lora[2].bias
        if decay_bias is None:  # pragma: no cover - checkpoint contract guard
            raise RuntimeError("RWKV7 decay projection requires the w0 bias")
        decay = torch.exp(
            -RWKV7_DECAY_BASE
            * torch.sigmoid(raw_decay.float() + decay_bias.float())
        )

        shape = (
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        recurrent_output, recurrent_state = rwkv7_recurrent(
            receptance.view(shape),
            decay.view(shape),
            key.view(shape),
            value.view(shape),
            (-normalized_key).view(shape),
            (normalized_key * in_context_learning).view(shape),
            recurrent_state,
            attention_mask,
        )

        recurrent_output = recurrent_output.reshape(
            batch_size * sequence_length, self.attention_hidden_size
        )
        recurrent_output = _group_norm_reference(recurrent_output, self.g_norm)
        recurrent_output = recurrent_output.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        direct = (
            receptance.view(shape)
            * key.view(shape)
            * self.r_k.view(1, 1, self.num_heads, self.head_dim)
        ).sum(dim=-1, keepdim=True) * value.view(shape)

        mixed = (recurrent_output + direct).reshape(
            batch_size, sequence_length, self.attention_hidden_size
        )
        output = self.o_proj(mixed * gate)
        output = output * attention_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output, recurrent_state, final_shift, v_first


class RWKV7ChannelMix(nn.Module):
    """RWKV-7 CMix feed-forward block."""

    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.value = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        shift_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted, final_shift = _masked_token_shift(
            hidden_states, shift_state, attention_mask
        )
        mixed = hidden_states + (shifted - hidden_states) * self.x_k.view(1, 1, -1)
        activated = torch.relu(self.key(mixed)).square()
        output = self.value(activated)
        output = output * attention_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output, final_shift


class RWKV7Block(nn.Module):
    """One explicit pre-norm TMix + CMix residual block."""

    def __init__(self, config: RWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.attn = RWKV7TimeMix(config, layer_idx)
        self.ffn = RWKV7ChannelMix(config)
        self.attn_norm = _layer_norm(
            config.hidden_size, config.norm_eps, config.norm_bias
        )
        self.ffn_norm = _layer_norm(
            config.hidden_size, config.norm_eps, config.norm_bias
        )
        if self.layer_idx == 0:
            self.pre_norm = _layer_norm(
                config.hidden_size, config.norm_eps, config.norm_bias
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor,
        attention_shift: torch.Tensor,
        ffn_shift: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        residual = (
            self.pre_norm(hidden_states)
            if hasattr(self, "pre_norm")
            else hidden_states
        )
        attention_input = self.attn_norm(residual)
        attention_output, recurrent_state, attention_shift, v_first = self.attn(
            attention_input,
            recurrent_state,
            attention_shift,
            v_first,
            attention_mask,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        ffn_input = self.ffn_norm(hidden_states)
        ffn_output, ffn_shift = self.ffn(
            ffn_input, ffn_shift, attention_mask
        )
        hidden_states = residual + ffn_output
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )
        return (
            hidden_states,
            recurrent_state,
            attention_shift,
            ffn_shift,
            v_first,
        )


class RWKV7PreTrainedModel(PreTrainedModel):
    config_class = RWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Block"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _tied_weights_keys = {}
    # Official RWKV-7 promotes w0 for both FP16 and BF16 execution.  This also
    # tells ``from_pretrained(dtype=...)`` not to downcast the parameter after
    # the self-contained model has been instantiated.
    _keep_in_fp32_modules_strict = ["w_lora.lora.2.bias"]

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RWKV recurrent state is not a Transformer key/value cache.
        return False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=float(self.config.initializer_range)
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=float(self.config.initializer_range)
            )
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class RWKV7Model(RWKV7PreTrainedModel):
    """Embedding, readable RWKV-7 layer loop, final normalization, and cache."""

    def __init__(self, config: RWKV7Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [RWKV7Block(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = _layer_norm(
            config.hidden_size, config.norm_eps, config.norm_bias
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def resize_token_embeddings(
        self, new_num_tokens: int | None = None, *args, **kwargs
    ):
        del args, kwargs
        if new_num_tokens is None or int(new_num_tokens) == self.config.vocab_size:
            return self.embeddings
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; resizing is unsupported"
        )

    def _empty_layer_state(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(hidden_states.shape[0])
        recurrent = torch.zeros(
            batch_size,
            self.config.num_heads,
            self.config.head_dim,
            self.config.head_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        shift = torch.zeros(
            batch_size,
            self.config.hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return recurrent, shift, shift.clone()

    def _layer_state(
        self,
        cache: RWKV7Cache,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recurrent = cache.recurrent_state[layer_idx]
        attention_shift = cache.attention_shift[layer_idx]
        ffn_shift = cache.ffn_shift[layer_idx]
        if recurrent is None or attention_shift is None or ffn_shift is None:
            return self._empty_layer_state(hidden_states)
        batch_size = int(hidden_states.shape[0])
        if int(recurrent.shape[0]) != batch_size:
            raise ValueError(
                "past_key_values batch size does not match the current input"
            )
        return (
            recurrent.to(device=hidden_states.device),
            attention_shift.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ),
            ffn_shift.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

    def _checkpointed_layer(
        self,
        layer: RWKV7Block,
        hidden_states: torch.Tensor,
        recurrent: torch.Tensor,
        attention_shift: torch.Tensor,
        ffn_shift: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        def custom_forward(hidden, state, attn_shift, channel_shift, first_value):
            return layer(
                hidden,
                state,
                attn_shift,
                channel_shift,
                first_value,
                attention_mask,
            )

        checkpoint_fn = getattr(self, "_gradient_checkpointing_func", None)
        if checkpoint_fn is not None:
            return checkpoint_fn(
                custom_forward,
                hidden_states,
                recurrent,
                attention_shift,
                ffn_shift,
                v_first,
            )
        from torch.utils.checkpoint import checkpoint

        return checkpoint(
            custom_forward,
            hidden_states,
            recurrent,
            attention_shift,
            ffn_shift,
            v_first,
            use_reentrant=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RWKV7Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        del cache_position, position_ids, kwargs
        if output_attentions is None:
            output_attentions = bool(self.config.output_attentions)
        if output_attentions:
            raise NotImplementedError(
                "RWKV7 does not expose Transformer-style attention matrices"
            )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("specify input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")

        if inputs_embeds is None:
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2:
                raise ValueError("input_ids must be shaped [batch, sequence]")
            inputs_embeds = self.embeddings(input_ids)
        elif inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must be shaped [batch, sequence, hidden]")

        batch_size, sequence_length, hidden_size = inputs_embeds.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError("inputs_embeds hidden dimension does not match config")
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("RWKV7 requires a non-empty batch and sequence")

        mask = _normalize_attention_mask(
            attention_mask,
            int(batch_size),
            int(sequence_length),
            inputs_embeds.device,
        )
        hidden_states = inputs_embeds * mask.unsqueeze(-1).to(
            dtype=inputs_embeds.dtype
        )

        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        if self.training or (self.gradient_checkpointing and torch.is_grad_enabled()):
            use_cache = False
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else bool(output_hidden_states)
        )
        return_dict = (
            self.config.use_return_dict if return_dict is None else bool(return_dict)
        )

        if past_key_values is None:
            working_cache = RWKV7Cache(num_layers=len(self.layers))
        elif isinstance(past_key_values, RWKV7Cache):
            working_cache = past_key_values
            while len(working_cache) < len(self.layers):
                working_cache.recurrent_state.append(None)
                working_cache.attention_shift.append(None)
                working_cache.ffn_shift.append(None)
        else:
            working_cache = RWKV7Cache.from_legacy_cache(past_key_values)

        cached_batch = working_cache.get_batch_size()
        if cached_batch is not None and cached_batch != int(batch_size):
            raise ValueError(
                "past_key_values batch size does not match the current input"
            )

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        v_first = torch.zeros(
            batch_size,
            sequence_length,
            self.config.attention_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for layer_idx, layer in enumerate(self.layers):
            recurrent, attention_shift, ffn_shift = self._layer_state(
                working_cache, layer_idx, hidden_states
            )
            v_first = v_first.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            layer_mask = mask.to(device=hidden_states.device)
            if self.gradient_checkpointing and self.training:
                outputs = self._checkpointed_layer(
                    layer,
                    hidden_states,
                    recurrent,
                    attention_shift,
                    ffn_shift,
                    v_first,
                    layer_mask,
                )
            else:
                outputs = layer(
                    hidden_states,
                    recurrent,
                    attention_shift,
                    ffn_shift,
                    v_first,
                    layer_mask,
                )
            (
                hidden_states,
                recurrent,
                attention_shift,
                ffn_shift,
                v_first,
            ) = outputs
            if use_cache:
                working_cache.set_layer(
                    layer_idx, recurrent, attention_shift, ffn_shift
                )
            if output_hidden_states and layer_idx + 1 < len(self.layers):
                all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * mask.to(
            hidden_states.device
        ).unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            working_cache.seen_tokens += int(sequence_length)
            output_cache = working_cache
        else:
            output_cache = None

        if not return_dict:
            values = (hidden_states, output_cache, all_hidden_states)
            return tuple(value for value in values if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=output_cache,
            hidden_states=all_hidden_states,
        )


def _select_logits(
    logits: torch.Tensor,
    logits_to_keep: int | torch.Tensor | None,
) -> torch.Tensor:
    if logits_to_keep is None:
        return logits
    if isinstance(logits_to_keep, torch.Tensor):
        if logits_to_keep.ndim != 1:
            raise ValueError("tensor logits_to_keep must be one-dimensional")
        return logits.index_select(1, logits_to_keep.to(logits.device))
    count = int(logits_to_keep)
    if count <= 0:
        return logits
    return logits[:, -min(count, int(logits.shape[1])) :]


class RWKV7ForCausalLM(RWKV7PreTrainedModel, GenerationMixin):
    """Standard HF causal language-model wrapper around RWKV7Model."""

    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: RWKV7Config):
        super().__init__(config)
        self.model = RWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def resize_token_embeddings(
        self, new_num_tokens: int | None = None, *args, **kwargs
    ):
        return self.model.resize_token_embeddings(new_num_tokens, *args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: RWKV7Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if past_key_values is not None:
            seen = int(past_key_values.get_seq_length())
            if input_ids.shape[1] > seen:
                input_ids = input_ids[:, seen:]
            else:
                input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs: dict[str, Any]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "use_cache": use_cache,
            }
        )
        for key in ("logits_to_keep", "num_logits_to_keep"):
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        return model_inputs

    def _reorder_cache(
        self, past_key_values: RWKV7Cache, beam_idx: torch.LongTensor
    ):
        return past_key_values.reorder_cache(beam_idx)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: RWKV7Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        num_logits_to_keep: int | torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        if num_logits_to_keep is not None:
            if logits_to_keep not in (None, 0):
                if not (
                    isinstance(logits_to_keep, torch.Tensor)
                    and isinstance(num_logits_to_keep, torch.Tensor)
                    and torch.equal(logits_to_keep, num_logits_to_keep)
                ) and logits_to_keep != num_logits_to_keep:
                    raise ValueError(
                        "logits_to_keep and num_logits_to_keep disagree"
                    )
            logits_to_keep = num_logits_to_keep

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        full_logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            if tuple(labels.shape[:2]) != tuple(full_logits.shape[:2]):
                raise ValueError("labels must have the same [batch, sequence] shape")
            shifted_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                label_mask = attention_mask[:, -labels.shape[1] :].to(
                    device=shifted_labels.device, dtype=torch.bool
                )
                shifted_labels = shifted_labels.masked_fill(
                    ~label_mask[:, 1:], -100
                )
            shifted_logits = full_logits[:, :-1].contiguous()
            if shifted_logits.numel() == 0 or not bool(
                (shifted_labels != -100).any().detach().cpu()
            ):
                loss = full_logits.float().sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]).float(),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                )

        logits = _select_logits(full_logits, logits_to_keep)
        return_dict = (
            self.config.use_return_dict if return_dict is None else bool(return_dict)
        )
        if not return_dict:
            values = (
                loss,
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
            )
            return tuple(value for value in values if value is not None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# Compatibility aliases for 0.8 callers. New model repositories use RWKV7*.
NativeRWKV7Model = RWKV7Model
NativeRWKV7ForCausalLM = RWKV7ForCausalLM


try:
    RWKV7Model.register_for_auto_class("AutoModel")
    RWKV7ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
except Exception:  # pragma: no cover - older Transformers
    pass


__all__ = [
    "RWKV7TimeMix",
    "RWKV7ChannelMix",
    "RWKV7Block",
    "RWKV7PreTrainedModel",
    "RWKV7Model",
    "RWKV7ForCausalLM",
    "NativeRWKV7Model",
    "NativeRWKV7ForCausalLM",
]
