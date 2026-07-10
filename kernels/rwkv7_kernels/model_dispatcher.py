"""Whole-model capability dispatch for the optional kernel package.

This module is the only public bridge from a clean Hugging Face model object to
performance implementations.  Backend implementations live below
``rwkv7_kernels.model`` and may inspect the documented RWKV-7 module structure,
but may not import or replace ``rwkv7_hf.modeling_rwkv7``.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from .protocol import (
    support_result,
    validate_model_request,
)
from .trace import record_model
_NOT_MIGRATED = (
    "whole-model backend-v2 is not available for this shape; "
    "the adapter will use its readable reference layer loop"
)
_MODEL_IMPL_ENV = "RWKV7_MODEL_KERNEL_IMPL"
_MODEL_IMPLS = ("auto", "dense", "native")
_DENSE_IMPLEMENTATION = "native-torchscript-dense-sequential-v2"
_NATIVE_PREFILL_IMPLEMENTATION = "native-nvidia-prefill-v2"
_NATIVE_DECODE_IMPLEMENTATION = "native-nvidia-fused-decode-v2"
_NATIVE_TRAINING_IMPLEMENTATION = "native-nvidia-train-temp-autograd-v2"


def _phase(request: dict[str, Any]) -> str:
    if bool(request["training"]) or bool(request.get("grad_enabled", False)):
        return "training"
    cache = request.get("past_key_values")
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length) and int(get_seq_length()) > 0:
        return "decode"
    value = request.get("hidden_states")
    if value is None:
        value = request.get("input_ids")
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        if request.get("model_kind") == "causal_lm":
            return "prefill"
        return "decode" if int(value.shape[1]) == 1 else "prefill"
    return "prefill"


def _requested_implementation() -> str:
    value = os.environ.get(_MODEL_IMPL_ENV, "auto").strip().lower()
    if value not in _MODEL_IMPLS:
        choices = ", ".join(_MODEL_IMPLS)
        raise ValueError(f"{_MODEL_IMPL_ENV} must be one of {choices}; got {value!r}")
    return value


def _probe_dense(owner: Any, request: dict[str, Any]):
    if request["model_kind"] != "base":
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 currently accepts the base model boundary only",
            phase=_phase(request),
        )
    hidden = request.get("hidden_states")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 requires [B,T,D] hidden_states",
            phase=_phase(request),
        )
    if hidden.device.type != "cuda":
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 is an NVIDIA CUDA implementation",
            phase=_phase(request),
        )
    if hidden.dtype not in (torch.float16, torch.bfloat16):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 requires FP16 or BF16 model tensors",
            phase=_phase(request),
        )
    if bool(request["training"]) or bool(request.get("grad_enabled", False)):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="dense-v2 is inference-only while training kernels migrate",
            phase=_phase(request),
        )
    if not hasattr(owner, "layers") or not hasattr(owner, "norm"):
        return support_result(
            supported=False,
            implementation=_DENSE_IMPLEMENTATION,
            reason="owner does not expose the clean RWKV7 base-model structure",
            phase=_phase(request),
        )
    return support_result(
        supported=True,
        implementation=_DENSE_IMPLEMENTATION,
        reason="explicit dense-v2 diagnostic implementation selected",
        phase=_phase(request),
    )


def _unsupported_native(request: dict[str, Any], reason: str):
    implementation = (
        _NATIVE_DECODE_IMPLEMENTATION
        if _phase(request) == "decode"
        else _NATIVE_PREFILL_IMPLEMENTATION
    )
    return support_result(
        supported=False,
        implementation=implementation,
        reason=reason,
        phase=_phase(request),
    )


def _effective_attention_mask(
    request: dict[str, Any], input_ids: torch.Tensor
) -> torch.Tensor:
    batch, sequence = int(input_ids.shape[0]), int(input_ids.shape[1])
    mask = request.get("attention_mask")
    if mask is None:
        return torch.ones(
            batch,
            sequence,
            device=input_ids.device,
            dtype=torch.bool,
        )
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (1, 2):
        raise ValueError("native attention_mask must be rank 1 or 2")
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if int(mask.shape[0]) not in (1, batch) or int(mask.shape[1]) < sequence:
        raise ValueError(
            "native attention_mask must broadcast to the input batch and cover "
            "the current sequence"
        )
    if int(mask.shape[0]) == 1 and batch != 1:
        mask = mask.expand(batch, -1)
    return mask[:, -sequence:].to(device=input_ids.device, dtype=torch.bool)


def _probe_native(owner: Any, request: dict[str, Any]):
    """Capability gate for the migrated fused prefill runtime."""

    if request["model_kind"] != "causal_lm":
        return _unsupported_native(
            request, "native prefill requires the causal-LM boundary"
        )
    if bool(request["training"]):
        return _probe_native_training(owner, request)
    if bool(request.get("grad_enabled", False)):
        return _unsupported_native(
            request, "native inference requires torch.no_grad or inference_mode"
        )
    if request.get("labels") is not None:
        return _unsupported_native(request, "native prefill does not accept labels")
    if request.get("inputs_embeds") is not None:
        return _unsupported_native(
            request, "native prefill currently requires input_ids"
        )
    if bool(request.get("output_hidden_states")):
        return _unsupported_native(
            request, "native prefill hidden-state history is not enabled"
        )
    if bool(request.get("output_attentions")):
        return _unsupported_native(
            request, "RWKV7 does not expose Transformer attention matrices"
        )
    input_ids = request.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return _unsupported_native(request, "native prefill requires input_ids")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        return _unsupported_native(
            request, "native prefill requires non-empty [B,T] input_ids"
        )
    if input_ids.device.type != "cuda":
        return _unsupported_native(request, "native prefill requires CUDA input_ids")
    if not hasattr(owner, "model") or not hasattr(owner, "lm_head"):
        return _unsupported_native(
            request, "owner does not expose the clean RWKV7 causal-LM structure"
        )
    dtype = owner.model.embeddings.weight.dtype
    if dtype != torch.float16:
        return _unsupported_native(
            request, "native prefill v2 currently accepts FP16 checkpoints"
        )
    mask = request.get("attention_mask")
    if mask is not None:
        try:
            _effective_attention_mask(request, input_ids)
        except ValueError as exc:
            return _unsupported_native(
                request, str(exc)
            )
    cache = request.get("past_key_values")
    if cache is None or not hasattr(cache, "get_seq_length"):
        return _unsupported_native(
            request, "native prefill requires the canonical RWKV7 cache envelope"
        )
    seen_tokens = int(cache.get_seq_length())
    initialized = bool(cache.is_initialized())
    sequence = int(input_ids.shape[1])
    if seen_tokens > 0:
        if sequence != 1:
            return _unsupported_native(
                request, "native cached decode currently accepts one token"
            )
        if not initialized:
            return _unsupported_native(
                request, "native cached decode requires every layer state"
            )
    elif initialized:
        return _unsupported_native(
            request, "zero-length cache must not contain initialized state"
        )
    keep = request.get("logits_to_keep")
    if isinstance(keep, torch.Tensor):
        return _unsupported_native(
            request, "native prefill tensor logits_to_keep is not enabled"
        )
    implementation = (
        _NATIVE_DECODE_IMPLEMENTATION
        if seen_tokens > 0
        else _NATIVE_PREFILL_IMPLEMENTATION
    )
    return support_result(
        supported=True,
        implementation=implementation,
        reason=f"explicit migrated NVIDIA {_phase(request)} implementation selected",
        phase=_phase(request),
    )


def _unsupported_training(reason: str):
    return support_result(
        supported=False,
        implementation=_NATIVE_TRAINING_IMPLEMENTATION,
        reason=reason,
        phase="training",
    )


def _probe_native_training(owner: Any, request: dict[str, Any]):
    if not bool(request.get("grad_enabled")):
        return _unsupported_training("native training requires autograd to be enabled")
    if bool(request.get("use_cache")):
        return _unsupported_training("native training is a dense no-cache path")
    if bool(request.get("output_hidden_states")):
        return _unsupported_training(
            "native training hidden-state history is not enabled"
        )
    if bool(request.get("output_attentions")):
        return _unsupported_training(
            "RWKV7 does not expose Transformer attention matrices"
        )
    input_ids = request.get("input_ids")
    inputs_embeds = request.get("inputs_embeds")
    if (input_ids is None) == (inputs_embeds is None):
        return _unsupported_training(
            "native training requires exactly one of input_ids or inputs_embeds"
        )
    value = inputs_embeds if inputs_embeds is not None else input_ids
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        return _unsupported_training("native training input shape is invalid")
    if not hasattr(owner, "model") or not hasattr(owner.model, "layers"):
        return _unsupported_training(
            "owner does not expose the clean RWKV7 causal-LM structure"
        )
    if any(
        type(layer.ffn.key) is not torch.nn.Linear
        or type(layer.ffn.value) is not torch.nn.Linear
        for layer in owner.model.layers
    ):
        return _unsupported_training(
            "native train_temp bypasses wrapped FFN modules; adapters use the "
            "reference autograd path"
        )
    if value.device.type != "cuda":
        return _unsupported_training("native training requires CUDA tensors")
    tokens = int(value.shape[1])
    if tokens <= 0 or tokens % 16:
        return _unsupported_training(
            "native training sequence length must be divisible by 16"
        )
    if owner.model.embeddings.weight.dtype != torch.bfloat16:
        return _unsupported_training("native training requires a BF16 checkpoint")
    if inputs_embeds is not None and inputs_embeds.dtype != torch.bfloat16:
        return _unsupported_training("native training inputs_embeds must be BF16")
    if int(owner.config.head_dim) != 64:
        return _unsupported_training("native training requires head_dim=64")
    mask = request.get("attention_mask")
    if mask is not None and not bool(mask.to(dtype=torch.bool).all().detach().cpu()):
        return _unsupported_training("native training does not accept padded batches")
    labels = request.get("labels")
    if labels is not None:
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
            return _unsupported_training("native training labels must be int64")
        if labels.device != value.device:
            return _unsupported_training(
                "native training labels and inputs must share a device"
            )
        if tuple(labels.shape) != tuple(value.shape[:2]):
            return _unsupported_training(
                "native training labels must match the input batch and sequence"
            )
        if bool((labels < 0).any().detach().cpu()):
            return _unsupported_training(
                "native fused loss does not accept -100 or negative labels"
            )
    from .nvidia.train_temp_cuda import train_temp_cuda_available

    if not train_temp_cuda_available(build=False):
        return _unsupported_training(
            "native train_temp CUDA runtime is unavailable on this device"
        )
    return support_result(
        supported=True,
        implementation=_NATIVE_TRAINING_IMPLEMENTATION,
        reason="migrated train_temp autograd implementation selected",
        phase="training",
    )


def _run_native_prefill(owner: Any, request: dict[str, Any]):
    from types import SimpleNamespace

    from .nvidia import native_jit

    input_ids = request["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    proxy = SimpleNamespace(model=owner.model, lm_head=owner.lm_head)
    packs, _heads, _head_dim, _eps = native_jit.extract_graph(proxy)
    mask = _effective_attention_mask(request, input_ids)
    masked = not bool(mask.all().detach().cpu())
    if not masked:
        logits, state_vk, attention_shift, ffn_shift = native_jit.prefill(
            proxy,
            input_ids,
            packs,
            logits_to_keep=request.get("logits_to_keep"),
        )
    else:
        batch, sequence = int(input_ids.shape[0]), int(input_ids.shape[1])
        vocab = int(owner.lm_head.out_features)
        logits = owner.model.embeddings.weight.new_zeros(batch, sequence, vocab)
        state_vk = [
            torch.zeros(
                batch,
                int(_heads),
                int(_head_dim),
                int(_head_dim),
                device=input_ids.device,
                dtype=torch.float32,
            )
            for _ in packs
        ]
        attention_shift = [
            owner.model.embeddings.weight.new_zeros(
                batch, int(owner.config.hidden_size)
            )
            for _ in packs
        ]
        ffn_shift = [value.clone() for value in attention_shift]
        for batch_index in range(batch):
            positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            compact_ids = input_ids[batch_index : batch_index + 1].index_select(
                1, positions
            )
            row_logits, row_state, row_attention, row_ffn = native_jit.prefill(
                proxy,
                compact_ids,
                packs,
                logits_to_keep=0,
            )
            logits[batch_index].index_copy_(0, positions, row_logits[0])
            for layer_index in range(len(packs)):
                state_vk[layer_index][batch_index : batch_index + 1].copy_(
                    row_state[layer_index].float()
                )
                attention_shift[layer_index][
                    batch_index : batch_index + 1
                ].copy_(row_attention[layer_index])
                ffn_shift[layer_index][batch_index : batch_index + 1].copy_(
                    row_ffn[layer_index]
                )
        keep = request.get("logits_to_keep")
        if keep is not None and int(keep) > 0:
            logits = logits[:, -min(int(keep), sequence) :]

    output_cache = None
    if bool(request["use_cache"]):
        output_cache = request["past_key_values"]
        for layer_idx in range(len(packs)):
            output_cache.set_layer(
                layer_idx,
                state_vk[layer_idx].float().transpose(-1, -2).contiguous(),
                attention_shift[layer_idx],
                ffn_shift[layer_idx],
            )
        output_cache.seen_tokens += int(input_ids.shape[1])

    effective = []
    for field, label in (
        ("_rwkv7_native_prefill_clampw_scan_effective", "clampw"),
        ("_rwkv7_native_prefill_self_chunk_effective", "self_chunk"),
        ("_rwkv7_native_prefill_sequence_ffn_effective", "sequence_ffn"),
        ("_rwkv7_native_prefill_stacked_rkv_effective", "stacked_rkv"),
        ("_rwkv7_native_prefill_wavg_lora_effective", "wavg_lora"),
        ("_rwkv7_native_prefill_fp16_recurrent_effective", "fp16_state"),
    ):
        if bool(getattr(proxy, field, False)):
            effective.append(label)
    if masked:
        effective.append("masked_compact")
    suffix = "+".join(effective) if effective else "dense_fallback"
    return {
        "output_kind": "causal_lm",
        "logits": logits,
        "loss": None,
        "past_key_values": output_cache,
        "hidden_states": None,
        "implementation": f"{_NATIVE_PREFILL_IMPLEMENTATION}[{suffix}]",
        "phase": _phase(request),
    }


def _run_native_decode(owner: Any, request: dict[str, Any]):
    from types import SimpleNamespace

    from .nvidia import native_jit

    input_ids = request["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    cache = request["past_key_values"]
    proxy = SimpleNamespace(model=owner.model, lm_head=owner.lm_head)
    packs, heads, head_dim, _eps = native_jit.extract_graph(proxy)
    batch = int(input_ids.shape[0])
    dtype = owner.model.embeddings.weight.dtype
    mask = _effective_attention_mask(request, input_ids)[:, 0]

    if bool(request["use_cache"]) and bool(mask.all().detach().cpu()):
        from .nvidia.graph_pool import get_native_graph_runner
        from .nvidia.native_graph_runtime import native_graph_available

        if native_graph_available():
            runner = get_native_graph_runner(owner, packs, batch)
            logits = runner.replay(input_ids[:, 0], cache, copy_logits=True)
            cache.seen_tokens += 1
            stats = runner.copy_stats()
            effective = [
                name
                for name in (
                    "ada_wagv_bmm",
                    "sm120_wagv_bmm_g",
                    "sm120_compiled_ffn",
                    "sm70_wagv_lora",
                    "fused_wavg_lora",
                )
                if bool(stats.get(f"{name}_effective"))
            ]
            effective.append("cuda_graph")
            return {
                "output_kind": "causal_lm",
                "logits": logits,
                "loss": None,
                "past_key_values": cache,
                "hidden_states": None,
                "implementation": (
                    f"{_NATIVE_DECODE_IMPLEMENTATION}[{'+'.join(effective)}]"
                ),
                "phase": "decode",
            }

    state_vk = []
    attention_shift = []
    ffn_shift = []
    for layer_idx in range(len(packs)):
        recurrent = cache.recurrent_state[layer_idx]
        attn_previous = cache.attention_shift[layer_idx]
        ffn_previous = cache.ffn_shift[layer_idx]
        if recurrent is None or attn_previous is None or ffn_previous is None:
            raise RuntimeError("native decode received an incomplete canonical cache")
        state_vk.append(recurrent.transpose(-1, -2).contiguous())
        attention_shift.append(attn_previous.clone())
        ffn_shift.append(ffn_previous.clone())

    active = torch.nonzero(mask, as_tuple=False).flatten()
    if active.numel() == 0:
        logits = owner.model.embeddings.weight.new_zeros(
            batch, 1, int(owner.lm_head.out_features)
        )
        cache.seen_tokens += 1
        return {
            "output_kind": "causal_lm",
            "logits": logits,
            "loss": None,
            "past_key_values": cache if bool(request["use_cache"]) else None,
            "hidden_states": None,
            "implementation": f"{_NATIVE_DECODE_IMPLEMENTATION}[masked_noop]",
            "phase": "decode",
        }

    compact = int(active.numel()) != batch
    work_state = (
        [value.index_select(0, active) for value in state_vk]
        if compact
        else state_vk
    )
    work_attention = (
        [value.index_select(0, active) for value in attention_shift]
        if compact
        else attention_shift
    )
    work_ffn = (
        [value.index_select(0, active) for value in ffn_shift]
        if compact
        else ffn_shift
    )
    active_ids = input_ids.index_select(0, active) if compact else input_ids
    token = F.embedding(active_ids[:, 0], owner.model.embeddings.weight)
    v_first = torch.zeros(
        int(active.numel()),
        int(heads * head_dim),
        device=token.device,
        dtype=dtype,
    )
    events: set[str] = set()

    def observe(name: str, _layer_idx: int) -> None:
        events.add(str(name).removesuffix("_selected").removesuffix("_effective"))

    for layer_idx, pack in enumerate(packs):
        token = native_jit._block_ip_batched(
            token,
            work_state[layer_idx],
            work_attention[layer_idx],
            work_ffn[layer_idx],
            v_first,
            pack,
            route_observer=observe,
            state_layout="vk_v1",
        )
    token = owner.model.norm(token)
    active_logits = owner.lm_head(token).unsqueeze(1)
    if compact:
        logits = active_logits.new_zeros(
            batch, 1, int(active_logits.shape[-1])
        )
        logits.index_copy_(0, active, active_logits)
        for layer_idx in range(len(packs)):
            state_vk[layer_idx].index_copy_(0, active, work_state[layer_idx])
            attention_shift[layer_idx].index_copy_(
                0, active, work_attention[layer_idx]
            )
            ffn_shift[layer_idx].index_copy_(0, active, work_ffn[layer_idx])
    else:
        logits = active_logits

    output_cache = None
    if bool(request["use_cache"]):
        output_cache = cache
        for layer_idx in range(len(packs)):
            output_cache.set_layer(
                layer_idx,
                state_vk[layer_idx].transpose(-1, -2).contiguous(),
                attention_shift[layer_idx],
                ffn_shift[layer_idx],
            )
        output_cache.seen_tokens += 1

    if compact:
        events.add("masked_compact")
    suffix = "+".join(sorted(events)) if events else "dense_fallback"
    return {
        "output_kind": "causal_lm",
        "logits": logits,
        "loss": None,
        "past_key_values": output_cache,
        "hidden_states": None,
        "implementation": f"{_NATIVE_DECODE_IMPLEMENTATION}[{suffix}]",
        "phase": "decode",
    }


def probe_model_forward_v1(owner: Any, request: dict[str, Any]):
    """Return whether a migrated whole-model implementation accepts a call."""

    validate_model_request(request)
    if _requested_implementation() == "dense":
        return _probe_dense(owner, request)
    if _requested_implementation() == "native":
        return _probe_native(owner, request)
    # Keep production auto disabled until every phase in the frozen one-shot
    # inventory has passed. Explicit dense diagnostics exercise the final ABI
    # without advertising a half-migrated production route.
    return support_result(
        supported=False,
        implementation="rwkv7-model-backend-v2",
        reason=_NOT_MIGRATED,
        phase=_phase(request),
    )


def model_forward_v1(owner: Any, request: dict[str, Any]):
    """Execute a supported whole-model request.

    Calling this after a negative probe is a protocol error.  Concrete phase
    dispatch is added here only after decode, prefill, cache and training
    implementations all satisfy the frozen backend-v2 acceptance matrix.
    """

    validate_model_request(request)

    def traced(result: dict[str, Any]) -> dict[str, Any]:
        record_model(
            str(result.get("implementation", "unknown-model-implementation")),
            str(result.get("phase", _phase(request))),
        )
        return result

    implementation = _requested_implementation()
    if implementation == "native":
        support = _probe_native(owner, request)
        if not support["supported"]:
            raise RuntimeError(support["reason"])
        if bool(request["training"]):
            from .nvidia.training_runtime import run_training

            return traced(run_training(owner, request))
        cache = request["past_key_values"]
        if int(cache.get_seq_length()) > 0:
            return traced(_run_native_decode(owner, request))
        return traced(_run_native_prefill(owner, request))
    if implementation != "dense":
        raise RuntimeError(_NOT_MIGRATED)
    support = _probe_dense(owner, request)
    if not support["supported"]:
        raise RuntimeError(support["reason"])
    from .model.dense import run_base_model

    return traced(run_base_model(owner, request))


__all__ = ["model_forward_v1", "probe_model_forward_v1"]
