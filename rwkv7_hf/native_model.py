# coding=utf-8
"""Canonical FLA-free RWKV-7 model for Hugging Face Transformers.

Inference dispatches to compiled full-sequence prefill and fixed-batch CUDA
graphs when the runtime is eligible, with native JIT and eager PyTorch fallbacks.
Training keeps the ordinary differentiable PyTorch path unless an explicitly
selected native training backend owns the full forward/backward contract.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import (
    _init_state_batched,
    _ordered_to_device,
    _step_token_batched,
)
from .biren_runtime import validate_biren_forward_dtype
from .kernel_policy import current_kernel_policy, single_cuda_device_from_device_map
from .model_cache import (
    NativeRWKV7Cache,
    _NativeRWKV7LegacyCache,
    _cache_seen,
    _copy_native_cache_tuple,
    _maybe_legacy_native_cache,
    _native_cache_batch_size,
    _native_cache_tuple_or_none,
    _native_last_token_slice,
    _validate_native_cache_batch_size,
)
from .model_config import NativeRWKV7Config
from .model_fast_api import _NativeFastAPIMixin
from .model_generation import _NativeGenerationContractMixin
from .model_layers import (
    NativeRWKV7Attention,
    NativeRWKV7FFN,
    NativeRWKV7Layer,
    _LoRA,
)
from .model_prefill_graph import _NativePrefillGraphRunner
from .model_quantization import _NativeQuantizationMixin
from .model_runtime_policy import (
    FALSE_VALUES as _RUNTIME_FALSE_VALUES,
    bnb_int8_threshold_override as _runtime_bnb_int8_threshold_override,
    bnb_prefill_value_stride as _runtime_bnb_prefill_value_stride,
    bnb_skip_policy as _runtime_bnb_skip_policy,
    cuda_device_guard as _runtime_cuda_device_guard,
    native_model_backend_requested as _runtime_native_model_backend_requested,
    native_model_jit_enabled as _runtime_native_model_jit_enabled,
    native_prefill_external_quant_graph_enabled as _runtime_external_quant_enabled,
    native_prefill_graph_cache_size as _runtime_native_prefill_graph_cache_size,
    native_prefill_graph_enabled as _runtime_native_prefill_graph_enabled,
    native_prefill_graph_signature as _runtime_native_prefill_graph_signature,
)
from .model_runtime import _NativeRuntimeMixin
from .model_speculative import _NativeSpeculativeGenerationMixin
from .model_backbone import (
    NativeRWKV7Model,
    _blend_native_recurrent_state,
    _step_token_batched_with_hidden,
    _validate_native_attention_mask,
    _validate_native_output_attentions,
)

# Keep the historical public module identity used by Auto* metadata and
# save_pretrained remote-code discovery while implementations live in focused
# sibling modules.
NativeRWKV7Cache.__module__ = __name__
NativeRWKV7Config.__module__ = __name__
NativeRWKV7Attention.__module__ = __name__
NativeRWKV7FFN.__module__ = __name__
NativeRWKV7Layer.__module__ = __name__
NativeRWKV7Model.__module__ = __name__
_NativePrefillGraphRunner.__module__ = __name__
_LoRA.__module__ = __name__

# Some Transformers releases only copy files directly referenced by the
# remote-code entrypoint. Keep static discovery edges to the dependencies
# reached through native.py/native_jit.py/native_quant_mm*.py without importing
# optional Triton kernels at runtime.
if False:  # pragma: no cover
    from .biren_runtime import enable_biren as _native_biren_runtime_dependency_sentinel
    from .extension_build import cuda_extension_build_environment as _native_extension_build_dependency_sentinel
    from .musa_build import load_musa_inline as _native_musa_build_dependency_sentinel
    from .musa_fused import try_musa_attn_shift_mix as _native_musa_fused_dependency_sentinel
    from .musa_wkv import musa_wkv as _native_musa_wkv_dependency_sentinel
    from .musa_wkv_source import WKV7_MUSA_HEADER as _native_musa_wkv_source_dependency_sentinel
    from .metax_runtime import enable_metax as _native_metax_runtime_dependency_sentinel
    from .ascend_graph_runtime import AscendGraphRunner as _native_ascend_graph_dependency_sentinel
    from .ascend_quant import AscendW8A16Linear as _native_ascend_w8_dependency_sentinel
    from .ascend_quant_w4 import AscendWeightOnlyLinear as _native_ascend_w4_dependency_sentinel
    from .ascend_runtime import enable_ascend as _native_ascend_runtime_dependency_sentinel
    from .ascend_w4_cle import calibrate_sqrelu_value_w4 as _native_ascend_w4_cle_dependency_sentinel
    from .ada_lora import ada_wagv_lora as _native_ada_lora_dependency_sentinel
    from .ada_sparse_ffn import ada_linear as _native_ada_sparse_ffn_dependency_sentinel
    from .blackwell_norm_mix import blackwell_ffn_add_norm_mix as _native_sm120_norm_mix_dependency_sentinel
    from .dplr_prefill import dplr_chunk_scan as _native_dplr_dependency_sentinel
    from .dplr_prefill_triton import dplr_chunk_scan_triton as _native_dplr_triton_dependency_sentinel
    from .fused_attention_projection import fused_rkv_wag_projection as _native_fused_attn_projection_dependency_sentinel
    from .fused_decode_norm_mix import fused_attn_norm_mix6_decode as _native_fused_decode_norm_mix_dependency_sentinel
    from .fused_elementwise import fused_relu_square as _native_fused_elementwise_dependency_sentinel
    from .fused_ffn import fused_sequence_ffn as _native_fused_ffn_dependency_sentinel
    from .fused_lora import fused_wag_lora as _native_fused_lora_dependency_sentinel
    from .fused_output import fused_attn_output_prepare as _native_fused_output_dependency_sentinel
    from .fused_prefill import fused_prefill_state_prep as _native_fused_prefill_dependency_sentinel
    from .fused_recurrent_update import fused_recurrent_update as _native_fused_recurrent_dependency_sentinel
    from .fused_time_mix import fused_attn_shift_mix as _native_fused_time_mix_dependency_sentinel
    from .kernel_policy import current_kernel_policy as _native_kernel_policy_dependency_sentinel
    from .native_quant_bnb8 import fused_bnb8_relu_square_quant as _native_bnb8_dependency_sentinel
    from .native_quant_mm4 import quantize_model_mm4 as _native_mm4_dependency_sentinel
    from .native_quant_mm8 import quantize_model_mm8 as _native_mm8_dependency_sentinel
    from .native_quant_fp8 import quantize_model_fp8 as _native_fp8_dependency_sentinel
    from .native_quant_policy import normalize_native_mm_policy as _native_quant_policy_dependency_sentinel
    from .native_wkv_fp16 import native_fp16_sequence as _native_wkv_fp16_dependency_sentinel  # noqa: F401
    from .native_jit_linear import graph_linear_operand as _native_jit_linear_dependency_sentinel
    from .native_jit_bnb8 import _bnb8_direct_linear as _native_jit_bnb8_dependency_sentinel
    from .native_jit_dense_step import block_step as _native_jit_dense_step_dependency_sentinel
    from .native_jit_decode import step_batched as _native_jit_decode_dependency_sentinel
    from .native_jit_graph_dispatch import _native_graph_rkv_policy as _native_jit_graph_dispatch_dependency_sentinel
    from .native_jit_packing import extract_dense_packs as _native_jit_packing_dependency_sentinel
    from .native_jit_prefill import _prefill_current_device as _native_jit_prefill_dependency_sentinel
    from .native_jit_prefill_policy import model_shape_selected as _native_jit_prefill_policy_dependency_sentinel
    from .native_jit_prefill_runtime_policy import _native_prefill_fused_scan_enabled as _native_jit_prefill_runtime_policy_dependency_sentinel
    from .native_jit_recurrent import _recurrent_update_batched as _native_jit_recurrent_dependency_sentinel
    from .self_chunk_A_fwd import chunk_dplr_fwd_intra as _native_self_chunk_a_dependency_sentinel
    from .self_chunk_cumsum import chunk_rwkv6_fwd_cumsum as _native_self_chunk_cumsum_dependency_sentinel
    from .self_chunk_h_fwd import chunk_dplr_fwd_h as _native_self_chunk_h_dependency_sentinel
    from .self_chunk_o_fwd import chunk_dplr_fwd_o as _native_self_chunk_o_dependency_sentinel
    from .self_chunk_rwkv7 import self_chunk_rwkv7 as _native_self_chunk_dependency_sentinel
    from .self_chunk_utils import check_shared_mem as _native_self_chunk_utils_dependency_sentinel
    from .self_chunk_wy_fwd import prepare_wy_repr_fwd as _native_self_chunk_wy_dependency_sentinel
    from .sm70_linear import sm70_linear as _native_sm70_linear_dependency_sentinel
    from .sm70_quant import w4_linear as _native_sm70_quant_dependency_sentinel
    from .sm70_wagv import sm70_wagv_lora as _native_sm70_wagv_dependency_sentinel
    from .sm120_compiled_ffn import sm120_compiled_ffn as _native_sm120_compiled_ffn_dependency_sentinel

_FALSE_VALUES = _RUNTIME_FALSE_VALUES


def _cuda_device_guard(device):
    return _runtime_cuda_device_guard(device, torch_module=torch)


def _bnb_skip_policy(
    policy: str | None = None,
    *,
    policy_device: int | str | None = None,
    hardware_policy: bool = True,
) -> str:
    return _runtime_bnb_skip_policy(
        policy,
        policy_device=policy_device,
        hardware_policy=hardware_policy,
        kernel_policy_fn=current_kernel_policy,
    )


def _bnb_prefill_value_stride() -> int:
    return _runtime_bnb_prefill_value_stride()


def _bnb_int8_threshold_override(
    *,
    policy_device: int | str | None = None,
    hardware_policy: bool = True,
) -> float | None:
    return _runtime_bnb_int8_threshold_override(
        policy_device=policy_device,
        hardware_policy=hardware_policy,
        kernel_policy_fn=current_kernel_policy,
    )

try:
    from .native_jit import extract as _native_jit_extract
    from .native_jit import extract_graph as _native_graph_extract
    from .native_jit import prefill as _native_jit_prefill
    from .native_jit import step_batched as _native_jit_step_batched
except Exception:  # pragma: no cover - optional native acceleration
    _native_jit_extract = None
    _native_graph_extract = None
    _native_jit_prefill = None
    _native_jit_step_batched = None

try:
    from .native_graph_runtime import (
        NativeGraphRunner as _NativeGraphRunner,
        native_graph_available as _native_graph_available,
        native_graph_cache_size as _native_graph_cache_size,
        native_graph_runtime_signature as _native_graph_runtime_signature,
        native_graph_stats_template as _native_graph_stats_template,
    )
except Exception:  # pragma: no cover - optional CUDA graph acceleration
    _NativeGraphRunner = None
    _native_graph_available = lambda: False
    _native_graph_cache_size = lambda: 8
    _native_graph_runtime_signature = lambda: ()
    _native_graph_stats_template = lambda: {"requests": 0, "hits": 0, "misses": 0, "evictions": 0}

try:
    from .ascend_graph_runtime import (
        AscendGraphRunner as _AscendGraphRunner,
        ascend_graph_available as _ascend_graph_available,
        ascend_graph_cache_size as _ascend_graph_cache_size,
        ascend_graph_module_signature as _ascend_graph_module_signature,
        ascend_graph_runtime_signature as _ascend_graph_runtime_signature,
    )
except Exception:  # pragma: no cover - optional torch-npu graph acceleration
    _AscendGraphRunner = None
    _ascend_graph_available = lambda: False
    _ascend_graph_cache_size = lambda: 3
    _ascend_graph_module_signature = lambda owner: ()
    _ascend_graph_runtime_signature = lambda: ()


def _native_model_jit_enabled() -> bool:
    return _runtime_native_model_jit_enabled()


def _native_model_backend_requested() -> str:
    return _runtime_native_model_backend_requested(
        jit_enabled_fn=_native_model_jit_enabled,
    )


def _native_prefill_graph_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    device: int | str | torch.device | None = None,
) -> bool:
    return _runtime_native_prefill_graph_enabled(
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
        device,
        native_jit_prefill_available=_native_jit_prefill is not None,
        torch_module=torch,
        kernel_policy_fn=current_kernel_policy,
    )


def _native_prefill_external_quant_graph_enabled(
    device: int | str | torch.device | None = None,
) -> bool:
    return _runtime_external_quant_enabled(
        device,
        torch_module=torch,
        kernel_policy_fn=current_kernel_policy,
    )


def _native_prefill_graph_cache_size(
    device: int | str | torch.device | None = None,
) -> int:
    return _runtime_native_prefill_graph_cache_size(
        device,
        torch_module=torch,
        kernel_policy_fn=current_kernel_policy,
    )


def _native_prefill_graph_signature() -> tuple[tuple[str, str], ...]:
    return _runtime_native_prefill_graph_signature()


def _zero3_pad_native_training_batch(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    pad_token_id: int,
):
    """Pad rank-local ZeRO-3 training inputs to one global sequence length.

    Native recurrence invokes child modules once per token. ZeRO-3 installs
    parameter-gather hooks on those children, so every rank must execute the
    same number of hooks even when its locally padded batch is shorter.
    """

    try:
        first_param = next(model.parameters())
    except StopIteration:
        return input_ids, inputs_embeds, attention_mask, labels, int(labels.shape[1])
    is_zero3 = hasattr(first_param, "ds_id") and hasattr(first_param, "ds_status")
    distributed = torch.distributed
    if not (
        is_zero3
        and distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    ):
        return input_ids, inputs_embeds, attention_mask, labels, int(labels.shape[1])

    local_seq_len = int(labels.shape[1])
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    global_length = torch.tensor(local_seq_len, device=device, dtype=torch.int64)
    distributed.all_reduce(global_length, op=distributed.ReduceOp.MAX)
    global_seq_len = int(global_length.item())
    pad_len = global_seq_len - local_seq_len
    if pad_len <= 0:
        return input_ids, inputs_embeds, attention_mask, labels, local_seq_len

    if input_ids is not None:
        input_ids = F.pad(input_ids, (0, pad_len), value=int(pad_token_id))
    if inputs_embeds is not None:
        inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len), value=0.0)
    if attention_mask is None:
        attention_mask = torch.ones(
            labels.shape[0],
            local_seq_len,
            device=device,
            dtype=torch.long,
        )
    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
    labels = F.pad(labels, (0, pad_len), value=-100)
    return input_ids, inputs_embeds, attention_mask, labels, local_seq_len


def _resolve_native_logits_to_keep(logits_to_keep=None, num_logits_to_keep=None):
    if logits_to_keep is None:
        return num_logits_to_keep
    if num_logits_to_keep is None:
        return logits_to_keep
    if isinstance(logits_to_keep, torch.Tensor) or isinstance(num_logits_to_keep, torch.Tensor):
        try:
            left = torch.as_tensor(logits_to_keep).detach().cpu()
            right = torch.as_tensor(num_logits_to_keep).detach().cpu()
            same = torch.equal(left, right)
        except Exception:
            same = False
    else:
        same = int(logits_to_keep) == int(num_logits_to_keep)
    if not same:
        raise ValueError("logits_to_keep and num_logits_to_keep must match when both are provided")
    return logits_to_keep


def _slice_native_logits(logits: torch.Tensor, logits_to_keep):
    if logits_to_keep is None:
        return logits
    if isinstance(logits_to_keep, torch.Tensor):
        if logits_to_keep.dim() == 0:
            logits_to_keep = int(logits_to_keep.detach().cpu().item())
        else:
            positions = logits_to_keep.to(device=logits.device, dtype=torch.long)
            return logits.index_select(1, positions)
    keep = int(logits_to_keep)
    if keep <= 0:
        return logits
    return logits[:, -min(keep, int(logits.shape[1])) :, :]


def _native_tensor_parallel_active(model) -> bool:
    """Detect Transformers TP without requiring a concrete model instance."""

    return int(getattr(model, "_tp_size", 1) or 1) > 1


class NativeRWKV7ForCausalLM(
    _NativeRuntimeMixin,
    _NativeQuantizationMixin,
    _NativeSpeculativeGenerationMixin,
    _NativeGenerationContractMixin,
    _NativeFastAPIMixin,
    PreTrainedModel,
    GenerationMixin,
):
    """Experimental batched native PyTorch CausalLM for converted RWKV-7 weights."""

    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["NativeRWKV7Layer"]
    # A recurrent cache is sharded alongside the layers under a pipeline
    # device map. Accelerate must not collapse it back onto the input device.
    _skip_keys_device_placement = ["past_key_values"]
    supports_gradient_checkpointing = True
    # Transformers >=5 expects dict-like _tied_weights_keys; RWKV-7 ties nothing.
    _tied_weights_keys = {}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _rwkv7_bnb_skip_modules = ["lm_head", r".*_lora\.lora\.[02]"]
    _rwkv7_bnb_policy_extra_skips = {
        "memory": [],
        "output_hot": [r".*attn\.o_proj"],
        "decode_rk": [r".*attn\.(r_proj|k_proj)"],
        "decode_hot": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)"],
        "prefill_hot": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)", r".*ffn\.key"],
        "dense": [r".*attn\.(r_proj|k_proj|v_proj|o_proj)", r".*ffn\.(key|value)"],
    }

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RWKV recurrent state is not a Transformer KV cache.  Returning False
        # keeps GenerationMixin from pre-allocating DynamicCache for this model
        # family, while forward still treats an empty DynamicCache as no cache
        # for compatibility with older/newer Transformers variants.
        return False

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.model = NativeRWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()
        # ``post_init`` composes plans from child models but Transformers 5.x
        # does not retain a top-level class plan while doing so.
        self._tp_plan["lm_head"] = "colwise_gather_output"

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        self._clear_native_jit_pack_cache()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder
        self._clear_native_jit_pack_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self._clear_native_jit_pack_cache()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, *args, **kwargs):
        """RWKV checkpoints use the fixed official trie vocabulary."""

        if new_num_tokens is None or int(new_num_tokens) == int(self.config.vocab_size):
            return self.get_input_embeddings()
        raise NotImplementedError(
            "RWKV-7 uses the fixed official trie vocabulary; changing vocab size "
            "with resize_token_embeddings is not supported by this adapter."
        )

    def _rwkv7_has_multi_cuda_device_map(self) -> bool:
        """Detect an Accelerate model split across multiple CUDA devices.

        Native prefill/decode packs assume every layer shares one device.
        Accelerate's ordinary module hooks can move eager inputs and recurrent
        state across a pipeline split, so all packed/graph routes must fail
        closed while that split is active.
        """

        cacheable = isinstance(self, NativeRWKV7ForCausalLM)
        if cacheable:
            cached = getattr(self, "_rwkv7_multi_cuda_device_map_cache", None)
            if cached is not None:
                return bool(cached)

        devices: set[tuple[str, int | None]] = set()
        device_map = getattr(self, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            for value in device_map.values():
                if isinstance(value, int):
                    devices.add(("cuda", int(value)))
                    continue
                if not isinstance(value, str) or value == "disk":
                    continue
                device = torch.device(value)
                if device.type == "cuda":
                    devices.add(("cuda", device.index))
            if len(devices) > 1:
                if cacheable:
                    self._rwkv7_multi_cuda_device_map_cache = True
                return True
            # Accelerate's recorded map is authoritative for a dispatched
            # model. Avoid a full parameter walk on every decode token.
            if cacheable:
                self._rwkv7_multi_cuda_device_map_cache = False
            return False

        parameter_devices = {
            (parameter.device.type, parameter.device.index)
            for parameter in self.parameters()
            if parameter.device.type == "cuda"
        }
        result = len(parameter_devices) > 1
        if cacheable:
            self._rwkv7_multi_cuda_device_map_cache = result
        return result

    def _rwkv7_has_tensor_parallel(self) -> bool:
        """Return whether Transformers distributed this instance over a TP mesh."""

        return _native_tensor_parallel_active(self)


    def _run(
        self,
        token_ids: torch.Tensor | None,
        state,
        xpa,
        xpf,
        v_first,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_jit: bool = False,
        collect_all: bool = False,
        output_hidden_states: bool = False,
    ):
        """Sequentially advance over token ids or embeddings.

        The eager fallback is sequential over time but vectorized over batch.
        Optimized inference prefill and decode are selected before this helper.

        When ``collect_all`` is enabled, returns per-token logits shaped
        ``[batch, seq, vocab]``. This keeps the FLA-free native path compatible
        with standard CausalLM training losses without changing the optimized
        decode path, which only materializes the final token logits.
        """
        if token_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7ForCausalLM._run requires token_ids or inputs_embeds")
        if token_ids is not None and token_ids.dim() != 2:
            raise ValueError("NativeRWKV7ForCausalLM._run expects token ids shaped [batch, seq]")
        if inputs_embeds is not None and inputs_embeds.dim() != 3:
            raise ValueError("NativeRWKV7ForCausalLM._run expects inputs_embeds shaped [batch, seq, hidden]")
        seq_len = int(inputs_embeds.shape[1] if inputs_embeds is not None else token_ids.shape[1])
        batch_size = int(inputs_embeds.shape[0] if inputs_embeds is not None else token_ids.shape[0])
        base = self.model
        x = None
        packs = self._native_jit_packs() if use_jit and not output_hidden_states and attention_mask is None else None
        backend = "native_jit" if packs is not None else "eager"
        all_logits = [] if collect_all else None
        all_hidden = [] if collect_all or output_hidden_states else None
        hidden_buckets = [[] for _ in range(self.config.num_hidden_layers + 1)] if output_hidden_states else None
        hidden_size = int(self.config.hidden_size)
        dtype = inputs_embeds.dtype if inputs_embeds is not None else base.embeddings.weight.dtype
        device = inputs_embeds.device if inputs_embeds is not None else token_ids.device
        last_normed = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        last_layer_hiddens = (
            [torch.zeros(batch_size, hidden_size, device=device, dtype=dtype) for _ in range(self.config.num_hidden_layers + 1)]
            if hidden_buckets is not None
            else None
        )
        for t in range(seq_len):
            x = inputs_embeds[:, t] if inputs_embeds is not None else base.embeddings(token_ids[:, t])
            token_mask = attention_mask[:, t] if attention_mask is not None else None
            if token_mask is not None:
                old_state, old_xpa, old_xpf, old_v_first = list(state), list(xpa), list(xpf), v_first
            if hidden_buckets is not None:
                emb_hidden = x
                if token_mask is not None:
                    emb_hidden = torch.where(token_mask.view(batch_size, 1).to(x.device), emb_hidden, last_layer_hiddens[0])
                hidden_buckets[0].append(emb_hidden)
            if packs is not None:
                x, state, xpa, xpf, v_first = _native_jit_step_batched(self, x, state, xpa, xpf, v_first, packs)
            elif hidden_buckets is not None:
                x, state, xpa, xpf, v_first, layer_hiddens = _step_token_batched_with_hidden(
                    self, x, state, xpa, xpf, v_first
                )
            else:
                x, state, xpa, xpf, v_first = _step_token_batched(self, x, state, xpa, xpf, v_first)
            normed = base.norm(x)
            if token_mask is not None:
                state, xpa, xpf, v_first = _blend_native_recurrent_state(
                    token_mask, old_state, state, old_xpa, xpa, old_xpf, xpf, old_v_first, v_first
                )
                mask_h = token_mask.view(batch_size, 1).to(normed.device)
                normed = torch.where(mask_h, normed, last_normed)
                if hidden_buckets is not None:
                    layer_hiddens = [
                        torch.where(mask_h.to(layer_hidden.device), layer_hidden, last_layer_hiddens[layer_idx + 1])
                        for layer_idx, layer_hidden in enumerate(layer_hiddens)
                    ]
            if hidden_buckets is not None:
                for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1):
                    hidden_buckets[layer_idx].append(
                        normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    )
                last_layer_hiddens = [emb_hidden] + [
                    normed if layer_idx == self.config.num_hidden_layers else layer_hidden
                    for layer_idx, layer_hidden in enumerate(layer_hiddens, start=1)
                ]
            if all_hidden is not None:
                all_hidden.append(normed)
            if all_logits is not None:
                all_logits.append(self.lm_head(normed))
            last_normed = normed
        if x is None:
            raise ValueError("NativeRWKV7ForCausalLM requires at least one token")
        if use_jit:
            self._rwkv7_native_model_last_decode_backend = backend
        if all_logits is not None:
            logits = torch.stack(all_logits, dim=1)
        else:
            logits = self.lm_head(normed).view(batch_size, 1, -1)
        last_hidden_state = torch.stack(all_hidden, dim=1) if all_hidden is not None else normed.view(batch_size, 1, -1)
        hidden_states = None
        if hidden_buckets is not None:
            hidden_states = tuple(torch.stack(bucket, dim=1) for bucket in hidden_buckets)
        # Accelerate normally returns model-parallel outputs to the input
        # device. Do that copy here with an explicit source-stream dependency;
        # otherwise the destination stream can race the last pipeline stage.
        logits = _ordered_to_device(logits, device)
        last_hidden_state = _ordered_to_device(last_hidden_state, device)
        if hidden_states is not None:
            hidden_states = tuple(_ordered_to_device(value, device) for value in hidden_states)
        return logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states

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
        labels: torch.LongTensor | None = None,
        logits_to_keep=None,
        num_logits_to_keep=None,
        position_ids=None,
        cache_position=None,
        token_type_ids=None,
        head_mask=None,
        return_legacy_cache: bool | None = None,
        **kwargs,
    ):
        prefill_graph_continuation = bool(
            kwargs.pop("_rwkv7_prefill_graph_continuation", False)
        )
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
                labels=labels,
                logits_to_keep=logits_to_keep,
                num_logits_to_keep=num_logits_to_keep,
                position_ids=position_ids,
                cache_position=cache_position,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                return_legacy_cache=return_legacy_cache,
                **kwargs,
            )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("NativeRWKV7ForCausalLM accepts either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("NativeRWKV7ForCausalLM requires input_ids or inputs_embeds")
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.view(1, -1)
        if input_ids is not None and input_ids.dim() != 2:
            raise ValueError("Experimental NativeRWKV7ForCausalLM expects input_ids shaped [batch, seq]")
        if inputs_embeds is not None:
            if inputs_embeds.dim() != 3:
                raise ValueError("NativeRWKV7ForCausalLM expects inputs_embeds shaped [batch, seq, hidden]")
            if int(inputs_embeds.shape[-1]) != int(self.config.hidden_size):
                raise ValueError("NativeRWKV7ForCausalLM inputs_embeds last dimension must match hidden_size")
        batch_size = int(input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0])
        seq_len = int(input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1])
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("NativeRWKV7ForCausalLM requires a non-empty batch and sequence")
        native_cache = _native_cache_tuple_or_none(past_key_values)
        _validate_native_cache_batch_size(native_cache, batch_size)
        _validate_native_output_attentions(output_attentions, self.config)
        if return_dict is None:
            return_dict = bool(getattr(self.config, "return_dict", True))
        base = self.model
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        dtype = inputs_embeds.dtype if inputs_embeds is not None else base.embeddings.weight.dtype
        validate_biren_forward_dtype(
            dtype,
            input_device=device,
            model_device=base.embeddings.weight.device,
            model_dtype=base.embeddings.weight.dtype,
        )
        native_attention_mask = _validate_native_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            device=device,
            allow_trailing=native_cache is not None,
        )
        output_hidden_states = bool(
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.view(1, -1)
            if tuple(labels.shape[:2]) != (batch_size, seq_len):
                raise ValueError("NativeRWKV7ForCausalLM labels must have the same shape as inputs")
            if native_cache is not None:
                raise ValueError("NativeRWKV7ForCausalLM does not support labels with past_key_values")
            input_ids, inputs_embeds, attention_mask, labels, local_seq_len = _zero3_pad_native_training_batch(
                self,
                input_ids,
                inputs_embeds,
                attention_mask,
                labels,
                pad_token_id=int(getattr(self.config, "pad_token_id", 0) or 0),
            )
            seq_len = int(labels.shape[1])
            native_attention_mask = _validate_native_attention_mask(
                attention_mask,
                batch_size,
                seq_len,
                device=device,
            )
            state, xpa, xpf, v_first = _init_state_batched(self, batch_size, device, dtype)
            logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states = self._run(
                input_ids,
                state,
                xpa,
                xpf,
                v_first,
                inputs_embeds=inputs_embeds if input_ids is None else None,
                attention_mask=native_attention_mask,
                use_jit=False,
                collect_all=True,
                output_hidden_states=output_hidden_states,
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_logits.numel() == 0 or not bool((shift_labels != -100).any().detach().cpu().item()):
                loss = logits.float().sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]).float(),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            if seq_len != local_seq_len:
                logits = logits[:, :local_seq_len]
                if hidden_states is not None:
                    hidden_states = tuple(value[:, :local_seq_len] for value in hidden_states)
            new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=local_seq_len) if use_cache else None
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                values = (loss, logits, new_cache, hidden_states)
                return tuple(v for v in values if v is not None)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=new_cache,
                hidden_states=hidden_states,
            )

        logits_to_keep = _resolve_native_logits_to_keep(logits_to_keep, num_logits_to_keep)
        if native_cache is None and self._native_prefill_can_run(
            input_ids,
            attention_mask=native_attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        ):
            logits, new_cache = self._native_prefill(
                input_ids,
                logits_to_keep=logits_to_keep,
                seen_tokens=seq_len,
                graph_continuation=prefill_graph_continuation,
            )
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if native_cache is not None and self._native_prefill_can_run(
            input_ids,
            attention_mask=native_attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        ):
            logits, new_cache = self._native_prefill(
                input_ids,
                logits_to_keep=logits_to_keep,
                seen_tokens=_cache_seen(past_key_values) + seq_len,
                initial_cache=native_cache,
                graph_continuation=prefill_graph_continuation,
            )
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if (
            native_cache is not None
            and use_cache
            and isinstance(past_key_values, NativeRWKV7Cache)
            and self._native_graph_can_run(
                input_ids,
                past_key_values,
                attention_mask=native_attention_mask,
                output_hidden_states=output_hidden_states,
            )
        ):
            runner = self._native_graph_runner(batch_size)
            logits = runner.replay(input_ids, past_key_values)
            past_key_values.seen_tokens = _cache_seen(past_key_values) + 1
            self._rwkv7_native_model_last_decode_backend = "native_graph"
            logits = _slice_native_logits(logits, logits_to_keep)
            new_cache = _maybe_legacy_native_cache(past_key_values, return_legacy_cache)
            if not return_dict:
                return logits, new_cache
            return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)
        if native_cache is None:
            state, xpa, xpf, v_first = _init_state_batched(self, batch_size, device, dtype)
            toks = input_ids
            use_jit = False
            seen = seq_len
            collect_all = True  # full forward -> all-token logits [B, seq, vocab] (HF CausalLM semantics; DPO/eval need per-token logprobs)
        else:
            state, xpa, xpf, v_first = _copy_native_cache_tuple(native_cache)
            toks = input_ids
            use_jit = seq_len == 1
            seen = _cache_seen(past_key_values) + seq_len
            collect_all = seq_len > 1
        logits, state, xpa, xpf, v_first, last_hidden_state, hidden_states = self._run(
            toks,
            state,
            xpa,
            xpf,
            v_first,
            inputs_embeds=inputs_embeds if toks is None else None,
            attention_mask=native_attention_mask,
            use_jit=use_jit,
            collect_all=collect_all,
            output_hidden_states=output_hidden_states,
        )
        logits = _slice_native_logits(logits, logits_to_keep)
        new_cache = NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=seen) if use_cache else None
        new_cache = _maybe_legacy_native_cache(new_cache, return_legacy_cache)
        if not return_dict:
            values = (logits, new_cache, hidden_states)
            return tuple(v for v in values if v is not None)
        return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache, hidden_states=hidden_states)


try:  # pragma: no cover - exercised through save_pretrained/AutoModel smoke.
    NativeRWKV7Config.register_for_auto_class()
    NativeRWKV7Model.register_for_auto_class("AutoModel")
    NativeRWKV7ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
except Exception:
    pass
