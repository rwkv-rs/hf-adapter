"""Opt-in RWKV-LM train_temp CUDA training backend for HF RWKV-7 models.

The kernels under ``csrc/train_temp`` are vendored from RWKV-LM at the exact
commit recorded in that directory.  This module keeps them lazy and isolated:
normal HF inference and training do not compile or route through these ops.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


TRAIN_TEMP_SOURCE_COMMIT = "e6f74b63a06e08606d130043599d218209628bad"
TRAIN_TEMP_HEAD_SIZE = 64
TRAIN_TEMP_CHUNK_LEN = 16

_LOAD_LOCK = threading.Lock()
_LOADED = False
_LOAD_ERROR: BaseException | None = None
_L2WRAP_EXTENSION: Any | None = None

_COMMON_CUDA_FLAGS = [
    "-res-usage",
    "--use_fast_math",
    "-O3",
    "-Xptxas",
    "-O3",
    "--extra-device-vectorization",
]
_OP_SOURCES = {
    "rwkv7_cmix_bf16_v5": (
        "rwkv7_cmix_bf16_v5.cpp",
        "rwkv7_cmix_bf16_v5.cu",
    ),
    "rwkv7_tmix_mix6_bf16_v5": (
        "rwkv7_tmix_mix6_bf16_v5.cpp",
        "rwkv7_tmix_mix6_bf16_v5.cu",
    ),
    "rwkv7_tmix_kk_pre_bf16_v5": (
        "rwkv7_tmix_kk_pre_bf16_v5.cpp",
        "rwkv7_tmix_kk_pre_bf16_v5.cu",
    ),
    "rwkv7_tmix_lnx_rkvres_xg_bf16_v1": (
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp",
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu",
    ),
    "rwkv7_tmix_a_gate_bf16": (
        "rwkv7_tmix_a_gate_bf16.cpp",
        "rwkv7_tmix_a_gate_bf16.cu",
    ),
    "rwkv7_tmix_vres_gate_bf16_v3": (
        "rwkv7_tmix_vres_gate_bf16_v3.cpp",
        "rwkv7_tmix_vres_gate_bf16_v3.cu",
    ),
}


def _train_temp_checkpoint_backend() -> str:
    requested = (
        os.environ.get("RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND", "auto")
        .strip()
        .lower()
    )
    if requested not in {"auto", "deepspeed", "torch"}:
        raise ValueError(
            "RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND must be auto, deepspeed or torch"
        )
    if requested == "torch":
        return "torch_non_reentrant"
    if importlib.util.find_spec("deepspeed") is not None:
        return "deepspeed"
    if requested == "deepspeed":
        raise RuntimeError("DeepSpeed checkpointing was requested but is unavailable")
    return "torch_non_reentrant"


def _train_temp_checkpoint(function, *args):
    """Checkpoint one layer with the official train_temp backend when present."""

    backend = _train_temp_checkpoint_backend()
    if backend == "deepspeed":
        import deepspeed

        return deepspeed.checkpointing.checkpoint(function, *args)

    from torch.utils.checkpoint import checkpoint

    return checkpoint(function, *args, use_reentrant=False)


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "train_temp"


def _cuda_include_paths(
    cuda_home: str | os.PathLike[str], *, include_target: bool = False
) -> list[str]:
    """Resolve both conventional and pip-split CUDA development headers."""

    home = Path(cuda_home)
    candidates = [home / "include"]
    if include_target:
        candidates.append(home / "targets" / "x86_64-linux" / "include")
    toolkit_headers = any(
        (candidate / "cuda_runtime.h").is_file() for candidate in candidates
    )
    if not toolkit_headers:
        site_packages = Path(torch.__file__).resolve().parent.parent
        nvidia_packages = site_packages / "nvidia"
        if nvidia_packages.is_dir():
            candidates.extend(sorted(nvidia_packages.glob("*/include")))
    resolved: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if candidate.is_dir() and value not in resolved:
            resolved.append(value)
    return resolved


def _op_registered(namespace: str) -> bool:
    try:
        getattr(getattr(torch.ops, namespace), "forward")
    except (AttributeError, RuntimeError):
        return False
    return True


def _validate_runtime() -> None:
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError(
            "train_temp CUDA backend requires Linux with an available CUDA GPU"
        )
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        raise RuntimeError(
            "train_temp BF16 CUDA backend requires compute capability sm_80 or newer; "
            f"found sm_{major}{minor}"
        )


def _resolve_cuda_home(cpp_extension: Any) -> Path | None:
    candidates = [
        os.environ.get("CUDA_HOME"),
        f"/usr/local/cuda-{torch.version.cuda}" if torch.version.cuda else None,
        getattr(cpp_extension, "CUDA_HOME", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "bin" / "nvcc").is_file():
            cpp_extension.CUDA_HOME = str(path)
            return path
    return None


def load_train_temp_cuda_extension(*, verbose: bool | None = None) -> None:
    """Build and load the vendored train_temp operators once."""

    global _L2WRAP_EXTENSION, _LOADED, _LOAD_ERROR
    _validate_runtime()
    if _LOADED:
        return
    if _LOAD_ERROR is not None:
        raise RuntimeError(
            "train_temp CUDA extension previously failed to load"
        ) from _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED:
            return
        try:
            from torch.utils import cpp_extension

            cuda_home = _resolve_cuda_home(cpp_extension)
            if cuda_home is None:
                raise RuntimeError(
                    "train_temp CUDA JIT requires a local CUDA toolkit; set CUDA_HOME "
                    "to the toolkit matching the PyTorch CUDA build"
                )
            if verbose is None:
                verbose = os.environ.get("RWKV7_TRAIN_TEMP_VERBOSE", "0").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            root = _source_root()
            include_paths = _cuda_include_paths(cuda_home)
            cuda_cpp_include_paths = _cuda_include_paths(cuda_home, include_target=True)
            for namespace, filenames in _OP_SOURCES.items():
                if _op_registered(namespace):
                    continue
                cpp_extension.load(
                    name=f"rwkv7_hf_{namespace}",
                    sources=[str(root / filename) for filename in filenames],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                    extra_include_paths=include_paths,
                    is_python_module=False,
                    verbose=bool(verbose),
                )
            if not _op_registered("rwkv7_clampw_v3"):
                cpp_extension.load(
                    name="rwkv7_hf_clampw_v3",
                    sources=[
                        str(root / "rwkv7_clampw_v3_for_h100.cu"),
                        str(root / "rwkv7_clampw_v3.cpp"),
                    ],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=[
                        *_COMMON_CUDA_FLAGS,
                        f"-D_N_={TRAIN_TEMP_HEAD_SIZE}",
                        f"-D_CHUNK_LEN_={TRAIN_TEMP_CHUNK_LEN}",
                    ],
                    extra_include_paths=cuda_cpp_include_paths,
                    is_python_module=False,
                    verbose=bool(verbose),
                )
            _L2WRAP_EXTENSION = cpp_extension.load(
                name="rwkv7_hf_l2wrap_ce_bf16_v2",
                sources=[
                    str(root / "rwkv7_l2wrap_ce_bf16_v2.cpp"),
                    str(root / "rwkv7_l2wrap_ce_bf16_v2.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                extra_include_paths=cuda_cpp_include_paths,
                verbose=bool(verbose),
            )
            missing = [
                namespace for namespace in _OP_SOURCES if not _op_registered(namespace)
            ]
            if not _op_registered("rwkv7_clampw_v3"):
                missing.append("rwkv7_clampw_v3")
            if missing:
                raise RuntimeError(
                    f"train_temp extension did not register required ops: {missing}"
                )
            _LOADED = True
        except BaseException as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(
                f"train_temp CUDA extension failed to load: {exc}"
            ) from exc


def train_temp_cuda_available(*, build: bool = False) -> bool:
    """Return whether the backend is supported, optionally compiling it."""

    try:
        _validate_runtime()
        if build:
            load_train_temp_cuda_extension()
    except Exception:
        return False
    return True


class _Mix6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
        inputs = tuple(
            value.contiguous() for value in (x, x_r, x_w, x_k, x_v, x_a, x_g)
        )
        ctx.save_for_backward(*inputs)
        return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(*inputs))

    @staticmethod
    def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
        grads = tuple(
            value.contiguous()
            for value in (grad_r, grad_w, grad_k, grad_v, grad_a, grad_g)
        )
        return tuple(
            torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(*grads, *ctx.saved_tensors)
        )


class _KkPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, k_k, a, k_a):
        inputs = tuple(value.contiguous() for value in (k, k_k, a, k_a))
        outputs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
            *inputs, TRAIN_TEMP_HEAD_SIZE
        )
        ctx.save_for_backward(*inputs, outputs[3])
        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def backward(ctx, grad_k, grad_neg_kk, grad_kka):
        k, k_k, a, k_a, inv_d = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
                grad_k.contiguous(),
                grad_neg_kk.contiguous(),
                grad_kka.contiguous(),
                k,
                k_k,
                a,
                k_a,
                inv_d,
                TRAIN_TEMP_HEAD_SIZE,
            )
        )


class _LnxOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, r_k, weight, bias, g):
        inputs = tuple(
            value.contiguous() for value in (x, r, k, v, r_k, weight, bias, g)
        )
        outputs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(*inputs)
        ctx.save_for_backward(*inputs, outputs[1], outputs[2])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
                grad_output.contiguous(), x, r, k, v, r_k, weight, bias, g, mean, rstd
            )
        )


class _AGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a0, a12):
        inputs = a0.contiguous(), a12.contiguous()
        ctx.save_for_backward(*inputs)
        return torch.ops.rwkv7_tmix_a_gate_bf16.forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(
            torch.ops.rwkv7_tmix_a_gate_bf16.backward(
                grad_output.contiguous(), *ctx.saved_tensors
            )
        )


class _VResGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_first, v0, v12):
        inputs = tuple(value.contiguous() for value in (v, v_first, v0, v12))
        ctx.save_for_backward(*inputs)
        return torch.ops.rwkv7_tmix_vres_gate_bf16_v3.forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(
            torch.ops.rwkv7_tmix_vres_gate_bf16_v3.backward(
                grad_output.contiguous(), *ctx.saved_tensors
            )
        )


class _ClampW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        batch, tokens, heads, head_size = r.shape
        if head_size != TRAIN_TEMP_HEAD_SIZE or tokens % TRAIN_TEMP_CHUNK_LEN:
            raise ValueError(
                f"train_temp clampw requires head_size={TRAIN_TEMP_HEAD_SIZE} and "
                f"tokens divisible by {TRAIN_TEMP_CHUNK_LEN}; got {head_size=} {tokens=}"
            )
        inputs = tuple(value.contiguous() for value in (r, w, k, v, a, b))
        output = torch.empty_like(v)
        state = torch.empty(
            batch,
            heads,
            tokens // TRAIN_TEMP_CHUNK_LEN,
            head_size,
            head_size,
            dtype=torch.float32,
            device=w.device,
        )
        state_aux = torch.empty(
            batch, tokens, heads, head_size, dtype=torch.float32, device=w.device
        )
        torch.ops.rwkv7_clampw_v3.forward(*inputs, output, state, state_aux)
        ctx.save_for_backward(*inputs, state, state_aux)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        r, w, k, v, a, b, state, state_aux = ctx.saved_tensors
        grads = [torch.empty_like(value) for value in (r, w, k, v, a, b)]
        torch.ops.rwkv7_clampw_v3.backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_output.contiguous(),
            state,
            state_aux,
            *grads,
        )
        return tuple(grads)


class _CMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        inputs = tuple(
            value.contiguous() for value in (x, x_k, key_weight, value_weight)
        )
        output, mixed, activation = torch.ops.rwkv7_cmix_bf16_v5.forward(*inputs)
        ctx.save_for_backward(*inputs, mixed, activation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, x_k, key_weight, value_weight, mixed, activation = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_cmix_bf16_v5.backward(
                grad_output.contiguous(),
                x,
                x_k,
                key_weight,
                value_weight,
                mixed,
                activation,
            )
        )


class _L2WrapCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        assert _L2WRAP_EXTENSION is not None
        logits = logits.contiguous()
        targets = targets.contiguous()
        loss, lse, max_values, argmax = _L2WRAP_EXTENSION.forward(logits, targets)
        ctx.save_for_backward(logits, targets.reshape(-1), lse, max_values, argmax)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        assert _L2WRAP_EXTENSION is not None
        logits, targets, lse, max_values, argmax = ctx.saved_tensors
        grad_logits = _L2WRAP_EXTENSION.backward(
            grad_output.contiguous().float(), logits, targets, lse, max_values, argmax
        )
        return grad_logits, None


def train_temp_fused_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Run the exact train_temp fused FP32 CE plus L2Wrap gradient."""

    load_train_temp_cuda_extension()
    return _L2WrapCrossEntropy.apply(logits, targets)


def train_temp_causal_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Apply the fused train_temp loss to standard causal-LM logits and labels.

    Unlike :func:`train_temp_fused_cross_entropy`, this helper performs the
    next-token shift expected by Hugging Face causal-language-model batches.
    The current CUDA kernel accepts dense int64 labels only; padding and the
    usual ``-100`` ignore index are intentionally rejected by its contract.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [batch, tokens, vocab], got {tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"labels must have shape [batch, tokens], got {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must share batch/token dimensions; got "
            f"{tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal train_temp loss requires at least two tokens")
    if labels.dtype != torch.long:
        raise TypeError(f"labels must be torch.int64, got {labels.dtype}")
    if labels.device != logits.device:
        raise ValueError(
            "logits and labels must share a device, got "
            f"{logits.device} and {labels.device}"
        )
    if bool(torch.any((labels < 0) | (labels >= logits.shape[-1])).item()):
        raise ValueError(
            "train_temp CUDA loss requires dense labels in [0, vocab_size); "
            "-100 is unsupported"
        )
    return train_temp_fused_cross_entropy(
        logits[:, :-1].contiguous(), labels[:, 1:].contiguous()
    )


def _dense_mask_only(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is not None and not bool(torch.all(attention_mask != 0).item()):
        raise ValueError(
            "train_temp CUDA backend does not support padded or masked batches"
        )


def _train_temp_attention_forward(
    self, hidden_states, v_first, *, native_lora_math: bool
):
    """Run fused TMix while preserving each backend's LoRA activation ownership."""

    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] % TRAIN_TEMP_CHUNK_LEN
    ):
        raise ValueError(
            "train_temp CUDA backend requires BF16 and sequence length divisible by "
            f"{TRAIN_TEMP_CHUNK_LEN}; got {hidden_states.dtype} and T={hidden_states.shape[1]}"
        )
    xr, xw, xk, xv, xa, xg = _Mix6.apply(
        hidden_states,
        self.x_r.reshape(-1),
        self.x_w.reshape(-1),
        self.x_k.reshape(-1),
        self.x_v.reshape(-1),
        self.x_a.reshape(-1),
        self.x_g.reshape(-1),
    )
    r = self.r_proj(xr)
    w = (
        self.w_lora.lora[2](torch.tanh(self.w_lora.lora[0](xw)))
        if native_lora_math
        else self.w_lora(xw)
    )
    k = self.k_proj(xk)
    v = self.v_proj(xv)
    if self.layer_idx == 0:
        v_first = v
    else:
        v12 = F.linear(self.v_lora.lora[0](xv), self.v_lora.lora[2].weight, None)
        v = _VResGate.apply(v, v_first, self.v_lora.lora[2].bias, v12)
    a12 = F.linear(self.a_lora.lora[0](xa), self.a_lora.lora[2].weight, None)
    a = _AGate.apply(self.a_lora.lora[2].bias, a12)
    g = (
        self.g_lora.lora[2](torch.sigmoid(self.g_lora.lora[0](xg)))
        if native_lora_math
        else self.g_lora(xg)
    )
    k, neg_kk, kka = _KkPre.apply(k, self.k_k.reshape(-1), a, self.k_a.reshape(-1))
    batch, tokens, _ = r.shape
    heads = int(self.num_heads)
    head_dim = int(self.head_dim)
    head_v_dim = int(getattr(self, "head_v_dim", head_dim))
    if head_dim != TRAIN_TEMP_HEAD_SIZE or head_v_dim != TRAIN_TEMP_HEAD_SIZE:
        raise ValueError(
            "train_temp CUDA backend currently requires K/V head dimensions of 64"
        )
    values = _ClampW.apply(
        r.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        w.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        k.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        v.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        neg_kk.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        kka.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
    ).reshape(batch, tokens, -1)
    values = _LnxOutput.apply(
        values,
        r,
        k,
        v,
        self.r_k,
        self.g_norm.weight,
        self.g_norm.bias,
        g,
    )
    return self.o_proj(values), v_first


def native_train_temp_attention_forward(self, hidden_states, v_first):
    """Run one NativeRWKV7Attention over a complete BF16 sequence."""

    return _train_temp_attention_forward(
        self, hidden_states, v_first, native_lora_math=True
    )


def native_train_temp_ffn_forward(self, hidden_states):
    """Run one NativeRWKV7FFN over a complete BF16 sequence."""

    return _CMix.apply(
        hidden_states,
        self.x_k.reshape(-1),
        self.key.weight,
        self.value.weight,
    )


def native_train_temp_layer_forward(self, hidden_states, v_first):
    """Run one NativeRWKV7Layer with the same checkpoint boundary as train_temp."""

    residual = self.pre_norm(hidden_states) if hasattr(self, "pre_norm") else hidden_states
    attn_output, v_first = self.attn(self.attn_norm(residual), v_first)
    hidden_states = residual + attn_output
    hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
    return hidden_states, v_first


def _attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
    output_attentions=False,
    v_first=None,
    cu_seqlens=None,
    **kwargs,
):
    _dense_mask_only(attention_mask)
    if (
        past_key_values is not None
        or use_cache
        or cu_seqlens is not None
        or output_attentions
    ):
        raise ValueError("train_temp CUDA backend is a dense no-cache training path")
    output, v_first = _train_temp_attention_forward(
        self, hidden_states, v_first, native_lora_math=False
    )
    return output, None, past_key_values, v_first


def _ffn_forward(self, x, attention_mask=None, state=None, cu_seqlens=None, **kwargs):
    _dense_mask_only(attention_mask)
    if state is not None or cu_seqlens is not None:
        raise ValueError("train_temp CUDA backend is a dense no-cache training path")
    return native_train_temp_ffn_forward(self, x), state


def native_train_temp_causal_lm_forward(
    model,
    input_ids=None,
    *,
    attention_mask=None,
    inputs_embeds=None,
    past_key_values=None,
    use_cache=None,
    output_hidden_states=None,
    output_attentions=None,
    return_dict=None,
    labels=None,
    logits_to_keep=None,
    num_logits_to_keep=None,
    **kwargs,
):
    """Canonical NativeRWKV7ForCausalLM full-sequence train_temp forward."""

    from transformers.modeling_outputs import CausalLMOutputWithPast

    _dense_mask_only(attention_mask)
    if past_key_values is not None or bool(use_cache):
        raise ValueError("train_temp CUDA backend is a dense no-cache training path")
    if bool(output_attentions) or bool(output_hidden_states):
        raise ValueError(
            "train_temp CUDA backend does not emit attention or hidden-state histories"
        )
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "train_temp CUDA backend accepts input_ids or inputs_embeds, not both"
        )
    if input_ids is None and inputs_embeds is None:
        raise ValueError("train_temp CUDA backend requires input_ids or inputs_embeds")
    if input_ids is not None:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError(
                "train_temp CUDA backend expects input_ids shaped [batch, tokens]"
            )
        hidden_states = model.model.embeddings(input_ids)
    else:
        if inputs_embeds.dim() != 3:
            raise ValueError(
                "train_temp CUDA backend expects inputs_embeds shaped [batch, tokens, hidden]"
            )
        hidden_states = inputs_embeds
    if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.bfloat16:
        raise ValueError("train_temp CUDA backend requires BF16 tensors on CUDA")
    if int(hidden_states.shape[1]) % TRAIN_TEMP_CHUNK_LEN:
        raise ValueError(
            f"train_temp CUDA backend requires sequence length divisible by {TRAIN_TEMP_CHUNK_LEN}"
        )

    gradient_checkpointing = bool(
        model.training
        and (
            getattr(model, "gradient_checkpointing", False)
            or getattr(model.model, "gradient_checkpointing", False)
        )
    )
    # Layer zero ignores the incoming value and emits the full V-first tensor.
    # A scalar placeholder avoids allocating a second BxTxC activation before
    # the first checkpointed layer runs.
    v_first = hidden_states.new_zeros(1)

    for layer in model.model.layers:
        if gradient_checkpointing:
            hidden_states, v_first = _train_temp_checkpoint(
                layer,
                hidden_states,
                v_first,
            )
        else:
            hidden_states, v_first = layer(hidden_states, v_first)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    requested_keep = (
        logits_to_keep if logits_to_keep is not None else num_logits_to_keep
    )
    loss = None
    if labels is not None:
        loss = train_temp_causal_cross_entropy(logits, labels)
    elif requested_keep is not None:
        keep = (
            int(requested_keep.detach().cpu().item())
            if isinstance(requested_keep, torch.Tensor)
            else int(requested_keep)
        )
        if keep > 0:
            logits = logits[:, -min(keep, int(logits.shape[1])) :, :]
    if return_dict is None:
        return_dict = bool(getattr(model.config, "return_dict", True))
    if not return_dict:
        values = (loss, logits) if loss is not None else (logits,)
        return values
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


def enable_train_temp_cuda_backend(model) -> dict[str, Any]:
    """Enable official train_temp kernels on the canonical Native or FLA reference model."""

    load_train_temp_cuda_extension()
    config_model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    native_model = (
        type(model).__name__ == "NativeRWKV7ForCausalLM"
        or config_model_type == "rwkv7_native"
    )
    if native_model:
        modules = tuple(model.modules())
        attention_modules = tuple(
            module
            for module in modules
            if type(module).__name__ == "NativeRWKV7Attention"
        )
        ffn_modules = tuple(
            module for module in modules if type(module).__name__ == "NativeRWKV7FFN"
        )
        layer_modules = tuple(
            module for module in modules if type(module).__name__ == "NativeRWKV7Layer"
        )
        if not attention_modules or len(attention_modules) != len(ffn_modules):
            raise TypeError(
                "expected a balanced Native RWKV-7 model, found "
                f"{len(attention_modules)} attention and {len(ffn_modules)} FFN modules"
            )
        if len(layer_modules) != len(attention_modules):
            raise TypeError(
                "expected one Native RWKV-7 layer per attention module, found "
                f"{len(layer_modules)} layers and {len(attention_modules)} attention modules"
            )
        for module in attention_modules:
            if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
                module._rwkv7_train_temp_original_forward = module.forward
            module._rwkv7_train_temp_cuda_enabled = True
            module._rwkv7_train_temp_forward = types.MethodType(
                native_train_temp_attention_forward, module
            )
            module.forward = module._rwkv7_train_temp_forward
        for module in ffn_modules:
            if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
                module._rwkv7_train_temp_original_forward = module.forward
            module._rwkv7_train_temp_cuda_enabled = True
            module._rwkv7_train_temp_forward = types.MethodType(
                native_train_temp_ffn_forward, module
            )
            module.forward = module._rwkv7_train_temp_forward
        for module in layer_modules:
            if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
                module._rwkv7_train_temp_original_forward = module.forward
            module._rwkv7_train_temp_cuda_enabled = True
            module._rwkv7_train_temp_forward = types.MethodType(
                native_train_temp_layer_forward, module
            )
            module.forward = module._rwkv7_train_temp_forward
        if not hasattr(model, "_rwkv7_train_temp_original_use_cache"):
            model._rwkv7_train_temp_original_use_cache = model.config.use_cache
        model.config.use_cache = False
        if getattr(model, "_rwkv7_train_temp_original_forward", None) is None:
            model._rwkv7_train_temp_original_forward = model.forward
        model._rwkv7_train_temp_cuda_enabled = True
        model._rwkv7_train_temp_forward = types.MethodType(
            native_train_temp_causal_lm_forward, model
        )
        model.forward = model._rwkv7_train_temp_forward
        return {
            "backend": "native_train_temp_cuda",
            "source_commit": TRAIN_TEMP_SOURCE_COMMIT,
            "attention_modules": len(attention_modules),
            "ffn_modules": len(ffn_modules),
            "layer_modules": len(layer_modules),
            "head_size": TRAIN_TEMP_HEAD_SIZE,
            "chunk_len": TRAIN_TEMP_CHUNK_LEN,
            "checkpoint_backend": _train_temp_checkpoint_backend(),
            "forward_dispatch": "direct",
        }

    from fla.layers.rwkv7 import RWKV7Attention
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward

    modules = tuple(model.modules())
    attention_modules = tuple(
        module for module in modules if isinstance(module, RWKV7Attention)
    )
    ffn_modules = tuple(
        module for module in modules if isinstance(module, RWKV7FeedForward)
    )
    attention_count = len(attention_modules)
    ffn_count = len(ffn_modules)
    if attention_count == 0 or ffn_count == 0 or attention_count != ffn_count:
        raise TypeError(
            "expected a balanced FLA RWKV-7 model, found "
            f"{attention_count} attention and {ffn_count} FFN modules"
        )
    for module in attention_modules:
        if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
            module._rwkv7_train_temp_original_forward = module.forward
            module.forward = types.MethodType(_attention_forward, module)
    for module in ffn_modules:
        if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
            module._rwkv7_train_temp_original_forward = module.forward
            module.forward = types.MethodType(_ffn_forward, module)
    if not hasattr(model, "_rwkv7_train_temp_original_use_cache"):
        model._rwkv7_train_temp_original_use_cache = model.config.use_cache
    model.config.use_cache = False
    model._rwkv7_train_temp_cuda_enabled = True
    return {
        "backend": "train_temp_cuda",
        "source_commit": TRAIN_TEMP_SOURCE_COMMIT,
        "attention_modules": attention_count,
        "ffn_modules": ffn_count,
        "head_size": TRAIN_TEMP_HEAD_SIZE,
        "chunk_len": TRAIN_TEMP_CHUNK_LEN,
    }


def disable_train_temp_cuda_backend(model) -> None:
    """Restore ordinary Native or FLA forwards after backend use."""

    for module in model.modules():
        original = getattr(module, "_rwkv7_train_temp_original_forward", None)
        if original is not None:
            module.forward = original
            delattr(module, "_rwkv7_train_temp_original_forward")
        if hasattr(module, "_rwkv7_train_temp_cuda_enabled"):
            delattr(module, "_rwkv7_train_temp_cuda_enabled")
        if hasattr(module, "_rwkv7_train_temp_forward"):
            delattr(module, "_rwkv7_train_temp_forward")
    if hasattr(model, "_rwkv7_train_temp_original_use_cache"):
        model.config.use_cache = model._rwkv7_train_temp_original_use_cache
        delattr(model, "_rwkv7_train_temp_original_use_cache")
    model._rwkv7_train_temp_cuda_enabled = False


__all__ = [
    "TRAIN_TEMP_CHUNK_LEN",
    "TRAIN_TEMP_HEAD_SIZE",
    "TRAIN_TEMP_SOURCE_COMMIT",
    "disable_train_temp_cuda_backend",
    "enable_train_temp_cuda_backend",
    "load_train_temp_cuda_extension",
    "native_train_temp_attention_forward",
    "native_train_temp_causal_lm_forward",
    "native_train_temp_ffn_forward",
    "native_train_temp_layer_forward",
    "train_temp_causal_cross_entropy",
    "train_temp_cuda_available",
    "train_temp_fused_cross_entropy",
]
