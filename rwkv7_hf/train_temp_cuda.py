"""Opt-in RWKV-LM train_temp CUDA training backend for HF RWKV-7 models.

The kernels under ``csrc/train_temp`` are vendored from RWKV-LM at the exact
commit recorded in that directory.  This module keeps them lazy and isolated:
normal HF/FLA inference and training do not compile or route through these ops.
"""

from __future__ import annotations

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


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "train_temp"


def _op_registered(namespace: str) -> bool:
    try:
        getattr(getattr(torch.ops, namespace), "forward")
    except (AttributeError, RuntimeError):
        return False
    return True


def _validate_runtime() -> None:
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("train_temp CUDA backend requires Linux with an available CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        raise RuntimeError(
            "train_temp BF16 CUDA backend requires Ampere (sm_80) or newer; "
            f"found sm_{major}{minor}"
        )


def load_train_temp_cuda_extension(*, verbose: bool | None = None) -> None:
    """Build and load the vendored train_temp operators once."""

    global _L2WRAP_EXTENSION, _LOADED, _LOAD_ERROR
    _validate_runtime()
    if _LOADED:
        return
    if _LOAD_ERROR is not None:
        raise RuntimeError("train_temp CUDA extension previously failed to load") from _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED:
            return
        try:
            from torch.utils.cpp_extension import CUDA_HOME, load

            if CUDA_HOME is None:
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
            for namespace, filenames in _OP_SOURCES.items():
                if _op_registered(namespace):
                    continue
                load(
                    name=f"rwkv7_hf_{namespace}",
                    sources=[str(root / filename) for filename in filenames],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                    is_python_module=False,
                    verbose=bool(verbose),
                )
            if not _op_registered("rwkv7_clampw_v3"):
                load(
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
                    is_python_module=False,
                    verbose=bool(verbose),
                )
            _L2WRAP_EXTENSION = load(
                name="rwkv7_hf_l2wrap_ce_bf16_v2",
                sources=[
                    str(root / "rwkv7_l2wrap_ce_bf16_v2.cpp"),
                    str(root / "rwkv7_l2wrap_ce_bf16_v2.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                verbose=bool(verbose),
            )
            missing = [namespace for namespace in _OP_SOURCES if not _op_registered(namespace)]
            if not _op_registered("rwkv7_clampw_v3"):
                missing.append("rwkv7_clampw_v3")
            if missing:
                raise RuntimeError(f"train_temp extension did not register required ops: {missing}")
            _LOADED = True
        except BaseException as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(f"train_temp CUDA extension failed to load: {exc}") from exc


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
        inputs = tuple(value.contiguous() for value in (x, x_r, x_w, x_k, x_v, x_a, x_g))
        ctx.save_for_backward(*inputs)
        return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(*inputs))

    @staticmethod
    def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
        grads = tuple(value.contiguous() for value in (grad_r, grad_w, grad_k, grad_v, grad_a, grad_g))
        return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(*grads, *ctx.saved_tensors))


class _KkPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, k_k, a, k_a):
        inputs = tuple(value.contiguous() for value in (k, k_k, a, k_a))
        outputs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(*inputs, TRAIN_TEMP_HEAD_SIZE)
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
        inputs = tuple(value.contiguous() for value in (x, r, k, v, r_k, weight, bias, g))
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
        state_aux = torch.empty(batch, tokens, heads, head_size, dtype=torch.float32, device=w.device)
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
        inputs = tuple(value.contiguous() for value in (x, x_k, key_weight, value_weight))
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


def train_temp_fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Run the exact train_temp fused FP32 CE plus L2Wrap gradient."""

    load_train_temp_cuda_extension()
    return _L2WrapCrossEntropy.apply(logits, targets)


def _dense_mask_only(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is not None and not bool(torch.all(attention_mask != 0).item()):
        raise ValueError("train_temp CUDA backend does not support padded or masked batches")


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
    if past_key_values is not None or use_cache or cu_seqlens is not None or output_attentions:
        raise ValueError("train_temp CUDA backend is a dense no-cache training path")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.shape[1] % TRAIN_TEMP_CHUNK_LEN:
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
    w = self.w_lora(xw)
    k = self.k_proj(xk)
    v = self.v_proj(xv)
    if self.layer_idx == 0:
        v_first = v
    else:
        v12 = F.linear(self.v_lora.lora[0](xv), self.v_lora.lora[2].weight, None)
        v = _VResGate.apply(v, v_first, self.v_lora.lora[2].bias, v12)
    a12 = F.linear(self.a_lora.lora[0](xa), self.a_lora.lora[2].weight, None)
    a = _AGate.apply(self.a_lora.lora[2].bias, a12)
    g = self.g_lora(xg)
    k, neg_kk, kka = _KkPre.apply(k, self.k_k.reshape(-1), a, self.k_a.reshape(-1))
    batch, tokens, _ = r.shape
    heads = int(self.num_heads)
    if int(self.head_dim) != TRAIN_TEMP_HEAD_SIZE or int(self.head_v_dim) != TRAIN_TEMP_HEAD_SIZE:
        raise ValueError("train_temp CUDA backend currently requires K/V head dimensions of 64")
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
    return self.o_proj(values), None, past_key_values, v_first


def _ffn_forward(self, x, attention_mask=None, state=None, cu_seqlens=None, **kwargs):
    _dense_mask_only(attention_mask)
    if state is not None or cu_seqlens is not None:
        raise ValueError("train_temp CUDA backend is a dense no-cache training path")
    return _CMix.apply(x, self.x_k.reshape(-1), self.key.weight, self.value.weight), state


def enable_train_temp_cuda_backend(model) -> dict[str, Any]:
    """Patch an FLA-backed HF RWKV-7 model with official train_temp kernels."""

    load_train_temp_cuda_extension()
    from fla.layers.rwkv7 import RWKV7Attention
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward

    attention_count = 0
    ffn_count = 0
    for module in model.modules():
        if isinstance(module, RWKV7Attention):
            if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
                module._rwkv7_train_temp_original_forward = module.forward
                module.forward = types.MethodType(_attention_forward, module)
            attention_count += 1
        elif isinstance(module, RWKV7FeedForward):
            if getattr(module, "_rwkv7_train_temp_original_forward", None) is None:
                module._rwkv7_train_temp_original_forward = module.forward
                module.forward = types.MethodType(_ffn_forward, module)
            ffn_count += 1
    if attention_count == 0 or ffn_count == 0 or attention_count != ffn_count:
        raise TypeError(
            "expected a balanced FLA RWKV-7 model, found "
            f"{attention_count} attention and {ffn_count} FFN modules"
        )
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
    """Restore FLA forwards after :func:`enable_train_temp_cuda_backend`."""

    for module in model.modules():
        original = getattr(module, "_rwkv7_train_temp_original_forward", None)
        if original is not None:
            module.forward = original
            delattr(module, "_rwkv7_train_temp_original_forward")
    model._rwkv7_train_temp_cuda_enabled = False


__all__ = [
    "TRAIN_TEMP_CHUNK_LEN",
    "TRAIN_TEMP_HEAD_SIZE",
    "TRAIN_TEMP_SOURCE_COMMIT",
    "disable_train_temp_cuda_backend",
    "enable_train_temp_cuda_backend",
    "load_train_temp_cuda_extension",
    "train_temp_cuda_available",
    "train_temp_fused_cross_entropy",
]
