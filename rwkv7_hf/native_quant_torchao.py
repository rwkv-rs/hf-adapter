# coding=utf-8
"""Optional TorchAO tensor-core weight-only quantization integration.

This backend complements the dependency-free MM8/MM4 reference formats.  It
uses TorchAO's packed CUDA layouts, which can dispatch the PyTorch tensor-core
weight-only kernels while the HF adapter's ``native_graph`` runner removes
their Python dispatch overhead.  W4 currently requires a bf16 model because
the underlying ``aten::_weight_int4pack_mm`` CUDA contract is bf16.
"""
from __future__ import annotations

import gc
import inspect
import os
import types

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear
from .kernel_policy import current_kernel_policy


TORCHAO_QUANTIZATIONS = ("torchao_w8", "torchao_w4")
_NATIVE_QUANT_CACHE_ATTRS = (
    "_rwkv7_native_jit_pack_cache",
    "_rwkv7_native_graph_pack_cache",
    "_rwkv7_native_graph_runner_cache",
    "_rwkv7_native_prefill_graph_runner_cache",
    "_rwkv7_native_prefill_graph_hot_runner",
)


def _clear_native_quant_caches(model, *, release_cuda: bool = False) -> None:
    """Drop dense operand/graph references before or after in-place packing."""

    for attr in _NATIVE_QUANT_CACHE_ATTRS:
        if hasattr(model, attr):
            delattr(model, attr)
    if release_cuda and torch is not None and torch.cuda.is_available():
        gc.collect()
        try:
            visible_cuda_devices = int(torch.cuda.device_count())
        except Exception:
            visible_cuda_devices = 1
        allow_multi_gpu = os.environ.get(
            "RWKV7_TORCHAO_EMPTY_CACHE_MULTI_GPU",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        # empty_cache has process-wide allocator consequences on multi-GPU
        # workers. Isolated quantization keeps the lower construction peak;
        # heterogeneous serving workers retain other cards' warm pools.
        if visible_cuda_devices <= 1 or allow_multi_gpu:
            torch.cuda.empty_cache()


def _fla_rwkv7_ffn_forward_fused_relu2(
    self,
    x,
    attention_mask=None,
    state=None,
    cu_seqlens=None,
    **kwargs,
):
    """FLA RWKV7 FFN forward with the key activation in Marlin's epilogue.

    This mirrors FLA's public RWKV7FeedForward contract.  It is installed only
    on an exact, recognized FLA module; generic Marlin ``forward`` remains a
    plain Linear so arbitrary HF callers cannot accidentally square twice.
    """

    from fla.modules.token_shift import token_shift

    if attention_mask is not None:
        x = x.mul(attention_mask[:, -x.shape[-2] :, None])
    if state is not None:
        delta, ffn_state = token_shift(
            x,
            cu_seqlens,
            cache=state[self.layer_idx]["ffn_state"],
            output_cache=True,
        )
    else:
        delta, ffn_state = token_shift(x, cu_seqlens, output_cache=True)
    if state is not None:
        state.update(ffn_state=ffn_state, layer_idx=self.layer_idx, offset=0)
    mixed = x.addcmul(delta, self.x_k)
    return self.value(self.key.rwkv7_forward_relu2(mixed)), state


def _enable_fla_fused_relu2_ffn(model, key_names) -> int:
    """Patch only recognized FLA RWKV7 FFNs; fail safely on other models."""

    enabled = 0
    for key_name in key_names:
        if not key_name.endswith(".ffn.key"):
            continue
        ffn_name = key_name[: -len(".key")]
        try:
            ffn = model.get_submodule(ffn_name)
        except Exception:
            continue
        module_name = type(ffn).__module__
        if (
            not module_name.startswith("fla.models.rwkv7.")
            or type(ffn).__name__ != "RWKV7FeedForward"
        ):
            continue
        if not all(
            hasattr(ffn, attr)
            for attr in ("key", "value", "act_fn", "x_k", "layer_idx")
        ):
            continue
        try:
            parameters = tuple(inspect.signature(type(ffn).forward).parameters)
        except (TypeError, ValueError):
            continue
        if parameters[:5] != (
            "self",
            "x",
            "attention_mask",
            "state",
            "cu_seqlens",
        ):
            continue
        if not callable(getattr(ffn.key, "rwkv7_forward_relu2", None)):
            continue
        if getattr(ffn, "_rwkv7_fused_relu2_forward", False):
            enabled += 1
            continue
        ffn.forward = types.MethodType(_fla_rwkv7_ffn_forward_fused_relu2, ffn)
        ffn._rwkv7_fused_relu2_forward = True
        enabled += 1
    return enabled


def _torchao_w4_5090_speed_shape_supported(
    name: str,
    shape: tuple[int, int],
    dtype,
    capability: tuple[int, int],
    enabled_shapes: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether a BF16 W4 FFN shape is enabled by exact-card policy.

    TorchAO's tiled INT4 kernel established the safe role/shape boundary but
    regressed long-prompt prefill.  The production route for this exact gate is
    therefore Marlin BF16/W4: the measured FFN pair beats dense BF16 at rows
    1, 8, 128, and 1024.  Keep this deliberately narrow: the generic ``speed``
    policy remains head-only everywhere except the exact card, dtype, role, and
    matrix shapes covered by paired-baseline evidence.
    """

    role = str(name).lower()
    return bool(
        dtype == torch.bfloat16
        and tuple(int(v) for v in capability) == (12, 0)
        and (role.endswith(".ffn.key") or role.endswith(".ffn.value"))
        and tuple(int(v) for v in shape) in {
            tuple(int(v) for v in item) for item in enabled_shapes
        }
    )


def _torchao_w4_5090_speed_module(name: str, module) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    weight = getattr(module, "weight", None)
    device = getattr(weight, "device", None)
    if device is None or torch.device(device).type != "cuda":
        return False
    try:
        capability = tuple(torch.cuda.get_device_capability(device))
        policy = current_kernel_policy(device=device, torch_module=torch)
        enabled_shapes = tuple(getattr(policy, "marlin_w4_ffn_shapes", ()))
    except Exception:
        return False
    return _torchao_w4_5090_speed_shape_supported(
        name,
        tuple(int(v) for v in weight.shape),
        weight.dtype,
        capability,
        enabled_shapes,
    )


class TorchAOW4FP16Linear(torch.nn.Module):
    """Inference bridge from an fp16 model to TorchAO's bf16 INT4 kernel.

    PyTorch's CUDA ``_weight_int4pack_mm`` contract accepts bf16 activations.
    Converting only a speed-policy head keeps the recurrent body and its graph
    in fp16, while the packed head reads four times fewer weight bytes.  The
    output is cast back to the caller dtype so the HF logits contract remains
    unchanged.  Training intentionally falls back to the unquantized path at
    model construction time; this module is an inference-only deployment op.
    """

    def __init__(self, inner, *, output_dtype):
        super().__init__()
        self.inner = inner
        self.in_features = int(inner.in_features)
        self.out_features = int(inner.out_features)
        self.output_dtype = output_dtype

    def forward(self, x):
        return self.inner(x.to(torch.bfloat16)).to(self.output_dtype)

    def rwkv7_forward_into(self, x, out):
        out.copy_(self.forward(x))
        return out


def _torchao_api():
    try:
        from torchao.quantization import int4_weight_only, int8_weight_only, quantize_
    except Exception:
        try:
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )
            from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
                Int4PackingFormat,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TorchAO quantization requires a torch-compatible torchao install"
            ) from exc

        def int4_weight_only(*, group_size: int = 128):
            # TorchAO 0.17 defaults to the v2 plain layout, which requires the
            # optional mslk package. The tiled 4D layout preserves the
            # tinygemm CUDA route exposed by the removed helper.
            return Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
                version=2,
            )

        int8_weight_only = Int8WeightOnlyConfig
    return quantize_, int8_weight_only, int4_weight_only


def torchao_quantization_available() -> bool:
    try:
        _torchao_api()
    except Exception:
        return False
    return True


def _marlin_w4_model_profile(model, group_size: int):
    """Return an exact-card model profile or ``None`` before target selection."""

    config = getattr(model, "config", None)
    identity = (
        int(getattr(config, "hidden_size", 0)),
        int(getattr(config, "intermediate_size", 0)),
        int(getattr(config, "num_hidden_layers", 0)),
        int(group_size),
    )
    if not all(identity):
        return None
    for module in model.modules():
        weight = getattr(module, "weight", None)
        device = getattr(weight, "device", None)
        if device is None or torch.device(device).type != "cuda":
            continue
        try:
            profiles = current_kernel_policy(
                device=device, torch_module=torch
            ).marlin_w4_model_profiles
        except Exception:
            return None
        for profile in profiles:
            if tuple(int(value) for value in profile[:4]) == identity:
                return profile
        return None
    return None


def quantize_model_torchao(
    model,
    quantization: str,
    *,
    min_params: int = 1_000_000,
    policy: str = "memory",
    group_size: int = 128,
    quantize_head: bool | None = None,
    marlin_skip_last_layers: int | None = None,
) -> int:
    """Quantize selected ``nn.Linear`` weights in place with TorchAO.

    ``torchao_w8`` uses per-output int8 weight-only quantization.
    ``torchao_w4`` uses the tensor-core tiled int4 layout and groupwise affine
    scales. The latter currently requires bf16 activations/weights.
    """

    if torch is None:
        raise RuntimeError("TorchAO quantization requires torch")
    quantization = str(quantization).strip().lower().replace("-", "_")
    aliases = {"ao8": "torchao_w8", "aow8": "torchao_w8", "ao4": "torchao_w4", "aow4": "torchao_w4"}
    quantization = aliases.get(quantization, quantization)
    if quantization not in TORCHAO_QUANTIZATIONS:
        raise ValueError(f"unsupported TorchAO quantization: {quantization!r}")
    policy = normalize_native_mm_policy(policy)
    quantize_, int8_weight_only, int4_weight_only = _torchao_api()

    model_profile = (
        _marlin_w4_model_profile(model, int(group_size))
        if quantization == "torchao_w4" and policy == "speed"
        else None
    )
    effective_quantize_head = (
        bool(model_profile[4])
        if quantize_head is None and model_profile is not None
        else True
        if quantize_head is None
        else bool(quantize_head)
    )
    skip_last_layers = (
        int(model_profile[5])
        if marlin_skip_last_layers is None and model_profile is not None
        else 0
        if marlin_skip_last_layers is None
        else max(0, int(marlin_skip_last_layers))
    )

    targets: list[tuple[str, bool]] = []
    exact_5090_speed_targets = 0
    num_hidden_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        selected = should_quantize_linear(
            name,
            int(module.weight.numel()),
            min_params=int(min_params),
            policy=policy,
        )
        if selected and not effective_quantize_head and name.lower().endswith("lm_head"):
            selected = False
        exact_5090_speed = bool(
            quantization == "torchao_w4"
            and policy == "speed"
            and int(group_size) in (32, 64, 128)
            and int(module.weight.numel()) >= int(min_params)
            and _torchao_w4_5090_speed_module(name, module)
        )
        if exact_5090_speed and skip_last_layers and num_hidden_layers:
            parts = name.split(".")
            try:
                layer_index = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                layer_index = -1
            if layer_index >= num_hidden_layers - skip_last_layers:
                exact_5090_speed = False
        if selected or exact_5090_speed:
            targets.append((name, exact_5090_speed))
            exact_5090_speed_targets += int(exact_5090_speed)
    fp16_w4_bridge = False
    if quantization == "torchao_w4":
        bad = [
            name
            for name, _ in targets
            if model.get_submodule(name).weight.dtype != torch.bfloat16
        ]
        if bad:
            # The speed policy selects only lm_head.  Exact RTX 3090 evidence
            # shows the bf16 int4pack kernel plus two small casts is 2.2x-3.9x
            # faster than the fp16 head for rows 1..8.  Keep memory-policy block
            # quantization conservative: mixed-dtype wrappers inside every
            # recurrent block need a separate training/correctness contract.
            fp16_only = all(
                model.get_submodule(name).weight.dtype == torch.float16
                for name, _ in targets
            )
            if policy != "speed" or not fp16_only:
                raise ValueError(
                    "torchao_w4 requires a bf16 model, except for the measured "
                    "fp16 speed-policy head bridge; load with dtype=torch.bfloat16 "
                    f"before quantizing (first incompatible module: {bad[0]})"
                )
            fp16_w4_bridge = True
        config = int4_weight_only(group_size=int(group_size))
    else:
        config = int8_weight_only()

    marlin_linear_cls = None
    if exact_5090_speed_targets:
        from .native_quant_marlin import MarlinW4Linear

        marlin_linear_cls = MarlinW4Linear

        # Large checkpoints can fit dense BF16 but OOM if the asymmetric
        # TorchAO head is packed before any block payload has been released.
        # Replace the Marlin FFN matrices first so every completed layer frees
        # substantial dense storage before the head's temporary quant buffers
        # are allocated.  This changes construction peak only, not dispatch.
        targets.sort(key=lambda item: not item[1])
        _clear_native_quant_caches(model, release_cuda=True)

    for name, exact_5090_speed in targets:
        module = model.get_submodule(name)
        if exact_5090_speed:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                attr,
                marlin_linear_cls(
                    module,
                    group_size=int(group_size),
                    fp32_reduce=False,
                    production_bn_tn=True,
                    fuse_relu2=name.lower().endswith(".ffn.key"),
                ),
            )
            continue
        if fp16_w4_bridge:
            module.to(dtype=torch.bfloat16)
        quantize_(module, config)
        if fp16_w4_bridge:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, TorchAOW4FP16Linear(module, output_dtype=torch.float16))
    fused_relu2_ffn_modules = _enable_fla_fused_relu2_ffn(
        model,
        [
            name
            for name, exact in targets
            if exact and name.lower().endswith(".ffn.key")
        ],
    )
    setattr(
        model,
        "_rwkv7_native_mm_quantization",
        (
            "torchao_w4_fp16_head"
            if fp16_w4_bridge
            else "marlin_w4_5090_hybrid"
            if exact_5090_speed_targets
            else quantization
        ),
    )
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(
        model,
        "_rwkv7_native_mm_exact_5090_speed_modules",
        int(exact_5090_speed_targets),
    )
    setattr(
        model,
        "_rwkv7_native_mm_exact_5090_kernel",
        "bntn_marlin_bf16_w4" if exact_5090_speed_targets else None,
    )
    setattr(
        model,
        "_rwkv7_native_mm_fused_relu2_ffn_modules",
        int(fused_relu2_ffn_modules),
    )
    setattr(model, "_rwkv7_native_mm_group_size", int(group_size))
    setattr(
        model,
        "_rwkv7_native_mm_quantized_head",
        any(name.lower().endswith("lm_head") for name, _ in targets),
    )
    setattr(model, "_rwkv7_native_mm_marlin_skip_last_layers", skip_last_layers)
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name, _ in targets),
    )
    # Quantization mutates Linear weights in place. Any previously extracted
    # operand packs or captured graphs are now stale.
    _clear_native_quant_caches(model)
    return len(targets)


def quantize_model_torchao_w8(model, **kwargs) -> int:
    return quantize_model_torchao(model, "torchao_w8", **kwargs)


def quantize_model_torchao_w4(model, **kwargs) -> int:
    return quantize_model_torchao(model, "torchao_w4", **kwargs)
