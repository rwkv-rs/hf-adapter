"""Structural quantization adapters for the optional RWKV-7 kernel wheel.

The readable Hugging Face model intentionally owns no quantization policy.
Callers opt in after loading a model, or pass the BitsAndBytes configuration
returned here to ``from_pretrained``.  Converted modules keep the ordinary
``nn.Linear`` call contract, so reference HF execution remains available.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch


_REPORTS: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
_METHODS = {
    "w8": "native_w8",
    "mm8": "native_w8",
    "native_w8": "native_w8",
    "w4": "native_w4",
    "mm4": "native_w4",
    "native_w4": "native_w4",
    "a8w8": "a8w8",
    "torchao_w8": "torchao_w8",
    "torchao_w4": "torchao_w4",
    "marlin": "marlin_w4",
    "marlin_w4": "marlin_w4",
    "marlin_bntn_w4": "marlin_bntn_w4",
    "bnb8": "bnb8",
    "bitsandbytes_w8": "bnb8",
    "bnb4": "bnb4",
    "bitsandbytes_w4": "bnb4",
}

_NATIVE_GRAPH_METHODS = {
    "native_w8",
    "native_w4",
    "a8w8",
    "marlin_w4",
    "marlin_bntn_w4",
}


def _owned_graph_modules(model: Any) -> tuple[list[Any], str | None]:
    """Return native packed Linear replacements or a fail-closed reason.

    Decode and prefill graphs need static output storage for every packed
    projection. Adapter-owned modules expose ``rwkv7_forward_into`` for that
    contract. Dense ``nn.Linear`` modules are ordinary graph operands and are
    intentionally ignored here.
    """

    packed: list[Any] = []
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and type(module.weight) is torch.nn.Parameter
        ):
            continue
        if not (
            hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            continue
        if not type(module).__module__.startswith("rwkv7_kernels."):
            return [], (
                "quantized Linear wrapper is not owned by rwkv7-kernels: "
                f"{type(module).__module__}.{type(module).__name__}"
            )
        if not callable(getattr(module, "rwkv7_forward_into", None)):
            return [], (
                "native packed Linear lacks rwkv7_forward_into: "
                f"{type(module).__name__}"
            )
        packed.append(module)
    if not packed:
        return [], "no graph-safe native packed Linear was found"
    return packed, None


def quantization_graph_support(
    model: Any,
    *,
    phase: str,
) -> tuple[bool, str]:
    """Return whether one quantized model may enter a CUDA Graph.

    Native MM8/MM4/A8W8/Marlin modules have an explicit preallocated-output
    ABI and are safe to attempt in the diagnostic graph lane. TorchAO and
    BitsAndBytes are external implementations; they remain fail-closed unless
    the card policy or an explicit environment override enables the historical
    measured bridge. Numerical acceptance is still enforced by the GPU
    validators and is not implied by this structural capability result.
    """

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"prefill", "decode"}:
        raise ValueError("quantization graph phase must be prefill or decode")
    report = _REPORTS.get(model)
    if report is None:
        return True, "model has no package-owned quantization report"
    method = str(report.get("method", ""))
    if method in _NATIVE_GRAPH_METHODS:
        packed, reason = _owned_graph_modules(model)
        if reason is not None:
            return False, reason
        return True, (
            f"{len(packed)} native packed Linear modules expose stable graph outputs"
        )

    from .nvidia.kernel_policy import current_kernel_policy, env_flag

    try:
        parameter = next(model.parameters())
        device = parameter.device
    except (StopIteration, AttributeError):
        device = None
    policy = current_kernel_policy(device=device, torch_module=torch)
    if normalized_phase == "prefill":
        enabled = env_flag(
            "RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH",
            bool(getattr(policy, "native_external_quant_prefill_graph", False)),
        )
    else:
        enabled = env_flag(
            "RWKV7_NATIVE_GRAPH_EXTERNAL_QUANT",
            bool(getattr(policy, "native_external_quant_graph", False)),
        )
    if not enabled:
        return False, (
            f"{method or 'external'} quantization has no validated "
            f"{normalized_phase} CUDA Graph policy on this device"
        )
    return True, (
        f"{method or 'external'} quantization is enabled by the exact-device "
        f"{normalized_phase} graph policy"
    )


def _normalize_method(method: str) -> str:
    value = str(method).strip().lower().replace("-", "_")
    try:
        return _METHODS[value]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_METHODS.values())))
        raise ValueError(f"unsupported quantization {method!r}; expected: {choices}") from exc


def _clear_runtime_state(model: Any) -> None:
    """Invalidate only optional-package caches after module replacement."""

    from .nvidia.graph_pool import clear_native_graph_runners
    from .nvidia.prefill_graph_pool import clear_native_prefill_graph_runners

    clear_native_graph_runners(model)
    clear_native_prefill_graph_runners(model)
    # Historical quantizers attach diagnostic values while packing.  The v2
    # package owns those values instead of putting policy on the HF model.
    for name in tuple(vars(model)):
        if name.startswith("_rwkv7_native_mm_") or name in {
            "_rwkv7_native_jit_pack_cache",
            "_rwkv7_native_graph_pack_cache",
            "_rwkv7_native_graph_runner_cache",
            "_rwkv7_native_prefill_graph_runner_cache",
            "_rwkv7_native_prefill_graph_hot_runner",
            "_rwkv7_native_model_jit_pack_cache",
        }:
            delattr(model, name)


def _module_device_dtype(model: Any) -> tuple[str | None, str | None]:
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError):
        return None, None
    return str(parameter.device), str(parameter.dtype)


def _bitsandbytes_modules(model: Any, method: str) -> list[str]:
    expected = "Linear8bitLt" if method == "bnb8" else "Linear4bit"
    return [
        name
        for name, module in model.named_modules()
        if type(module).__name__ == expected
        and type(module).__module__.startswith("bitsandbytes.")
    ]


def prepare_bitsandbytes_config(
    method: str,
    *,
    config: Any | None = None,
    policy: str = "memory",
    skip_modules: list[str] | tuple[str, ...] | None = None,
    int8_threshold: float = 6.0,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    double_quant: bool = True,
):
    """Create a standard HF ``BitsAndBytesConfig`` without changing RWKV config."""

    normalized = _normalize_method(method)
    if normalized not in {"bnb8", "bnb4"}:
        raise ValueError("prepare_bitsandbytes_config accepts only bnb8 or bnb4")
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - optional environment
        raise RuntimeError("BitsAndBytes loading requires Transformers") from exc
    common: dict[str, Any] = {
        "llm_int8_skip_modules": rwkv7_bitsandbytes_skip_modules(
            config,
            policy=policy,
            extra=skip_modules,
        )
    }
    if normalized == "bnb8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(int8_threshold),
            **common,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=str(quant_type),
        bnb_4bit_use_double_quant=bool(double_quant),
        **common,
    )


def rwkv7_bitsandbytes_skip_modules(
    config: Any | None,
    *,
    policy: str = "memory",
    extra: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return concrete BnB skips needed by native RWKV projection packing."""

    policies = {
        "memory",
        "output_hot",
        "decode_rk",
        "decode_hot",
        "prefill_hot",
        "dense",
    }
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized not in policies:
        raise ValueError(
            f"unsupported BitsAndBytes skip policy {policy!r}; "
            f"expected: {', '.join(sorted(policies))}"
        )
    skips = ["lm_head"]
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    for layer_index in range(num_layers):
        prefix = f"model.layers.{layer_index}"
        for lora_name in ("w_lora", "a_lora", "g_lora", "v_lora"):
            for linear_index in (0, 2):
                skips.append(
                    f"{prefix}.attn.{lora_name}.lora.{linear_index}"
                )
        if normalized == "output_hot":
            skips.append(f"{prefix}.attn.o_proj")
        if normalized in {"decode_rk", "decode_hot", "prefill_hot", "dense"}:
            projection_names = (
                ("r_proj", "k_proj")
                if normalized == "decode_rk"
                else ("r_proj", "k_proj", "v_proj", "o_proj")
            )
            skips.extend(f"{prefix}.attn.{name}" for name in projection_names)
        if normalized == "prefill_hot":
            skips.append(f"{prefix}.ffn.key")
            # Preserve most value projections as dense hot operands. The
            # historical measured policy quantized every fourth layer.
            if (layer_index + 1) % 4:
                skips.append(f"{prefix}.ffn.value")
        if normalized == "dense":
            skips.extend((f"{prefix}.ffn.key", f"{prefix}.ffn.value"))
    if extra:
        skips.extend(str(name) for name in extra)
    return list(dict.fromkeys(skips))


def _quantize_marlin(
    model: Any,
    *,
    min_params: int,
    policy: str,
    group_size: int,
    fp32_reduce: bool,
    production_bn_tn: bool,
) -> int:
    from .nvidia.native_quant_marlin import MarlinW4Linear
    from .nvidia.native_quant_policy import (
        normalize_native_mm_policy,
        should_quantize_linear,
    )

    policy = normalize_native_mm_policy(policy)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and should_quantize_linear(
            name,
            int(module.weight.numel()),
            min_params=int(min_params),
            policy=policy,
        )
    ]
    for name in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attribute,
            MarlinW4Linear(
                getattr(parent, attribute),
                group_size=int(group_size),
                fp32_reduce=bool(fp32_reduce),
                production_bn_tn=bool(production_bn_tn),
                fuse_relu2=name.endswith(".ffn.key"),
            ),
        )
    return len(targets)


def quantize_model(
    model: Any,
    method: str,
    *,
    min_params: int = 8_000_000,
    policy: str = "memory",
    group_size: int = 0,
    group_policy: str = "all",
    fused: bool = True,
    fp32_reduce: bool = True,
    production_bn_tn: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply or adopt one quantization family and return its actual report.

    ``bnb8``/``bnb4`` adopt modules already created by standard Transformers
    loading; use :func:`prepare_bitsandbytes_config` before ``from_pretrained``.
    All other methods structurally replace eligible ``nn.Linear`` modules.
    """

    normalized = _normalize_method(method)
    _clear_runtime_state(model)
    if normalized == "native_w8":
        from .nvidia.native_quant_mm8 import quantize_model_mm8

        replaced = quantize_model_mm8(
            model,
            min_params=int(min_params),
            fused=bool(fused),
            policy=policy,
        )
        implementation = "native-mm8-w8"
    elif normalized == "native_w4":
        from .nvidia.native_quant_mm4 import quantize_model_mm4

        replaced = quantize_model_mm4(
            model,
            min_params=int(min_params),
            fused=bool(fused),
            policy=policy,
            group_size=int(group_size),
            group_policy=group_policy,
        )
        implementation = "native-mm4-w4"
    elif normalized == "a8w8":
        from .nvidia.native_quant_a8w8 import quantize_model_a8w8

        replaced = quantize_model_a8w8(
            model,
            min_params=int(min_params),
            policy=policy,
        )
        implementation = "native-dynamic-a8w8"
    elif normalized in {"torchao_w8", "torchao_w4"}:
        from .nvidia.native_quant_torchao import quantize_model_torchao

        replaced = quantize_model_torchao(
            model,
            normalized,
            min_params=int(min_params),
            policy=policy,
            group_size=int(group_size or 128),
            **kwargs,
        )
        implementation = normalized.replace("_", "-")
    elif normalized in {"marlin_w4", "marlin_bntn_w4"}:
        exact_bn_tn = (
            normalized == "marlin_bntn_w4"
            if production_bn_tn is None
            else bool(production_bn_tn)
        )
        if exact_bn_tn:
            if not torch.cuda.is_available():
                raise RuntimeError("Marlin BN/TN requires a CUDA device")
            capability = tuple(torch.cuda.get_device_capability())
            if capability != (12, 0):
                raise RuntimeError(
                    "marlin_bntn_w4 is an exact SM120/Blackwell route; "
                    f"detected sm{capability[0]}{capability[1]}"
                )
        replaced = _quantize_marlin(
            model,
            min_params=int(min_params),
            policy=policy,
            group_size=int(group_size or 128),
            fp32_reduce=bool(fp32_reduce),
            production_bn_tn=exact_bn_tn,
        )
        implementation = (
            "marlin-bntn-bf16-w4" if exact_bn_tn else "marlin-bf16-w4"
        )
    else:
        names = _bitsandbytes_modules(model, normalized)
        if not names:
            raise RuntimeError(
                f"no {normalized} modules found; pass prepare_bitsandbytes_config() "
                "to AutoModelForCausalLM.from_pretrained first"
            )
        replaced = len(names)
        implementation = f"bitsandbytes-{normalized[-1]}bit-adapter"

    _clear_runtime_state(model)
    device, dtype = _module_device_dtype(model)
    report = {
        "method": normalized,
        "implementation": implementation,
        "replaced_modules": int(replaced),
        "device": device,
        "dtype": dtype,
        "policy": str(policy),
        "group_size": int(group_size),
        "graph_cache_invalidated": True,
    }
    _REPORTS[model] = report
    decode_graph, decode_reason = quantization_graph_support(model, phase="decode")
    prefill_graph, prefill_reason = quantization_graph_support(model, phase="prefill")
    report.update(
        {
            "decode_cuda_graph_structurally_supported": bool(decode_graph),
            "decode_cuda_graph_reason": decode_reason,
            "prefill_cuda_graph_structurally_supported": bool(prefill_graph),
            "prefill_cuda_graph_reason": prefill_reason,
        }
    )
    _REPORTS[model] = report
    return dict(report)


def quantization_report(model: Any) -> dict[str, Any] | None:
    """Return package-owned quantization metadata for ``model`` if present."""

    report = _REPORTS.get(model)
    return None if report is None else dict(report)


__all__ = [
    "prepare_bitsandbytes_config",
    "quantization_graph_support",
    "quantization_report",
    "quantize_model",
    "rwkv7_bitsandbytes_skip_modules",
]
