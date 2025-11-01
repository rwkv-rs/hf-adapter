# coding=utf-8
"""Optional TorchAO tensor-core weight-only quantization integration.

This backend complements the dependency-free MM8/MM4 reference formats.  It
uses TorchAO's packed CUDA layouts, which can dispatch the PyTorch tensor-core
weight-only kernels while the HF adapter's ``native_graph`` runner removes
their Python dispatch overhead.  W4 currently requires a bf16 model because
the underlying ``aten::_weight_int4pack_mm`` CUDA contract is bf16.
"""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear


TORCHAO_QUANTIZATIONS = ("torchao_w8", "torchao_w4")


def _torchao_api():
    try:
        from torchao.quantization import int4_weight_only, int8_weight_only, quantize_
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TorchAO quantization requires a torch-compatible torchao install"
        ) from exc
    return quantize_, int8_weight_only, int4_weight_only


def torchao_quantization_available() -> bool:
    try:
        _torchao_api()
    except Exception:
        return False
    return True


def quantize_model_torchao(
    model,
    quantization: str,
    *,
    min_params: int = 1_000_000,
    policy: str = "memory",
    group_size: int = 128,
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

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if should_quantize_linear(
            name,
            int(module.weight.numel()),
            min_params=int(min_params),
            policy=policy,
        ):
            targets.append((name, module))
    if quantization == "torchao_w4":
        bad = [name for name, module in targets if module.weight.dtype != torch.bfloat16]
        if bad:
            raise ValueError(
                "torchao_w4 requires a bf16 model; load with dtype=torch.bfloat16 "
                f"before quantizing (first incompatible module: {bad[0]})"
            )
        config = int4_weight_only(group_size=int(group_size))
    else:
        config = int8_weight_only()

    for _, module in targets:
        quantize_(module, config)
    setattr(model, "_rwkv7_native_mm_quantization", quantization)
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    # Quantization mutates Linear weights in place. Any previously extracted
    # operand packs or captured graphs are now stale.
    for attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    return len(targets)


def quantize_model_torchao_w8(model, **kwargs) -> int:
    return quantize_model_torchao(model, "torchao_w8", **kwargs)


def quantize_model_torchao_w4(model, **kwargs) -> int:
    return quantize_model_torchao(model, "torchao_w4", **kwargs)
