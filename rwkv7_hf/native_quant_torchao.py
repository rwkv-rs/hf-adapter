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
    fp16_w4_bridge = False
    if quantization == "torchao_w4":
        bad = [name for name, module in targets if module.weight.dtype != torch.bfloat16]
        if bad:
            # The speed policy selects only lm_head.  Exact RTX 3090 evidence
            # shows the bf16 int4pack kernel plus two small casts is 2.2x-3.9x
            # faster than the fp16 head for rows 1..8.  Keep memory-policy block
            # quantization conservative: mixed-dtype wrappers inside every
            # recurrent block need a separate training/correctness contract.
            fp16_only = all(module.weight.dtype == torch.float16 for _, module in targets)
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

    for name, module in targets:
        if fp16_w4_bridge:
            module.to(dtype=torch.bfloat16)
        quantize_(module, config)
        if fp16_w4_bridge:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, TorchAOW4FP16Linear(module, output_dtype=torch.float16))
    setattr(
        model,
        "_rwkv7_native_mm_quantization",
        "torchao_w4_fp16_head" if fp16_w4_bridge else quantization,
    )
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name, _ in targets),
    )
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
