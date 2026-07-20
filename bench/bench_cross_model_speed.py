#!/usr/bin/env python3
"""Benchmark one RWKV-7 or official Qwen3.5 HF configuration.

The matrix orchestrator invokes this worker in a fresh process for every raw
row.  It intentionally benchmarks exact tensor shapes; model-quality evaluation
uses separate task runners.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")


def _bootstrap_qwen_backend(argv: list[str]) -> str:
    """Read the backend before importing Transformers, which binds Qwen ops."""

    for index, value in enumerate(argv):
        if value == "--qwen-backend" and index + 1 < len(argv):
            return argv[index + 1].strip().lower()
        if value.startswith("--qwen-backend="):
            return value.split("=", 1)[1].strip().lower()
    return "auto"


QWEN_BACKEND_BOOTSTRAP = _bootstrap_qwen_backend(sys.argv[1:])
QWEN_FORCE_TORCH = (
    os.environ.get("RWKV7_QWEN35_FORCE_TORCH", "0").lower() in {"1", "true", "yes", "on"}
    or QWEN_BACKEND_BOOTSTRAP == "torch"
)
if QWEN_FORCE_TORCH:
    sys.path[:] = [path for path in sys.path if "flash-linear-attention" not in path.replace("\\", "/").lower()]
    _original_find_spec = importlib.util.find_spec

    def _find_spec_without_fla(name: str, *args, **kwargs):
        if name == "fla" or name.startswith("fla."):
            return None
        return _original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec_without_fla

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PROMPT_SEED = (
    "RWKV and Qwen are language models evaluated with identical tensor shapes. "
    "This sentence is repeated only to build deterministic benchmark tokens. "
)


def package_version(name: str) -> str | None:
    candidates = [name]
    if name == "triton":
        candidates.append("triton-windows")
    for candidate in candidates:
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
    return None


def validate_args(args: argparse.Namespace) -> None:
    for field in ("batch_size", "prompt_tokens", "decode_tokens", "runs"):
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if int(args.warmup) < 0:
        raise ValueError("--warmup must be non-negative")
    if int(getattr(args, "prefill_chunk_size", 0) or 0) < 0:
        raise ValueError("--prefill-chunk-size must be non-negative")
    if args.model_role not in {"candidate", "reference"}:
        raise ValueError("--model-role must be candidate or reference")
    if args.model_kind not in {"rwkv", "qwen35"}:
        raise ValueError("--model-kind must be rwkv or qwen35")
    native_quantizations = {
        "torchao_w8",
        "torchao_w4",
        "a8w8",
        "mm8",
        "mm4",
        "bnb8_a8w8_head",
    }
    if args.model_kind != "rwkv" and str(args.quantization) in native_quantizations:
        raise ValueError(
            f"{args.quantization} is an RWKV candidate backend; use bnb8/bnb4 for Qwen reference rows"
        )
    if args.qwen_backend not in {"auto", "fla", "torch"}:
        raise ValueError("--qwen-backend must be auto, fla, or torch")
    qwen_conv_backend = str(getattr(args, "qwen_conv_backend", "auto"))
    if qwen_conv_backend not in {"auto", "causal_conv1d", "fla_triton"}:
        raise ValueError("--qwen-conv-backend must be auto, causal_conv1d, or fla_triton")
    if args.qwen_backend == "torch" and qwen_conv_backend != "auto":
        raise ValueError("an accelerated Qwen conv backend cannot be combined with --qwen-backend torch")
    if args.probe_output and args.probe_tokens <= 0:
        raise ValueError("--probe-tokens must be positive when --probe-output is set")


def build_exact_prompt(tokenizer, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    encoded = tokenizer(PROMPT_SEED * 32, return_tensors="pt", add_special_tokens=False).input_ids
    if encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] == 0:
        raise RuntimeError(f"tokenizer returned invalid input shape {tuple(encoded.shape)}")
    repeats = (prompt_tokens + int(encoded.shape[1]) - 1) // int(encoded.shape[1])
    ids = encoded.repeat(1, repeats)[:, :prompt_tokens].repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def model_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "model_name": Path(args.model).name,
        "model_id_or_path": args.model,
        "model_size_label": args.model_size_label,
        "model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "num_heads", None)),
        "head_dim": getattr(config, "head_dim", None),
    }


def base_row(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": args.benchmark_matrix,
        "model_pair": args.model_pair,
        "model_role": args.model_role,
        "model_kind": args.model_kind,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "qwen_backend_requested": args.qwen_backend,
        "qwen_conv_backend_requested": getattr(args, "qwen_conv_backend", "auto"),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "prefill_chunk_size": int(getattr(args, "prefill_chunk_size", 0) or 0),
        "native_quant_min_params_requested": int(getattr(args, "native_quant_min_params", 1_000_000)),
        "native_quant_policy_requested": str(getattr(args, "native_quant_policy", "memory")),
        "torchao_group_size_requested": int(getattr(args, "torchao_group_size", 128)),
        **model_metadata(args),
    }


def failure_row(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        **base_row(args),
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": repr(exc),
    }


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def device_map_for(device: str):
    return {"": cuda_device_index(device)} if device.startswith("cuda") else None


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(cuda_device_index(device))
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(cuda_device_index(device)) / 1024 / 1024, 1)


def _tensor_payload_bytes(tensor, seen: set[int]) -> int:
    """Count physical payloads for ordinary and wrapper-subclass tensors."""

    ident = id(tensor)
    if ident in seen:
        return 0
    seen.add(ident)
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten) and type(tensor) not in {torch.Tensor, torch.nn.Parameter}:
        try:
            names = flatten()[0]
            payload = 0
            for name in names:
                value = getattr(tensor, name)
                if isinstance(value, torch.Tensor):
                    payload += _tensor_payload_bytes(value, seen)
            if payload:
                return payload
        except Exception:
            pass
    return int(tensor.numel()) * int(tensor.element_size())


def model_footprint_mb(model) -> float:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += _tensor_payload_bytes(tensor, seen)
    return round(total / 1024 / 1024, 1)


def _logical_parameter_numel(parameter) -> int:
    """Return the unpacked logical size of dense or bitsandbytes parameters."""

    quant_state = getattr(parameter, "quant_state", None)
    shape = getattr(quant_state, "shape", None)
    if shape is not None:
        try:
            logical = math.prod(int(dim) for dim in shape)
            if logical > 0:
                return logical
        except (TypeError, ValueError):
            pass
    return int(parameter.numel())


def model_parameter_metadata(model, args: argparse.Namespace) -> dict[str, Any]:
    """Count unique total and per-token active logical parameters.

    Dense models activate every parameter. For a future top-k MoE reference,
    shared parameters stay active while expert parameters are scaled by the
    configured experts-per-token fraction.
    """

    unique: dict[int, tuple[str, Any]] = {}
    for name, parameter in model.named_parameters():
        unique.setdefault(id(parameter), (name, parameter))
    total = sum(_logical_parameter_numel(parameter) for _name, parameter in unique.values())
    expert = sum(
        _logical_parameter_numel(parameter)
        for name, parameter in unique.values()
        if ".experts." in name or ".expert." in name
    )
    config = getattr(model, "config", None)
    num_experts = int(getattr(config, "num_experts", 0) or 0)
    experts_per_token = int(getattr(config, "num_experts_per_tok", 0) or 0)
    if expert > 0 and num_experts > 0 and 0 < experts_per_token <= num_experts:
        active = total - expert + round(expert * experts_per_token / num_experts)
        method = "moe_topk_logical"
    else:
        active = total
        method = "dense_all_logical"
    prefill_applications = active * int(args.batch_size) * int(args.prompt_tokens)
    decode_applications = active * int(args.batch_size) * int(args.decode_tokens)
    return {
        "logical_parameter_count": total,
        "active_parameter_count": active,
        "active_parameter_fraction": (active / total) if total else None,
        "active_parameter_method": method,
        "prefill_active_parameter_applications": prefill_applications,
        "decode_active_parameter_applications": decode_applications,
    }


def quantization_config(args: argparse.Namespace, dtype: torch.dtype):
    if args.quantization in {"none", "torchao_w8", "torchao_w4", "a8w8", "mm8", "mm4"}:
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is required for bnb8/bnb4 rows")
    from transformers import BitsAndBytesConfig

    if args.quantization in {"bnb8", "bnb8_a8w8_head"}:
        # bitsandbytes' LLM.int8 outlier path evaluates ``outliers.any()`` on
        # the host and therefore cannot be captured by a CUDA graph.  Keep the
        # library default for ordinary runs, while allowing the strict native
        # graph matrix to disable that path explicitly and reproducibly.
        threshold = float(os.environ.get("RWKV7_BNB_INT8_THRESHOLD", "6.0"))
        if threshold < 0.0:
            raise ValueError("RWKV7_BNB_INT8_THRESHOLD must be non-negative")
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=threshold)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )


def set_rwkv_runtime(model, args: argparse.Namespace) -> None:
    if not hasattr(model.config, "attn_mode"):
        return
    model.config.attn_mode = args.rwkv_attn_mode
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = args.rwkv_attn_mode


def prepare_rwkv_model_dir(model_path: str, code_source: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if code_source == "model":
        return model_path, None
    source = Path(model_path).resolve()
    repo_code = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    if not source.is_dir():
        raise ValueError("--rwkv-code-source repo requires a local converted model directory")
    temporary = tempfile.TemporaryDirectory(prefix="rwkv7_qwen35_repo_code_", dir=source.parent)
    target = Path(temporary.name)
    for item in source.iterdir():
        if item.name == "__pycache__" or item.suffix == ".py":
            continue
        link = target / item.name
        try:
            link.symlink_to(item, target_is_directory=item.is_dir())
        except OSError:
            if item.is_dir():
                shutil.copytree(item, link)
            else:
                os.link(item, link)
    for py_file in repo_code.glob("*.py"):
        shutil.copy2(py_file, target / py_file.name)
    return str(target), temporary


def load_model(args: argparse.Namespace, dtype: torch.dtype, model_path: str | None = None):
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device),
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args, dtype)
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    if args.model_kind == "rwkv":
        kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path or args.model, **kwargs).eval()
        if args.quantization == "bnb8_a8w8_head":
            from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8

            replaced = quantize_model_a8w8(
                model,
                min_params=int(getattr(args, "native_quant_min_params", 1_000_000)),
                policy="speed",
            )
            setattr(model, "_rwkv7_cross_model_quant_backend", args.quantization)
            setattr(model, "_rwkv7_cross_model_quant_replaced_modules", int(replaced))
        elif args.quantization in {"torchao_w8", "torchao_w4"}:
            from rwkv7_hf.native_quant_torchao import quantize_model_torchao

            replaced = quantize_model_torchao(
                model,
                args.quantization,
                min_params=int(getattr(args, "native_quant_min_params", 1_000_000)),
                policy=str(getattr(args, "native_quant_policy", "memory")),
                group_size=int(getattr(args, "torchao_group_size", 128)),
            )
            setattr(model, "_rwkv7_cross_model_quant_backend", args.quantization)
            setattr(model, "_rwkv7_cross_model_quant_replaced_modules", int(replaced))
        elif args.quantization in {"a8w8", "mm8", "mm4"}:
            if args.quantization == "a8w8":
                from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8 as quantize_model
            elif args.quantization == "mm8":
                from rwkv7_hf.native_quant_mm8 import quantize_model_mm8 as quantize_model
            else:
                from rwkv7_hf.native_quant_mm4 import quantize_model_mm4 as quantize_model
            replaced = quantize_model(
                model,
                min_params=int(getattr(args, "native_quant_min_params", 1_000_000)),
                policy=str(getattr(args, "native_quant_policy", "memory")),
            )
            setattr(model, "_rwkv7_cross_model_quant_backend", args.quantization)
            setattr(model, "_rwkv7_cross_model_quant_replaced_modules", int(replaced))
        set_rwkv_runtime(model, args)
        return model

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("installed Transformers does not provide Qwen3_5ForCausalLM") from exc
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, **kwargs).eval()
    if (
        args.qwen_backend == "fla"
        and str(args.device).startswith("cuda")
        and torch.cuda.get_device_capability(cuda_device_index(args.device)) == (7, 0)
    ):
        try:
            from bench.qwen35_sm70_fla import bind_qwen35_sm70_fla
        except ModuleNotFoundError:
            from qwen35_sm70_fla import bind_qwen35_sm70_fla

        bind_qwen35_sm70_fla(model)
    if getattr(args, "qwen_conv_backend", "auto") == "fla_triton":
        try:
            from bench.qwen35_fla_triton_conv import bind_qwen35_fla_triton_conv
        except ModuleNotFoundError:
            from qwen35_fla_triton_conv import bind_qwen35_fla_triton_conv

        model._qwen35_fla_triton_conv_layers = bind_qwen35_fla_triton_conv(model)
    return model


def _operator_origin(value: Any) -> str:
    """Return a stable module-qualified name for a bound kernel or module."""

    inner = getattr(value, "func", None)
    if inner is not None and inner is not value:
        value = inner
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
    if module and qualname:
        return f"{module}.{qualname}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _origin_is(origin: str, prefixes: tuple[str, ...]) -> bool:
    return any(origin == prefix or origin.startswith(prefix + ".") for prefix in prefixes)


_QWEN35_FLA_TRITON_CONV_PREFIXES = (
    "bench.qwen35_fla_triton_conv",
    "qwen35_fla_triton_conv",
)
_QWEN35_SM70_FLA_PREFIXES = (
    "bench.qwen35_sm70_fla",
    "qwen35_sm70_fla",
)
_QWEN35_FLA_CORE_PREFIXES = ("fla",) + _QWEN35_SM70_FLA_PREFIXES
_QWEN35_ACCELERATED_CONV_PREFIXES = ("causal_conv1d",) + _QWEN35_FLA_TRITON_CONV_PREFIXES


def qwen_fla_operator_contract(model) -> dict[str, Any]:
    """Inspect the operators actually bound by every Qwen3.5 linear layer.

    Transformers silently substitutes Python/Torch reference functions when
    FLA or causal-conv1d is unavailable. Package presence alone therefore is
    not sufficient evidence that the benchmark used accelerated kernels. The
    Gated DeltaNet and norm operators form the required FLA core contract;
    causal convolution is tracked separately so strict rows can reject the
    Transformers Torch fallback while older FLA-core-only rows remain readable.
    """

    operator_attrs = (
        "chunk_gated_delta_rule",
        "recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    layers: list[tuple[str, Any]] = []
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            if type(module).__name__ == "Qwen3_5GatedDeltaNet" or any(
                hasattr(module, attr) for attr in operator_attrs
            ):
                layers.append((name, module))

    prefill_origins = sorted({_operator_origin(getattr(layer, "chunk_gated_delta_rule", None)) for _, layer in layers})
    decode_origins = sorted(
        {_operator_origin(getattr(layer, "recurrent_gated_delta_rule", None)) for _, layer in layers}
    )
    conv_update_origins = sorted(
        {_operator_origin(getattr(layer, "causal_conv1d_update", None)) for _, layer in layers}
    )
    conv_prefill_origins = sorted({_operator_origin(getattr(layer, "causal_conv1d_fn", None)) for _, layer in layers})
    norm_origins = sorted({_operator_origin(layer.norm) for _, layer in layers if hasattr(layer, "norm")})

    prefill_fla_layers = sum(
        _origin_is(
            _operator_origin(getattr(layer, "chunk_gated_delta_rule", None)),
            _QWEN35_FLA_CORE_PREFIXES,
        )
        for _, layer in layers
    )
    decode_fla_layers = sum(
        _origin_is(_operator_origin(getattr(layer, "recurrent_gated_delta_rule", None)), ("fla",))
        for _, layer in layers
    )
    conv_update_fused_layers = sum(
        _origin_is(
            _operator_origin(getattr(layer, "causal_conv1d_update", None)),
            _QWEN35_ACCELERATED_CONV_PREFIXES,
        )
        for _, layer in layers
    )
    conv_prefill_fused_layers = sum(
        getattr(layer, "causal_conv1d_fn", None) is not None
        and _origin_is(
            _operator_origin(getattr(layer, "causal_conv1d_fn", None)),
            _QWEN35_ACCELERATED_CONV_PREFIXES,
        )
        for _, layer in layers
    )
    norm_fla_layers = sum(
        hasattr(layer, "norm") and _origin_is(_operator_origin(layer.norm), ("fla",)) for _, layer in layers
    )

    total = len(layers)
    core_missing: list[str] = []
    if total == 0:
        core_missing.append("qwen3.5 linear-attention layers")
    if prefill_fla_layers != total:
        core_missing.append("FLA chunk_gated_delta_rule prefill")
    if decode_fla_layers != total:
        core_missing.append("FLA fused_recurrent_gated_delta_rule decode")
    if norm_fla_layers != total:
        core_missing.append("FLA FusedRMSNormGated")

    conv_missing: list[str] = []
    if conv_prefill_fused_layers != total:
        conv_missing.append("causal_conv1d prefill")
    if conv_update_fused_layers != total:
        conv_missing.append("causal_conv1d cached update")

    conv_backend = "fallback"
    conv_origins = conv_prefill_origins + conv_update_origins
    if conv_origins and all(_origin_is(origin, ("causal_conv1d",)) for origin in conv_origins):
        conv_backend = "causal_conv1d"
    elif conv_origins and all(_origin_is(origin, _QWEN35_FLA_TRITON_CONV_PREFIXES) for origin in conv_origins):
        conv_backend = "fla_triton"
    elif not conv_missing:
        conv_backend = "mixed_accelerated"

    return {
        "qwen_linear_attention_layers": total,
        "qwen_fla_prefill_layers": prefill_fla_layers,
        "qwen_fla_decode_layers": decode_fla_layers,
        "qwen_causal_conv1d_prefill_layers": conv_prefill_fused_layers,
        "qwen_causal_conv1d_update_layers": conv_update_fused_layers,
        "qwen_fla_norm_layers": norm_fla_layers,
        "qwen_prefill_operator_origins": prefill_origins,
        "qwen_decode_operator_origins": decode_origins,
        "qwen_conv_prefill_operator_origins": conv_prefill_origins,
        "qwen_conv_update_operator_origins": conv_update_origins,
        "qwen_norm_operator_origins": norm_origins,
        "qwen_fla_core_contract_missing": core_missing,
        "qwen_fla_core_contract_pass": not core_missing,
        "qwen_sm70_recurrent_prefill_layers": sum(
            _origin_is(
                _operator_origin(getattr(layer, "chunk_gated_delta_rule", None)),
                _QWEN35_SM70_FLA_PREFIXES,
            )
            for _, layer in layers
        ),
        "qwen_causal_conv1d_contract_missing": conv_missing,
        "qwen_causal_conv1d_contract_pass": not conv_missing,
        "qwen_conv_backend_effective": conv_backend,
        "qwen_full_fused_contract_pass": not core_missing and not conv_missing,
        # Kept as the comparator-facing compatibility key. It intentionally
        # means the required FLA core, not optional causal-conv1d availability.
        "qwen_operator_contract_missing": core_missing,
        "qwen_operator_contract_pass": not core_missing,
    }


def enforce_qwen_backend(model, args: argparse.Namespace) -> dict[str, Any]:
    if args.model_kind != "qwen35":
        return {}
    contract = qwen_fla_operator_contract(model)
    if args.qwen_backend == "fla" and not contract["qwen_operator_contract_pass"]:
        missing = ", ".join(contract["qwen_operator_contract_missing"])
        raise RuntimeError(
            "Qwen3.5 FLA backend was required but Transformers bound fallback operators; "
            f"missing: {missing}. Install a card-compatible "
            "flash-linear-attention, PyTorch, and Triton stack."
        )
    if args.qwen_backend == "torch" and contract["qwen_operator_contract_pass"]:
        raise RuntimeError("Qwen3.5 torch backend was requested but FLA operators remain bound")
    return contract


def qwen_effective_backend(args: argparse.Namespace, contract: dict[str, Any]) -> str:
    if args.model_kind != "qwen35":
        return ""
    if contract.get("qwen_operator_contract_pass"):
        if contract.get("qwen_causal_conv1d_contract_pass"):
            if contract.get("qwen_conv_backend_effective") == "fla_triton":
                if contract.get("qwen_sm70_recurrent_prefill_layers"):
                    return "qwen_fla_recurrent_prefill_sm70_fla_triton_conv"
                return "qwen_fla_gated_delta_rule_fla_triton_conv"
            return "qwen_fla_gated_delta_rule"
        return "qwen_fla_gated_delta_rule_torch_conv"
    return "transformers_torch_fallback"


def last_rwkv_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def last_rwkv_prefill_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_prefill_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_prefill_backend", None)


def step_function(model, model_kind: str, batch_size: int) -> tuple[Callable[..., Any], str]:
    if model_kind == "rwkv":
        fast = getattr(model, "rwkv7_forward_token", None)
        if fast is None and batch_size == 1:
            fast = getattr(model, "rwkv7_forward_one", None)
        if fast is not None:
            return lambda token, state: fast(token, past_key_values=state), "rwkv_fast_token"
    return (
        lambda token, state: model(token, past_key_values=state, use_cache=True, logits_to_keep=1),
        "module_call",
    )


def forward_prefill(args: argparse.Namespace, model, ids):
    chunk_size = int(getattr(args, "prefill_chunk_size", 0) or 0)
    if chunk_size <= 0 or int(ids.shape[1]) <= chunk_size:
        return model(ids, use_cache=True, logits_to_keep=1)

    # RWKV exposes a cache-correct serving helper. Qwen follows the same HF
    # cache contract directly. This permits memory-safe, apples-to-apples
    # reruns of large (batch, prompt) cells without weakening the matrix key.
    rwkv_chunks = getattr(model, "rwkv7_prefill_chunks", None)
    if args.model_kind == "rwkv" and callable(rwkv_chunks):
        return rwkv_chunks(ids, chunk_size=chunk_size, logits_to_keep=1)

    out = None
    past = None
    for start in range(0, int(ids.shape[1]), chunk_size):
        out = model(
            ids[:, start : start + chunk_size],
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = out.past_key_values
    if out is None:
        raise RuntimeError("chunked prefill produced no output")
    return out


def timed_prefill(args: argparse.Namespace, model, ids) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = forward_prefill(args, model, ids)
    cuda_sync(args.device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(args.runs):
            cuda_sync(args.device)
            started = time.perf_counter()
            _ = forward_prefill(args, model, ids)
            cuda_sync(args.device)
            samples.append(time.perf_counter() - started)
    median_s = float(statistics.median(samples))
    return median_s, (args.batch_size * args.prompt_tokens) / median_s


def decode_once(args: argparse.Namespace, model, ids, step: Callable[..., Any]) -> tuple[float, Any]:
    with torch.inference_mode():
        out = forward_prefill(args, model, ids)
        state = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = step(token, state)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        started = time.perf_counter()
        for _ in range(args.decode_tokens):
            out = step(token, state)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
    return time.perf_counter() - started, state


def timed_decode(args: argparse.Namespace, model, ids) -> tuple[float, str, str | None, str]:
    step, step_backend = step_function(model, args.model_kind, args.batch_size)
    samples: list[float] = []
    state = None
    for _ in range(args.runs):
        elapsed, state = decode_once(args, model, ids, step)
        samples.append(elapsed)
    return (
        float(statistics.median(samples)),
        step_backend,
        last_rwkv_backend(model),
        type(state).__name__ if state is not None else "None",
    )


def save_backend_probe(args: argparse.Namespace, model, ids) -> dict[str, Any]:
    """Save deterministic logits and greedy tokens for cross-process checks."""

    step, _ = step_function(model, args.model_kind, 1)
    probe_ids = ids[:1]
    greedy_tokens: list[int] = []
    with torch.inference_mode():
        out = forward_prefill(args, model, probe_ids)
        state = out.past_key_values
        prompt_logits = out.logits[:, -1].float().cpu()
        token = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.probe_tokens):
            greedy_tokens.append(int(token[0, 0].item()))
            out = step(token, state)
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        final_logits = out.logits[:, -1].float().cpu()

    output = Path(args.probe_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": probe_ids.cpu(),
            "prompt_logits": prompt_logits,
            "final_logits": final_logits,
            "greedy_tokens": torch.tensor(greedy_tokens, dtype=torch.int64),
            "qwen_backend_requested": args.qwen_backend,
        },
        output,
    )
    return {
        "probe_output": str(output),
        "probe_tokens": args.probe_tokens,
        "probe_greedy_tokens": greedy_tokens,
    }


def environment_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    qwen_fast_path = None
    if args.model_kind == "qwen35":
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

            qwen_fast_path = bool(is_fast_path_available)
        except Exception:
            qwen_fast_path = False
    self_chunk_h_bv_effective = None
    self_chunk_h_bc_effective = None
    scan_block_m_effective = None
    if args.model_kind == "rwkv" and str(args.device).startswith("cuda"):
        try:
            from rwkv7_hf.native_jit import (
                _native_prefill_scan_block_m,
                _native_prefill_self_chunk_size,
            )
            from rwkv7_hf.self_chunk_h_fwd import resolve_chunk_h_tiles

            self_chunk_size = _native_prefill_self_chunk_size(
                int(args.batch_size), int(args.prompt_tokens)
            )
            self_chunk_h_bv_effective, self_chunk_h_bc_effective = resolve_chunk_h_tiles(
                torch.cuda.current_device(),
                self_chunk_size,
                batch_size=int(args.batch_size),
                tokens=int(args.prompt_tokens),
            )
            config = getattr(model, "config", None)
            hidden_size = int(getattr(config, "hidden_size"))
            num_heads = getattr(config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(config, "num_heads")
            num_heads = int(num_heads)
            scan_block_m_effective = _native_prefill_scan_block_m(
                hidden_size // num_heads,
                int(args.batch_size),
                int(args.prompt_tokens),
                hidden_size,
            )
        except Exception:
            pass
    capability = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        capability = list(torch.cuda.get_device_capability(cuda_device_index(args.device)))
    arch = f"sm_{capability[0]}{capability[1]}" if capability is not None else None
    qwen_device_route = None
    if args.model_kind == "qwen35" and arch is not None:
        qwen_device_route = "fla_triton_sm70" if capability == [7, 0] else f"fla_runtime_dispatch_{arch}"
    return {
        "device": device_name(args.device),
        "gpu_compute_capability": capability,
        "gpu_arch": arch,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": package_version("triton"),
        "transformers_version": package_version("transformers"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "fla_version": package_version("flash-linear-attention"),
        "causal_conv1d_version": package_version("causal-conv1d"),
        "qwen_fla_importable": importlib.util.find_spec("fla") is not None,
        "qwen_causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
        "qwen_force_torch": QWEN_FORCE_TORCH,
        "qwen_fast_path_available": qwen_fast_path,
        "qwen_fla_expected_device_route": qwen_device_route,
        "rwkv_fast_token_backend_requested": os.environ.get("RWKV7_FAST_TOKEN_BACKEND"),
        "rwkv_fast_token_quant_requested": os.environ.get("RWKV7_FAST_TOKEN_QUANT"),
        "rwkv_fast_prefill_requested": os.environ.get("RWKV7_FAST_PREFILL"),
        "rwkv_fast_prefill_quant_requested": os.environ.get("RWKV7_FAST_PREFILL_QUANT"),
        "rwkv_prefill_graph_requested": os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH"),
        "rwkv_prefill_fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN"),
        "rwkv_prefill_external_quant_graph_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH"
        ),
        "rwkv_prefill_blas_requested": os.environ.get("RWKV7_NATIVE_PREFILL_BLAS"),
        "rwkv_prefill_self_chunk_requested": os.environ.get("RWKV7_NATIVE_PREFILL_SELF_CHUNK"),
        "rwkv_prefill_self_chunk_min_tokens_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS"
        ),
        "rwkv_prefill_self_chunk_size_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE"
        ),
        "rwkv_prefill_self_chunk_safe_gate_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_SAFE_GATE"
        ),
        "rwkv_prefill_self_chunk_h_bv_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BV"
        ),
        "rwkv_prefill_self_chunk_h_bc_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_H_BC"
        ),
        "rwkv_prefill_self_chunk_h_bv_effective": self_chunk_h_bv_effective,
        "rwkv_prefill_self_chunk_h_bc_effective": self_chunk_h_bc_effective,
        "rwkv_prefill_scan_block_m_requested": os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M"),
        "rwkv_prefill_scan_block_m_effective": scan_block_m_effective,
        "rwkv_prefill_scan_num_warps_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS"
        ),
        "rwkv_native_bnb8_direct_requested": os.environ.get("RWKV7_NATIVE_BNB8_DIRECT"),
        "rwkv_native_bnb8_relu_quant_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_RELU_QUANT"
        ),
        "rwkv_native_bnb8_rkv_mix_quant_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT"
        ),
        "rwkv_native_bnb8_ffn_mix_quant_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT"
        ),
        "rwkv_native_bnb8_attn_mix_block_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK"
        ),
        "rwkv_native_bnb8_ffn_mix_block_requested": os.environ.get(
            "RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK"
        ),
        "bnb_int8_threshold_requested": (
            float(os.environ["RWKV7_BNB_INT8_THRESHOLD"])
            if args.quantization in {"bnb8", "bnb8_a8w8_head"}
            and "RWKV7_BNB_INT8_THRESHOLD" in os.environ
            else None
        ),
    }


def effective_quantization_metadata(model, args: argparse.Namespace) -> dict[str, Any]:
    quantizer = getattr(model, "hf_quantizer", None)
    config = getattr(quantizer, "quantization_config", None)
    if config is None:
        config = getattr(getattr(model, "config", None), "quantization_config", None)
    getter = config.get if isinstance(config, dict) else lambda name, default=None: getattr(config, name, default)
    if args.quantization == "bnb8_a8w8_head":
        backend = "bitsandbytes+rwkv_native"
    elif args.quantization.startswith("bnb"):
        backend = "bitsandbytes"
    elif args.quantization.startswith("torchao"):
        backend = "torchao"
    elif args.quantization in {"a8w8", "mm8", "mm4"}:
        backend = "rwkv_native"
    else:
        backend = "dense"
    native_quant = args.quantization in {
        "a8w8",
        "mm8",
        "mm4",
        "torchao_w8",
        "torchao_w4",
        "bnb8_a8w8_head",
    }
    a8w8_effective_rows = None
    mm4_launch: dict[str, int] = {}
    if args.quantization in {"a8w8", "bnb8_a8w8_head"}:
        from rwkv7_hf.native_quant_a8w8 import a8w8_gemv_max_rows

        try:
            native_device = next(model.parameters()).device
        except Exception:
            native_device = None
        a8w8_effective_rows = a8w8_gemv_max_rows(native_device)
    elif args.quantization == "mm4":
        from rwkv7_hf.native_quant_mm4 import mm4_effective_launch_config

        try:
            native_device = next(model.parameters()).device
        except Exception:
            native_device = None
        mm4_launch = mm4_effective_launch_config(native_device)
    native_jit_module = None
    if args.model_kind == "rwkv":
        method = getattr(model, "rwkv7_prefill_native", None)
        fn = getattr(method, "__func__", method)
        globals_dict = getattr(fn, "__globals__", {})
        native_jit_module = globals_dict.get("native_jit")
        if native_jit_module is None:
            # The production remote-code wrapper imports individual helpers as
            # ``_native_jit_prefill`` rather than retaining the module object.
            # Resolve the exact dynamically loaded module through that helper
            # so telemetry follows the code used by this model instance.
            prefill_fn = globals_dict.get("_native_jit_prefill")
            native_jit_module = sys.modules.get(getattr(prefill_fn, "__module__", ""))
        if native_jit_module is None:
            # Some Transformers remote-code loaders wrap/copy imported
            # callables without preserving their original module in
            # ``sys.modules``.  ``--rwkv-code-source repo`` guarantees this
            # fallback is byte-identical to the files overlaid on the model.
            from rwkv7_hf import native_jit as native_jit_module

    def bnb8_flag(env_name: str, policy_name: str) -> bool | None:
        if args.quantization not in {"bnb8", "bnb8_a8w8_head"} or native_jit_module is None:
            return None
        return bool(native_jit_module._native_bnb8_policy_flag(env_name, policy_name))

    def bnb8_block(env_name: str, policy_name: str, fallback: int) -> int | None:
        if args.quantization not in {"bnb8", "bnb8_a8w8_head"} or native_jit_module is None:
            return None
        return int(native_jit_module._native_bnb8_policy_block(env_name, policy_name, fallback))

    return {
        "bnb_int8_threshold": (
            float(getter("llm_int8_threshold", 6.0))
            if args.quantization in {"bnb8", "bnb8_a8w8_head"}
            else None
        ),
        "rwkv_bnb_skip_policy": (
            getattr(model, "_rwkv7_bnb_skip_policy", None) if args.model_kind == "rwkv" else None
        ),
        "rwkv_bnb_prefill_value_stride": (
            int(os.environ.get("RWKV7_BNB_PREFILL_VALUE_STRIDE", "8"))
            if args.model_kind == "rwkv" and args.quantization.startswith("bnb")
            else None
        ),
        # Record resolved hardware-policy values as well as the raw requested
        # environment variables in ``runtime_metadata``.  This makes a result
        # produced with zero tuning variables fully reproducible and proves
        # that the exact-card defaults, rather than shell-only overrides, were
        # active during the acceptance run.
        "rwkv_native_bnb8_direct_effective": bnb8_flag(
            "RWKV7_NATIVE_BNB8_DIRECT", "native_bnb8_direct"
        ),
        "rwkv_native_bnb8_relu_quant_effective": bnb8_flag(
            "RWKV7_NATIVE_BNB8_RELU_QUANT", "native_bnb8_relu_quant"
        ),
        "rwkv_native_bnb8_rkv_mix_quant_effective": bnb8_flag(
            "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT", "native_bnb8_rkv_mix_quant"
        ),
        "rwkv_native_bnb8_ffn_mix_quant_effective": bnb8_flag(
            "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT", "native_bnb8_ffn_mix_quant"
        ),
        "rwkv_native_bnb8_attn_mix_block_effective": bnb8_block(
            "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK", "native_bnb8_attn_mix_block", 1024
        ),
        "rwkv_native_bnb8_ffn_mix_block_effective": bnb8_block(
            "RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK", "native_bnb8_ffn_mix_block", 1024
        ),
        "quantization_backend": backend,
        "quantized_modules": getattr(model, "_rwkv7_cross_model_quant_replaced_modules", None),
        "native_quant_block_modules": getattr(
            model, "_rwkv7_native_mm_block_replaced_modules", None
        ),
        # Capture every native-kernel launch knob that can change an acceptance
        # row. Missing values mean the documented kernel default was used.
        "a8w8_gemv_max_rows": (
            a8w8_effective_rows
            if args.quantization in {"a8w8", "bnb8_a8w8_head"}
            else None
        ),
        "a8w8_gemv_block_k": (
            int(os.environ.get("RWKV7_A8W8_GEMV_BLOCK_K", "256"))
            if args.quantization in {"a8w8", "bnb8_a8w8_head"}
            else None
        ),
        "a8w8_gemv_block_n": (
            int(os.environ.get("RWKV7_A8W8_GEMV_BLOCK_N", "64"))
            if args.quantization in {"a8w8", "bnb8_a8w8_head"}
            else None
        ),
        "a8w8_gemv_warps": (
            int(os.environ.get("RWKV7_A8W8_GEMV_WARPS", "1"))
            if args.quantization in {"a8w8", "bnb8_a8w8_head"}
            else None
        ),
        "mm4_fused_max_rows": (
            mm4_launch.get("fused_max_rows")
            if args.quantization == "mm4"
            else None
        ),
        "mm4_gemv_block_pairs": (
            mm4_launch.get("gemv_block_pairs") if args.quantization == "mm4" else None
        ),
        "mm4_gemv_block_n": (
            mm4_launch.get("gemv_block_n") if args.quantization == "mm4" else None
        ),
        "mm4_dot_min_rows": (
            mm4_launch.get("dot_min_rows") if args.quantization == "mm4" else None
        ),
        "mm4_dot_block_b": (
            mm4_launch.get("dot_block_b")
            if args.quantization == "mm4"
            else None
        ),
        "mm4_dot_block_pairs": (
            mm4_launch.get("dot_block_pairs")
            if args.quantization == "mm4"
            else None
        ),
        "mm4_dot_block_n": (
            mm4_launch.get("dot_block_n")
            if args.quantization == "mm4"
            else None
        ),
        "mm4_dot_warps": (
            mm4_launch.get("dot_warps")
            if args.quantization == "mm4"
            else None
        ),
        "native_quant_kernel_active": native_quant,
    }


_QWEN35_FAST_BINDING_PREFIXES = {
    "causal_conv1d_fn": ("causal_conv1d.", "bench.qwen35_fla_triton_conv", "qwen35_fla_triton_conv"),
    "causal_conv1d_update": ("causal_conv1d.", "bench.qwen35_fla_triton_conv", "qwen35_fla_triton_conv"),
    "chunk_gated_delta_rule": (
        "fla.",
        "bench.qwen35_sm70_fla",
        "qwen35_sm70_fla",
    ),
    "recurrent_gated_delta_rule": ("fla.",),
}


def qwen35_fast_path_bindings(model) -> dict[str, Any]:
    """Verify the operators bound by live Qwen3.5 GatedDeltaNet layers.

    The Transformers module-level ``is_fast_path_available`` flag proves that
    optional packages imported, but not that a loaded layer retained those
    callables.  Inspecting the live bindings makes optimized-Qwen comparison a
    fail-closed contract rather than an availability hint.
    """

    layers = [
        module
        for module in model.modules()
        if all(hasattr(module, name) for name in _QWEN35_FAST_BINDING_PREFIXES)
    ]
    bindings: dict[str, str | None] = {}
    if layers:
        first = layers[0]
        for name in _QWEN35_FAST_BINDING_PREFIXES:
            fn = getattr(first, name, None)
            bindings[name] = getattr(fn, "__module__", None) if callable(fn) else None
    verified = bool(layers) and all(
        isinstance(bindings.get(name), str)
        and any(str(bindings[name]).startswith(prefix) for prefix in prefixes)
        for name, prefixes in _QWEN35_FAST_BINDING_PREFIXES.items()
    )
    return {
        "verified": verified,
        "layer_count": len(layers),
        "bindings": bindings,
    }


def validate_loaded_model(args: argparse.Namespace, model) -> None:
    if args.model_kind != "qwen35" or not args.require_qwen_fast_path:
        return
    binding_check = qwen35_fast_path_bindings(model)
    if not bool(binding_check["verified"]):
        raise RuntimeError(
            "Qwen3.5 full optimized path was required but the loaded GatedDeltaNet "
            f"layers are not bound to FLA plus accelerated causal conv: {binding_check}"
        )


def benchmark_loaded(
    args: argparse.Namespace,
    tokenizer,
    model,
    *,
    load_s: float,
    qwen_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qwen_contract = qwen_contract or {}
    input_device = str(next(model.parameters()).device)
    ids = build_exact_prompt(tokenizer, args.prompt_tokens, args.batch_size, input_device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cuda_device_index(args.device))
    prefill_s, prefill_tokps = timed_prefill(args, model, ids)
    prefill_backend = last_rwkv_prefill_backend(model) if args.model_kind == "rwkv" else None
    prefill_clampw_scan = (
        bool(getattr(model, "_rwkv7_native_prefill_clampw_scan_effective", False))
        if args.model_kind == "rwkv"
        else None
    )
    prefill_stacked_rkv = (
        bool(getattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", False))
        if args.model_kind == "rwkv"
        else None
    )
    prefill_self_chunk = (
        bool(getattr(model, "_rwkv7_native_prefill_self_chunk_effective", False))
        if args.model_kind == "rwkv"
        else None
    )
    prefill_sequence_ffn = (
        bool(getattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", False))
        if args.model_kind == "rwkv"
        else None
    )
    decode_s, step_backend, effective_backend, cache_type = timed_decode(args, model, ids)
    logits_finite = True
    with torch.inference_mode():
        check = forward_prefill(args, model, ids[:, : min(8, ids.shape[1])])
        logits_finite = bool(torch.isfinite(check.logits).all().item())
    if not logits_finite:
        raise RuntimeError("model produced non-finite logits")
    probe_metadata = save_backend_probe(args, model, ids) if args.probe_output else {}

    qwen_bindings = qwen35_fast_path_bindings(model) if args.model_kind == "qwen35" else None
    footprint = model_footprint_mb(model)
    peak = peak_mb(args.device)
    runtime_working_set = round(max(0.0, peak - footprint), 1) if peak is not None else None
    parameter_metadata = model_parameter_metadata(model, args)
    active_parameters = int(parameter_metadata["active_parameter_count"])
    active_parameter_billions = active_parameters / 1e9
    decode_tokps = (args.batch_size * args.decode_tokens) / decode_s
    row = {
        **base_row(args),
        **model_metadata(args, model),
        **environment_metadata(args, model),
        **effective_quantization_metadata(model, args),
        **parameter_metadata,
        **qwen_contract,
        **probe_metadata,
        "status": "pass",
        "input_device": input_device,
        "prefill_sec_median": round(prefill_s, 6),
        "prefill_tokps_total": round(prefill_tokps, 3),
        "prefill_tokps_per_active_billion": round(
            prefill_tokps / active_parameter_billions, 6
        ),
        "prefill_active_parameter_tops": round(prefill_tokps * active_parameters / 1e12, 6),
        "decode_sec_median": round(decode_s, 6),
        "decode_tokps_total": round(decode_tokps, 3),
        "decode_tokps_per_seq": round(args.decode_tokens / decode_s, 3),
        "decode_tokps_per_active_billion": round(
            decode_tokps / active_parameter_billions, 6
        ),
        "decode_ms_per_step": round(1000 * decode_s / args.decode_tokens, 6),
        "decode_active_parameter_tops": round(decode_tokps * active_parameters / 1e12, 6),
        "step_backend": step_backend,
        "prefill_effective_backend": prefill_backend or ("module_call" if args.model_kind == "qwen35" else None),
        "prefill_backend_effective": prefill_backend or ("module_call" if args.model_kind == "qwen35" else None),
        "rwkv_prefill_clampw_scan_effective": prefill_clampw_scan,
        "rwkv_prefill_stacked_rkv_effective": prefill_stacked_rkv,
        "rwkv_prefill_self_chunk_effective": prefill_self_chunk,
        "rwkv_prefill_sequence_ffn_effective": prefill_sequence_ffn,
        "effective_backend": qwen_effective_backend(args, qwen_contract) or effective_backend or step_backend,
        "qwen_fast_path_verified": qwen_bindings["verified"] if qwen_bindings is not None else None,
        "qwen_fast_path_layer_count": qwen_bindings["layer_count"] if qwen_bindings is not None else None,
        "qwen_fast_path_bindings": qwen_bindings["bindings"] if qwen_bindings is not None else None,
        "cache_type": cache_type,
        "model_footprint_mb": footprint,
        "peak_vram_mb": peak,
        "runtime_working_set_mb": runtime_working_set,
        "load_s": round(load_s, 3),
        "logits_finite": logits_finite,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    return row


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    started = time.perf_counter()
    effective_model_path = args.model
    temporary = None
    model = None
    try:
        if args.model_kind == "rwkv":
            effective_model_path, temporary = prepare_rwkv_model_dir(args.model, args.rwkv_code_source)
        tokenizer = AutoTokenizer.from_pretrained(
            effective_model_path,
            trust_remote_code=args.model_kind == "rwkv",
        )
        model = load_model(args, dtype, effective_model_path)
        qwen_contract = enforce_qwen_backend(model, args)
        validate_loaded_model(args, model)
        load_s = time.perf_counter() - started
        return benchmark_loaded(
            args,
            tokenizer,
            model,
            load_s=load_s,
            qwen_contract=qwen_contract,
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if temporary is not None:
            temporary.cleanup()


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-kind", required=True, choices=["rwkv", "qwen35"])
    ap.add_argument("--model-role", required=True, choices=["candidate", "reference"])
    ap.add_argument("--model-pair", required=True)
    ap.add_argument("--model-size-label", required=True)
    ap.add_argument("--benchmark-matrix", default="qwen35_hf")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument(
        "--quantization",
        default="none",
        choices=[
            "none",
            "bnb8",
            "bnb4",
            "bnb8_a8w8_head",
            "torchao_w8",
            "torchao_w4",
            "a8w8",
            "mm8",
            "mm4",
        ],
    )
    ap.add_argument("--native-quant-min-params", type=int, default=1_000_000)
    ap.add_argument("--native-quant-policy", choices=["memory", "speed"], default="memory")
    ap.add_argument("--torchao-group-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Split prefill into cache-carrying chunks; 0 benchmarks one full call.",
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rwkv-attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    ap.add_argument("--rwkv-code-source", choices=["repo", "model"], default="repo")
    ap.add_argument(
        "--qwen-backend",
        choices=["auto", "fla", "torch"],
        default="fla",
        help="Require verified FLA operators by default; torch is an explicit diagnostic fallback lane",
    )
    ap.add_argument(
        "--qwen-conv-backend",
        choices=["auto", "causal_conv1d", "fla_triton"],
        default="auto",
        help="Select the Qwen causal-conv implementation independently of the FLA core",
    )
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument("--results", default="")
    ap.add_argument("--probe-output", default="")
    ap.add_argument("--probe-tokens", type=int, default=8)
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args()
    validate_args(args)
    return args


def main() -> int:
    args = parse_args()
    try:
        row = benchmark(args)
    except Exception as exc:
        row = failure_row(args, exc)
        append_row(args.results, row)
        print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        if not args.optional:
            raise
        return 0
    append_row(args.results, row)
    print("QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
