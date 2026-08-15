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
    os.environ.get("RWKV7_QWEN35_FORCE_TORCH", "0").lower()
    in {"1", "true", "yes", "on"}
    or QWEN_BACKEND_BOOTSTRAP == "torch"
)
if QWEN_FORCE_TORCH:
    sys.path[:] = [
        path
        for path in sys.path
        if "flash-linear-attention" not in path.replace("\\", "/").lower()
    ]
    _original_find_spec = importlib.util.find_spec

    def _find_spec_without_fla(name: str, *args, **kwargs):
        if name == "fla" or name.startswith("fla."):
            return None
        return _original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec_without_fla

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
QWEN_STATIC_GRAPH_ROUTES = {
    "static_cache_inductor_cudagraph",
    "static_cache_raw_cudagraph",
}


def _is_finite_real_number(value: Any) -> bool:
    """Return true only for finite int/float telemetry, never bool sentinels."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


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
    rwkv_implementation = str(getattr(args, "rwkv_implementation", "auto"))
    if rwkv_implementation not in {"auto", "wrapper_repo"}:
        raise ValueError("--rwkv-implementation must be auto or wrapper_repo")
    if rwkv_implementation == "wrapper_repo" and (
        args.model_kind != "rwkv"
        or str(getattr(args, "rwkv_code_source", "repo")) != "repo"
    ):
        raise ValueError(
            "--rwkv-implementation wrapper_repo requires --model-kind rwkv "
            "and --rwkv-code-source repo"
        )
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
    qwen_sdpa_policy = str(getattr(args, "qwen_sdpa_policy", "auto"))
    if qwen_sdpa_policy not in {"auto", "math_only"}:
        raise ValueError("--qwen-sdpa-policy must be auto or math_only")
    if args.model_kind != "qwen35" and qwen_sdpa_policy != "auto":
        raise ValueError("--qwen-sdpa-policy is only valid for Qwen3.5")
    cross_cache_greedy_policy = str(
        getattr(args, "qwen_cross_cache_full_greedy_policy", "strict")
    )
    if cross_cache_greedy_policy not in {"strict", "informational"}:
        raise ValueError(
            "--qwen-cross-cache-full-greedy-policy must be strict or informational"
        )
    if args.model_kind != "qwen35" and cross_cache_greedy_policy != "strict":
        raise ValueError(
            "--qwen-cross-cache-full-greedy-policy is only valid for Qwen3.5"
        )
    qwen_conv_backend = str(getattr(args, "qwen_conv_backend", "auto"))
    if qwen_conv_backend not in {"auto", "causal_conv1d", "fla_triton"}:
        raise ValueError(
            "--qwen-conv-backend must be auto, causal_conv1d, or fla_triton"
        )
    if args.qwen_backend == "torch" and qwen_conv_backend != "auto":
        raise ValueError(
            "an accelerated Qwen conv backend cannot be combined with --qwen-backend torch"
        )
    qwen_decode_optimization = str(
        getattr(args, "qwen_decode_optimization", "module_call_dynamic")
    )
    if qwen_decode_optimization not in {
        "module_call_dynamic",
        *QWEN_STATIC_GRAPH_ROUTES,
    }:
        raise ValueError(
            "--qwen-decode-optimization must be module_call_dynamic or "
            "static_cache_inductor_cudagraph, or static_cache_raw_cudagraph"
        )
    if qwen_decode_optimization in QWEN_STATIC_GRAPH_ROUTES:
        if qwen_decode_optimization == "static_cache_inductor_cudagraph":
            qwen_compile_mode = str(getattr(args, "qwen_compile_mode", "max-autotune"))
            if qwen_compile_mode not in {"reduce-overhead", "max-autotune"}:
                raise ValueError(
                    "--qwen-compile-mode must be reduce-overhead or max-autotune"
                )
        qwen_graph_conv_contract = (
            "fla_triton"
            if str(getattr(args, "benchmark_matrix", ""))
            == "qwen35_v100_best_optimized_hf_v1"
            else "causal_conv1d"
        )
        graph_requirements = {
            "model_kind": (args.model_kind, "qwen35"),
            "model_role": (args.model_role, "reference"),
            "device": (str(args.device).split(":", 1)[0], "cuda"),
            "dtype": (args.dtype, "fp16"),
            "quantization": (args.quantization, "none"),
            "qwen_backend": (args.qwen_backend, "fla"),
            "qwen_conv_backend": (qwen_conv_backend, qwen_graph_conv_contract),
            "require_qwen_fast_path": (
                bool(getattr(args, "require_qwen_fast_path", False)),
                True,
            ),
            "optimization_lane": (
                str(getattr(args, "optimization_lane", "") or ""),
                "qwen_best_optimized_hf",
            ),
        }
        mismatches = [
            f"{field}={actual!r} (expected {expected!r})"
            for field, (actual, expected) in graph_requirements.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                f"{qwen_decode_optimization} is a strict Qwen reference lane: "
                + "; ".join(mismatches)
            )
        if int(getattr(args, "qwen_graph_probe_tokens", 16)) <= 0:
            raise ValueError("--qwen-graph-probe-tokens must be positive")
    if args.probe_output and args.probe_tokens <= 0:
        raise ValueError("--probe-tokens must be positive when --probe-output is set")
    probe_batch_size = int(getattr(args, "probe_batch_size", 1))
    if args.probe_output and not 1 <= probe_batch_size <= int(args.batch_size):
        raise ValueError(
            "--probe-batch-size must be in [1, --batch-size] when --probe-output is set"
        )


def build_exact_prompt(
    tokenizer, prompt_tokens: int, batch_size: int, device: str
) -> torch.Tensor:
    encoded = tokenizer(
        PROMPT_SEED * 32, return_tensors="pt", add_special_tokens=False
    ).input_ids
    if encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] == 0:
        raise RuntimeError(
            f"tokenizer returned invalid input shape {tuple(encoded.shape)}"
        )
    repeats = (prompt_tokens + int(encoded.shape[1]) - 1) // int(encoded.shape[1])
    ids = encoded.repeat(1, repeats)[:, :prompt_tokens].repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def model_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    config = getattr(model, "config", None)
    implementation_effective = None
    if args.model_kind == "rwkv" and model is not None:
        implementation_effective = getattr(
            model, "_rwkv7_benchmark_implementation_effective", None
        )
        if implementation_effective is None:
            class_name = type(model).__name__
            if class_name == "NativeRWKV7ForCausalLM":
                implementation_effective = "native_model"
            elif class_name == "RWKV7ForCausalLM":
                implementation_effective = "wrapper_repo"
    return {
        "model_name": Path(args.model).name,
        "model_id_or_path": args.model,
        "model_size_label": args.model_size_label,
        "model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(
            config, "num_attention_heads", getattr(config, "num_heads", None)
        ),
        "head_dim": getattr(config, "head_dim", None),
        "rwkv_implementation_effective": implementation_effective,
    }


def base_row(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": args.benchmark_matrix,
        "benchmark_repository_commit": os.environ.get("REPOSITORY_COMMIT"),
        "optimization_lane": str(getattr(args, "optimization_lane", "") or ""),
        "model_pair": args.model_pair,
        "model_role": args.model_role,
        "model_kind": args.model_kind,
        "rwkv_implementation_requested": str(
            getattr(args, "rwkv_implementation", "auto")
        )
        if args.model_kind == "rwkv"
        else None,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "qwen_backend_requested": args.qwen_backend,
        "qwen_conv_backend_requested": getattr(args, "qwen_conv_backend", "auto"),
        "qwen_sdpa_policy_requested": (
            str(getattr(args, "qwen_sdpa_policy", "auto"))
            if args.model_kind == "qwen35"
            else None
        ),
        "qwen_cross_cache_full_greedy_policy_requested": (
            str(getattr(args, "qwen_cross_cache_full_greedy_policy", "strict"))
            if args.model_kind == "qwen35"
            else None
        ),
        "qwen_fast_path_required": bool(getattr(args, "require_qwen_fast_path", False)),
        "qwen_decode_optimization_requested": str(
            getattr(args, "qwen_decode_optimization", "module_call_dynamic")
        ),
        "qwen_compile_mode_requested": (
            str(getattr(args, "qwen_compile_mode", "max-autotune"))
            if str(getattr(args, "qwen_decode_optimization", "module_call_dynamic"))
            == "static_cache_inductor_cudagraph"
            else None
        ),
        "qwen_graph_logits_probe_tokens_requested": int(
            getattr(args, "qwen_graph_probe_tokens", 16)
        ),
        "timing_statistic": "median",
        "mtp_enabled": False,
        "speculative_decoding_enabled": False,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "prefill_chunk_size": int(getattr(args, "prefill_chunk_size", 0) or 0),
        "native_quant_min_params_requested": int(
            getattr(args, "native_quant_min_params", 1_000_000)
        ),
        "native_quant_policy_requested": str(
            getattr(args, "native_quant_policy", "memory")
        ),
        "torchao_group_size_requested": int(getattr(args, "torchao_group_size", 128)),
        **model_metadata(args),
    }


def failure_row(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    row = {
        **base_row(args),
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": repr(exc),
    }
    parity = getattr(exc, "qwen_graph_parity", None)
    if isinstance(parity, dict):
        row.update(parity)
    return row


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def configure_qwen_sdpa_policy(args: argparse.Namespace) -> None:
    """Apply and verify the explicit Qwen SDPA backend policy."""

    if args.model_kind != "qwen35":
        return
    policy = str(getattr(args, "qwen_sdpa_policy", "auto"))
    enabled = policy == "auto"
    setters = (
        ("enable_flash_sdp", enabled),
        ("enable_mem_efficient_sdp", enabled),
        ("enable_math_sdp", True),
        ("enable_cudnn_sdp", enabled),
    )
    for name, value in setters:
        setter = getattr(torch.backends.cuda, name, None)
        if callable(setter):
            setter(value)

    expected = {
        "flash_sdp_enabled": enabled,
        "mem_efficient_sdp_enabled": enabled,
        "math_sdp_enabled": True,
        "cudnn_sdp_enabled": enabled,
    }
    mismatches: list[str] = []
    for name, value in expected.items():
        getter = getattr(torch.backends.cuda, name, None)
        if callable(getter) and bool(getter()) is not value:
            mismatches.append(f"{name}={bool(getter())!r}, expected {value!r}")
    if mismatches:
        raise RuntimeError(
            f"Qwen SDPA policy {policy!r} was not applied: " + "; ".join(mismatches)
        )
    setattr(args, "_qwen_sdpa_policy_effective", policy)


def device_map_for(device: str):
    return {"": cuda_device_index(device)} if device.startswith("cuda") else None


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(cuda_device_index(device))
    return device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return round(
        torch.cuda.max_memory_allocated(cuda_device_index(device)) / 1024 / 1024, 1
    )


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
    total = sum(
        _logical_parameter_numel(parameter) for _name, parameter in unique.values()
    )
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


def prepare_rwkv_model_dir(
    model_path: str, code_source: str
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if code_source == "model":
        return model_path, None
    source = Path(model_path).resolve()
    repo_code = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    if not source.is_dir():
        raise ValueError(
            "--rwkv-code-source repo requires a local converted model directory"
        )
    temporary = tempfile.TemporaryDirectory(
        prefix="rwkv7_qwen35_repo_code_", dir=source.parent
    )
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


def load_model(
    args: argparse.Namespace, dtype: torch.dtype, model_path: str | None = None
):
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map_for(args.device),
        "low_cpu_mem_usage": True,
    }
    qconfig = quantization_config(args, dtype)
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    if args.model_kind == "rwkv":
        implementation = str(getattr(args, "rwkv_implementation", "auto"))
        if implementation == "wrapper_repo":
            if str(getattr(args, "rwkv_code_source", "repo")) != "repo":
                raise ValueError(
                    "wrapper_repo loading requires --rwkv-code-source repo"
                )
            from rwkv7_hf.configuration_rwkv7 import RWKV7Config
            from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

            # Bypass converted config auto_map explicitly. The canonical model
            # directory remains the weight/config source and recorded identity;
            # importing the repository classes directly makes the FLA wrapper
            # implementation auditable without rewriting model inputs.
            canonical_model_path = str(Path(args.model).resolve(strict=True))
            config = RWKV7Config.from_pretrained(canonical_model_path)
            previous_native_model = os.environ.get("RWKV7_NATIVE_MODEL")
            os.environ["RWKV7_NATIVE_MODEL"] = "0"
            try:
                model = RWKV7ForCausalLM.from_pretrained(
                    canonical_model_path,
                    config=config,
                    **kwargs,
                ).eval()
            finally:
                if previous_native_model is None:
                    os.environ.pop("RWKV7_NATIVE_MODEL", None)
                else:
                    os.environ["RWKV7_NATIVE_MODEL"] = previous_native_model
            setattr(model, "_rwkv7_benchmark_implementation_effective", "wrapper_repo")
        else:
            kwargs["trust_remote_code"] = True
            model = AutoModelForCausalLM.from_pretrained(
                model_path or args.model, **kwargs
            ).eval()
            effective = (
                "native_model"
                if type(model).__name__ == "NativeRWKV7ForCausalLM"
                else "wrapper_repo"
                if type(model).__name__ == "RWKV7ForCausalLM"
                else "unknown"
            )
            setattr(model, "_rwkv7_benchmark_implementation_effective", effective)
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
                from rwkv7_hf.native_quant_a8w8 import (
                    quantize_model_a8w8 as quantize_model,
                )
            elif args.quantization == "mm8":
                from rwkv7_hf.native_quant_mm8 import (
                    quantize_model_mm8 as quantize_model,
                )
            else:
                from rwkv7_hf.native_quant_mm4 import (
                    quantize_model_mm4 as quantize_model,
                )
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
        raise RuntimeError(
            "installed Transformers does not provide Qwen3_5ForCausalLM"
        ) from exc
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, **kwargs).eval()
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
    return any(
        origin == prefix or origin.startswith(prefix + ".") for prefix in prefixes
    )


_QWEN35_FLA_TRITON_CONV_PREFIXES = (
    "bench.qwen35_fla_triton_conv",
    "qwen35_fla_triton_conv",
)
_QWEN35_ACCELERATED_CONV_PREFIXES = (
    "causal_conv1d",
) + _QWEN35_FLA_TRITON_CONV_PREFIXES


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

    prefill_origins = sorted(
        {
            _operator_origin(getattr(layer, "chunk_gated_delta_rule", None))
            for _, layer in layers
        }
    )
    decode_origins = sorted(
        {
            _operator_origin(getattr(layer, "recurrent_gated_delta_rule", None))
            for _, layer in layers
        }
    )
    conv_update_origins = sorted(
        {
            _operator_origin(getattr(layer, "causal_conv1d_update", None))
            for _, layer in layers
        }
    )
    conv_prefill_origins = sorted(
        {
            _operator_origin(getattr(layer, "causal_conv1d_fn", None))
            for _, layer in layers
        }
    )
    norm_origins = sorted(
        {_operator_origin(layer.norm) for _, layer in layers if hasattr(layer, "norm")}
    )

    prefill_fla_layers = sum(
        _origin_is(
            _operator_origin(getattr(layer, "chunk_gated_delta_rule", None)), ("fla",)
        )
        for _, layer in layers
    )
    decode_fla_layers = sum(
        _origin_is(
            _operator_origin(getattr(layer, "recurrent_gated_delta_rule", None)),
            ("fla",),
        )
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
        hasattr(layer, "norm") and _origin_is(_operator_origin(layer.norm), ("fla",))
        for _, layer in layers
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
    if conv_origins and all(
        _origin_is(origin, ("causal_conv1d",)) for origin in conv_origins
    ):
        conv_backend = "causal_conv1d"
    elif conv_origins and all(
        _origin_is(origin, _QWEN35_FLA_TRITON_CONV_PREFIXES) for origin in conv_origins
    ):
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
        raise RuntimeError(
            "Qwen3.5 torch backend was requested but FLA operators remain bound"
        )
    requested_conv = str(getattr(args, "qwen_conv_backend", "auto"))
    effective_conv = str(contract.get("qwen_conv_backend_effective", "fallback"))
    if requested_conv != "auto" and effective_conv != requested_conv:
        raise RuntimeError(
            "Qwen3.5 causal-conv backend mismatch: "
            f"requested {requested_conv!r}, but live layers bound {effective_conv!r}."
        )
    if bool(getattr(args, "require_qwen_fast_path", False)) and not bool(
        contract.get("qwen_full_fused_contract_pass")
    ):
        missing = list(contract.get("qwen_fla_core_contract_missing", [])) + list(
            contract.get("qwen_causal_conv1d_contract_missing", [])
        )
        raise RuntimeError(
            "Qwen3.5 full optimized path was required but the live operator contract failed; "
            f"missing: {', '.join(missing) or 'unknown operators'}."
        )
    if requested_conv == "causal_conv1d" and bool(
        getattr(args, "require_qwen_fast_path", False)
    ):
        official = qwen_official_fast_path_environment()
        missing = [name for name, available in official.items() if not available]
        if missing:
            raise RuntimeError(
                "Qwen3.5 official FLA + causal_conv1d path was required but the "
                f"environment gate failed: {', '.join(missing)}."
            )
    return contract


def qwen_effective_backend(args: argparse.Namespace, contract: dict[str, Any]) -> str:
    if args.model_kind != "qwen35":
        return ""
    if contract.get("qwen_operator_contract_pass"):
        if contract.get("qwen_causal_conv1d_contract_pass"):
            if contract.get("qwen_conv_backend_effective") == "fla_triton":
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


def step_function(
    model, model_kind: str, batch_size: int
) -> tuple[Callable[..., Any], str]:
    if model_kind == "rwkv":
        fast = getattr(model, "rwkv7_forward_token", None)
        if fast is None and batch_size == 1:
            fast = getattr(model, "rwkv7_forward_one", None)
        if fast is not None:
            return lambda token, state: fast(
                token, past_key_values=state
            ), "rwkv_fast_token"
    return (
        lambda token, state: model(
            token, past_key_values=state, use_cache=True, logits_to_keep=1
        ),
        "module_call",
    )


def forward_prefill(
    args: argparse.Namespace,
    model,
    ids,
    *,
    past_key_values=None,
):
    chunk_size = int(getattr(args, "prefill_chunk_size", 0) or 0)
    if chunk_size <= 0 or int(ids.shape[1]) <= chunk_size:
        kwargs: dict[str, Any] = {"use_cache": True, "logits_to_keep": 1}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        return model(ids, **kwargs)

    # RWKV exposes a cache-correct serving helper. Qwen follows the same HF
    # cache contract directly. This permits memory-safe, apples-to-apples
    # reruns of large (batch, prompt) cells without weakening the matrix key.
    rwkv_chunks = getattr(model, "rwkv7_prefill_chunks", None)
    if args.model_kind == "rwkv" and past_key_values is None and callable(rwkv_chunks):
        return rwkv_chunks(ids, chunk_size=chunk_size, logits_to_keep=1)

    out = None
    past = past_key_values
    for start in range(0, int(ids.shape[1]), chunk_size):
        out = model(
            ids[:, start : start + chunk_size],
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        if past_key_values is not None and out.past_key_values is not past_key_values:
            raise RuntimeError("chunked prefill replaced the supplied persistent cache")
        past = out.past_key_values
    if out is None:
        raise RuntimeError("chunked prefill produced no output")
    return out


def timed_prefill_details(args: argparse.Namespace, model, ids) -> dict[str, Any]:
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
    return {
        "median_s": median_s,
        "tokps": (args.batch_size * args.prompt_tokens) / median_s,
        "samples": samples,
    }


def timed_prefill(args: argparse.Namespace, model, ids) -> tuple[float, float]:
    details = timed_prefill_details(args, model, ids)
    return float(details["median_s"]), float(details["tokps"])


def decode_once(
    args: argparse.Namespace, model, ids, step: Callable[..., Any]
) -> tuple[float, Any]:
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


def _cache_sequence_length(cache) -> int:
    value = cache.get_seq_length()
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _cache_tensor_pointer_signature(
    cache,
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    signature: list[tuple[str, int, tuple[int, ...]]] = []
    for layer_index, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                signature.append(
                    (
                        f"layers.{layer_index}.{name}",
                        int(value.data_ptr()),
                        tuple(value.shape),
                    )
                )
    return tuple(sorted(signature))


def _compile_counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        counters = torch._dynamo.utils.counters
    except AttributeError:
        return {}
    return {
        str(group): {
            str(name): int(value)
            for name, value in values.items()
            if isinstance(value, int)
        }
        for group, values in counters.items()
    }


def _compile_counter_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
    group: str,
    name: str,
) -> int:
    return int(after.get(group, {}).get(name, 0)) - int(
        before.get(group, {}).get(name, 0)
    )


def _distinct_batch_probe_ids(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if int(ids.shape[0]) <= 1:
        return ids
    offsets = torch.arange(int(ids.shape[0]), device=ids.device).reshape(-1, 1)
    return torch.remainder(ids + offsets, int(vocab_size))


def _clear_qwen_compile_state(model) -> None:
    for name in ("_compiled_call", "_last_compile_config"):
        if hasattr(model, name):
            delattr(model, name)
    reset = getattr(torch.compiler, "reset", None)
    if callable(reset):
        reset()


class QwenStaticCacheInductorCudaGraphDecode:
    """Compile one exact Qwen decode cell with HF StaticCache and CUDAGraph Trees."""

    graph_scope = "single_token_hf_qwen_forward"
    compile_backend = "inductor"
    compile_fullgraph = False
    compile_dynamic = False
    capture_warmup_steps = 3

    def __init__(self, args: argparse.Namespace, model, ids) -> None:
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "static_cache_inductor_cudagraph requires a live CUDA device"
            )
        try:
            from transformers import CompileConfig, StaticCache
        except ImportError as exc:
            raise RuntimeError(
                "installed Transformers does not provide CompileConfig and StaticCache"
            ) from exc

        self.args = args
        self.model = model
        self.ids = ids
        self.compile_mode = str(getattr(args, "qwen_compile_mode", "max-autotune"))
        self.max_cache_len = (
            int(args.prompt_tokens) + int(args.warmup) + int(args.decode_tokens)
        )
        setup_started = time.perf_counter()
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_cache_len)
        first = self.prefill(ids)
        token = first.logits[:, -1:].argmax(dim=-1)
        del first
        initial_pointer_signature = _cache_tensor_pointer_signature(self.cache)
        if not initial_pointer_signature:
            raise RuntimeError("StaticCache did not initialize any persistent tensors")

        compile_config = CompileConfig(
            backend=self.compile_backend,
            mode=self.compile_mode,
            fullgraph=self.compile_fullgraph,
            dynamic=self.compile_dynamic,
        )
        before = _compile_counter_snapshot()
        self.compiled = model.get_compiled_call(compile_config)
        capture_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(self.capture_warmup_steps):
                torch.compiler.cudagraph_mark_step_begin()
                out = self.compiled(
                    token,
                    past_key_values=self.cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                token = out.logits[:, -1:].argmax(dim=-1)
                del out
        cuda_sync(args.device)
        self.capture_s = time.perf_counter() - capture_started
        after = _compile_counter_snapshot()
        self.graph_break_count = sum(after.get("graph_break", {}).values()) - sum(
            before.get("graph_break", {}).values()
        )
        self.cudagraph_skip_count = _compile_counter_delta(
            before, after, "inductor", "cudagraph_skips"
        )
        self.cudagraph_recorded_non_static_inputs = _compile_counter_delta(
            before,
            after,
            "inductor",
            "cudagraph_recorded_non_static_inputs",
        )
        if self.graph_break_count != 0 or self.cudagraph_skip_count != 0:
            raise RuntimeError(
                "Qwen compile did not satisfy the no-break/no-skip CUDA Graph gate: "
                f"graph_breaks={self.graph_break_count}, "
                f"cudagraph_skips={self.cudagraph_skip_count}"
            )
        if self.cudagraph_recorded_non_static_inputs <= 0:
            raise RuntimeError(
                "Inductor did not record a CUDA Graph node for Qwen decode"
            )

        profile_prefill = self.prefill(ids)
        profile_token = profile_prefill.logits[:, -1:].argmax(dim=-1)
        del profile_prefill
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                profile_out = self.compiled(
                    profile_token,
                    past_key_values=self.cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
            cuda_sync(args.device)
        del profile_out
        graph_events = [
            event
            for event in profile.key_averages()
            if str(event.key) == "cudaGraphLaunch"
        ]
        self.cuda_graph_launch_count = sum(int(event.count) for event in graph_events)
        self.cuda_graph_verified = self.cuda_graph_launch_count > 0
        if not self.cuda_graph_verified:
            raise RuntimeError(
                "profiler did not observe cudaGraphLaunch for Qwen decode"
            )

        self.prefill(ids)
        final_pointer_signature = _cache_tensor_pointer_signature(self.cache)
        self.cache_pointer_stable = initial_pointer_signature == final_pointer_signature
        self.cache_tensor_pointer_count = len(final_pointer_signature)
        if not self.cache_pointer_stable:
            raise RuntimeError(
                "StaticCache tensor pointers changed across reset/prefill"
            )
        self.setup_s = time.perf_counter() - setup_started

    def prefill(self, ids):
        with torch.inference_mode():
            self.cache.reset()
            if _cache_sequence_length(self.cache) != 0:
                raise RuntimeError(
                    "StaticCache reset did not restore sequence length to zero"
                )
            out = forward_prefill(
                self.args,
                self.model,
                ids,
                past_key_values=self.cache,
            )
        if out.past_key_values is not self.cache:
            raise RuntimeError(
                "chunked StaticCache prefill replaced the persistent cache object"
            )
        actual = _cache_sequence_length(self.cache)
        expected = int(ids.shape[1])
        if actual != expected:
            raise RuntimeError(
                f"StaticCache prefill length mismatch: got {actual}, expected {expected}"
            )
        return out

    def compiled_step(self, token):
        torch.compiler.cudagraph_mark_step_begin()
        out = self.compiled(
            token,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        return out

    def compiled_greedy_tokens(self, ids, steps: int) -> torch.Tensor:
        with torch.inference_mode():
            out = self.prefill(ids)
            token = out.logits[:, -1:].argmax(dim=-1)
            tokens = [token.detach().cpu()]
            del out
            for _ in range(steps):
                out = self.compiled_step(token)
                token = out.logits[:, -1:].argmax(dim=-1)
                tokens.append(token.detach().cpu())
                del out
        expected = int(ids.shape[1]) + steps
        actual = _cache_sequence_length(self.cache)
        if actual != expected:
            raise RuntimeError(
                f"compiled StaticCache length mismatch: got {actual}, expected {expected}"
            )
        return torch.cat(tokens, dim=1)

    def candidate_greedy_tokens(self, ids, steps: int) -> torch.Tensor:
        return self.compiled_greedy_tokens(ids, steps)

    def candidate_logits_trace(self, ids, steps: int) -> list[torch.Tensor]:
        with torch.inference_mode():
            out = self.prefill(ids)
            token = out.logits[:, -1:].argmax(dim=-1)
            logits = [out.logits[:, -1].detach().cpu()]
            del out
            for _ in range(steps):
                out = self.compiled_step(token)
                token = out.logits[:, -1:].argmax(dim=-1)
                logits.append(out.logits[:, -1].detach().cpu())
                del out
        expected = int(ids.shape[1]) + steps
        actual = _cache_sequence_length(self.cache)
        if actual != expected:
            raise RuntimeError(
                f"compiled StaticCache length mismatch: got {actual}, expected {expected}"
            )
        return logits

    def timed(self) -> float:
        with torch.inference_mode():
            out = self.prefill(self.ids)
            token = out.logits[:, -1:].argmax(dim=-1)
            del out
            for _ in range(int(self.args.warmup)):
                out = self.compiled_step(token)
                token = out.logits[:, -1:].argmax(dim=-1)
                del out
            cuda_sync(self.args.device)
            started = time.perf_counter()
            for _ in range(int(self.args.decode_tokens)):
                out = self.compiled_step(token)
                token = out.logits[:, -1:].argmax(dim=-1)
                del out
            cuda_sync(self.args.device)
            elapsed = time.perf_counter() - started
        expected = (
            int(self.args.prompt_tokens)
            + int(self.args.warmup)
            + int(self.args.decode_tokens)
        )
        actual = _cache_sequence_length(self.cache)
        if actual != expected:
            raise RuntimeError(
                f"timed StaticCache length mismatch: got {actual}, expected {expected}"
            )
        return elapsed

    def cleanup(self) -> None:
        self.compiled = None
        _clear_qwen_compile_state(self.model)


class QwenStaticCacheRawCudaGraphDecode:
    """Capture one exact eager Qwen decode step as a raw CUDA Graph."""

    graph_scope = "single_token_hf_qwen_forward_argmax_token_copy"
    compile_backend = None
    compile_mode = None
    compile_fullgraph = None
    compile_dynamic = None
    graph_break_count = None
    cudagraph_skip_count = None
    cudagraph_recorded_non_static_inputs = None
    capture_warmup_steps = 3

    def __init__(self, args: argparse.Namespace, model, ids) -> None:
        self.args = args
        self.model = model
        self.ids = ids
        self.max_cache_len = None
        self.cache = None
        self.static_token = None
        self.capture_stream = None
        self.graph = None
        self.static_logits = None
        self._cleanup_complete = False
        try:
            self._initialize()
        except BaseException:
            # __init__ failures do not assign the runner in timed_decode_details.
            # Release any graph/cache tensors acquired before the failure here so
            # a resident benchmark process can safely continue to the next cell.
            self.cleanup(suppress_errors=True)
            raise

    def _initialize(self) -> None:
        args = self.args
        model = self.model
        ids = self.ids
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("static_cache_raw_cudagraph requires a live CUDA device")
        try:
            from transformers import StaticCache
        except ImportError as exc:
            raise RuntimeError(
                "installed Transformers does not provide StaticCache"
            ) from exc

        self.max_cache_len = (
            int(args.prompt_tokens) + int(args.warmup) + int(args.decode_tokens)
        )
        setup_started = time.perf_counter()
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_cache_len)
        self.static_token = torch.empty(
            (int(args.batch_size), 1), dtype=torch.long, device=ids.device
        )
        first = self.prefill(ids)
        self.static_token.copy_(first.logits[:, -1:].argmax(dim=-1))
        del first
        initial_cache_signature = _cache_tensor_pointer_signature(self.cache)
        if not initial_cache_signature:
            raise RuntimeError("StaticCache did not initialize persistent tensors")
        self.cache_pointer_signature = initial_cache_signature
        self.static_token_pointer = int(self.static_token.data_ptr())

        self.capture_stream = torch.cuda.Stream(device=ids.device)
        current = torch.cuda.current_stream(device=ids.device)
        self.capture_stream.wait_stream(current)
        with torch.cuda.stream(self.capture_stream), torch.inference_mode():
            for _ in range(self.capture_warmup_steps):
                out = model(
                    self.static_token,
                    past_key_values=self.cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                self.static_token.copy_(out.logits[:, -1:].argmax(dim=-1))
                del out
        current.wait_stream(self.capture_stream)

        first = self.prefill(ids)
        self.static_token.copy_(first.logits[:, -1:].argmax(dim=-1))
        del first
        self.graph = torch.cuda.CUDAGraph()
        capture_started = time.perf_counter()
        self.capture_stream.wait_stream(current)
        with (
            torch.inference_mode(),
            torch.cuda.graph(
                self.graph, stream=self.capture_stream, capture_error_mode="global"
            ),
        ):
            captured = model(
                self.static_token,
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            self.static_logits = captured.logits[:, -1]
            self.static_token.copy_(self.static_logits.argmax(dim=-1, keepdim=True))
        current.wait_stream(self.capture_stream)
        cuda_sync(args.device)
        self.capture_s = time.perf_counter() - capture_started
        self.static_logits_pointer = int(self.static_logits.data_ptr())

        first = self.prefill(ids)
        self.static_token.copy_(first.logits[:, -1:].argmax(dim=-1))
        del first
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.inference_mode():
                self.graph.replay()
            cuda_sync(args.device)
        self.cuda_graph_launch_count = sum(
            int(event.count)
            for event in profile.key_averages()
            if str(event.key) == "cudaGraphLaunch"
        )
        self.cuda_graph_verified = self.cuda_graph_launch_count == 1
        if not self.cuda_graph_verified:
            raise RuntimeError(
                "raw CUDA Graph profiler expected exactly one cudaGraphLaunch, got "
                f"{self.cuda_graph_launch_count}"
            )
        self._refresh_pointer_gate()
        self.setup_s = time.perf_counter() - setup_started

    def _refresh_pointer_gate(self) -> None:
        self.cache_pointer_stable = (
            _cache_tensor_pointer_signature(self.cache) == self.cache_pointer_signature
            and int(self.static_token.data_ptr()) == self.static_token_pointer
            and int(self.static_logits.data_ptr()) == self.static_logits_pointer
        )
        self.cache_tensor_pointer_count = len(self.cache_pointer_signature)
        if not self.cache_pointer_stable:
            raise RuntimeError("raw CUDA Graph static tensor pointers changed")

    def prefill(self, ids):
        with torch.inference_mode():
            self.cache.reset()
            if _cache_sequence_length(self.cache) != 0:
                raise RuntimeError("StaticCache reset did not restore length zero")
            out = forward_prefill(
                self.args, self.model, ids, past_key_values=self.cache
            )
        if out.past_key_values is not self.cache:
            raise RuntimeError("chunked StaticCache prefill replaced the cache object")
        actual = _cache_sequence_length(self.cache)
        if actual != int(ids.shape[1]):
            raise RuntimeError(
                f"StaticCache prefill length mismatch: {actual} != {ids.shape[1]}"
            )
        return out

    def candidate_step(self, token):
        self.static_token.copy_(token)
        self.graph.replay()
        return self.static_logits

    def candidate_greedy_tokens(self, ids, steps: int) -> torch.Tensor:
        with torch.inference_mode():
            out = self.prefill(ids)
            self.static_token.copy_(out.logits[:, -1:].argmax(dim=-1))
            tokens = [self.static_token.detach().cpu()]
            del out
            for _ in range(steps):
                self.graph.replay()
                tokens.append(self.static_token.detach().cpu())
        actual = _cache_sequence_length(self.cache)
        expected = int(ids.shape[1]) + steps
        if actual != expected:
            raise RuntimeError(
                f"raw graph cache length mismatch: {actual} != {expected}"
            )
        self._refresh_pointer_gate()
        return torch.cat(tokens, dim=1)

    def candidate_logits_trace(self, ids, steps: int) -> list[torch.Tensor]:
        with torch.inference_mode():
            out = self.prefill(ids)
            self.static_token.copy_(out.logits[:, -1:].argmax(dim=-1))
            logits = [out.logits[:, -1].detach().cpu()]
            del out
            for _ in range(steps):
                self.graph.replay()
                logits.append(self.static_logits.detach().cpu())
        actual = _cache_sequence_length(self.cache)
        expected = int(ids.shape[1]) + steps
        if actual != expected:
            raise RuntimeError(
                f"raw graph cache length mismatch: {actual} != {expected}"
            )
        self._refresh_pointer_gate()
        return logits

    def timed(self) -> float:
        with torch.inference_mode():
            out = self.prefill(self.ids)
            self.static_token.copy_(out.logits[:, -1:].argmax(dim=-1))
            del out
            for _ in range(int(self.args.warmup)):
                self.graph.replay()
            cuda_sync(self.args.device)
            started = time.perf_counter()
            for _ in range(int(self.args.decode_tokens)):
                self.graph.replay()
            cuda_sync(self.args.device)
            elapsed = time.perf_counter() - started
        expected = self.max_cache_len
        actual = _cache_sequence_length(self.cache)
        if actual != expected:
            raise RuntimeError(
                f"timed raw graph cache length mismatch: {actual} != {expected}"
            )
        self._refresh_pointer_gate()
        return elapsed

    def cleanup(self, *, suppress_errors: bool = False) -> None:
        if self._cleanup_complete:
            return
        self._cleanup_complete = True
        cleanup_errors: list[BaseException] = []
        try:
            cuda_sync(self.args.device)
        except BaseException as exc:  # keep releasing resources after a failed capture
            cleanup_errors.append(exc)
        graph = self.graph
        if graph is not None:
            try:
                graph.reset()
            except BaseException as exc:
                cleanup_errors.append(exc)
        self.static_logits = None
        self.static_token = None
        self.graph = None
        self.cache = None
        self.capture_stream = None
        if cleanup_errors and not suppress_errors:
            raise RuntimeError("raw CUDA Graph cleanup failed") from cleanup_errors[0]


def _minimum_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    values = torch.nn.functional.cosine_similarity(
        left.float(),
        right.float(),
        dim=-1,
        eps=1e-12,
    )
    return float(values.min().item())


def _eager_greedy_tokens(
    args: argparse.Namespace,
    model,
    ids,
    steps: int,
    *,
    cache=None,
) -> torch.Tensor:
    with torch.inference_mode():
        if cache is not None:
            cache.reset()
        out = forward_prefill(args, model, ids, past_key_values=cache)
        state = out.past_key_values
        if cache is not None and state is not cache:
            raise RuntimeError("eager StaticCache oracle replaced the supplied cache")
        token = out.logits[:, -1:].argmax(dim=-1)
        tokens = [token.detach().cpu()]
        del out
        for _ in range(steps):
            out = model(
                token,
                past_key_values=state,
                use_cache=True,
                logits_to_keep=1,
            )
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            tokens.append(token.detach().cpu())
            del out
    return torch.cat(tokens, dim=1)


def _eager_logits_trace(
    args: argparse.Namespace,
    model,
    ids,
    steps: int,
    *,
    cache=None,
) -> list[torch.Tensor]:
    """Collect an untimed logits trace without retaining GPU outputs."""

    with torch.inference_mode():
        if cache is not None:
            cache.reset()
        out = forward_prefill(args, model, ids, past_key_values=cache)
        state = out.past_key_values
        if cache is not None and state is not cache:
            raise RuntimeError("eager StaticCache oracle replaced the supplied cache")
        token = out.logits[:, -1:].argmax(dim=-1)
        logits = [out.logits[:, -1].detach().cpu()]
        del out
        for _ in range(steps):
            out = model(
                token,
                past_key_values=state,
                use_cache=True,
                logits_to_keep=1,
            )
            state = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
            logits.append(out.logits[:, -1].detach().cpu())
            del out
    return logits


def _candidate_logits_trace(
    runner,
    ids,
    steps: int,
) -> list[torch.Tensor]:
    """Collect the candidate StaticCache graph logits trace."""

    return runner.candidate_logits_trace(ids, steps)


def _logits_trace_metrics(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise RuntimeError("logits traces must have the same non-zero length")
    non_finite_indices = [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())
    ]
    if non_finite_indices:
        return {
            "finite": False,
            "min_cosine": None,
            "max_abs_diff": None,
            "worst_index": int(non_finite_indices[0]),
            "greedy_match": False,
        }
    cosines = [_minimum_cosine(a, b) for a, b in zip(left, right)]
    max_abs = max(
        float((a.float() - b.float()).abs().max().item()) for a, b in zip(left, right)
    )
    return {
        "finite": True,
        "min_cosine": min(cosines),
        "max_abs_diff": max_abs,
        "worst_index": min(range(len(cosines)), key=cosines.__getitem__),
        "greedy_match": all(
            torch.equal(a.argmax(dim=-1), b.argmax(dim=-1)) for a, b in zip(left, right)
        ),
    }


def _token_trace_mismatch_metrics(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, int | None]:
    if left.shape != right.shape:
        raise RuntimeError(
            f"token traces must have the same shape: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    mismatches = (left != right).nonzero(as_tuple=False)
    return {
        "count": int(mismatches.shape[0]),
        "first_index": (
            None if int(mismatches.shape[0]) == 0 else int(mismatches[0, 1].item())
        ),
    }


class QwenCudaGraphParityError(RuntimeError):
    def __init__(self, parity: dict[str, Any]) -> None:
        self.qwen_graph_parity = dict(parity)
        super().__init__(
            "Qwen CUDA Graph parity failed: "
            + json.dumps(parity, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )


def verify_qwen_cuda_graph_parity(
    args: argparse.Namespace,
    model,
    ids,
    runner,
) -> dict[str, Any]:
    """Gate the full decode horizon and a logits probe against eager HF."""

    probe_ids = _distinct_batch_probe_ids(ids, int(model.config.vocab_size))
    parity_steps = int(args.warmup) + int(args.decode_tokens)
    eager_tokens = _eager_greedy_tokens(args, model, probe_ids, parity_steps)
    static_eager_tokens = _eager_greedy_tokens(
        args,
        model,
        probe_ids,
        parity_steps,
        cache=runner.cache,
    )
    candidate_tokens = runner.candidate_greedy_tokens(probe_ids, parity_steps)
    static_eager_match = bool(torch.equal(eager_tokens, static_eager_tokens))
    greedy_match = bool(torch.equal(eager_tokens, candidate_tokens))
    same_cache_greedy_match = bool(torch.equal(static_eager_tokens, candidate_tokens))
    dynamic_static_mismatches = _token_trace_mismatch_metrics(
        eager_tokens, static_eager_tokens
    )
    dynamic_candidate_mismatches = _token_trace_mismatch_metrics(
        eager_tokens, candidate_tokens
    )
    same_cache_mismatches = _token_trace_mismatch_metrics(
        static_eager_tokens, candidate_tokens
    )
    prefill_next_token_match = bool(
        torch.equal(eager_tokens[:, :1], candidate_tokens[:, :1])
    )
    logits_probe_tokens = min(
        int(getattr(args, "qwen_graph_probe_tokens", 16)), parity_steps
    )
    dynamic_logits = _eager_logits_trace(args, model, probe_ids, logits_probe_tokens)
    static_logits = _eager_logits_trace(
        args,
        model,
        probe_ids,
        logits_probe_tokens,
        cache=runner.cache,
    )
    candidate_logits = _candidate_logits_trace(runner, probe_ids, logits_probe_tokens)
    dynamic_candidate = _logits_trace_metrics(dynamic_logits, candidate_logits)
    dynamic_static = _logits_trace_metrics(dynamic_logits, static_logits)
    same_cache = _logits_trace_metrics(static_logits, candidate_logits)
    minimum_cosine = dynamic_candidate["min_cosine"]
    max_abs_diff = dynamic_candidate["max_abs_diff"]
    same_cache_cosine_pass = (
        bool(same_cache["finite"])
        and isinstance(same_cache["min_cosine"], (int, float))
        and float(same_cache["min_cosine"]) >= 0.9999
    )
    logits_greedy_match = bool(dynamic_candidate["greedy_match"])
    cross_cache_policy = str(
        getattr(args, "qwen_cross_cache_full_greedy_policy", "strict")
    )
    cross_cache_full_greedy_required = cross_cache_policy == "strict"
    cross_cache_full_greedy_pass = (
        static_eager_match and greedy_match
        if cross_cache_full_greedy_required
        else True
    )
    verified = (
        runner.cuda_graph_verified
        and runner.cache_pointer_stable
        and prefill_next_token_match
        and cross_cache_full_greedy_pass
        and same_cache_greedy_match
        and logits_greedy_match
        and bool(dynamic_static["greedy_match"])
        and bool(same_cache["greedy_match"])
        and bool(dynamic_candidate["finite"])
        and bool(dynamic_static["finite"])
        and bool(same_cache["finite"])
        and same_cache_cosine_pass
    )
    return {
        "qwen_graph_parity_verified": verified,
        "qwen_cross_cache_full_greedy_policy_effective": cross_cache_policy,
        "qwen_cross_cache_full_greedy_required": cross_cache_full_greedy_required,
        "qwen_graph_prefill_next_token_match": prefill_next_token_match,
        "qwen_graph_greedy_match": greedy_match,
        "qwen_same_cache_greedy_match": same_cache_greedy_match,
        "qwen_dynamic_static_full_greedy_mismatch_count": dynamic_static_mismatches[
            "count"
        ],
        "qwen_dynamic_static_full_greedy_first_mismatch_index": dynamic_static_mismatches[
            "first_index"
        ],
        "qwen_dynamic_candidate_full_greedy_mismatch_count": dynamic_candidate_mismatches[
            "count"
        ],
        "qwen_dynamic_candidate_full_greedy_first_mismatch_index": dynamic_candidate_mismatches[
            "first_index"
        ],
        "qwen_same_cache_full_greedy_mismatch_count": same_cache_mismatches["count"],
        "qwen_same_cache_full_greedy_first_mismatch_index": same_cache_mismatches[
            "first_index"
        ],
        "qwen_static_cache_eager_greedy_match": static_eager_match,
        "qwen_graph_logits_greedy_match": logits_greedy_match,
        "qwen_dynamic_static_logits_greedy_match": bool(dynamic_static["greedy_match"]),
        "qwen_same_cache_logits_greedy_match": bool(same_cache["greedy_match"]),
        "qwen_graph_probe_tokens": parity_steps,
        "qwen_graph_logits_probe_tokens": logits_probe_tokens,
        "qwen_graph_distinct_batch_prompts": int(ids.shape[0]) > 1,
        "qwen_graph_logits_min_cosine": minimum_cosine,
        "qwen_graph_logits_max_abs_diff": max_abs_diff,
        "qwen_graph_logits_trace_finite": bool(dynamic_candidate["finite"]),
        "qwen_graph_logits_worst_index": int(dynamic_candidate["worst_index"]),
        "qwen_dynamic_static_logits_min_cosine": dynamic_static["min_cosine"],
        "qwen_dynamic_static_logits_max_abs_diff": dynamic_static["max_abs_diff"],
        "qwen_dynamic_static_logits_finite": bool(dynamic_static["finite"]),
        "qwen_dynamic_static_logits_worst_index": int(dynamic_static["worst_index"]),
        "qwen_same_cache_logits_min_cosine": same_cache["min_cosine"],
        "qwen_same_cache_logits_max_abs_diff": same_cache["max_abs_diff"],
        "qwen_same_cache_logits_finite": bool(same_cache["finite"]),
        "qwen_same_cache_logits_worst_index": int(same_cache["worst_index"]),
        # Compatibility aliases for pre-v2 Inductor-only evidence consumers.
        "qwen_static_compiled_logits_min_cosine": same_cache["min_cosine"],
        "qwen_static_compiled_logits_max_abs_diff": same_cache["max_abs_diff"],
        "qwen_static_compiled_logits_finite": bool(same_cache["finite"]),
        "qwen_static_compiled_logits_worst_index": int(same_cache["worst_index"]),
    }


def timed_decode_details(args: argparse.Namespace, model, ids) -> dict[str, Any]:
    qwen_optimization = str(
        getattr(args, "qwen_decode_optimization", "module_call_dynamic")
    )
    if args.model_kind == "qwen35" and qwen_optimization in QWEN_STATIC_GRAPH_ROUTES:
        runner = None
        try:
            runner_class = (
                QwenStaticCacheInductorCudaGraphDecode
                if qwen_optimization == "static_cache_inductor_cudagraph"
                else QwenStaticCacheRawCudaGraphDecode
            )
            runner = runner_class(args, model, ids)
            parity = verify_qwen_cuda_graph_parity(args, model, ids, runner)
            if not bool(parity["qwen_graph_parity_verified"]):
                raise QwenCudaGraphParityError(parity)
            samples = [runner.timed() for _ in range(int(args.runs))]
            return {
                "median_s": float(statistics.median(samples)),
                "samples": samples,
                "step_backend": f"qwen_{qwen_optimization}",
                "effective_backend": None,
                "cache_type": type(runner.cache).__name__,
                "qwen_decode_optimization_effective": qwen_optimization,
                "qwen_cuda_graph_requested": True,
                "qwen_cuda_graph_effective": True,
                "qwen_decode_cuda_graph_verified": runner.cuda_graph_verified,
                "qwen_graph_scope": runner.graph_scope,
                "qwen_graph_capture_s": runner.capture_s,
                "qwen_graph_setup_s": runner.setup_s,
                "qwen_graph_max_cache_len": runner.max_cache_len,
                "qwen_graph_break_count": runner.graph_break_count,
                "qwen_cudagraph_skip_count": runner.cudagraph_skip_count,
                "qwen_cudagraph_recorded_non_static_inputs": (
                    runner.cudagraph_recorded_non_static_inputs
                ),
                "qwen_cuda_graph_launch_count": runner.cuda_graph_launch_count,
                "qwen_cache_pointer_stable": runner.cache_pointer_stable,
                "qwen_cache_tensor_pointer_count": runner.cache_tensor_pointer_count,
                "qwen_compile_backend_effective": runner.compile_backend,
                "qwen_compile_mode_effective": runner.compile_mode,
                "qwen_compile_fullgraph_effective": runner.compile_fullgraph,
                "qwen_compile_dynamic_effective": runner.compile_dynamic,
                **parity,
            }
        finally:
            if runner is not None:
                runner.cleanup()
            elif qwen_optimization == "static_cache_inductor_cudagraph":
                _clear_qwen_compile_state(model)
            runner = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    step, step_backend = step_function(model, args.model_kind, args.batch_size)
    samples: list[float] = []
    state = None
    for _ in range(args.runs):
        elapsed, state = decode_once(args, model, ids, step)
        samples.append(elapsed)
    return {
        "median_s": float(statistics.median(samples)),
        "samples": samples,
        "step_backend": step_backend,
        "effective_backend": last_rwkv_backend(model),
        "cache_type": type(state).__name__ if state is not None else "None",
        "qwen_decode_optimization_effective": (
            "module_call_dynamic" if args.model_kind == "qwen35" else None
        ),
        "qwen_cuda_graph_requested": False if args.model_kind == "qwen35" else None,
        "qwen_cuda_graph_effective": False if args.model_kind == "qwen35" else None,
        "qwen_decode_cuda_graph_verified": False
        if args.model_kind == "qwen35"
        else None,
        "qwen_graph_parity_verified": None,
        "qwen_cross_cache_full_greedy_policy_effective": None,
        "qwen_cross_cache_full_greedy_required": None,
        "qwen_graph_prefill_next_token_match": None,
        "qwen_graph_greedy_match": None,
        "qwen_same_cache_greedy_match": None,
        "qwen_dynamic_static_full_greedy_mismatch_count": None,
        "qwen_dynamic_static_full_greedy_first_mismatch_index": None,
        "qwen_dynamic_candidate_full_greedy_mismatch_count": None,
        "qwen_dynamic_candidate_full_greedy_first_mismatch_index": None,
        "qwen_same_cache_full_greedy_mismatch_count": None,
        "qwen_same_cache_full_greedy_first_mismatch_index": None,
        "qwen_graph_probe_tokens": None,
        "qwen_graph_logits_probe_tokens": None,
        "qwen_graph_distinct_batch_prompts": None,
        "qwen_static_cache_eager_greedy_match": None,
        "qwen_graph_logits_greedy_match": None,
        "qwen_dynamic_static_logits_greedy_match": None,
        "qwen_same_cache_logits_greedy_match": None,
        "qwen_graph_logits_min_cosine": None,
        "qwen_graph_logits_max_abs_diff": None,
        "qwen_graph_logits_trace_finite": None,
        "qwen_graph_logits_worst_index": None,
        "qwen_dynamic_static_logits_min_cosine": None,
        "qwen_dynamic_static_logits_max_abs_diff": None,
        "qwen_dynamic_static_logits_finite": None,
        "qwen_dynamic_static_logits_worst_index": None,
        "qwen_static_compiled_logits_min_cosine": None,
        "qwen_static_compiled_logits_max_abs_diff": None,
        "qwen_static_compiled_logits_finite": None,
        "qwen_static_compiled_logits_worst_index": None,
        "qwen_same_cache_logits_min_cosine": None,
        "qwen_same_cache_logits_max_abs_diff": None,
        "qwen_same_cache_logits_finite": None,
        "qwen_same_cache_logits_worst_index": None,
        "qwen_graph_scope": None,
        "qwen_graph_capture_s": None,
        "qwen_graph_setup_s": None,
        "qwen_graph_max_cache_len": None,
        "qwen_graph_break_count": None,
        "qwen_cudagraph_skip_count": None,
        "qwen_cudagraph_recorded_non_static_inputs": None,
        "qwen_cuda_graph_launch_count": None,
        "qwen_cache_pointer_stable": None,
        "qwen_cache_tensor_pointer_count": None,
        "qwen_compile_backend_effective": None,
        "qwen_compile_mode_effective": None,
        "qwen_compile_fullgraph_effective": None,
        "qwen_compile_dynamic_effective": None,
    }


def timed_decode(
    args: argparse.Namespace, model, ids
) -> tuple[float, str, str | None, str]:
    details = timed_decode_details(args, model, ids)
    return (
        float(details["median_s"]),
        str(details["step_backend"]),
        details["effective_backend"],
        str(details["cache_type"]),
    )


def save_backend_probe(args: argparse.Namespace, model, ids) -> dict[str, Any]:
    """Save deterministic logits and greedy tokens for cross-process checks."""

    probe_batch_size = int(getattr(args, "probe_batch_size", 1))
    step, _ = step_function(model, args.model_kind, probe_batch_size)
    probe_ids = _distinct_batch_probe_ids(
        ids[:probe_batch_size], int(model.config.vocab_size)
    )
    greedy_tokens: list[Any] = []
    decode_logits_finite_by_batch = torch.ones(probe_batch_size, dtype=torch.bool)
    with torch.inference_mode():
        out = forward_prefill(args, model, probe_ids)
        state = out.past_key_values
        prompt_logits = out.logits[:, -1].float().cpu()
        token = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.probe_tokens):
            if probe_batch_size == 1:
                greedy_tokens.append(int(token[0, 0].item()))
            else:
                greedy_tokens.append(
                    [int(value) for value in token[:, 0].detach().cpu().tolist()]
                )
            out = step(token, state)
            state = out.past_key_values
            decode_logits_finite_by_batch &= torch.isfinite(
                out.logits[:, -1].float().cpu()
            ).all(dim=-1)
            token = out.logits[:, -1:].argmax(dim=-1)
        final_logits = out.logits[:, -1].float().cpu()

    output = Path(args.probe_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "probe_schema_version": 2,
            "benchmark_repository_commit": os.environ.get("REPOSITORY_COMMIT"),
            "model_pair": args.model_pair,
            "model_size_label": args.model_size_label,
            "model_id_or_path": args.model,
            "probe_output": str(output.resolve()),
            "input_ids": probe_ids.cpu(),
            "prompt_logits": prompt_logits,
            "final_logits": final_logits,
            "greedy_tokens": torch.tensor(greedy_tokens, dtype=torch.int64),
            "decode_logits_finite_by_batch": decode_logits_finite_by_batch,
            "decode_logits_all_finite": bool(
                decode_logits_finite_by_batch.all().item()
            ),
            "qwen_backend_requested": args.qwen_backend,
        },
        output,
    )
    return {
        "probe_output": str(output),
        "probe_tokens": args.probe_tokens,
        "probe_batch_size": probe_batch_size,
        "probe_distinct_batch_prompts": probe_batch_size > 1,
        "probe_decode_logits_all_finite": bool(
            decode_logits_finite_by_batch.all().item()
        ),
        "probe_decode_logits_finite_by_batch": (decode_logits_finite_by_batch.tolist()),
        "probe_greedy_tokens": greedy_tokens,
    }


def qwen_official_fast_path_environment() -> dict[str, bool]:
    """Return the fail-closed official Qwen3.5 optional-kernel environment gate."""

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        available = (
            is_fast_path_available()
            if callable(is_fast_path_available)
            else is_fast_path_available
        )
        fast_path_available = bool(available)
    except Exception:
        fast_path_available = False
    return {
        "qwen_fast_path_available": fast_path_available,
        "qwen_fla_importable": importlib.util.find_spec("fla") is not None,
        "qwen_causal_conv1d_importable": importlib.util.find_spec("causal_conv1d")
        is not None,
        "qwen_force_torch_disabled": not QWEN_FORCE_TORCH,
    }


def environment_metadata(args: argparse.Namespace, model=None) -> dict[str, Any]:
    qwen_environment = (
        qwen_official_fast_path_environment() if args.model_kind == "qwen35" else {}
    )
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
            self_chunk_h_bv_effective, self_chunk_h_bc_effective = (
                resolve_chunk_h_tiles(
                    torch.cuda.current_device(),
                    self_chunk_size,
                    batch_size=int(args.batch_size),
                    tokens=int(args.prompt_tokens),
                )
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
        capability = list(
            torch.cuda.get_device_capability(cuda_device_index(args.device))
        )
    arch = f"sm_{capability[0]}{capability[1]}" if capability is not None else None
    qwen_device_route = None
    if args.model_kind == "qwen35" and arch is not None:
        qwen_device_route = (
            "fla_triton_sm70"
            if capability == [7, 0]
            else f"fla_runtime_dispatch_{arch}"
        )
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
        "qwen_fla_importable": qwen_environment.get("qwen_fla_importable"),
        "qwen_causal_conv1d_importable": qwen_environment.get(
            "qwen_causal_conv1d_importable"
        ),
        "qwen_force_torch": QWEN_FORCE_TORCH,
        "qwen_fast_path_available": qwen_environment.get("qwen_fast_path_available"),
        "qwen_fla_expected_device_route": qwen_device_route,
        "qwen_sdpa_policy_effective": (
            str(getattr(args, "_qwen_sdpa_policy_effective", "auto"))
            if args.model_kind == "qwen35"
            else None
        ),
        "qwen_sdp_flash_enabled": (
            bool(torch.backends.cuda.flash_sdp_enabled())
            if args.model_kind == "qwen35"
            and callable(getattr(torch.backends.cuda, "flash_sdp_enabled", None))
            else None
        ),
        "qwen_sdp_mem_efficient_enabled": (
            bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            if args.model_kind == "qwen35"
            and callable(
                getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", None)
            )
            else None
        ),
        "qwen_sdp_math_enabled": (
            bool(torch.backends.cuda.math_sdp_enabled())
            if args.model_kind == "qwen35"
            and callable(getattr(torch.backends.cuda, "math_sdp_enabled", None))
            else None
        ),
        "qwen_sdp_cudnn_enabled": (
            bool(torch.backends.cuda.cudnn_sdp_enabled())
            if args.model_kind == "qwen35"
            and callable(getattr(torch.backends.cuda, "cudnn_sdp_enabled", None))
            else None
        ),
        "rwkv_fast_token_backend_requested": os.environ.get("RWKV7_FAST_TOKEN_BACKEND"),
        "rwkv_native_model_backend_requested": os.environ.get(
            "RWKV7_NATIVE_MODEL_BACKEND"
        ),
        "rwkv_fast_token_quant_requested": os.environ.get("RWKV7_FAST_TOKEN_QUANT"),
        "rwkv_fast_prefill_requested": os.environ.get("RWKV7_FAST_PREFILL"),
        "rwkv_fast_prefill_quant_requested": os.environ.get("RWKV7_FAST_PREFILL_QUANT"),
        "rwkv_prefill_graph_requested": os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH"),
        "rwkv_prefill_fused_scan_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_FUSED_SCAN"
        ),
        "rwkv_prefill_global_fp16_accum_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM"
        ),
        "rwkv_prefill_block_fp16_accum_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM"
        ),
        "rwkv_prefill_external_quant_graph_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH"
        ),
        "rwkv_prefill_blas_requested": os.environ.get("RWKV7_NATIVE_PREFILL_BLAS"),
        "rwkv_prefill_self_chunk_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK"
        ),
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
        "rwkv_prefill_scan_block_m_requested": os.environ.get(
            "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M"
        ),
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
    getter = (
        config.get
        if isinstance(config, dict)
        else lambda name, default=None: getattr(config, name, default)
    )
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
        if (
            args.quantization not in {"bnb8", "bnb8_a8w8_head"}
            or native_jit_module is None
        ):
            return None
        return bool(native_jit_module._native_bnb8_policy_flag(env_name, policy_name))

    def bnb8_block(env_name: str, policy_name: str, fallback: int) -> int | None:
        if (
            args.quantization not in {"bnb8", "bnb8_a8w8_head"}
            or native_jit_module is None
        ):
            return None
        return int(
            native_jit_module._native_bnb8_policy_block(env_name, policy_name, fallback)
        )

    return {
        "bnb_int8_threshold": (
            float(getter("llm_int8_threshold", 6.0))
            if args.quantization in {"bnb8", "bnb8_a8w8_head"}
            else None
        ),
        "rwkv_bnb_skip_policy": (
            getattr(model, "_rwkv7_bnb_skip_policy", None)
            if args.model_kind == "rwkv"
            else None
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
        "quantized_modules": getattr(
            model, "_rwkv7_cross_model_quant_replaced_modules", None
        ),
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
            mm4_launch.get("fused_max_rows") if args.quantization == "mm4" else None
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
            mm4_launch.get("dot_block_b") if args.quantization == "mm4" else None
        ),
        "mm4_dot_block_pairs": (
            mm4_launch.get("dot_block_pairs") if args.quantization == "mm4" else None
        ),
        "mm4_dot_block_n": (
            mm4_launch.get("dot_block_n") if args.quantization == "mm4" else None
        ),
        "mm4_dot_warps": (
            mm4_launch.get("dot_warps") if args.quantization == "mm4" else None
        ),
        "native_quant_kernel_active": native_quant,
    }


_QWEN35_FAST_BINDING_PREFIXES = {
    "causal_conv1d_fn": (
        "causal_conv1d.",
        "bench.qwen35_fla_triton_conv",
        "qwen35_fla_triton_conv",
    ),
    "causal_conv1d_update": (
        "causal_conv1d.",
        "bench.qwen35_fla_triton_conv",
        "qwen35_fla_triton_conv",
    ),
    "chunk_gated_delta_rule": ("fla.",),
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
    if (
        args.model_kind == "rwkv"
        and str(getattr(args, "rwkv_implementation", "auto")) == "wrapper_repo"
    ):
        effective = getattr(model, "_rwkv7_benchmark_implementation_effective", None)
        if effective != "wrapper_repo" or type(model).__name__ != "RWKV7ForCausalLM":
            raise RuntimeError(
                "explicit RWKV repository wrapper was required but the loaded model "
                f"is {type(model).__module__}.{type(model).__name__} "
                f"(effective={effective!r})"
            )
    if args.model_kind != "qwen35" or not args.require_qwen_fast_path:
        return
    binding_check = qwen35_fast_path_bindings(model)
    if not bool(binding_check["verified"]):
        raise RuntimeError(
            "Qwen3.5 full optimized path was required but the loaded GatedDeltaNet "
            f"layers are not bound to FLA plus accelerated causal conv: {binding_check}"
        )


def validate_qwen_result_contract(
    args: argparse.Namespace, row: dict[str, Any]
) -> None:
    """Reject a row unless it proves the exact requested Qwen fast path.

    The official comparison lane is selected by the existing public CLI
    combination ``--qwen-conv-backend causal_conv1d`` plus
    ``--require-qwen-fast-path``. This keeps the repository Triton conv lane
    available for explicit experiments without allowing it into the official
    causal-conv main table.
    """

    if args.model_kind != "qwen35" or not bool(
        getattr(args, "require_qwen_fast_path", False)
    ):
        return
    required: dict[str, Any] = {
        "status": "pass",
        "qwen_fast_path_verified": True,
        "qwen_full_fused_contract_pass": True,
        "qwen_conv_backend_effective": str(getattr(args, "qwen_conv_backend", "auto")),
        "qwen_force_torch": False,
    }
    if str(getattr(args, "qwen_conv_backend", "auto")) == "causal_conv1d":
        required.update(
            {
                "qwen_fast_path_available": True,
                "qwen_causal_conv1d_importable": True,
            }
        )
    qwen_route = str(getattr(args, "qwen_decode_optimization", "module_call_dynamic"))
    if qwen_route in QWEN_STATIC_GRAPH_ROUTES:
        cross_cache_full_greedy_required = (
            str(getattr(args, "qwen_cross_cache_full_greedy_policy", "strict"))
            == "strict"
        )
        required.update(
            {
                "optimization_lane": "qwen_best_optimized_hf",
                "qwen_decode_optimization_effective": qwen_route,
                "qwen_cuda_graph_requested": True,
                "qwen_cuda_graph_effective": True,
                "qwen_decode_cuda_graph_verified": True,
                "qwen_graph_parity_verified": True,
                "qwen_cross_cache_full_greedy_policy_effective": str(
                    getattr(args, "qwen_cross_cache_full_greedy_policy", "strict")
                ),
                "qwen_cross_cache_full_greedy_required": cross_cache_full_greedy_required,
                "qwen_graph_prefill_next_token_match": True,
                "qwen_same_cache_greedy_match": True,
                "qwen_same_cache_full_greedy_mismatch_count": 0,
                "qwen_graph_logits_trace_finite": True,
                "qwen_graph_logits_greedy_match": True,
                "qwen_dynamic_static_logits_greedy_match": True,
                "qwen_same_cache_logits_greedy_match": True,
                "qwen_dynamic_static_logits_finite": True,
                "qwen_same_cache_logits_finite": True,
                "qwen_cache_pointer_stable": True,
                "step_backend": f"qwen_{qwen_route}",
                "prefill_backend_effective": "module_call_dynamic_cache",
                "prefill_cache_type": "DynamicCache",
                "cache_type": "StaticCache",
            }
        )
        if cross_cache_full_greedy_required:
            required.update(
                {
                    "qwen_graph_greedy_match": True,
                    "qwen_static_cache_eager_greedy_match": True,
                    "qwen_dynamic_static_full_greedy_mismatch_count": 0,
                    "qwen_dynamic_candidate_full_greedy_mismatch_count": 0,
                }
            )
        if qwen_route == "static_cache_inductor_cudagraph":
            required.update(
                {
                    "qwen_graph_break_count": 0,
                    "qwen_cudagraph_skip_count": 0,
                    "qwen_compile_backend_effective": "inductor",
                    "qwen_compile_mode_effective": str(
                        getattr(args, "qwen_compile_mode", "max-autotune")
                    ),
                    "qwen_graph_scope": "single_token_hf_qwen_forward",
                }
            )
        else:
            for field in (
                "qwen_graph_break_count",
                "qwen_cudagraph_skip_count",
                "qwen_cudagraph_recorded_non_static_inputs",
                "qwen_compile_backend_effective",
                "qwen_compile_mode_effective",
                "qwen_compile_fullgraph_effective",
                "qwen_compile_dynamic_effective",
            ):
                required[field] = None
            required["qwen_graph_scope"] = (
                "single_token_hf_qwen_forward_argmax_token_copy"
            )

    def matches_expected(actual: Any, expected: Any) -> bool:
        if isinstance(expected, bool):
            return type(actual) is bool and actual is expected
        if isinstance(expected, int):
            return type(actual) is int and actual == expected
        return actual == expected

    mismatches = [
        f"{field}={row.get(field)!r} (expected {expected!r})"
        for field, expected in required.items()
        if not matches_expected(row.get(field), expected)
    ]
    if mismatches:
        raise RuntimeError(
            "Qwen3.5 result row failed the requested fast-path contract: "
            + "; ".join(mismatches)
        )
    if qwen_route in QWEN_STATIC_GRAPH_ROUTES:
        for field in (
            "qwen_graph_logits_min_cosine",
            "qwen_dynamic_static_logits_min_cosine",
        ):
            value = row.get(field)
            if not _is_finite_real_number(value):
                raise RuntimeError(
                    "Qwen3.5 result row failed the requested CUDA Graph contract: "
                    f"{field}={value!r} (expected finite cross-cache telemetry)"
                )
        same_cache = row.get("qwen_same_cache_logits_min_cosine")
        if not _is_finite_real_number(same_cache) or same_cache < 0.9999:
            raise RuntimeError(
                "Qwen3.5 result row failed the requested CUDA Graph contract: "
                f"qwen_same_cache_logits_min_cosine={same_cache!r} "
                "(expected >=0.9999)"
            )
        launches = row.get("qwen_cuda_graph_launch_count")
        launch_count_valid = _is_finite_real_number(launches) and (
            launches == 1
            if qwen_route == "static_cache_raw_cudagraph"
            else launches > 0
        )
        if not launch_count_valid:
            expected_launches = (
                "exactly 1" if qwen_route == "static_cache_raw_cudagraph" else "> 0"
            )
            raise RuntimeError(
                "Qwen3.5 result row did not prove CUDA Graph replay: "
                f"qwen_cuda_graph_launch_count={launches!r} "
                f"(expected {expected_launches})"
            )


_RWKV_NATIVE_GRAPH_DECODE_ROUTE_FIELDS = (
    "ada_wagv_lora_extension_requested",
    "ada_wagv_lora_extension_selected",
    "ada_wagv_lora_extension_effective",
    "ada_wagv_lora_extension_selected_layers",
    "ada_wagv_lora_extension_effective_layers",
    "ada_wagv_lora_extension_effective_layer_count",
    "ada_wagv_lora_extension_full_model_effective",
    "rkv_policy",
    "fused_norm_mix_num_warps",
    "state_dtype",
    "triton_fp16_state",
    "fp16_recurrent",
    "sm70_wagv_lora_selected",
    "sm70_wagv_lora_effective",
    "sm70_wagv_lora_selected_layers",
    "sm70_wagv_lora_effective_layers",
    "sm70_wagv_lora_effective_layer_count",
    "sm70_wagv_lora_full_eligible_layers_effective",
    "sm70_wagv_lora_extension_required",
    "sm70_wagv_lora_extension_available",
    "fused_wavg_lora_selected",
    "fused_wavg_lora_effective",
    "fused_wavg_lora_selected_layers",
    "fused_wavg_lora_effective_layers",
    "fused_wavg_lora_effective_layer_count",
    "fused_wavg_lora_full_eligible_layers_effective",
    "ada_wagv_bmm_requested",
    "ada_wagv_bmm_selected",
    "ada_wagv_bmm_effective",
    "ada_wagv_bmm_selected_layers",
    "ada_wagv_bmm_effective_layers",
    "ada_wagv_bmm_effective_layer_count",
    "ada_wagv_bmm_full_model_effective",
    "sm120_wagv_bmm_g_requested",
    "sm120_wagv_bmm_g_selected",
    "sm120_wagv_bmm_g_effective",
    "sm120_wagv_bmm_g_selected_layers",
    "sm120_wagv_bmm_g_effective_layers",
    "sm120_wagv_bmm_g_effective_layer_count",
    "sm120_wagv_bmm_g_full_model_effective",
    "sm120_compiled_ffn_requested",
    "sm120_compiled_ffn_selected",
    "sm120_compiled_ffn_effective",
    "sm120_compiled_ffn_selected_layers",
    "sm120_compiled_ffn_effective_layers",
    "sm120_compiled_ffn_effective_layer_count",
    "sm120_compiled_ffn_full_model_effective",
    "sm120_compiled_ffn_compile_effective",
    "sm120_compiled_ffn_compile_reused",
    "sm120_compiled_ffn_unique_graphs",
    "sm120_compiled_ffn_graph_breaks",
    "sm120_compiled_ffn_compile_mode",
    "sm120_compiled_ffn_prewarm_all_finite",
    "sm120_compiled_ffn_prewarm_min_cosine",
    "sm120_compiled_ffn_prewarm_argmax_all_equal",
    "sm120_compiled_ffn_prewarm_max_abs_diff",
    "sm120_compiled_ffn_prewarm_layer_indices",
    "sm120_compiled_ffn_prewarm_layer_count",
)


def rwkv_native_graph_decode_route(model, batch_size: int) -> dict[str, Any]:
    """Return route truth from the exact fixed-batch runner, if present."""

    empty = {
        f"rwkv_native_graph_{name}": None
        for name in _RWKV_NATIVE_GRAPH_DECODE_ROUTE_FIELDS
    }
    getter = getattr(model, "rwkv7_native_graph_runner_copy_stats", None)
    if not callable(getter):
        return empty
    try:
        runners = getter().get("runners", [])
        match = next(
            (
                item
                for item in reversed(runners)
                if int(item.get("batch_size", -1)) == int(batch_size)
            ),
            None,
        )
    except Exception:
        return empty
    if not isinstance(match, dict):
        return empty
    return {
        f"rwkv_native_graph_{name}": match.get(name)
        for name in _RWKV_NATIVE_GRAPH_DECODE_ROUTE_FIELDS
    }


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
    ids = build_exact_prompt(
        tokenizer, args.prompt_tokens, args.batch_size, input_device
    )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cuda_device_index(args.device))
    prefill_timing = timed_prefill_details(args, model, ids)
    prefill_s = float(prefill_timing["median_s"])
    prefill_tokps = float(prefill_timing["tokps"])
    prefill_backend = (
        last_rwkv_prefill_backend(model) if args.model_kind == "rwkv" else None
    )
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
    prefill_global_fp16_accum = (
        bool(
            getattr(
                model,
                "_rwkv7_native_prefill_global_fp16_accum_effective",
                False,
            )
        )
        if args.model_kind == "rwkv"
        else None
    )
    prefill_block_fp16_accum = (
        bool(
            getattr(
                model,
                "_rwkv7_native_prefill_block_fp16_accum_effective",
                False,
            )
        )
        if args.model_kind == "rwkv"
        else None
    )
    decode_timing = timed_decode_details(args, model, ids)
    decode_s = float(decode_timing["median_s"])
    step_backend = str(decode_timing["step_backend"])
    effective_backend = decode_timing["effective_backend"]
    cache_type = str(decode_timing["cache_type"])
    rwkv_decode_route = (
        rwkv_native_graph_decode_route(model, args.batch_size)
        if args.model_kind == "rwkv"
        else {}
    )
    logits_finite = True
    with torch.inference_mode():
        check = forward_prefill(args, model, ids[:, : min(8, ids.shape[1])])
        logits_finite = bool(torch.isfinite(check.logits).all().item())
    if not logits_finite:
        raise RuntimeError("model produced non-finite logits")
    probe_metadata = save_backend_probe(args, model, ids) if args.probe_output else {}

    qwen_bindings = (
        qwen35_fast_path_bindings(model) if args.model_kind == "qwen35" else None
    )
    footprint = model_footprint_mb(model)
    peak = peak_mb(args.device)
    runtime_working_set = (
        round(max(0.0, peak - footprint), 1) if peak is not None else None
    )
    parameter_metadata = model_parameter_metadata(model, args)
    active_parameters = int(parameter_metadata["active_parameter_count"])
    active_parameter_billions = active_parameters / 1e9
    decode_tokps = (args.batch_size * args.decode_tokens) / decode_s
    row = {
        **base_row(args),
        **model_metadata(args, model),
        **environment_metadata(args, model),
        **effective_quantization_metadata(model, args),
        **rwkv_decode_route,
        **parameter_metadata,
        **qwen_contract,
        **probe_metadata,
        "status": "pass",
        "input_device": input_device,
        "prefill_sec_median": round(prefill_s, 6),
        "prefill_sec_median_raw": prefill_s,
        "prefill_sec_samples": [float(value) for value in prefill_timing["samples"]],
        "prefill_tokps_total": round(prefill_tokps, 3),
        "prefill_tokps_total_raw": prefill_tokps,
        "prefill_tokps_per_active_billion": round(
            prefill_tokps / active_parameter_billions, 6
        ),
        "prefill_active_parameter_tops": round(
            prefill_tokps * active_parameters / 1e12, 6
        ),
        "decode_sec_median": round(decode_s, 6),
        "decode_sec_median_raw": decode_s,
        "decode_sec_samples": [float(value) for value in decode_timing["samples"]],
        "decode_tokps_total": round(decode_tokps, 3),
        "decode_tokps_total_raw": decode_tokps,
        "decode_tokps_per_seq": round(args.decode_tokens / decode_s, 3),
        "decode_tokps_per_active_billion": round(
            decode_tokps / active_parameter_billions, 6
        ),
        "decode_ms_per_step": round(1000 * decode_s / args.decode_tokens, 6),
        "decode_active_parameter_tops": round(
            decode_tokps * active_parameters / 1e12, 6
        ),
        "step_backend": step_backend,
        "prefill_effective_backend": prefill_backend
        or (
            "module_call_dynamic_cache"
            if args.model_kind == "qwen35"
            and str(getattr(args, "qwen_decode_optimization", "module_call_dynamic"))
            in QWEN_STATIC_GRAPH_ROUTES
            else "module_call"
            if args.model_kind == "qwen35"
            else None
        ),
        "prefill_backend_effective": prefill_backend
        or (
            "module_call_dynamic_cache"
            if args.model_kind == "qwen35"
            and str(getattr(args, "qwen_decode_optimization", "module_call_dynamic"))
            in QWEN_STATIC_GRAPH_ROUTES
            else "module_call"
            if args.model_kind == "qwen35"
            else None
        ),
        "prefill_cache_type": "DynamicCache" if args.model_kind == "qwen35" else None,
        "rwkv_prefill_clampw_scan_effective": prefill_clampw_scan,
        "rwkv_prefill_stacked_rkv_effective": prefill_stacked_rkv,
        "rwkv_prefill_self_chunk_effective": prefill_self_chunk,
        "rwkv_prefill_sequence_ffn_effective": prefill_sequence_ffn,
        "rwkv_prefill_global_fp16_accum_effective": prefill_global_fp16_accum,
        "rwkv_prefill_block_fp16_accum_effective": prefill_block_fp16_accum,
        "effective_backend": qwen_effective_backend(args, qwen_contract)
        or effective_backend
        or step_backend,
        "qwen_fast_path_verified": qwen_bindings["verified"]
        if qwen_bindings is not None
        else None,
        "qwen_fast_path_layer_count": qwen_bindings["layer_count"]
        if qwen_bindings is not None
        else None,
        "qwen_fast_path_bindings": qwen_bindings["bindings"]
        if qwen_bindings is not None
        else None,
        "cache_type": cache_type,
        "qwen_axis_composition": (
            "independent_best_prefill_and_decode"
            if args.model_kind == "qwen35"
            and str(getattr(args, "qwen_decode_optimization", "module_call_dynamic"))
            in QWEN_STATIC_GRAPH_ROUTES
            else "continuous_single_cache_path"
            if args.model_kind == "qwen35"
            else None
        ),
        "qwen_decode_optimization_effective": decode_timing[
            "qwen_decode_optimization_effective"
        ],
        "qwen_cuda_graph_requested": decode_timing["qwen_cuda_graph_requested"],
        "qwen_cuda_graph_effective": decode_timing["qwen_cuda_graph_effective"],
        "qwen_decode_cuda_graph_verified": decode_timing[
            "qwen_decode_cuda_graph_verified"
        ],
        "qwen_graph_parity_verified": decode_timing["qwen_graph_parity_verified"],
        "qwen_cross_cache_full_greedy_policy_effective": decode_timing[
            "qwen_cross_cache_full_greedy_policy_effective"
        ],
        "qwen_cross_cache_full_greedy_required": decode_timing[
            "qwen_cross_cache_full_greedy_required"
        ],
        "qwen_graph_prefill_next_token_match": decode_timing[
            "qwen_graph_prefill_next_token_match"
        ],
        "qwen_graph_greedy_match": decode_timing["qwen_graph_greedy_match"],
        "qwen_same_cache_greedy_match": decode_timing["qwen_same_cache_greedy_match"],
        "qwen_dynamic_static_full_greedy_mismatch_count": decode_timing[
            "qwen_dynamic_static_full_greedy_mismatch_count"
        ],
        "qwen_dynamic_static_full_greedy_first_mismatch_index": decode_timing[
            "qwen_dynamic_static_full_greedy_first_mismatch_index"
        ],
        "qwen_dynamic_candidate_full_greedy_mismatch_count": decode_timing[
            "qwen_dynamic_candidate_full_greedy_mismatch_count"
        ],
        "qwen_dynamic_candidate_full_greedy_first_mismatch_index": decode_timing[
            "qwen_dynamic_candidate_full_greedy_first_mismatch_index"
        ],
        "qwen_same_cache_full_greedy_mismatch_count": decode_timing[
            "qwen_same_cache_full_greedy_mismatch_count"
        ],
        "qwen_same_cache_full_greedy_first_mismatch_index": decode_timing[
            "qwen_same_cache_full_greedy_first_mismatch_index"
        ],
        "qwen_static_cache_eager_greedy_match": decode_timing[
            "qwen_static_cache_eager_greedy_match"
        ],
        "qwen_graph_logits_greedy_match": decode_timing[
            "qwen_graph_logits_greedy_match"
        ],
        "qwen_dynamic_static_logits_greedy_match": decode_timing[
            "qwen_dynamic_static_logits_greedy_match"
        ],
        "qwen_same_cache_logits_greedy_match": decode_timing[
            "qwen_same_cache_logits_greedy_match"
        ],
        "qwen_graph_probe_tokens": decode_timing["qwen_graph_probe_tokens"],
        "qwen_graph_logits_probe_tokens": decode_timing[
            "qwen_graph_logits_probe_tokens"
        ],
        "qwen_graph_distinct_batch_prompts": decode_timing[
            "qwen_graph_distinct_batch_prompts"
        ],
        "qwen_graph_logits_min_cosine": decode_timing["qwen_graph_logits_min_cosine"],
        "qwen_graph_logits_max_abs_diff": decode_timing[
            "qwen_graph_logits_max_abs_diff"
        ],
        "qwen_graph_logits_trace_finite": decode_timing[
            "qwen_graph_logits_trace_finite"
        ],
        "qwen_graph_logits_worst_index": decode_timing["qwen_graph_logits_worst_index"],
        "qwen_dynamic_static_logits_min_cosine": decode_timing[
            "qwen_dynamic_static_logits_min_cosine"
        ],
        "qwen_dynamic_static_logits_max_abs_diff": decode_timing[
            "qwen_dynamic_static_logits_max_abs_diff"
        ],
        "qwen_dynamic_static_logits_finite": decode_timing[
            "qwen_dynamic_static_logits_finite"
        ],
        "qwen_dynamic_static_logits_worst_index": decode_timing[
            "qwen_dynamic_static_logits_worst_index"
        ],
        "qwen_static_compiled_logits_min_cosine": decode_timing[
            "qwen_static_compiled_logits_min_cosine"
        ],
        "qwen_static_compiled_logits_max_abs_diff": decode_timing[
            "qwen_static_compiled_logits_max_abs_diff"
        ],
        "qwen_static_compiled_logits_finite": decode_timing[
            "qwen_static_compiled_logits_finite"
        ],
        "qwen_static_compiled_logits_worst_index": decode_timing[
            "qwen_static_compiled_logits_worst_index"
        ],
        "qwen_same_cache_logits_min_cosine": decode_timing[
            "qwen_same_cache_logits_min_cosine"
        ],
        "qwen_same_cache_logits_max_abs_diff": decode_timing[
            "qwen_same_cache_logits_max_abs_diff"
        ],
        "qwen_same_cache_logits_finite": decode_timing["qwen_same_cache_logits_finite"],
        "qwen_same_cache_logits_worst_index": decode_timing[
            "qwen_same_cache_logits_worst_index"
        ],
        "qwen_graph_scope": decode_timing["qwen_graph_scope"],
        "qwen_graph_capture_s": decode_timing["qwen_graph_capture_s"],
        "qwen_graph_setup_s": decode_timing["qwen_graph_setup_s"],
        "qwen_graph_max_cache_len": decode_timing["qwen_graph_max_cache_len"],
        "qwen_graph_break_count": decode_timing["qwen_graph_break_count"],
        "qwen_cudagraph_skip_count": decode_timing["qwen_cudagraph_skip_count"],
        "qwen_cudagraph_recorded_non_static_inputs": decode_timing[
            "qwen_cudagraph_recorded_non_static_inputs"
        ],
        "qwen_cuda_graph_launch_count": decode_timing["qwen_cuda_graph_launch_count"],
        "qwen_cache_pointer_stable": decode_timing["qwen_cache_pointer_stable"],
        "qwen_cache_tensor_pointer_count": decode_timing[
            "qwen_cache_tensor_pointer_count"
        ],
        "qwen_compile_backend_effective": decode_timing[
            "qwen_compile_backend_effective"
        ],
        "qwen_compile_mode_effective": decode_timing["qwen_compile_mode_effective"],
        "qwen_compile_fullgraph_effective": decode_timing[
            "qwen_compile_fullgraph_effective"
        ],
        "qwen_compile_dynamic_effective": decode_timing[
            "qwen_compile_dynamic_effective"
        ],
        "model_footprint_mb": footprint,
        "peak_vram_mb": peak,
        "runtime_working_set_mb": runtime_working_set,
        "load_s": round(load_s, 3),
        "logits_finite": logits_finite,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    validate_qwen_result_contract(args, row)
    return row


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    started = time.perf_counter()
    effective_model_path = args.model
    temporary = None
    model = None
    try:
        configure_qwen_sdpa_policy(args)
        if args.model_kind == "rwkv":
            effective_model_path, temporary = prepare_rwkv_model_dir(
                args.model, args.rwkv_code_source
            )
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
    ap.add_argument(
        "--optimization-lane",
        default="",
        help="Auditable protocol lane, for example best_optimized_hf or diagnostic_no_graph.",
    )
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
    ap.add_argument(
        "--native-quant-policy", choices=["memory", "speed"], default="memory"
    )
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
    ap.add_argument(
        "--rwkv-attn-mode",
        choices=["chunk", "fused_recurrent"],
        default="fused_recurrent",
    )
    ap.add_argument("--rwkv-code-source", choices=["repo", "model"], default="repo")
    ap.add_argument(
        "--rwkv-implementation",
        choices=["auto", "wrapper_repo"],
        default="auto",
        help=(
            "RWKV model class loader. wrapper_repo bypasses converted auto_map and "
            "loads the repository FLA wrapper directly from the canonical model path."
        ),
    )
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
    ap.add_argument(
        "--qwen-sdpa-policy",
        choices=["auto", "math_only"],
        default="auto",
        help=(
            "Explicit full-attention SDPA backend policy. math_only disables "
            "flash, memory-efficient, and cuDNN SDPA before model loading."
        ),
    )
    ap.add_argument(
        "--qwen-cross-cache-full-greedy-policy",
        choices=["strict", "informational"],
        default="strict",
        help=(
            "Whether full-horizon DynamicCache-vs-StaticCache greedy equality is a "
            "hard gate. Same-cache eager-vs-graph full greedy equality always remains "
            "a hard gate; short logits traces and finite cross-cache telemetry remain "
            "hard gates under both policies."
        ),
    )
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument(
        "--qwen-decode-optimization",
        choices=["module_call_dynamic", *sorted(QWEN_STATIC_GRAPH_ROUTES)],
        default="module_call_dynamic",
        help=(
            "Decode invocation layer. The optimized Qwen lane uses HF StaticCache, "
            "and either verified Inductor CUDAGraph Trees or raw CUDA Graph replay."
        ),
    )
    ap.add_argument(
        "--qwen-graph-probe-tokens",
        type=int,
        default=16,
        help="Untimed eager/Graph greedy and logits parity steps for each captured cell.",
    )
    ap.add_argument(
        "--qwen-compile-mode",
        choices=["reduce-overhead", "max-autotune"],
        default="max-autotune",
        help=(
            "Inductor mode for the explicit Qwen StaticCache/CUDA-Graph lane. "
            "The effective mode is recorded per row and must pass the same "
            "fail-closed logits and greedy gates."
        ),
    )
    ap.add_argument("--results", default="")
    ap.add_argument("--probe-output", default="")
    ap.add_argument("--probe-tokens", type=int, default=8)
    ap.add_argument("--probe-batch-size", type=int, default=1)
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
        print(
            "QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False),
            flush=True,
        )
        if not args.optional:
            raise
        return 0
    append_row(args.results, row)
    print(
        "QWEN35_CROSS_MODEL_SPEED_RESULT " + json.dumps(row, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
