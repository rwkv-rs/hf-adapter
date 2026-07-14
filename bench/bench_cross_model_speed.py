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
    if args.model_role not in {"candidate", "reference"}:
        raise ValueError("--model-role must be candidate or reference")
    if args.model_kind not in {"rwkv", "qwen35"}:
        raise ValueError("--model-kind must be rwkv or qwen35")
    if args.qwen_backend not in {"auto", "fla", "torch"}:
        raise ValueError("--qwen-backend must be auto, fla, or torch")
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
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
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


def model_footprint_mb(model) -> float:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)
        total += tensor.numel() * tensor.element_size()
    return round(total / 1024 / 1024, 1)


def quantization_config(args: argparse.Namespace, dtype: torch.dtype):
    if args.quantization == "none":
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is required for bnb8/bnb4 rows")
    from transformers import BitsAndBytesConfig

    if args.quantization == "bnb8":
        return BitsAndBytesConfig(load_in_8bit=True)
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
        set_rwkv_runtime(model, args)
        return model

    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("installed Transformers does not provide Qwen3_5ForCausalLM") from exc
    return Qwen3_5ForCausalLM.from_pretrained(args.model, **kwargs).eval()


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


def qwen_fla_operator_contract(model) -> dict[str, Any]:
    """Inspect the operators actually bound by every Qwen3.5 linear layer.

    Transformers silently substitutes Python/Torch reference functions when
    FLA or causal-conv1d is unavailable. Package presence alone therefore is
    not sufficient evidence that the benchmark used accelerated kernels. The
    Gated DeltaNet and norm operators form the required FLA core contract;
    causal-conv1d is tracked as a separate optional acceleration capability so
    Windows exact-card rows can exercise FLA without overstating full fusion.
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
        _origin_is(_operator_origin(getattr(layer, "chunk_gated_delta_rule", None)), ("fla",))
        for _, layer in layers
    )
    decode_fla_layers = sum(
        _origin_is(_operator_origin(getattr(layer, "recurrent_gated_delta_rule", None)), ("fla",))
        for _, layer in layers
    )
    conv_update_fused_layers = sum(
        _origin_is(_operator_origin(getattr(layer, "causal_conv1d_update", None)), ("causal_conv1d",))
        for _, layer in layers
    )
    conv_prefill_fused_layers = sum(
        getattr(layer, "causal_conv1d_fn", None) is not None
        and _origin_is(_operator_origin(getattr(layer, "causal_conv1d_fn", None)), ("causal_conv1d",))
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
            return "qwen_fla_gated_delta_rule"
        return "qwen_fla_gated_delta_rule_torch_conv"
    return "transformers_torch_fallback"


def last_rwkv_backend(model) -> str | None:
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


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


def forward_prefill(model, ids):
    return model(ids, use_cache=True, logits_to_keep=1)


def timed_prefill(args: argparse.Namespace, model, ids) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = forward_prefill(model, ids)
    cuda_sync(args.device)
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(args.runs):
            cuda_sync(args.device)
            started = time.perf_counter()
            _ = forward_prefill(model, ids)
            cuda_sync(args.device)
            samples.append(time.perf_counter() - started)
    median_s = float(statistics.median(samples))
    return median_s, (args.batch_size * args.prompt_tokens) / median_s


def decode_once(args: argparse.Namespace, model, ids, step: Callable[..., Any]) -> tuple[float, Any]:
    with torch.inference_mode():
        out = forward_prefill(model, ids)
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
        out = forward_prefill(model, probe_ids)
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


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
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
        "qwen_fla_expected_device_route": qwen_device_route,
        "rwkv_fast_token_backend_requested": os.environ.get("RWKV7_FAST_TOKEN_BACKEND"),
        "rwkv_fast_token_quant_requested": os.environ.get("RWKV7_FAST_TOKEN_QUANT"),
        "rwkv_fast_prefill_requested": os.environ.get("RWKV7_FAST_PREFILL"),
        "rwkv_fast_prefill_quant_requested": os.environ.get("RWKV7_FAST_PREFILL_QUANT"),
        "rwkv_prefill_graph_requested": os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH"),
        "rwkv_prefill_fused_scan_requested": os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_SCAN"),
        "rwkv_prefill_scan_block_m_requested": os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M"),
        "rwkv_prefill_scan_num_warps_requested": os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS"),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    started = time.perf_counter()
    effective_model_path = args.model
    temporary = None
    if args.model_kind == "rwkv":
        effective_model_path, temporary = prepare_rwkv_model_dir(args.model, args.rwkv_code_source)
    tokenizer = AutoTokenizer.from_pretrained(effective_model_path, trust_remote_code=args.model_kind == "rwkv")
    model = load_model(args, dtype, effective_model_path)
    qwen_contract = enforce_qwen_backend(model, args)
    load_s = time.perf_counter() - started
    input_device = str(next(model.parameters()).device)
    ids = build_exact_prompt(tokenizer, args.prompt_tokens, args.batch_size, input_device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cuda_device_index(args.device))
    prefill_s, prefill_tokps = timed_prefill(args, model, ids)
    decode_s, step_backend, effective_backend, cache_type = timed_decode(args, model, ids)
    logits_finite = True
    with torch.inference_mode():
        check = forward_prefill(model, ids[:, : min(8, ids.shape[1])])
        logits_finite = bool(torch.isfinite(check.logits).all().item())
    if not logits_finite:
        raise RuntimeError("model produced non-finite logits")
    probe_metadata = save_backend_probe(args, model, ids) if args.probe_output else {}

    row = {
        **base_row(args),
        **model_metadata(args, model),
        **environment_metadata(args),
        **qwen_contract,
        **probe_metadata,
        "status": "pass",
        "input_device": input_device,
        "prefill_sec_median": round(prefill_s, 6),
        "prefill_tokps_total": round(prefill_tokps, 3),
        "decode_sec_median": round(decode_s, 6),
        "decode_tokps_total": round((args.batch_size * args.decode_tokens) / decode_s, 3),
        "decode_tokps_per_seq": round(args.decode_tokens / decode_s, 3),
        "decode_ms_per_step": round(1000 * decode_s / args.decode_tokens, 6),
        "step_backend": step_backend,
        "effective_backend": qwen_effective_backend(args, qwen_contract) or effective_backend or step_backend,
        "prefill_backend_effective": getattr(model, "rwkv7_last_fast_prefill_backend", lambda: None)(),
        "cache_type": cache_type,
        "model_footprint_mb": model_footprint_mb(model),
        "peak_vram_mb": peak_mb(args.device),
        "load_s": round(load_s, 3),
        "logits_finite": logits_finite,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if temporary is not None:
        temporary.cleanup()
    return row


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
    ap.add_argument("--quantization", default="none", choices=["none", "bnb8", "bnb4"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=128)
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
