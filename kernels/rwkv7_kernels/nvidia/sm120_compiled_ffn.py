# coding=utf-8
"""Strict opt-in compiled dense FFN route for SM86/SM89/SM120 native-graph decode.

The runtime boundary is the exact RWKV dense FFN expression used by the
native CUDA-graph token loop::

    residual + linear(relu(linear(x, up_weight)) ** 2, down_weight)

One shape-polymorphic ``torch.compile`` callable is cached per device/shape and
accepts every layer's weights as tensor inputs.  Preparation compiles and
executes the callable with all 24 distinct layer-weight pairs before raw CUDA
graph capture.  There is intentionally no eager fallback after the route is
requested: unsupported contracts, compatibility patches, compilation errors,
recompiles, or numerical-gate failures raise an actionable ``RuntimeError``.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F


ENV_NAME = "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN"
EXPECTED_BATCH = 8
EXPECTED_LAYERS = 24
SUPPORTED_HIDDEN = (1024, 2048)
MIN_COSINE = 0.9999
COMPILE_MODE = "max-autotune-no-cudagraphs"

_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _CompiledFFNEntry:
    callable: Callable[..., torch.Tensor]
    unique_graphs_after_compile: int
    graph_breaks_after_compile: int


_COMPILED: dict[tuple[Any, ...], _CompiledFFNEntry] = {}
_PREPARED: dict[tuple[Any, ...], "CompiledFFNPreparation"] = {}
_FAILURES: dict[tuple[Any, ...], str] = {}
_LOCK = threading.RLock()


@dataclass(frozen=True)
class CompiledFFNPreparation:
    """Evidence recorded after all layer operands pass lazy warmup."""

    hidden_size: int
    batch_size: int
    layer_indices: tuple[int, ...]
    min_cosine: float
    max_abs_diff: float
    argmax_all_equal: bool
    all_finite: bool
    compile_effective: bool = True
    compile_reused: bool = True
    unique_graphs_compiled: int = 1
    graph_breaks: int = 0
    compile_mode: str = COMPILE_MODE


_PREPARATION_TELEMETRY_DEFAULTS: dict[str, Any] = {
    "sm120_compiled_ffn_compile_effective": None,
    "sm120_compiled_ffn_compile_reused": None,
    "sm120_compiled_ffn_unique_graphs": None,
    "sm120_compiled_ffn_graph_breaks": None,
    "sm120_compiled_ffn_compile_mode": None,
    "sm120_compiled_ffn_prewarm_all_finite": None,
    "sm120_compiled_ffn_prewarm_min_cosine": None,
    "sm120_compiled_ffn_prewarm_argmax_all_equal": None,
    "sm120_compiled_ffn_prewarm_max_abs_diff": None,
    "sm120_compiled_ffn_prewarm_layer_indices": None,
    "sm120_compiled_ffn_prewarm_layer_count": None,
}


def sm120_compiled_ffn_preparation_stats(
    preparation: CompiledFFNPreparation | None,
) -> dict[str, Any]:
    """Return stable runner/cross-row telemetry for compile and prewarm truth."""

    if not isinstance(preparation, CompiledFFNPreparation):
        return dict(_PREPARATION_TELEMETRY_DEFAULTS)
    layer_indices = [int(value) for value in preparation.layer_indices]
    return {
        "sm120_compiled_ffn_compile_effective": bool(preparation.compile_effective),
        "sm120_compiled_ffn_compile_reused": bool(preparation.compile_reused),
        "sm120_compiled_ffn_unique_graphs": int(preparation.unique_graphs_compiled),
        "sm120_compiled_ffn_graph_breaks": int(preparation.graph_breaks),
        "sm120_compiled_ffn_compile_mode": str(preparation.compile_mode),
        "sm120_compiled_ffn_prewarm_all_finite": bool(preparation.all_finite),
        "sm120_compiled_ffn_prewarm_min_cosine": float(preparation.min_cosine),
        "sm120_compiled_ffn_prewarm_argmax_all_equal": bool(
            preparation.argmax_all_equal
        ),
        "sm120_compiled_ffn_prewarm_max_abs_diff": float(preparation.max_abs_diff),
        "sm120_compiled_ffn_prewarm_layer_indices": layer_indices,
        "sm120_compiled_ffn_prewarm_layer_count": len(layer_indices),
    }


def sm120_compiled_ffn_contract(
    *,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dtype_name: str,
    capability: tuple[int, int],
) -> bool:
    """Return whether a model signature is inside the measured route."""

    return bool(
        int(batch_size) == EXPECTED_BATCH
        and int(hidden_size) in SUPPORTED_HIDDEN
        and int(num_layers) == EXPECTED_LAYERS
        and str(dtype_name).lower() in {"torch.float16", "float16", "fp16"}
        and tuple(int(value) for value in capability) in {(8, 6), (8, 9), (12, 0)}
    )


def _enabled_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _capability(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    return tuple(
        int(value) for value in torch.cuda.get_device_capability(_device_index(device))
    )


def _cache_key(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
) -> tuple[Any, ...]:
    return (
        device.type,
        _device_index(device),
        dtype,
        int(batch_size),
        int(hidden_size),
        4 * int(hidden_size),
        COMPILE_MODE,
    )


def _dense_ffn_linear_add(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.relu(F.linear(x, up_weight)).square()
    return residual + F.linear(hidden, down_weight)


def _validate_weight_pair(
    up_weight: Any,
    down_weight: Any,
    *,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    layer_index: int,
) -> None:
    if not isinstance(up_weight, torch.Tensor) or not isinstance(
        down_weight, torch.Tensor
    ):
        raise RuntimeError(
            f"{ENV_NAME}=1 requires dense Tensor FFN weights; layer {layer_index} is quantized or callable"
        )
    expected_up = (4 * int(hidden_size), int(hidden_size))
    expected_down = (int(hidden_size), 4 * int(hidden_size))
    if (
        tuple(up_weight.shape) != expected_up
        or tuple(down_weight.shape) != expected_down
    ):
        raise RuntimeError(
            f"{ENV_NAME}=1 requires FFN shapes {expected_up}/{expected_down}; "
            f"layer {layer_index} has {tuple(up_weight.shape)}/{tuple(down_weight.shape)}"
        )
    if up_weight.device != device or down_weight.device != device:
        raise RuntimeError(f"{ENV_NAME}=1 requires every FFN weight on {device}")
    if up_weight.dtype != dtype or down_weight.dtype != dtype:
        raise RuntimeError(f"{ENV_NAME}=1 requires every FFN weight to use {dtype}")
    if not up_weight.is_contiguous() or not down_weight.is_contiguous():
        raise RuntimeError(
            f"{ENV_NAME}=1 requires standard contiguous dense FFN weights; layer {layer_index} "
            "was already relaid out for sparse FFN. Enable the compiled route before model/graph "
            "loading and restart this process."
        )


def _resolve_compile() -> Callable[..., Any]:
    for name in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        if _enabled_environment(name):
            raise RuntimeError(
                f"{ENV_NAME}=1 cannot run while {name}=1; set the compiled FFN switch before "
                "importing the RWKV remote-code model"
            )
    compile_fn = getattr(torch, "_rwkv7_original_compile", None)
    if not callable(compile_fn):
        compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError(f"{ENV_NAME}=1 requires torch.compile")
    return compile_fn


def _dynamo_compile_counters() -> tuple[int, int]:
    """Read the same compile/reuse evidence used by the isolated probe."""

    dynamo = getattr(torch, "_dynamo", None)
    utils = getattr(dynamo, "utils", None)
    counters = getattr(utils, "counters", None)
    if counters is None or not hasattr(counters, "get"):
        raise RuntimeError(
            f"{ENV_NAME}=1 requires torch._dynamo compile counters so shared-callable "
            "reuse can be verified"
        )
    stats = counters.get("stats", {})
    graph_breaks = counters.get("graph_break", {})
    try:
        unique_graphs = int(stats.get("unique_graphs", 0))
        graph_break_count = sum(int(value) for value in graph_breaks.values())
    except Exception as exc:
        raise RuntimeError(
            f"{ENV_NAME}=1 could not read torch._dynamo compile counters"
        ) from exc
    return unique_graphs, graph_break_count


def _require_compile_counter_delta(
    before: tuple[int, int],
    after: tuple[int, int],
    *,
    expected_unique_graphs: int,
    context: str,
) -> None:
    unique_delta = int(after[0]) - int(before[0])
    graph_break_delta = int(after[1]) - int(before[1])
    if unique_delta != int(expected_unique_graphs) or graph_break_delta != 0:
        raise RuntimeError(
            f"compiled FFN {context}: unique_graph_delta={unique_delta}, "
            f"graph_break_delta={graph_break_delta}"
        )


def _deterministic_inputs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(
        -0.125,
        0.125,
        int(hidden_size),
        device=device,
        dtype=dtype,
    )
    rows = [torch.roll(base, shifts=row * 17) for row in range(int(batch_size))]
    x = torch.stack(rows, dim=0).contiguous()
    residual = torch.stack([row.flip(0) for row in rows], dim=0).contiguous()
    return x, residual


def _compile_callable(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    residual: torch.Tensor,
) -> _CompiledFFNEntry:
    compile_fn = _resolve_compile()
    unique_before, graph_breaks_before = _dynamo_compile_counters()
    compiled = compile_fn(
        _dense_ffn_linear_add,
        backend="inductor",
        mode=COMPILE_MODE,
        fullgraph=True,
        dynamic=False,
    )
    if compiled is _dense_ffn_linear_add:
        raise RuntimeError(
            f"{ENV_NAME}=1 requested a compiled route, but torch.compile returned the eager callable"
        )
    for _ in range(3):
        compiled(x, up_weight, down_weight, residual)
    torch.cuda.synchronize(x.device)
    unique_after, graph_breaks_after = _dynamo_compile_counters()
    _require_compile_counter_delta(
        (unique_before, graph_breaks_before),
        (unique_after, graph_breaks_after),
        expected_unique_graphs=1,
        context="did not produce exactly one full graph",
    )
    return _CompiledFFNEntry(
        callable=compiled,
        unique_graphs_after_compile=unique_after,
        graph_breaks_after_compile=graph_breaks_after,
    )


def prepare_sm120_compiled_ffn(
    packs,
    batch_size: int,
) -> CompiledFFNPreparation:
    """Compile once and warm every layer before native CUDA graph capture."""

    packs = list(packs)
    if not packs:
        raise RuntimeError(f"{ENV_NAME}=1 requires non-empty native graph packs")
    layer_indices = tuple(int(pack[0]) for pack in packs)
    if layer_indices != tuple(range(EXPECTED_LAYERS)):
        raise RuntimeError(
            f"{ENV_NAME}=1 requires exactly layers 0..{EXPECTED_LAYERS - 1}; got {layer_indices}"
        )
    up_weights = [pack[-3] for pack in packs]
    down_weights = [pack[-2] for pack in packs]
    first_up = up_weights[0]
    if not isinstance(first_up, torch.Tensor) or first_up.dim() != 2:
        raise RuntimeError(f"{ENV_NAME}=1 requires dense rank-2 FFN weights")
    hidden_size = int(first_up.shape[1])
    device = first_up.device
    dtype = first_up.dtype
    capability = _capability(device)
    if not sm120_compiled_ffn_contract(
        batch_size=int(batch_size),
        hidden_size=hidden_size,
        num_layers=len(packs),
        dtype_name=str(dtype),
        capability=capability,
    ):
        raise RuntimeError(
            f"{ENV_NAME}=1 supports only SM86/SM89/SM120, FP16, B8, 24 layers, and hidden "
            f"{SUPPORTED_HIDDEN}; got capability={capability}, dtype={dtype}, "
            f"batch={batch_size}, layers={len(packs)}, hidden={hidden_size}"
        )
    for layer_index, (up_weight, down_weight) in enumerate(
        zip(up_weights, down_weights, strict=True)
    ):
        _validate_weight_pair(
            up_weight,
            down_weight,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            layer_index=layer_index,
        )
    key = _cache_key(
        device=device,
        dtype=dtype,
        batch_size=int(batch_size),
        hidden_size=hidden_size,
    )
    with _LOCK:
        failure = _FAILURES.get(key)
        if failure is not None:
            raise RuntimeError(f"{ENV_NAME}=1 previously failed preparation: {failure}")
        x, residual = _deterministic_inputs(
            device=device,
            dtype=dtype,
            batch_size=int(batch_size),
            hidden_size=hidden_size,
        )
        try:
            with torch.no_grad():
                entry = _COMPILED.get(key)
                if entry is None:
                    entry = _compile_callable(
                        x, up_weights[0], down_weights[0], residual
                    )
                    _COMPILED[key] = entry
                compiled = entry.callable
                unique_before_reuse, graph_breaks_before_reuse = (
                    _dynamo_compile_counters()
                )
                min_cosine = 1.0
                max_abs_diff = 0.0
                argmax_all_equal = True
                all_finite = True
                for up_weight, down_weight in zip(
                    up_weights, down_weights, strict=True
                ):
                    reference = _dense_ffn_linear_add(
                        x, up_weight, down_weight, residual
                    )
                    candidate = compiled(x, up_weight, down_weight, residual)
                    reference32 = reference.float()
                    candidate32 = candidate.float()
                    finite = bool(
                        torch.isfinite(reference32).all().item()
                        and torch.isfinite(candidate32).all().item()
                    )
                    all_finite = bool(all_finite and finite)
                    cosine = F.cosine_similarity(reference32, candidate32, dim=-1)
                    min_cosine = min(min_cosine, float(cosine.min().item()))
                    max_abs_diff = max(
                        max_abs_diff,
                        float((reference32 - candidate32).abs().max().item()),
                    )
                    argmax_all_equal = bool(
                        argmax_all_equal
                        and torch.equal(
                            reference32.argmax(dim=-1),
                            candidate32.argmax(dim=-1),
                        )
                    )
                torch.cuda.synchronize(device)
                unique_after_reuse, graph_breaks_after_reuse = (
                    _dynamo_compile_counters()
                )
                _require_compile_counter_delta(
                    (unique_before_reuse, graph_breaks_before_reuse),
                    (unique_after_reuse, graph_breaks_after_reuse),
                    expected_unique_graphs=0,
                    context="callable recompiled across layer weights",
                )
                if (
                    not all_finite
                    or not math.isfinite(min_cosine)
                    or min_cosine < MIN_COSINE
                    or not argmax_all_equal
                ):
                    raise RuntimeError(
                        "compiled FFN numerical warmup failed: "
                        f"finite={all_finite}, min_cosine={min_cosine}, "
                        f"argmax_all_equal={argmax_all_equal}, max_abs_diff={max_abs_diff}"
                    )
                preparation = CompiledFFNPreparation(
                    hidden_size=hidden_size,
                    batch_size=int(batch_size),
                    layer_indices=layer_indices,
                    min_cosine=min_cosine,
                    max_abs_diff=max_abs_diff,
                    argmax_all_equal=argmax_all_equal,
                    all_finite=all_finite,
                    unique_graphs_compiled=1,
                    graph_breaks=0,
                )
                _PREPARED[key] = preparation
                return preparation
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _FAILURES[key] = message
            _COMPILED.pop(key, None)
            _PREPARED.pop(key, None)
            raise RuntimeError(
                f"{ENV_NAME}=1 preparation failed and eager fallback is forbidden: {message}"
            ) from exc


def sm120_compiled_ffn(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Execute a previously prepared callable, or fail without fallback."""

    if x.dim() != 2:
        raise RuntimeError(f"{ENV_NAME}=1 requires a rank-2 B8 FFN input")
    rows = int(x.shape[0])
    hidden_size = int(x.shape[-1])
    capability = _capability(x.device)
    if (
        rows != EXPECTED_BATCH
        or hidden_size not in SUPPORTED_HIDDEN
        or x.dtype != torch.float16
        or capability not in {(8, 6), (8, 9), (12, 0)}
    ):
        raise RuntimeError(
            f"{ENV_NAME}=1 reached an unsupported decode input: "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, capability={capability}"
        )
    if not x.is_contiguous():
        raise RuntimeError(f"{ENV_NAME}=1 requires a contiguous FFN input")
    if (
        tuple(residual.shape) != (EXPECTED_BATCH, hidden_size)
        or residual.dtype != torch.float16
        or residual.device != x.device
        or not residual.is_contiguous()
    ):
        raise RuntimeError(
            f"{ENV_NAME}=1 requires residual to be exact contiguous "
            f"[{EXPECTED_BATCH}, {hidden_size}] FP16 on {x.device}"
        )
    key = _cache_key(
        device=x.device,
        dtype=x.dtype,
        batch_size=rows,
        hidden_size=hidden_size,
    )
    preparation = _PREPARED.get(key)
    entry = _COMPILED.get(key)
    if preparation is None or entry is None:
        raise RuntimeError(
            f"{ENV_NAME}=1 reached decode before successful prewarm; raw CUDA graph "
            "capture is forbidden until prepare_sm120_compiled_ffn passes"
        )
    if torch.is_grad_enabled():
        raise RuntimeError(f"{ENV_NAME}=1 is inference-only")
    _validate_weight_pair(
        up_weight,
        down_weight,
        hidden_size=hidden_size,
        device=x.device,
        dtype=x.dtype,
        layer_index=-1,
    )
    return entry.callable(x, up_weight, down_weight, residual)


def clear_sm120_compiled_ffn_cache() -> None:
    """Clear process-local compiled state; intended for tests and teardown."""

    with _LOCK:
        _COMPILED.clear()
        _PREPARED.clear()
        _FAILURES.clear()


__all__ = [
    "COMPILE_MODE",
    "CompiledFFNPreparation",
    "ENV_NAME",
    "clear_sm120_compiled_ffn_cache",
    "prepare_sm120_compiled_ffn",
    "sm120_compiled_ffn",
    "sm120_compiled_ffn_contract",
    "sm120_compiled_ffn_preparation_stats",
]
