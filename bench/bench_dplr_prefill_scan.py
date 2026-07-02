#!/usr/bin/env python3
# coding=utf-8
"""Synthetic RWKV-7 DPLR/chunked prefill scan prototype benchmark.

This benchmark intentionally does not load a model and does not import/call
``native_jit``.  It only compares the pure torch recurrent reference against the
standalone ``rwkv7_hf.dplr_prefill.dplr_chunk_scan`` prototype so server runs
can validate the chunked API surface independently of the HF prefill path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Make the script runnable as either `PYTHONPATH=. python bench/...` or directly
# from the repository checkout.  This does not import native_jit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - py_compile/lightweight hosts may not have torch
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised by benchmark/test hosts with torch
    _TORCH_IMPORT_ERROR = None


DTYPE_CHOICES = ("bf16", "fp16", "fp32")
ALGORITHMS = ("sequential", "affine", "wy", "lowrank", "triton_wy", "cuda_wy")
WY_ALGORITHM_ALIASES = ("wy", "lowrank")
TRITON_WY_ALIASES = ("triton_wy", "cuda_wy")


@dataclass(frozen=True)
class AlgorithmPlan:
    requested_algorithm: str
    effective_algorithm: str | None
    status: str
    call_algorithm: str | None
    reason: str | None = None


class UnsupportedAlgorithmError(RuntimeError):
    def __init__(
        self,
        *,
        requested_algorithm: str,
        status: str = "skip_unsupported_algorithm",
        reason: str,
        effective_algorithm: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.requested_algorithm = requested_algorithm
        self.effective_algorithm = effective_algorithm
        self.status = status
        self.reason = reason


def _torch_unavailable_message() -> str:
    msg = "bench_dplr_prefill_scan.py requires torch at runtime; install torch or run on the benchmark server."
    if _TORCH_IMPORT_ERROR is not None:
        msg += f" torch import error: {_TORCH_IMPORT_ERROR}"
    return msg


def _require_positive(values: list[int], *, name: str) -> None:
    bad = [v for v in values if int(v) <= 0]
    if bad:
        raise SystemExit(f"{name} must contain positive integers; got {bad}")


def _normalize_algorithm_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_")


def _algorithm_family(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = _normalize_algorithm_name(name)
    if normalized in {"torch_recurrent_scan", "torch_reference", "reference"}:
        return "torch_reference"
    if normalized in {"sequential", "sequential_fallback"}:
        return "sequential"
    if normalized == "affine":
        return "dense_affine"
    if normalized in WY_ALGORITHM_ALIASES:
        return "lowrank_wy"
    if normalized in TRITON_WY_ALIASES:
        return "triton_wy"
    return "unknown"


def _is_dense_affine_algorithm(name: str | None) -> bool:
    return _algorithm_family(name) == "dense_affine"


def _normalize_supported_algorithms(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        values = raw.keys()
    elif isinstance(raw, str):
        values = raw.replace(",", " ").replace("/", " ").split()
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]

    found = {_normalize_algorithm_name(v) for v in values}
    return tuple(algorithm for algorithm in ALGORITHMS if algorithm in found)


def _detect_supported_algorithms(fn: Callable[..., Any], supports_algorithm: bool | None) -> tuple[str, ...] | None:
    if supports_algorithm is False:
        return ("sequential",)

    owners: list[Any] = [fn]
    module = inspect.getmodule(fn)
    if module is not None:
        owners.append(module)
    globals_dict = getattr(fn, "__globals__", None)
    if isinstance(globals_dict, dict):
        owners.append(globals_dict)

    for owner in owners:
        for attr in ("SUPPORTED_ALGORITHMS", "_SUPPORTED_ALGORITHMS", "DPLR_SUPPORTED_ALGORITHMS", "ALGORITHMS"):
            if isinstance(owner, dict):
                raw = owner.get(attr)
            else:
                raw = getattr(owner, attr, None)
            detected = _normalize_supported_algorithms(raw)
            if detected:
                return detected
    return None


def _unexpected_keyword_error(exc: TypeError, keyword: str) -> bool:
    text = str(exc).lower()
    return keyword.lower() in text and ("unexpected keyword" in text or "unexpected" in text)


def _looks_like_unsupported_algorithm(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "algorithm" in text and any(
        marker in text
        for marker in (
            "must be one of",
            "unsupported",
            "unknown",
            "not implemented",
            "invalid",
            "got",
        )
    )


def _annotate_algorithm_row(
    row: dict[str, Any],
    *,
    requested_algorithm: str,
    effective_algorithm: str | None,
    detected_algorithms: tuple[str, ...] | None,
) -> None:
    requested = _normalize_algorithm_name(requested_algorithm)
    effective = _normalize_algorithm_name(effective_algorithm) if effective_algorithm is not None else None
    row["algorithm"] = requested
    row["requested_algorithm"] = requested
    row["effective_algorithm"] = effective
    row["algorithm_family"] = _algorithm_family(effective or requested)
    row["is_dense_affine"] = _is_dense_affine_algorithm(effective or requested)
    row["detected_algorithms"] = list(detected_algorithms) if detected_algorithms is not None else None


def _dtype_from_name(name: str):
    assert torch is not None
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _cuda_sync(device: str) -> None:
    assert torch is not None
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _device_name(device: str) -> str:
    assert torch is not None
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)
    return str(dev)


def _peak_mb(device: str) -> float | None:
    assert torch is not None
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return round(torch.cuda.max_memory_allocated(idx) / 1024 / 1024, 1)


def _normalize_last_dim(x: Any):
    assert torch is not None
    return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(dtype=x.dtype)


def make_inputs(
    *,
    B: int,
    T: int,
    H: int,
    N: int,
    device: str,
    dtype: Any,
    seed: int,
) -> dict[str, Any]:
    """Create small synthetic post-projection tensors for DPLR scan only."""

    assert torch is not None
    dev = torch.device(device)
    gen_device = dev if dev.type == "cuda" else torch.device("cpu")
    g = torch.Generator(device=gen_device)
    g.manual_seed(int(seed))

    shape = (B, T, H, N)
    r = torch.randn(shape, device=gen_device, dtype=dtype, generator=g) * 0.20
    k = torch.randn(shape, device=gen_device, dtype=dtype, generator=g) * 0.20
    v = torch.randn(shape, device=gen_device, dtype=dtype, generator=g) * 0.20
    kk = _normalize_last_dim(torch.randn(shape, device=gen_device, dtype=dtype, generator=g))
    a = torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype, generator=g))

    # Match native_jit/fused_recurrent_update convention: this benchmark feeds
    # the already exponentiated positive decay vector, not raw W logits.
    w_log = (-0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype, generator=g).float())).to(dtype)
    w_decay = torch.exp(w_log.float()).to(dtype)
    state = torch.randn(B, H, N, N, device=gen_device, dtype=torch.float32, generator=g) * 0.01

    if gen_device != dev:
        r = r.to(dev)
        k = k.to(dev)
        v = v.to(dev)
        kk = kk.to(dev)
        a = a.to(dev)
        w_decay = w_decay.to(dev)
        state = state.to(dev)

    return {"r": r, "w": w_decay, "k": k, "v": v, "kk": kk, "a": a, "state": state}


def timed(fn: Callable[[], Any], *, device: str, warmup: int, steps: int) -> float:
    assert torch is not None
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            fn()
    _cuda_sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(1, int(steps))):
            fn()
    _cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0 / max(1, int(steps))


def pair_diff(got: tuple[Any, Any], ref: tuple[Any, Any]) -> dict[str, float]:
    assert torch is not None
    out_got, state_got = got
    out_ref, state_ref = ref
    out_delta = (out_got.float() - out_ref.float()).abs()
    state_delta = (state_got.float() - state_ref.float()).abs()

    flat_got = out_got.float().reshape(out_got.shape[0], -1)
    flat_ref = out_ref.float().reshape(out_ref.shape[0], -1)
    denom = flat_got.norm(dim=-1) * flat_ref.norm(dim=-1)
    cosine = torch.where(
        denom > 0,
        (flat_got * flat_ref).sum(dim=-1) / denom.clamp_min(1e-12),
        torch.ones_like(denom),
    )
    return {
        "out_max_abs_diff": float(out_delta.max().detach().cpu()) if out_delta.numel() else 0.0,
        "state_max_abs_diff": float(state_delta.max().detach().cpu()) if state_delta.numel() else 0.0,
        "out_min_cosine": float(cosine.min().detach().cpu()) if cosine.numel() else 1.0,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _supports_keyword(fn: Callable[..., Any], keyword: str) -> bool | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    if keyword in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class DplrInvoker:
    """Compatibility wrapper for current and future dplr_chunk_scan signatures."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.supports_algorithm = _supports_keyword(fn, "algorithm")
        self.supports_force_fallback = _supports_keyword(fn, "force_fallback")
        self.supported_algorithms = _detect_supported_algorithms(fn, self.supports_algorithm)

    def _plans_for_no_algorithm_keyword(self, requested_algorithm: str) -> list[AlgorithmPlan]:
        if requested_algorithm == "sequential":
            return [
                AlgorithmPlan(
                    requested_algorithm=requested_algorithm,
                    effective_algorithm="sequential",
                    status="pass",
                    call_algorithm=None,
                )
            ]
        if requested_algorithm == "affine":
            return [
                AlgorithmPlan(
                    requested_algorithm=requested_algorithm,
                    effective_algorithm="sequential_fallback",
                    status="fallback_no_algorithm_arg",
                    call_algorithm=None,
                    reason="dplr_chunk_scan has no algorithm= parameter; using sequential fallback",
                )
            ]
        return []

    def plans_for(self, algorithm: str) -> list[AlgorithmPlan]:
        requested_algorithm = _normalize_algorithm_name(algorithm)

        if self.supports_algorithm is False:
            return self._plans_for_no_algorithm_keyword(requested_algorithm)

        if self.supported_algorithms is not None:
            if requested_algorithm in self.supported_algorithms:
                return [
                    AlgorithmPlan(
                        requested_algorithm=requested_algorithm,
                        effective_algorithm=requested_algorithm,
                        status="pass",
                        call_algorithm=requested_algorithm,
                    )
                ]
            if requested_algorithm in WY_ALGORITHM_ALIASES:
                for alias in WY_ALGORITHM_ALIASES:
                    if alias != requested_algorithm and alias in self.supported_algorithms:
                        return [
                            AlgorithmPlan(
                                requested_algorithm=requested_algorithm,
                                effective_algorithm=alias,
                                status="fallback_algorithm_alias",
                                call_algorithm=alias,
                                reason=f"{requested_algorithm!r} not detected; using {alias!r} WY/lowrank alias",
                            )
                        ]
            if requested_algorithm in TRITON_WY_ALIASES:
                for alias in TRITON_WY_ALIASES:
                    if alias != requested_algorithm and alias in self.supported_algorithms:
                        return [
                            AlgorithmPlan(
                                requested_algorithm=requested_algorithm,
                                effective_algorithm=alias,
                                status="fallback_algorithm_alias",
                                call_algorithm=alias,
                                reason=f"{requested_algorithm!r} not detected; using {alias!r} compiled DPLR/WY alias",
                            )
                        ]
            return []

        plans = [
            AlgorithmPlan(
                requested_algorithm=requested_algorithm,
                effective_algorithm=requested_algorithm,
                status="pass",
                call_algorithm=requested_algorithm,
            )
        ]
        if requested_algorithm in WY_ALGORITHM_ALIASES:
            for alias in WY_ALGORITHM_ALIASES:
                if alias != requested_algorithm:
                    plans.append(
                        AlgorithmPlan(
                            requested_algorithm=requested_algorithm,
                            effective_algorithm=alias,
                            status="fallback_algorithm_alias",
                            call_algorithm=alias,
                            reason=f"trying {alias!r} as a WY/lowrank alias",
                        )
                    )
        if requested_algorithm in TRITON_WY_ALIASES:
            for alias in TRITON_WY_ALIASES:
                if alias != requested_algorithm:
                    plans.append(
                        AlgorithmPlan(
                            requested_algorithm=requested_algorithm,
                            effective_algorithm=alias,
                            status="fallback_algorithm_alias",
                            call_algorithm=alias,
                            reason=f"trying {alias!r} as a compiled DPLR/WY alias",
                        )
                    )
        return plans

    def call_plan(self, xs: dict[str, Any], *, chunk_size: int, plan: AlgorithmPlan) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {"chunk_size": int(chunk_size)}
        if plan.call_algorithm is not None:
            kwargs["algorithm"] = plan.call_algorithm
        elif self.supports_force_fallback is True and plan.status == "fallback_no_algorithm_arg":
            kwargs["force_fallback"] = True
        return self.fn(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"], **kwargs)

    def __call__(self, xs: dict[str, Any], *, chunk_size: int, algorithm: str) -> tuple[tuple[Any, Any], AlgorithmPlan]:
        requested_algorithm = _normalize_algorithm_name(algorithm)
        plans = self.plans_for(requested_algorithm)
        if not plans:
            detected = list(self.supported_algorithms) if self.supported_algorithms is not None else None
            raise UnsupportedAlgorithmError(
                requested_algorithm=requested_algorithm,
                reason=f"dplr_chunk_scan does not advertise support for {requested_algorithm!r}; detected_algorithms={detected}",
            )

        unsupported_errors: list[str] = []
        for plan in plans:
            try:
                return self.call_plan(xs, chunk_size=chunk_size, plan=plan), plan
            except TypeError as exc:
                if self.supports_algorithm is None and plan.call_algorithm is not None and _unexpected_keyword_error(exc, "algorithm"):
                    # Last-resort compatibility fallback for callables where
                    # inspect cannot see the signature.  Only sequential and
                    # historical affine rows can safely fall back to the
                    # no-algorithm API; WY/lowrank should skip instead.
                    self.supports_algorithm = False
                    self.supported_algorithms = ("sequential",)
                    fallback_plans = self._plans_for_no_algorithm_keyword(requested_algorithm)
                    if fallback_plans:
                        fallback_plan = fallback_plans[0]
                        return self.call_plan(xs, chunk_size=chunk_size, plan=fallback_plan), fallback_plan
                    raise UnsupportedAlgorithmError(
                        requested_algorithm=requested_algorithm,
                        reason="dplr_chunk_scan does not accept algorithm= and no safe fallback exists for WY/lowrank",
                    ) from exc
                if _looks_like_unsupported_algorithm(exc):
                    unsupported_errors.append(str(exc))
                    continue
                raise
            except (ValueError, NotImplementedError) as exc:
                if _looks_like_unsupported_algorithm(exc):
                    unsupported_errors.append(str(exc))
                    continue
                raise

        raise UnsupportedAlgorithmError(
            requested_algorithm=requested_algorithm,
            reason="; ".join(unsupported_errors) or f"dplr_chunk_scan rejected {requested_algorithm!r}",
        )


def base_row(args: argparse.Namespace, *, B: int, T: int, H: int, N: int) -> dict[str, Any]:
    return {
        "axis": "dplr_prefill_scan_proto",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": _device_name(args.device),
        "B": B,
        "T": T,
        "H": H,
        "N": N,
        "batch_size": B,
        "tokens": T,
        "heads": H,
        "head_dim": N,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
    }


def finish_timing_row(row: dict[str, Any], *, B: int, T: int, ms: float | None) -> None:
    row["ms"] = round(ms, 5) if ms is not None else None
    row["tokps"] = round((B * T) / (ms / 1000.0), 1) if ms else None


def emit(row: dict[str, Any], *, results: str) -> None:
    print(json.dumps(row, ensure_ascii=False), flush=True)
    append_jsonl(results, row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic DPLR/chunked prefill scan benchmark; no model and no native_jit.")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    ap.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    ap.add_argument("--seed", type=int, default=7007)
    args = ap.parse_args()

    if torch is None:
        print(_torch_unavailable_message(), file=sys.stderr)
        return 2

    _require_positive(args.batch_sizes, name="--batch-sizes")
    _require_positive(args.tokens, name="--tokens")
    _require_positive(args.chunk_sizes, name="--chunk-sizes")
    _require_positive([args.heads], name="--heads")
    _require_positive([args.head_dim], name="--head-dim")
    if int(args.warmup) < 0:
        raise SystemExit("--warmup must be >= 0")
    _require_positive([args.steps], name="--steps")

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA device requested ({args.device}) but torch.cuda.is_available() is false", file=sys.stderr)
        return 2

    try:
        from rwkv7_hf.dplr_prefill import dplr_chunk_scan
        from rwkv7_hf.fused_recurrent_update import torch_recurrent_scan
    except Exception as exc:
        print(f"failed to import DPLR benchmark dependencies: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        from rwkv7_hf.dplr_prefill_triton import dplr_chunk_scan_triton_available
    except Exception:
        dplr_chunk_scan_triton_available = None  # type: ignore[assignment]

    dtype = _dtype_from_name(args.dtype)
    dplr_call = DplrInvoker(dplr_chunk_scan)
    triton_wy_available = bool(
        dplr_chunk_scan_triton_available is not None and dplr_chunk_scan_triton_available()
    )
    triton_wy_block_m = int(os.environ.get("RWKV7_DPLR_TRITON_BLOCK_M", "8"))
    rows = 0

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    for B in args.batch_sizes:
        for T in args.tokens:
            H = int(args.heads)
            N = int(args.head_dim)
            xs = make_inputs(B=int(B), T=int(T), H=H, N=N, device=args.device, dtype=dtype, seed=args.seed + int(B) * 100000 + int(T))

            try:
                with torch.inference_mode():
                    ref = torch_recurrent_scan(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"])
                ref_ms = timed(
                    lambda: torch_recurrent_scan(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"]),
                    device=args.device,
                    warmup=args.warmup,
                    steps=args.steps,
                )
                row = base_row(args, B=int(B), T=int(T), H=H, N=N)
                _annotate_algorithm_row(
                    row,
                    requested_algorithm="torch_recurrent_scan",
                    effective_algorithm="torch_recurrent_scan",
                    detected_algorithms=dplr_call.supported_algorithms,
                )
                row.update(
                    {
                        "chunk_size": None,
                        "status": "pass",
                        "out_max_abs_diff": 0.0,
                        "state_max_abs_diff": 0.0,
                        "out_min_cosine": 1.0,
                        "peak_vram_mb": _peak_mb(args.device),
                    }
                )
                finish_timing_row(row, B=int(B), T=int(T), ms=ref_ms)
                emit(row, results=args.results)
                rows += 1
            except Exception as exc:
                row = base_row(args, B=int(B), T=int(T), H=H, N=N)
                _annotate_algorithm_row(
                    row,
                    requested_algorithm="torch_recurrent_scan",
                    effective_algorithm="torch_recurrent_scan",
                    detected_algorithms=dplr_call.supported_algorithms,
                )
                row.update(
                    {
                        "chunk_size": None,
                        "status": f"error:{type(exc).__name__}",
                        "error": str(exc),
                        "ms": None,
                        "tokps": None,
                        "out_max_abs_diff": None,
                        "state_max_abs_diff": None,
                        "out_min_cosine": None,
                        "peak_vram_mb": _peak_mb(args.device),
                    }
                )
                emit(row, results=args.results)
                rows += 1
                continue

            for chunk_size in args.chunk_sizes:
                for algorithm in args.algorithms:
                    requested_algorithm = _normalize_algorithm_name(algorithm)
                    row = base_row(args, B=int(B), T=int(T), H=H, N=N)
                    _annotate_algorithm_row(
                        row,
                        requested_algorithm=requested_algorithm,
                        effective_algorithm=None,
                        detected_algorithms=dplr_call.supported_algorithms,
                    )
                    row.update({"chunk_size": int(chunk_size)})
                    try:
                        with torch.inference_mode():
                            got, plan = dplr_call(xs, chunk_size=int(chunk_size), algorithm=requested_algorithm)
                        ms = timed(
                            lambda: dplr_call.call_plan(xs, chunk_size=int(chunk_size), plan=plan),
                            device=args.device,
                            warmup=args.warmup,
                            steps=args.steps,
                        )
                        row.update(pair_diff(got, ref))
                        _annotate_algorithm_row(
                            row,
                            requested_algorithm=plan.requested_algorithm,
                            effective_algorithm=plan.effective_algorithm,
                            detected_algorithms=dplr_call.supported_algorithms,
                        )
                        row.update(
                            {
                                "status": plan.status,
                                "triton_wy_available": triton_wy_available,
                                "triton_wy_block_m": triton_wy_block_m if (row.get("algorithm_family") == "triton_wy") else None,
                                "peak_vram_mb": _peak_mb(args.device),
                            }
                        )
                        if plan.reason is not None:
                            row["fallback_reason"] = plan.reason
                        finish_timing_row(row, B=int(B), T=int(T), ms=ms)
                    except UnsupportedAlgorithmError as exc:
                        _annotate_algorithm_row(
                            row,
                            requested_algorithm=exc.requested_algorithm,
                            effective_algorithm=exc.effective_algorithm,
                            detected_algorithms=dplr_call.supported_algorithms,
                        )
                        row.update(
                            {
                                "status": exc.status,
                                "skip_reason": exc.reason,
                                "triton_wy_available": triton_wy_available,
                                "triton_wy_block_m": triton_wy_block_m if row.get("algorithm_family") == "triton_wy" else None,
                                "ms": None,
                                "tokps": None,
                                "out_max_abs_diff": None,
                                "state_max_abs_diff": None,
                                "out_min_cosine": None,
                                "peak_vram_mb": _peak_mb(args.device),
                            }
                        )
                    except Exception as exc:
                        row.update(
                            {
                                "status": f"error:{type(exc).__name__}",
                                "error": str(exc),
                                "triton_wy_available": triton_wy_available,
                                "triton_wy_block_m": triton_wy_block_m if row.get("algorithm_family") == "triton_wy" else None,
                                "ms": None,
                                "tokps": None,
                                "out_max_abs_diff": None,
                                "state_max_abs_diff": None,
                                "out_min_cosine": None,
                                "peak_vram_mb": _peak_mb(args.device),
                            }
                        )
                    emit(row, results=args.results)
                    rows += 1

    print(f"\nappended {rows} row(s) -> {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
