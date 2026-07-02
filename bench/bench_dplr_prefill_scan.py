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
import inspect
import json
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
ALGORITHMS = ("sequential", "affine")


def _torch_unavailable_message() -> str:
    msg = "bench_dplr_prefill_scan.py requires torch at runtime; install torch or run on the benchmark server."
    if _TORCH_IMPORT_ERROR is not None:
        msg += f" torch import error: {_TORCH_IMPORT_ERROR}"
    return msg


def _require_positive(values: list[int], *, name: str) -> None:
    bad = [v for v in values if int(v) <= 0]
    if bad:
        raise SystemExit(f"{name} must contain positive integers; got {bad}")


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

    def __call__(self, xs: dict[str, Any], *, chunk_size: int, algorithm: str) -> tuple[tuple[Any, Any], str, str]:
        kwargs: dict[str, Any] = {"chunk_size": int(chunk_size)}
        status = "pass"
        effective_algorithm = algorithm

        if self.supports_algorithm is True:
            kwargs["algorithm"] = algorithm
        elif self.supports_algorithm is False:
            # The current prototype has no algorithm= parameter and is explicitly
            # sequential inside chunks.  Keep affine rows runnable, but mark them
            # as fallback so server JSONL cannot be mistaken for a real affine
            # implementation result.
            if self.supports_force_fallback is True:
                kwargs["force_fallback"] = True
            if algorithm == "affine":
                status = "fallback_no_algorithm_arg"
                effective_algorithm = "sequential_fallback"
        else:
            # Signature introspection failed; optimistically try the future API.
            kwargs["algorithm"] = algorithm

        try:
            out = self.fn(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"], **kwargs)
            return out, status, effective_algorithm
        except TypeError as exc:
            if self.supports_algorithm is None and "algorithm" in kwargs:
                # Last-resort compatibility fallback for callables where inspect
                # cannot see the signature.
                kwargs.pop("algorithm", None)
                if self.supports_force_fallback is True:
                    kwargs["force_fallback"] = True
                out = self.fn(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"], **kwargs)
                fallback_status = "fallback_no_algorithm_arg" if algorithm == "affine" else "pass"
                fallback_effective = "sequential_fallback" if algorithm == "affine" else "sequential"
                return out, fallback_status, fallback_effective
            raise exc


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

    dtype = _dtype_from_name(args.dtype)
    dplr_call = DplrInvoker(dplr_chunk_scan)
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
                row.update(
                    {
                        "algorithm": "torch_recurrent_scan",
                        "effective_algorithm": "torch_recurrent_scan",
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
                row.update(
                    {
                        "algorithm": "torch_recurrent_scan",
                        "effective_algorithm": "torch_recurrent_scan",
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
                    row = base_row(args, B=int(B), T=int(T), H=H, N=N)
                    row.update({"algorithm": algorithm, "chunk_size": int(chunk_size)})
                    try:
                        with torch.inference_mode():
                            got, status, effective_algorithm = dplr_call(xs, chunk_size=int(chunk_size), algorithm=algorithm)
                        ms = timed(lambda: dplr_call(xs, chunk_size=int(chunk_size), algorithm=algorithm)[0], device=args.device, warmup=args.warmup, steps=args.steps)
                        row.update(pair_diff(got, ref))
                        row.update(
                            {
                                "effective_algorithm": effective_algorithm,
                                "status": status,
                                "peak_vram_mb": _peak_mb(args.device),
                            }
                        )
                        finish_timing_row(row, B=int(B), T=int(T), ms=ms)
                    except Exception as exc:
                        row.update(
                            {
                                "effective_algorithm": None,
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

    print(f"\nappended {rows} row(s) -> {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
