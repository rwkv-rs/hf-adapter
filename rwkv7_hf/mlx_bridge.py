# coding=utf-8
"""Small MLX bridge utilities for the Apple Silicon backend lane.

The current Apple path is the FLA-free PyTorch/MPS native backend.  This module
is the first MLX-native seam: it loads selected tensors from converted HF
``model.safetensors`` files into MLX arrays, saves MLX safetensors, and records
portable tensor telemetry.  It deliberately stays optional; importing
``rwkv7_hf`` on Linux/CPU hosts must not require MLX.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


try:  # pragma: no cover - exercised on Apple hosts with optional mlx extra
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]

try:  # pragma: no cover - optional in static import tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None  # type: ignore[assignment]


def mlx_available() -> bool:
    """Return whether MLX is importable in this Python environment."""

    return mx is not None


def require_mlx():
    if mx is None:
        raise RuntimeError("MLX is not installed. On Apple Silicon install the optional extra: pip install -e '.[mlx]'")
    return mx


def mlx_dtype(name: str | None):
    """Map a user dtype string to an MLX dtype. ``None`` / ``keep`` preserves dtype."""

    if name is None or name == "" or name == "keep":
        return None
    m = require_mlx()
    table = {
        "fp32": m.float32,
        "float32": m.float32,
        "fp16": m.float16,
        "float16": m.float16,
        "bf16": m.bfloat16,
        "bfloat16": m.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported MLX dtype {name!r}; expected keep/fp32/fp16/bf16") from exc


def torch_dtype_for_mlx(name: str | None):
    if torch is None:
        raise RuntimeError("torch is required to load HF safetensors before MLX conversion")
    if name is None or name == "" or name == "keep":
        return None
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; expected keep/fp32/fp16/bf16") from exc


def torch_tensor_to_mlx(tensor: Any, *, dtype: str | None = "keep"):
    """Convert a torch tensor to an MLX array on the default MLX device."""

    m = require_mlx()
    if torch is None:
        raise RuntimeError("torch is required for torch_tensor_to_mlx")
    target_torch_dtype = torch_dtype_for_mlx(dtype)
    t = tensor.detach().cpu().contiguous()
    if target_torch_dtype is not None:
        t = t.to(target_torch_dtype)
    # NumPy cannot represent torch.bfloat16 directly; stage through fp32 and
    # cast to bf16 in MLX below when requested.
    if str(t.dtype) == "torch.bfloat16":
        arr = m.array(t.float().numpy()).astype(m.bfloat16)
    else:
        arr = m.array(t.numpy())
        target_mx_dtype = mlx_dtype(dtype)
        if target_mx_dtype is not None:
            arr = arr.astype(target_mx_dtype)
    m.eval(arr)
    return arr


def mlx_array_nbytes(array: Any) -> int:
    return int(array.size) * int(array.itemsize)


def summarize_mlx_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    bytes_by_dtype: dict[str, int] = {}
    total_params = 0
    total_bytes = 0
    for value in arrays.values():
        dtype = str(value.dtype)
        n = int(value.size)
        b = mlx_array_nbytes(value)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + n
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + b
        total_params += n
        total_bytes += b
    return {
        "tensor_count": len(arrays),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
        "bytes_by_dtype": bytes_by_dtype,
    }


def reset_mlx_peak_memory() -> None:
    """Reset MLX peak-memory counters when the runtime exposes them."""

    m = require_mlx()
    reset = getattr(m, "reset_peak_memory", None)
    if callable(reset):
        reset()


def mlx_memory_telemetry() -> dict[str, int]:
    """Return best-effort MLX memory counters in bytes.

    MLX exposes these counters on Apple hosts.  Keeping them behind a helper
    lets scripts record memory telemetry without making non-MLX imports fail.
    """

    m = require_mlx()
    out: dict[str, int] = {}
    for key, attr in (
        ("mlx_active_memory_bytes", "get_active_memory"),
        ("mlx_peak_memory_bytes", "get_peak_memory"),
        ("mlx_cache_memory_bytes", "get_cache_memory"),
    ):
        fn = getattr(m, attr, None)
        if callable(fn):
            out[key] = int(fn())
    return out


def hf_safetensor_files(model_dir: str | Path) -> list[Path]:
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {root}")
    return files


def list_hf_safetensor_keys(model_dir: str | Path) -> list[str]:
    if safe_open is None:
        raise RuntimeError("safetensors is required to inspect HF model tensors")
    keys: list[str] = []
    for path in hf_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.extend(list(handle.keys()))
    return sorted(keys)


def load_selected_hf_tensors_as_mlx(
    model_dir: str | Path,
    *,
    include: Iterable[str] | None = None,
    tensor_regex: str | None = None,
    dtype: str | None = "keep",
    max_tensors: int | None = None,
) -> dict[str, Any]:
    """Load selected HF safetensors into MLX arrays.

    Selection is by exact tensor names in ``include`` and/or a regex.  This
    keeps Apple smoke light: e.g. load only one projection matrix instead of a
    full 1.5B checkpoint when validating the MLX bridge.
    """

    if safe_open is None:
        raise RuntimeError("safetensors is required to load HF model tensors")
    if torch is None:
        raise RuntimeError("torch is required to load HF safetensors before MLX conversion")
    require_mlx()
    include_set = set(include or [])
    pattern = re.compile(tensor_regex) if tensor_regex else None
    arrays: dict[str, Any] = {}
    for path in hf_safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                matched = key in include_set or (pattern.search(key) is not None if pattern is not None else False)
                if not matched:
                    continue
                arrays[key] = torch_tensor_to_mlx(handle.get_tensor(key), dtype=dtype)
                if max_tensors is not None and len(arrays) >= max_tensors:
                    return arrays
    missing = sorted(include_set.difference(arrays))
    if missing:
        raise KeyError(f"missing required tensor(s) in {model_dir}: {missing}")
    if not arrays:
        raise ValueError("no tensors selected for MLX loading")
    return arrays


def save_mlx_safetensors(arrays: dict[str, Any], output: str | Path, metadata: dict[str, str] | None = None) -> Path:
    m = require_mlx()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save_safetensors(out, arrays, metadata=metadata)
    return out


def write_mlx_manifest(
    output_dir: str | Path,
    *,
    source_model: str | Path,
    arrays: dict[str, Any],
    dtype: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "rwkv7_hf_mlx_safetensors_v1",
        "source_model": str(source_model),
        "dtype": dtype,
        **summarize_mlx_arrays(arrays),
    }
    if extra:
        manifest.update(extra)
    path = root / "mlx_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def copy_hf_metadata_files(model_dir: str | Path, output_dir: str | Path) -> list[str]:
    """Copy lightweight HF metadata files next to an MLX tensor export."""

    copied: list[str] = []
    root = Path(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = root / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)
    return copied
