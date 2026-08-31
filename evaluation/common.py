from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_TRAINING_PADDING_MODES = frozenset(("none", "left", "right"))


def training_case_seed(
    base_seed: int,
    *,
    batch: int,
    tokens: int,
    padding: str = "none",
    sample_index: int = 0,
) -> int:
    """Derive one order-independent training sample seed.

    Checkpointing is intentionally not an input: checkpoint-on and
    checkpoint-off lanes must consume identical token IDs.  Additional random
    samples are represented explicitly by ``sample_index`` instead of by the
    iteration order of one mutable generator.
    """

    if batch <= 0 or tokens <= 0:
        raise ValueError("batch and tokens must be positive")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if padding not in _TRAINING_PADDING_MODES:
        choices = ", ".join(sorted(_TRAINING_PADDING_MODES))
        raise ValueError(f"padding must be one of {choices}; got {padding!r}")
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "batch": int(batch),
            "tokens": int(tokens),
            "padding": padding,
            "sample_index": int(sample_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="little",
        signed=False,
    ) & ((1 << 63) - 1)


def input_ids_sha256(value) -> str:
    """Hash token IDs together with their dtype and exact shape."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("input IDs must be a torch.Tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict:
    payload_paths = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and (
            path.suffix in {".json", ".jinja", ".model", ".py", ".safetensors", ".txt"}
            or path.name.endswith(".safetensors.index.json")
        )
    )
    payloads = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_paths
    }
    aggregate = hashlib.sha256()
    for name, row in payloads.items():
        aggregate.update(f"{name}\0{row['sha256']}\n".encode())
    config = model_dir / "config.json"
    return {
        "path": str(model_dir.resolve()),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "resolved_revision": aggregate.hexdigest(),
        "payloads": payloads,
        "weights": [
            {"name": name, **row}
            for name, row in payloads.items()
            if name.endswith(".safetensors")
        ],
    }


def cuda_toolkit_provenance() -> dict:
    """Record the external CUDA compiler used by lazy extension builds."""

    raw_home = os.environ.get("CUDA_HOME")
    home = Path(raw_home).expanduser().resolve() if raw_home else None
    nvcc = home / "bin" / "nvcc" if home is not None else None
    provenance = home / "PROVENANCE.txt" if home is not None else None
    nvcc_version = None
    if nvcc is not None and nvcc.is_file() and not nvcc.is_symlink():
        try:
            nvcc_version = subprocess.check_output(
                [str(nvcc), "--version"],
                text=True,
                stderr=subprocess.STDOUT,
            ).splitlines()
        except Exception as exc:
            nvcc_version = [f"error: {type(exc).__name__}: {exc}"]
    return {
        "cuda_home": str(home) if home is not None else None,
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "nvcc": str(nvcc) if nvcc is not None and nvcc.is_file() else None,
        "nvcc_version": nvcc_version,
        "provenance": (
            {
                "path": str(provenance),
                "sha256": sha256_file(provenance),
            }
            if provenance is not None
            and provenance.is_file()
            and not provenance.is_symlink()
            else None
        ),
    }


def environment() -> dict:
    import torch
    import transformers

    def package(name: str):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except Exception:
        driver = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_linear_attention": package("flash-linear-attention"),
        "triton": package("triton"),
        "accelerate": package("accelerate"),
        "datasets": package("datasets"),
        "peft": package("peft"),
        "trl": package("trl"),
        "wandb": package("wandb"),
        "bitsandbytes": package("bitsandbytes"),
        "torchao": package("torchao"),
        "lm_eval": package("lm_eval"),
        "rwkv7_hf": package("rwkv7-hf"),
        "rwkv7_kernels": package("rwkv7-kernels"),
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend_environment": {
            name: os.environ.get(name)
            for name in (
                "RWKV7_BACKEND",
                "RWKV7_KERNEL_IMPL",
                "RWKV7_MODEL_KERNEL_IMPL",
                "RWKV7_TRAINING_KERNEL_IMPL",
            )
        },
        "cuda_toolkit": cuda_toolkit_provenance(),
    }


def write_bundle(output_dir: Path, name: str, report: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    jsonl_path = output_dir / f"{name}.jsonl"
    markdown_path = output_dir / f"{name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    jsonl_path.write_text(
        json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {name}",
        "",
        f"- status: **{report.get('status', 'unknown')}**",
        f"- model: {report.get('model', {}).get('path')}",
        f"- dtype: {report.get('dtype')}",
        f"- device: {report.get('environment', {}).get('gpu')}",
        f"- code: {report.get('code_sha')}",
    ]
    if report.get("fla_commit") is not None:
        lines.append(f"- FLA: {report.get('fla_commit')}")
    lines.extend(
        [
            "",
            "| case | cosine | max abs | mean abs | argmax |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for case, row in report.get("comparisons", {}).items():
        lines.append(
            f"| {case} | {row['cosine']:.8f} | {row['max_abs']:.8f} | "
            f"{row['mean_abs']:.8f} | {row['argmax_same']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, markdown_path
