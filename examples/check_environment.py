#!/usr/bin/env python3
"""Check whether this machine is ready to run an RWKV-7 HF model."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path


MIN_PYTHON = (3, 10)
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer_config.json",
    "rwkv_vocab_v20230424.txt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional converted HF model directory to validate",
    )
    return parser


def inspect_model_directory(model_dir: Path) -> list[str]:
    if not model_dir.exists():
        return [f"model path does not exist: {model_dir}"]
    if not model_dir.is_dir():
        return [f"model path is not a directory: {model_dir}"]

    problems = [
        f"missing {name}"
        for name in REQUIRED_MODEL_FILES
        if not (model_dir / name).is_file()
    ]
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    if not has_weights:
        problems.append("missing model weights (*.safetensors or *.bin)")
    return problems


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def report_musa_devices(musa: object) -> None:
    """Report optional MUSA metadata without overriding availability."""
    try:
        device_count = getattr(musa, "device_count")
        if not callable(device_count):
            raise TypeError("device_count is not callable")
        count = int(device_count())
    except Exception as exc:
        print("[PASS] MUSA: available")
        print(f"[WARN] MUSA device count unavailable: {type(exc).__name__}: {exc}")
        return

    print(f"[PASS] MUSA: available ({count} device(s))")
    get_device_name = getattr(musa, "get_device_name", None)
    for index in range(count):
        try:
            if not callable(get_device_name):
                raise TypeError("get_device_name is not callable")
            name = get_device_name(index)
        except Exception as exc:
            print(
                f"[WARN] MUSA device {index} name unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(f"[INFO] MUSA device {index}: {name}")


def report_ascend_devices(npu: object) -> None:
    """Report torch-npu devices after its optional private backend is loaded."""

    try:
        device_count = getattr(npu, "device_count")
        if not callable(device_count):
            raise TypeError("device_count is not callable")
        count = int(device_count())
    except Exception as exc:
        print("[PASS] Huawei Ascend NPU: available")
        print(
            f"[WARN] Ascend NPU device count unavailable: {type(exc).__name__}: {exc}"
        )
        return

    print(f"[PASS] Huawei Ascend NPU: available ({count} device(s))")
    get_device_name = getattr(npu, "get_device_name", None)
    for index in range(count):
        try:
            if not callable(get_device_name):
                raise TypeError("get_device_name is not callable")
            name = get_device_name(index)
        except Exception as exc:
            print(
                f"[WARN] Ascend NPU {index} name unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(f"[INFO] Ascend NPU {index}: {name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = 0

    print("RWKV-7 HF environment check")
    print(f"[INFO] OS: {platform.platform()}")
    python_ok = sys.version_info >= MIN_PYTHON
    print(
        f"[{'PASS' if python_ok else 'FAIL'}] Python: "
        f"{platform.python_version()} (requires 3.10 or newer)"
    )
    failures += not python_ok

    for package in ("torch", "transformers", "safetensors"):
        version = package_version(package)
        print(
            f"[{'PASS' if version else 'FAIL'}] {package}: {version or 'not installed'}"
        )
        failures += version is None
    torch_npu_version = package_version("torch-npu")
    print(f"[INFO] torch-npu: {torch_npu_version or 'not installed (optional)'}")

    try:
        import torch
    except (
        Exception
    ) as exc:  # Import failures can include missing CUDA/DLL dependencies.
        print(f"[FAIL] PyTorch import: {type(exc).__name__}: {exc}")
        failures += 1
    else:
        try:
            from rwkv7_hf.ascend_runtime import import_torch_npu

            import_torch_npu(required=False)
        except Exception:
            pass
        npu = getattr(torch, "npu", None)
        npu_is_available = getattr(npu, "is_available", None)
        try:
            has_ascend = bool(callable(npu_is_available) and npu_is_available())
        except Exception:
            has_ascend = False
        musa = getattr(torch, "musa", None)
        musa_is_available = getattr(musa, "is_available", None)
        try:
            has_musa = bool(callable(musa_is_available) and musa_is_available())
        except Exception:
            has_musa = False
        if torch.cuda.is_available():
            print(f"[PASS] CUDA: available ({torch.cuda.device_count()} device(s))")
            for index in range(torch.cuda.device_count()):
                print(
                    f"[INFO] CUDA device {index}: {torch.cuda.get_device_name(index)}"
                )
            print(
                "[INFO] Recommended first run: --device cuda --backend auto --dtype fp16"
            )
        elif has_ascend:
            report_ascend_devices(npu)
            print(
                "[INFO] Recommended first run: --device npu --backend native --dtype bf16"
            )
        elif has_musa:
            report_musa_devices(musa)
            print(
                "[INFO] Recommended first run: --device musa --backend native --dtype fp16"
            )
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            print("[PASS] Apple MPS: available")
            print(
                "[INFO] Recommended first run: --device mps --backend native --dtype fp16"
            )
        else:
            print("[INFO] GPU backend: unavailable; CPU fallback will be used")
            print(
                "[INFO] Recommended first run: --device cpu --backend native --dtype fp32"
            )

    if args.model is not None:
        problems = inspect_model_directory(args.model)
        if problems:
            for problem in problems:
                print(f"[FAIL] Model: {problem}")
            failures += len(problems)
        else:
            print(f"[PASS] Model directory: {args.model}")

    if failures:
        print(f"RESULT: NOT READY ({failures} problem(s))")
        print(
            "Read docs/USER_GUIDE_ZH.md or docs/USER_GUIDE.md, fix the first "
            "FAIL, and run this check again."
        )
        return 1

    print("RESULT: READY")
    if args.model is None:
        print("Next: download/convert a model, then rerun with --model PATH_TO_MODEL")
    else:
        print("Next: run examples/generate.py with this model directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
