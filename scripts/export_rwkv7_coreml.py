#!/usr/bin/env python3
# coding=utf-8
"""Prototype RWKV-7 HF -> CoreML export entry point.

This is the Apple/mobile counterpart to the MLX baseline work.  The script is
import-safe on machines without CoreMLTools and has two layers:

1. ``--dry-run`` writes a CoreML export manifest and command/evidence plan.
2. live export currently supports a first ``full_logits`` CoreML package from a
   HF model, with optional CoreMLTools torch quantization/palettization knobs.

The final production lane still needs stateful ``decode``/``prefill`` CoreML
functions and ANE benchmark rows.  The manifest schema is intentionally already
stateful-aware so the follow-up can extend the implementation without changing
documentation or result parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

AXIS = "rwkv7_coreml_export"
SUPPORTED_STATE_MODES = {"tensor", "coreml", "wkv-coreml"}
SUPPORTED_EXPORT_KINDS = {"full-logits", "stateful-plan"}
SUPPORTED_QUANTIZATION = {"none", "int8", "int4", "lut8", "lut6", "lut4"}
SUPPORTED_COMPUTE_UNITS = {"all", "cpu-and-ne", "cpu-and-gpu", "cpu-only"}
SUPPORTED_DEPLOYMENT_TARGETS = {"iOS18", "macOS15"}


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_config(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing config.json in {model_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"--{name} must be positive, got {value}")
    return value


def validate_args(args: argparse.Namespace) -> None:
    positive_int(args.chunks, name="chunks")
    positive_int(args.prefill_seq_length, name="prefill-seq-length")
    positive_int(args.sample_seq_length, name="sample-seq-length")
    if args.state_mode not in SUPPORTED_STATE_MODES:
        raise ValueError(f"unsupported state mode {args.state_mode!r}")
    if args.export_kind not in SUPPORTED_EXPORT_KINDS:
        raise ValueError(f"unsupported export kind {args.export_kind!r}")
    if args.quantization not in SUPPORTED_QUANTIZATION:
        raise ValueError(f"unsupported quantization {args.quantization!r}")
    if args.compute_units not in SUPPORTED_COMPUTE_UNITS:
        raise ValueError(f"unsupported compute units {args.compute_units!r}")
    if args.deployment_target not in SUPPORTED_DEPLOYMENT_TARGETS:
        raise ValueError(f"unsupported deployment target {args.deployment_target!r}")


def model_shape_summary(config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(config.get("hidden_size", config.get("n_embd", 0)) or 0)
    num_layers = int(config.get("num_hidden_layers", config.get("n_layer", 0)) or 0)
    num_heads = int(config.get("num_heads", config.get("n_head", 0)) or 0)
    head_dim = int(config.get("head_dim", hidden_size // num_heads if num_heads else 0) or 0)
    vocab_size = int(config.get("vocab_size", 0) or 0)
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": config.get("max_position_embeddings"),
    }


def make_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(args.model)
    output_dir = Path(args.output)
    base_name = args.basename or model_dir.name
    functions: list[dict[str, Any]] = []
    if args.export_kind == "full-logits":
        functions.append(
            {
                "name": "full_logits",
                "implemented": True,
                "input": {"input_ids": [1, int(args.sample_seq_length)]},
                "output": {"logits": [1, int(args.sample_seq_length), model_shape_summary(config).get("vocab_size")]},
            }
        )
    functions.extend(
        [
            {
                "name": "decode",
                "implemented": False,
                "planned_input": {"input_ids": [1, 1]},
                "state_mode": args.state_mode,
            },
            {
                "name": "prefill",
                "implemented": False,
                "planned_input": {"input_ids": [1, int(args.prefill_seq_length)]},
                "state_mode": args.state_mode,
            },
        ]
    )
    return {
        "format": "rwkv7_coreml_export_manifest_v1",
        "axis": AXIS,
        "source_model": str(model_dir),
        "output_dir": str(output_dir),
        "basename": base_name,
        "export_kind": args.export_kind,
        "state_mode": args.state_mode,
        "chunks": int(args.chunks),
        "prefill_seq_length": int(args.prefill_seq_length),
        "sample_seq_length": int(args.sample_seq_length),
        "quantization": args.quantization,
        "compute_units": args.compute_units,
        "deployment_target": args.deployment_target,
        "minimum_os_note": "Stateful CoreML RWKV decode/prefill is planned for iOS18/macOS15+.",
        "shape": model_shape_summary(config),
        "functions": functions,
        "follow_up_required": [
            "stateful decode/prefill CoreML functions",
            "CoreML state serialization correctness test",
            "ANE runtime benchmark rows in qwen35_apple_baseline schema",
            "CoreML int4/lut4 accuracy and speed gates",
        ],
    }


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "coreml_export_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def import_coreml_stack(require: bool) -> tuple[Any, Any, Any, Any] | None:
    try:
        import numpy as np
        import torch
        import coremltools as ct
        from transformers import AutoModelForCausalLM

        return np, torch, ct, AutoModelForCausalLM
    except Exception:
        # Keep the CLI CI-safe and import-safe on non-Apple or minimal machines.
        # ``main`` maps this skip to exit code 2 when --require-coremltools is
        # requested, so automation can make the dependency mandatory without
        # getting an unstructured Python traceback for the common missing-stack
        # case.
        return None


def coreml_compute_unit(ct: Any, value: str) -> Any:
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu-and-ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-only": ct.ComputeUnit.CPU_ONLY,
    }[value]


def coreml_target(ct: Any, value: str) -> Any:
    return {
        "iOS18": ct.target.iOS18,
        "macOS15": ct.target.macOS15,
    }[value]


def apply_torch_compression(model: Any, quantization: str) -> Any:
    if quantization == "none":
        return model
    if quantization in {"int8", "int4"}:
        from coremltools.optimize.torch.quantization import PostTrainingQuantizer, PostTrainingQuantizerConfig

        if quantization == "int8":
            config = PostTrainingQuantizerConfig.from_dict(
                {"global_config": {"weight_dtype": "int8", "granularity": "per_channel"}}
            )
        else:
            config = PostTrainingQuantizerConfig.from_dict(
                {"global_config": {"weight_dtype": "int4", "granularity": "per_block", "block_size": 128}}
            )
        return PostTrainingQuantizer(model, config).compress()
    if quantization in {"lut8", "lut6", "lut4"}:
        from coremltools.optimize.torch.palettization import PostTrainingPalettizer, PostTrainingPalettizerConfig

        bits = int(quantization.replace("lut", ""))
        group_size = 128 if bits == 8 else 16 if bits == 6 else 32
        config = PostTrainingPalettizerConfig.from_dict(
            {"global_config": {"n_bits": bits, "granularity": "per_grouped_channel", "group_size": group_size}}
        )
        return PostTrainingPalettizer(model, config).compress()
    raise ValueError(f"unsupported quantization {quantization!r}")


def export_full_logits(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    stack = import_coreml_stack(args.require_coremltools)
    if stack is None:
        row = {
            "axis": AXIS,
            "status": "skip",
            "reason": "coremltools/torch/transformers stack not installed",
            "model": Path(args.model).name,
            "export_kind": args.export_kind,
            "manifest": str(Path(args.output) / "coreml_export_manifest.json"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        return row
    np, torch, ct, AutoModelForCausalLM = stack

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any) -> Any:
            out = self.wrapped(input_ids=input_ids, use_cache=False)
            return out.logits

    os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    model = apply_torch_compression(model, args.quantization)
    wrapper = LogitsWrapper(model).eval()
    sample = torch.zeros((1, int(args.sample_seq_length)), dtype=torch.int64)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=sample.shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        minimum_deployment_target=coreml_target(ct, args.deployment_target),
        compute_units=coreml_compute_unit(ct, args.compute_units),
        compute_precision=ct.precision.FLOAT16,
    )
    package_name = f"{manifest['basename']}-full-logits"
    if args.quantization != "none":
        package_name += f"-{args.quantization}"
    output_path = Path(args.output) / f"{package_name}.mlpackage"
    mlmodel.save(str(output_path))
    elapsed = time.perf_counter() - t0
    return {
        "axis": AXIS,
        "status": "pass",
        "model": Path(args.model).name,
        "export_kind": args.export_kind,
        "quantization": args.quantization,
        "state_mode": args.state_mode,
        "coreml_package": str(output_path),
        "manifest": str(Path(args.output) / "coreml_export_manifest.json"),
        "elapsed_s": round(float(elapsed), 6),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Prototype RWKV-7 HF -> CoreML export.")
    ap.add_argument("model", help="Converted RWKV-7 HF model directory")
    ap.add_argument("output", help="Output CoreML export directory")
    ap.add_argument("--basename", default="", help="Output basename; defaults to model directory name")
    ap.add_argument("--export-kind", default="full-logits", choices=sorted(SUPPORTED_EXPORT_KINDS))
    ap.add_argument("--state-mode", default="wkv-coreml", choices=sorted(SUPPORTED_STATE_MODES))
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--prefill-seq-length", type=int, default=16)
    ap.add_argument("--sample-seq-length", type=int, default=16)
    ap.add_argument("--quantization", default="none", choices=sorted(SUPPORTED_QUANTIZATION))
    ap.add_argument("--compute-units", default="cpu-and-ne", choices=sorted(SUPPORTED_COMPUTE_UNITS))
    ap.add_argument("--deployment-target", default="iOS18", choices=sorted(SUPPORTED_DEPLOYMENT_TARGETS))
    ap.add_argument("--dry-run", action="store_true", help="Only write manifest/plan; do not import CoreMLTools.")
    ap.add_argument("--require-coremltools", action="store_true", help="Return failure if CoreML stack is unavailable.")
    ap.add_argument("--results", default="", help="Optional JSONL result path")
    args = ap.parse_args()
    validate_args(args)
    config = read_config(args.model)
    manifest = make_manifest(args, config)
    manifest_path = write_manifest(args.output, manifest)
    if args.dry_run or args.export_kind == "stateful-plan":
        row = {
            "axis": AXIS,
            "status": "plan",
            "model": Path(args.model).name,
            "export_kind": args.export_kind,
            "quantization": args.quantization,
            "state_mode": args.state_mode,
            "manifest": str(manifest_path),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "shape": manifest["shape"],
        }
    else:
        row = export_full_logits(args, manifest)
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results, row)
    if args.require_coremltools and row.get("status") == "skip":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
