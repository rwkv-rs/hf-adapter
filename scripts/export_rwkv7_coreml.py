#!/usr/bin/env python3
# coding=utf-8
"""RWKV-7 HF -> CoreML export entry point.

This is the Apple/mobile counterpart to the MLX baseline work.  The script is
import-safe on machines without CoreMLTools and has two layers:

1. ``--dry-run`` writes a CoreML export manifest and command/evidence plan.
2. live export supports both the compatibility ``full_logits`` package and a
   stateful ``prefill`` + ``decode`` multifunction package.  The stateful path
   stores the RWKV recurrent cache in Core ML state tensors and uses a fixed
   prefill chunk with a token mask, so arbitrary prompt lengths can be streamed
   through the same package without rebuilding a Transformer-style KV cache.

Core ML 8 only exposes fp16 state tensors.  RWKV normally accumulates WKV state
in fp32, so the stateful package keeps fp32 accumulation *inside* each function
call and writes fp16 at function boundaries.  Runtime parity/quality gates must
therefore remain mandatory before an ANE precision claim.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

AXIS = "rwkv7_coreml_export"
SUPPORTED_STATE_MODES = {"tensor", "coreml", "wkv-coreml"}
SUPPORTED_EXPORT_KINDS = {"full-logits", "stateful-multifunction", "stateful-plan"}
SUPPORTED_QUANTIZATION = {"none", "int8", "int4", "lut8", "lut6", "lut4"}
SUPPORTED_COMPUTE_UNITS = {"all", "cpu-and-ne", "cpu-and-gpu", "cpu-only"}
SUPPORTED_COMPUTE_PRECISIONS = {"auto", "fp16", "fp32"}
SUPPORTED_DEPLOYMENT_TARGETS = {"iOS18", "macOS15"}
MAX_STATEFUL_PREFILL_SEQ_LENGTH = 128
COREML_STATE_NAMES = (
    "rwkv_recurrent_state",
    "rwkv_recurrent_state_residual",
    "rwkv_attn_x_prev",
    "rwkv_ffn_x_prev",
    "rwkv_v_first",
)


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


def parse_csv(raw: str | None) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


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
    if getattr(args, "coreml_compute_precision", "fp16") not in SUPPORTED_COMPUTE_PRECISIONS:
        raise ValueError(f"unsupported CoreML compute precision {args.coreml_compute_precision!r}")
    if args.deployment_target not in SUPPORTED_DEPLOYMENT_TARGETS:
        raise ValueError(f"unsupported deployment target {args.deployment_target!r}")
    if (
        args.export_kind == "stateful-multifunction"
        and int(args.prefill_seq_length) > MAX_STATEFUL_PREFILL_SEQ_LENGTH
    ):
        raise ValueError(
            "stateful TorchScript prefill is statically unrolled; "
            f"--prefill-seq-length must be <= {MAX_STATEFUL_PREFILL_SEQ_LENGTH}. "
            "Longer prompts are streamed through repeated masked chunks."
        )


def model_shape_summary(config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(config.get("hidden_size", config.get("n_embd", 0)) or 0)
    num_layers = int(config.get("num_hidden_layers", config.get("n_layer", 0)) or 0)
    num_heads = int(config.get("num_heads", config.get("n_head", 0)) or 0)
    head_dim = int(config.get("head_dim", hidden_size // num_heads if num_heads else 0) or 0)
    attention_hidden_size = int(
        config.get("attention_hidden_size", num_heads * head_dim) or 0
    )
    if num_heads > 0 and head_dim > 0 and attention_hidden_size != num_heads * head_dim:
        raise ValueError("attention_hidden_size must equal num_heads * head_dim")
    vocab_size = int(config.get("vocab_size", 0) or 0)
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": hidden_size,
        "attention_hidden_size": attention_hidden_size,
        "num_hidden_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": config.get("max_position_embeddings"),
    }


def resolved_coreml_compute_precision(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "coreml_compute_precision", "auto"))
    if requested != "auto":
        return requested
    # Stateful recurrence is correctness-first: the fp16 MIL lane currently
    # diverges from HF greedy tokens on the live 0.1B probe. Keep full-logits
    # compatibility exports fp16, but require explicit opt-in for stateful fp16.
    return "fp32" if args.export_kind in {"stateful-multifunction", "stateful-plan"} else "fp16"


def state_layout(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Packed Core ML state layout shared by prefill and decode functions.

    Core ML state currently supports fp16 only.  Batch is fixed to one because
    independent ``MLState`` objects are the serving/session isolation unit; a
    dynamic-batch MLX path remains available for batched Apple serving.
    """

    shape = model_shape_summary(config)
    layers = int(shape["num_hidden_layers"] or 0)
    heads = int(shape["num_heads"] or 0)
    head_dim = int(shape["head_dim"] or 0)
    hidden = int(shape["hidden_size"] or 0)
    attention_hidden = int(shape["attention_hidden_size"] or 0)
    if min(layers, heads, head_dim, hidden, attention_hidden) <= 0:
        raise ValueError(f"incomplete RWKV-7 shape for CoreML state: {shape}")
    return [
        {
            "name": "rwkv_recurrent_state",
            "shape": [layers, heads, head_dim, head_dim],
            "dtype": "float16",
            "encoding": "fp16_high",
        },
        {
            "name": "rwkv_recurrent_state_residual",
            "shape": [layers, heads, head_dim, head_dim],
            "dtype": "float16",
            "encoding": "fp16_residual",
        },
        {"name": "rwkv_attn_x_prev", "shape": [layers, hidden], "dtype": "float16"},
        {"name": "rwkv_ffn_x_prev", "shape": [layers, hidden], "dtype": "float16"},
        {"name": "rwkv_v_first", "shape": [1, attention_hidden], "dtype": "float16"},
    ]


def make_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(args.model)
    output_dir = Path(args.output)
    base_name = args.basename or model_dir.name
    functions: list[dict[str, Any]] = []
    shape = model_shape_summary(config)
    if args.export_kind == "full-logits":
        functions.append(
            {
                "name": "full_logits",
                "implemented": True,
                "input": {"input_ids": [1, int(args.sample_seq_length)]},
                "output": {"logits": [1, int(args.sample_seq_length), shape.get("vocab_size")]},
            }
        )
    stateful_implemented = args.export_kind == "stateful-multifunction"
    if args.export_kind != "full-logits" or stateful_implemented:
        functions.extend(
            [
            {
                "name": "decode",
                "implemented": stateful_implemented,
                "input": {"input_ids": [1, 1], "token_mask": [1, 1]},
                "output": {"logits": [1, shape.get("vocab_size")]},
                "state_mode": args.state_mode,
            },
            {
                "name": "prefill",
                "implemented": stateful_implemented,
                "input": {
                    "input_ids": [1, int(args.prefill_seq_length)],
                    "token_mask": [1, int(args.prefill_seq_length)],
                },
                "output": {"logits": [1, shape.get("vocab_size")]},
                "state_mode": args.state_mode,
            },
            ]
        )
    elif args.export_kind == "full-logits":
        # Keep planned stateful functions visible in old compatibility exports.
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
    packed_states = state_layout(config)
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
        "quant_skip_modules": parse_csv(getattr(args, "quant_skip_modules", "")),
        "compute_units": args.compute_units,
        "coreml_compute_precision_requested": getattr(args, "coreml_compute_precision", "auto"),
        "coreml_compute_precision": resolved_coreml_compute_precision(args),
        "deployment_target": args.deployment_target,
        "minimum_os_note": "Stateful CoreML RWKV decode/prefill is planned for iOS18/macOS15+.",
        "shape": shape,
        "functions": functions,
        "state_contract": {
            "version": 1,
            "batch_size": 1,
            "boundary_dtype": "float16",
            "internal_accumulation_dtype": "float32",
            "recurrent_state_encoding": "fp16_high_plus_fp16_residual",
            "states": packed_states,
            "transfer": "MLState.read_state/write_state between multifunction handles",
        },
        "follow_up_required": [
            "ANE runtime benchmark rows in qwen35_apple_baseline schema",
            "CoreML int4/lut4 accuracy and speed gates",
            "fp16 CoreML-state parity and long-context drift gates against HF/MLX",
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


def coreml_compute_precision(ct: Any, value: str) -> Any:
    return {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[value]


def apply_torch_compression(model: Any, quantization: str, *, skip_modules: list[str] | None = None) -> Any:
    skipped = {str(name): None for name in (skip_modules or [])}
    if quantization == "none":
        return model
    if quantization in {"int8", "int4"}:
        from coremltools.optimize.torch.quantization import PostTrainingQuantizer, PostTrainingQuantizerConfig

        if quantization == "int8":
            config_dict = {"global_config": {"weight_dtype": "int8", "granularity": "per_channel"}}
        else:
            config_dict = {
                "global_config": {"weight_dtype": "int4", "granularity": "per_block", "block_size": 128}
            }
        if skipped:
            config_dict["module_name_configs"] = skipped
        config = PostTrainingQuantizerConfig.from_dict(config_dict)
        return PostTrainingQuantizer(model, config).compress()
    if quantization in {"lut8", "lut6", "lut4"}:
        from coremltools.optimize.torch.palettization import PostTrainingPalettizer, PostTrainingPalettizerConfig

        bits = int(quantization.replace("lut", ""))
        group_size = 128 if bits == 8 else 16 if bits == 6 else 32
        config_dict = {
            "global_config": {"n_bits": bits, "granularity": "per_grouped_channel", "group_size": group_size}
        }
        if skipped:
            config_dict["module_name_configs"] = skipped
        config = PostTrainingPalettizerConfig.from_dict(config_dict)
        return PostTrainingPalettizer(model, config).compress()
    raise ValueError(f"unsupported quantization {quantization!r}")


def _state_specs(ct: Any, np: Any, manifest: dict[str, Any]) -> list[Any]:
    contract = manifest.get("state_contract") or {}
    result = []
    for item in contract.get("states") or []:
        result.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(int(x) for x in item["shape"]), dtype=np.float16),
                name=str(item["name"]),
            )
        )
    if [item.name for item in result] != list(COREML_STATE_NAMES):
        raise ValueError("CoreML state contract is incomplete or out of order")
    return result


def _make_stateful_wrapper(torch: Any, model: Any, *, seq_length: int) -> Any:
    """Create a fixed-chunk RWKV wrapper whose registered buffers are ML states."""

    from rwkv7_hf.native import native_decode_step_batched

    base = model.model
    layers = len(base.layers)
    heads = int(base.layers[0].attn.num_heads)
    head_dim = int(base.layers[0].attn.head_dim)
    hidden = int(base.layers[0].attn.hidden_size)
    attention_hidden = int(
        getattr(base.layers[0].attn, "attention_hidden_size", heads * head_dim)
    )

    class StatefulRWKV7(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped
            # Core ML state tensors must be fp16 as of iOS 18 / macOS 15.
            self.register_buffer(
                "rwkv_recurrent_state",
                torch.zeros((layers, heads, head_dim, head_dim), dtype=torch.float16),
            )
            self.register_buffer(
                "rwkv_recurrent_state_residual",
                torch.zeros((layers, heads, head_dim, head_dim), dtype=torch.float16),
            )
            self.register_buffer("rwkv_attn_x_prev", torch.zeros((layers, hidden), dtype=torch.float16))
            self.register_buffer("rwkv_ffn_x_prev", torch.zeros((layers, hidden), dtype=torch.float16))
            self.register_buffer(
                "rwkv_v_first",
                torch.zeros((1, attention_hidden), dtype=torch.float16),
            )

        def forward(self, input_ids: Any, token_mask: Any) -> Any:
            # State is expanded to the native batch-one list layout.  WKV state
            # is promoted back to fp32 for the recurrence within this call.
            state = [
                self.rwkv_recurrent_state[i : i + 1].float()
                + self.rwkv_recurrent_state_residual[i : i + 1].float()
                for i in range(layers)
            ]
            xpa = [self.rwkv_attn_x_prev[i : i + 1] for i in range(layers)]
            xpf = [self.rwkv_ffn_x_prev[i : i + 1] for i in range(layers)]
            v_first = self.rwkv_v_first
            logits = torch.zeros(
                (1, int(self.wrapped.config.vocab_size)),
                dtype=self.wrapped.lm_head.weight.dtype,
                device=input_ids.device,
            )
            # ``seq_length`` is deliberately static so TorchScript/Core ML can
            # unroll one production chunk.  token_mask skips padded tail slots.
            for token_index in range(int(seq_length)):
                # native_decode_step_batched updates the Python lists in place;
                # copy the list containers so masked padding can select the
                # true pre-token values rather than aliases to the new values.
                old_state = list(state)
                old_xpa = list(xpa)
                old_xpf = list(xpf)
                old_v_first = v_first
                new_logits, new_state, new_xpa, new_xpf, new_v_first = native_decode_step_batched(
                    self.wrapped,
                    input_ids[:, token_index],
                    state,
                    xpa,
                    xpf,
                    v_first,
                )
                active = token_mask[:, token_index] != 0
                state_mask = active.reshape(1, 1, 1, 1)
                residual_mask = active.reshape(1, 1)
                attention_mask = active.reshape(1, 1)
                state = [
                    torch.where(state_mask, new_value, old_value)
                    for old_value, new_value in zip(old_state, new_state)
                ]
                xpa = [
                    torch.where(residual_mask, new_value, old_value)
                    for old_value, new_value in zip(old_xpa, new_xpa)
                ]
                xpf = [
                    torch.where(residual_mask, new_value, old_value)
                    for old_value, new_value in zip(old_xpf, new_xpf)
                ]
                v_first = torch.where(attention_mask, new_v_first, old_v_first)
                logits = torch.where(residual_mask, new_logits, logits)

                # Decode crosses a Core ML state boundary after every token.
                # Apply the same boundary encoding between tokens inside the
                # unrolled prefill function; otherwise a 2-token call keeps
                # higher precision after token 1 and diverges from two 1-token
                # calls even though both use the same recurrence.
                rounded_state = []
                for value in state:
                    value_fp32 = value.float()
                    value_high = value_fp32.to(torch.float16)
                    value_residual = (value_fp32 - value_high.float()).to(torch.float16)
                    rounded_state.append(value_high.float() + value_residual.float())
                state = rounded_state
                xpa = [value.to(torch.float16) for value in xpa]
                xpf = [value.to(torch.float16) for value in xpf]
                v_first = v_first.to(torch.float16)

            # Slice assignment is intentional: CoreMLTools recognizes
            # ``slice -> copy_`` as a state update, while a whole-buffer
            # ``copy_`` has no matching tensor-assignment slice in Torch IR.
            packed_state = torch.cat(state, dim=0).float()
            packed_state_high = packed_state.to(torch.float16)
            packed_state_residual = (packed_state - packed_state_high.float()).to(torch.float16)
            self.rwkv_recurrent_state[:] = packed_state_high
            self.rwkv_recurrent_state_residual[:] = packed_state_residual
            self.rwkv_attn_x_prev[:] = torch.cat(xpa, dim=0).to(torch.float16)
            self.rwkv_ffn_x_prev[:] = torch.cat(xpf, dim=0).to(torch.float16)
            self.rwkv_v_first[:] = v_first.to(torch.float16)
            return logits

    return StatefulRWKV7(model).eval()


def _convert_stateful_function(
    *,
    torch: Any,
    np: Any,
    ct: Any,
    model: Any,
    manifest: dict[str, Any],
    seq_length: int,
    args: argparse.Namespace,
) -> Any:
    wrapper = _make_stateful_wrapper(torch, model, seq_length=int(seq_length))
    sample_ids = torch.zeros((1, int(seq_length)), dtype=torch.int64)
    sample_mask = torch.ones((1, int(seq_length)), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample_ids, sample_mask), check_trace=False)
        # Tracing executes the stateful forward once. Restore zero initial
        # state before conversion so MLModel.make_state() never inherits the
        # synthetic trace sample.
        for state_name in COREML_STATE_NAMES:
            getattr(wrapper, state_name).zero_()
            getattr(traced, state_name).zero_()
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=sample_ids.shape, dtype=np.int32),
            ct.TensorType(name="token_mask", shape=sample_mask.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=_state_specs(ct, np, manifest),
        minimum_deployment_target=coreml_target(ct, args.deployment_target),
        compute_units=coreml_compute_unit(ct, args.compute_units),
        compute_precision=coreml_compute_precision(ct, resolved_coreml_compute_precision(args)),
    )


def export_stateful_multifunction(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    stack = import_coreml_stack(args.require_coremltools)
    if stack is None:
        return {
            "axis": AXIS,
            "status": "skip",
            "reason": "coremltools/torch/transformers stack not installed",
            "model": Path(args.model).name,
            "export_kind": args.export_kind,
            "manifest": str(Path(args.output) / "coreml_export_manifest.json"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
    np, torch, ct, AutoModelForCausalLM = stack
    os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        # Trace in fp32 so LayerNorm gamma and epsilon have the same source
        # dtype. CoreMLTools lowers the ML Program to fp16 via
        # compute_precision below; tracing an fp16 LayerNorm currently leaves
        # epsilon fp32 and is rejected by the MIL layer_norm op.
        dtype=torch.float32,
    ).eval()
    model = apply_torch_compression(
        model,
        args.quantization,
        skip_modules=parse_csv(getattr(args, "quant_skip_modules", "")),
    )
    output_dir = Path(args.output)
    intermediate = output_dir / "functions"
    intermediate.mkdir(parents=True, exist_ok=True)
    prefill_path = intermediate / "prefill.mlpackage"
    decode_path = intermediate / "decode.mlpackage"

    prefill_model = _convert_stateful_function(
        torch=torch,
        np=np,
        ct=ct,
        model=model,
        manifest=manifest,
        seq_length=int(args.prefill_seq_length),
        args=args,
    )
    prefill_model.save(str(prefill_path))
    decode_model = _convert_stateful_function(
        torch=torch,
        np=np,
        ct=ct,
        model=model,
        manifest=manifest,
        seq_length=1,
        args=args,
    )
    decode_model.save(str(decode_path))

    descriptor = ct.utils.MultiFunctionDescriptor()
    descriptor.add_function(str(prefill_path), src_function_name="main", target_function_name="prefill")
    descriptor.add_function(str(decode_path), src_function_name="main", target_function_name="decode")
    descriptor.default_function_name = "decode"
    package_name = f"{manifest['basename']}-stateful"
    if args.quantization != "none":
        package_name += f"-{args.quantization}"
    output_path = output_dir / f"{package_name}.mlpackage"
    ct.utils.save_multifunction(descriptor, str(output_path))
    if not bool(getattr(args, "keep_intermediate_functions", False)):
        shutil.rmtree(intermediate, ignore_errors=True)
    elapsed = time.perf_counter() - t0
    manifest["coreml_package"] = str(output_path)
    manifest["multifunction"] = {"functions": ["prefill", "decode"], "default_function": "decode"}
    write_manifest(args.output, manifest)
    return {
        "axis": AXIS,
        "status": "pass",
        "model": Path(args.model).name,
        "export_kind": args.export_kind,
        "quantization": args.quantization,
        "quant_skip_modules": parse_csv(getattr(args, "quant_skip_modules", "")),
        "coreml_compute_precision": resolved_coreml_compute_precision(args),
        "state_mode": args.state_mode,
        "coreml_package": str(output_path),
        "functions": ["prefill", "decode"],
        "state_names": list(COREML_STATE_NAMES),
        "prefill_seq_length": int(args.prefill_seq_length),
        "kept_intermediate_functions": bool(getattr(args, "keep_intermediate_functions", False)),
        "manifest": str(Path(args.output) / "coreml_export_manifest.json"),
        "elapsed_s": round(float(elapsed), 6),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


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
    model = apply_torch_compression(
        model,
        args.quantization,
        skip_modules=parse_csv(getattr(args, "quant_skip_modules", "")),
    )
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
        compute_precision=coreml_compute_precision(ct, resolved_coreml_compute_precision(args)),
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
        "quant_skip_modules": parse_csv(getattr(args, "quant_skip_modules", "")),
        "coreml_compute_precision": resolved_coreml_compute_precision(args),
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
    ap.add_argument(
        "--quant-skip-modules",
        default="",
        help="Comma-separated exact torch module names to keep uncompressed (for example lm_head).",
    )
    ap.add_argument("--compute-units", default="cpu-and-ne", choices=sorted(SUPPORTED_COMPUTE_UNITS))
    ap.add_argument(
        "--coreml-compute-precision",
        default="auto",
        choices=sorted(SUPPORTED_COMPUTE_PRECISIONS),
        help="MIL compute precision; auto uses fp32 for stateful recurrence and fp16 for full-logits.",
    )
    ap.add_argument("--deployment-target", default="iOS18", choices=sorted(SUPPORTED_DEPLOYMENT_TARGETS))
    ap.add_argument("--keep-intermediate-functions", action="store_true", help="Keep prefill/decode source packages after multifunction deduplication")
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
    elif args.export_kind == "stateful-multifunction":
        row = export_stateful_multifunction(args, manifest)
    else:
        row = export_full_logits(args, manifest)
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results, row)
    if args.require_coremltools and row.get("status") == "skip":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
