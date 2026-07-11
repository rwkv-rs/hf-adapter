#!/usr/bin/env python3
# coding=utf-8
"""CoreML runtime row generator for the Apple/Qwen3.5 baseline schema.

This is the runtime counterpart to ``scripts/export_rwkv7_coreml.py``.  It is
safe on machines without CoreMLTools: dry-run writes a plan row, and live runs
emit structured ``skip`` rows when the CoreML stack or package is unavailable.

Live support includes the compatibility ``full_logits`` package (reported as a
partial row) and the stateful ``prefill`` + ``decode`` multifunction package.
The latter uses exact shared prompt text, transfers packed RWKV Core ML state
between function handles, performs greedy decode, and records TTFT, throughput,
state-transfer correctness, package footprint, and optional chunk-boundary
drift evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, Iterable

BASELINE_AXIS = "qwen35_apple_baseline"
PLAN_AXIS = "rwkv7_coreml_runtime_plan"
SUPPORTED_COMPUTE_UNITS = {"all", "cpu-and-ne", "cpu-and-gpu", "cpu-only"}
DEFAULT_PROMPT_SEED = (
    "User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. "
    "Report throughput, latency, memory, state-cache behavior, and quantization stability.\n"
    "Assistant: "
)


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv(raw: str) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    values = [int(item) for item in parse_csv(raw)]
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"all integer values must be positive: {values}")
    return values


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path}: expected object manifest")
    if data.get("format") not in {"rwkv7_coreml_export_manifest_v1", "rwkv7_coreml_export_manifest_v2"}:
        raise ValueError(f"{manifest_path}: unsupported manifest format {data.get('format')!r}")
    return data


def manifest_function(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in manifest.get("functions") or []:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def full_logits_seq_len(manifest: dict[str, Any]) -> int | None:
    fn = manifest_function(manifest, "full_logits")
    if not fn:
        return None
    input_ids = ((fn.get("input") or {}).get("input_ids") or []) if isinstance(fn.get("input"), dict) else []
    if len(input_ids) != 2:
        return None
    return safe_int(input_ids[1])


def stateful_function_status(manifest: dict[str, Any]) -> dict[str, Any]:
    decode = manifest_function(manifest, "decode") or {}
    prefill = manifest_function(manifest, "prefill") or {}
    contract = manifest.get("state_contract") if isinstance(manifest.get("state_contract"), dict) else {}
    return {
        "stateful_contract_present": bool(contract),
        "state_contract_version": contract.get("version"),
        "state_mode": manifest.get("state_mode"),
        "decode_implemented": bool(decode.get("implemented")),
        "prefill_implemented": bool(prefill.get("implemented")),
        "state_tensors_per_layer": sorted((contract.get("state_tensors_per_layer") or {}).keys()),
        "global_state_tensors": sorted((contract.get("global_state_tensors") or {}).keys()),
    }


def infer_coreml_package(manifest: dict[str, Any]) -> Path:
    explicit = manifest.get("coreml_package")
    if explicit:
        return Path(str(explicit))
    output_dir = Path(str(manifest.get("output_dir") or "."))
    basename = str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name)
    suffix = "stateful" if manifest.get("export_kind") == "stateful-multifunction" else "full-logits"
    package_name = f"{basename}-{suffix}"
    quantization = str(manifest.get("quantization") or "none")
    if quantization != "none":
        package_name += f"-{quantization}"
    return output_dir / f"{package_name}.mlpackage"


def import_coreml_runtime(require: bool) -> tuple[Any, Any] | None:
    try:
        import numpy as np
        import coremltools as ct

        return np, ct
    except Exception:
        return None


def coreml_compute_unit(ct: Any, value: str) -> Any:
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu-and-ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-only": ct.ComputeUnit.CPU_ONLY,
    }[value]


def make_input_ids(np: Any, seq_len: int, vocab_size: int | None) -> Any:
    vocab = int(vocab_size or 65536)
    # Keep token ids deterministic and non-constant for runtime smoke while
    # avoiding tokenizer dependencies.  These rows stay partial until stateful
    # CoreML can consume the exact text prompt used by Qwen/RWKV MLX rows.
    values = np.arange(int(seq_len), dtype=np.int32) % max(vocab, 1)
    return values.reshape(1, int(seq_len))


def output_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def make_prompt(seed: str, target_chars: int) -> str:
    target = int(target_chars)
    if target <= 0 or not seed:
        raise ValueError("prompt target and seed must be non-empty")
    return (seed * ((target + len(seed) - 1) // len(seed)))[:target]


def package_nbytes(path: Path) -> int | None:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())
    except OSError:
        return None


def process_peak_memory_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # ru_maxrss is bytes on Darwin and KiB on Linux.
    return value if platform.system() == "Darwin" else value * 1024


def state_names(manifest: dict[str, Any]) -> list[str]:
    contract = manifest.get("state_contract") if isinstance(manifest.get("state_contract"), dict) else {}
    names = [str(item.get("name")) for item in contract.get("states") or [] if isinstance(item, dict)]
    return [name for name in names if name]


def plan_row(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    package: Path,
    prompt_target_chars: list[int],
    decode_lengths: list[int],
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    return {
        "axis": PLAN_AXIS,
        "status": "plan",
        "manifest": str(manifest_path),
        "coreml_package": str(package),
        "model": str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name),
        "export_kind": manifest.get("export_kind"),
        "state_mode": manifest.get("state_mode"),
        "quantization": manifest.get("quantization"),
        "coreml_compute_precision": manifest.get("coreml_compute_precision"),
        "full_logits_seq_len": full_logits_seq_len(manifest),
        **stateful_function_status(manifest),
        "prompt_target_chars": prompt_target_chars,
        "decode_lengths": decode_lengths,
        "repeat": int(repeat),
        "warmup": int(warmup),
        "will_emit_baseline_axis": True,
        "pass_status_requires_stateful_decode": True,
    }


def skip_row(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    package: Path,
    reason: str,
    prompt_case: str,
    prompt_target_chars: int,
    requested_generated_tokens: int,
) -> dict[str, Any]:
    return {
        "axis": BASELINE_AXIS,
        "status": "skip",
        "engine": "rwkv7_hf",
        "runtime": "coreml",
        "model": str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name),
        "family": "rwkv7",
        "manifest": str(manifest_path),
        "coreml_package": str(package),
        "export_kind": manifest.get("export_kind"),
        "state_mode": manifest.get("state_mode"),
        "quantization": manifest.get("quantization"),
        "prompt_case": prompt_case,
        "prompt_target_chars": int(prompt_target_chars),
        "requested_generated_tokens": int(requested_generated_tokens),
        "reason": reason,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_full_logits_rows(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    package: Path,
    prompt_target_chars: list[int],
    decode_lengths: list[int],
    repeat: int,
    warmup: int,
    compute_units: str,
    require_coremltools: bool,
) -> list[dict[str, Any]]:
    seq_len = full_logits_seq_len(manifest)
    rows: list[dict[str, Any]] = []
    shapes = [(f"chars{chars}", int(chars), int(tokens)) for chars in prompt_target_chars for tokens in decode_lengths]
    if seq_len is None:
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason="manifest has no implemented full_logits input shape",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]
    if not package.exists():
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason="CoreML package does not exist; run scripts/export_rwkv7_coreml.py without --dry-run first",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]
    stack = import_coreml_runtime(require_coremltools)
    if stack is None:
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason="coremltools/numpy runtime stack not installed",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]
    np, ct = stack
    mlmodel = ct.models.MLModel(str(package), compute_units=coreml_compute_unit(ct, compute_units))
    vocab_size = safe_int((manifest.get("shape") or {}).get("vocab_size")) if isinstance(manifest.get("shape"), dict) else None
    input_ids = make_input_ids(np, int(seq_len), vocab_size)
    t_warmup = time.perf_counter()
    for _ in range(int(warmup)):
        mlmodel.predict({"input_ids": input_ids})
    warmup_s = time.perf_counter() - t_warmup
    for prompt_case, chars, tokens in shapes:
        for repeat_index in range(1, int(repeat) + 1):
            t0 = time.perf_counter()
            out = mlmodel.predict({"input_ids": input_ids})
            elapsed = time.perf_counter() - t0
            logits = out.get("logits") if isinstance(out, dict) else None
            rows.append(
                {
                    "axis": BASELINE_AXIS,
                    "status": "partial",
                    "partial_reason": "full_logits CoreML package lacks stateful recurrent decode/prefill; not eligible for Qwen3.5 pass/fail comparison",
                    "engine": "rwkv7_hf",
                    "runtime": "coreml",
                    "model": str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name),
                    "family": "rwkv7",
                    "manifest": str(manifest_path),
                    "coreml_package": str(package),
                    "export_kind": manifest.get("export_kind"),
                    "state_mode": manifest.get("state_mode"),
                    "quantization": manifest.get("quantization"),
                    "compute_units": compute_units,
                    **stateful_function_status(manifest),
                    "prompt_case": prompt_case,
                    "prompt_target_chars": int(chars),
                    "prompt_chars": int(chars),
                    "prompt_source": "synthetic_input_ids",
                    "prompt_eval_tokens": int(seq_len),
                    "generated_tokens": 0,
                    "requested_generated_tokens": int(tokens),
                    "repeat_index": int(repeat_index),
                    "repeat": int(repeat),
                    "warmup": int(warmup),
                    "warmup_s": round(float(warmup_s), 6),
                    "wall_s": round(float(elapsed), 6),
                    "prefill_s": round(float(elapsed), 6),
                    "ttft_s": round(float(elapsed), 6),
                    "prefill_tok_s": round(float(seq_len / elapsed), 6) if elapsed > 0 else None,
                    "decode_tok_s": None,
                    "logits_shape": output_shape(logits),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                }
            )
    return rows


def _state_values(state: Any, names: list[str]) -> dict[str, Any]:
    if not hasattr(state, "read_state"):
        raise RuntimeError("installed coremltools MLState lacks read_state; upgrade coremltools")
    return {name: state.read_state(name) for name in names}


def _write_state_values(state: Any, values: dict[str, Any]) -> None:
    if not hasattr(state, "write_state"):
        raise RuntimeError("installed coremltools MLState lacks write_state; upgrade coremltools")
    for name, value in values.items():
        state.write_state(name, value)


def _max_abs(np: Any, left: Any, right: Any) -> float:
    return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _run_stateful_prefill(
    *,
    np: Any,
    model: Any,
    state: Any,
    prompt_ids: list[int],
    model_chunk_size: int,
    active_chunk_size: int,
) -> tuple[Any, int]:
    """Run exact prompt ids through a fixed CoreML chunk with masked tail."""

    if not prompt_ids:
        raise ValueError("stateful CoreML prefill requires at least one token")
    active = max(1, min(int(active_chunk_size), int(model_chunk_size)))
    logits = None
    calls = 0
    for start in range(0, len(prompt_ids), active):
        part = prompt_ids[start : start + active]
        input_ids = np.zeros((1, int(model_chunk_size)), dtype=np.int32)
        token_mask = np.zeros((1, int(model_chunk_size)), dtype=np.int32)
        input_ids[0, : len(part)] = np.asarray(part, dtype=np.int32)
        token_mask[0, : len(part)] = 1
        out = model.predict({"input_ids": input_ids, "token_mask": token_mask}, state=state)
        logits = out.get("logits") if isinstance(out, dict) else None
        calls += 1
    if logits is None:
        raise RuntimeError("stateful CoreML prefill produced no logits")
    return logits, calls


def run_stateful_rows(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    package: Path,
    prompt_target_chars: list[int],
    decode_lengths: list[int],
    repeat: int,
    warmup: int,
    compute_units: str,
    require_coremltools: bool,
    prompt_seed: str,
    store_responses: bool,
    verify_chunked_prefill: bool,
    verify_chunk_size: int,
    require_chunked_prefill_match: bool,
    state_atol: float,
    verify_hf_parity: bool,
    hf_parity_dtype: str,
    require_hf_greedy_match: bool,
) -> list[dict[str, Any]]:
    shapes = [(f"chars{chars}", int(chars), int(tokens)) for chars in prompt_target_chars for tokens in decode_lengths]
    if not package.exists():
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason="stateful CoreML package does not exist; export stateful-multifunction first",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]
    stack = import_coreml_runtime(require_coremltools)
    if stack is None:
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason="coremltools/numpy runtime stack not installed",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]
    np, ct = stack
    try:
        from transformers import AutoTokenizer

        source_model = str(manifest.get("source_model") or "")
        tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)
    except Exception as exc:
        return [
            skip_row(
                manifest_path=manifest_path,
                manifest=manifest,
                package=package,
                reason=f"RWKV tokenizer unavailable for exact shared prompt: {type(exc).__name__}: {exc}",
                prompt_case=prompt_case,
                prompt_target_chars=chars,
                requested_generated_tokens=tokens,
            )
            for prompt_case, chars, tokens in shapes
        ]

    hf_model = None
    torch = None
    if verify_hf_parity:
        try:
            import torch as _torch
            from transformers import AutoModelForCausalLM

            os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
            torch = _torch
            hf_dtype = _torch.float32 if hf_parity_dtype == "fp32" else _torch.float16
            hf_model = AutoModelForCausalLM.from_pretrained(
                source_model,
                trust_remote_code=True,
                dtype=hf_dtype,
            ).eval()
        except Exception as exc:
            return [
                skip_row(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    package=package,
                    reason=f"HF parity model unavailable: {type(exc).__name__}: {exc}",
                    prompt_case=prompt_case,
                    prompt_target_chars=chars,
                    requested_generated_tokens=tokens,
                )
                for prompt_case, chars, tokens in shapes
            ]

    compute_unit = coreml_compute_unit(ct, compute_units)
    prefill_model = ct.models.MLModel(str(package), compute_units=compute_unit, function_name="prefill")
    decode_model = ct.models.MLModel(str(package), compute_units=compute_unit, function_name="decode")
    prefill_fn = manifest_function(manifest, "prefill") or {}
    input_spec = prefill_fn.get("input") if isinstance(prefill_fn.get("input"), dict) else {}
    input_shape = input_spec.get("input_ids") or [1, manifest.get("prefill_seq_length", 1)]
    model_chunk_size = int(input_shape[1])
    names = state_names(manifest)
    if not names:
        raise ValueError("stateful CoreML manifest has no state_contract states")
    pkg_bytes = package_nbytes(package)
    warmup_s = 0.0
    if int(warmup) > 0:
        t_warmup = time.perf_counter()
        for _ in range(int(warmup)):
            warm_prefill_state = prefill_model.make_state()
            warm_out = prefill_model.predict(
                {
                    "input_ids": np.zeros((1, model_chunk_size), dtype=np.int32),
                    "token_mask": np.ones((1, model_chunk_size), dtype=np.int32),
                },
                state=warm_prefill_state,
            )
            warm_decode_state = decode_model.make_state()
            _write_state_values(warm_decode_state, _state_values(warm_prefill_state, names))
            warm_token = int(np.argmax(warm_out["logits"].reshape(-1)))
            decode_model.predict(
                {
                    "input_ids": np.asarray([[warm_token]], dtype=np.int32),
                    "token_mask": np.asarray([[1]], dtype=np.int32),
                },
                state=warm_decode_state,
            )
        warmup_s = time.perf_counter() - t_warmup
    rows: list[dict[str, Any]] = []

    for prompt_case, chars, requested_tokens in shapes:
        prompt = make_prompt(prompt_seed, chars)
        prompt_ids = [int(x) for x in tokenizer(prompt, add_special_tokens=False).input_ids]
        if not prompt_ids:
            raise ValueError("RWKV tokenizer produced zero prompt tokens")
        for repeat_index in range(1, int(repeat) + 1):
            prefill_state = prefill_model.make_state()
            t_prefill = time.perf_counter()
            logits, prefill_calls = _run_stateful_prefill(
                np=np,
                model=prefill_model,
                state=prefill_state,
                prompt_ids=prompt_ids,
                model_chunk_size=model_chunk_size,
                active_chunk_size=model_chunk_size,
            )
            prefill_s = time.perf_counter() - t_prefill

            t_transfer = time.perf_counter()
            values = _state_values(prefill_state, names)
            decode_state = decode_model.make_state()
            _write_state_values(decode_state, values)
            copied = _state_values(decode_state, names)
            transfer_s = time.perf_counter() - t_transfer
            transfer_max_abs = max(_max_abs(np, values[name], copied[name]) for name in names)
            transfer_bytes = sum(int(value.nbytes) for value in values.values())

            chunk_logits_diff = None
            chunk_state_diff = None
            chunk_token_match = None
            verification_calls = None
            if verify_chunked_prefill:
                verification_state = prefill_model.make_state()
                verification_logits, verification_calls = _run_stateful_prefill(
                    np=np,
                    model=prefill_model,
                    state=verification_state,
                    prompt_ids=prompt_ids,
                    model_chunk_size=model_chunk_size,
                    active_chunk_size=int(verify_chunk_size or max(1, model_chunk_size // 2)),
                )
                verification_values = _state_values(verification_state, names)
                chunk_logits_diff = _max_abs(np, logits, verification_logits)
                chunk_state_diff = max(_max_abs(np, values[name], verification_values[name]) for name in names)
                chunk_token_match = bool(int(np.argmax(logits)) == int(np.argmax(verification_logits)))

            t_first = time.perf_counter()
            next_token = int(np.argmax(logits.reshape(-1)))
            first_s = time.perf_counter() - t_first
            generated: list[int] = []
            t_decode = time.perf_counter()
            for _ in range(int(requested_tokens)):
                generated.append(int(next_token))
                out = decode_model.predict(
                    {
                        "input_ids": np.asarray([[next_token]], dtype=np.int32),
                        "token_mask": np.asarray([[1]], dtype=np.int32),
                    },
                    state=decode_state,
                )
                logits = out["logits"]
                next_token = int(np.argmax(logits.reshape(-1)))
            decode_s = (time.perf_counter() - t_decode) + first_s
            response_text = tokenizer.decode(generated, skip_special_tokens=True) if store_responses else ""
            hf_generated = None
            hf_greedy_match = None
            hf_first_mismatch = None
            if hf_model is not None and torch is not None:
                with torch.no_grad():
                    hf_out = hf_model(torch.tensor([prompt_ids], dtype=torch.long), use_cache=True)
                    hf_cache = hf_out.past_key_values
                    hf_logits = hf_out.logits[:, -1, :]
                    hf_generated = []
                    for _ in range(int(requested_tokens)):
                        hf_token = int(torch.argmax(hf_logits, dim=-1).item())
                        hf_generated.append(hf_token)
                        hf_out = hf_model(
                            torch.tensor([[hf_token]], dtype=torch.long),
                            past_key_values=hf_cache,
                            use_cache=True,
                        )
                        hf_cache = hf_out.past_key_values
                        hf_logits = hf_out.logits[:, -1, :]
                hf_greedy_match = bool(generated == hf_generated)
                if not hf_greedy_match:
                    hf_first_mismatch = next(
                        (
                            index
                            for index, (coreml_token, hf_token) in enumerate(zip(generated, hf_generated))
                            if int(coreml_token) != int(hf_token)
                        ),
                        min(len(generated), len(hf_generated)),
                    )
            chunk_match = bool(
                not verify_chunked_prefill
                or (
                    chunk_token_match
                    and float(chunk_logits_diff or 0.0) <= float(state_atol)
                    and float(chunk_state_diff or 0.0) <= float(state_atol)
                )
            )
            correctness_pass = (
                transfer_max_abs <= float(state_atol)
                and (not require_hf_greedy_match or hf_greedy_match is True)
                and (not require_chunked_prefill_match or chunk_match)
            )
            row = {
                "axis": BASELINE_AXIS,
                "status": "pass" if correctness_pass else "fail",
                "engine": "rwkv7_hf",
                "runtime": "coreml_stateful",
                "model": str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name),
                "model_path": str(manifest.get("source_model") or ""),
                "family": "rwkv7",
                "manifest": str(manifest_path),
                "coreml_package": str(package),
                "export_kind": manifest.get("export_kind"),
                "state_mode": manifest.get("state_mode"),
                "state_boundary_dtype": ((manifest.get("state_contract") or {}).get("boundary_dtype")),
                "quantization": manifest.get("quantization"),
                "quant_skip_modules": manifest.get("quant_skip_modules"),
                "coreml_compute_precision": manifest.get("coreml_compute_precision"),
                "compute_units": compute_units,
                "prompt_case": prompt_case,
                "prompt_target_chars": int(chars),
                "prompt_chars": len(prompt),
                "prompt_source": "qwen35_shared_prompt_seed",
                "prompt_eval_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
                "requested_generated_tokens": int(requested_tokens),
                "repeat_index": int(repeat_index),
                "repeat": int(repeat),
                "warmup": int(warmup),
                "warmup_s": round(float(warmup_s), 6),
                "prefill_chunk_size": int(model_chunk_size),
                "prefill_calls": int(prefill_calls),
                "prefill_s": round(float(prefill_s), 6),
                "first_token_s": round(float(first_s), 6),
                "ttft_s": round(float(prefill_s + first_s), 6),
                "decode_s": round(float(decode_s), 6),
                "prefill_tok_s": round(float(len(prompt_ids) / prefill_s), 6) if prefill_s > 0 else None,
                "decode_tok_s": round(float(len(generated) / decode_s), 6) if decode_s > 0 else None,
                "seen_tokens_after_generate": int(len(prompt_ids) + len(generated)),
                "expected_seen_tokens": int(len(prompt_ids) + requested_tokens),
                "generated_preview": generated[:16],
                "state_names": names,
                "state_transfer_s": round(float(transfer_s), 6),
                "state_transfer_bytes": int(transfer_bytes),
                "state_transfer_max_abs": round(float(transfer_max_abs), 8),
                "state_atol": float(state_atol),
                "chunked_prefill_verified": bool(verify_chunked_prefill),
                "chunked_prefill_required": bool(require_chunked_prefill_match),
                "chunked_prefill_status": "pass" if chunk_match else "fail",
                "chunked_prefill_calls": int(verification_calls) if verification_calls is not None else None,
                "chunked_prefill_logits_max_abs": round(float(chunk_logits_diff), 8) if chunk_logits_diff is not None else None,
                "chunked_prefill_state_max_abs": round(float(chunk_state_diff), 8) if chunk_state_diff is not None else None,
                "chunked_prefill_greedy_match": chunk_token_match,
                "hf_parity_verified": bool(verify_hf_parity),
                "hf_parity_required": bool(require_hf_greedy_match),
                "hf_parity_dtype": hf_parity_dtype if verify_hf_parity else None,
                "hf_greedy_match": hf_greedy_match,
                "hf_first_mismatch_index": hf_first_mismatch,
                "hf_generated_preview": hf_generated[:16] if hf_generated is not None else None,
                "coreml_package_bytes": pkg_bytes,
                "peak_memory_bytes": process_peak_memory_bytes(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            }
            if store_responses:
                row["generated_token_ids"] = generated
                row["response_text"] = response_text
                row["response_preview"] = response_text[:160]
                row["response_chars"] = len(response_text)
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RWKV-7 CoreML runtime rows in the Apple/Qwen3.5 schema.")
    ap.add_argument("--manifest", required=True, help="coreml_export_manifest.json from scripts/export_rwkv7_coreml.py")
    ap.add_argument("--coreml-package", default="", help="Optional .mlpackage path; inferred from manifest when omitted")
    ap.add_argument("--results", default="bench/results_qwen35_apple_baseline.jsonl")
    ap.add_argument("--prompt-target-chars", default="1024", help="Comma-separated prompt text sizes for schema alignment")
    ap.add_argument("--prompt-seed", default=DEFAULT_PROMPT_SEED, help="Exact shared prompt seed used by the Qwen3.5/MLX runner")
    ap.add_argument("--decode-lengths", default="128", help="Comma-separated requested decode lengths for schema alignment")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1, help="Untimed CoreML prefill/decode calls before measured rows")
    ap.add_argument("--compute-units", default="cpu-and-ne", choices=sorted(SUPPORTED_COMPUTE_UNITS))
    ap.add_argument("--store-responses", action="store_true", help="Store generated ids/text for quality scoring")
    ap.add_argument("--verify-chunked-prefill", action="store_true", help="Repeat prefill with smaller active chunks and compare state/logits")
    ap.add_argument("--verify-chunk-size", type=int, default=0, help="Active-token subchunk for the optional chunk-boundary check")
    ap.add_argument("--require-chunked-prefill-match", action="store_true", help="Fail the baseline row when the optional chunk-boundary check drifts")
    ap.add_argument("--state-atol", type=float, default=0.125, help="Maximum state/logit drift allowed by state correctness gates")
    ap.add_argument("--verify-hf-parity", action="store_true", help="Compare CoreML greedy tokens with the source HF native model")
    ap.add_argument("--require-hf-greedy-match", action="store_true", help="Fail the row when optional HF greedy parity mismatches")
    ap.add_argument("--hf-parity-dtype", default="fp32", choices=["fp16", "fp32"])
    ap.add_argument("--dry-run", action="store_true", help="Only emit a runtime plan row")
    ap.add_argument("--require-coremltools", action="store_true", help="Exit 2 if live runtime prerequisites are missing")
    args = ap.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.verify_chunk_size < 0:
        raise ValueError("--verify-chunk-size must be non-negative")
    if args.state_atol < 0:
        raise ValueError("--state-atol must be non-negative")
    if args.require_chunked_prefill_match and not args.verify_chunked_prefill:
        raise ValueError("--require-chunked-prefill-match requires --verify-chunked-prefill")
    if args.require_hf_greedy_match and not args.verify_hf_parity:
        raise ValueError("--require-hf-greedy-match requires --verify-hf-parity")
    prompt_targets = parse_int_csv(args.prompt_target_chars)
    decode_lengths = parse_int_csv(args.decode_lengths)
    manifest_path = Path(args.manifest)
    manifest = read_manifest(manifest_path)
    package = Path(args.coreml_package) if args.coreml_package else infer_coreml_package(manifest)

    if args.dry_run:
        row = plan_row(
            manifest_path=manifest_path,
            manifest=manifest,
            package=package,
            prompt_target_chars=prompt_targets,
            decode_lengths=decode_lengths,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
        )
        print(json.dumps(row, ensure_ascii=False))
        append_jsonl(args.results, row)
        return 0

    stateful = bool(
        manifest.get("export_kind") == "stateful-multifunction"
        and (manifest_function(manifest, "prefill") or {}).get("implemented")
        and (manifest_function(manifest, "decode") or {}).get("implemented")
    )
    if stateful:
        rows = run_stateful_rows(
            manifest_path=manifest_path,
            manifest=manifest,
            package=package,
            prompt_target_chars=prompt_targets,
            decode_lengths=decode_lengths,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            compute_units=args.compute_units,
            require_coremltools=bool(args.require_coremltools),
            prompt_seed=args.prompt_seed,
            store_responses=bool(args.store_responses),
            verify_chunked_prefill=bool(args.verify_chunked_prefill),
            verify_chunk_size=int(args.verify_chunk_size),
            require_chunked_prefill_match=bool(args.require_chunked_prefill_match),
            state_atol=float(args.state_atol),
            verify_hf_parity=bool(args.verify_hf_parity),
            hf_parity_dtype=args.hf_parity_dtype,
            require_hf_greedy_match=bool(args.require_hf_greedy_match),
        )
    else:
        rows = run_full_logits_rows(
            manifest_path=manifest_path,
            manifest=manifest,
            package=package,
            prompt_target_chars=prompt_targets,
            decode_lengths=decode_lengths,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            compute_units=args.compute_units,
            require_coremltools=bool(args.require_coremltools),
        )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
        append_jsonl(args.results, row)
    if args.require_coremltools and any(row.get("status") == "skip" and "coremltools" in str(row.get("reason")) for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
