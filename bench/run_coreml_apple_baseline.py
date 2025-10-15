#!/usr/bin/env python3
# coding=utf-8
"""CoreML runtime row generator for the Apple/Qwen3.5 baseline schema.

This is the runtime counterpart to ``scripts/export_rwkv7_coreml.py``.  It is
safe on machines without CoreMLTools: dry-run writes a plan row, and live runs
emit structured ``skip`` rows when the CoreML stack or package is unavailable.

Current live support is intentionally conservative.  It can measure a first
``full_logits`` CoreML package, but those rows are marked ``partial`` because
full-logits export does not yet provide recurrent stateful decode/prefill.  The
same JSON fields are already aligned with ``qwen35_apple_baseline`` so the later
stateful CoreML runner can flip to ``status=pass`` only after it records real
TTFT/prefill/decode/state-cache evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

BASELINE_AXIS = "qwen35_apple_baseline"
PLAN_AXIS = "rwkv7_coreml_runtime_plan"
SUPPORTED_COMPUTE_UNITS = {"all", "cpu-and-ne", "cpu-and-gpu", "cpu-only"}


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
    if data.get("format") != "rwkv7_coreml_export_manifest_v1":
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
    output_dir = Path(str(manifest.get("output_dir") or "."))
    basename = str(manifest.get("basename") or Path(str(manifest.get("source_model") or "rwkv7")).name)
    package_name = f"{basename}-full-logits"
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


def plan_row(
    *,
    manifest_path: str | Path,
    manifest: dict[str, Any],
    package: Path,
    prompt_target_chars: list[int],
    decode_lengths: list[int],
    repeat: int,
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
        "full_logits_seq_len": full_logits_seq_len(manifest),
        **stateful_function_status(manifest),
        "prompt_target_chars": prompt_target_chars,
        "decode_lengths": decode_lengths,
        "repeat": int(repeat),
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RWKV-7 CoreML runtime rows in the Apple/Qwen3.5 schema.")
    ap.add_argument("--manifest", required=True, help="coreml_export_manifest.json from scripts/export_rwkv7_coreml.py")
    ap.add_argument("--coreml-package", default="", help="Optional .mlpackage path; inferred from manifest when omitted")
    ap.add_argument("--results", default="bench/results_qwen35_apple_baseline.jsonl")
    ap.add_argument("--prompt-target-chars", default="1024", help="Comma-separated prompt text sizes for schema alignment")
    ap.add_argument("--decode-lengths", default="128", help="Comma-separated requested decode lengths for schema alignment")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--compute-units", default="cpu-and-ne", choices=sorted(SUPPORTED_COMPUTE_UNITS))
    ap.add_argument("--dry-run", action="store_true", help="Only emit a runtime plan row")
    ap.add_argument("--require-coremltools", action="store_true", help="Exit 2 if live runtime prerequisites are missing")
    args = ap.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
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
        )
        print(json.dumps(row, ensure_ascii=False))
        append_jsonl(args.results, row)
        return 0

    rows = run_full_logits_rows(
        manifest_path=manifest_path,
        manifest=manifest,
        package=package,
        prompt_target_chars=prompt_targets,
        decode_lengths=decode_lengths,
        repeat=int(args.repeat),
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
