#!/usr/bin/env python3
"""Verify conservative MLX defaults on real converted checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AXIS = "mlx_safe_defaults"


def append(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "platform": platform.platform(),
    }


def isolated_parent(args: argparse.Namespace, models: list[str]) -> int:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rwkv7-mlx-safe-") as temporary:
        for index, model in enumerate(models):
            child = Path(temporary) / f"{index}.jsonl"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--models",
                model,
                "--dtype",
                args.dtype,
                "--prompt",
                args.prompt,
                "--results",
                str(child),
                "--isolated-child",
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            child_rows = []
            if child.exists():
                child_rows = [json.loads(line) for line in child.read_text().splitlines() if line]
            matches = [row for row in child_rows if row.get("axis") == AXIS]
            if completed.returncode or len(matches) != 1:
                row = {
                    "axis": AXIS,
                    "status": "fail",
                    "model": Path(model).name,
                    "reason": "isolated safe-default child failed",
                    "child_returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                }
            else:
                row = matches[0]
                row["process_isolated"] = True
            print(json.dumps(row, ensure_ascii=False))
            append(args.results, row)
            rows.append(row)
    summary = {
        "axis": AXIS + "_summary",
        "status": "pass" if rows and all(row.get("status") == "pass" for row in rows) else "fail",
        "models": len(rows),
        "pass_models": sum(row.get("status") == "pass" for row in rows),
        "process_isolated": True,
    }
    print(json.dumps(summary, ensure_ascii=False))
    append(args.results, summary)
    return 1 if args.fail_on_gate and summary["status"] != "pass" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16", "keep"])
    parser.add_argument("--prompt", default="User: verify conservative MLX defaults. Assistant:")
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not models:
        raise ValueError("--models must not be empty")
    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "models": models,
        "dtype": args.dtype,
        **provenance(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append(args.results, env)
    if args.dry_run:
        return 0
    if len(models) > 1 and not args.isolated_child:
        return isolated_parent(args, models)

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    model_path = models[0]
    model = load_mlx_rwkv7_model(model_path, dtype=args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = [int(value) for value in tokenizer(args.prompt, add_special_tokens=False).input_ids]
    if not ids:
        raise RuntimeError("safe-default prompt tokenized to zero tokens")

    initial = model.telemetry()
    logits, state = model.prefill([ids])
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    logits, state = model.decode_step(token, state)
    mx.eval(logits)
    single = model.telemetry()

    batch_ids = [ids, ids]
    batch_logits, batch_state = model.prefill(batch_ids)
    model.prepare_compiled_decode(batch_size=2)
    before_auto_validated = sorted(model._compiled_decode_validated_batches)
    batch_token = mx.argmax(batch_logits[:, -1, :], axis=-1).astype(mx.int32)
    batch_logits, batch_state = model.decode_step(batch_token, batch_state)
    mx.eval(batch_logits)
    after = model.telemetry()

    gates = {
        "no_implicit_quantization": len(model.quantized_linears) == 0,
        "reference_wkv_default": initial["wkv_backend"] == "reference",
        "recurrent_prefill_default": initial["prefill_backend"] == "recurrent",
        "scan_prefill_default_off": initial["wkv_scan_prefill_mode"] == "off",
        "auto_decode_default": initial["decode_backend"] == "auto",
        "single_unvalidated_auto_uses_eager": single["decode_backend_last"] == "eager",
        "single_prefill_uses_recurrent": single["prefill_backend_last"] == "recurrent",
        "prepared_batch_not_validated": 2 not in before_auto_validated,
        "prepared_unvalidated_batch_uses_eager": after["decode_backend_last"] == "eager",
        "no_rejected_route_promoted": 2 not in after["decode_compiled_validated_batches"],
        "seen_tokens_exact": int(state.seen_tokens) == len(ids) + 1
        and int(batch_state.seen_tokens) == len(ids) + 1,
    }
    row = {
        "axis": AXIS,
        "status": "pass" if all(gates.values()) else "fail",
        "model": Path(model_path).name,
        "model_path": model_path,
        "dtype": args.dtype,
        "batch_sizes": [1, 2],
        "prompt_tokens": len(ids),
        "gates": gates,
        "initial": {
            "wkv_backend": initial["wkv_backend"],
            "prefill_backend": initial["prefill_backend"],
            "decode_backend": initial["decode_backend"],
            "wkv_scan_prefill_mode": initial["wkv_scan_prefill_mode"],
        },
        "batch2_compile_prepared": 2 in after["decode_compiled_batches"],
        "batch2_compile_validated": 2 in after["decode_compiled_validated_batches"],
        "decode_backend_used": after["decode_backend_last"],
        **provenance(),
    }
    print(json.dumps(row, ensure_ascii=False))
    append(args.results, row)
    del model
    gc.collect()
    mx.clear_cache()
    return 1 if args.fail_on_gate and row["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
