#!/usr/bin/env python3
# coding=utf-8
"""Capture and compare matched-precision Native HF and official v3a inference.

The two engines are captured in separate processes so a 32 GiB card never has
to hold two large checkpoints at once. Both captures consume the prompt one
token at a time from zero FP16 recurrent state, then greedily decode the same
number of steps. The compare phase is CPU-only and checks logits, tokens,
per-layer recurrent state, shift state, and elapsed-position state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch


DEFAULT_PROMPT = (
    "User: Summarize recurrent neural networks and cache reuse.\n\nAssistant:" * 16
)

ALIGNMENT_THRESHOLDS = {
    "logits": {"min_cosine": 0.9999, "max_abs": 0.125},
    "state": {"min_cosine": 0.9999, "max_abs": 1.0},
    "xpa": {"min_cosine": 0.9999, "max_abs": 0.125},
    "xpf": {"min_cosine": 0.9999, "max_abs": 0.125},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"tensor shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    left_flat = left.detach().cpu().reshape(-1)
    right_flat = right.detach().cpu().reshape(-1)
    max_abs = 0.0
    sum_abs = 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    finite = True
    total = int(left_flat.numel())
    chunk_size = 1 << 20
    for start in range(0, total, chunk_size):
        lhs = left_flat[start : start + chunk_size].float()
        rhs = right_flat[start : start + chunk_size].float()
        finite = finite and bool(torch.isfinite(lhs).all()) and bool(torch.isfinite(rhs).all())
        diff = (lhs - rhs).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max()))
            sum_abs += float(diff.double().sum())
            dot += float((lhs.double() * rhs.double()).sum())
            left_norm += float((lhs.double() * lhs.double()).sum())
            right_norm += float((rhs.double() * rhs.double()).sum())
    denominator = max((left_norm * right_norm) ** 0.5, 1.0e-30)
    cosine = 1.0 if left_norm == 0.0 and right_norm == 0.0 and max_abs == 0.0 else dot / denominator
    return {
        "shape": list(left.shape),
        "dtype_native": str(left.dtype),
        "dtype_official": str(right.dtype),
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": sum_abs / max(total, 1),
        "cosine": cosine,
    }


def metrics_pass(metrics: dict[str, Any], kind: str) -> bool:
    threshold = ALIGNMENT_THRESHOLDS[kind]
    return bool(
        metrics["finite"]
        and metrics["cosine"] >= threshold["min_cosine"]
        and metrics["max_abs"] <= threshold["max_abs"]
    )


def git_revision(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_official_source(
    path: str | Path,
    *,
    expected_commit: str,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    source_dir = Path(path)
    if (source_dir / ".git").is_dir():
        revision = git_revision(source_dir)
        if revision != expected_commit:
            raise RuntimeError(
                f"official checkout is not pinned: expected {expected_commit}, got {revision}"
            )
        return {"method": "git", "commit": revision, "files": {}}
    if not manifest_path:
        raise RuntimeError(
            "official source has no .git directory; --official-source-manifest is required"
        )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("commit") != expected_commit:
        raise RuntimeError("official source manifest commit does not match the required pin")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError("official source manifest must contain file SHA256 entries")
    verified: dict[str, str] = {}
    for relative, expected_hash in sorted(expected_files.items()):
        candidate = source_dir / relative
        if not candidate.is_file():
            raise RuntimeError(f"official source manifest file is missing: {relative}")
        actual = sha256_file(candidate)
        if actual != expected_hash:
            raise RuntimeError(
                f"official source hash mismatch for {relative}: expected {expected_hash}, got {actual}"
            )
        verified[str(relative)] = actual
    return {"method": "sha256_manifest", "commit": expected_commit, "files": verified}


def zero_native_cache(model: Any, batch_size: int, device: str):
    from rwkv7_hf.native_model import NativeRWKV7Cache

    config = model.config
    layers = int(config.num_hidden_layers)
    heads = int(config.num_heads)
    head_dim = int(config.head_dim)
    hidden = int(config.hidden_size)
    attention_hidden = int(getattr(config, "attention_hidden_size", heads * head_dim))
    state = [
        torch.zeros(batch_size, heads, head_dim, head_dim, device=device, dtype=torch.float16)
        for _ in range(layers)
    ]
    xpa = [
        torch.zeros(batch_size, hidden, device=device, dtype=torch.float16)
        for _ in range(layers)
    ]
    xpf = [
        torch.zeros(batch_size, hidden, device=device, dtype=torch.float16)
        for _ in range(layers)
    ]
    v_first = torch.zeros(batch_size, attention_hidden, device=device, dtype=torch.float16)
    return NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=0)


def snapshot_native(model: Any, cache: Any) -> dict[str, torch.Tensor]:
    runners = getattr(model, "_rwkv7_native_graph_runner_cache", {})
    bound = cache._native_graph_bound_runner() if hasattr(cache, "_native_graph_bound_runner") else None
    if bound is None and runners:
        bound = next(reversed(runners.values()))
    if bound is None:
        raise RuntimeError("native capture requires an active NativeGraphRunner")
    heads = int(model.config.num_heads)
    elapsed = bound.elapsed.view(1, bound.batch_size, 1).expand(
        int(model.config.num_hidden_layers), bound.batch_size, heads
    )
    return {
        "state": torch.stack([item.detach().cpu() for item in cache._state]),
        "xpa": torch.stack([item.detach().cpu() for item in cache._xpa]),
        "xpf": torch.stack([item.detach().cpu() for item in cache._xpf]),
        "elapsed": elapsed.detach().cpu(),
    }


def snapshot_official(state: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "state": state[1].detach().cpu(),
        "xpa": state[0][:, 0].detach().cpu(),
        "xpf": state[0][:, 1].detach().cpu(),
        "elapsed": state[2].detach().cpu(),
    }


def run_tokens(
    *,
    step: Callable[[torch.Tensor], torch.Tensor],
    prompt_ids: torch.Tensor,
    decode_steps: int,
    snapshot: Callable[[], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    logits = None
    for index in range(int(prompt_ids.shape[1])):
        logits = step(prompt_ids[:, index : index + 1])
    if logits is None:
        raise ValueError("prompt must contain at least one token")
    prefill = snapshot()
    logits_rows = [logits.detach().cpu()]
    token = logits.argmax(dim=-1, keepdim=True)
    greedy: list[torch.Tensor] = []
    for _ in range(int(decode_steps)):
        logits = step(token)
        token = logits.argmax(dim=-1, keepdim=True)
        logits_rows.append(logits.detach().cpu())
        greedy.append(token.reshape(-1).detach().cpu())
    final = snapshot()
    return {
        "logits": torch.stack(logits_rows),
        "greedy_tokens": torch.stack(greedy) if greedy else torch.empty(0, prompt_ids.shape[0], dtype=torch.long),
        "prefill": prefill,
        "final": final,
    }


def capture_native(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    ids = tokenizer(
        args.prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[:, : args.prompt_tokens]
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    ).eval()
    captures: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        if hasattr(model, "rwkv7_clear_native_graph_cache"):
            model.rwkv7_clear_native_graph_cache()
        batch_ids = ids.repeat(batch_size, 1).to(args.device)
        cache = zero_native_cache(model, batch_size, args.device)

        def step(tokens: torch.Tensor) -> torch.Tensor:
            nonlocal cache
            output = model.rwkv7_forward_token(
                tokens,
                past_key_values=cache,
                return_dict=True,
                copy_logits=True,
            )
            cache = output.past_key_values
            return output.logits[:, -1].detach()

        captures[str(batch_size)] = run_tokens(
            step=step,
            prompt_ids=batch_ids,
            decode_steps=args.decode_steps,
            snapshot=lambda: snapshot_native(model, cache),
        )
    return {
        "engine": "native_hf",
        "source_revision": args.native_source_revision,
        "precision": "fp16_state_fp16_io",
        "prompt": args.prompt,
        "prompt_tokens": int(ids.shape[1]),
        "prompt_ids": ids.cpu(),
        "decode_steps": args.decode_steps,
        "batch_sizes": args.batch_sizes,
        "captures": captures,
    }


def load_official(args: argparse.Namespace):
    source_verification = verify_official_source(
        args.official_dir,
        expected_commit=args.official_commit,
        manifest_path=args.official_source_manifest,
    )
    revision = str(source_verification["commit"])
    os.environ.setdefault("RWKV_V7_ON", "1")
    sys.path.insert(0, args.official_dir)
    module = importlib.import_module(args.official_module)
    module.MODEL_PATH = args.official_model
    module.WKV_MODE = "fp16"
    module.EMB_DEVICE = "gpu"
    module.RKV_MODE = "off"
    module.CMIX_SPARSE = "no-fc"
    module.LOWRANK_WEIGHT = "both"
    module.ORIG_LINEAR_GROUPS = module.parse_orig_linear_groups(
        "att_c2c,ffn_key,head"
    )
    os.chdir(args.official_dir)
    torch.set_grad_enabled(False)
    module.load_extensions(module.WKV_MODE)
    return module.RWKV7(), revision, source_verification


def capture_official(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    ids = tokenizer(
        args.prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[:, : args.prompt_tokens]
    model, revision, source_verification = load_official(args)
    captures: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        batch_ids = ids.repeat(batch_size, 1).to(args.device)
        state = model.zero_state(batch_size)

        def step(tokens: torch.Tensor) -> torch.Tensor:
            return model.forward(tokens, state).view(batch_size, -1).detach()

        captures[str(batch_size)] = run_tokens(
            step=step,
            prompt_ids=batch_ids,
            decode_steps=args.decode_steps,
            snapshot=lambda: snapshot_official(state),
        )
    return {
        "engine": "official_v3a",
        "source_revision": revision,
        "source_verification": source_verification,
        "precision": "fp16_state_fp16_io",
        "prompt": args.prompt,
        "prompt_tokens": int(ids.shape[1]),
        "prompt_ids": ids.cpu(),
        "decode_steps": args.decode_steps,
        "batch_sizes": args.batch_sizes,
        "captures": captures,
    }


def compare_captures(
    native: dict[str, Any],
    official: dict[str, Any],
    *,
    expected_official_commit: str,
) -> dict[str, Any]:
    if native.get("engine") != "native_hf" or official.get("engine") != "official_v3a":
        raise ValueError("capture engines are missing or reversed")
    if official.get("source_revision") != expected_official_commit:
        raise ValueError("official capture commit does not match the required pin")
    for key in ("precision", "prompt_tokens", "decode_steps", "batch_sizes"):
        if native.get(key) != official.get(key):
            raise ValueError(f"capture metadata mismatch for {key}")
    if not torch.equal(native["prompt_ids"], official["prompt_ids"]):
        raise ValueError("capture prompt token IDs do not match")

    rows: list[dict[str, Any]] = []
    for batch_size in native["batch_sizes"]:
        label = str(batch_size)
        nat = native["captures"][label]
        off = official["captures"][label]
        logits_metrics = tensor_metrics(nat["logits"], off["logits"])
        native_top1 = nat["logits"].argmax(dim=-1)
        official_top1 = off["logits"].argmax(dim=-1)
        logits_metrics["top1_matches"] = int((native_top1 == official_top1).sum())
        logits_metrics["top1_total"] = int(native_top1.numel())
        logits_metrics["top1_match_rate"] = (
            logits_metrics["top1_matches"] / max(logits_metrics["top1_total"], 1)
        )
        logits_metrics["threshold_pass"] = metrics_pass(logits_metrics, "logits")
        states: dict[str, Any] = {}
        for phase in ("prefill", "final"):
            states[phase] = {
                name: tensor_metrics(nat[phase][name], off[phase][name])
                for name in ("state", "xpa", "xpf")
            }
            for name in ("state", "xpa", "xpf"):
                states[phase][name]["threshold_pass"] = metrics_pass(
                    states[phase][name], name
                )
            native_elapsed = nat[phase]["elapsed"]
            expected_elapsed = off[phase]["elapsed"].view(1, batch_size, 1).expand_as(native_elapsed)
            states[phase]["elapsed_exact"] = bool(torch.equal(native_elapsed, expected_elapsed))
        greedy_exact = bool(torch.equal(nat["greedy_tokens"], off["greedy_tokens"]))
        rows.append(
            {
                "batch_size": batch_size,
                "logits": logits_metrics,
                "greedy_exact": greedy_exact,
                "greedy_matches": int((nat["greedy_tokens"] == off["greedy_tokens"]).sum()),
                "greedy_total": int(nat["greedy_tokens"].numel()),
                "states": states,
                "quality_pass": bool(
                    logits_metrics["threshold_pass"]
                    and all(
                        states[phase][name]["threshold_pass"]
                        for phase in ("prefill", "final")
                        for name in ("state", "xpa", "xpf")
                    )
                ),
            }
        )
    passed = all(
        row["greedy_exact"]
        and row["quality_pass"]
        and row["logits"]["top1_match_rate"] == 1.0
        and row["states"]["prefill"]["elapsed_exact"]
        and row["states"]["final"]["elapsed_exact"]
        for row in rows
    )
    return {
        "axis": "official_native_inference_alignment",
        "status": "pass" if passed else "fail",
        "precision": native["precision"],
        "official_commit": expected_official_commit,
        "prompt_tokens": native["prompt_tokens"],
        "decode_steps": native["decode_steps"],
        "thresholds": ALIGNMENT_THRESHOLDS,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("capture-native", "capture-official"):
        capture = subparsers.add_parser(name)
        capture.add_argument("--hf-dir", required=True)
        capture.add_argument("--output", required=True)
        capture.add_argument("--device", default="cuda")
        capture.add_argument("--prompt", default=DEFAULT_PROMPT)
        capture.add_argument("--prompt-tokens", type=int, default=8)
        capture.add_argument("--decode-steps", type=int, default=128)
        capture.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
        capture.add_argument("--native-source-revision", default="unknown")
        capture.add_argument("--official-dir", default="")
        capture.add_argument("--official-model", default="")
        capture.add_argument("--official-module", default="rwkv7_fast_v3a")
        capture.add_argument("--official-commit", default="")
        capture.add_argument("--official-source-manifest", default="")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--native", required=True)
    compare.add_argument("--official", required=True)
    compare.add_argument("--official-commit", required=True)
    compare.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "capture-native":
        capture = capture_native(args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(capture, output)
        print(json.dumps({"output": str(output), "sha256": sha256_file(output)}))
        return 0
    if args.command == "capture-official":
        if not args.official_dir or not args.official_model or not args.official_commit:
            raise ValueError("official capture requires --official-dir, --official-model, and --official-commit")
        capture = capture_official(args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(capture, output)
        print(json.dumps({"output": str(output), "sha256": sha256_file(output)}))
        return 0
    native = torch.load(args.native, map_location="cpu", weights_only=False)
    official = torch.load(args.official, map_location="cpu", weights_only=False)
    report = compare_captures(
        native,
        official,
        expected_official_commit=args.official_commit,
    )
    report["native_capture_sha256"] = sha256_file(args.native)
    report["official_capture_sha256"] = sha256_file(args.official)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
