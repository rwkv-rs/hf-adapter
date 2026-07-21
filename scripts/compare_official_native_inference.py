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
    "logits": {
        "min_cosine": 0.9999,
        "max_abs": 0.125,
        "fp16_max_ulps_at_max": 4.0,
        "fp16_max_abs_tail": 0.1875,
        "fp16_max_fraction_over_abs": 5.0e-5,
    },
    "state": {"min_cosine": 0.9999, "max_abs": 1.0},
    "xpa": {
        "min_cosine": 0.9999,
        "max_abs": 0.125,
        "fp16_max_ulps_at_max": 4.0,
        "fp16_max_abs_tail": 0.1875,
        "fp16_max_fraction_over_abs": 5.0e-5,
    },
    "xpf": {
        "min_cosine": 0.9999,
        "max_abs": 0.125,
        "fp16_max_ulps_at_max": 4.0,
        "fp16_max_abs_tail": 0.1875,
        "fp16_max_fraction_over_abs": 5.0e-5,
    },
}

# Native HF and Albatross intentionally use different fused GEMM schedules.
# With FP16 IO, a long greedy trajectory can therefore be bitwise identical
# while a sparse tail of non-winning logits or shift-state elements differs by
# more than a small fixed absolute threshold.  Keep the strict elementwise
# profile above, but also expose a production trajectory profile: it has
# independent cosine, mean-error, and worst-element ceilings and is accepted
# only together with exact top-1, greedy-token, and elapsed-position checks in
# ``compare_captures``.  This is not used for FP32 or mixed-dtype captures.
FP16_TRAJECTORY_THRESHOLDS = {
    "logits": {"min_cosine": 0.9999, "max_mean_abs": 0.05, "max_abs": 1.125},
    "xpa": {"min_cosine": 0.9999, "max_mean_abs": 0.01, "max_abs": 0.375},
    "xpf": {"min_cosine": 0.9999, "max_mean_abs": 0.01, "max_abs": 0.375},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    absolute_threshold: float = 0.125,
) -> dict[str, Any]:
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
    count_over_abs = 0
    max_abs_ulps_at_max: float | None = None
    max_abs_native_value: float | None = None
    max_abs_official_value: float | None = None
    total = int(left_flat.numel())
    chunk_size = 1 << 20
    for start in range(0, total, chunk_size):
        lhs_raw = left_flat[start : start + chunk_size]
        rhs_raw = right_flat[start : start + chunk_size]
        lhs = lhs_raw.float()
        rhs = rhs_raw.float()
        finite = finite and bool(torch.isfinite(lhs).all()) and bool(torch.isfinite(rhs).all())
        diff = (lhs - rhs).abs()
        if diff.numel():
            chunk_max, chunk_index = diff.max(dim=0)
            chunk_max_value = float(chunk_max)
            if chunk_max_value > max_abs:
                max_abs = chunk_max_value
                index = int(chunk_index)
                max_abs_native_value = float(lhs[index])
                max_abs_official_value = float(rhs[index])
                if right.dtype == torch.float16:
                    reference = rhs_raw[index]
                    direction = torch.tensor(
                        float("inf") if lhs[index] >= rhs[index] else float("-inf"),
                        dtype=torch.float16,
                    )
                    spacing = float((torch.nextafter(reference, direction) - reference).abs())
                    max_abs_ulps_at_max = max_abs / spacing if spacing > 0.0 else None
            count_over_abs += int((diff > float(absolute_threshold)).sum())
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
        "max_abs_native_value": max_abs_native_value,
        "max_abs_official_value": max_abs_official_value,
        "max_abs_ulps_at_max": max_abs_ulps_at_max,
        "count_over_abs_threshold": count_over_abs,
        "fraction_over_abs_threshold": count_over_abs / max(total, 1),
        "mean_abs": sum_abs / max(total, 1),
        "cosine": cosine,
    }


def metrics_pass(metrics: dict[str, Any], kind: str) -> bool:
    threshold = ALIGNMENT_THRESHOLDS[kind]
    fixed_abs_pass = metrics["max_abs"] <= threshold["max_abs"]
    fp16_tail_pass = bool(
        metrics["dtype_native"] == "torch.float16"
        and metrics["dtype_official"] == "torch.float16"
        and "fp16_max_ulps_at_max" in threshold
        and metrics["max_abs_ulps_at_max"] is not None
        and (
            metrics["max_abs_ulps_at_max"] <= threshold["fp16_max_ulps_at_max"]
            or metrics["max_abs"] <= threshold["fp16_max_abs_tail"]
        )
        and metrics["fraction_over_abs_threshold"]
        <= threshold["fp16_max_fraction_over_abs"]
    )
    metrics["fixed_abs_pass"] = bool(fixed_abs_pass)
    metrics["fp16_tail_pass"] = bool(fp16_tail_pass)
    return bool(
        metrics["finite"]
        and metrics["cosine"] >= threshold["min_cosine"]
        and (fixed_abs_pass or fp16_tail_pass)
    )


def metrics_pass_fp16_trajectory(metrics: dict[str, Any], kind: str) -> bool:
    """Gate bounded FP16-IO drift for an otherwise exact token trajectory."""

    threshold = FP16_TRAJECTORY_THRESHOLDS[kind]
    passed = bool(
        metrics["dtype_native"] == "torch.float16"
        and metrics["dtype_official"] == "torch.float16"
        and metrics["finite"]
        and metrics["cosine"] >= threshold["min_cosine"]
        and metrics["mean_abs"] <= threshold["max_mean_abs"]
        and metrics["max_abs"] <= threshold["max_abs"]
    )
    metrics["fp16_trajectory_pass"] = passed
    return passed


def metrics_pass_official_envelope(
    metrics: dict[str, Any],
    envelope: dict[str, Any] | None,
    *,
    multiplier: float,
) -> bool:
    if not envelope or multiplier < 1.0 or not metrics["finite"]:
        return False
    cosine_floor = 1.0 - (1.0 - float(envelope["min_cosine"])) * multiplier
    return bool(
        metrics["max_abs"] <= float(envelope["max_abs"]) * multiplier
        and metrics["fraction_over_abs_threshold"]
        <= float(envelope["max_fraction_over_abs_threshold"]) * multiplier
        and metrics["cosine"] >= cosine_floor
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
    # ``official_dir`` commonly points at an engine subdirectory inside the
    # pinned Albatross checkout. Ask Git whether the directory belongs to a
    # worktree instead of requiring a physically nested ``.git`` directory.
    try:
        root_result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_root: Path | None = Path(root_result.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        git_root = None
    if git_root is not None:
        revision = git_revision(source_dir)
        if revision != expected_commit:
            raise RuntimeError(
                f"official checkout is not pinned: expected {expected_commit}, got {revision}"
            )
        relative = source_dir.resolve().relative_to(git_root)
        pathspec = str(relative) if str(relative) != "." else "."
        for command in (
            ["git", "-C", str(git_root), "diff", "--quiet", "HEAD", "--", pathspec],
            ["git", "-C", str(git_root), "diff", "--cached", "--quiet", "HEAD", "--", pathspec],
        ):
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"official checkout has modified tracked files under {pathspec}"
                )
        return {
            "method": "git",
            "commit": revision,
            "root": str(git_root),
            "subdirectory": pathspec,
            "files": {},
        }
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
        torch.zeros(batch_size, heads, head_dim, head_dim, device=device, dtype=torch.float32)
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
    if cache._state is None or cache._xpa is None or cache._xpf is None:
        raise RuntimeError("native capture requires an initialized NativeRWKV7Cache")
    batch_size = int(cache.get_batch_size())
    heads = int(model.config.num_heads)
    # FP32 recurrent state intentionally has no FP16 dithering tensor. The
    # public cache still records the exact logical position and is available
    # for both native_graph and the >32-layer native_jit route.
    elapsed = torch.full(
        (int(model.config.num_hidden_layers), batch_size, heads),
        int(cache.get_seq_length()),
        dtype=torch.int32,
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
        "runtime": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("RWKV7_NATIVE_GRAPH_", "RWKV7_FUSED_"))
        },
        "precision": "fp32_state_fp16_io",
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
    module.WKV_MODE = args.official_wkv
    official_emb = getattr(args, "official_emb", "gpu")
    official_batched_rkv = getattr(args, "official_batched_rkv", "off")
    official_cmix_sparse = getattr(args, "official_cmix_sparse", "no-fc")
    official_lowrank_weight = getattr(args, "official_lowrank_weight", "both")
    official_orig_linear_groups = getattr(
        args, "official_orig_linear_groups", "att_c2c,ffn_key,head"
    )
    module.EMB_DEVICE = official_emb
    module.RKV_MODE = official_batched_rkv
    module.CMIX_SPARSE = official_cmix_sparse
    module.LOWRANK_WEIGHT = official_lowrank_weight
    module.ORIG_LINEAR_GROUPS = module.parse_orig_linear_groups(
        official_orig_linear_groups
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
        "runtime": {
            "wkv": args.official_wkv,
            "emb": args.official_emb,
            "batched_rkv": args.official_batched_rkv,
            "cmix_sparse": args.official_cmix_sparse,
            "lowrank_weight": args.official_lowrank_weight,
            "orig_linear_groups": sorted(
                value.strip()
                for value in args.official_orig_linear_groups.split(",")
                if value.strip() and value.strip() != "none"
            ),
        },
        "precision": (
            "fp32_state_fp16_io"
            if args.official_wkv == "fp32io16"
            else "fp16_state_fp16_io"
        ),
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
    official_self_envelope: dict[str, Any] | None = None,
    envelope_multiplier: float = 1.25,
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
        logits_metrics = tensor_metrics(
            nat["logits"],
            off["logits"],
            absolute_threshold=ALIGNMENT_THRESHOLDS["logits"]["max_abs"],
        )
        native_top1 = nat["logits"].argmax(dim=-1)
        official_top1 = off["logits"].argmax(dim=-1)
        logits_metrics["top1_matches"] = int((native_top1 == official_top1).sum())
        logits_metrics["top1_total"] = int(native_top1.numel())
        logits_metrics["top1_match_rate"] = (
            logits_metrics["top1_matches"] / max(logits_metrics["top1_total"], 1)
        )
        logits_metrics["standard_threshold_pass"] = metrics_pass(logits_metrics, "logits")
        metrics_pass_fp16_trajectory(logits_metrics, "logits")
        logits_metrics["official_self_envelope_pass"] = metrics_pass_official_envelope(
            logits_metrics,
            (official_self_envelope or {}).get("envelope", {}).get("logits"),
            multiplier=envelope_multiplier,
        )
        logits_metrics["threshold_pass"] = bool(
            logits_metrics["standard_threshold_pass"]
            or logits_metrics["official_self_envelope_pass"]
        )
        states: dict[str, Any] = {}
        for phase in ("prefill", "final"):
            states[phase] = {
                name: tensor_metrics(
                    nat[phase][name],
                    off[phase][name],
                    absolute_threshold=ALIGNMENT_THRESHOLDS[name]["max_abs"],
                )
                for name in ("state", "xpa", "xpf")
            }
            for name in ("state", "xpa", "xpf"):
                states[phase][name]["standard_threshold_pass"] = metrics_pass(
                    states[phase][name], name
                )
                if name in FP16_TRAJECTORY_THRESHOLDS:
                    metrics_pass_fp16_trajectory(states[phase][name], name)
                states[phase][name]["official_self_envelope_pass"] = (
                    metrics_pass_official_envelope(
                        states[phase][name],
                        (official_self_envelope or {})
                        .get("envelope", {})
                        .get(f"{phase}.{name}"),
                        multiplier=envelope_multiplier,
                    )
                )
                states[phase][name]["threshold_pass"] = bool(
                    states[phase][name]["standard_threshold_pass"]
                    or states[phase][name]["official_self_envelope_pass"]
                )
            native_elapsed = nat[phase]["elapsed"]
            expected_elapsed = off[phase]["elapsed"].view(1, batch_size, 1).expand_as(native_elapsed)
            states[phase]["elapsed_exact"] = bool(torch.equal(native_elapsed, expected_elapsed))
        greedy_exact = bool(torch.equal(nat["greedy_tokens"], off["greedy_tokens"]))
        standard_quality_pass = bool(
            logits_metrics["threshold_pass"]
            and all(
                states[phase][name]["threshold_pass"]
                for phase in ("prefill", "final")
                for name in ("state", "xpa", "xpf")
            )
        )
        fp16_trajectory_quality_pass = bool(
            logits_metrics["fp16_trajectory_pass"]
            and all(
                states[phase]["state"]["threshold_pass"]
                for phase in ("prefill", "final")
            )
            and all(
                states[phase][name]["fp16_trajectory_pass"]
                for phase in ("prefill", "final")
                for name in ("xpa", "xpf")
            )
        )
        rows.append(
            {
                "batch_size": batch_size,
                "logits": logits_metrics,
                "greedy_exact": greedy_exact,
                "greedy_matches": int((nat["greedy_tokens"] == off["greedy_tokens"]).sum()),
                "greedy_total": int(nat["greedy_tokens"].numel()),
                "states": states,
                "standard_quality_pass": standard_quality_pass,
                "fp16_trajectory_quality_pass": fp16_trajectory_quality_pass,
                "quality_pass": bool(
                    standard_quality_pass or fp16_trajectory_quality_pass
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
    report = {
        "axis": "official_native_inference_alignment",
        "status": "pass" if passed else "fail",
        "precision": native["precision"],
        "official_commit": expected_official_commit,
        "prompt_tokens": native["prompt_tokens"],
        "decode_steps": native["decode_steps"],
        "thresholds": ALIGNMENT_THRESHOLDS,
        "fp16_trajectory_thresholds": FP16_TRAJECTORY_THRESHOLDS,
        "rows": rows,
    }
    if official_self_envelope is not None:
        report["official_self_envelope"] = {
            "axis": official_self_envelope.get("axis"),
            "official_commit": official_self_envelope.get("official_commit"),
            "precision": official_self_envelope.get("precision"),
            "prompt_tokens": official_self_envelope.get("prompt_tokens"),
            "decode_steps": official_self_envelope.get("decode_steps"),
            "batch_sizes": official_self_envelope.get("batch_sizes"),
            "multiplier": envelope_multiplier,
            "envelope": official_self_envelope.get("envelope"),
        }
    return report


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
        capture.add_argument(
            "--official-wkv", choices=("fp16", "fp32io16"), default="fp32io16"
        )
        capture.add_argument("--official-emb", choices=("gpu", "cpu"), default="gpu")
        capture.add_argument(
            "--official-batched-rkv", choices=("auto", "on", "off"), default="off"
        )
        capture.add_argument(
            "--official-cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc"
        )
        capture.add_argument(
            "--official-lowrank-weight",
            choices=("orig", "transpose", "both"),
            default="both",
        )
        capture.add_argument(
            "--official-orig-linear-groups", default="att_c2c,ffn_key,head"
        )
    compare = subparsers.add_parser("compare")
    compare.add_argument("--native", required=True)
    compare.add_argument("--official", required=True)
    compare.add_argument("--official-commit", required=True)
    compare.add_argument("--official-self-envelope", default="")
    compare.add_argument("--official-envelope-multiplier", type=float, default=1.25)
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
    official_self_envelope = (
        json.loads(Path(args.official_self_envelope).read_text(encoding="utf-8"))
        if args.official_self_envelope
        else None
    )
    if official_self_envelope is not None:
        for key, expected in (
            ("axis", "official_fp16_self_repeat"),
            ("official_commit", args.official_commit),
            ("precision", native["precision"]),
            ("prompt_tokens", native["prompt_tokens"]),
            ("decode_steps", native["decode_steps"]),
            ("batch_sizes", native["batch_sizes"]),
        ):
            if official_self_envelope.get(key) != expected:
                raise ValueError(f"official self-envelope mismatch for {key}")
    report = compare_captures(
        native,
        official,
        expected_official_commit=args.official_commit,
        official_self_envelope=official_self_envelope,
        envelope_multiplier=args.official_envelope_multiplier,
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
