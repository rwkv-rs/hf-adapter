#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@contextmanager
def fast_forward_env(enabled: bool):
    old = os.environ.get("RWKV7_FAST_FORWARD")
    os.environ["RWKV7_FAST_FORWARD"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def run_decode_case(model, input_ids: torch.Tensor, decode_steps: int, max_diff_limit: float, fast_fn, label: str) -> None:
    prefill_ids = input_ids[:, :-1]
    next_forward = input_ids[:, -1:]
    next_fast = next_forward.clone()

    max_diff = 0.0
    greedy_equal = 0
    with torch.inference_mode():
        forward_out = model(prefill_ids, use_cache=True, logits_to_keep=1)
        fast_out = model(prefill_ids, use_cache=True, logits_to_keep=1)
        forward_state = forward_out.past_key_values
        fast_state = fast_out.past_key_values
        for _ in range(decode_steps):
            with fast_forward_env(False):
                forward_out = model(next_forward, past_key_values=forward_state, use_cache=True, logits_to_keep=1)
            fast_out = fast_fn(next_fast, past_key_values=fast_state)
            forward_state = forward_out.past_key_values
            fast_state = fast_out.past_key_values
            diff = float((forward_out.logits.float() - fast_out.logits.float()).abs().max().detach().cpu())
            max_diff = max(max_diff, diff)
            next_forward = forward_out.logits[:, -1:].argmax(dim=-1)
            next_fast = fast_out.logits[:, -1:].argmax(dim=-1)
            greedy_equal += int(torch.equal(next_forward, next_fast))
        # The optimized cache must remain compatible with the standard HF
        # recurrent forward path so serving code can fall back after a fast step.
        with fast_forward_env(False):
            forward_out = model(next_forward, past_key_values=forward_state, use_cache=True, logits_to_keep=1)
            fallback_out = model(next_fast, past_key_values=fast_state, use_cache=True, logits_to_keep=1)
        forward_state = forward_out.past_key_values
        fast_state = fallback_out.past_key_values
        fallback_diff = float((forward_out.logits.float() - fallback_out.logits.float()).abs().max().detach().cpu())
        max_diff = max(max_diff, fallback_diff)

    print(f"{label} max_abs_diff", max_diff)
    print(f"{label} fallback_max_abs_diff", fallback_diff)
    print(f"{label} greedy_equal", greedy_equal, "/", decode_steps)
    print(f"{label} seq_length_forward", forward_state.get_seq_length())
    print(f"{label} seq_length_fast", fast_state.get_seq_length())
    assert max_diff <= max_diff_limit, (label, max_diff)
    assert greedy_equal == decode_steps, label
    assert forward_state.get_seq_length() == fast_state.get_seq_length(), label


def run_forward_fast_path_case(model, input_ids: torch.Tensor, max_diff_limit: float, label: str) -> None:
    prefill_ids = input_ids[:, :-1]
    token = input_ids[:, -1:]
    with torch.inference_mode():
        ref_prefill = model(prefill_ids, use_cache=True, logits_to_keep=1)
        fast_prefill = model(prefill_ids, use_cache=True, logits_to_keep=1)
        with fast_forward_env(False):
            ref = model(token, past_key_values=ref_prefill.past_key_values, use_cache=True, logits_to_keep=1)
        with fast_forward_env(True):
            fast = model(token, past_key_values=fast_prefill.past_key_values, use_cache=True, logits_to_keep=1)
    diff = float((ref.logits.float() - fast.logits.float()).abs().max().detach().cpu())
    effective = last_fast_token_backend(model)
    print(f"{label} forward_fast_path_max_abs_diff", diff)
    print(f"{label} forward_fast_path_effective_backend", effective)
    assert diff <= max_diff_limit, (label, diff)
    assert effective in {"native_graph", "native_jit", "fla"}, effective


def graph_cache_limit() -> int:
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_CACHE_SIZE", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def native_graph_cache_batch_sizes(model) -> list[int]:
    cache = getattr(model, "_rwkv7_native_graph_runner_cache", None)
    if isinstance(cache, tuple) and len(cache) == 2:
        key = cache[0]
        return [int(key[-1])] if isinstance(key, tuple) and key else []
    if hasattr(cache, "keys"):
        return sorted({int(key[-1]) for key in cache.keys() if isinstance(key, tuple) and key})
    return []


def check_native_graph_cache(model, batch_sizes: list[int]) -> None:
    cached = native_graph_cache_batch_sizes(model)
    print("native_graph_cache_batch_sizes", cached)
    expected = sorted(set(int(v) for v in batch_sizes))
    if graph_cache_limit() >= len(expected):
        assert set(expected).issubset(cached), (expected, cached)
    assert 1 in cached, cached
    assert hasattr(model, "rwkv7_clear_native_graph_cache"), "missing native graph cache clear API"
    cleared = model.rwkv7_clear_native_graph_cache()
    assert cleared == len(cached), (cleared, cached)
    assert native_graph_cache_batch_sizes(model) == []
    assert hasattr(model, "rwkv7_warmup_fast_token"), "missing native graph warmup API"
    warmed = model.rwkv7_warmup_fast_token(expected, backend="native_graph")
    print("native_graph_warmup", warmed)
    assert all(warmed[int(b)] == "native_graph" for b in expected), warmed
    if hasattr(model, "rwkv7_native_graph_cache_batch_sizes"):
        cached_after_warmup = model.rwkv7_native_graph_cache_batch_sizes()
    else:
        cached_after_warmup = native_graph_cache_batch_sizes(model)
    print("native_graph_cache_batch_sizes_after_warmup", cached_after_warmup)
    if graph_cache_limit() >= len(expected):
        assert set(expected).issubset(cached_after_warmup), (expected, cached_after_warmup)
    assert model.rwkv7_clear_native_graph_cache() == len(cached_after_warmup)


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-diff", type=float, default=0.15)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--fast-token-layouts", nargs="+", default=["3d"], choices=["3d", "2d"],
                    help="Fast-token tensor layouts to validate; 3d is the current production baseline")
    ap.add_argument("--fast-token-backends", nargs="+", default=["auto"], choices=["auto", "fla", "native_jit", "native_graph"],
                    help="Fast-token backends to validate; auto picks native_graph/native_jit/fla in that order when available")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    assert hasattr(model, "rwkv7_forward_one"), "Model does not expose rwkv7_forward_one"
    assert hasattr(model, "rwkv7_forward_token"), "Model does not expose rwkv7_forward_token"
    set_attn_mode(model, args.attn_mode)

    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(args.device) if args.device.startswith("cuda") else enc.input_ids
    assert input_ids.shape[1] >= 2, "Prompt must tokenize to at least two tokens"

    old_layout = os.environ.get("RWKV7_FAST_TOKEN_LAYOUT")
    old_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")
    old_native_backend = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    try:
        for backend in args.fast_token_backends:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = backend
            if backend in {"auto", "native_jit", "native_graph"}:
                # NativeRWKV7ForCausalLM routes its FLA-free model core through
                # the native backend selector; the legacy FAST selector remains
                # necessary for wrapper checkpoints.  Set both so this public
                # contract exercises the requested backend on either model.
                os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = backend
            else:
                os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
            for layout in args.fast_token_layouts:
                os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = layout
                for bsz in args.batch_sizes:
                    ids = input_ids.repeat(bsz, 1)
                    run_decode_case(
                        model,
                        ids,
                        args.decode_steps,
                        args.max_diff,
                        model.rwkv7_forward_token,
                        label=f"rwkv7_forward_token backend={backend} layout={layout} bsz={bsz}",
                    )
                    effective = last_fast_token_backend(model)
                    print(f"rwkv7_forward_token backend={backend} layout={layout} bsz={bsz} effective_backend", effective)
                    if backend != "auto":
                        assert effective == backend, (backend, effective)
                    else:
                        assert effective in {"native_graph", "native_jit", "fla"}, effective
                run_decode_case(
                    model,
                    input_ids,
                    args.decode_steps,
                    args.max_diff,
                    model.rwkv7_forward_one,
                    label=f"rwkv7_forward_one backend={backend} layout={layout} bsz=1",
                )
                effective = last_fast_token_backend(model)
                print(f"rwkv7_forward_one backend={backend} layout={layout} bsz=1 effective_backend", effective)
                if backend != "auto":
                    assert effective == backend, (backend, effective)
                else:
                    assert effective in {"native_graph", "native_jit", "fla"}, effective
                run_forward_fast_path_case(
                    model,
                    input_ids,
                    args.max_diff,
                    label=f"hf_forward backend={backend} layout={layout} bsz=1",
                )
                if backend == "native_graph" or effective == "native_graph":
                    check_native_graph_cache(model, args.batch_sizes)
    finally:
        if old_layout is None:
            os.environ.pop("RWKV7_FAST_TOKEN_LAYOUT", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = old_layout
        if old_backend is None:
            os.environ.pop("RWKV7_FAST_TOKEN_BACKEND", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = old_backend
        if old_native_backend is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old_native_backend
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
