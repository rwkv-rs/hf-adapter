#!/usr/bin/env python3
# coding=utf-8
"""Dynamic-batch cache/reorder smoke test for RWKV-7 HF adapter.

The repeated-prompt batch tests catch shape/layout issues, but dynamic batching
also needs row independence and correct cache reordering. This test uses
heterogeneous same-length prompts, advances batched and per-row states, reorders
the batched cache, then verifies the next logits match the independently decoded
rows in the reordered order.
"""
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_CACHE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PROMPTS = [
    "Alpha user asks about graph theory, eigenvalues, and sparse matrices. ",
    "Beta dialogue covers cooking rice, mountain weather, and train tickets. ",
    "Gamma note discusses compilers, register allocation, and loop fusion. ",
    "Delta report mentions batteries, camera lenses, and market volatility. ",
    "Epsilon story has robots, ancient maps, and a quiet library at night. ",
]


@contextmanager
def reference_forward_env():
    old = os.environ.get("RWKV7_FAST_FORWARD")
    os.environ["RWKV7_FAST_FORWARD"] = "0"
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


def build_heterogeneous_ids(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.Tensor:
    rows = []
    for i in range(batch_size):
        text = PROMPTS[i % len(PROMPTS)] * 128
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() < prompt_tokens:
            raise ValueError(f"Prompt {i} only tokenized to {ids.numel()} tokens; need {prompt_tokens}")
        rows.append(ids[:prompt_tokens])
    out = torch.stack(rows, dim=0)
    return out.to(device) if device.startswith("cuda") else out


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def assert_close_logits(label: str, got: torch.Tensor, expected: torch.Tensor, max_diff_limit: float) -> None:
    diff = max_abs_diff(got, expected)
    print(f"{label} max_abs_diff={diff}")
    assert diff <= max_diff_limit, (label, diff)
    got_next = got[:, -1:].argmax(dim=-1)
    expected_next = expected[:, -1:].argmax(dim=-1)
    assert torch.equal(got_next, expected_next), label


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def run_case(model, ids: torch.Tensor, mode: str, decode_steps: int, max_diff_limit: float) -> None:
    assert ids.ndim == 2 and ids.shape[0] >= 2 and ids.shape[1] >= 2
    if mode == "forward":
        def step_fn(token, state):
            with reference_forward_env():
                return model(token, past_key_values=state, use_cache=True, logits_to_keep=1)
    elif mode == "fast_token":
        if not hasattr(model, "rwkv7_forward_token"):
            raise AssertionError("Model does not expose rwkv7_forward_token")
        step_fn = model.rwkv7_forward_token
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(mode)

    with torch.inference_mode():
        batched = model(ids, use_cache=True, logits_to_keep=1)
        batched_state = batched.past_key_values
        batched_next = batched.logits[:, -1:].argmax(dim=-1)
        assert hasattr(batched_state, "reorder_cache"), type(batched_state).__name__
        assert hasattr(batched_state, "select_batch"), type(batched_state).__name__
        assert hasattr(batched_state, "clone"), type(batched_state).__name__
        assert hasattr(batched_state, "detach"), type(batched_state).__name__
        assert hasattr(batched_state, "to"), type(batched_state).__name__
        assert batched_state.get_batch_size() == ids.shape[0], batched_state.get_batch_size()

        indiv_states = []
        indiv_next = []
        indiv_logits = []
        for row in range(ids.shape[0]):
            out = model(ids[row:row + 1], use_cache=True, logits_to_keep=1)
            indiv_states.append(out.past_key_values)
            indiv_next.append(out.logits[:, -1:].argmax(dim=-1))
            indiv_logits.append(out.logits)
        assert_close_logits(f"{mode} prefill batch-vs-individual", batched.logits, torch.cat(indiv_logits, dim=0), max_diff_limit)

        for step in range(decode_steps):
            batched = step_fn(batched_next, batched_state)
            batched_state = batched.past_key_values
            batched_next = batched.logits[:, -1:].argmax(dim=-1)

            expected_logits = []
            for row in range(ids.shape[0]):
                out = step_fn(indiv_next[row], indiv_states[row])
                indiv_states[row] = out.past_key_values
                indiv_next[row] = out.logits[:, -1:].argmax(dim=-1)
                expected_logits.append(out.logits)
            assert_close_logits(
                f"{mode} heterogeneous decode step={step + 1}",
                batched.logits,
                torch.cat(expected_logits, dim=0),
                max_diff_limit,
            )

        perm_list = list(reversed(range(ids.shape[0])))
        if ids.shape[0] >= 3:
            perm_list = [ids.shape[0] - 1, 0, *range(1, ids.shape[0] - 1)]
        perm = torch.tensor(perm_list, dtype=torch.long, device=ids.device)
        original_state = batched_state
        batched_state = batched_state.select_batch(perm, inplace=False)
        assert batched_state is not original_state
        assert batched_state.get_seq_length() == original_state.get_seq_length()
        assert batched_state.get_batch_size() == ids.shape[0]
        batched_next = batched_next.index_select(0, perm)
        batched = step_fn(batched_next, batched_state)
        expected_logits = []
        for src in perm_list:
            out = step_fn(indiv_next[src], indiv_states[src])
            indiv_states[src] = out.past_key_values
            indiv_next[src] = out.logits[:, -1:].argmax(dim=-1)
            expected_logits.append(out.logits)
        expected = torch.cat(expected_logits, dim=0)
        assert_close_logits(f"{mode} reordered decode", batched.logits, expected, max_diff_limit)
        assert batched.past_key_values.get_seq_length() == ids.shape[1] + decode_steps + 1

        inplace_perm_list = list(reversed(range(ids.shape[0])))
        inplace_perm = torch.tensor(inplace_perm_list, dtype=torch.long, device=ids.device)
        inplace_sources = [perm_list[i] for i in inplace_perm_list]
        inplace_state = batched.past_key_values
        metrics_before = (
            inplace_state.rwkv7_cache_metrics().get("native_graph_bound_selects")
            if hasattr(inplace_state, "rwkv7_cache_metrics")
            else None
        )
        selected_state = inplace_state.select_batch(inplace_perm, inplace=True)
        assert selected_state is inplace_state
        metrics_after = (
            inplace_state.rwkv7_cache_metrics().get("native_graph_bound_selects")
            if hasattr(inplace_state, "rwkv7_cache_metrics")
            else None
        )
        if mode == "fast_token" and metrics_before is not None and metrics_after is not None:
            effective_backend = (
                model.rwkv7_last_fast_token_backend()
                if hasattr(model, "rwkv7_last_fast_token_backend")
                else None
            )
            if effective_backend == "native_graph":
                assert metrics_after == metrics_before + 1
            else:
                assert metrics_after >= metrics_before
        batched_next = batched.logits[:, -1:].argmax(dim=-1).index_select(0, inplace_perm)
        batched = step_fn(batched_next, selected_state)
        inplace_expected = []
        for src in inplace_sources:
            out = step_fn(indiv_next[src], indiv_states[src])
            indiv_states[src] = out.past_key_values
            indiv_next[src] = out.logits[:, -1:].argmax(dim=-1)
            inplace_expected.append(out.logits)
        assert_close_logits(
            f"{mode} inplace-reordered decode",
            batched.logits,
            torch.cat(inplace_expected, dim=0),
            max_diff_limit,
        )
        assert batched.past_key_values.get_seq_length() == ids.shape[1] + decode_steps + 2
        perm_list = inplace_sources

        if ids.shape[0] > 2:
            keep_rows = torch.arange(ids.shape[0] - 1, dtype=torch.long, device=ids.device)
            keep_sources = perm_list[: ids.shape[0] - 1]
            compact_state = batched.past_key_values.batch_select(keep_rows, inplace=False)
            compact_next = batched.logits[:, -1:].argmax(dim=-1).index_select(0, keep_rows)
            assert compact_state.get_batch_size() == len(keep_sources)
            detached = compact_state.detach(inplace=False)
            assert detached is not compact_state
            assert first_tensor(detached.states).requires_grad is False
            if ids.device.type == "cuda":
                offloaded = detached.to("cpu", inplace=False)
                assert first_tensor(offloaded.states).device.type == "cpu"
                restored = offloaded.to(ids.device, inplace=False)
                assert first_tensor(restored.states).device.type == "cuda"
            else:
                restored = detached.to(ids.device, inplace=False)
            compact = step_fn(compact_next, restored)
            compact_expected = []
            for src in keep_sources:
                out = step_fn(indiv_next[src], indiv_states[src])
                indiv_states[src] = out.past_key_values
                indiv_next[src] = out.logits[:, -1:].argmax(dim=-1)
                compact_expected.append(out.logits)
            assert_close_logits(
                f"{mode} compacted decode",
                compact.logits,
                torch.cat(compact_expected, dim=0),
                max_diff_limit,
            )
            assert compact.past_key_values.get_seq_length() == ids.shape[1] + decode_steps + 3
        print(f"{mode} PASS cache_type={type(batched.past_key_values).__name__} perm={perm_list}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--max-diff", type=float, default=0.2)
    ap.add_argument("--modes", nargs="+", default=["forward", "fast_token"], choices=["forward", "fast_token"])
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
    set_attn_mode(model, args.attn_mode)

    ids = build_heterogeneous_ids(tok, args.batch_size, args.prompt_tokens, args.device)
    print(f"ids_shape={tuple(ids.shape)} modes={args.modes}")
    for mode in args.modes:
        run_case(model, ids, mode, args.decode_steps, args.max_diff)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
