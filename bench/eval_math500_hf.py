#!/usr/bin/env python3
# coding=utf-8
"""MATH500 evaluator for the RWKV-7 HF adapter.

The output schema mirrors the key fields from BlinkDL/Albatross
`faster3a_2605/eval_math500.py`: rollout accuracy, pass@rollout accuracy,
sample/sec, token/sec, and generation/verification artifacts.  This script is
intended for acceptance testing of the HF adapter; full acceptance should use
`--rollout 64 --max-new-tokens 1500` on the complete MATH500 file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True)
class Task:
    index: int
    problem: str
    answer: str
    subject: str = ""
    level: str = ""
    unique_id: str = ""


@torch.jit.script
def sample_logits_from_uniform(
    logits: torch.Tensor,
    uniform01: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Albatross-compatible sampler using caller-provided U[0,1)."""

    k = min(max(1, int(top_k)), logits.size(-1))
    if temperature <= 0.0 or top_p <= 0.0 or k == 1:
        return torch.argmax(logits, dim=-1)
    vals, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    probs = torch.softmax(vals if temperature == 1.0 else vals / float(temperature), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    if top_p < 1.0:
        keep = torch.argmax((cdf >= float(top_p)).to(torch.int32), dim=-1)
        mass = cdf.gather(1, keep.view(-1, 1)).view(-1)
    else:
        mass = cdf[:, -1]
    u = torch.clamp(uniform01.view(-1, 1).to(device=logits.device, dtype=torch.float32), 0.0, 0.9999999403953552)
    r = u * mass.view(-1, 1)
    picked = torch.searchsorted(cdf, r).view(-1, 1)
    return ids.gather(1, picked).view(-1)


@torch.jit.script
def sample_logits(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """Albatross-compatible sampler: temperature -> top-k -> top-p."""

    uniform01 = torch.rand((logits.size(0), 1), device=logits.device)
    return sample_logits_from_uniform(logits, uniform01, temperature, top_p, top_k)


_SPLITMIX64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _SPLITMIX64_MASK
    z = value
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _SPLITMIX64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _SPLITMIX64_MASK
    return (z ^ (z >> 31)) & _SPLITMIX64_MASK


def deterministic_uniform01(seed: int, task_index: int, sample_id: int, draw_index: int, salt: int) -> float:
    """Stable per-row RNG independent of slot/refill order."""

    key = int(seed) & _SPLITMIX64_MASK
    key ^= (int(task_index) + 0x100000001B3) * 0x9E3779B185EBCA87
    key ^= (int(sample_id) + 0xC2B2AE3D27D4EB4F) * 0xBF58476D1CE4E5B9
    key ^= (int(draw_index) + 0x165667B19E3779F9) * 0x94D049BB133111EB
    key ^= (int(salt) + 0xD6E8FEB86659FD93) * 0xD2B74407B1CE6E93
    bits = _splitmix64(key) >> 11
    return bits * (1.0 / float(1 << 53))


def deterministic_uniforms_for_work(
    args: argparse.Namespace,
    work_items: list[tuple[int, int]],
    draw_indices: list[int],
    device: str,
) -> torch.Tensor:
    vals = [
        deterministic_uniform01(args.seed, task_idx, sample_id, draw_idx, args.rng_salt)
        for (task_idx, sample_id), draw_idx in zip(work_items, draw_indices, strict=True)
    ]
    return torch.tensor(vals, dtype=torch.float32, device=device).view(-1, 1)


def load_tasks(dataset: str, limit: int = 0) -> list[Task]:
    tasks: list[Task] = []
    with open(dataset, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append(
                Task(
                    index=index,
                    problem=str(item["problem"]),
                    answer=str(item["answer"]),
                    subject=str(item.get("subject", "")),
                    level=str(item.get("level", "")),
                    unique_id=str(item.get("unique_id", index)),
                )
            )
            if limit > 0 and len(tasks) >= limit:
                break
    return tasks


def build_prompt(problem: str, prompt_style: str) -> str:
    problem = problem.strip().replace("\r\n", "\n")
    if prompt_style == "fake_think":
        return f"User: {problem}\n\nAssistant: <think></think"
    if prompt_style == "plain":
        return f"User: {problem}\n\nAssistant:"
    raise ValueError(f"unknown prompt style: {prompt_style}")


def encode_prompt(args: argparse.Namespace, tokenizer, problem: str) -> torch.Tensor:
    prompt = build_prompt(problem, args.prompt_style)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if args.add_bos:
        bos = torch.tensor([[args.bos_token_id]], dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=1)
    prompt_tokens = int(ids.shape[1])
    if prompt_tokens + args.max_new_tokens > args.ctx_limit:
        keep = max(1, args.ctx_limit - args.max_new_tokens)
        ids = ids[:, -keep:]
    return ids


def verify_completion(answer: str, completion: str) -> tuple[bool, str]:
    try:
        gold = parse(f"$\\boxed{{{answer}}}$")
        pred = parse(str(completion))
        return bool(pred and verify(gold, pred, strict=False)), ""
    except Exception as exc:  # pragma: no cover - depends on math_verify parsers
        return False, f"{type(exc).__name__}: {exc}"


def _verify_completion_worker(payload: tuple[int, str, str]) -> tuple[int, bool, str]:
    """Process-pool friendly wrapper for deferred verification."""

    row_index, answer, completion = payload
    correct, error = verify_completion(answer, completion)
    return row_index, correct, error


def verify_rows_deferred(
    rows: list[dict[str, Any]],
    *,
    workers: int,
    progress_every: int,
) -> dict[str, Any]:
    """Verify completed generations after the GPU decode loop has finished.

    The default evaluator verifies each row at slot-finish time.  That is fine
    for correctness, but it can stall dynamic batching because expensive SymPy /
    math_verify work runs on the main thread while GPU slots wait for refill.
    This helper keeps the generated completions identical and moves only the
    CPU verifier to a post-decode phase.
    """

    started = time.perf_counter()
    if workers <= 0:
        workers = min(4, max(1, os.cpu_count() or 1))
    payloads = [(idx, str(row["answer"]), str(row["completion"])) for idx, row in enumerate(rows)]
    done = 0

    if workers == 1 or len(payloads) <= 1:
        for payload in payloads:
            idx, correct, error = _verify_completion_worker(payload)
            rows[idx]["correct"] = correct
            rows[idx]["verify_error"] = error
            done += 1
            if progress_every > 0 and done % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"math500_hf verify {done}/{len(rows)} elapsed_s={elapsed:.3f}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            chunksize = max(1, len(payloads) // max(1, workers * 16))
            for idx, correct, error in pool.map(_verify_completion_worker, payloads, chunksize=chunksize):
                rows[idx]["correct"] = correct
                rows[idx]["verify_error"] = error
                done += 1
                if progress_every > 0 and done % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"math500_hf verify {done}/{len(rows)} elapsed_s={elapsed:.3f}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"math500_hf verify done rows={len(rows)} workers={workers} elapsed_s={elapsed:.3f}", flush=True)
    return {
        "deferred_verification": True,
        "verify_workers": workers,
        "verification_sec": elapsed,
    }


def trim_completion(text: str) -> str:
    text = text.split("\nUser:", 1)[0]
    if text.startswith(">"):
        text = text[1:]
    return text.strip()


def generate_one(args: argparse.Namespace, model, tokenizer, task: Task, sample_id: int) -> dict[str, Any]:
    input_ids = encode_prompt(args, tokenizer, task.problem).to(args.device)
    prompt_tokens = int(input_ids.shape[1])
    enc = {"input_ids": input_ids}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "pad_token_id": args.pad_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = out[0, prompt_tokens:]
    completion = trim_completion(tokenizer.decode(new_ids.detach().cpu().tolist(), skip_special_tokens=False))
    if args.defer_verification:
        correct, verify_error = None, ""
    else:
        correct, verify_error = verify_completion(task.answer, completion)
    generated_tokens = int(new_ids.numel())
    ended_eod = bool(generated_tokens > 0 and int(new_ids[-1].detach().cpu()) == args.eos_token_id)
    ended_user_stop = "\nUser:" in tokenizer.decode(new_ids.detach().cpu().tolist(), skip_special_tokens=False)
    return {
        "task_index": task.index,
        "local_task_index": task.index,
        "sample_id": sample_id,
        "problem": task.problem,
        "answer": task.answer,
        "subject": task.subject,
        "level": task.level,
        "unique_id": task.unique_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_including_eod": generated_tokens,
        "tokens_including_stop": generated_tokens,
        "ended_eod": ended_eod,
        "ended_user_stop": ended_user_stop,
        "stop_reason": "eod" if ended_eod else "user_stop" if ended_user_stop else "max_tokens",
        "truncated": generated_tokens >= args.max_new_tokens and not ended_eod and not ended_user_stop,
        "completion": completion,
        "correct": correct,
        "verify_error": verify_error,
        "generate_sec": elapsed,
        "generate_tokps": generated_tokens / max(elapsed, 1e-9),
    }


def cache_tensors(cache) -> list[dict[str, Any]]:
    states = getattr(cache, "states", None)
    if states is not None:
        return list(states)
    return list(cache)


def _is_native_tuple_cache(cache) -> bool:
    return all(hasattr(cache, attr) for attr in ("_state", "_xpa", "_xpf", "_v_first"))


def _zeros_like_batch(value, batch_size: int):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        if not shape:
            raise ValueError("cache tensor is scalar")
        shape[0] = batch_size
        return torch.zeros(shape, device=value.device, dtype=value.dtype)
    if isinstance(value, list):
        return [_zeros_like_batch(v, batch_size) for v in value]
    if isinstance(value, tuple):
        return tuple(_zeros_like_batch(v, batch_size) for v in value)
    if isinstance(value, dict):
        return {k: _zeros_like_batch(v, batch_size) for k, v in value.items()}
    return value


def _copy_cache_row(dst, row: int, src) -> None:
    if src is None or dst is None:
        return
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        with torch.inference_mode():
            dst[row : row + 1].copy_(src.to(device=dst.device, dtype=dst.dtype))
        return
    if isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
        for d, s in zip(dst, src, strict=False):
            _copy_cache_row(d, row, s)
        return
    if isinstance(src, dict) and isinstance(dst, dict):
        for key, value in src.items():
            if key in dst:
                _copy_cache_row(dst[key], row, value)


def build_batch_cache(cache_cls, example_cache, batch_size: int):
    if _is_native_tuple_cache(example_cache):
        return cache_cls(
            _zeros_like_batch(getattr(example_cache, "_state", None), batch_size),
            _zeros_like_batch(getattr(example_cache, "_xpa", None), batch_size),
            _zeros_like_batch(getattr(example_cache, "_xpf", None), batch_size),
            _zeros_like_batch(getattr(example_cache, "_v_first", None), batch_size),
            seen_tokens=0,
        )
    batch_cache = cache_cls(seen_tokens=0)
    batch_cache.states = []
    for state in cache_tensors(example_cache):
        row: dict[str, Any] = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
        for key in ("recurrent_state", "conv_state", "ffn_state"):
            value = state.get(key) if isinstance(state, dict) else None
            if isinstance(value, torch.Tensor):
                shape = list(value.shape)
                if not shape:
                    raise ValueError(f"cache tensor {key} is scalar")
                shape[0] = batch_size
                row[key] = torch.zeros(shape, device=value.device, dtype=value.dtype)
        batch_cache.states.append(row)
    return batch_cache


def copy_single_cache_into_batch(batch_cache, row: int, single_cache) -> None:
    if _is_native_tuple_cache(batch_cache) and _is_native_tuple_cache(single_cache):
        for attr in ("_state", "_xpa", "_xpf", "_v_first"):
            _copy_cache_row(getattr(batch_cache, attr), row, getattr(single_cache, attr))
        return
    for layer_idx, src in enumerate(cache_tensors(single_cache)):
        dst = batch_cache._ensure_layer(layer_idx)
        if not isinstance(src, dict):
            continue
        for key in ("recurrent_state", "conv_state", "ffn_state"):
            src_tensor = src.get(key)
            dst_tensor = dst.get(key)
            if isinstance(src_tensor, torch.Tensor) and isinstance(dst_tensor, torch.Tensor):
                with torch.inference_mode():
                    dst_tensor[row : row + 1].copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
        dst["attn_state"] = None


def prefill_one(args: argparse.Namespace, model, ids: torch.Tensor):
    if args.prefill_backend == "native" and hasattr(model, "rwkv7_prefill_native"):
        return model.rwkv7_prefill_native(ids, logits_to_keep=1, return_dict=True)
    return model(ids, use_cache=True, logits_to_keep=1, return_dict=True)




def clone_detach_cache(cache):
    """Clone an HF/RWKV recurrent cache and detach tensors when the cache supports it.

    Older NativeRWKV7Cache snapshots expose ``clone()`` but not ``detach()``.
    The MATH500 prompt-cache path only needs an inference-safe immutable copy,
    so recursively detaching common private tensor containers is sufficient and
    keeps the evaluator compatible with already-converted model dirs.
    """

    cloned = cache.clone() if hasattr(cache, "clone") else cache
    detach = getattr(cloned, "detach", None)
    if callable(detach):
        return detach(inplace=True)

    def detach_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, list):
            return [detach_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(detach_value(v) for v in value)
        if isinstance(value, dict):
            return {k: detach_value(v) for k, v in value.items()}
        return value

    for attr in ("_state", "_xpa", "_xpf", "_v_first", "states"):
        if hasattr(cloned, attr):
            setattr(cloned, attr, detach_value(getattr(cloned, attr)))
    return cloned

def build_prefill_cache(
    args: argparse.Namespace,
    tasks: list[Task],
    model,
    tokenizer,
) -> tuple[dict[int, tuple[Any, torch.Tensor, int]], float]:
    cache: dict[int, tuple[Any, torch.Tensor, int]] = {}
    started = time.perf_counter()
    for done, task in enumerate(tasks, 1):
        ids = encode_prompt(args, tokenizer, task.problem).to(args.device)
        with torch.inference_mode():
            out = prefill_one(args, model, ids)
        state = clone_detach_cache(out.past_key_values)
        logits = out.logits[:, -1, :].reshape(-1).detach().clone()
        cache[task.index] = (state, logits, int(ids.shape[1]))
        if args.progress_every > 0 and done % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"math500_hf prefill_cache {done}/{len(tasks)} elapsed_s={elapsed:.3f}", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"math500_hf prefill_cache done prompts={len(tasks)} elapsed_s={elapsed:.3f}", flush=True)
    return cache, elapsed


def run_dynamic_batched(args: argparse.Namespace, model, tokenizer, tasks: list[Task]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Albatross-style dynamic rollout with prompt-state cache and slot refill."""

    prefill_cache, prefill_sec = build_prefill_cache(args, tasks, model, tokenizer)
    work = [(task.index, sample_id) for task in tasks for sample_id in range(args.rollout)]
    if not work:
        return [], {"prefill_sec": prefill_sec, "decode_sec": 0.0, "decoded_token_events": 0, "forward_steps": 0}
    task_by_index = {task.index: task for task in tasks}
    batch_size = min(max(1, args.bsz), len(work))
    first_state = next(iter(prefill_cache.values()))[0]
    batch_cache = build_batch_cache(type(first_state), first_state, batch_size)

    pending_pos = 0
    slot_work: list[tuple[int, int] | None] = [None] * batch_size
    prompt_lengths = [0] * batch_size
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    active = [False] * batch_size
    token_counts = [0] * batch_size
    draw_counts = [0] * batch_size
    out_texts = ["" for _ in range(batch_size)]
    out_last = [0 for _ in range(batch_size)]
    next_cpu = [0 for _ in range(batch_size)]
    token_tensor = torch.empty((batch_size, 1), dtype=torch.long, device=args.device)
    rows: list[dict[str, Any]] = []
    decode_started = time.perf_counter()
    decoded_token_events = 0
    forward_steps = 0
    last_progress_t = decode_started
    last_progress_tokens = 0

    def finish_row(row: int, stop_reason: str) -> None:
        work_item = slot_work[row]
        assert work_item is not None
        task_idx, sample_id = work_item
        token_ids = generated[row]
        if args.defer_text_decode:
            raw_completion = tokenizer.decode(token_ids, skip_special_tokens=False)
        elif out_last[row] < len(token_ids):
            pending = tokenizer.decode(token_ids[out_last[row] :], skip_special_tokens=False)
            if "\ufffd" not in pending:
                out_texts[row] += pending
            raw_completion = out_texts[row]
        else:
            raw_completion = out_texts[row]
        completion = trim_completion(raw_completion)
        task = task_by_index[task_idx]
        post_user_stop = "\nUser:" in raw_completion
        final_stop_reason = "user_stop" if stop_reason == "max_tokens" and post_user_stop else stop_reason
        if args.defer_verification:
            correct, verify_error = None, ""
        else:
            correct, verify_error = verify_completion(task.answer, completion)
        rows.append(
            {
                "task_index": task.index,
                "local_task_index": task_idx,
                "sample_id": sample_id,
                "problem": task.problem,
                "answer": task.answer,
                "subject": task.subject,
                "level": task.level,
                "unique_id": task.unique_id,
                "prompt_tokens": prompt_lengths[row],
                "generated_tokens": len(token_ids),
                "tokens_including_eod": token_counts[row],
                "tokens_including_stop": token_counts[row],
                "ended_eod": final_stop_reason == "eod",
                "ended_user_stop": final_stop_reason == "user_stop",
                "stop_reason": final_stop_reason,
                "truncated": final_stop_reason == "max_tokens",
                "completion": completion,
                "correct": correct,
                "verify_error": verify_error,
            }
        )
        slot_work[row] = None
        prompt_lengths[row] = 0
        generated[row] = []
        active[row] = False
        token_counts[row] = 0
        draw_counts[row] = 0
        out_texts[row] = ""
        out_last[row] = 0
        next_cpu[row] = 0

    def refill_rows(refill: list[int]) -> list[int]:
        nonlocal pending_pos
        assigned: list[int] = []
        init_logits: list[torch.Tensor] = []
        for row in refill:
            if pending_pos >= len(work):
                break
            task_idx, sample_id = work[pending_pos]
            pending_pos += 1
            state, logits, prompt_len = prefill_cache[task_idx]
            copy_single_cache_into_batch(batch_cache, row, state)
            slot_work[row] = (task_idx, sample_id)
            prompt_lengths[row] = prompt_len
            generated[row] = []
            active[row] = True
            token_counts[row] = 0
            draw_counts[row] = 0
            out_texts[row] = ""
            out_last[row] = 0
            assigned.append(row)
            init_logits.append(logits)
        if assigned:
            init = torch.stack(init_logits, dim=0)
            if args.rng_mode == "per_sample":
                work_items = [slot_work[row] for row in assigned]
                if any(item is None for item in work_items):
                    raise RuntimeError("assigned row without work item")
                uniforms = deterministic_uniforms_for_work(
                    args,
                    [item for item in work_items if item is not None],
                    [draw_counts[row] for row in assigned],
                    args.device,
                )
                sampled = sample_logits_from_uniform(init, uniforms, args.temperature, args.top_p, args.top_k)
            else:
                sampled = sample_logits(init, args.temperature, args.top_p, args.top_k)
            for row, token in zip(assigned, sampled.detach().cpu().tolist(), strict=False):
                next_cpu[row] = int(token)
                draw_counts[row] += 1
        return assigned

    def process_next_token(row: int) -> bool:
        nonlocal decoded_token_events
        token = int(next_cpu[row])
        token_counts[row] += 1
        decoded_token_events += 1
        if token == args.eos_token_id:
            finish_row(row, "eod")
            return False
        generated[row].append(token)
        if args.defer_text_decode:
            if token_counts[row] >= args.max_new_tokens:
                finish_row(row, "max_tokens")
                return False
            return True
        pending = tokenizer.decode(generated[row][out_last[row] :], skip_special_tokens=False)
        if "\ufffd" not in pending:
            out_texts[row] += pending
            out_last[row] = len(generated[row])
            if "\nUser:" in out_texts[row]:
                finish_row(row, "user_stop")
                return False
        if token_counts[row] >= args.max_new_tokens:
            finish_row(row, "max_tokens")
            return False
        return True

    refill_rows(list(range(batch_size)))
    while any(active) or pending_pos < len(work):
        scan_rows = list(range(batch_size))
        forward_rows: list[int] = []
        while scan_rows:
            refill: list[int] = []
            next_scan: list[int] = []
            for row in scan_rows:
                if active[row]:
                    if process_next_token(row):
                        forward_rows.append(row)
                    else:
                        refill.append(row)
                elif pending_pos < len(work):
                    refill.append(row)
            if refill:
                next_scan = refill_rows(refill)
            scan_rows = next_scan

        if not forward_rows:
            continue
        token_tensor.fill_(0)
        for row in forward_rows:
            token_tensor[row, 0] = next_cpu[row]
        with torch.inference_mode():
            if args.decode_backend == "fast_token" and hasattr(model, "rwkv7_forward_token"):
                out = model.rwkv7_forward_token(token_tensor, past_key_values=batch_cache, return_dict=True)
            else:
                out = model(token_tensor, past_key_values=batch_cache, use_cache=True, logits_to_keep=1, return_dict=True)
        batch_cache = out.past_key_values
        logits = out.logits[:, -1, :]
        if args.rng_mode == "global":
            sampled = sample_logits(logits, args.temperature, args.top_p, args.top_k).detach().cpu().tolist()
            for row in forward_rows:
                next_cpu[row] = int(sampled[row])
                draw_counts[row] += 1
        else:
            row_logits = logits[torch.tensor(forward_rows, dtype=torch.long, device=logits.device)]
            if args.rng_mode == "per_sample":
                work_items = [slot_work[row] for row in forward_rows]
                if any(item is None for item in work_items):
                    raise RuntimeError("forward row without work item")
                uniforms = deterministic_uniforms_for_work(
                    args,
                    [item for item in work_items if item is not None],
                    [draw_counts[row] for row in forward_rows],
                    args.device,
                )
                row_sampled = sample_logits_from_uniform(row_logits, uniforms, args.temperature, args.top_p, args.top_k)
            else:
                row_sampled = sample_logits(row_logits, args.temperature, args.top_p, args.top_k)
            for row, token in zip(forward_rows, row_sampled.detach().cpu().tolist(), strict=True):
                next_cpu[row] = int(token)
                draw_counts[row] += 1
        forward_steps += 1
        if args.progress_every > 0 and forward_steps % args.progress_every == 0:
            now = time.perf_counter()
            dt_total = max(now - decode_started, 1e-9)
            dt_window = max(now - last_progress_t, 1e-9)
            delta_tokens = decoded_token_events - last_progress_tokens
            last_progress_t = now
            last_progress_tokens = decoded_token_events
            print(
                f"math500_hf dynamic step={forward_steps} active={sum(int(x) for x in active)}/{batch_size} "
                f"done={len(rows)}/{len(work)} pending={len(work) - pending_pos} "
                f"tokens={decoded_token_events} tps={decoded_token_events / dt_total:.1f} "
                f"window_tps={delta_tokens / dt_window:.1f}",
                flush=True,
            )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_sec = time.perf_counter() - decode_started
    stats = {
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "generation_elapsed_sec": prefill_sec + decode_sec,
        "decoded_token_events": decoded_token_events,
        "forward_steps": forward_steps,
        "dynamic_bsz": batch_size,
        "fast_token_backend_effective": getattr(model, "rwkv7_last_fast_token_backend", lambda: None)(),
        "cache_metrics": batch_cache.rwkv7_cache_metrics() if hasattr(batch_cache, "rwkv7_cache_metrics") else None,
        "rng_mode": args.rng_mode,
        "rng_salt": args.rng_salt,
        "defer_text_decode": args.defer_text_decode,
    }
    print(
        f"math500_hf dynamic done B={batch_size} rows={len(rows)} decode_s={decode_sec:.3f} "
        f"tokens={decoded_token_events}",
        flush=True,
    )
    return rows, stats


def summarize(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    started: float,
    out_dir: Path,
    extra_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    total = len(rows)
    correct_generations = sum(int(row["correct"]) for row in rows)
    pass_tasks = sum(1 for task_rows in by_task.values() if any(row["correct"] for row in task_rows))
    elapsed = time.perf_counter() - started
    tokens = sum(int(row.get("tokens_including_stop", row["generated_tokens"])) for row in rows)
    generation_elapsed = None
    if extra_stats:
        value = extra_stats.get("generation_elapsed_sec")
        if value is not None:
            generation_elapsed = float(value)
    if generation_elapsed is None:
        generation_elapsed = elapsed
    speed_elapsed = generation_elapsed if args.summary_speed_timing == "generation" else elapsed
    summary = {
        "axis": "math500_hf_adapter",
        "backend": "hf_adapter",
        "status": "pass",
        "model": args.hf_dir,
        "dataset": args.dataset,
        "num_tasks": len(by_task),
        "rollout": args.rollout,
        "total_generations": total,
        "correct_generations": correct_generations,
        "rollout_accuracy": correct_generations / max(total, 1),
        "pass_at_rollout_accuracy": pass_tasks / max(len(by_task), 1),
        "ended_eod": sum(int(row["ended_eod"]) for row in rows),
        "ended_user_stop": sum(int(row["ended_user_stop"]) for row in rows),
        "truncated": sum(int(row["truncated"]) for row in rows),
        "truncated_rate": sum(int(row["truncated"]) for row in rows) / max(total, 1),
        "mean_generated_tokens": sum(int(row["generated_tokens"]) for row in rows) / max(total, 1),
        "mean_tokens_including_stop": tokens / max(total, 1),
        "elapsed_sec": elapsed,
        "speed_timing": args.summary_speed_timing,
        "speed_elapsed_sec": speed_elapsed,
        "sample_per_sec": total / max(speed_elapsed, 1e-9),
        "token_per_sec": tokens / max(speed_elapsed, 1e-9),
        "wall_sample_per_sec": total / max(elapsed, 1e-9),
        "wall_token_per_sec": tokens / max(elapsed, 1e-9),
        "generation_elapsed_sec": generation_elapsed,
        "generation_sample_per_sec": total / max(generation_elapsed, 1e-9),
        "generation_token_per_sec": tokens / max(generation_elapsed, 1e-9),
        "generations_jsonl": str(out_dir / "generations.jsonl"),
        "config": {
            "hf_dir": args.hf_dir,
            "dataset": args.dataset,
            "rollout": args.rollout,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "ctx_limit": args.ctx_limit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "prompt_style": args.prompt_style,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "dynamic_batching": args.dynamic_batching,
            "bsz": args.bsz,
            "add_bos": args.add_bos,
            "prefill_backend": args.prefill_backend,
            "decode_backend": args.decode_backend,
            "rng_mode": args.rng_mode,
            "rng_salt": args.rng_salt,
            "defer_verification": args.defer_verification,
            "verify_workers": args.verify_workers,
            "summary_speed_timing": args.summary_speed_timing,
            "defer_text_decode": args.defer_text_decode,
        },
    }
    if extra_stats:
        summary.update(extra_stats)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="math500_hf_runs")
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--ctx-limit", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.28)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=("fake_think", "plain"), default="fake_think")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad-token-id", type=int, default=0)
    ap.add_argument("--eos-token-id", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--dynamic-batching", action="store_true", help="Use Albatross-style prompt cache + dynamic slot refill")
    ap.add_argument("--bsz", type=int, default=64, help="Dynamic decode batch size")
    ap.add_argument("--add-bos", action="store_true", help="Prepend token id 0 to prompts, matching Albatross eval")
    ap.add_argument("--bos-token-id", type=int, default=0)
    ap.add_argument("--prefill-backend", choices=("native", "forward"), default="native")
    ap.add_argument("--decode-backend", choices=("fast_token", "forward"), default="fast_token")
    ap.add_argument(
        "--rng-mode",
        choices=("global", "active_global", "per_sample"),
        default="global",
        help=(
            "Dynamic-batching sampler RNG mode. global matches Albatross and samples the full batch; "
            "active_global samples only active forward rows; per_sample uses deterministic per task/sample/token draws."
        ),
    )
    ap.add_argument("--rng-salt", type=int, default=0, help="Extra salt for --rng-mode=per_sample deterministic draws")
    ap.add_argument(
        "--defer-verification",
        action="store_true",
        help=(
            "Do not run math_verify inside the generation/refill loop. Generated rows are verified after decode, "
            "which keeps GPU throughput measurement from being stalled by CPU verifier work."
        ),
    )
    ap.add_argument(
        "--verify-workers",
        type=int,
        default=4,
        help="Number of deferred verifier worker processes. Use 1 for sequential; only used with --defer-verification.",
    )
    ap.add_argument(
        "--summary-speed-timing",
        choices=("wall", "generation"),
        default="wall",
        help=(
            "Timing denominator for sample_per_sec/token_per_sec. wall preserves the original end-to-end schema; "
            "generation uses prefill+decode time and is intended for GPU speed acceptance when verification is deferred."
        ),
    )
    ap.add_argument(
        "--defer-text-decode",
        action="store_true",
        help=(
            "Dynamic batching only: collect token ids and decode once when a row finishes instead of calling "
            "tokenizer.decode after every generated token. Default early user-stop behavior remains unchanged."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    tasks = load_tasks(args.dataset, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()

    started = time.perf_counter()
    extra_stats: dict[str, Any] | None = None
    if args.dynamic_batching:
        rows, extra_stats = run_dynamic_batched(args, model, tokenizer, tasks)
    else:
        rows = []
        work_total = len(tasks) * args.rollout
        for task in tasks:
            for sample_id in range(args.rollout):
                row = generate_one(args, model, tokenizer, task, sample_id)
                rows.append(row)
                if args.progress_every > 0 and len(rows) % args.progress_every == 0:
                    tokens = sum(int(r["generated_tokens"]) for r in rows)
                    elapsed = time.perf_counter() - started
                    print(
                        f"math500_hf progress rows={len(rows)}/{work_total} "
                        f"tokens={tokens} tps={tokens / max(elapsed, 1e-9):.1f}",
                        flush=True,
                    )

    generation_done = time.perf_counter()
    if extra_stats is None:
        extra_stats = {}
    extra_stats.setdefault("generation_elapsed_sec", generation_done - started)
    extra_stats.setdefault("deferred_verification", False)
    if args.defer_verification:
        extra_stats.update(
            verify_rows_deferred(rows, workers=args.verify_workers, progress_every=args.progress_every)
        )

    rows.sort(key=lambda x: (x["task_index"], x["sample_id"]))
    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(args, rows, started, out_dir, extra_stats=extra_stats)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MATH500_HF_ADAPTER_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
