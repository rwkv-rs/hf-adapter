#!/usr/bin/env python3
# coding=utf-8
"""End-to-end native-graph A/B for projection-axis W4 R/K/V decode.

Unlike the R/K/V microbenchmark, this captures one complete RWKV-7 token step:
embedding, every model layer, final norm, and lm_head. Dense and W4 graphs start
from the same dense-prefilled recurrent state. The W4 graph uses the fused
shift/mix producer that writes `[batch, 3, hidden]` directly, so no timed
`torch.stack` is present.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_jit import _block_ip_batched, extract

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "User: Explain recurrent inference and quantized state-space models.\nAssistant: " * 32


@contextmanager
def w4_rkv_mode(enabled: bool):
    old = os.environ.get("RWKV7_NATIVE_GRAPH_W4_RKV")
    os.environ["RWKV7_NATIVE_GRAPH_W4_RKV"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_GRAPH_W4_RKV", None)
        else:
            os.environ["RWKV7_NATIVE_GRAPH_W4_RKV"] = old


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def make_state(packs, batch: int, device, dtype):
    hidden = int(packs[0][1] * packs[0][2])
    state = [
        torch.zeros(batch, int(p[1]), int(p[2]), int(p[2]), device=device, dtype=torch.float32)
        for p in packs
    ]
    xpa = [torch.zeros(batch, hidden, device=device, dtype=dtype) for _ in packs]
    xpf = [torch.zeros(batch, hidden, device=device, dtype=dtype) for _ in packs]
    v_first = torch.zeros(batch, hidden, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def clone_state(values):
    state, xpa, xpf, v_first = values
    return [x.clone() for x in state], [x.clone() for x in xpa], [x.clone() for x in xpf], v_first.clone()


def token_step(model, token: torch.Tensor, packs, values):
    state, xpa, xpf, v_first = values
    base = model.model
    hidden = int(packs[0][1] * packs[0][2])
    x = F.embedding(token.reshape(-1), base.embeddings.weight).reshape(int(token.numel()), hidden)
    for layer, pack in enumerate(packs):
        x = _block_ip_batched(x, state[layer], xpa[layer], xpf[layer], v_first, pack)
    x = F.layer_norm(x, [hidden], base.norm.weight, base.norm.bias, 1e-5)
    logits = F.linear(x, model.lm_head.weight, getattr(model.lm_head, "bias", None))
    return logits


def dense_prefill(model, ids: torch.Tensor, packs):
    values = make_state(packs, int(ids.shape[0]), ids.device, model.model.embeddings.weight.dtype)
    with w4_rkv_mode(False), torch.inference_mode():
        for position in range(int(ids.shape[1])):
            token_step(model, ids[:, position], packs, values)
    return values


def greedy_rollout(model, token: torch.Tensor, packs, initial_values, *, w4: bool, steps: int) -> torch.Tensor:
    values = clone_state(initial_values)
    current = token.reshape(-1).clone()
    generated = []
    with w4_rkv_mode(w4), torch.inference_mode():
        for _ in range(steps):
            logits = token_step(model, current, packs, values)
            current = logits.argmax(dim=-1)
            generated.append(current.clone())
    return torch.stack(generated, dim=1)


def capture_graph(model, token: torch.Tensor, packs, initial_values, *, w4: bool, warmup: int):
    values = clone_state(initial_values)
    base = model.model
    batch = int(token.numel())
    hidden = int(packs[0][1] * packs[0][2])
    fixed_token = token.reshape(batch).clone()
    logits = torch.empty(batch, int(model.lm_head.weight.shape[0]), device=token.device, dtype=base.embeddings.weight.dtype)

    def one_step() -> None:
        state, xpa, xpf, v_first = values
        x = F.embedding(fixed_token, base.embeddings.weight).reshape(batch, hidden)
        for layer, pack in enumerate(packs):
            x = _block_ip_batched(x, state[layer], xpa[layer], xpf[layer], v_first, pack)
        x = F.layer_norm(x, [hidden], base.norm.weight, base.norm.bias, 1e-5)
        logits.copy_(F.linear(x, model.lm_head.weight, getattr(model.lm_head, "bias", None)))

    with w4_rkv_mode(w4):
        stream = torch.cuda.Stream(device=token.device)
        stream.wait_stream(torch.cuda.current_stream(token.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                one_step()
        torch.cuda.current_stream(token.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            one_step()
    return graph, logits, values


def graph_ms(graph, steps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / steps


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).min().detach().cpu())


def packed_bytes(packs) -> int:
    return sum(int(p[-2].numel()) * int(p[-2].element_size()) + int(p[-1].numel()) * int(p[-1].element_size()) for p in packs)


def dense_rkv_bytes(packs) -> int:
    total = 0
    for p in packs:
        for weight in p[20:23]:
            total += int(weight.numel()) * int(weight.element_size())
    return total


def run_batch(args, model, dense_packs, quant_packs, ids: torch.Tensor, batch: int) -> dict[str, Any]:
    batch_ids = ids.repeat(batch, 1)
    initial = dense_prefill(model, batch_ids, dense_packs)
    token = batch_ids[:, -1].contiguous()

    with w4_rkv_mode(False), torch.inference_mode():
        dense_logits = token_step(model, token, dense_packs, clone_state(initial)).detach()
    with w4_rkv_mode(True), torch.inference_mode():
        quant_logits = token_step(model, token, quant_packs, clone_state(initial)).detach()

    dense_tokens = greedy_rollout(
        model,
        token,
        dense_packs,
        initial,
        w4=False,
        steps=args.greedy_check_tokens,
    )
    quant_tokens = greedy_rollout(
        model,
        token,
        quant_packs,
        initial,
        w4=True,
        steps=args.greedy_check_tokens,
    )
    token_matches = dense_tokens == quant_tokens
    matching_prefix = []
    for row in token_matches:
        prefix = 0
        for matched in row:
            if not bool(matched):
                break
            prefix += 1
        matching_prefix.append(prefix)

    dense_graph, _, _ = capture_graph(model, token, dense_packs, initial, w4=False, warmup=args.warmup)
    dense_ms = graph_ms(dense_graph, args.steps)
    del dense_graph
    quant_graph, _, _ = capture_graph(model, token, quant_packs, initial, w4=True, warmup=args.warmup)
    quant_ms = graph_ms(quant_graph, args.steps)
    del quant_graph

    diff = dense_logits.float() - quant_logits.float()
    block_m = int(os.environ.get("RWKV7_NATIVE_GRAPH_W4_RKV_BLOCK_M", 8 if batch == 1 else 16))
    block_k = int(os.environ.get("RWKV7_NATIVE_GRAPH_W4_RKV_BLOCK_K", 64 if batch == 1 else 128))
    default_warps = 1 if batch == 1 else (4 if batch <= 4 else 2)
    num_warps = int(os.environ.get("RWKV7_NATIVE_GRAPH_W4_RKV_NUM_WARPS", default_warps))
    return {
        "axis": "native_graph_w4_rkv",
        "backend": "hf_adapter_native_graph",
        "status": "pass",
        "device": device_name(args.device),
        "dtype": args.dtype,
        "batch_size": batch,
        "prompt_tokens": int(batch_ids.shape[1]),
        "steps": args.steps,
        "hidden_size": int(dense_packs[0][1] * dense_packs[0][2]),
        "num_layers": len(dense_packs),
        "dense_ms_per_token_step": round(dense_ms, 6),
        "w4_ms_per_token_step": round(quant_ms, 6),
        "w4_speedup_vs_dense": round(dense_ms / quant_ms, 4),
        "dense_tokps_total": round(1000.0 * batch / dense_ms, 2),
        "w4_tokps_total": round(1000.0 * batch / quant_ms, 2),
        "logit_min_cosine": cosine_min(dense_logits, quant_logits),
        "logit_max_abs_diff": float(diff.abs().max().detach().cpu()),
        "logit_mean_abs_diff": float(diff.abs().mean().detach().cpu()),
        "argmax_matches": int((dense_logits.argmax(dim=-1) == quant_logits.argmax(dim=-1)).sum().detach().cpu()),
        "argmax_total": batch,
        "greedy_check_tokens": args.greedy_check_tokens,
        "greedy_token_matches": int(token_matches.sum().detach().cpu()),
        "greedy_token_total": int(token_matches.numel()),
        "greedy_min_matching_prefix": min(matching_prefix) if matching_prefix else 0,
        "w4_block_m": block_m,
        "w4_block_k": block_k,
        "w4_num_warps": num_warps,
        "dense_rkv_weight_mb": round(dense_rkv_bytes(dense_packs) / 1024 / 1024, 4),
        "packed_rkv_weight_mb": round(packed_bytes(quant_packs) / 1024 / 1024, 4),
        "packed_rkv_footprint_ratio": round(packed_bytes(quant_packs) / dense_rkv_bytes(dense_packs), 4),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--greedy-check-tokens", type=int, default=16)
    parser.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise ValueError("native-graph benchmark requires CUDA")
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device,
    ).eval()
    ids = tokenizer(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.prompt_tokens].to(args.device)

    with w4_rkv_mode(False):
        dense_packs, _, _, _ = extract(model)
    with w4_rkv_mode(True):
        quant_packs, _, _, _ = extract(model)

    rows = [run_batch(args, model, dense_packs, quant_packs, ids, int(batch)) for batch in args.batch_sizes]
    for row in rows:
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        output = Path(args.results)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} rows -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
