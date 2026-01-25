#!/usr/bin/env python3
# coding=utf-8
"""Two-rank Transformers-native tensor-parallel acceptance for RWKV-7.

Run with::

    torchrun --standalone --nproc-per-node=2 \
      tests/test_tensor_parallel_generate.py --model MODEL_DIR

Unlike ``test_device_map_generate.py``, this loads one shard of every planned
matrix per rank through ``tp_plan=\"auto\"`` and executes collectives inside
each RWKV layer.  The recurrent state remains replicated until a dedicated
head-local WKV kernel is available; this test therefore checks weight-shard
shapes explicitly and never labels pipeline layer placement as TP.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _broadcast_reference(value: torch.Tensor | None, shape: tuple[int, ...], dtype, rank: int, device):
    if rank == 0:
        assert value is not None
        result = value.to(device=device, dtype=dtype).contiguous()
    else:
        result = torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(result, src=0)
    return result


def _local_weight_shapes(model) -> dict[str, list[int]]:
    layer = model.model.layers[0]
    return {
        "embeddings": list(model.model.embeddings.weight.shape),
        "attn_r_proj": list(layer.attn.r_proj.weight.shape),
        "attn_o_proj": list(layer.attn.o_proj.weight.shape),
        "ffn_key": list(layer.ffn.key.weight.shape),
        "ffn_value": list(layer.ffn.value.weight.shape),
        "lm_head": list(model.lm_head.weight.shape),
    }


def main() -> int:
    os.environ.setdefault("RWKV7_NATIVE_MODEL_BACKEND", "eager")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--input-ids", default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--max-abs", type=float, default=2e-1)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("tensor-parallel acceptance requires at least two CUDA devices")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    if world_size != 2:
        raise SystemExit(f"this acceptance lane requires exactly 2 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = DTYPES[args.dtype]
    ids_list = [int(value) for value in args.input_ids.split(",") if value.strip()]
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = int(config.vocab_size)
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    input_ids = torch.tensor(
        [
            [((token + batch_index) % max(1, vocab_size - 1)) + 1 for token in ids_list]
            for batch_index in range(args.batch_size)
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    hidden_size = int(config.hidden_size)
    attention_hidden_size = int(
        getattr(config, "attention_hidden_size", config.num_heads * config.head_dim)
    )
    intermediate_size = int(config.intermediate_size)
    dimensions = (vocab_size, hidden_size, attention_hidden_size, intermediate_size)
    if any(value % world_size for value in dimensions):
        raise SystemExit(
            "vocab_size, hidden_size, attention_hidden_size, and intermediate_size "
            f"must be divisible by TP world size {world_size}; got {dimensions}"
        )

    reference_logits = reference_tokens = None
    reference_peak_mb = 0.0
    if rank == 0:
        torch.cuda.reset_peak_memory_stats(device)
        reference = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device).eval()
        with torch.inference_mode():
            reference_logits = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            ).logits
            reference_tokens = reference.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
            )
        reference_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        del reference
        torch.cuda.empty_cache()
    reference_logits = _broadcast_reference(
        reference_logits,
        (args.batch_size, 1, vocab_size),
        dtype,
        rank,
        device,
    )
    reference_tokens = _broadcast_reference(
        reference_tokens,
        (args.batch_size, len(ids_list) + args.max_new_tokens),
        torch.long,
        rank,
        device,
    )
    reference_peak = torch.tensor(reference_peak_mb, device=device)
    dist.broadcast(reference_peak, src=0)
    reference_peak_mb = float(reference_peak.item())
    dist.barrier()

    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        tp_plan="auto",
    ).eval()
    dist.barrier()
    load_s = time.perf_counter() - load_started

    shapes = _local_weight_shapes(model)
    if rank == 0:
        layer0 = model.model.layers[0]
        print(
            json.dumps(
                {
                    "tp_diagnostic": {
                        name: {
                            "plan": getattr(module, "_hf_tp_plan", None),
                            "pre_hooks": len(module._forward_pre_hooks),
                            "post_hooks": len(module._forward_hooks),
                        }
                        for name, module in {
                            "r_proj": layer0.attn.r_proj,
                            "a_lora_out": layer0.attn.a_lora.lora[2],
                            "ffn_key": layer0.ffn.key,
                            "ffn_value": layer0.ffn.value,
                            "lm_head": model.lm_head,
                        }.items()
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    expected_shapes = {
        "embeddings": [vocab_size // world_size, hidden_size],
        "attn_r_proj": [attention_hidden_size // world_size, hidden_size],
        "attn_o_proj": [hidden_size, attention_hidden_size // world_size],
        "ffn_key": [intermediate_size // world_size, hidden_size],
        "ffn_value": [hidden_size, intermediate_size // world_size],
        "lm_head": [vocab_size // world_size, hidden_size],
    }
    shape_contract = shapes == expected_shapes

    run_started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
        )
    dist.barrier()
    run_s = time.perf_counter() - run_started

    logits = output.logits
    max_abs_diff = float((logits.float() - reference_logits.float()).abs().max().item())
    logits_close = bool(
        torch.allclose(logits.float(), reference_logits.float(), atol=args.atol, rtol=args.rtol)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            logits.float().reshape(1, -1), reference_logits.float().reshape(1, -1)
        ).item()
    )
    generated_equal = bool(torch.equal(generated, reference_tokens))
    rank_tokens = [torch.empty_like(generated) for _ in range(world_size)]
    dist.all_gather(rank_tokens, generated.contiguous())
    ranks_equal = all(torch.equal(rank_tokens[0], value) for value in rank_tokens[1:])

    cache = output.past_key_values
    local_state_heads = int(cache._state[0].shape[1])
    recurrent_state_replicated = local_state_heads == int(config.num_heads)
    backend = getattr(model, "_rwkv7_native_model_last_decode_backend", None)
    local_pass = bool(
        int(getattr(model, "_tp_size", 0)) == world_size
        and shape_contract
        and max_abs_diff <= args.max_abs
        and cosine >= args.min_cosine
        and generated_equal
        and ranks_equal
        and recurrent_state_replicated
        and backend in {None, "eager"}
    )
    passed_tensor = torch.tensor(int(local_pass), dtype=torch.int32, device=device)
    dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
    passed = bool(passed_tensor.item())

    row = {
        "axis": "transformers_native_tp",
        "status": "pass" if passed else "fail",
        "world_size": world_size,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(device),
        "tp_size": int(getattr(model, "_tp_size", 0)),
        "tp_plan": dict(model.tp_plan),
        "local_weight_shapes": shapes,
        "expected_local_weight_shapes": expected_shapes,
        "shape_contract": shape_contract,
        "recurrent_state": "replicated",
        "local_state_heads": local_state_heads,
        "logits_close": logits_close,
        "logits_cosine": cosine,
        "max_abs_diff": max_abs_diff,
        "generated_equal_reference": generated_equal,
        "all_ranks_generated_equal": ranks_equal,
        "generated_tail": generated[0, -args.max_new_tokens :].detach().cpu().tolist(),
        "decode_backend": backend,
        "load_s": round(load_s, 4),
        "run_s": round(run_s, 4),
        "single_gpu_reference_peak_vram_mb": round(reference_peak_mb, 1),
        "tp_local_peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 1),
        "tp_local_to_reference_peak_ratio": round(
            (torch.cuda.max_memory_allocated(device) / 1024 / 1024) / reference_peak_mb,
            6,
        ) if reference_peak_mb else None,
    }
    if rank == 0:
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
        if args.results:
            result_path = Path(args.results)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
