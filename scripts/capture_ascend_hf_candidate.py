#!/usr/bin/env python3
"""Capture canonical HF/Ascend logits and recurrent state for oracle comparison."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, __version__ as transformers_version

from rwkv7_hf import enable_ascend
from rwkv7_hf.ascend_reference_oracle import sha256_file, tensor_map_sha256, tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="repaired native HF view")
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path, required=True)
    return parser.parse_args()


def _cache_tensors(cache, prefix: str) -> dict[str, torch.Tensor]:
    recurrent, attn_shift, ffn_shift, v_first = tuple(cache)
    batch = int(v_first.shape[0])
    seen = int(cache.seen_tokens)
    tensors = {
        f"{prefix}.v_first": v_first.detach().cpu().contiguous(),
        f"{prefix}.valid_tokens": torch.full((batch,), seen, dtype=torch.int64),
        f"{prefix}.processed_width": torch.tensor(seen, dtype=torch.int64),
    }
    for layer, value in enumerate(recurrent):
        tensors[f"{prefix}.recurrent.{layer:02d}"] = value.detach().cpu().contiguous()
    for layer, value in enumerate(attn_shift):
        tensors[f"{prefix}.attn_shift.{layer:02d}"] = value.detach().cpu().contiguous()
    for layer, value in enumerate(ffn_shift):
        tensors[f"{prefix}.ffn_shift.{layer:02d}"] = value.detach().cpu().contiguous()
    return tensors


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
    scenario = reference["scenario"]
    if scenario.get("name") != "common_hello_b1_greedy3":
        raise ValueError("this capture entry point requires the fixed B1 Hello scenario")
    info = enable_ascend("npu:0", backend="eager")
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    load_cpu_s = time.perf_counter() - load_start
    move_start = time.perf_counter()
    model.to("npu:0")
    torch.npu.synchronize()
    move_npu_s = time.perf_counter() - move_start
    torch.npu.reset_peak_memory_stats()
    input_ids = torch.tensor([scenario["input_token_ids"]], device="npu:0", dtype=torch.long)
    attention_mask = torch.tensor([scenario["attention_mask"]], device="npu:0", dtype=torch.long)
    tensors: dict[str, torch.Tensor] = {
        "b1.input.token_ids": input_ids.cpu(),
        "b1.input.attention_mask": attention_mask.cpu(),
    }
    timings = {}
    with torch.inference_mode():
        start = time.perf_counter()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        torch.npu.synchronize()
        timings["prefill_s"] = time.perf_counter() - start
        tensors["b1.prefill.logits"] = output.logits.detach().cpu().contiguous()
        tensors.update(_cache_tensors(output.past_key_values, "b1.prefill.state"))
        cache = output.past_key_values
        logits = output.logits[:, -1]
        generated = []
        for step in range(int(scenario["max_new_tokens"])):
            token = logits.argmax(dim=-1).to(torch.int64)
            generated.append(token.detach().cpu())
            start = time.perf_counter()
            output = model(
                input_ids=token[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            torch.npu.synchronize()
            timings[f"decode_{step:02d}_s"] = time.perf_counter() - start
            cache = output.past_key_values
            logits = output.logits[:, -1]
            tensors[f"b1.decode.{step:02d}.input_token_ids"] = token[:, None].cpu()
            tensors[f"b1.decode.{step:02d}.logits"] = logits.detach().cpu().contiguous()
        generated_ids = torch.stack(generated, dim=1)
        tensors["b1.greedy.token_ids"] = generated_ids
        tensors.update(_cache_tensors(cache, "b1.final.state"))
    ordered = {name: tensors[name].contiguous() for name in sorted(tensors)}
    args.output_tensors.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        ordered,
        args.output_tensors,
        metadata={"format": "rwkv7-ascend-hf-candidate-v1", "dtype": "bf16"},
    )
    observed_ids = generated_ids.reshape(-1).tolist()
    capture = {
        "tensor_file": args.output_tensors.name,
        "tensor_file_sha256": sha256_file(args.output_tensors),
        "capture_sha256": tensor_map_sha256(ordered),
        "tensor_count": len(ordered),
        "tensor_sha256": {name: tensor_sha256(value) for name, value in ordered.items()},
    }
    report = {
        "format_version": reference["format_version"],
        "axis": "huawei_ascend_hf_candidate_capture",
        "status": "candidate_captured",
        "fla_source": reference["fla_source"],
        "checkpoint_files_sha256": reference["checkpoint_files_sha256"],
        "tokenizer_files_sha256": reference["tokenizer_files_sha256"],
        "config_repair": reference["config_repair"],
        "scenario": scenario,
        "measurement": {
            "observed_greedy_token_ids": observed_ids,
            "oracle_greedy_exact": observed_ids == scenario["oracle_greedy_token_ids"],
            "timings": timings,
            "memory": {
                "allocated_bytes": int(torch.npu.memory_allocated()),
                "reserved_bytes": int(torch.npu.memory_reserved()),
                "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
            },
        },
        "runtime": {
            **info.to_dict(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "dtype": "bfloat16",
            "backend": "eager",
            "load_cpu_s": load_cpu_s,
            "move_npu_s": move_npu_s,
        },
        "capture": capture,
        "command": "scripts/capture_ascend_hf_candidate.py --model <native-view> --reference-json <reference.json> --output-json <candidate.json> --output-tensors <candidate.safetensors>",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
