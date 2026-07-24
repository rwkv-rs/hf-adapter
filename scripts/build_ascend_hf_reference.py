#!/usr/bin/env python3
"""Build the pinned, adapter-independent RWKV-7 CPU reference capture."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from rwkv7_hf.ascend_reference_oracle import (
    DEFAULT_ACCEPTANCE_THRESHOLDS,
    NaiveRWKV7Oracle,
    REFERENCE_FORMAT_VERSION,
    RWKV7_7P2_CHECKPOINT_SHA256,
    RWKV7_7P2_TOKENIZER_SHA256,
    SafetensorStore,
    sha256_file,
    state_tensor_map,
    tensor_map_sha256,
    tensor_sha256,
    verify_files,
    verify_fla_checkout,
)

PROMPT = "Hello"
EXPECTED_INPUT_IDS = [33155]
CROSS_BACKEND_VLLM_GREEDY_IDS = [45, 308, 459]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fla-checkout", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--decode-steps", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decode_steps != 3:
        raise ValueError("the common Huawei gate is fixed at max_new_tokens=3")
    fla = verify_fla_checkout(args.fla_checkout)
    checkpoint_hashes = verify_files(args.model, RWKV7_7P2_CHECKPOINT_SHA256)
    tokenizer_hashes = verify_files(args.model, RWKV7_7P2_TOKENIZER_SHA256)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    if input_ids.tolist() != [EXPECTED_INPUT_IDS]:
        raise RuntimeError(
            f"tokenizer drift for {PROMPT!r}: {input_ids.tolist()} != {[EXPECTED_INPUT_IDS]}"
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tensors: dict[str, torch.Tensor] = {
        "b1.input.token_ids": input_ids.to(torch.int64),
        "b1.input.attention_mask": torch.ones_like(input_ids, dtype=torch.int64),
    }
    with SafetensorStore(args.model) as store:
        oracle = NaiveRWKV7Oracle(store, dtype=dtype)
        prefill = oracle.forward(input_ids)
        tensors["b1.prefill.logits"] = prefill.logits.contiguous()
        tensors.update(state_tensor_map(prefill.state.clone(), "b1.prefill.state"))
        state = prefill.state
        logits = prefill.logits[:, -1]
        generated = []
        for step in range(args.decode_steps):
            token = logits.argmax(dim=-1).to(torch.int64)
            generated.append(token)
            decoded = oracle.forward(token[:, None], state=state)
            state = decoded.state
            logits = decoded.logits[:, -1]
            tensors[f"b1.decode.{step:02d}.input_token_ids"] = token[:, None]
            tensors[f"b1.decode.{step:02d}.logits"] = logits.contiguous()
        generated_ids = torch.stack(generated, dim=1)
        tensors["b1.greedy.token_ids"] = generated_ids
        tensors.update(state_tensor_map(state.clone(), "b1.final.state"))
        config_repair = oracle.config_repair
    floating = [value for value in tensors.values() if value.is_floating_point()]
    if not all(bool(torch.isfinite(value).all()) for value in floating):
        raise RuntimeError("reference capture contains non-finite values")
    args.output_tensors.parent.mkdir(parents=True, exist_ok=True)
    ordered = {name: tensors[name].detach().cpu().contiguous() for name in sorted(tensors)}
    save_file(
        ordered,
        args.output_tensors,
        metadata={
            "format": "rwkv7-ascend-independent-reference-v1",
            "dtype": args.dtype,
        },
    )
    tensor_hashes = {name: tensor_sha256(value) for name, value in ordered.items()}
    logits_names = [name for name in ordered if ".logits" in name]
    state_names = [name for name in ordered if ".state." in name]
    observed_ids = generated_ids.reshape(-1).tolist()
    report = {
        "format_version": REFERENCE_FORMAT_VERSION,
        "axis": "huawei_ascend_hf_independent_cpu_oracle",
        "status": "reference_generated",
        "reference_backend": "pinned_fla_formula_naive_pytorch_cpu",
        "candidate_adapter_forward_called": False,
        "fla_source": fla,
        "checkpoint_files_sha256": checkpoint_hashes,
        "tokenizer_files_sha256": tokenizer_hashes,
        "config_repair": config_repair,
        "scenario": {
            "name": "common_hello_b1_greedy3",
            "batch_size": 1,
            "prompt": PROMPT,
            "input_token_ids": EXPECTED_INPUT_IDS,
            "attention_mask": [1],
            "temperature": 0.0,
            "max_new_tokens": 3,
            "ignore_eos": True,
            "oracle_greedy_token_ids": observed_ids,
            "cross_backend_vllm_greedy_token_ids": CROSS_BACKEND_VLLM_GREEDY_IDS,
            "cross_backend_vllm_exact": observed_ids == CROSS_BACKEND_VLLM_GREEDY_IDS,
        },
        "runtime": {
            "device": "cpu",
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "acceptance_thresholds": DEFAULT_ACCEPTANCE_THRESHOLDS,
        "capture": {
            "tensor_file": args.output_tensors.name,
            "tensor_file_sha256": sha256_file(args.output_tensors),
            "capture_sha256": tensor_map_sha256(ordered),
            "logits_capture_sha256": tensor_map_sha256(ordered, logits_names),
            "state_capture_sha256": tensor_map_sha256(ordered, state_names),
            "tensor_count": len(ordered),
            "tensor_sha256": tensor_hashes,
        },
        "pending_npu_gates": [
            "hf_npu_b1_prefill_and_three_decode_steps",
            "hf_npu_b2_prefill_and_multi_step_decode",
            "hf_npu_b2_ragged_attention_mask",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
