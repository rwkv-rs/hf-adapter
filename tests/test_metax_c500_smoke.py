#!/usr/bin/env python3
"""Deterministic native-HF acceptance for an exact MetaX C500."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any


def _cosine(left, right) -> float:
    import torch.nn.functional as F

    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)
    )


def _tiny_model():
    import torch
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(20260726)
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=127,
            hidden_size=32,
            num_hidden_layers=2,
            head_dim=8,
            intermediate_size=64,
            decay_low_rank_dim=8,
            gate_low_rank_dim=8,
            a_low_rank_dim=8,
            v_low_rank_dim=8,
            use_cache=True,
        )
    )


def run_tiny(device: str) -> list[dict[str, Any]]:
    import torch

    base = _tiny_model().eval()
    state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    ids_cpu = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        oracle = base(ids_cpu, use_cache=True).logits.detach().float().cpu()
    rows: list[dict[str, Any]] = []
    for label, dtype in (
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        model = _tiny_model()
        model.load_state_dict(state)
        model.to(device=device, dtype=dtype).eval()
        ids = ids_cpu.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            model(ids, use_cache=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            full = model(ids, use_cache=True)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            chunked = model.rwkv7_prefill_chunks(ids, chunk_size=3, logits_to_keep=1)
            selected = full.past_key_values.clone().select_batch(
                torch.tensor([1], device=device), inplace=True
            )
            generated = model.generate(
                ids[:1, :4],
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
            )
        forward_cosine = _cosine(full.logits.detach().cpu(), oracle)
        chunk_cosine = _cosine(
            chunked.logits[:, -1].detach().cpu(),
            full.logits[:, -1].detach().cpu(),
        )

        model.train()
        model.zero_grad(set_to_none=True)
        trained = model(ids, labels=ids, use_cache=False)
        trained.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        threshold = 0.99999 if label == "fp32" else 0.999
        passed = bool(
            torch.isfinite(full.logits).all()
            and forward_cosine >= threshold
            and chunk_cosine >= 0.999
            and selected.get_batch_size() == 1
            and generated.shape == (1, 8)
            and torch.isfinite(trained.loss)
            and gradients
            and all(torch.isfinite(gradient).all() for gradient in gradients)
        )
        rows.append(
            {
                "dtype": label,
                "status": "pass" if passed else "fail",
                "forward_ms": elapsed_ms,
                "forward_cosine_vs_cpu_fp32": forward_cosine,
                "chunked_prefill_cosine": chunk_cosine,
                "cache_batch_select_size": selected.get_batch_size(),
                "generated_token_ids": generated[0, 4:].detach().cpu().tolist(),
                "training_loss": float(trained.loss.detach().float().cpu()),
                "finite_gradient_tensors": len(gradients),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        del model, full, chunked, selected, generated, trained, gradients
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def run_real_checkpoint(model_path: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoded = tokenizer("User: Hello!\n\nAssistant:", return_tensors="pt")
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        cpu_last = cpu_model(**encoded, use_cache=True).logits[:, -1].detach().cpu()
    del cpu_model
    gc.collect()

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.float16
        )
        .eval()
        .to(device)
    )
    inputs = {name: value.to(device) for name, value in encoded.items()}
    ids = inputs["input_ids"]
    with torch.inference_mode():
        full = model(**inputs, use_cache=True)
        chunked = model.rwkv7_prefill_chunks(
            ids,
            chunk_size=3,
            attention_mask=inputs.get("attention_mask"),
            logits_to_keep=1,
        )
        batch_ids = ids.repeat(8, 1)
        batched = model(batch_ids, use_cache=True)
        selected = batched.past_key_values.select_batch(
            torch.tensor([7, 0, 3], device=device), inplace=False
        )
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
        )
    gpu_last = full.logits[:, -1].detach().float().cpu()
    chunk_last = chunked.logits[:, -1].detach().float().cpu()
    with tempfile.TemporaryDirectory(prefix="rwkv7-metax-reload-") as directory:
        model.save_pretrained(directory, safe_serialization=True)
        reloaded = (
            AutoModelForCausalLM.from_pretrained(
                directory, trust_remote_code=True, dtype=torch.float16
            )
            .eval()
            .to(device)
        )
        with torch.inference_mode():
            reload_last = reloaded(**inputs, use_cache=True).logits[:, -1].float().cpu()
    cpu_cosine = _cosine(gpu_last, cpu_last)
    chunk_cosine = _cosine(chunk_last, gpu_last)
    reload_cosine = _cosine(reload_last, gpu_last)
    passed = bool(
        torch.isfinite(gpu_last).all()
        and cpu_cosine >= 0.999
        and chunk_cosine >= 0.999
        and reload_cosine >= 0.999
        and selected.get_batch_size() == 3
    )
    return {
        "status": "pass" if passed else "fail",
        "model": Path(model_path).name,
        "dtype": "fp16",
        "forward_cosine_vs_cpu_fp32": cpu_cosine,
        "chunked_prefill_cosine": chunk_cosine,
        "save_reload_cosine": reload_cosine,
        "selected_cache_batch_size": selected.get_batch_size(),
        "generated_token_ids": generated[0, ids.shape[1] :].detach().cpu().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    from rwkv7_hf.metax_runtime import SOURCE_COMMIT, enable_metax

    info = enable_metax("cuda:0", required=True)
    tiny = run_tiny(info.device)
    real = run_real_checkpoint(args.model, info.device) if args.model else None
    passed = all(row["status"] == "pass" for row in tiny) and (
        real is None or real["status"] == "pass"
    )
    result = {
        "schema": "rwkv7-metax-c500-hf-current-smoke-v1",
        "status": "pass" if passed else "fail",
        "source_evidence_commit": SOURCE_COMMIT,
        "runtime": info.to_dict(),
        "tiny_rows": tiny,
        "real_checkpoint": real,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
