#!/usr/bin/env python3
"""Small, correctness-first HF/Transformers smoke on Huawei Ascend."""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--backend", choices=["eager", "native_jit", "auto"], default="eager")
    ap.add_argument("--results", default="")
    ap.add_argument("--training-smoke", action="store_true")
    args = ap.parse_args()

    from rwkv7_hf.ascend_runtime import enable_ascend, memory_stats, synchronize
    info = enable_ascend(args.device, backend=args.backend, required=True)
    import torch
    from rwkv7_hf.native_model import NativeRWKV7Cache, NativeRWKV7Config, NativeRWKV7ForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(20260724)
    cfg = NativeRWKV7Config(
        vocab_size=320, hidden_size=16, num_hidden_layers=2, head_dim=4,
        intermediate_size=32, decay_low_rank_dim=4, gate_low_rank_dim=4,
        a_low_rank_dim=4, v_low_rank_dim=4, use_cache=True,
        pad_token_id=0, eos_token_id=0, bos_token_id=1,
    )
    model = NativeRWKV7ForCausalLM(cfg).eval().to(device=args.device, dtype=dtype)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], device=args.device)
    with torch.inference_mode():
        full = model(ids, use_cache=True, logits_to_keep=1)
        chunked = model.rwkv7_prefill_chunks(ids, chunk_size=2, logits_to_keep=1)
        assert torch.isfinite(full.logits).all() and torch.isfinite(chunked.logits).all()
        assert torch.allclose(full.logits.float(), chunked.logits.float(), rtol=8e-3, atol=8e-2)
        cache = chunked.past_key_values
        assert isinstance(cache, NativeRWKV7Cache) and cache.seen_tokens == ids.shape[1]
        selected = cache.clone().select_batch(torch.tensor([1], device=args.device), inplace=True)
        assert selected.get_batch_size() == 1
        step = model(torch.tensor([[13]], device=args.device), past_key_values=selected, use_cache=True)
        assert step.past_key_values.seen_tokens == ids.shape[1] + 1
        generated = model.generate(ids[:1, :3], max_new_tokens=2, do_sample=False, use_cache=True,
                                   pad_token_id=0, eos_token_id=None)
        assert generated.shape == (1, 5)
    training = {}
    if args.training_smoke:
        model.train()
        model.zero_grad(set_to_none=True)
        train_ids = ids[:1, :4]
        train_out = model(train_ids, labels=train_ids, use_cache=False)
        assert train_out.loss is not None and torch.isfinite(train_out.loss).detach().cpu().item()
        train_out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all().detach().cpu().item() for g in grads)
        parameters = [p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad).detach().cpu().item()]
        assert parameters
        parameter = max(parameters, key=lambda p: float(p.grad.detach().abs().max().cpu()))
        before = parameter.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        optimizer.step()
        changed = not torch.equal(before, parameter.detach())
        assert changed
        training = {
            "training_forward_backward": "pass",
            "training_loss": float(train_out.loss.detach().cpu()),
            "finite_gradient_tensors": len(grads),
            "parameter_update": "pass",
        }
        model.eval()
    synchronize(args.device)

    auto = {}
    if args.model:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
        acfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        base = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).eval().to(args.device)
        loaded = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=dtype
        ).eval().to(args.device)
        encoded = tokenizer("Ascend", return_tensors="pt").input_ids[:, :4].to(args.device)
        with torch.inference_mode():
            base_out = base(encoded, use_cache=True)
            out = loaded(encoded, use_cache=True)
            assert torch.isfinite(base_out.last_hidden_state).all()
            assert torch.isfinite(out.logits).all()
            gen = loaded.generate(encoded, max_new_tokens=1, do_sample=False, use_cache=True,
                                  pad_token_id=0, eos_token_id=None)
        with tempfile.TemporaryDirectory(prefix="rwkv7-ascend-roundtrip-") as temp:
            loaded.save_pretrained(temp, safe_serialization=True)
            reloaded = NativeRWKV7ForCausalLM.from_pretrained(temp, torch_dtype=dtype).eval().to(args.device)
            with torch.inference_mode():
                again = reloaded(encoded, use_cache=True)
            assert torch.allclose(out.logits.float(), again.logits.float(), rtol=1e-3, atol=1e-2)
        auto = {
            "auto_config": acfg.__class__.__name__, "auto_model": base.__class__.__name__,
            "auto_causal_lm": loaded.__class__.__name__, "auto_tokenizer": tokenizer.__class__.__name__,
            "save_reload": "pass", "auto_generate": list(gen.shape),
        }

    row = {
        "axis": "huawei_ascend_hf_acceptance", "status": "pass", **info.to_dict(),
        "dtype": args.dtype, "forward": "pass", "generate": "pass",
        "dynamic_cache": "pass", "chunked_prefill": "pass", **training, **auto,
        "memory": memory_stats(args.device),
    }
    text = json.dumps(row, ensure_ascii=False)
    print(text)
    if args.results:
        path = Path(args.results); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
