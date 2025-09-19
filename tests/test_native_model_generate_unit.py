#!/usr/bin/env python3
# coding=utf-8
"""CPU/no-CUDA generation smoke for the experimental native RWKV-7 CausalLM.

This uses a tiny random config and does not require converted weights, FLA, CUDA,
or external model files. It guards the CPU fallback GenerationMixin path that
upstream Transformers / AMD / no-GPU contributors depend on.
"""
from __future__ import annotations

import types

import torch

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(2026)
    cfg = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(cfg).eval()


def main() -> int:
    model = build_tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    calls: list[tuple[tuple[int, ...], bool, bool]] = []
    original_forward = model.forward

    def counted_forward(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        calls.append((tuple(input_ids.shape), past_key_values is not None, bool(use_cache)))
        return original_forward(input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

    model.forward = types.MethodType(counted_forward, model)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )

    assert out.shape == (1, 7), tuple(out.shape)
    assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
    assert calls, "generate should call forward"
    assert calls[0] == ((1, input_ids.shape[1]), False, True), calls
    assert all(shape == (1, 1) and has_cache and use_cache for shape, has_cache, use_cache in calls[1:]), calls
    print("NATIVE CPU GENERATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
