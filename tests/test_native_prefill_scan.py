#!/usr/bin/env python3
# coding=utf-8
"""Unit coverage for the native layer-wise prefill path.

The test builds a tiny synthetic RWKV-7-style pack and checks that the new
layer-wise `native_jit.prefill` path matches the existing token-by-token
`native_jit.forward` reference.  CUDA/Triton-specific fused scan performance is
validated by benchmarks; this CPU-friendly shape test keeps the math contract
stable on normal CI.
"""
from __future__ import annotations

import os
import types

try:
    import torch
except Exception:  # pragma: no cover - local lightweight environments
    torch = None  # type: ignore[assignment]


def _linear_weight(out_features: int, in_features: int, *, scale: float = 0.02):
    return torch.randn(out_features, in_features, dtype=torch.float32) * scale


def _build_fake_model_and_packs():
    from rwkv7_hf import native_jit

    torch.manual_seed(7)
    H, N = 2, 4
    hidden = H * N
    vocab = 13
    rank = 3
    layers = 2

    emb = torch.randn(vocab, hidden, dtype=torch.float32) * 0.03
    norm_w = torch.ones(hidden, dtype=torch.float32)
    norm_b = torch.zeros(hidden, dtype=torch.float32)
    head_w = _linear_weight(vocab, hidden)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(head_w)

    fake_layers = [
        types.SimpleNamespace(attn=types.SimpleNamespace(num_heads=H, head_dim=N, hidden_size=hidden))
        for _ in range(layers)
    ]
    base = types.SimpleNamespace(
        embeddings=types.SimpleNamespace(weight=emb),
        norm=types.SimpleNamespace(weight=norm_w, bias=norm_b),
        layers=fake_layers,
    )
    model = types.SimpleNamespace(
        model=base,
        lm_head=lm_head,
    )

    packs = []
    for i in range(layers):
        has_pre = 1 if i == 0 else 0
        pre_w = torch.ones(hidden, dtype=torch.float32)
        pre_b = torch.zeros(hidden, dtype=torch.float32)
        an_w = torch.ones(hidden, dtype=torch.float32) + torch.randn(hidden) * 0.01
        an_b = torch.randn(hidden) * 0.01
        fn_w = torch.ones(hidden, dtype=torch.float32) + torch.randn(hidden) * 0.01
        fn_b = torch.randn(hidden) * 0.01
        x_mix = [torch.rand(hidden, dtype=torch.float32) for _ in range(6)]
        k_k = torch.randn(hidden, dtype=torch.float32) * 0.1
        k_a = torch.randn(hidden, dtype=torch.float32) * 0.1
        r_k = torch.randn(H, N, dtype=torch.float32) * 0.1
        Rw = _linear_weight(hidden, hidden)
        Kw = _linear_weight(hidden, hidden)
        Vw = _linear_weight(hidden, hidden)
        Ow = _linear_weight(hidden, hidden)
        w1 = _linear_weight(rank, hidden)
        w2 = _linear_weight(hidden, rank)
        w0 = torch.randn(hidden, dtype=torch.float32) * 0.01
        a1 = _linear_weight(rank, hidden)
        a2 = _linear_weight(hidden, rank)
        a0 = torch.randn(hidden, dtype=torch.float32) * 0.01
        v1 = _linear_weight(rank, hidden)
        v2 = _linear_weight(hidden, rank)
        v0 = torch.randn(hidden, dtype=torch.float32) * 0.01
        g1 = _linear_weight(rank, hidden)
        g2 = _linear_weight(hidden, rank)
        gn_w = torch.ones(hidden, dtype=torch.float32) + torch.randn(hidden) * 0.01
        gn_b = torch.randn(hidden, dtype=torch.float32) * 0.01
        fx_k = torch.rand(hidden, dtype=torch.float32)
        fK = _linear_weight(hidden, hidden)
        fV = _linear_weight(hidden, hidden)
        RKVw = torch.stack((Rw.t(), Kw.t(), Vw.t())).contiguous()
        packs.append(
            (
                i,
                H,
                N,
                float(N * 1e-5),
                has_pre,
                pre_w,
                pre_b,
                an_w,
                an_b,
                fn_w,
                fn_b,
                *x_mix,
                k_k,
                k_a,
                r_k,
                Rw,
                Kw,
                Vw,
                Ow,
                w1,
                w2,
                w0,
                a1,
                a2,
                a0,
                v1,
                v2,
                v0,
                g1,
                g2,
                gn_w,
                gn_b,
                fx_k,
                fK,
                fV,
                RKVw,
            )
        )
    return native_jit, model, packs


def test_prefill_matches_token_loop() -> None:
    native_jit, model, packs = _build_fake_model_and_packs()
    ids = torch.tensor([[1, 5, 4, 2]], dtype=torch.long)
    with torch.no_grad():
        ref = native_jit.forward(model, ids, packs).float().view(1, -1)
        logits, state, xpa, xpf = native_jit.prefill(model, ids, packs, logits_to_keep=1)
    got = logits[:, -1, :].float()
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5), (got - ref).abs().max()
    assert len(state) == len(packs)
    assert state[0].shape == (1, 2, 4, 4)
    assert xpa[0].shape == (1, 8)
    assert xpf[0].shape == (1, 8)


def test_prefill_opt_in_lora_state_prep_fallback_matches_token_loop() -> None:
    native_jit, model, packs = _build_fake_model_and_packs()
    ids = torch.tensor([[1, 5, 4, 2]], dtype=torch.long)
    old_env = {
        key: os.environ.get(key)
        for key in (
            "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP",
            "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT",
            "RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA",
            "RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M",
        )
    }
    old_state_avail = native_jit.fused_prefill_state_prep_available
    old_output_avail = native_jit.fused_attn_output_prepare_available
    old_wavg_avail = native_jit.fused_wavg_lora_available
    try:
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_OUTPUT"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M"] = "999"
        native_jit.fused_prefill_state_prep_available = lambda: True
        native_jit.fused_attn_output_prepare_available = lambda: True
        native_jit.fused_wavg_lora_available = lambda: True
        with torch.no_grad():
            ref = native_jit.forward(model, ids, packs).float().view(1, -1)
            logits, state, xpa, xpf = native_jit.prefill(model, ids, packs, logits_to_keep=1)
    finally:
        native_jit.fused_prefill_state_prep_available = old_state_avail
        native_jit.fused_attn_output_prepare_available = old_output_avail
        native_jit.fused_wavg_lora_available = old_wavg_avail
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    got = logits[:, -1, :].float()
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5), (got - ref).abs().max()
    assert len(state) == len(packs)
    assert state[1].shape == (1, 2, 4, 4)
    assert xpa[1].shape == (1, 8)
    assert xpf[1].shape == (1, 8)


def test_prefill_opt_in_fused_state_scan_fallback_matches_token_loop() -> None:
    native_jit, model, packs = _build_fake_model_and_packs()
    ids = torch.tensor([[1, 5, 4, 2]], dtype=torch.long)
    old_env = {
        key: os.environ.get(key)
        for key in (
            "RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN",
            "RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH",
            "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT",
            "RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT",
        )
    }
    old_state_scan_avail = native_jit.fused_recurrent_scan_state_prep_available
    old_output_avail = native_jit.fused_attn_output_prepare_available
    try:
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_FUSED_OUTPUT"] = "1"
        # The state-scan path intentionally stays separate from the older
        # scan+output fusion probe, which consumes already-prepared W/K/V/KK.
        os.environ.pop("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", None)
        native_jit.fused_recurrent_scan_state_prep_available = lambda: True
        native_jit.fused_attn_output_prepare_available = lambda: True
        assert native_jit._native_prefill_fused_state_scan_enabled(1)
        assert not native_jit._native_prefill_fused_state_scan_enabled(2)
        with torch.no_grad():
            ref = native_jit.forward(model, ids, packs).float().view(1, -1)
            logits, state, xpa, xpf = native_jit.prefill(model, ids, packs, logits_to_keep=1)
    finally:
        native_jit.fused_recurrent_scan_state_prep_available = old_state_scan_avail
        native_jit.fused_attn_output_prepare_available = old_output_avail
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    got = logits[:, -1, :].float()
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5), (got - ref).abs().max()
    assert len(state) == len(packs)
    assert state[1].shape == (1, 2, 4, 4)
    assert xpa[1].shape == (1, 8)
    assert xpf[1].shape == (1, 8)


def test_sm70_scan_tile_policy_is_batch_aware_and_exact_arch() -> None:
    from rwkv7_hf import native_jit

    key = "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M"
    old_env = os.environ.get(key)
    old_available = native_jit.torch.cuda.is_available
    old_capability = native_jit.torch.cuda.get_device_capability
    old_device_name = native_jit.torch.cuda.get_device_name
    try:
        os.environ.pop(key, None)
        native_jit.torch.cuda.is_available = lambda: True
        native_jit.torch.cuda.get_device_capability = lambda: (7, 0)
        assert native_jit._native_prefill_scan_block_m(64, 1) == 16
        assert native_jit._native_prefill_scan_block_m(64, 4) == 32
        native_jit.torch.cuda.get_device_capability = lambda: (7, 5)
        assert native_jit._native_prefill_scan_block_m(64, 1) == 64
        native_jit.torch.cuda.get_device_capability = lambda: (8, 9)
        native_jit.torch.cuda.get_device_name = lambda: "NVIDIA GeForce RTX 4090"
        assert native_jit._native_prefill_scan_block_m(64, 1) == 4
        assert native_jit._native_prefill_scan_block_m(64, 4) == 8
        native_jit.torch.cuda.get_device_name = lambda: "NVIDIA GeForce RTX 4070"
        assert native_jit._native_prefill_scan_block_m(64, 1) == 64
        os.environ[key] = "8"
        assert native_jit._native_prefill_scan_block_m(64, 4) == 8
    finally:
        native_jit.torch.cuda.is_available = old_available
        native_jit.torch.cuda.get_device_capability = old_capability
        native_jit.torch.cuda.get_device_name = old_device_name
        if old_env is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_env


def main() -> int:
    if torch is None:
        print("SKIP native prefill scan test: torch unavailable")
        return 0
    test_prefill_matches_token_loop()
    test_prefill_opt_in_lora_state_prep_fallback_matches_token_loop()
    test_prefill_opt_in_fused_state_scan_fallback_matches_token_loop()
    test_sm70_scan_tile_policy_is_batch_aware_and_exact_arch()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
