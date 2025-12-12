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


def test_stacked_rkv_exact_shape_gate_does_not_alias_equal_row_counts() -> None:
    from rwkv7_hf import native_jit

    keys = (
        "RWKV7_NATIVE_PREFILL_STACKED_RKV",
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS",
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS",
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS",
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES",
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS"] = "1"
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS"] = ""
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES"] = ""
        os.environ["RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES"] = "4096x32x1x512"
        assert native_jit._native_prefill_stacked_rkv_enabled(512, 1, 512, 4096, 32)
        assert not native_jit._native_prefill_stacked_rkv_enabled(512, 1, 512, 2560, 24)
        assert not native_jit._native_prefill_stacked_rkv_enabled(512, 4, 128, 4096, 32)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_clampw_scan_exact_shape_policy_and_env_override(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    env_name = "RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN"
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        native_jit,
        "_kernel_policy",
        lambda: types.SimpleNamespace(
            fused_prefill_clampw_scan=False,
            prefill_clampw_scan_model_shapes=((2048, 24, 8, 512),),
        ),
    )
    monkeypatch.setattr(native_jit, "_native_prefill_fused_scan_enabled", lambda: True)
    monkeypatch.setattr(native_jit, "fused_recurrent_scan_clampw", object())
    monkeypatch.setattr(native_jit, "fused_recurrent_scan_clampw_available", lambda: True)

    assert native_jit._native_prefill_fused_clampw_scan_enabled(8, 512, 2048, 24)
    assert not native_jit._native_prefill_fused_clampw_scan_enabled(8, 512, 2560, 24)
    monkeypatch.setenv(env_name, "1")
    assert native_jit._native_prefill_fused_clampw_scan_enabled(8, 128, 2560, 32)
    monkeypatch.setenv(env_name, "0")
    assert not native_jit._native_prefill_fused_clampw_scan_enabled(8, 512, 2048, 24)


def test_native_prefill_scan_passes_model_layers_to_clampw_gate(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    calls = []

    def clampw_enabled(batch_size, prompt_tokens, hidden_size, num_layers):
        calls.append((batch_size, prompt_tokens, hidden_size, num_layers))
        return num_layers == 7

    def clampw_scan(r, w, k, v, kk, a, state, **kwargs):
        return torch.zeros_like(r), state

    monkeypatch.setattr(native_jit, "_native_prefill_fused_clampw_scan_enabled", clampw_enabled)
    monkeypatch.setattr(native_jit, "_native_prefill_scan_block_m", lambda *args: 1)
    monkeypatch.setattr(native_jit, "_native_prefill_scan_num_warps", lambda *args: 1)
    monkeypatch.setattr(native_jit, "fused_recurrent_scan_clampw", clampw_scan)

    B, T, H, N = 1, 2, 2, 2
    token_tensor = torch.zeros(B, T, H * N)
    state = torch.zeros(B, H, N, N)
    out, new_state = native_jit._native_prefill_scan(
        token_tensor,
        token_tensor,
        token_tensor,
        token_tensor,
        token_tensor,
        token_tensor,
        state,
        B,
        T,
        H,
        N,
        w_is_raw=True,
        num_layers=7,
    )

    assert calls == [(B, T, H * N, 7)]
    assert out.shape == token_tensor.shape
    assert new_state is state


def test_self_chunk_exact_model_shape_can_lower_the_generic_token_floor(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    monkeypatch.setattr(native_jit, "self_chunk_rwkv7", object())
    monkeypatch.setattr(native_jit, "self_chunk_rwkv7_available", lambda: True)
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK", "1")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS", "1024")
    monkeypatch.setenv(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES",
        "4096x32x8x512",
    )
    assert native_jit._native_prefill_self_chunk_enabled(512, 64, 8, 4096, 32)
    assert not native_jit._native_prefill_self_chunk_enabled(512, 64, 4, 4096, 32)
    assert not native_jit._native_prefill_self_chunk_enabled(512, 64, 8, 2560, 24)


def test_self_chunk_safe_gate_is_explicitly_tunable(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SAFE_GATE", raising=False)
    assert native_jit._native_prefill_self_chunk_safe_gate()
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SAFE_GATE", "0")
    assert not native_jit._native_prefill_self_chunk_safe_gate()


def test_self_chunk_size_can_be_exact_shape_specific(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    policy = types.SimpleNamespace(
        prefill_self_chunk_size=32,
        prefill_self_chunk_shape_sizes=((2, 512, 16), (2, 2048, 16), (8, 128, 16)),
    )
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: policy)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE", raising=False)
    assert native_jit._native_prefill_self_chunk_size(2, 512) == 16
    assert native_jit._native_prefill_self_chunk_size(2, 2048) == 16
    assert native_jit._native_prefill_self_chunk_size(8, 128) == 16
    assert native_jit._native_prefill_self_chunk_size(4, 2048) == 32
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE", "64")
    assert native_jit._native_prefill_self_chunk_size(2, 512) == 64


def test_sequence_ffn_exact_model_shape_does_not_alias_equal_rows(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    monkeypatch.setattr(native_jit, "fused_sequence_ffn", object())
    monkeypatch.setattr(native_jit, "fused_sequence_ffn_available", lambda: True)
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_FUSED_SEQUENCE_FFN", "1")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MIN_ROWS", "1")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MAX_ROWS", "1")
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_EXTRA_ROWS", "")
    monkeypatch.setenv(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MODEL_SHAPES",
        "4096x32x8x512",
    )
    assert native_jit._native_prefill_fused_sequence_ffn_enabled(4096, 8, 512, 4096, 32)
    assert not native_jit._native_prefill_fused_sequence_ffn_enabled(4096, 2, 2048, 4096, 32)
    assert not native_jit._native_prefill_fused_sequence_ffn_enabled(4096, 8, 512, 2560, 24)


def test_fp16_accum_ffn_key_is_exact_shape_and_explicitly_disableable(monkeypatch) -> None:
    from rwkv7_hf import native_jit

    monkeypatch.setattr(
        native_jit,
        "_kernel_policy",
        lambda: types.SimpleNamespace(
            prefill_fp16_accum_ffn_key_model_shapes=((4096, 32, 8, 128),),
        ),
    )
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY", raising=False)
    monkeypatch.delenv(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_MODEL_SHAPES",
        raising=False,
    )
    assert native_jit._native_prefill_fp16_accum_ffn_key_enabled(
        8, 128, 4096, 32, torch.float16
    )
    assert not native_jit._native_prefill_fp16_accum_ffn_key_enabled(
        1, 1024, 4096, 32, torch.float16
    )
    assert not native_jit._native_prefill_fp16_accum_ffn_key_enabled(
        8, 128, 4096, 32, torch.bfloat16
    )
    monkeypatch.setenv("RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY", "0")
    assert not native_jit._native_prefill_fp16_accum_ffn_key_enabled(
        8, 128, 4096, 32, torch.float16
    )


_TorchModule = torch.nn.Module if torch is not None else object


class _FakeQuantLinear(_TorchModule):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.register_buffer("packed_weight", weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.packed_weight)


def test_graph_quant_linear_promotes_scalar_activation_rank() -> None:
    from rwkv7_hf import native_jit

    operand = _FakeQuantLinear(torch.randn(5, 7, dtype=torch.float32))
    x = torch.randn(7, dtype=torch.float32)
    got = native_jit._graph_linear_call(x, operand)
    expected = torch.nn.functional.linear(x, operand.packed_weight)
    assert got.shape == (5,)
    torch.testing.assert_close(got, expected)


def test_prefill_accepts_quantized_projection_operands() -> None:
    native_jit, model, packs = _build_fake_model_and_packs()
    ids = torch.tensor([[1, 5, 4, 2]], dtype=torch.long)
    quant_packs = []
    quant_indices = {20, 21, 22, 23, 38, 39}
    for pack in packs:
        values = list(pack)
        for index in quant_indices:
            values[index] = _FakeQuantLinear(values[index])
        quant_packs.append(tuple(values))

    with torch.no_grad():
        expected, *_ = native_jit.prefill(model, ids, packs, logits_to_keep=1)
        actual, state, xpa, xpf = native_jit.prefill(model, ids, quant_packs, logits_to_keep=1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert len(state) == len(packs)
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


def test_state_prep_deferred_sigmoid_fallback_matches_materialized() -> None:
    from rwkv7_hf.fused_prefill import fused_prefill_state_prep

    torch.manual_seed(17)
    shape = (2, 3, 8)
    w = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    a_raw = torch.randn(shape)
    v_first = torch.randn(shape)
    v_gate_raw = torch.randn(shape)
    k_k = torch.randn(8)
    k_a = torch.randn(8)
    common = dict(
        v_first=v_first,
        num_heads=2,
        head_dim=4,
        w_transform="log_decay",
        force_fallback=True,
    )
    expected = fused_prefill_state_prep(
        w,
        k,
        v,
        torch.sigmoid(a_raw),
        k_k,
        k_a,
        v_gate=torch.sigmoid(v_gate_raw),
        **common,
    )
    actual = fused_prefill_state_prep(
        w,
        k,
        v,
        a_raw,
        k_k,
        k_a,
        v_gate=v_gate_raw,
        a_is_raw=True,
        v_gate_is_raw=True,
        **common,
    )
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


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
        native_jit.torch.cuda.get_device_capability = lambda *_args: (7, 0)
        assert native_jit._native_prefill_scan_block_m(64, 1) == 16
        assert native_jit._native_prefill_scan_block_m(64, 4) == 32
        native_jit.torch.cuda.get_device_capability = lambda *_args: (7, 5)
        assert native_jit._native_prefill_scan_block_m(64, 1) == 64
        native_jit.torch.cuda.get_device_capability = lambda *_args: (8, 9)
        native_jit.torch.cuda.get_device_name = lambda *_args: "NVIDIA GeForce RTX 4090"
        assert native_jit._native_prefill_scan_block_m(64, 1) == 4
        assert native_jit._native_prefill_scan_block_m(64, 4) == 8
        assert native_jit._native_prefill_scan_block_m(64, 8, 128) == 32
        assert native_jit._native_prefill_scan_block_m(64, 8, 512) == 8
        assert native_jit._native_prefill_scan_block_m(64, 8, 512, 2048) == 32
        assert native_jit._native_prefill_scan_block_m(64, 8, 512, 4096) == 8
        native_jit.torch.cuda.get_device_name = lambda *_args: "NVIDIA GeForce RTX 4070"
        assert native_jit._native_prefill_scan_block_m(64, 1) == 64
        native_jit.torch.cuda.get_device_capability = lambda *_args: (12, 0)
        native_jit.torch.cuda.get_device_name = lambda *_args: "NVIDIA GeForce RTX 5070 Laptop GPU"
        assert native_jit._native_prefill_scan_block_m(64, 1) == 8
        assert native_jit._native_prefill_scan_block_m(64, 2) == 16
        assert native_jit._native_prefill_scan_block_m(64, 4) == 32
        assert native_jit._native_prefill_scan_block_m(64, 8) == 64
        assert native_jit._native_prefill_scan_num_warps(64, 8) == 1
        assert native_jit._native_prefill_scan_num_warps(64, 32) == 1
        assert native_jit._native_prefill_scan_num_warps(64, 64) == 4
        native_jit.torch.cuda.get_device_name = lambda *_args: "NVIDIA GeForce RTX 5090"
        assert native_jit._native_prefill_scan_block_m(64, 8, 512, 2048) == 8
        assert native_jit._native_prefill_scan_block_m(64, 8, 512, 4096) == 64
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


def test_callable_graph_linear_promotes_vector_input() -> None:
    from rwkv7_hf import native_jit

    class RequiresMatrixInput:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            assert value.dim() == 2
            return value * 2

    vector = torch.tensor([1.0, 2.0])
    result = native_jit._graph_linear_call(vector, RequiresMatrixInput())
    assert result.shape == vector.shape
    assert torch.equal(result, vector * 2)


def main() -> int:
    if torch is None:
        print("SKIP native prefill scan test: torch unavailable")
        return 0
    test_prefill_matches_token_loop()
    test_prefill_opt_in_lora_state_prep_fallback_matches_token_loop()
    test_state_prep_deferred_sigmoid_fallback_matches_materialized()
    test_prefill_opt_in_fused_state_scan_fallback_matches_token_loop()
    test_sm70_scan_tile_policy_is_batch_aware_and_exact_arch()
    test_callable_graph_linear_promotes_vector_input()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
