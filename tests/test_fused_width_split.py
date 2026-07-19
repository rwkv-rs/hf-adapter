from __future__ import annotations

import torch


def test_fused_output_project_supports_residual_attention_width_split() -> None:
    from rwkv7_hf.fused_output import fused_attn_output_project

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(29)
    batch, heads, head_dim, residual_hidden = 3, 2, 32, 48
    attention_hidden = heads * head_dim
    recurrent = torch.randn(batch, attention_hidden, device=device, dtype=dtype) * 0.1
    r = torch.randn(batch, heads, head_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn_like(r)
    v = torch.randn_like(r)
    g = torch.randn_like(recurrent)
    r_k = torch.randn(heads, head_dim, device=device, dtype=dtype) * 0.1
    norm_weight = torch.randn(attention_hidden, device=device, dtype=dtype) * 0.1
    norm_bias = torch.randn_like(norm_weight) * 0.1
    output_weight = torch.randn(
        residual_hidden,
        attention_hidden,
        device=device,
        dtype=dtype,
    ) * 0.01
    args = (recurrent, r, k, v, g, r_k, norm_weight, norm_bias, output_weight)
    kwargs = {
        "num_heads": heads,
        "head_dim": head_dim,
        "head_v_dim": head_dim,
        "eps": head_dim * 1e-5,
    }
    reference = fused_attn_output_project(*args, **kwargs, force_fallback=True)
    actual = fused_attn_output_project(*args, **kwargs)
    assert actual.shape == (batch, residual_hidden)
    torch.testing.assert_close(actual.float(), reference.float(), atol=4e-3, rtol=4e-3)


def test_fused_wavg_lora_supports_residual_attention_width_split() -> None:
    from rwkv7_hf.fused_lora import fused_wavg_lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(31)
    batch, residual_hidden, attention_hidden = 3, 32, 48
    ranks = (8, 6, 10, 4)
    inputs = [
        torch.randn(batch, residual_hidden, device=device, dtype=dtype) * 0.1
        for _ in range(4)
    ]
    down = [
        torch.randn(rank, residual_hidden, device=device, dtype=dtype) * 0.01
        for rank in ranks
    ]
    up = [
        torch.randn(attention_hidden, rank, device=device, dtype=dtype) * 0.01
        for rank in ranks
    ]
    bias = [
        torch.randn(attention_hidden, device=device, dtype=dtype) * 0.01
        for _ in ranks
    ]
    args = (*inputs, *down, *up, *bias)
    reference = fused_wavg_lora(*args, force_fallback=True)
    actual = fused_wavg_lora(*args, block_m=32, block_r=16, block_k=32)
    for reference_tensor, actual_tensor in zip(reference, actual, strict=True):
        assert actual_tensor.shape == (batch, attention_hidden)
        torch.testing.assert_close(
            actual_tensor.float(),
            reference_tensor.float(),
            atol=4e-3,
            rtol=4e-3,
        )
