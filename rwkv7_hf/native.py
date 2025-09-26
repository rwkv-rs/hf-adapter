# coding=utf-8
"""Native RWKV-7 forward — port of the official TMix_one / CMix_one per-token
math (BlinkDL/RWKV-LM), driven by the FLA-loaded weights. Same math as FLA
(verified equal at fp32), but written as tight per-layer functions so the whole
step can later be torch.jit.script-ed to remove inter-op Python dispatch — the
decode bottleneck (see bench/profile_decode.py).

Run `python -m rwkv7_hf.native <hf_dir>` to check correctness vs FLA.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EXP_HALF = 0.606531  # = exp(-0.5), RWKV-7 decay base


def attn_step(layer, layer_id: int, x: torch.Tensor, x_prev: torch.Tensor,
              v_first: torch.Tensor, state: torch.Tensor):
    """Port of RWKV_x070_TMix_one. x,x_prev:[hidden] state:[H,N,N] v_first:[hidden].
    Returns (out:[hidden], x_new_prev, new_state:[H,N,N], new_v_first)."""
    H, N = layer.num_heads, layer.head_dim
    xx = x_prev - x
    xr = x + xx * layer.x_r.reshape(-1)
    xw = x + xx * layer.x_w.reshape(-1)
    xk = x + xx * layer.x_k.reshape(-1)
    xv = x + xx * layer.x_v.reshape(-1)
    xa = x + xx * layer.x_a.reshape(-1)
    xg = x + xx * layer.x_g.reshape(-1)

    r = layer.r_proj(xr)
    w = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw)))
    k = layer.k_proj(xk)
    v = layer.v_proj(xv)
    a = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xa)))
    g = layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xg)))

    kk = F.normalize((k * layer.k_k).view(H, N), dim=-1, p=2).view(H * N)
    k = k * (1 + (a - 1) * layer.k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(
            layer.v_lora.lora[2](layer.v_lora.lora[0](xv)))
    w = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(x.dtype) @ r.view(H, N, 1)
    out = out.view(H * N)
    out = F.group_norm(out.view(1, H * N), num_groups=H,
                       weight=layer.g_norm.weight, bias=layer.g_norm.bias,
                       eps=N * 1e-5).view(H * N)
    sk = (r.view(H, N) * k.view(H, N) * layer.r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(H * N)
    out = layer.o_proj(out * g)
    return out, x, state, v_first


def ffn_step(layer, x: torch.Tensor, x_prev: torch.Tensor):
    """Port of RWKV_x070_CMix_one. Returns (out, x_new_prev)."""
    xx = x_prev - x
    k = x + xx * layer.x_k
    k = torch.relu(layer.key(k)) ** 2
    return layer.value(k), x


def attn_step_batched(layer, layer_id: int, x: torch.Tensor, x_prev: torch.Tensor,
                      v_first: torch.Tensor, state: torch.Tensor):
    """Batched RWKV_x070_TMix_one.

    x/x_prev/v_first: [B, hidden], state: [B, H, N, N].  This is a
    correctness-first pure PyTorch path for the experimental native model; it
    intentionally mirrors :func:`attn_step` and avoids FLA-specific runtime
    dependencies.
    """
    B = int(x.shape[0])
    H, N = layer.num_heads, layer.head_dim
    hidden = H * N
    xx = x_prev - x
    xr = x + xx * layer.x_r.reshape(1, hidden)
    xw = x + xx * layer.x_w.reshape(1, hidden)
    xk = x + xx * layer.x_k.reshape(1, hidden)
    xv = x + xx * layer.x_v.reshape(1, hidden)
    xa = x + xx * layer.x_a.reshape(1, hidden)
    xg = x + xx * layer.x_g.reshape(1, hidden)

    r = layer.r_proj(xr)
    w = layer.w_lora.lora[2](torch.tanh(layer.w_lora.lora[0](xw)))
    k = layer.k_proj(xk)
    v = layer.v_proj(xv)
    a = torch.sigmoid(layer.a_lora.lora[2](layer.a_lora.lora[0](xa)))
    g = layer.g_lora.lora[2](torch.sigmoid(layer.g_lora.lora[0](xg)))

    kk = F.normalize((k * layer.k_k.reshape(1, hidden)).view(B, H, N), dim=-1, p=2).view(B, hidden)
    k = k * (1 + (a - 1) * layer.k_a.reshape(1, hidden))
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(
            layer.v_lora.lora[2](layer.v_lora.lora[0](xv)))
    w = torch.exp(-EXP_HALF * torch.sigmoid(w.float()))

    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(x.dtype) @ r.view(B, H, N, 1)
    out = out.view(B, hidden)
    out = F.group_norm(out, num_groups=H,
                       weight=layer.g_norm.weight, bias=layer.g_norm.bias,
                       eps=N * 1e-5)
    sk = (r.view(B, H, N) * k.view(B, H, N) * layer.r_k.reshape(1, H, N)).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, hidden)
    out = layer.o_proj(out * g)
    return out, x, state, v_first


def ffn_step_batched(layer, x: torch.Tensor, x_prev: torch.Tensor):
    """Batched RWKV_x070_CMix_one. Returns (out, x_new_prev)."""
    xx = x_prev - x
    k = x + xx * layer.x_k.reshape(1, -1)
    k = torch.relu(layer.key(k)) ** 2
    return layer.value(k), x


def _init_state_batched(model, batch_size: int, device, dtype):
    base = model.model
    n = len(base.layers)
    H = base.layers[0].attn.num_heads
    N = base.layers[0].attn.head_dim
    hid = base.layers[0].attn.hidden_size
    B = int(batch_size)
    state = [torch.zeros(B, H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(B, hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(B, hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(B, hid, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def _step_token_batched(model, x, state, xpa, xpf, v_first):
    for i, layer in enumerate(model.model.layers):
        attn = layer.attn
        residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
        h = layer.attn_norm(residual)
        # Call the attention / FFN modules instead of passing them directly to
        # the functional helpers.  DeepSpeed ZeRO-3 uses module pre-forward
        # hooks to gather partitioned parameters; bypassing ``Module.__call__``
        # leaves raw parameters such as ``x_r``, ``r_k``, ``g_norm.weight`` and
        # ``ffn.x_k`` sharded during backward.
        a, xpa[i], state[i], v_first = attn(h, xpa[i], v_first, state[i])
        x = residual + a
        residual = x
        h2 = layer.ffn_norm(x)
        f, xpf[i] = layer.ffn(h2, xpf[i])
        x = residual + f
    return x, state, xpa, xpf, v_first


def native_forward_batched(model, input_ids: torch.Tensor):
    """Sequential pure-PyTorch native forward for input_ids shaped [B, T]."""
    if input_ids.dim() != 2:
        raise ValueError("native_forward_batched expects input_ids shaped [batch, seq]")
    base = model.model
    state, xpa, xpf, v_first = _init_state_batched(model, input_ids.shape[0], input_ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(input_ids.shape[1]):
        x = F.embedding(input_ids[:, t], base.embeddings.weight)
        x, state, xpa, xpf, v_first = _step_token_batched(model, x, state, xpa, xpf, v_first)
    if x is None:
        raise ValueError("native_forward_batched requires at least one token")
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight)


def native_prefill_batched(model, input_ids):
    """Batched prefill returning (logits, state, xpa, xpf, v_first)."""
    if input_ids.dim() != 2:
        raise ValueError("native_prefill_batched expects input_ids shaped [batch, seq]")
    base = model.model
    state, xpa, xpf, v_first = _init_state_batched(model, input_ids.shape[0], input_ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(input_ids.shape[1]):
        x = F.embedding(input_ids[:, t], base.embeddings.weight)
        x, state, xpa, xpf, v_first = _step_token_batched(model, x, state, xpa, xpf, v_first)
    if x is None:
        raise ValueError("native_prefill_batched requires at least one token")
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight), state, xpa, xpf, v_first


def native_decode_step_batched(model, token_ids, state, xpa, xpf, v_first):
    """One batched incremental decode step. token_ids: [B] or [B, 1]."""
    base = model.model
    token_ids = token_ids.reshape(-1)
    x = F.embedding(token_ids, base.embeddings.weight)
    x, state, xpa, xpf, v_first = _step_token_batched(model, x, state, xpa, xpf, v_first)
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight), state, xpa, xpf, v_first


def _init_state(model, device, dtype):
    base = model.model
    n = len(base.layers)
    H = base.layers[0].attn.num_heads
    N = base.layers[0].attn.head_dim
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def _step_token(model, x, state, xpa, xpf, v_first):
    for i, layer in enumerate(model.model.layers):
        attn = layer.attn
        residual = layer.pre_norm(x) if hasattr(layer, "pre_norm") else x
        h = layer.attn_norm(residual)
        a, xpa[i], state[i], v_first = attn_step(attn, i, h, xpa[i], v_first, state[i])
        x = residual + a
        residual = x
        h2 = layer.ffn_norm(x)
        f, xpf[i] = ffn_step(layer.ffn, h2, xpf[i])
        x = residual + f
    return x, state, xpa, xpf, v_first


def native_forward(model, input_ids: torch.Tensor):
    """Sequential full forward. Returns final-token logits [vocab]."""
    base = model.model
    state, xpa, xpf, v_first = _init_state(model, input_ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(input_ids.shape[1]):
        x = F.embedding(input_ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = _step_token(model, x, state, xpa, xpf, v_first)
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight)


def native_prefill(model, input_ids):
    """Prefill, returning (logits, state, xpa, xpf, v_first) for incremental decode."""
    base = model.model
    state, xpa, xpf, v_first = _init_state(model, input_ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(input_ids.shape[1]):
        x = F.embedding(input_ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = _step_token(model, x, state, xpa, xpf, v_first)
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight), state, xpa, xpf, v_first


def native_decode_step(model, token_id, state, xpa, xpf, v_first):
    """One incremental decode step. token_id: scalar tensor. Returns (logits, ...state)."""
    base = model.model
    x = F.embedding(token_id.reshape(1, 1), base.embeddings.weight).reshape(-1)
    x, state, xpa, xpf, v_first = _step_token(model, x, state, xpa, xpf, v_first)
    x = base.norm(x)
    return F.linear(x, model.lm_head.weight), state, xpa, xpf, v_first


if __name__ == "__main__":
    import os
    import sys
    import time
    os.environ.setdefault("RWKV_V7_ON", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/rwkv7-models/rwkv7-g1d-0.1b-hf"
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()

    # correctness vs fla
    for prompt in ["The quick brown fox jumps over the lazy dog.",
                   "Once upon a time, in a faraway land,"]:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            fla = model(ids).logits[0, -1].float().cpu()
            nat = native_forward(model, ids).float().cpu()
        cos = F.cosine_similarity(fla.unsqueeze(0), nat.unsqueeze(0)).item()
        maxabs = (fla - nat).abs().max().item()
        print(f"[correctness] cos={cos:.6f} maxabs={maxabs:.4f} "
              f"argmax={int(fla.argmax() == nat.argmax())}  {prompt[:36]!r}")

    # eager native decode speed
    ids = tok("The quick brown fox.", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        logits, state, xpa, xpf, v_first = native_prefill(model, ids)
        nx = logits.argmax()
        for _ in range(5):
            logits, state, xpa, xpf, v_first = native_decode_step(model, nx, state, xpa, xpf, v_first)
            nx = logits.argmax()
        torch.cuda.synchronize()
        t0 = time.time()
        N = 128
        for _ in range(N):
            logits, state, xpa, xpf, v_first = native_decode_step(model, nx, state, xpa, xpf, v_first)
            nx = logits.argmax()
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"[decode] eager native: {N / dt:.1f} tok/s  ({1000 * dt / N:.2f} ms/tok)")
