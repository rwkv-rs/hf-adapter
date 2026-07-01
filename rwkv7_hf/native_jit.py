# coding=utf-8
"""TorchScript-native RWKV-7 decode. The ENTIRE per-layer block (LayerNorms +
TMix_one + CMix_one) is fused into one torch.jit.script function, so per token
there is only ~1 C++ call per layer + embedding/head. Math ports the official
RWKV_x070 TMix_one/CMix_one (bit-exact vs FLA, see native.py).

Run: python -m rwkv7_hf.native_jit <hf_dir>
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.jit.script
def block_step(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
               v_first: torch.Tensor, state: torch.Tensor,
               layer_id: int, H: int, N: int, eps: float, has_pre: int,
               pre_w: torch.Tensor, pre_b: torch.Tensor,
               an_w: torch.Tensor, an_b: torch.Tensor,
               fn_w: torch.Tensor, fn_b: torch.Tensor,
               x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
               x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
               k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
               Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
               w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
               a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
               v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
               g1: torch.Tensor, g2: torch.Tensor,
               gn_w: torch.Tensor, gn_b: torch.Tensor,
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor):
    # --- block wiring (fuse_norm=False) ---
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

    # --- TMix_one ---
    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(H, N, 1)
    out = out.view(H * N)
    out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    # --- CMix_one ---
    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


@torch.jit.script
def block_step_batched(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
                       v_first: torch.Tensor, state: torch.Tensor,
                       layer_id: int, H: int, N: int, eps: float, has_pre: int,
                       pre_w: torch.Tensor, pre_b: torch.Tensor,
                       an_w: torch.Tensor, an_b: torch.Tensor,
                       fn_w: torch.Tensor, fn_b: torch.Tensor,
                       x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
                       x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
                       k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
                       Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
                       w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
                       a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
                       v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
                       g1: torch.Tensor, g2: torch.Tensor,
                       gn_w: torch.Tensor, gn_b: torch.Tensor,
                       fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor):
    # Batched variant of block_step. Shapes:
    # x/xpa/xpf/v_first:[B,H*N], state:[B,H,N,N].
    B = x.shape[0]
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(B, H, N, 1)
    out = out.view(B, H * N)
    out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


def extract(model):
    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(ref.shape[0], 1, device=ref.device, dtype=ref.dtype)
        v0 = vl.lora[2].bias if vl is not None else torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
        if hasattr(layer, "pre_norm"):
            pre_w, pre_b, has_pre = layer.pre_norm.weight, layer.pre_norm.bias, 1
        else:
            pre_w = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            pre_b = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            has_pre = 0
        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            a.r_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
            a.w_lora.lora[0].weight, a.w_lora.lora[2].weight, a.w_lora.lora[2].bias,
            a.a_lora.lora[0].weight, a.a_lora.lora[2].weight, a.a_lora.lora[2].bias,
            v1, v2, v0,
            a.g_lora.lora[0].weight, a.g_lora.lora[2].weight,
            a.g_norm.weight, a.g_norm.bias,
            layer.ffn.x_k, layer.ffn.key.weight, layer.ffn.value.weight,
        ))
    return packs, H, N, eps


def _init(model, device, dtype):
    layers = model.model.layers
    n = len(layers)
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    hid = layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def step(model, x, state, xpa, xpf, v_first, packs):
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step(x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p)
    return x, state, xpa, xpf, v_first


def step_batched(model, x, state, xpa, xpf, v_first, packs):
    """Batched TorchScript block-step decode for native_model caches.

    Shapes mirror ``rwkv7_hf.native._step_token_batched``: x/xpa/xpf/v_first
    are ``[B, hidden]`` and recurrent state is ``[B, H, N, N]`` per layer.
    Keeping this helper in native_jit lets the experimental FLA-free model use
    the same reduced-dispatch H2 decode idea without importing the wrapper.
    """
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step_batched(
            x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p
        )
    return x, state, xpa, xpf, v_first


def forward(model, ids, packs):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    x = F.layer_norm(x, [H * N], base.norm.weight, base.norm.bias, 1e-5)
    return F.linear(x, model.lm_head.weight)


def decode_speed(model, ids, packs, n=128):
    import time
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    emb = base.embeddings.weight
    head = model.lm_head.weight
    norm_w = base.norm.weight
    norm_b = base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
    with torch.no_grad():
        for _ in range(5):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
        torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def _block_ip(x, state, xpa, xpf, v_first, p):
    """In-place (eager) block step for CUDA-graph capture: state/xpa/xpf/v_first
    are fixed buffers updated in place. Same math as block_step."""
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV) = p
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
    xx = xpa - h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    new_state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(h.dtype) @ r.view(H, N, 1)
    out = out.view(H * N)
    out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(H * N)
    out = F.linear(out * g, Ow)
    xpa.copy_(h)
    state.copy_(new_state)
    x = residual + out
    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    xpf.copy_(h2)
    return residual + F.linear(fk, fV)


def _block_ip_batched(x, state, xpa, xpf, v_first, p):
    """In-place batched block step for CUDA-graph capture.

    Shapes:
      x/xpa/xpf/v_first: [B, H*N]
      state: [B, H, N, N]

    This mirrors `block_step_batched` but writes recurrent/cache buffers in
    place so a captured CUDA graph can replay across decode tokens.
    """
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV) = p
    B = x.shape[0]
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
    xx = xpa - h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
    k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    new_state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(h.dtype) @ r.view(B, H, N, 1)
    out = out.view(B, H * N)
    out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, H * N)
    out = F.linear(out * g, Ow)
    xpa.copy_(h)
    state.copy_(new_state)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    xpf.copy_(h2)
    return residual + F.linear(fk, fV)


def cuda_graph_decode(model, ids, packs, n=128):
    import time
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb = base.embeddings.weight
    head = model.lm_head.weight
    nw, nb = base.norm.weight, base.norm.bias

    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head).argmax())

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step()
            tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()

    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        g.replay()
        tok_id.copy_(logits.argmax())
    torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def greedy_jit(model, ids, packs, n=40):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    nw, nb = base.norm.weight, base.norm.bias
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), model.lm_head.weight).argmax().clone()
    toks = [int(nx)]
    with torch.no_grad():
        for _ in range(n - 1):
            x = F.embedding(nx.reshape(1, 1), base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), model.lm_head.weight).argmax()
            toks.append(int(nx))
    return toks


def greedy_graph(model, ids, packs, n=40):
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb, head = base.embeddings.weight, model.lm_head.weight
    nw, nb = base.norm.weight, base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head).argmax())
    # snapshot post-prefill state so we can realign after warmup advances it
    st_s = [s.clone() for s in state]
    xpa_s = [s.clone() for s in xpa]
    xpf_s = [s.clone() for s in xpf]
    vf_s = v_first.clone()
    tok_s = tok_id.clone()

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head))

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step(); tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()
    # restore post-prefill state so the captured graph replays from the right point
    for i in range(len(state)):
        state[i].copy_(st_s[i]); xpa[i].copy_(xpa_s[i]); xpf[i].copy_(xpf_s[i])
    v_first.copy_(vf_s)
    tok_id.copy_(tok_s)
    toks = [int(tok_id)]
    for _ in range(n - 1):
        g.replay()
        nt = logits.argmax()
        tok_id.copy_(nt)
        toks.append(int(nt))
    return toks


def fast_generate(model, tokenizer, prompt, max_new_tokens=48, use_graph=True):
    """End-to-end greedy generation via the native (CUDA-graph) decode path.
    Returns the full decoded text (prompt + new tokens). Same result as the
    FLA model's greedy generate(), but ~10x faster on the 5070."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    packs, _, _, _ = extract(model)
    fn = greedy_graph if use_graph else greedy_jit
    new_tokens = fn(model, ids, packs, n=max_new_tokens)
    full = ids[0].tolist() + new_tokens
    return tokenizer.decode(full, skip_special_tokens=True)


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("RWKV_V7_ON", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/rwkv7-models/rwkv7-g1d-0.1b-hf"
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    # correctness at fp32 vs fla
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()
    packs, H, N, eps = extract(model)
    for prompt in ["The quick brown fox jumps over the lazy dog.",
                   "Once upon a time, in a faraway land,"]:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            fla = model(ids).logits[0, -1].float().cpu()
            nat = forward(model, ids, packs).float().cpu()
        cos = F.cosine_similarity(fla.unsqueeze(0), nat.unsqueeze(0)).item()
        maxabs = (fla - nat).abs().max().item()
        print(f"[correctness] cos={cos:.6f} maxabs={maxabs:.4f} "
              f"argmax={int(fla.argmax() == nat.argmax())}  {prompt[:36]!r}")
    del model; torch.cuda.empty_cache()
    # speed
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=dt, device_map="cuda").eval()
        packs, H, N, eps = extract(model)
        ids = tok("The quick brown fox.", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            tps_jit = decode_speed(model, ids, packs)
            tps_cg = cuda_graph_decode(model, ids, packs)
            tj = greedy_jit(model, ids, packs)
            tg = greedy_graph(model, ids, packs)
        match = sum(int(a == b) for a, b in zip(tj, tg))
        print(f"[decode {dt_name}] jit-fused {tps_jit:.1f} | cuda-graph {tps_cg:.1f} tok/s | "
              f"graph-correct {match}/{len(tj)} tokens == jit")
        del model; torch.cuda.empty_cache()

    # end-to-end: native greedy token ids vs fla model.generate (must match)
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    packs, _, _, _ = extract(model)
    prompt = "User: Hello!\n\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        fla_out = model.generate(ids, max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=0)
    fla_ids = fla_out[0, ids.shape[1]:].tolist()
    nat_ids = greedy_graph(model, ids, packs, n=32)
    print(f"[e2e] fla   : {tok.decode(fla_ids)!r}")
    print(f"[e2e] native: {tok.decode(nat_ids)!r}")
    print(f"[e2e] token-identical: {fla_ids == nat_ids} ({sum(int(a==b) for a,b in zip(fla_ids,nat_ids))}/{len(fla_ids)})")
