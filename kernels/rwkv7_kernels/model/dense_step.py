# coding=utf-8
"""Pure dense TorchScript RWKV-7 per-layer token steps.

Migrated from the historical native JIT runner into the optional kernel wheel.
The functions intentionally contain no hardware policy, optional dispatch,
Hugging Face output types, or model-container logic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.jit.script
def block_step(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
               v_first: torch.Tensor, state: torch.Tensor,
               layer_id: int, H: int, N: int, group_eps: float,
               norm_eps: float, has_pre: int,
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
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
               RKVw: torch.Tensor):
    D = int(an_w.numel())
    A = H * N
    # --- block wiring (fuse_norm=False) ---
    if has_pre == 1:
        residual = F.layer_norm(x, [D], pre_w, pre_b, norm_eps)
    else:
        residual = x
    h = F.layer_norm(residual, [D], an_w, an_b, norm_eps)

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
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(A)
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
    out = out.view(A)
    out = F.group_norm(out.view(1, A), H, gn_w, gn_b, group_eps).view(A)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(A)
    out = F.linear(out * g, Ow)
    x = residual + out

    # --- CMix_one ---
    residual = x
    h2 = F.layer_norm(x, [D], fn_w, fn_b, norm_eps)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


@torch.jit.script
def block_step_batched(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
                       v_first: torch.Tensor, state: torch.Tensor,
                       layer_id: int, H: int, N: int, group_eps: float,
                       norm_eps: float, has_pre: int,
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
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
               RKVw: torch.Tensor):
    # Batched variant of block_step. Shapes:
    # x/xpa/xpf:[B,D], v_first:[B,A], state:[B,H,N,N].
    B = x.shape[0]
    D = int(an_w.numel())
    A = H * N
    if has_pre == 1:
        residual = F.layer_norm(x, [D], pre_w, pre_b, norm_eps)
    else:
        residual = x
    h = F.layer_norm(residual, [D], an_w, an_b, norm_eps)

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
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, A)
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
    out = out.view(B, A)
    out = F.group_norm(out, H, gn_w, gn_b, group_eps).view(B, A)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, A)
    out = F.linear(out * g, Ow)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [D], fn_w, fn_b, norm_eps)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


__all__ = ["block_step", "block_step_batched"]
