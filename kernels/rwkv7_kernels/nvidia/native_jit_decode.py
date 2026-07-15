# coding=utf-8
"""Native dense-JIT and CUDA-graph decode execution.

The facade binds stable helpers once and re-exports every function here as a
direct alias. External graph runners therefore retain the historical per-layer
call cost while decode orchestration is isolated from prefill and policy code.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


_OWNED_NAMES = {'cuda_graph_decode', 'fast_generate', 'step', 'step_batched', '_block_ip_batched', 'greedy_graph', 'decode_speed', '_block_ip', 'greedy_jit', 'forward'} | {"bind_runtime"}
_RUNTIME_NAMES = ('_graph_linear_call', '_graph_linear_call_with_explicit_bias', '_graph_linear_is_dense', '_graph_linear_shape', '_graph_linears_are_dense', '_init', '_linear_module', '_lm_head', '_native_graph_ada_wag_lora_enabled', '_native_graph_ada_wagv_bmm_enabled', '_native_graph_sm120_wagv_bmm_g_enabled', '_native_graph_ada_wagv_lora_enabled', '_native_graph_blackwell_norm_mix_enabled', '_native_graph_ffn_dispatch', '_native_graph_fp16_recurrent_enabled', '_native_graph_fused_norm_mix_enabled', '_native_graph_fused_norm_mix_num_warps', '_native_graph_fused_output_enabled', '_native_graph_fused_output_project_block_m', '_native_graph_fused_output_project_enabled', '_native_graph_fused_projection_enabled', '_native_graph_fused_recurrent_output_enabled', '_native_graph_fused_recurrent_raw_enabled', '_native_graph_fused_recurrent_raw_num_warps', '_native_graph_fused_wag_lora_blocks', '_native_graph_fused_wag_lora_enabled', '_native_graph_fused_wavg_lora_blocks', '_native_graph_fused_wavg_lora_enabled', '_native_graph_fused_wavg_lora_num_warps', '_native_graph_linear_dispatch', '_native_graph_rkv_project', '_native_graph_sm70_wagv_lora_enabled', '_native_graph_vkwr_rkv_dispatch', '_recurrent_update_batched', '_recurrent_update_unbatched', 'ada_wag_lora', 'ada_wagv_bmm', 'ada_wagv_lora', 'blackwell_ffn_add_norm_mix', 'block_step', 'block_step_batched', 'extract', 'fused_attn_norm_mix6_decode', 'fused_attn_output_prepare', 'fused_attn_output_project', 'fused_ffn_add_norm_mix_decode', 'fused_recurrent_output_prepare', 'fused_recurrent_output_prepare_raw', 'fused_rkv_wavg_projection', 'fused_wag_lora', 'fused_wavg_lora', 'native_fp16_recurrent_output_prepare_raw', 'sm70_wagv_lora')


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _ada_wagv_lora_extension_required() -> bool:
    return (
        os.environ.get(
            "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION", "0"
        )
        .strip()
        .lower()
        in _TRUE_VALUES
    )


def _sm70_wagv_lora_extension_required() -> bool:
    return (
        os.environ.get(
            "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION", "0"
        )
        .strip()
        .lower()
        in _TRUE_VALUES
    )


def bind_runtime(runtime: dict[str, object]) -> None:
    for name in _RUNTIME_NAMES:
        if name in runtime and name not in _OWNED_NAMES:
            globals()[name] = runtime[name]


def _native_decay_projection(x, down, up, bias):
    """Match the clean HF ``project_without_bias(...).float() + w0`` rule."""

    projected = _graph_linear_call(torch.tanh(_graph_linear_call(x, down)), up)
    if _graph_linear_is_dense(up) and bias is not None:
        return projected.float() + bias.float()
    # Packed quantized modules retain and apply their own bias.
    return projected


def step(model, x, state, xpa, xpf, v_first, packs):
    for p in packs:
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step(x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p)
    return x, state, xpa, xpf, v_first

def step_batched(model, x, state, xpa, xpf, v_first, packs):
    """Batched TorchScript block-step decode for native_model caches.

    Shapes mirror ``rwkv7_hf.native._step_token_batched``: x/xpa/xpf are
    ``[B,D]``, v_first is ``[B,A]``, and recurrent state is
    ``[B,H,N,N]`` per layer.
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
    return _lm_head(model, x)

def decode_speed(model, ids, packs, n=128):
    import time
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    emb = base.embeddings.weight
    head = model.lm_head
    norm_w = base.norm.weight
    norm_b = base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
    with torch.no_grad():
        for _ in range(5):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _linear_module(head, F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5)).argmax()
        torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt

def _block_ip(
    x,
    state,
    xpa,
    xpf,
    v_first,
    p,
    sparse_ffn_out=None,
    fp16_elapsed=None,
    fp16_advance_elapsed=False,
    route_observer=None,
    state_layout="vk_v1",
):
    """In-place (eager) block step for CUDA-graph capture: state/xpa/xpf/v_first
    are fixed buffers updated in place. Same math as block_step."""
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    D = int(an_w.numel())
    A = int(H * N)
    equal_width = D == A
    residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5) if has_pre else x
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled(1, D)
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(1, D) and RKVw.numel() != 0
        xr, xw, xk, xv, xa, xg = fused_attn_norm_mix6_decode(
            residual,
            xpa,
            an_w,
            an_b,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            num_warps=_native_graph_fused_norm_mix_num_warps(),
            stack_rkv=stack_rkv,
        )
    else:
        h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)
        xx = xpa - h
        xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
        xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    v_mixed = False
    lora_dense = bool(
        _graph_linears_are_dense(w1, w2, a1, a2, v1, v2, g1, g2)
        and w0.dtype == x.dtype
    )
    extension_rank = max(
        _graph_linear_shape(w1)[0],
        _graph_linear_shape(a1)[0],
        _graph_linear_shape(g1)[0],
        _graph_linear_shape(v1)[0],
    )
    require_ada_extension = _ada_wagv_lora_extension_required()
    extension_eligible = bool(
        equal_width
        and lora_dense
        and _native_graph_ada_wagv_lora_enabled(1, D, extension_rank)
    )
    if require_ada_extension:
        if not extension_eligible:
            raise RuntimeError(
                "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=1 "
                "reached an ineligible layer; fallback is forbidden"
            )
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        if i > 0:
            w, a, g, v = ada_wagv_lora(
                xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
                w0, a0, v0, v, v_first, sigmoid_a=True,
                require_extension=True,
            )
            v_mixed = True
        else:
            w, a, g, _unused_v = ada_wagv_lora(
                xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
                w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
                require_extension=True,
            )
    elif _native_graph_fused_projection_enabled() and lora_dense and _graph_linears_are_dense(Rw, Kw, Vw):
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr.view(1, D),
            xk.view(1, D),
            xv.view(1, D),
            xw.view(1, D),
            xa.view(1, D),
            xg.view(1, D),
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        r = r.view(A)
        k = k.view(A)
        v = v.view(A)
        w = w.view(A)
        a = torch.sigmoid(a.view(A))
        g = g.view(A)
        v_gate = torch.sigmoid(v_gate.view(A))
    elif equal_width and i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif equal_width and i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        1,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif equal_width and lora_dense and _native_graph_ada_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w, a, g = ada_wag_lora(
            xw, xa, xg, w1, a1, g1, w2, a2, g2, w0, a0,
        )
        a = torch.sigmoid(a)
    elif equal_width and i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(1, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        if route_observer is not None:
            route_observer("sm70_wagv_lora_selected", int(i))
        w, a, g, v = sm70_wagv_lora(
            xw.view(1, D), xa.view(1, D), xg.view(1, D), xv.view(1, D),
            w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0,
            v.view(1, A), v_first.view(1, A),
            require_extension=_sm70_wagv_lora_extension_required(),
        )
        if route_observer is not None:
            route_observer("sm70_wagv_lora_effective", int(i))
        w = w.view(A); a = torch.sigmoid(a.view(A)); g = g.view(A); v = v.view(A)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(1, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks(1)
            if route_observer is not None:
                route_observer("fused_wavg_lora_selected", int(i))
            w, a, g, v_gate = fused_wavg_lora(
                xw.view(1, D),
                xa.view(1, D),
                xg.view(1, D),
                xv.view(1, D),
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
                num_warps=_native_graph_fused_wavg_lora_num_warps(1),
            )
            if route_observer is not None:
                route_observer("fused_wavg_lora_effective", int(i))
            w = w.view(A)
            a = a.view(A)
            g = g.view(A)
            v_gate = v_gate.view(A)
        a = torch.sigmoid(a)
    elif lora_dense and _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw.view(1, D),
            xa.view(1, D),
            xg.view(1, D),
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        w = w.view(A)
        a = torch.sigmoid(a.view(A))
        g = g.view(A)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, D)
        w = _native_decay_projection(xw, w1, w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fp16_recurrent = bool(
        w.dtype == r.dtype
        and _native_graph_fp16_recurrent_enabled(state, fp16_elapsed)
    )
    use_kv_v2_recurrent = (
        str(getattr(state_layout, "value", state_layout)).strip().lower()
        == "kv_v2"
    )
    use_fused_recurrent_output = (
        use_kv_v2_recurrent
        or use_fp16_recurrent
        or _native_graph_fused_recurrent_output_enabled()
    )
    use_fused_recurrent_raw = bool(
        w.dtype == r.dtype
        and (
            use_kv_v2_recurrent
            or use_fp16_recurrent
            or (
                use_fused_recurrent_output
                and _native_graph_fused_recurrent_raw_enabled(1, D)
            )
        )
    )
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(A)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    new_state = None
    if use_fp16_recurrent:
        out = native_fp16_recurrent_output_prepare_raw(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            fp16_elapsed,
            advance_elapsed=fp16_advance_elapsed,
            eps=eps,
        ).view(A)
    elif use_fused_recurrent_raw:
        out, new_state = fused_recurrent_output_prepare_raw(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
            num_warps=_native_graph_fused_recurrent_raw_num_warps(),
            state_layout=state_layout,
        )
        out = out.view(A)
        new_state = new_state.view(H, N, N)
    elif use_fused_recurrent_output:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = fused_recurrent_output_prepare(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.view(A)
        new_state = new_state.view(H, N, N)
    else:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = _recurrent_update_unbatched(r, w, k, v, kk, a, state, H, N)
    if use_fused_recurrent_output:
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    elif _native_graph_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
        out = fused_attn_output_project(
            out.view(1, A),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, A),
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        ).view(D)
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out.view(1, A),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, A),
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        ).view(A)
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    else:
        out = F.group_norm(out.view(1, A), H, gn_w, gn_b, eps).view(A)
        sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(H, N)).view(A)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    if new_state is not None:
        state.copy_(new_state)
    if use_fused_norm_mix:
        if _native_graph_blackwell_norm_mix_enabled(
            residual, out, xpf, layer_index=int(i)
        ):
            residual, fk = blackwell_ffn_add_norm_mix(
                residual, out, xpf, fn_w, fn_b, fx_k, eps=1.0e-5
            )
        else:
            residual, fk = fused_ffn_add_norm_mix_decode(
                residual,
                out,
                xpf,
                fn_w,
                fn_b,
                fx_k,
                num_warps=_native_graph_fused_norm_mix_num_warps(),
            )
    else:
        xpa.copy_(h)
        residual = residual + out
        h2 = F.layer_norm(residual, [D], fn_w, fn_b, 1e-5)
        fxx = xpf - h2
        fk = h2 + fxx * fx_k
        xpf.copy_(h2)
    return _native_graph_ffn_dispatch(
        fk,
        fK,
        fV,
        residual,
        sparse_out=sparse_ffn_out,
        layer_index=int(i),
    )

def _block_ip_batched(
    x,
    state,
    xpa,
    xpf,
    v_first,
    p,
    sparse_ffn_out=None,
    fp16_elapsed=None,
    fp16_advance_elapsed=False,
    route_observer=None,
    state_layout="vk_v1",
):
    """In-place batched block step for CUDA-graph capture.

    Shapes:
      x/xpa/xpf: [B,D], v_first: [B,A]
      state: [B, H, N, N]

    This mirrors `block_step_batched` but writes recurrent/cache buffers in
    place so a captured CUDA graph can replay across decode tokens.
    """
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    B = x.shape[0]
    D = int(an_w.numel())
    A = int(H * N)
    equal_width = D == A
    residual = F.layer_norm(x, [D], pre_w, pre_b, 1e-5) if has_pre else x
    lora_dense = bool(
        _graph_linears_are_dense(w1, w2, a1, a2, v1, v2, g1, g2)
        and w0.dtype == x.dtype
    )
    bmm_max_rank = (
        max(
            _graph_linear_shape(w1)[0],
            _graph_linear_shape(a1)[0],
            _graph_linear_shape(g1)[0],
            _graph_linear_shape(v1)[0],
        )
        if i > 0
        else max(
            _graph_linear_shape(w1)[0],
            _graph_linear_shape(a1)[0],
            _graph_linear_shape(g1)[0],
        )
    )
    use_ada_wagv_bmm = bool(
        equal_width
        and lora_dense
        and _native_graph_ada_wagv_bmm_enabled(B, D, bmm_max_rank, x.device)
    )
    use_fused_norm_mix = _native_graph_fused_norm_mix_enabled(B, D)
    use_sm120_wagv_bmm_g = bool(
        use_ada_wagv_bmm
        and use_fused_norm_mix
        and _graph_linears_are_dense(Rw, Kw, Vw)
        and _native_graph_sm120_wagv_bmm_g_enabled(
            B, D, bmm_max_rank, x.device
        )
    )
    if use_fused_norm_mix:
        stack_rkv = _native_graph_vkwr_rkv_dispatch(B, D) and RKVw.numel() != 0
        use_sm120_wagv_bmm_g = bool(use_sm120_wagv_bmm_g and stack_rkv)
        xr, xw, xk, xv, xa, xg = fused_attn_norm_mix6_decode(
            residual,
            xpa,
            an_w,
            an_b,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            num_warps=_native_graph_fused_norm_mix_num_warps(),
            stack_rkv=bool(stack_rkv and not use_sm120_wagv_bmm_g),
            # The grouped route consumes a W/A prefix in layer zero and W/A/V
            # elsewhere, so both cases can reuse this contiguous allocation.
            stack_wav=bool(
                use_ada_wagv_bmm
                and not stack_rkv
                and not use_sm120_wagv_bmm_g
            ),
            stack_rkv_wagv=use_sm120_wagv_bmm_g,
        )
    else:
        h = F.layer_norm(residual, [D], an_w, an_b, 1e-5)
        xx = xpa - h
        xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
        xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    v_mixed = False
    if i > 0 and use_sm120_wagv_bmm_g:
        r, k, v = _native_graph_rkv_project(
            xr, xk, xv, Rw, Kw, Vw, RKVw, B, D
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_selected", int(i))
            route_observer("sm120_wagv_bmm_g_selected", int(i))
        w, a, g, v = ada_wagv_bmm(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
            require_bmm=True, include_g=True, require_zero_copy=True,
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_effective", int(i))
            route_observer("sm120_wagv_bmm_g_effective", int(i))
        v_mixed = True
    elif i == 0 and use_sm120_wagv_bmm_g:
        r, k, v = _native_graph_rkv_project(
            xr, xk, xv, Rw, Kw, Vw, RKVw, B, D
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_selected", int(i))
            route_observer("sm120_wagv_bmm_g_selected", int(i))
        w, a, g, _unused_v = ada_wagv_bmm(
            xw, xa, xg, xv, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
            require_bmm=True, include_g=True, require_zero_copy=True,
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_effective", int(i))
            route_observer("sm120_wagv_bmm_g_effective", int(i))
    elif _native_graph_fused_projection_enabled() and lora_dense and _graph_linears_are_dense(Rw, Kw, Vw):
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr,
            xk,
            xv,
            xw,
            xa,
            xg,
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        a = torch.sigmoid(a)
        v_gate = torch.sigmoid(v_gate)
    elif i > 0 and use_ada_wagv_bmm:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        if route_observer is not None:
            route_observer("ada_wagv_bmm_selected", int(i))
        w, a, g, v = ada_wagv_bmm(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True, require_bmm=True,
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_effective", int(i))
        v_mixed = True
    elif i == 0 and use_ada_wagv_bmm:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        if route_observer is not None:
            route_observer("ada_wagv_bmm_selected", int(i))
        w, a, g, _unused_v = ada_wagv_bmm(
            xw, xa, xg, xv, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
            require_bmm=True,
        )
        if route_observer is not None:
            route_observer("ada_wagv_bmm_effective", int(i))
    elif equal_width and i > 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0], _graph_linear_shape(v1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g, v = ada_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2,
            w0, a0, v0, v, v_first, sigmoid_a=True,
        )
        v_mixed = True
    elif equal_width and i == 0 and lora_dense and _native_graph_ada_wagv_lora_enabled(
        B,
        D,
        max(_graph_linear_shape(w1)[0], _graph_linear_shape(a1)[0], _graph_linear_shape(g1)[0]),
    ):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g, _unused_v = ada_wagv_lora(
            xw, xa, xg, xg, w1, a1, g1, g1, w2, a2, g2, g2,
            w0, a0, a0, v, v, sigmoid_a=True, compute_v=False,
        )
    elif equal_width and lora_dense and _native_graph_ada_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w, a, g = ada_wag_lora(
            xw, xa, xg, w1, a1, g1, w2, a2, g2, w0, a0,
        )
        a = torch.sigmoid(a)
    elif equal_width and i > 0 and lora_dense and _native_graph_sm70_wagv_lora_enabled(B, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        if route_observer is not None:
            route_observer("sm70_wagv_lora_selected", int(i))
        w, a, g, v = sm70_wagv_lora(
            xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0, v, v_first,
            require_extension=_sm70_wagv_lora_extension_required(),
        )
        if route_observer is not None:
            route_observer("sm70_wagv_lora_effective", int(i))
        a = torch.sigmoid(a)
        v_mixed = True
    elif lora_dense and _native_graph_fused_wavg_lora_enabled(B, D):
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks(B)
            if route_observer is not None:
                route_observer("fused_wavg_lora_selected", int(i))
            w, a, g, v_gate = fused_wavg_lora(
                xw,
                xa,
                xg,
                xv,
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
                num_warps=_native_graph_fused_wavg_lora_num_warps(B),
            )
            if route_observer is not None:
                route_observer("fused_wavg_lora_effective", int(i))
        a = torch.sigmoid(a)
    elif lora_dense and _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw,
            xa,
            xg,
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        a = torch.sigmoid(a)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, D)
        w = _native_decay_projection(xw, w1, w2, w0)
        a = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xa, a1), a2, a0))
        g = _graph_linear_call(torch.sigmoid(_graph_linear_call(xg, g1)), g2)
    use_fp16_recurrent = bool(
        w.dtype == r.dtype
        and _native_graph_fp16_recurrent_enabled(state, fp16_elapsed)
    )
    use_kv_v2_recurrent = (
        str(getattr(state_layout, "value", state_layout)).strip().lower()
        == "kv_v2"
    )
    use_fused_recurrent_output = (
        use_kv_v2_recurrent
        or use_fp16_recurrent
        or _native_graph_fused_recurrent_output_enabled()
    )
    use_fused_recurrent_raw = bool(
        w.dtype == r.dtype
        and (
            use_kv_v2_recurrent
            or use_fp16_recurrent
            or (
                use_fused_recurrent_output
                and _native_graph_fused_recurrent_raw_enabled(B, D)
            )
        )
    )
    if not use_fused_recurrent_raw:
        kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, A)
        k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    elif not v_mixed:
        if v_gate is None:
            v_gate = torch.sigmoid(_graph_linear_call_with_explicit_bias(_graph_linear_call(xv, v1), v2, v0))
        v = v + (v_first - v) * v_gate
    new_state = None
    if use_fp16_recurrent:
        out = native_fp16_recurrent_output_prepare_raw(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            fp16_elapsed,
            advance_elapsed=fp16_advance_elapsed,
            eps=eps,
        ).reshape(B, A)
    elif use_fused_recurrent_raw:
        out, new_state = fused_recurrent_output_prepare_raw(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            k_k,
            k_a,
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
            state_layout=state_layout,
        )
        out = out.reshape(B, A)
    elif use_fused_recurrent_output:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = fused_recurrent_output_prepare(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.reshape(B, A)
    else:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
        out, new_state = _recurrent_update_batched(r, w, k, v, kk, a, state, B, H, N)
    if use_fused_recurrent_output:
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    elif _native_graph_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
        out = fused_attn_output_project(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        )
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        )
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    else:
        out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, A)
        sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(B, H, N)).view(B, A)) * g
        out = _native_graph_linear_dispatch(out, Ow, role="hidden")
    if new_state is not None:
        state.copy_(new_state)
    if use_fused_norm_mix:
        if _native_graph_blackwell_norm_mix_enabled(
            residual, out, xpf, layer_index=int(i)
        ):
            residual, fk = blackwell_ffn_add_norm_mix(
                residual, out, xpf, fn_w, fn_b, fx_k, eps=1.0e-5
            )
        else:
            residual, fk = fused_ffn_add_norm_mix_decode(
                residual,
                out,
                xpf,
                fn_w,
                fn_b,
                fx_k,
                num_warps=_native_graph_fused_norm_mix_num_warps(),
            )
    else:
        xpa.copy_(h)
        residual = residual + out
        h2 = F.layer_norm(residual, [D], fn_w, fn_b, 1e-5)
        fxx = xpf - h2
        fk = h2 + fxx * fx_k
        xpf.copy_(h2)
    return _native_graph_ffn_dispatch(
        fk,
        fK,
        fV,
        residual,
        sparse_out=sparse_ffn_out,
        route_observer=route_observer,
        layer_index=int(i),
    )

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
    head = model.lm_head
    nw, nb = base.norm.weight, base.norm.bias

    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax())

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)))

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
    nx = _lm_head(model, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax().clone()
    toks = [int(nx)]
    with torch.no_grad():
        for _ in range(n - 1):
            x = F.embedding(nx.reshape(1, 1), base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = _lm_head(model, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax()
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
    emb, head = base.embeddings.weight, model.lm_head
    nw, nb = base.norm.weight, base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)).argmax())
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
        logits.copy_(_linear_module(head, F.layer_norm(x, [H * N], nw, nb, 1e-5)))

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

__all__ = ['step', 'step_batched', 'forward', 'decode_speed', '_block_ip', '_block_ip_batched', 'cuda_graph_decode', 'greedy_jit', 'greedy_graph', 'fast_generate']
