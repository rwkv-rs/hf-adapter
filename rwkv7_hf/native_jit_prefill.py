# coding=utf-8
"""Native RWKV-7 sequence prefill execution engine.

This module owns sequence projections, recurrent scan routing, layer-wise
prefill math and cache handoff. Policy remains in the dedicated runtime-policy
module and is supplied through the stable native_jit facade.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_EXECUTION_NAMES = {'_native_prefill_project_residual', '_prefill_current_device', '_native_prefill_linear', '_native_prefill_scan', '_native_prefill_linear_add_residual', '_native_prefill_stacked_rkv_weights'}
_OWNED_NAMES = _EXECUTION_NAMES | {"bind_runtime"}
_RUNTIME_NAMES = ('_FP16_ACCUMULATION_LOCK', '_bnb8_direct_linear', '_bnb8_direct_relu_square_linear', '_bnb8_ffn_mix_quant_enabled', '_bnb8_prequant_linear', '_bnb8_rkv_mix_quant_enabled', '_graph_linear_is_dense', '_graph_linears_are_dense', '_init_batched_from_packs', '_lm_head', '_native_bnb8_policy_block', '_native_prefill_attn_shift_mix_block_size', '_native_prefill_dplr_chunk_size', '_native_prefill_dplr_scan_enabled', '_native_prefill_ffn_shift_mix_block_size', '_native_prefill_fp16_accum_ffn_key_enabled', '_native_prefill_fp16_accum_ffn_key_layers', '_native_prefill_fp16_recurrent_enabled', '_native_prefill_fp16_recurrent_requested', '_native_prefill_global_fp16_accum_enabled', '_native_prefill_fused_clampw_scan_enabled', '_native_prefill_fused_output_enabled', '_native_prefill_fused_output_project_block_m', '_native_prefill_fused_output_project_enabled', '_native_prefill_fused_residual_gemm_enabled', '_native_prefill_fused_scan_enabled', '_native_prefill_fused_scan_output_enabled', '_native_prefill_fused_sequence_ffn_enabled', '_native_prefill_fused_shift_mix_enabled', '_native_prefill_fused_state_prep_enabled', '_native_prefill_fused_state_scan_enabled', '_native_prefill_fused_wavg_lora_blocks', '_native_prefill_fused_wavg_lora_enabled', '_native_prefill_model_shape_selected', '_native_prefill_policy_model_shape_selected', '_native_prefill_scan_block_m', '_native_prefill_scan_num_warps', '_native_prefill_self_chunk_enabled', '_native_prefill_self_chunk_h_tiles', '_native_prefill_self_chunk_safe_gate', '_native_prefill_self_chunk_size', '_native_prefill_sequence_ffn_blocks', '_native_prefill_sequence_ffn_launch', '_native_prefill_shift_mix_layers', '_native_prefill_shift_mix_num_warps', '_native_prefill_stacked_rkv_enabled', '_native_prefill_state_prep_layers', '_native_prefill_state_prep_w_dtype', '_recurrent_update_batched', 'dplr_chunk_scan', 'env_flag', 'fused_attn_output_prepare', 'fused_attn_output_project', 'fused_attn_sequence_shift_mix', 'fused_bnb8_attn_sequence_mix_quant', 'fused_bnb8_ffn_sequence_mix_quant', 'fused_ffn_sequence_shift_mix', 'fused_prefill_kv_kk_prep', 'fused_prefill_state_prep', 'fused_recurrent_scan', 'fused_recurrent_scan_clampw', 'fused_recurrent_scan_output_prepare', 'fused_recurrent_scan_state_prep', 'fused_relu_square', 'fused_relu_square_available', 'fused_sequence_ffn', 'fused_wavg_lora', 'native_fp16_sequence', 'self_chunk_rwkv7')


def bind_runtime(runtime: dict[str, object]) -> None:
    for name in _RUNTIME_NAMES:
        if name in runtime and name not in _OWNED_NAMES:
            globals()[name] = runtime[name]
    implementations = globals().get("_IMPLEMENTATIONS", {})
    for name in _EXECUTION_NAMES:
        implementation = implementations.get(name)
        facade_value = runtime.get(name)
        if implementation is None or facade_value is None:
            continue
        if getattr(facade_value, "__wrapped__", None) is implementation:
            globals()[name] = implementation
        else:
            globals()[name] = facade_value


def _native_prefill_linear(
    x: torch.Tensor,
    operand,
    bias=None,
    *,
    allow_fp16_accumulation: bool = False,
) -> torch.Tensor:
    """Sequence linear supporting dense and HF/native quantized operands."""

    if _graph_linear_is_dense(operand):
        matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        can_select_accumulation = bool(
            allow_fp16_accumulation
            and x.is_cuda
            and x.dtype == torch.float16
            and matmul is not None
            and hasattr(matmul, "allow_fp16_accumulation")
        )
        if not can_select_accumulation:
            return F.linear(x, operand, bias)
        with _FP16_ACCUMULATION_LOCK:
            previous = bool(matmul.allow_fp16_accumulation)
            if not previous:
                matmul.allow_fp16_accumulation = True
            try:
                return F.linear(x, operand, bias)
            finally:
                if not previous:
                    matmul.allow_fp16_accumulation = False
    direct = _bnb8_direct_linear(x, operand)
    if direct is not None:
        return direct
    # Quantized modules retain and apply their own bias. Explicit packed biases
    # belong only to dense low-rank operands.
    return operand(x)

def _native_prefill_linear_add_residual(x, weight, residual):
    """Compute ``residual + linear(x, weight)`` with one GEMM output write."""

    hidden = int(weight.shape[0])
    out = residual.reshape(-1, hidden)
    out.addmm_(
        x.reshape(-1, int(weight.shape[1])),
        weight.t(),
    )
    return out.view_as(residual)

def _native_prefill_project_residual(x, operand, residual):
    """Use GEMM beta=1 for dense weights and a safe add for quant modules."""

    if _graph_linear_is_dense(operand) and _native_prefill_fused_residual_gemm_enabled():
        return _native_prefill_linear_add_residual(x, operand, residual)
    return residual + _native_prefill_linear(x, operand)

def _native_prefill_stacked_rkv_weights(model, packs) -> list[torch.Tensor]:
    """Lazily pack transposed dense R/K/V weights for one bmm per layer.

    The cache is an ordinary Python attribute (not a parameter/buffer), so it
    never changes checkpoints.  Weight data pointers and tensor versions form
    the key, which makes adapter merges or in-place edits rebuild safely.
    """

    signatures = []
    for p in packs:
        rw, kw, vw = p[20], p[21], p[22]
        if not all(isinstance(weight, torch.Tensor) and weight.dim() == 2 for weight in (rw, kw, vw)):
            return []
        signatures.append(
            tuple((int(weight.data_ptr()), int(getattr(weight, "_version", 0))) for weight in (rw, kw, vw))
        )
    key = tuple(signatures)
    cached = getattr(model, "_rwkv7_native_prefill_stacked_rkv_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]
    packed = [torch.stack((p[20].t(), p[21].t(), p[22].t()), dim=0).contiguous() for p in packs]
    setattr(model, "_rwkv7_native_prefill_stacked_rkv_cache", (key, packed))
    return packed

def _native_prefill_scan(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    T: int,
    H: int,
    N: int,
    *,
    w_is_raw: bool = False,
    w_is_log: bool = False,
    use_self_chunk: bool | None = None,
    num_layers: int | None = None,
):
    """Run the recurrent prefill scan, using Triton only when explicitly enabled."""

    if w_is_raw and _native_prefill_fused_clampw_scan_enabled(B, T, H * N, num_layers):
        scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
        out, new_state = fused_recurrent_scan_clampw(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_raw:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

    if use_self_chunk is None:
        use_self_chunk = _native_prefill_self_chunk_enabled(T, N)
    if use_self_chunk:
        chunk_size = _native_prefill_self_chunk_size(B, T)
        if T % chunk_size:
            chunk_size = 16
        out, new_state = self_chunk_rwkv7(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            chunk_size=chunk_size,
            w_is_log=w_is_log,
            safe_gate=_native_prefill_self_chunk_safe_gate(),
            h_tiles=_native_prefill_self_chunk_h_tiles(B, T),
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_log:
        w = torch.exp(w.float())

    if _native_prefill_fused_scan_enabled(B, T, H * N, num_layers):
        scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
        out, new_state = fused_recurrent_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if _native_prefill_dplr_scan_enabled() and T > 1:
        out, new_state = dplr_chunk_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            chunk_size=_native_prefill_dplr_chunk_size(),
        )
        return out.reshape(B, T, H * N), new_state

    cur_state = state
    outs = []
    for t in range(T):
        out, cur_state = _recurrent_update_batched(
            r[:, t],
            w[:, t],
            k[:, t],
            v[:, t],
            kk[:, t],
            a[:, t],
            cur_state,
            B,
            H,
            N,
        )
        outs.append(out)
    return torch.stack(outs, dim=1), cur_state

def _prefill_current_device_impl(
    model,
    ids,
    packs,
    *,
    state=None,
    xpa=None,
    xpf=None,
    logits_to_keep: int | None = 1,
    fp16_elapsed=None,
):
    """Layer-wise native RWKV-7 prefill over a full prompt.

    This is the first production-facing bridge for the fused recurrent scan
    prototype: it computes every layer over `[batch, tokens]` using vectorized
    projections and an optional fused recurrent scan instead of repeatedly
    calling the one-token decode path.  Returned state uses the native layout
    `[B,H,N,N]`; callers that expose HF/FLA cache state should transpose the
    final two dimensions, matching the native-graph decode runner.
    """

    base = model.model
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError("native_jit.prefill expects ids shaped [batch, tokens]")
    B = int(ids.shape[0])
    T = int(ids.shape[1])
    if T <= 0:
        raise ValueError("native_jit.prefill requires at least one token")
    H = int(packs[0][1])
    N = int(packs[0][2])
    attention_hidden = H * N
    residual_hidden = int(packs[0][7].numel())
    dtype = base.embeddings.weight.dtype
    use_fp16_recurrent_requested = bool(
        _native_prefill_fp16_recurrent_requested()
        and native_fp16_sequence is not None
        and dtype == torch.float16
        and N == 64
    )
    state_dtype = torch.float16 if use_fp16_recurrent_requested else torch.float32
    if state is None or xpa is None or xpf is None:
        state, xpa, xpf = _init_batched_from_packs(
            packs,
            B,
            ids.device,
            dtype,
            state_dtype=state_dtype,
        )
    else:
        state = [s.to(device=ids.device, dtype=state_dtype).contiguous() for s in state]
        xpa = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpa]
        xpf = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpf]
    if use_fp16_recurrent_requested:
        if fp16_elapsed is None:
            fp16_elapsed = torch.zeros(B, device=ids.device, dtype=torch.int32)
        elif not (
            fp16_elapsed.is_cuda
            and fp16_elapsed.device == ids.device
            and fp16_elapsed.dtype == torch.int32
            and fp16_elapsed.is_contiguous()
            and int(fp16_elapsed.numel()) == B
        ):
            raise ValueError("fp16_elapsed must be contiguous CUDA int32 [batch]")

    x = F.embedding(ids, base.embeddings.weight).reshape(B, T, residual_hidden)
    v_first_seq = torch.zeros(B, T, attention_hidden, device=ids.device, dtype=dtype)
    use_clampw_scan_requested = not use_fp16_recurrent_requested and _native_prefill_fused_clampw_scan_enabled(
        B,
        T,
        attention_hidden,
        len(packs),
    )
    clampw_scan_used = False
    use_prefill_sequence_ffn = _native_prefill_fused_sequence_ffn_enabled(
        B * T,
        B,
        T,
        residual_hidden,
        len(packs),
        dtype,
    )
    sequence_ffn_blocks = _native_prefill_sequence_ffn_blocks(B * T) if use_prefill_sequence_ffn else None
    sequence_ffn_launch = _native_prefill_sequence_ffn_launch() if use_prefill_sequence_ffn else None
    sequence_ffn_workspace = None
    sequence_attn_mix_workspace = None
    bnb8_attn_mix_workspace = None
    bnb8_attn_quant_workspace = None
    bnb8_attn_scale_workspace = None
    sequence_ffn_mix_workspace = None
    bnb8_ffn_quant_workspace = None
    bnb8_ffn_scale_workspace = None
    self_chunk_used = False
    sequence_ffn_used = False
    use_fp16_accum_ffn_key = _native_prefill_fp16_accum_ffn_key_enabled(
        B,
        T,
        residual_hidden,
        len(packs),
        dtype,
    )
    fp16_accum_ffn_key_layers = (
        _native_prefill_fp16_accum_ffn_key_layers(
            B,
            T,
            residual_hidden,
            len(packs),
        )
        if use_fp16_accum_ffn_key
        else set()
    )
    fp16_accum_ffn_key_used = False
    use_prefill_shift_mix = _native_prefill_fused_shift_mix_enabled(
        B, T, residual_hidden, len(packs)
    )
    strict_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag("RWKV7_NATIVE_PREFILL_SHIFT_MIX_STRICT_FP16", False)
    )
    strict_attn_default = _native_prefill_policy_model_shape_selected(
        "prefill_attn_shift_mix_strict_fp16_model_shapes",
        B,
        T,
        residual_hidden,
        len(packs),
    )
    strict_attn_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag(
            "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16",
            strict_shift_mix_fp16 or strict_attn_default,
        )
        and _native_prefill_model_shape_selected(
            "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_STRICT_FP16_MODEL_SHAPES",
            "prefill_attn_shift_mix_strict_fp16_model_shapes",
            B,
            T,
            residual_hidden,
            len(packs),
        )
    )
    strict_ffn_default = _native_prefill_policy_model_shape_selected(
        "prefill_ffn_shift_mix_strict_fp16_model_shapes",
        B,
        T,
        residual_hidden,
        len(packs),
    )
    strict_ffn_shift_mix_fp16 = bool(
        dtype == torch.float16
        and env_flag(
            "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16",
            strict_shift_mix_fp16 or strict_ffn_default,
        )
        and _native_prefill_model_shape_selected(
            "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_STRICT_FP16_MODEL_SHAPES",
            "prefill_ffn_shift_mix_strict_fp16_model_shapes",
            B,
            T,
            residual_hidden,
            len(packs),
        )
    )
    attn_shift_mix_block_size = _native_prefill_attn_shift_mix_block_size(
        strict_attn_shift_mix_fp16,
        B,
        T,
        residual_hidden,
        len(packs),
    )
    attn_shift_mix_num_warps = _native_prefill_shift_mix_num_warps(
        "ATTN", B, T, residual_hidden, len(packs)
    )
    ffn_shift_mix_block_size = _native_prefill_ffn_shift_mix_block_size(
        B, T, residual_hidden, len(packs)
    )
    ffn_shift_mix_num_warps = _native_prefill_shift_mix_num_warps(
        "FFN", B, T, residual_hidden, len(packs)
    )
    prefill_shift_mix_layers = (
        _native_prefill_shift_mix_layers(B, T, len(packs))
        if use_prefill_shift_mix
        else set()
    )
    use_prefill_state_prep = _native_prefill_fused_state_prep_enabled(
        B, T, residual_hidden, len(packs)
    )
    use_prefill_output = _native_prefill_fused_output_enabled(
        B, T, residual_hidden, len(packs)
    )
    prefill_state_prep_layers = (
        _native_prefill_state_prep_layers(B, T, residual_hidden, len(packs))
        if use_prefill_state_prep
        else set()
    )
    capture_layer_outputs = env_flag(
        "RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS",
        False,
    )
    layer_outputs = [] if capture_layer_outputs else None
    stacked_rkv_weights = (
        _native_prefill_stacked_rkv_weights(model, packs)
        if _native_prefill_stacked_rkv_enabled(B * T, B, T, residual_hidden, len(packs))
        else None
    )
    stacked_rkv_used = False
    wavg_lora_used = False

    for p in packs:
        (i, H, N, eps, has_pre,
         pre_w, pre_b, an_w, an_b, fn_w, fn_b,
         x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
         Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
         gn_w, gn_b, fx_k, fK, fV, _RKVw) = p
        layer_idx = int(i)
        H = int(H)
        N = int(N)
        attention_hidden = H * N
        residual_hidden = int(an_w.numel())
        use_fp16_recurrent = _native_prefill_fp16_recurrent_enabled(state[layer_idx])
        use_layer_state_prep = bool(
            use_prefill_state_prep
            and (
                prefill_state_prep_layers is None
                or layer_idx in prefill_state_prep_layers
            )
        )
        use_layer_shift_mix = bool(
            use_prefill_shift_mix
            and (
                prefill_shift_mix_layers is None
                or layer_idx in prefill_shift_mix_layers
            )
        )
        use_layer_attn_shift_mix = bool(
            use_layer_shift_mix
            and env_flag("RWKV7_NATIVE_PREFILL_FUSED_ATTN_SHIFT_MIX", True)
        )
        use_layer_ffn_shift_mix = bool(
            use_layer_shift_mix
            and env_flag("RWKV7_NATIVE_PREFILL_FUSED_FFN_SHIFT_MIX", True)
        )
        residual = F.layer_norm(x, [residual_hidden], pre_w, pre_b, 1e-5) if int(has_pre) == 1 else x
        h = F.layer_norm(residual, [residual_hidden], an_w, an_b, 1e-5)
        defer_state_sigmoid = bool(
            not use_fp16_recurrent
            and use_layer_state_prep
            and not _native_prefill_fused_state_scan_enabled(B)
            and not use_clampw_scan_requested
        )
        state_sigmoid_is_raw = False
        use_sequence_attn_mix = (
            use_layer_attn_shift_mix and fused_attn_sequence_shift_mix is not None
        )
        v_gate = None
        prequantized_rkv = None
        use_bnb8_rkv_mix = bool(
            use_sequence_attn_mix and _bnb8_rkv_mix_quant_enabled(Rw, Kw, Vw)
        )
        if use_bnb8_rkv_mix:
            (
                qr, sr, qk, sk, qv, sv,
                xw, xv, xa, xg, next_xpa,
                bnb8_attn_mix_workspace,
                bnb8_attn_quant_workspace,
                bnb8_attn_scale_workspace,
            ) = fused_bnb8_attn_sequence_mix_quant(
                h,
                xpa[layer_idx],
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                mix_workspace=bnb8_attn_mix_workspace,
                quant_workspace=bnb8_attn_quant_workspace,
                scale_workspace=bnb8_attn_scale_workspace,
                block=_native_bnb8_policy_block(
                    "RWKV7_NATIVE_BNB8_ATTN_MIX_BLOCK",
                    "native_bnb8_attn_mix_block",
                    1024,
                ),
            )
            prequantized_rkv = (qr, sr, qk, sk, qv, sv)
            xr = xk = None
        elif use_sequence_attn_mix:
            if sequence_attn_mix_workspace is None:
                sequence_attn_mix_workspace = torch.empty(
                    (6, B, T, residual_hidden), device=h.device, dtype=h.dtype
                )
            xr, xw, xk, xv, xa, xg, next_xpa = fused_attn_sequence_shift_mix(
                h,
                xpa[layer_idx],
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
                block_size=attn_shift_mix_block_size,
                num_warps=attn_shift_mix_num_warps,
                workspace=sequence_attn_mix_workspace,
                strict_fp16_rounding=strict_attn_shift_mix_fp16,
            )
        else:
            prev_h = torch.cat([xpa[layer_idx].view(B, 1, residual_hidden), h[:, :-1, :]], dim=1)
            xx = prev_h - h
            xr = h + xx * x_r.view(1, 1, residual_hidden)
            xw = h + xx * x_w.view(1, 1, residual_hidden)
            xk = h + xx * x_k.view(1, 1, residual_hidden)
            xv = h + xx * x_v.view(1, 1, residual_hidden)
            xa = h + xx * x_a.view(1, 1, residual_hidden)
            xg = h + xx * x_g.view(1, 1, residual_hidden)

        use_stacked_rkv = False
        if prequantized_rkv is not None:
            qr, sr, qk, sk, qv, sv = prequantized_rkv
            r = _bnb8_prequant_linear(qr, sr, Rw, dtype=h.dtype, output_shape=(B, T))
            k = _bnb8_prequant_linear(qk, sk, Kw, dtype=h.dtype, output_shape=(B, T))
            v = _bnb8_prequant_linear(qv, sv, Vw, dtype=h.dtype, output_shape=(B, T))
        elif stacked_rkv_weights:
            row_values = B * T * residual_hidden
            use_stacked_rkv = bool(
                xr.is_contiguous()
                and xk.is_contiguous()
                and xv.is_contiguous()
                and xr.untyped_storage().data_ptr() == xk.untyped_storage().data_ptr()
                and xr.untyped_storage().data_ptr() == xv.untyped_storage().data_ptr()
                and int(xk.storage_offset()) == int(xr.storage_offset()) + row_values
                and int(xv.storage_offset()) == int(xr.storage_offset()) + 2 * row_values
            )
        if prequantized_rkv is not None:
            pass
        elif use_stacked_rkv:
            stacked_rkv_used = True
            rkv_inputs = xr.as_strided(
                (3, B * T, residual_hidden),
                (B * T * residual_hidden, residual_hidden, 1),
            )
            rkv = torch.bmm(rkv_inputs, stacked_rkv_weights[layer_idx])
            r = rkv[0].view(B, T, attention_hidden)
            k = rkv[1].view(B, T, attention_hidden)
            v = rkv[2].view(B, T, attention_hidden)
        else:
            r = _native_prefill_linear(xr, Rw)
            k = _native_prefill_linear(xk, Kw)
            v = _native_prefill_linear(xv, Vw)
        use_prefill_wavg_lora = bool(
            (not use_fp16_recurrent or T > 16)
            and layer_idx > 0
            and _native_prefill_fused_wavg_lora_enabled(B * T)
            and _graph_linears_are_dense(w1, w2, a1, a2, g1, g2, v1, v2)
        )
        if use_prefill_wavg_lora:
            wavg_lora_used = True
            block_m, block_r, block_k = _native_prefill_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.reshape(B * T, residual_hidden),
                xa.reshape(B * T, residual_hidden),
                xg.reshape(B * T, residual_hidden),
                xv.reshape(B * T, residual_hidden),
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
            )
            w = w.view(B, T, attention_hidden)
            a = torch.sigmoid(a.view(B, T, attention_hidden))
            g = g.view(B, T, attention_hidden)
            v_gate = v_gate.view(B, T, attention_hidden)
        else:
            w_mid = _native_prefill_linear(xw, w1)
            w_mid.tanh_()
            if use_fp16_recurrent and T <= 16:
                w = _native_prefill_linear(w_mid, w2)
                fp16_w0 = w0.reshape(-1).contiguous()
            else:
                w = _native_prefill_linear(w_mid, w2, w0)
                fp16_w0 = None
            a_mid = _native_prefill_linear(xa, a1)
            a = _native_prefill_linear(a_mid, a2, a0)
            if not defer_state_sigmoid:
                a.sigmoid_()
            else:
                state_sigmoid_is_raw = True
            g_mid = _native_prefill_linear(xg, g1)
            g_mid.sigmoid_()
            g = _native_prefill_linear(g_mid, g2)
            if layer_idx != 0:
                v_mid = _native_prefill_linear(xv, v1)
                v_gate = _native_prefill_linear(v_mid, v2, v0)
                if not defer_state_sigmoid:
                    v_gate.sigmoid_()
        use_fused_scan_output = bool(
            not use_fp16_recurrent and _native_prefill_fused_scan_output_enabled()
        )
        use_self_chunk = _native_prefill_self_chunk_enabled(
            T,
            N,
            B,
            attention_hidden,
            len(packs),
        ) and not use_fused_scan_output and not use_fp16_recurrent
        self_chunk_used = bool(self_chunk_used or use_self_chunk)
        self_chunk_w_is_log = False
        use_clampw_scan = use_clampw_scan_requested and not use_fused_scan_output
        use_fused_state_scan = bool(
            not use_fp16_recurrent
            and _native_prefill_fused_state_scan_enabled(B)
            and not use_fused_scan_output
        )
        if use_clampw_scan and use_layer_state_prep and fused_prefill_kv_kk_prep is None:
            use_clampw_scan = False
        state_scan_done = False
        if use_fused_state_scan:
            scan_block_m = _native_prefill_scan_block_m(N, B, T, H * N)
            scan_num_warps = _native_prefill_scan_num_warps(N, scan_block_m)
            if layer_idx == 0:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    block_n=N,
                    block_m=scan_block_m,
                    num_warps=scan_num_warps,
                )
                v_first_seq = v.reshape(B, T, attention_hidden)
            else:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=scan_block_m,
                    num_warps=scan_num_warps,
                )
            out = out.reshape(B, T, attention_hidden)
            k = k.reshape(B, T, attention_hidden)
            v = v.reshape(B, T, attention_hidden)
            state_scan_done = True
        elif use_layer_state_prep and use_fp16_recurrent:
            if fused_prefill_kv_kk_prep is None:
                raise RuntimeError(
                    "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP requires the fused K/V/KK prep kernel"
                )
            if layer_idx == 0:
                k, v, kk = fused_prefill_kv_kk_prep(
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    num_heads=H,
                    head_dim=N,
                )
                v_first_seq = v
            else:
                k, v, kk = fused_prefill_kv_kk_prep(
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    v_first=v_first_seq,
                    v_gate=v_gate,
                    num_heads=H,
                    head_dim=N,
                )
        elif use_layer_state_prep and not use_fp16_recurrent:
            self_chunk_w_is_log = bool(use_self_chunk and not use_clampw_scan)
            if use_clampw_scan:
                if layer_idx == 0:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        num_heads=H,
                        head_dim=N,
                    )
                    v_first_seq = v
                else:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        v_first=v_first_seq,
                        v_gate=v_gate,
                        num_heads=H,
                        head_dim=N,
                    )
            elif layer_idx == 0:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                    w_transform="log_decay" if use_self_chunk else "decay",
                    a_is_raw=state_sigmoid_is_raw,
                    v_gate_is_raw=state_sigmoid_is_raw,
                )
                v_first_seq = v
            else:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    v_first=v_first_seq,
                    v_gate=v_gate,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                    w_transform="log_decay" if use_self_chunk else "decay",
                    a_is_raw=state_sigmoid_is_raw,
                    v_gate_is_raw=state_sigmoid_is_raw,
                )
        else:
            kk = F.normalize(
                (k * k_k.view(1, 1, attention_hidden)).view(B, T, H, N),
                dim=-1,
                p=2.0,
            ).view(B, T, attention_hidden)
            k = k * (1 + (a - 1) * k_a.view(1, 1, attention_hidden))
            if layer_idx == 0:
                v_first_seq = v
            else:
                v = v + (v_first_seq - v) * v_gate
            if not use_clampw_scan and not use_fp16_recurrent:
                w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

        if use_fp16_recurrent:
            assert fp16_elapsed is not None
            out = native_fp16_sequence(
                r.view(B, T, H, N).contiguous(),
                w.view(B, T, H, N).contiguous(),
                k.view(B, T, H, N).contiguous(),
                v.view(B, T, H, N).contiguous(),
                (-kk).view(B, T, H, N).contiguous(),
                (kk * a).view(B, T, H, N).contiguous(),
                state[layer_idx],
                fp16_elapsed,
                w0=fp16_w0,
            ).reshape(B, T, attention_hidden)
            new_state = state[layer_idx]
            state_scan_done = True
        elif use_fused_scan_output:
            out, new_state = fused_recurrent_scan_output_prepare(
                r.view(B, T, H, N),
                w.view(B, T, H, N),
                k.view(B, T, H, N),
                v.view(B, T, H, N),
                kk.view(B, T, H, N),
                a.view(B, T, H, N),
                state[layer_idx],
                g.view(B, T, H, N),
                r_k,
                gn_w,
                gn_b,
                eps=eps,
                block_n=N,
            )
            out = out.reshape(B, T, attention_hidden)
        elif not state_scan_done:
            clampw_scan_used = bool(clampw_scan_used or use_clampw_scan)
            out, new_state = _native_prefill_scan(
                r, w, k, v, kk, a, state[layer_idx], B, T, H, N,
                w_is_raw=use_clampw_scan,
                w_is_log=self_chunk_w_is_log,
                use_self_chunk=use_self_chunk,
                num_layers=len(packs),
            )
        out_projected = False
        if use_fused_scan_output:
            pass
        elif _native_prefill_fused_output_project_enabled() and _graph_linear_is_dense(Ow):
            out = fused_attn_output_project(
                out.reshape(B * T, attention_hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, attention_hidden),
                r_k,
                gn_w,
                gn_b,
                Ow,
                None,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
                block_m=_native_prefill_fused_output_project_block_m(),
            ).view(B, T, residual_hidden)
            out_projected = True
        elif use_prefill_output:
            out = fused_attn_output_prepare(
                out.reshape(B * T, attention_hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, attention_hidden),
                r_k,
                gn_w,
                gn_b,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
            ).view(B, T, attention_hidden)
        else:
            out = F.group_norm(
                out.reshape(B * T, attention_hidden), H, gn_w, gn_b, eps
            ).view(B, T, attention_hidden)
            sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
            out = (
                out + (sk * v.view(B, T, H, N)).view(B, T, attention_hidden)
            ) * g
        if not out_projected:
            x = _native_prefill_project_residual(out, Ow, residual)
        else:
            x = residual + out
        xpa[layer_idx] = (
            next_xpa
            if use_sequence_attn_mix
            else h[:, -1, :].contiguous()
        )
        state[layer_idx] = new_state.contiguous()

        residual = x
        h2 = F.layer_norm(x, [residual_hidden], fn_w, fn_b, 1e-5)
        use_layer_sequence_ffn = bool(
            use_prefill_sequence_ffn and _graph_linears_are_dense(fK, fV)
        )
        if use_layer_sequence_ffn:
            sequence_ffn_used = True
            assert sequence_ffn_blocks is not None
            assert sequence_ffn_launch is not None
            if sequence_ffn_workspace is None:
                sequence_ffn_workspace = (
                    torch.empty((B * T, residual_hidden), device=h2.device, dtype=h2.dtype),
                    torch.empty((B * T, int(fK.shape[0])), device=h2.device, dtype=h2.dtype),
                )
            ffn_out, next_xpf = fused_sequence_ffn(
                h2,
                xpf[layer_idx],
                fx_k,
                fK,
                fV,
                block_m=sequence_ffn_blocks[0],
                block_n=sequence_ffn_blocks[1],
                key_block_k=sequence_ffn_blocks[2],
                value_block_k=sequence_ffn_blocks[3],
                group_m=sequence_ffn_blocks[4],
                num_stages=sequence_ffn_launch[0],
                num_warps=sequence_ffn_launch[1],
                workspace=sequence_ffn_workspace,
            )
            x = residual + ffn_out
        else:
            ffn_up_prequantized = False
            if use_layer_ffn_shift_mix and _bnb8_ffn_mix_quant_enabled(fK):
                (
                    qfk,
                    sfk,
                    next_xpf,
                ) = fused_bnb8_ffn_sequence_mix_quant(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    quant_workspace=bnb8_ffn_quant_workspace,
                    scale_workspace=bnb8_ffn_scale_workspace,
                    block=_native_bnb8_policy_block(
                        "RWKV7_NATIVE_BNB8_FFN_MIX_BLOCK",
                        "native_bnb8_ffn_mix_block",
                        1024,
                    ),
                )
                bnb8_ffn_quant_workspace = qfk
                bnb8_ffn_scale_workspace = sfk
                fk = _bnb8_prequant_linear(qfk, sfk, fK, dtype=h2.dtype, output_shape=(B, T))
                ffn_up_prequantized = True
            elif use_layer_ffn_shift_mix and fused_ffn_sequence_shift_mix is not None:
                if sequence_ffn_mix_workspace is None:
                    sequence_ffn_mix_workspace = torch.empty_like(h2)
                fk, next_xpf = fused_ffn_sequence_shift_mix(
                    h2,
                    xpf[layer_idx],
                    fx_k,
                    block_size=ffn_shift_mix_block_size,
                    num_warps=ffn_shift_mix_num_warps,
                    workspace=sequence_ffn_mix_workspace,
                    strict_fp16_rounding=strict_ffn_shift_mix_fp16,
                )
            else:
                prev_h2 = torch.cat(
                    [xpf[layer_idx].view(B, 1, residual_hidden), h2[:, :-1, :]],
                    dim=1,
                )
                fxx = prev_h2 - h2
                fk = h2 + fxx * fx_k.view(1, 1, residual_hidden)
                next_xpf = h2[:, -1, :].contiguous()
            fused_quant_ffn = None
            if not ffn_up_prequantized:
                fused_quant = getattr(fK, "rwkv7_forward_ffn", None)
                if callable(fused_quant):
                    fused_quant_ffn = fused_quant(fk, fV, residual)
            fused_up_relu2 = False
            if fused_quant_ffn is None and not ffn_up_prequantized:
                fused = getattr(fK, "rwkv7_forward_relu2", None)
                fused_up_relu2 = bool(
                    getattr(fK, "fused_relu2", False) and callable(fused)
                )
                if fused_up_relu2:
                    fk = fused(fk)
                else:
                    fp16_accum_ffn_key_layer = bool(
                        use_fp16_accum_ffn_key and _graph_linear_is_dense(fK)
                        and (
                            fp16_accum_ffn_key_layers is None
                            or layer_idx in fp16_accum_ffn_key_layers
                        )
                    )
                    fp16_accum_ffn_key_used = bool(
                        fp16_accum_ffn_key_used or fp16_accum_ffn_key_layer
                    )
                    fk = _native_prefill_linear(
                        fk,
                        fK,
                        allow_fp16_accumulation=fp16_accum_ffn_key_layer,
                    )
            if fused_quant_ffn is not None:
                x = fused_quant_ffn
            else:
                fused_bnb8_ffn = (
                    None
                    if fused_up_relu2
                    else _bnb8_direct_relu_square_linear(fk, fV)
                )
                if fused_bnb8_ffn is not None:
                    x = residual + fused_bnb8_ffn
                elif fused_up_relu2:
                    x = _native_prefill_project_residual(fk, fV, residual)
                elif (
                    use_layer_ffn_shift_mix
                    and fused_relu_square is not None
                    and fused_relu_square_available is not None
                    and fused_relu_square_available()
                ):
                    fk = fused_relu_square(fk)
                    x = _native_prefill_project_residual(fk, fV, residual)
                else:
                    fk = torch.relu(fk) ** 2
                    x = _native_prefill_project_residual(fk, fV, residual)
        xpf[layer_idx] = next_xpf
        if layer_outputs is not None:
            layer_outputs.append(x[:, -1, :].detach().clone())

    keep = T if logits_to_keep is None or int(logits_to_keep) <= 0 else min(int(logits_to_keep), T)
    # Recurrent/shift state is already complete. Final norm is consumed only
    # by the language head, so serving requests that ask for the last logits
    # must not normalize and materialize the entire prompt sequence.
    x_for_logits = x if keep == T else x[:, -keep:, :]
    x_for_logits = F.layer_norm(
        x_for_logits,
        [residual_hidden],
        base.norm.weight,
        base.norm.bias,
        1e-5,
    )
    logits = _lm_head(model, x_for_logits)
    setattr(
        model,
        "_rwkv7_native_prefill_clampw_scan_effective",
        bool(clampw_scan_used),
    )
    setattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", bool(stacked_rkv_used))
    setattr(model, "_rwkv7_native_prefill_wavg_lora_effective", bool(wavg_lora_used))
    setattr(model, "_rwkv7_native_prefill_self_chunk_effective", bool(self_chunk_used))
    setattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", bool(sequence_ffn_used))
    setattr(
        model,
        "_rwkv7_native_prefill_fp16_accum_ffn_key_effective",
        bool(fp16_accum_ffn_key_used),
    )
    setattr(
        model,
        "_rwkv7_native_prefill_fp16_recurrent_effective",
        bool(use_fp16_recurrent_requested),
    )
    setattr(model, "_rwkv7_native_prefill_layer_outputs", layer_outputs)
    return logits, state, xpa, xpf


def _prefill_current_device(
    model,
    ids,
    packs,
    *,
    state=None,
    xpa=None,
    xpf=None,
    logits_to_keep: int | None = 1,
    fp16_elapsed=None,
):
    """Run exact-shape prefill under a scoped full-GEMM fp16 policy."""

    ids_shape = tuple(ids.shape)
    batch_size = int(ids_shape[0]) if len(ids_shape) == 2 else 1
    prompt_tokens = int(ids_shape[-1])
    hidden_size = int(packs[0][7].numel())
    num_layers = len(packs)
    dtype = model.model.embeddings.weight.dtype
    selected = _native_prefill_global_fp16_accum_enabled(
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
        dtype,
    )
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if not selected or matmul is None:
        setattr(model, "_rwkv7_native_prefill_global_fp16_accum_effective", False)
        return _prefill_current_device_impl(
            model,
            ids,
            packs,
            state=state,
            xpa=xpa,
            xpf=xpf,
            logits_to_keep=logits_to_keep,
            fp16_elapsed=fp16_elapsed,
        )

    with _FP16_ACCUMULATION_LOCK:
        previous = bool(matmul.allow_fp16_accumulation)
        if not previous:
            matmul.allow_fp16_accumulation = True
        try:
            result = _prefill_current_device_impl(
                model,
                ids,
                packs,
                state=state,
                xpa=xpa,
                xpf=xpf,
                logits_to_keep=logits_to_keep,
                fp16_elapsed=fp16_elapsed,
            )
        finally:
            if not previous:
                matmul.allow_fp16_accumulation = False
    setattr(model, "_rwkv7_native_prefill_global_fp16_accum_effective", True)
    return result

__all__ = ['_native_prefill_linear', '_native_prefill_linear_add_residual', '_native_prefill_project_residual', '_native_prefill_stacked_rkv_weights', '_native_prefill_scan', '_prefill_current_device']

_IMPLEMENTATIONS = {name: globals()[name] for name in __all__}
