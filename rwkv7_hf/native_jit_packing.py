# coding=utf-8
"""Native JIT model-pack extraction and recurrent-state allocation.

Pack construction is model-container work, not token-loop math.  Callers pass
current policy and operand adapters explicitly so the stable native_jit facade
retains its historical monkeypatch surface.
"""
from __future__ import annotations

import torch


def _should_stack_rkv(
    rkv_policy: str,
    hidden_size: int,
    pack_max_hidden: int,
) -> bool:
    """Bound the eager R/K/V copy to validated, memory-safe model widths."""

    return (
        rkv_policy == "vkwr_auto"
        and 0 < int(hidden_size) <= int(pack_max_hidden)
    )


def extract_dense_packs(
    model,
    *,
    rkv_policy: str,
    rkv_pack_max_hidden: int,
):
    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    attention_hidden = int(getattr(layers[0].attn, "attention_hidden_size", H * N))
    dense_ref = model.model.embeddings.weight
    stack_rkv = _should_stack_rkv(rkv_policy, hidden, rkv_pack_max_hidden)
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(attention_hidden, 1, device=ref.device, dtype=ref.dtype)
        v0 = vl.lora[2].bias if vl is not None else torch.zeros(attention_hidden, device=ref.device, dtype=ref.dtype)
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
            torch.stack((a.r_proj.weight.t(), a.k_proj.weight.t(), a.v_proj.weight.t())).contiguous()
            if stack_rkv
            else dense_ref.new_empty((0,)),
        ))
    return packs, H, N, eps


def extract_graph_packs(
    model,
    *,
    rkv_policy: str,
    rkv_pack_max_hidden: int,
    sparse_ffn_low_memory_pack_enabled,
    try_relayout_ffn_value_weight,
    graph_linear_operand,
    graph_linear_is_dense,
):
    """Pack CUDA-graph operands while preserving MM8/MM4 modules.

    Dense models keep the exact historical tensor tuple. Quantized projection
    modules are retained as callable operands and are consumed by the eager
    graph-capture dispatchers below. This function is intentionally separate
    from :func:`extract`: TorchScript decode still requires tensor-only packs.
    """

    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    attention_hidden = int(getattr(layers[0].attn, "attention_hidden_size", H * N))
    stack_rkv = _should_stack_rkv(rkv_policy, hidden, rkv_pack_max_hidden)
    embed_ref = model.model.embeddings.weight
    for i, layer in enumerate(layers):
        if sparse_ffn_low_memory_pack_enabled() and (
            type(layer.ffn.value) is torch.nn.Linear
            and type(layer.ffn.value.weight) is torch.nn.Parameter
            and layer.ffn.value.weight.device.type == "cuda"
            and layer.ffn.value.weight.dtype == torch.float16
        ):
            if model.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK is inference-only"
                )
            try_relayout_ffn_value_weight(layer.ffn.value)
        a = layer.attn
        vl = getattr(a, "v_lora", None)
        if vl is not None:
            v1 = graph_linear_operand(vl.lora[0])
            v2 = graph_linear_operand(vl.lora[2])
            v0 = vl.lora[2].bias
        else:
            v1 = torch.zeros(1, hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            v2 = torch.zeros(attention_hidden, 1, device=embed_ref.device, dtype=embed_ref.dtype)
            v0 = torch.zeros(attention_hidden, device=embed_ref.device, dtype=embed_ref.dtype)
        if hasattr(layer, "pre_norm"):
            pre_w, pre_b, has_pre = layer.pre_norm.weight, layer.pre_norm.bias, 1
        else:
            pre_w = torch.zeros(hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            pre_b = torch.zeros(hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            has_pre = 0

        r_op = graph_linear_operand(a.r_proj)
        k_op = graph_linear_operand(a.k_proj)
        v_op = graph_linear_operand(a.v_proj)
        if stack_rkv and all(graph_linear_is_dense(item) for item in (r_op, k_op, v_op)):
            stacked_rkv = torch.stack((r_op.t(), k_op.t(), v_op.t())).contiguous()
        else:
            stacked_rkv = embed_ref.new_empty((0,))

        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            r_op, k_op, v_op, graph_linear_operand(a.o_proj),
            graph_linear_operand(a.w_lora.lora[0]),
            graph_linear_operand(a.w_lora.lora[2]),
            a.w_lora.lora[2].bias,
            graph_linear_operand(a.a_lora.lora[0]),
            graph_linear_operand(a.a_lora.lora[2]),
            a.a_lora.lora[2].bias,
            v1, v2, v0,
            graph_linear_operand(a.g_lora.lora[0]),
            graph_linear_operand(a.g_lora.lora[2]),
            a.g_norm.weight, a.g_norm.bias,
            layer.ffn.x_k,
            graph_linear_operand(layer.ffn.key),
            graph_linear_operand(layer.ffn.value),
            stacked_rkv,
        ))
    return packs, H, N, eps


def init_state(model, device, dtype):
    layers = model.model.layers
    n = len(layers)
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    hid = layers[0].attn.hidden_size
    attention_hidden = getattr(layers[0].attn, "attention_hidden_size", H * N)
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(attention_hidden, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def init_batched_from_packs(
    packs,
    batch_size: int,
    device,
    dtype,
    *,
    state_dtype=None,
):
    n = len(packs)
    H = int(packs[0][1])
    N = int(packs[0][2])
    hid = int(packs[0][7].numel())
    if state_dtype is None:
        state_dtype = torch.float32
    state = [torch.zeros(batch_size, H, N, N, device=device, dtype=state_dtype) for _ in range(n)]
    xpa = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    return state, xpa, xpf


__all__ = [
    "extract_dense_packs",
    "extract_graph_packs",
    "init_batched_from_packs",
    "init_state",
]
