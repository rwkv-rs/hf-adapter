# coding=utf-8
"""Native JIT model-pack extraction and recurrent-state allocation.

Pack construction is model-container work, not token-loop math.  Callers pass
current policy and operand adapters explicitly so the stable native_jit facade
retains its historical monkeypatch surface.
"""
from __future__ import annotations

import torch


def _kernel_bias(bias, reference: torch.Tensor):
    """Return an ordinary packed bias in the native activation dtype."""

    if bias is None:
        return None
    return bias.to(device=reference.device, dtype=reference.dtype)


def _decay_bias(bias, reference: torch.Tensor):
    """Keep RWKV-7's decay bias in FP32 inside private native packs.

    The clean model adds ``w0`` after the low-rank projection has been
    promoted to FP32.  Casting this particular bias to FP16/BF16 changes the
    decay before every recurrent update and compounds across layers/tokens.
    Other LoRA biases retain the historical activation-dtype packing.
    """

    if bias is None:
        return None
    return bias.to(device=reference.device, dtype=torch.float32)


def _pre_norm_pack(layer, *, reference: torch.Tensor, hidden_size: int):
    """Return packed LayerNorm tensors; an explicit ``Identity`` is no norm."""

    pre_norm = getattr(layer, "pre_norm", None)
    weight = getattr(pre_norm, "weight", None)
    is_layer_norm = isinstance(pre_norm, torch.nn.LayerNorm)
    has_layer_norm_contract = isinstance(weight, torch.Tensor) and hasattr(
        pre_norm, "eps"
    )
    if is_layer_norm or has_layer_norm_contract:
        if not isinstance(weight, torch.Tensor):
            raise TypeError("RWKV7 native packing requires tensor pre_norm.weight")
        bias = getattr(pre_norm, "bias", None)
        if bias is None:
            bias = torch.zeros_like(weight)
        if not isinstance(bias, torch.Tensor):
            raise TypeError("RWKV7 native packing requires tensor pre_norm.bias")
        return weight, bias, 1
    weight = torch.zeros(
        hidden_size, device=reference.device, dtype=reference.dtype
    )
    return weight, torch.zeros_like(weight), 0


def extract_dense_packs(model, *, rkv_policy: str):
    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    attention_hidden = int(getattr(layers[0].attn, "attention_hidden_size", H * N))
    dense_ref = model.model.embeddings.weight
    stack_rkv = rkv_policy == "vkwr_auto"
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(attention_hidden, 1, device=ref.device, dtype=ref.dtype)
        v0 = _kernel_bias(vl.lora[2].bias, ref) if vl is not None else torch.zeros(attention_hidden, device=ref.device, dtype=ref.dtype)
        pre_w, pre_b, has_pre = _pre_norm_pack(
            layer,
            reference=ref,
            hidden_size=hidden,
        )
        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            a.r_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
            a.w_lora.lora[0].weight, a.w_lora.lora[2].weight,
            _decay_bias(a.w_lora.lora[2].bias, ref),
            a.a_lora.lora[0].weight, a.a_lora.lora[2].weight,
            _kernel_bias(a.a_lora.lora[2].bias, ref),
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
    stack_rkv = rkv_policy == "vkwr_auto"
    embed_ref = model.model.embeddings.weight
    for i, layer in enumerate(layers):
        if sparse_ffn_low_memory_pack_enabled() and (
            isinstance(layer.ffn.value, torch.nn.Linear)
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
            v0 = _kernel_bias(vl.lora[2].bias, embed_ref)
        else:
            v1 = torch.zeros(1, hidden, device=embed_ref.device, dtype=embed_ref.dtype)
            v2 = torch.zeros(attention_hidden, 1, device=embed_ref.device, dtype=embed_ref.dtype)
            v0 = torch.zeros(attention_hidden, device=embed_ref.device, dtype=embed_ref.dtype)
        pre_w, pre_b, has_pre = _pre_norm_pack(
            layer,
            reference=embed_ref,
            hidden_size=hidden,
        )

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
            _decay_bias(a.w_lora.lora[2].bias, embed_ref),
            graph_linear_operand(a.a_lora.lora[0]),
            graph_linear_operand(a.a_lora.lora[2]),
            _kernel_bias(a.a_lora.lora[2].bias, embed_ref),
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
