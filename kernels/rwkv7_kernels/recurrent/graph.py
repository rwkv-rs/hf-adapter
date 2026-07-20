"""Exact RWKV-7 recurrence accelerated with reusable CUDA graphs."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch


_MAX_GRAPHS = 2
_LOCK = RLock()
_GRAPHS: "OrderedDict[tuple[Any, ...], _GraphEntry]" = OrderedDict()


def _reference_recurrent(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
):
    batch, time = receptance.shape[:2]
    # Keep the same per-sample arithmetic contract as the public reference
    # operator. Batched matmul is mathematically equivalent, but FP16 CUDA
    # kernels may choose a different accumulation strategy for B=1 and B=8;
    # that is enough to flip close lm_eval choices. CUDA Graph capture removes
    # launch overhead without changing this batch-invariant execution rule.
    def run_sample(batch_index):
        state = initial_state[batch_index : batch_index + 1]
        sample_mask = (
            None
            if attention_mask is None
            else attention_mask[batch_index : batch_index + 1]
        )
        outputs = []
        for token in range(time):
            r_t = receptance[batch_index : batch_index + 1, token]
            w_t = decay[batch_index : batch_index + 1, token].to(
                dtype=state.dtype
            )
            k_t = key[batch_index : batch_index + 1, token]
            v_t = value[batch_index : batch_index + 1, token]
            a_t = a[batch_index : batch_index + 1, token]
            b_t = b[batch_index : batch_index + 1, token]
            state_vk = state.transpose(-1, -2)
            ab = a_t.unsqueeze(-1) @ b_t.unsqueeze(-2)
            vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
            candidate_vk = (
                state_vk * w_t.unsqueeze(-2)
                + state_vk @ ab.to(dtype=state.dtype)
                + vk.to(dtype=state.dtype)
            )
            candidate = candidate_vk.transpose(-1, -2)
            output = (
                candidate_vk.to(dtype=r_t.dtype) @ r_t.unsqueeze(-1)
            ).squeeze(-1)
            if sample_mask is not None:
                active = sample_mask[:, token]
                state = torch.where(active.view(1, 1, 1, 1), candidate, state)
                output = torch.where(
                    active.view(1, 1, 1), output, torch.zeros_like(output)
                )
            else:
                state = candidate
            outputs.append(output.to(dtype=value.dtype))
        return torch.stack(outputs, dim=1), state

    samples = [run_sample(batch_index) for batch_index in range(batch)]
    return (
        torch.cat([sample[0] for sample in samples], dim=0),
        torch.cat([sample[1] for sample in samples], dim=0),
    )


def _empty_like(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        value.size(), value.stride(), dtype=value.dtype, device=value.device
    )


@dataclass
class _GraphEntry:
    inputs: tuple[torch.Tensor, ...]
    mask: torch.Tensor | None
    graph: torch.cuda.CUDAGraph
    output: tuple[torch.Tensor, torch.Tensor]

    @classmethod
    def capture(cls, tensors, attention_mask):
        static = tuple(_empty_like(value) for value in tensors)
        static_mask = (
            None if attention_mask is None else _empty_like(attention_mask)
        )
        for target, source in zip(static, tensors):
            target.copy_(source)
        if static_mask is not None:
            static_mask.copy_(attention_mask)

        current = torch.cuda.current_stream(tensors[0].device)
        warmup = torch.cuda.Stream(device=tensors[0].device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(2):
                result = _reference_recurrent(*static, static_mask)
                del result
        current.wait_stream(warmup)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            output = _reference_recurrent(*static, static_mask)
        return cls(static, static_mask, graph, output)

    def run(self, tensors, attention_mask):
        # Static buffers may have been allocated while the caller was inside
        # ``torch.inference_mode``. PyTorch 2.5 rejects mutating such tensors
        # from an ordinary/no-grad context, so replay preparation must retain
        # inference mode as well.
        with torch.inference_mode():
            for target, source in zip(self.inputs, tensors):
                target.copy_(source)
            if self.mask is not None:
                self.mask.copy_(attention_mask)
            self.graph.replay()
        # The graph owns static output buffers.  Callers and HF caches must be
        # free to retain results after this entry is replayed by another layer.
        return self.output[0].clone(), self.output[1].clone()


def _key(tensors, attention_mask):
    first = tensors[0]
    return (
        first.device.type,
        first.device.index,
        tuple((tuple(value.shape), value.dtype, value.stride()) for value in tensors),
        None
        if attention_mask is None
        else (tuple(attention_mask.shape), attention_mask.dtype, attention_mask.stride()),
    )


def probe_recurrent_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
) -> dict[str, Any]:
    implementation = "torch-cuda-graph-reference-v1"
    tensors = (receptance, decay, key, value, a, b, initial_state)
    if not torch.cuda.is_available() or not all(value.is_cuda for value in tensors):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "the v1 graph backend requires CUDA tensors",
        }
    if any(value.requires_grad for value in tensors):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "the v1 graph backend is inference-only",
        }
    if receptance.ndim != 4 or initial_state.ndim != 4:
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "rank-four recurrent inputs and state are required",
        }
    expected = tuple(receptance.shape)
    if any(tuple(value.shape) != expected for value in (decay, key, a, b)):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "r/w/k/a/b must have identical shapes",
        }
    if tuple(value.shape[:3]) != tuple(receptance.shape[:3]):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "value must share the [B,T,H] dimensions",
        }
    if receptance.dtype != torch.float16:
        return {
            "supported": False,
            "implementation": implementation,
            "reason": f"unsupported input dtype {receptance.dtype}",
        }
    if int(receptance.shape[-1]) != 64 or int(value.shape[-1]) != 64:
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "the promoted v1 shape requires K=V=64",
        }
    if initial_state.dtype != torch.float32:
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "the promoted v1 graph requires FP32 state",
        }
    if any(item.dtype != torch.float16 for item in (key, value, a, b)):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "r/k/v/a/b must all use FP16",
        }
    if decay.dtype not in (torch.float16, torch.float32):
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "decay must use FP16 or FP32",
        }
    if attention_mask is not None and not attention_mask.is_cuda:
        return {
            "supported": False,
            "implementation": implementation,
            "reason": "attention_mask must be on CUDA",
        }
    return {
        "supported": True,
        "implementation": implementation,
        "reason": "exact FP16 inference recurrence is CUDA-graph compatible",
    }


def recurrent_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
):
    support = probe_recurrent_v1(
        receptance, decay, key, value, a, b, initial_state, attention_mask
    )
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    tensors = (receptance, decay, key, value, a, b, initial_state)
    key_value = _key(tensors, attention_mask)
    with _LOCK:
        entry = _GRAPHS.get(key_value)
        if entry is None:
            entry = _GraphEntry.capture(tensors, attention_mask)
            _GRAPHS[key_value] = entry
            while len(_GRAPHS) > _MAX_GRAPHS:
                _GRAPHS.popitem(last=False)
        else:
            _GRAPHS.move_to_end(key_value)
        return entry.run(tensors, attention_mask)


__all__ = ["probe_recurrent_v1", "recurrent_v1"]
