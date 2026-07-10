"""Fixed-shape CUDA Graph wrapper for the migrated sequence prefill engine.

The runner owns only private static buffers.  Replay returns clones that are
copied into the canonical Hugging Face cache by ``model_dispatcher``; graph
buffers and the historical ``[V,K]`` layout never escape this package.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch

from .kernel_policy import current_kernel_policy, env_flag
from . import native_jit


_SIGNATURE_PREFIXES = (
    "RWKV7_NATIVE_PREFILL_",
    "RWKV7_NATIVE_BNB8_",
    "RWKV7_FUSED_",
)


def prefill_graph_runtime_signature() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith(_SIGNATURE_PREFIXES)
        )
    )


def prefill_graph_cache_size(device: torch.device | None = None) -> int:
    policy = current_kernel_policy(device=device, torch_module=torch)
    default = int(getattr(policy, "prefill_graph_cache_size", 2))
    raw = os.environ.get("RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE", str(default))
    try:
        return min(16, max(1, int(raw)))
    except ValueError:
        return min(16, max(1, default))


def prefill_graph_supported(
    owner: Any,
    *,
    batch_size: int,
    prompt_tokens: int,
    quantized: bool,
) -> tuple[bool, str]:
    weight = owner.model.embeddings.weight
    if not torch.cuda.is_available() or weight.device.type != "cuda":
        return False, "prefill CUDA Graph requires a CUDA model"
    if weight.dtype != torch.float16:
        return False, "prefill CUDA Graph is validated only for FP16"
    if quantized:
        return False, "quantized prefill Graph needs a separately validated graph-safe route"
    policy = current_kernel_policy(device=weight.device, torch_module=torch)
    enabled = env_flag(
        "RWKV7_NATIVE_PREFILL_GRAPH", bool(getattr(policy, "prefill_graph", False))
    )
    if not enabled:
        return False, "prefill CUDA Graph is disabled by device policy"
    shapes = {
        tuple(int(item) for item in shape)
        for shape in getattr(policy, "prefill_graph_model_shapes", ())
        if len(shape) == 4
    }
    signature = (
        int(owner.config.hidden_size),
        int(owner.config.num_hidden_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    if shapes and signature not in shapes:
        return False, f"prefill CUDA Graph shape is not allowlisted: {signature}"
    return True, "fixed prefill shape is allowlisted"


class NativePrefillGraphRunner:
    """Capture and replay one exact ``(model, B, T, logits_to_keep)`` shape."""

    def __init__(
        self,
        owner: Any,
        packs: list[Any],
        batch_size: int,
        prompt_tokens: int,
        logits_to_keep: int | None,
    ) -> None:
        supported, reason = prefill_graph_supported(
            owner,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            quantized=False,
        )
        if not supported:
            raise RuntimeError(reason)
        self.owner = owner
        self.proxy = SimpleNamespace(model=owner.model, lm_head=owner.lm_head)
        self.packs = packs
        self.batch_size = int(batch_size)
        self.prompt_tokens = int(prompt_tokens)
        self.logits_to_keep = (
            None if logits_to_keep is None else int(logits_to_keep)
        )
        weight = owner.model.embeddings.weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.input_ids = torch.zeros(
            self.batch_size,
            self.prompt_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None
        self.state: list[torch.Tensor] = []
        self.attention_shift: list[torch.Tensor] = []
        self.ffn_shift: list[torch.Tensor] = []
        self._capture()

    def _run_once(self):
        return native_jit.prefill(
            self.proxy,
            self.input_ids,
            self.packs,
            logits_to_keep=self.logits_to_keep,
        )

    def _capture(self) -> None:
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm), torch.inference_mode():
            for _ in range(3):
                self._run_once()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            logits, state, attention_shift, ffn_shift = self._run_once()
        self.graph = graph
        self.logits = logits
        self.state = list(state)
        self.attention_shift = list(attention_shift)
        self.ffn_shift = list(ffn_shift)

    def replay(self, input_ids: torch.Tensor):
        if tuple(input_ids.shape) != (self.batch_size, self.prompt_tokens):
            raise ValueError("prefill CUDA Graph input shape changed after capture")
        if input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("prefill CUDA Graph input must be CUDA int64")
        if self.graph is None or self.logits is None:
            raise RuntimeError("prefill CUDA Graph is not captured")
        self.input_ids.copy_(input_ids)
        self.graph.replay()
        # Every value is caller-owned after this boundary.  A later replay must
        # not mutate a live HF output or cache.
        return (
            self.logits.clone(),
            [value.clone() for value in self.state],
            [value.clone() for value in self.attention_shift],
            [value.clone() for value in self.ffn_shift],
        )

    def effective_routes(self) -> tuple[str, ...]:
        rows = []
        for field, label in (
            ("_rwkv7_native_prefill_clampw_scan_effective", "clampw"),
            ("_rwkv7_native_prefill_self_chunk_effective", "self_chunk"),
            ("_rwkv7_native_prefill_sequence_ffn_effective", "sequence_ffn"),
            ("_rwkv7_native_prefill_stacked_rkv_effective", "stacked_rkv"),
            ("_rwkv7_native_prefill_wavg_lora_effective", "wavg_lora"),
            ("_rwkv7_native_prefill_fp16_recurrent_effective", "fp16_state"),
        ):
            if bool(getattr(self.proxy, field, False)):
                rows.append(label)
        rows.append("cuda_graph_prefill")
        return tuple(rows)


__all__ = [
    "NativePrefillGraphRunner",
    "prefill_graph_cache_size",
    "prefill_graph_runtime_signature",
    "prefill_graph_supported",
]
