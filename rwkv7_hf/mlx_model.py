# coding=utf-8
"""Correctness-first MLX RWKV-7 recurrent reference backend.

This module is the next Apple Silicon layer after :mod:`rwkv7_hf.mlx_bridge`:
it can run the native RWKV-7 recurrent equations directly on MLX arrays loaded
from a converted HuggingFace checkpoint.  The implementation intentionally
mirrors ``rwkv7_hf.native`` and stays optional/import-safe on non-Apple hosts.

It is **not** the final production Metal/WKV kernel.  The purpose is to pin down
the full MLX weight layout, recurrent state-cache semantics, chunked prefill,
and greedy decode behavior before replacing the inner WKV update with a fused
Metal/MLX kernel.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .mlx_bridge import load_selected_hf_tensors_as_mlx, mlx_array_nbytes, require_mlx, summarize_mlx_arrays
from .mlx_quant import (
    MLXQuantizedLinear,
    metal_quant_available,
    mm4_group_matmul_metal_inputs,
    mm4_triple_matmul_metal_inputs,
    mm8_group_matmul_metal_inputs,
    mm8_triple_matmul_metal_inputs,
    pack_mlx_mm4_group,
    pack_mlx_mm8_group,
)
from .mlx_mix import attn_mix, metal_attn_mix_available
from .mlx_wkv import metal_wkv_available, wkv_update
from .mlx_scan import metal_wkv_scan_available, wkv_scan


EXP_HALF = 0.606531  # = exp(-0.5), RWKV-7 decay base used by the native torch path.


def _mx():
    return require_mlx()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_scan_prefill_mode(name: str, default: str = "off") -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        raw = default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "force", "forced"}:
        return "on"
    if value in {"0", "false", "no", "off", "disable", "disabled"}:
        return "off"
    if value == "auto":
        return "auto"
    return default


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name)
    value = (raw if raw is not None and raw != "" else default).strip().lower()
    return value if value in choices else default


def _is_attn_rkv_projection_weight(key: str) -> bool:
    return key.endswith((".attn.r_proj.weight", ".attn.k_proj.weight", ".attn.v_proj.weight"))


def _as_list(values: Iterable[int] | Any) -> list[int]:
    if isinstance(values, list):
        return [int(v) for v in values]
    if isinstance(values, tuple):
        return [int(v) for v in values]
    try:
        return [int(v) for v in values.tolist()]
    except Exception:
        return [int(v) for v in values]


def _layer_indices(arrays: dict[str, Any]) -> list[int]:
    found: set[int] = set()
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    for key in arrays:
        match = pattern.match(key)
        if match:
            found.add(int(match.group(1)))
    if not found:
        raise ValueError("no model.layers.* tensors found in MLX weight bundle")
    return sorted(found)


@dataclass
class MLXRWKV7State:
    """MLX recurrent state cache for RWKV-7 decode.

    Layout mirrors the native PyTorch cache:

    - ``recurrent_state[layer]``: ``[B, H, N, N]`` fp32 WKV state
    - ``attn_x_prev[layer]``: ``[B, hidden]`` previous attention input
    - ``ffn_x_prev[layer]``: ``[B, hidden]`` previous FFN input
    - ``v_first``: ``[B, hidden]`` first-layer value stream

    ``select_batch`` / ``reorder_cache`` give the MLX path the same dynamic
    batching seam used by HF serving caches.
    """

    recurrent_state: list[Any]
    attn_x_prev: list[Any]
    ffn_x_prev: list[Any]
    v_first: Any
    seen_tokens: int = 0

    @property
    def batch_size(self) -> int:
        return int(self.v_first.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.recurrent_state)

    def clone(self) -> "MLXRWKV7State":
        mx = _mx()
        cloned = MLXRWKV7State(
            [mx.array(x) for x in self.recurrent_state],
            [mx.array(x) for x in self.attn_x_prev],
            [mx.array(x) for x in self.ffn_x_prev],
            mx.array(self.v_first),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(cloned.v_first, *cloned.recurrent_state, *cloned.attn_x_prev, *cloned.ffn_x_prev)
        return cloned

    def select_batch(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        mx = _mx()
        idx = mx.array(_as_list(indices), dtype=mx.int32)
        selected = MLXRWKV7State(
            [mx.take(x, idx, axis=0) for x in self.recurrent_state],
            [mx.take(x, idx, axis=0) for x in self.attn_x_prev],
            [mx.take(x, idx, axis=0) for x in self.ffn_x_prev],
            mx.take(self.v_first, idx, axis=0),
            seen_tokens=int(self.seen_tokens),
        )
        mx.eval(selected.v_first, *selected.recurrent_state, *selected.attn_x_prev, *selected.ffn_x_prev)
        return selected

    def reorder_cache(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def compact(self, indices: Iterable[int] | Any) -> "MLXRWKV7State":
        return self.select_batch(indices)

    def detach(self) -> "MLXRWKV7State":
        # MLX arrays are eager/lazy value arrays, not torch autograd tensors; a
        # clone gives callers an explicit cache boundary.
        return self.clone()


@dataclass
class MLXGenerateOutput:
    """Tokenizer-integrated MLX generation result."""

    prompt: str
    prompt_ids: list[int]
    generated_ids: list[int]
    text: str
    prefill_s: float
    decode_s: float
    prompt_tokens: int
    generated_tokens: int

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else None

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def telemetry(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "generated_tokens": int(self.generated_tokens),
            "prefill_s": round(float(self.prefill_s), 6),
            "decode_s": round(float(self.decode_s), 6),
            "prefill_tok_s": round(float(self.prefill_tok_s), 6) if self.prefill_tok_s is not None else None,
            "decode_tok_s": round(float(self.decode_tok_s), 6) if self.decode_tok_s is not None else None,
            "generated_preview": [int(x) for x in self.generated_ids[:16]],
        }


@dataclass
class MLXSessionStepOutput:
    """One incremental decode step from an :class:`MLXGenerationSession`."""

    step_index: int
    generated_ids: list[int]
    text: str
    decode_s: float
    total_generated_tokens: int
    seen_tokens: int

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def telemetry(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "generated_tokens": int(self.generated_tokens),
            "total_generated_tokens": int(self.total_generated_tokens),
            "seen_tokens": int(self.seen_tokens),
            "decode_s": round(float(self.decode_s), 6),
            "decode_tok_s": round(float(self.decode_tok_s), 6) if self.decode_tok_s is not None else None,
            "generated_preview": [int(x) for x in self.generated_ids[:16]],
        }


class MLXGenerationSession:
    """Stateful tokenizer-backed MLX generation helper.

    The plain ``generate_text`` helper is useful for one-shot demos.  Serving
    style callers need a stricter seam: prefill a prompt once, hold the RWKV
    recurrent state cache, then decode in multiple chunks without recomputing
    the prompt.  This class exposes that shape for Apple/MLX smoke tests and
    future Metal-backed serving integration.

    The session is intentionally single-prompt/tokenizer-backed.  Dynamic batch
    select/reorder remains covered at the lower ``MLXRWKV7State`` layer where
    batched cache tensors are explicit.
    """

    def __init__(
        self,
        *,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompt: str,
        prompt_ids: list[int],
        logits: Any,
        state: MLXRWKV7State,
        prefill_s: float,
        skip_special_tokens: bool = False,
    ):
        if state.batch_size != 1:
            raise ValueError("MLXGenerationSession currently expects one prompt / batch row")
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_ids = [int(x) for x in prompt_ids]
        self.logits = logits
        self.state = state
        self.prefill_s = float(prefill_s)
        self.decode_s = 0.0
        self.generated_ids: list[int] = []
        self.step_count = 0
        self.skip_special_tokens = bool(skip_special_tokens)

    @classmethod
    def from_prompt(
        cls,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompt: str,
        *,
        skip_special_tokens: bool = False,
    ) -> "MLXGenerationSession":
        """Encode and prefill a prompt, returning a reusable decode session."""

        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = [int(tok) for tok in encoded.input_ids]
        if not prompt_ids:
            raise ValueError("prompt produced no token ids")
        t0 = time.perf_counter()
        logits, state = model.prefill([prompt_ids])
        prefill_s = time.perf_counter() - t0
        return cls(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            logits=logits,
            state=state,
            prefill_s=prefill_s,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def text(self) -> str:
        return self.tokenizer.decode(self.generated_ids, skip_special_tokens=self.skip_special_tokens)

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else None

    @property
    def decode_tok_s(self) -> float | None:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else None

    def decode(self, max_new_tokens: int) -> MLXSessionStepOutput:
        """Decode ``max_new_tokens`` more tokens from the cached state."""

        mx = _mx()
        n = int(max_new_tokens)
        if n < 0:
            raise ValueError("max_new_tokens must be non-negative")
        t0 = time.perf_counter()
        generated = []
        next_token = mx.argmax(self.logits[:, -1, :], axis=-1).astype(mx.int32)
        for _ in range(n):
            generated.append(next_token)
            self.logits, self.state = self.model.decode_step(next_token, self.state)
            next_token = mx.argmax(self.logits[:, -1, :], axis=-1).astype(mx.int32)
        if generated:
            out = mx.stack(generated, axis=1)
            mx.eval(out, self.logits)
            step_ids = _as_list(out.reshape(-1))
        else:
            mx.eval(self.logits)
            step_ids = []
        elapsed = time.perf_counter() - t0
        self.generated_ids.extend(step_ids)
        self.decode_s += elapsed
        self.step_count += 1
        return MLXSessionStepOutput(
            step_index=self.step_count,
            generated_ids=step_ids,
            text=self.tokenizer.decode(step_ids, skip_special_tokens=self.skip_special_tokens),
            decode_s=elapsed,
            total_generated_tokens=self.generated_tokens,
            seen_tokens=int(self.state.seen_tokens),
        )

    def output(self) -> MLXGenerateOutput:
        """Return a cumulative one-shot-style generation output."""

        return MLXGenerateOutput(
            prompt=self.prompt,
            prompt_ids=list(self.prompt_ids),
            generated_ids=list(self.generated_ids),
            text=self.text,
            prefill_s=self.prefill_s,
            decode_s=self.decode_s,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
        )

    def telemetry(self) -> dict[str, Any]:
        out = self.output().telemetry()
        out.update(
            {
                "session_steps": int(self.step_count),
                "seen_tokens": int(self.state.seen_tokens),
            }
        )
        return out


class MLXGenerationSessionBatch:
    """Small interleaved session manager for MLX serving smoke tests.

    This is not the final fused quant+WKV Metal kernel.  It is a
    production-shaped API scaffold: multiple independent prompts are prefetched
    once, then advanced round-by-round while preserving each prompt's recurrent
    state cache.  The default path stays sequential for compatibility, while
    ``backend="batched"`` / ``"auto"`` stacks equal-length decode rounds into
    one MLX batch so Apple validation can exercise a dynamic-batching shaped
    path before the inner decode loop is replaced by deeper fused MLX/Metal
    kernels.
    """

    def __init__(self, sessions: list[MLXGenerationSession]):
        if not sessions:
            raise ValueError("MLXGenerationSessionBatch requires at least one session")
        model = sessions[0].model
        tokenizer = sessions[0].tokenizer
        for idx, session in enumerate(sessions):
            if session.model is not model:
                raise ValueError(f"session {idx} uses a different MLX model instance")
            if session.tokenizer is not tokenizer:
                raise ValueError(f"session {idx} uses a different tokenizer instance")
        self.model = model
        self.tokenizer = tokenizer
        self.sessions = list(sessions)
        self.round_count = 0
        self.round_backends: list[str] = []
        self.round_backend_reasons: list[str] = []
        self.round_decode_s: list[float] = []
        self.round_generated_tokens: list[int] = []
        self.round_stable_repair_counts: list[int] = []

    @classmethod
    def from_prompts(
        cls,
        model: "MLXRWKV7Model",
        tokenizer: Any,
        prompts: list[str],
        *,
        skip_special_tokens: bool = False,
    ) -> "MLXGenerationSessionBatch":
        if isinstance(prompts, str):
            raise TypeError("prompts must be a list of strings, not a single string")
        if not prompts:
            raise ValueError("prompts must contain at least one prompt")
        sessions = [
            MLXGenerationSession.from_prompt(
                model,
                tokenizer,
                prompt,
                skip_special_tokens=skip_special_tokens,
            )
            for prompt in prompts
        ]
        return cls(sessions)

    @property
    def batch_size(self) -> int:
        return len(self.sessions)

    def _normalize_steps(self, tokens_per_session: int | list[int]) -> list[int]:
        if isinstance(tokens_per_session, int):
            steps = [int(tokens_per_session)] * self.batch_size
        else:
            steps = [int(x) for x in tokens_per_session]
            if len(steps) != self.batch_size:
                raise ValueError(f"expected {self.batch_size} token counts, got {len(steps)}")
        if any(step < 0 for step in steps):
            raise ValueError("all token counts must be non-negative")
        return steps

    def _decode_round_sequential(self, steps: list[int], *, reason: str = "requested") -> list[MLXSessionStepOutput]:
        t0 = time.perf_counter()
        outputs = [session.decode(step) for session, step in zip(self.sessions, steps)]
        elapsed = time.perf_counter() - t0
        self.round_decode_s.append(float(elapsed))
        self.round_generated_tokens.append(int(sum(steps)))
        self.round_backends.append("sequential")
        self.round_backend_reasons.append(str(reason))
        self.round_stable_repair_counts.append(0)
        return outputs

    def _uses_metal_quant_projection_bits(self, bits: int) -> bool:
        active_bits = getattr(self.model, "quantized_linear_bits", None)
        quant_backend = getattr(self.model, "quantized_linear_backend", None)
        if active_bits != int(bits):
            return False
        if quant_backend == "metal":
            return True
        if quant_backend != "auto":
            return False
        return any(int(getattr(q, "auto_metal_max_rows", 0)) > 0 for q in self.model.quantized_linears.values())

    def _uses_w8_metal_projection(self) -> bool:
        return self._uses_metal_quant_projection_bits(8)

    def _uses_w4_metal_projection(self) -> bool:
        return self._uses_metal_quant_projection_bits(4)

    def _stable_argmax_tolerance_value(self) -> float:
        return max(0.0, _env_float("RWKV7_MLX_SESSION_STABLE_ARGMAX_TOLERANCE", 0.015625))

    def _stable_argmax_mode_value(self) -> str:
        return _env_choice("RWKV7_MLX_SESSION_STABLE_ARGMAX_MODE", "lower", {"lower", "repair"})

    def _auto_stable_argmax_tolerance(self) -> float:
        if self._uses_w8_metal_projection() and _env_flag("RWKV7_MLX_SESSION_AUTO_W8_STABLE", False):
            return self._stable_argmax_tolerance_value()
        if self._uses_w4_metal_projection() and _env_flag("RWKV7_MLX_SESSION_AUTO_W4_STABLE", False):
            return self._stable_argmax_tolerance_value()
        return 0.0

    def _auto_batch_disabled_reason(self) -> str | None:
        """Return why ``backend='auto'`` should avoid batched decode.

        Metal quant projection is correct for one-shot and sequential session
        paths, but long multi-session batched decode can diverge from one-shot
        greedy tokens in low-margin cases.  Keep automatic W8/W4 Metal batching
        guarded by default.  Developers can opt into stable auto batching with
        ``RWKV7_MLX_SESSION_AUTO_W8_STABLE=1`` or
        ``RWKV7_MLX_SESSION_AUTO_W4_STABLE=1`` after running strict gates.
        """

        if self._auto_stable_argmax_tolerance() > 0:
            return None
        if self._uses_w8_metal_projection():
            return "auto_mm8_metal_batch_exactness_guard"
        if self._uses_w4_metal_projection():
            return "auto_mm4_metal_batch_exactness_guard"
        return None

    def _greedy_argmax(self, logits):
        mx = _mx()
        scores = logits[:, -1, :].astype(mx.float32)
        return mx.argmax(scores, axis=-1).astype(mx.int32)

    def _stable_argmax_lower(self, logits, *, tolerance: float):
        mx = _mx()
        scores = logits[:, -1, :].astype(mx.float32)
        greedy = mx.argmax(scores, axis=-1).astype(mx.int32)
        tol = float(tolerance)
        if tol <= 0:
            return greedy
        top2_idx = mx.argpartition(-scores, kth=2, axis=-1)[:, :2].astype(mx.int32)
        top2_vals = mx.take_along_axis(scores, top2_idx, axis=-1)
        margins = mx.max(top2_vals, axis=-1) - mx.min(top2_vals, axis=-1)
        low_margin = margins <= tol
        low_token = mx.min(top2_idx, axis=-1).astype(mx.int32)
        return mx.where(low_margin, low_token, greedy).astype(mx.int32)

    def _low_margin_indices(self, logits, *, tolerance: float) -> list[int]:
        """Return batch rows whose top-2 logits are close enough to repair.

        Low-margin batched Metal quant logits can differ from the one-row path
        by enough fp16 ulps to flip greedy tokens.  The optional
        ``RWKV7_MLX_SESSION_STABLE_ARGMAX_MODE=repair`` path uses this detector
        to selectively replay those rows through the exact one-row decode path
        and then argmaxes the repaired logits normally.
        """

        tol = float(tolerance)
        if tol <= 0:
            return []
        mx = _mx()
        scores = logits[:, -1, :].astype(mx.float32)
        top2_idx = mx.argpartition(-scores, kth=2, axis=-1)[:, :2].astype(mx.int32)
        top2_vals = mx.take_along_axis(scores, top2_idx, axis=-1)
        margins = mx.max(top2_vals, axis=-1) - mx.min(top2_vals, axis=-1)
        mx.eval(margins)
        return [idx for idx, value in enumerate(margins.tolist()) if float(value) <= tol]

    def _concat_state_rows(self, rows: list[MLXRWKV7State], *, seen_tokens: int) -> MLXRWKV7State:
        mx = _mx()
        state = MLXRWKV7State(
            [
                mx.concatenate([row.recurrent_state[layer] for row in rows], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            [
                mx.concatenate([row.attn_x_prev[layer] for row in rows], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            [
                mx.concatenate([row.ffn_x_prev[layer] for row in rows], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            mx.concatenate([row.v_first for row in rows], axis=0),
            seen_tokens=int(seen_tokens),
        )
        mx.eval(state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev)
        return state

    def _repair_low_margin_rows(
        self,
        logits,
        *,
        state_before: MLXRWKV7State,
        state_after: MLXRWKV7State,
        token_ids,
        tolerance: float,
    ) -> tuple[Any, MLXRWKV7State, int]:
        """Replay low-margin batched rows through the one-row decode path.

        The repair is only used by explicit stable backends.  It preserves the
        batched fast path for ordinary rows while replacing ambiguous row logits
        and recurrent state with the same values a sequential session would have
        produced for the just-consumed token.
        """

        repair_indices = self._low_margin_indices(logits, tolerance=tolerance)
        if not repair_indices:
            return logits, state_after, 0

        mx = _mx()
        token_list = _as_list(token_ids)
        logit_rows = [mx.take(logits, mx.array([idx], dtype=mx.int32), axis=0) for idx in range(self.batch_size)]
        state_rows = [state_after.select_batch([idx]) for idx in range(self.batch_size)]
        for idx in repair_indices:
            row_state_before = state_before.select_batch([idx])
            exact_logits, exact_state = self.model.decode_step([int(token_list[idx])], row_state_before)
            exact_state.seen_tokens = int(state_after.seen_tokens)
            logit_rows[idx] = exact_logits
            state_rows[idx] = exact_state
        repaired_logits = mx.concatenate(logit_rows, axis=0)
        repaired_state = self._concat_state_rows(state_rows, seen_tokens=int(state_after.seen_tokens))
        mx.eval(repaired_logits)
        return repaired_logits, repaired_state, len(repair_indices)

    def _stack_state(self) -> tuple[MLXRWKV7State, list[int]]:
        mx = _mx()
        seen_tokens = [int(session.state.seen_tokens) for session in self.sessions]
        stacked = MLXRWKV7State(
            [
                mx.concatenate([session.state.recurrent_state[layer] for session in self.sessions], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            [
                mx.concatenate([session.state.attn_x_prev[layer] for session in self.sessions], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            [
                mx.concatenate([session.state.ffn_x_prev[layer] for session in self.sessions], axis=0)
                for layer in range(self.model.num_hidden_layers)
            ],
            mx.concatenate([session.state.v_first for session in self.sessions], axis=0),
            seen_tokens=min(seen_tokens) if seen_tokens else 0,
        )
        mx.eval(stacked.v_first, *stacked.recurrent_state, *stacked.attn_x_prev, *stacked.ffn_x_prev)
        return stacked, seen_tokens

    def _split_state(self, state: MLXRWKV7State, seen_tokens: list[int]) -> list[MLXRWKV7State]:
        split: list[MLXRWKV7State] = []
        for idx, seen in enumerate(seen_tokens):
            row = state.select_batch([idx])
            row.seen_tokens = int(seen)
            split.append(row)
        return split

    def _decode_round_batched(self, tokens_per_session: int, *, stable_argmax_tolerance: float = 0.0) -> list[MLXSessionStepOutput]:
        mx = _mx()
        n = int(tokens_per_session)
        if n <= 0:
            return self._decode_round_sequential([n] * self.batch_size, reason="zero_token_round")

        prior_seen = [int(session.state.seen_tokens) for session in self.sessions]
        stacked_state, _ = self._stack_state()
        logits = mx.concatenate([session.logits for session in self.sessions], axis=0)
        stable_mode = self._stable_argmax_mode_value() if stable_argmax_tolerance > 0 else "off"
        next_token = (
            self._stable_argmax_lower(logits, tolerance=stable_argmax_tolerance)
            if stable_mode == "lower"
            else self._greedy_argmax(logits)
        )
        generated = []
        stable_repair_count = 0

        t0 = time.perf_counter()
        for _ in range(n):
            generated.append(next_token)
            state_before = stacked_state.clone() if stable_mode == "repair" else stacked_state
            logits, stacked_state = self.model.decode_step(next_token, stacked_state)
            if stable_mode == "repair":
                logits, stacked_state, repaired = self._repair_low_margin_rows(
                    logits,
                    state_before=state_before,
                    state_after=stacked_state,
                    token_ids=next_token,
                    tolerance=stable_argmax_tolerance,
                )
                stable_repair_count += int(repaired)
            next_token = (
                self._stable_argmax_lower(logits, tolerance=stable_argmax_tolerance)
                if stable_mode == "lower"
                else self._greedy_argmax(logits)
            )
        out = mx.stack(generated, axis=1)
        mx.eval(out, logits, stacked_state.v_first, *stacked_state.recurrent_state, *stacked_state.attn_x_prev, *stacked_state.ffn_x_prev)
        elapsed = time.perf_counter() - t0

        split_states = self._split_state(stacked_state, [seen + n for seen in prior_seen])
        outputs: list[MLXSessionStepOutput] = []
        for idx, session in enumerate(self.sessions):
            row_ids = _as_list(out[idx].reshape(-1))
            session.logits = mx.take(logits, mx.array([idx], dtype=mx.int32), axis=0)
            session.state = split_states[idx]
            session.generated_ids.extend(row_ids)
            session.decode_s += elapsed
            session.step_count += 1
            outputs.append(
                MLXSessionStepOutput(
                    step_index=session.step_count,
                    generated_ids=row_ids,
                    text=self.tokenizer.decode(row_ids, skip_special_tokens=session.skip_special_tokens),
                    decode_s=elapsed,
                    total_generated_tokens=session.generated_tokens,
                    seen_tokens=int(session.state.seen_tokens),
                )
            )
        self.round_decode_s.append(float(elapsed))
        self.round_generated_tokens.append(int(n * self.batch_size))
        self.round_backends.append("batched_stable" if stable_argmax_tolerance > 0 else "batched")
        if stable_argmax_tolerance > 0 and stable_mode == "repair":
            reason = f"equal_positive_round_stable_argmax_tol_{stable_argmax_tolerance:g}_mode_repair_repairs_{stable_repair_count}"
        elif stable_argmax_tolerance > 0:
            reason = f"equal_positive_round_stable_argmax_tol_{stable_argmax_tolerance:g}"
        else:
            reason = "equal_positive_round"
        self.round_backend_reasons.append(reason)
        self.round_stable_repair_counts.append(int(stable_repair_count))
        return outputs

    def decode_round(
        self,
        tokens_per_session: int | list[int],
        *,
        backend: str = "sequential",
    ) -> list[MLXSessionStepOutput]:
        """Advance all sessions once and return per-session step outputs.

        ``backend="sequential"`` preserves the historical per-session loop.
        ``backend="batched"`` requires equal positive token counts and decodes
        all sessions as one MLX batch.  ``backend="batched_stable"`` adds a
        low-margin stable-argmax policy for W8/W4 Metal exactness bring-up.
        ``backend="auto"`` uses the batched path when all sessions request the
        same positive number of tokens and falls back to the sequential path for
        heterogeneous or zero-token rounds, plus guarded W8/W4 Metal quant
        paths until their long batched-session exactness gates pass.
        """

        steps = self._normalize_steps(tokens_per_session)
        selected_backend = (backend or "sequential").lower().strip()
        if selected_backend not in {"sequential", "batched", "batched_stable", "auto"}:
            raise ValueError(f"unsupported MLX session backend {backend!r}; expected sequential, batched, batched_stable, or auto")

        if selected_backend == "sequential":
            outputs = self._decode_round_sequential(steps, reason="requested")
        elif len(set(steps)) == 1 and steps[0] > 0 and (
            selected_backend in {"batched", "batched_stable"} or self._auto_batch_disabled_reason() is None
        ):
            tol = (
                self._stable_argmax_tolerance_value()
                if selected_backend == "batched_stable"
                else self._auto_stable_argmax_tolerance()
                if selected_backend == "auto"
                else 0.0
            )
            outputs = self._decode_round_batched(steps[0], stable_argmax_tolerance=tol)
        elif selected_backend == "auto":
            reason = self._auto_batch_disabled_reason()
            if reason is None:
                reason = "heterogeneous_or_zero_round"
            outputs = self._decode_round_sequential(steps, reason=reason)
        else:
            raise ValueError("backend='batched'/'batched_stable' requires equal positive token counts for every session")
        self.round_count += 1
        return outputs

    def outputs(self) -> list[MLXGenerateOutput]:
        return [session.output() for session in self.sessions]

    def telemetry(self) -> dict[str, Any]:
        prompt_tokens = [session.prompt_tokens for session in self.sessions]
        generated_tokens = [session.generated_tokens for session in self.sessions]
        seen_tokens = [int(session.state.seen_tokens) for session in self.sessions]
        decode_s = [round(float(session.decode_s), 6) for session in self.sessions]
        round_decode_s = [round(float(value), 6) for value in self.round_decode_s]
        round_stable_repair_counts = [int(value) for value in self.round_stable_repair_counts]
        return {
            "batch_size": int(self.batch_size),
            "session_rounds": int(self.round_count),
            "round_backends": list(self.round_backends),
            "round_backend_reasons": list(self.round_backend_reasons),
            "round_stable_repair_counts": round_stable_repair_counts,
            "last_round_stable_repair_count": (
                round_stable_repair_counts[-1] if round_stable_repair_counts else None
            ),
            "last_round_backend": self.round_backends[-1] if self.round_backends else None,
            "last_round_backend_reason": self.round_backend_reasons[-1] if self.round_backend_reasons else None,
            "round_decode_s": round_decode_s,
            "round_generated_tokens": [int(value) for value in self.round_generated_tokens],
            "round_decode_tok_s": [
                round(float(tokens / elapsed), 6) if elapsed > 0 else None
                for tokens, elapsed in zip(self.round_generated_tokens, self.round_decode_s)
            ],
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "seen_tokens": seen_tokens,
            "decode_s": decode_s,
            "decode_tok_s": [
                round(float(session.decode_tok_s), 6) if session.decode_tok_s is not None else None
                for session in self.sessions
            ],
            "generated_previews": [[int(x) for x in session.generated_ids[:16]] for session in self.sessions],
        }


class MLXRWKV7Model:
    """Minimal MLX-native RWKV-7 recurrent model loaded from HF safetensors."""

    def __init__(self, config: dict[str, Any], arrays: dict[str, Any], *, wkv_backend: str = "reference"):
        self.config = dict(config)
        self.arrays = dict(arrays)
        self.wkv_backend = (wkv_backend or "reference").lower().strip()
        if self.wkv_backend not in {"reference", "metal", "auto"}:
            raise ValueError(f"unsupported MLX WKV backend {wkv_backend!r}; expected reference, metal, or auto")
        self.wkv_backend_last: str | None = None
        self.wkv_backend_counts: dict[str, int] = {"reference": 0, "metal": 0}
        self.quantized_linears: dict[str, MLXQuantizedLinear] = {}
        self.quantized_dense_equivalent_bytes = 0
        self.quantized_linear_bytes = 0
        self.quantized_linear_bits: int | None = None
        self.quantized_linear_backend: str | None = None
        self.quantized_linear_min_params: int | None = None
        self.quantized_linear_rkv_min_params: int | None = None
        self.step_eval_interval = max(1, _env_int("RWKV7_MLX_STEP_EVAL_INTERVAL", 1))
        self.fused_ffn_key_relu2 = _env_flag("RWKV7_MLX_FUSED_FFN_KEY_RELU2", False)
        self.fused_ffn_key_relu2_counts: dict[str, int] = {"metal": 0, "fallback": 0}
        self.fused_attn_mix = _env_flag("RWKV7_MLX_FUSED_ATTN_MIX", False)
        self.fused_attn_mix_counts: dict[str, int] = {"metal": 0, "fallback": 0}
        self.wkv_scan_prefill_mode = _env_scan_prefill_mode("RWKV7_MLX_WKV_SCAN_PREFILL", "off")
        self.wkv_scan_prefill_min_tokens = max(2, _env_int("RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS", 32))
        self.wkv_scan_prefill_counts: dict[str, int] = {"reference": 0, "metal": 0, "fallback": 0}
        self.wkv_scan_prefill_reason_counts: dict[str, int] = {}
        self.state_only_prefill_calls = 0
        self.state_only_prefill_tokens = 0
        self.group_rkv_quant_projection = _env_flag("RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION", False)
        self.group_rkv_quant_projection_mode = _env_choice(
            "RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION_MODE",
            "direct",
            {"direct", "packed"},
        )
        self.group_rkv_quant_projection_counts: dict[str, int] = {"metal": 0, "fallback": 0}
        self._rkv_group_quant_cache: dict[tuple[int, int], Any] = {}
        self.layer_ids = _layer_indices(self.arrays)
        self.num_hidden_layers = int(self.config.get("num_hidden_layers", len(self.layer_ids)))
        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config.get("num_heads", self.config.get("n_head", 0)))
        self.head_dim = int(self.config.get("head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(self.config["vocab_size"])
        self.intermediate_size = int(self.config.get("intermediate_size", self.hidden_size * 4))
        self.norm_eps = float(self.config.get("norm_eps", 1e-5))
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"invalid RWKV-7 shape: num_heads({self.num_heads}) * head_dim({self.head_dim}) "
                f"!= hidden_size({self.hidden_size})"
            )
        if len(self.layer_ids) != self.num_hidden_layers:
            raise ValueError(f"config has {self.num_hidden_layers} layers but tensors contain {len(self.layer_ids)}")

    @classmethod
    def from_hf(
        cls,
        model_dir: str | Path,
        *,
        dtype: str | None = "fp16",
        quantization: str | None = None,
        quant_min_params: int = 8_000_000,
        quant_rkv_min_params: int | None = None,
        quant_backend: str = "affine",
        wkv_backend: str = "reference",
    ) -> "MLXRWKV7Model":
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        arrays = load_selected_hf_tensors_as_mlx(root, tensor_regex=r".*", dtype=dtype)
        model = cls(config, arrays, wkv_backend=wkv_backend)
        if quantization and quantization.lower() not in {"none", "off", "false", "0"}:
            model.quantize_linears(
                quantization,
                min_params=quant_min_params,
                rkv_min_params=quant_rkv_min_params,
                backend=quant_backend,
            )
        return model

    @classmethod
    def from_arrays(cls, config: dict[str, Any], arrays: dict[str, Any], *, wkv_backend: str = "reference") -> "MLXRWKV7Model":
        return cls(config, arrays, wkv_backend=wkv_backend)

    def _is_quantizable_linear_weight(
        self,
        key: str,
        value: Any,
        min_params: int,
        *,
        rkv_min_params: int | None = None,
    ) -> bool:
        if not key.endswith(".weight"):
            return False
        if key == "model.embeddings.weight":
            return False
        if getattr(value, "ndim", 0) != 2:
            return False
        threshold = (
            int(rkv_min_params)
            if rkv_min_params is not None and int(rkv_min_params) >= 0 and _is_attn_rkv_projection_weight(key)
            else int(min_params)
        )
        if int(value.size) < threshold:
            return False
        return True

    def quantize_linears(
        self,
        quantization: str,
        *,
        min_params: int = 8_000_000,
        rkv_min_params: int | None = None,
        backend: str = "affine",
    ) -> int:
        """Replace eligible dense MLX Linear weights with packed W8/W4 weights.

        This is the Apple packed-quant projection seam.  ``backend=affine`` runs
        dequant-matmul via MLX affine decomposition without materializing a dense
        dequantized fp16/fp32 weight; ``backend=reference`` keeps a correctness
        fallback; ``backend=metal`` enables the fused dequant-projection kernel;
        ``backend=auto`` selects the safe small-batch Metal path where current
        exactness and speed gates allow it.

        ``rkv_min_params`` is an Apple performance knob for the fused/grouped
        R/K/V projection path.  It lets callers quantize attention
        ``r_proj``/``k_proj``/``v_proj`` weights even when the general
        ``min_params`` threshold intentionally keeps smaller dense matrices
        unquantized.  Leave it as ``None`` to preserve the historical single
        threshold policy.
        """

        backend = (backend or "affine").lower().strip()
        if backend not in {"affine", "reference", "metal", "auto"}:
            raise ValueError(f"unsupported MLX quant backend {backend!r}; expected affine, reference, metal, or auto")
        q = quantization.lower().strip()
        if q in {"mm8", "w8", "8", "int8"}:
            bits = 8
        elif q in {"mm4", "w4", "4", "int4"}:
            bits = 4
        else:
            raise ValueError(f"unsupported MLX quantization {quantization!r}; expected mm8/mm4")
        effective_rkv_min_params = None if rkv_min_params is None or int(rkv_min_params) < 0 else int(rkv_min_params)
        selected = [
            key for key, value in list(self.arrays.items())
            if self._is_quantizable_linear_weight(key, value, min_params, rkv_min_params=effective_rkv_min_params)
        ]
        for key in selected:
            dense = self.arrays.pop(key)
            self.quantized_dense_equivalent_bytes += mlx_array_nbytes(dense)
            qlinear = MLXQuantizedLinear.from_linear_weight(dense, bits=bits, backend=backend)
            self.quantized_linears[key] = qlinear
            self.quantized_linear_bytes += qlinear.storage_bytes
        self._rkv_group_quant_cache.clear()
        self.quantized_linear_bits = bits if selected else None
        self.quantized_linear_backend = backend if selected else None
        self.quantized_linear_min_params = int(min_params) if selected else None
        self.quantized_linear_rkv_min_params = effective_rkv_min_params if selected else None
        return len(selected)

    def reset_telemetry_counters(self) -> None:
        """Reset per-run backend counters without changing weights or caches."""

        self.wkv_backend_last = None
        self.wkv_backend_counts = {"reference": 0, "metal": 0}
        self.fused_ffn_key_relu2_counts = {"metal": 0, "fallback": 0}
        self.fused_attn_mix_counts = {"metal": 0, "fallback": 0}
        self.wkv_scan_prefill_counts = {"reference": 0, "metal": 0, "fallback": 0}
        self.wkv_scan_prefill_reason_counts = {}
        self.state_only_prefill_calls = 0
        self.state_only_prefill_tokens = 0
        self.group_rkv_quant_projection_counts = {"metal": 0, "fallback": 0}
        for qlinear in self.quantized_linears.values():
            qlinear.last_backend = None
            qlinear.backend_counts = {"reference": 0, "affine": 0, "metal": 0}

    def telemetry(self) -> dict[str, Any]:
        out = {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "wkv_backend": self.wkv_backend,
            "wkv_backend_last": self.wkv_backend_last,
            "wkv_backend_counts": dict(self.wkv_backend_counts),
            "wkv_metal_available": metal_wkv_available(),
            "quant_metal_available": metal_quant_available(),
            "step_eval_interval": int(self.step_eval_interval),
            "fused_ffn_key_relu2": bool(self.fused_ffn_key_relu2),
            "fused_ffn_key_relu2_counts": dict(self.fused_ffn_key_relu2_counts),
            "fused_attn_mix": bool(self.fused_attn_mix),
            "fused_attn_mix_counts": dict(self.fused_attn_mix_counts),
            "fused_attn_mix_metal_available": metal_attn_mix_available(),
            "wkv_scan_prefill": self.wkv_scan_prefill_mode != "off",
            "wkv_scan_prefill_mode": self.wkv_scan_prefill_mode,
            "wkv_scan_prefill_min_tokens": int(self.wkv_scan_prefill_min_tokens),
            "wkv_scan_prefill_counts": dict(self.wkv_scan_prefill_counts),
            "wkv_scan_prefill_reason_counts": dict(self.wkv_scan_prefill_reason_counts),
            "wkv_scan_metal_available": metal_wkv_scan_available(),
            "state_only_prefill_calls": int(self.state_only_prefill_calls),
            "state_only_prefill_tokens": int(self.state_only_prefill_tokens),
            **summarize_mlx_arrays(self.arrays),
        }
        if self.quantized_linears:
            out.update(
                {
                    "quantized_linear_count": len(self.quantized_linears),
                    "quantized_linear_bits": self.quantized_linear_bits,
                    "quantized_linear_backend": self.quantized_linear_backend,
                    "quantized_linear_min_params": self.quantized_linear_min_params,
                    "quantized_linear_rkv_min_params": self.quantized_linear_rkv_min_params,
                    "quantized_linear_bytes": int(self.quantized_linear_bytes),
                    "quantized_dense_equivalent_bytes": int(self.quantized_dense_equivalent_bytes),
                    "quantized_footprint_ratio": round(
                        self.quantized_linear_bytes / max(self.quantized_dense_equivalent_bytes, 1), 6
                    ),
                    "quantized_linear_keys_preview": sorted(self.quantized_linears)[:8],
                    "quantized_linear_last_backend_counts": {
                        "reference": sum(int(q.backend_counts.get("reference", 0)) for q in self.quantized_linears.values()),
                        "affine": sum(int(q.backend_counts.get("affine", 0)) for q in self.quantized_linears.values()),
                        "metal": sum(int(q.backend_counts.get("metal", 0)) for q in self.quantized_linears.values()),
                    },
                    "group_rkv_quant_projection": bool(self.group_rkv_quant_projection),
                    "group_rkv_quant_projection_mode": self.group_rkv_quant_projection_mode,
                    "group_rkv_quant_projection_counts": dict(self.group_rkv_quant_projection_counts),
                }
            )
        return out

    def _get(self, key: str):
        try:
            return self.arrays[key]
        except KeyError as exc:
            raise KeyError(f"missing MLX RWKV-7 tensor {key!r}") from exc

    def _linear(self, x, weight_key: str, bias_key: str | None = None):
        qlinear = self.quantized_linears.get(weight_key)
        if qlinear is not None:
            y = qlinear(x)
        else:
            y = x @ self._get(weight_key).T
        if bias_key is not None and bias_key in self.arrays:
            y = y + self._get(bias_key)
        return y

    def _rkv_group_weight(self, layer: int, qlines: list[MLXQuantizedLinear]):
        bits = int(qlines[0].bits)
        cache_key = (int(layer), bits)
        cached = self._rkv_group_quant_cache.get(cache_key)
        if cached is not None:
            return cached
        weights = [q.weight for q in qlines]
        group = pack_mlx_mm8_group(weights) if bits == 8 else pack_mlx_mm4_group(weights)
        self._rkv_group_quant_cache[cache_key] = group
        return group

    def _grouped_rkv_projection(self, layer: int, xr, xk, xv, prefix: str):
        """Opt-in grouped R/K/V quant projection seam.

        This path is disabled by default and only activates when all three
        R/K/V projection weights are quantized, route to Metal, and share shape
        and bit-width. It keeps the default correctness-first path unchanged.
        """

        if not self.group_rkv_quant_projection:
            return None
        keys = [
            f"{prefix}.r_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ]
        qlines = [self.quantized_linears.get(key) for key in keys]
        if any(q is None for q in qlines):
            self.group_rkv_quant_projection_counts["fallback"] = int(
                self.group_rkv_quant_projection_counts.get("fallback", 0)
            ) + 1
            return None
        qlines = [q for q in qlines if q is not None]
        if len({int(q.bits) for q in qlines}) != 1:
            self.group_rkv_quant_projection_counts["fallback"] = int(
                self.group_rkv_quant_projection_counts.get("fallback", 0)
            ) + 1
            return None
        if any(q._selected_backend(x) != "metal" for q, x in zip(qlines, (xr, xk, xv), strict=True)):
            self.group_rkv_quant_projection_counts["fallback"] = int(
                self.group_rkv_quant_projection_counts.get("fallback", 0)
            ) + 1
            return None
        if self.group_rkv_quant_projection_mode == "packed":
            mx = _mx()
            group = self._rkv_group_weight(layer, qlines)
            x_group = mx.stack([xr, xk, xv], axis=0)
            y_group = (
                mm8_group_matmul_metal_inputs(x_group, group)
                if int(qlines[0].bits) == 8
                else mm4_group_matmul_metal_inputs(x_group, group)
            )
        else:
            weights = [q.weight for q in qlines]
            y_group = (
                mm8_triple_matmul_metal_inputs(xr, xk, xv, weights)
                if int(qlines[0].bits) == 8
                else mm4_triple_matmul_metal_inputs(xr, xk, xv, weights)
            )
        for q in qlines:
            q.last_backend = "metal"
            q.backend_counts["metal"] = int(q.backend_counts.get("metal", 0)) + 1
        self.group_rkv_quant_projection_counts["metal"] = int(
            self.group_rkv_quant_projection_counts.get("metal", 0)
        ) + 1
        return y_group[0], y_group[1], y_group[2]

    def _layer_norm(self, x, prefix: str):
        mx = _mx()
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        y = (xf - mean) * mx.rsqrt(var + self.norm_eps)
        y = y.astype(x.dtype)
        return y * self._get(f"{prefix}.weight") + self._get(f"{prefix}.bias")

    def _group_norm_heads(self, x, layer: int):
        mx = _mx()
        leading = tuple(int(dim) for dim in x.shape[:-1])
        xf = x.astype(mx.float32).reshape(*leading, self.num_heads, self.head_dim)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
        y = (xf - mean) * mx.rsqrt(var + self.head_dim * 1e-5)
        y = y.reshape(*leading, self.hidden_size).astype(x.dtype)
        prefix = f"model.layers.{layer}.attn.g_norm"
        return y * self._get(f"{prefix}.weight") + self._get(f"{prefix}.bias")

    def _normalize_last_dim(self, x, eps: float = 1e-12):
        mx = _mx()
        xf = x.astype(mx.float32)
        denom = mx.sqrt(mx.maximum(mx.sum(xf * xf, axis=-1, keepdims=True), eps))
        return (xf / denom).astype(x.dtype)

    def init_state(self, batch_size: int, *, dtype: Any | None = None) -> MLXRWKV7State:
        mx = _mx()
        if dtype is None:
            dtype = self._get("model.embeddings.weight").dtype
        B = int(batch_size)
        state = [
            mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
            for _ in range(self.num_hidden_layers)
        ]
        xpa = [mx.zeros((B, self.hidden_size), dtype=dtype) for _ in range(self.num_hidden_layers)]
        xpf = [mx.zeros((B, self.hidden_size), dtype=dtype) for _ in range(self.num_hidden_layers)]
        v_first = mx.zeros((B, self.hidden_size), dtype=dtype)
        mx.eval(v_first, *state, *xpa, *xpf)
        return MLXRWKV7State(state, xpa, xpf, v_first, seen_tokens=0)

    def _attn_step(self, layer: int, x, x_prev, v_first, state):
        mx = _mx()
        B = int(x.shape[0])
        hidden = self.hidden_size
        H = self.num_heads
        N = self.head_dim
        prefix = f"model.layers.{layer}.attn"
        if self.fused_attn_mix:
            (xr, xw, xk, xv, xa, xg), mix_backend = attn_mix(
                x,
                x_prev,
                self._get(f"{prefix}.x_r"),
                self._get(f"{prefix}.x_w"),
                self._get(f"{prefix}.x_k"),
                self._get(f"{prefix}.x_v"),
                self._get(f"{prefix}.x_a"),
                self._get(f"{prefix}.x_g"),
                backend="auto",
            )
            if mix_backend == "metal":
                self.fused_attn_mix_counts["metal"] = int(self.fused_attn_mix_counts.get("metal", 0)) + 1
            else:
                self.fused_attn_mix_counts["fallback"] = int(self.fused_attn_mix_counts.get("fallback", 0)) + 1
        else:
            xx = x_prev - x
            xr = x + xx * self._get(f"{prefix}.x_r").reshape(1, hidden)
            xw = x + xx * self._get(f"{prefix}.x_w").reshape(1, hidden)
            xk = x + xx * self._get(f"{prefix}.x_k").reshape(1, hidden)
            xv = x + xx * self._get(f"{prefix}.x_v").reshape(1, hidden)
            xa = x + xx * self._get(f"{prefix}.x_a").reshape(1, hidden)
            xg = x + xx * self._get(f"{prefix}.x_g").reshape(1, hidden)

        grouped_rkv = self._grouped_rkv_projection(layer, xr, xk, xv, prefix)
        if grouped_rkv is None:
            r = self._linear(xr, f"{prefix}.r_proj.weight")
            k = self._linear(xk, f"{prefix}.k_proj.weight")
            v = self._linear(xv, f"{prefix}.v_proj.weight")
        else:
            r, k, v = grouped_rkv
        w = self._linear(
            mx.tanh(self._linear(xw, f"{prefix}.w_lora.lora.0.weight")),
            f"{prefix}.w_lora.lora.2.weight",
            f"{prefix}.w_lora.lora.2.bias",
        )
        a = mx.sigmoid(
            self._linear(
                self._linear(xa, f"{prefix}.a_lora.lora.0.weight"),
                f"{prefix}.a_lora.lora.2.weight",
                f"{prefix}.a_lora.lora.2.bias",
            )
        )
        g = self._linear(
            mx.sigmoid(self._linear(xg, f"{prefix}.g_lora.lora.0.weight")),
            f"{prefix}.g_lora.lora.2.weight",
        )

        kk = self._normalize_last_dim((k * self._get(f"{prefix}.k_k").reshape(1, hidden)).reshape(B, H, N)).reshape(
            B, hidden
        )
        k = k * (1 + (a - 1) * self._get(f"{prefix}.k_a").reshape(1, hidden))
        if layer == 0:
            v_first = v
        else:
            v_mix = mx.sigmoid(
                self._linear(
                    self._linear(xv, f"{prefix}.v_lora.lora.0.weight"),
                    f"{prefix}.v_lora.lora.2.weight",
                    f"{prefix}.v_lora.lora.2.bias",
                )
            )
            v = v + (v_first - v) * v_mix
        w = mx.exp(-EXP_HALF * mx.sigmoid(w.astype(mx.float32)))

        out_heads, state, backend_used = wkv_update(
            state,
            w,
            v,
            k,
            kk,
            a,
            r,
            backend=self.wkv_backend,
        )
        self.wkv_backend_last = backend_used
        self.wkv_backend_counts[backend_used] = int(self.wkv_backend_counts.get(backend_used, 0)) + 1
        out = out_heads.reshape(B, hidden)
        out = self._group_norm_heads(out, layer)
        sk = (
            r.reshape(B, H, N)
            * k.reshape(B, H, N)
            * self._get(f"{prefix}.r_k").reshape(1, H, N)
        ).sum(axis=-1, keepdims=True)
        out = out + (sk * v.reshape(B, H, N)).reshape(B, hidden)
        out = self._linear(out * g, f"{prefix}.o_proj.weight")
        return out, x, state, v_first

    def _ffn_step(self, layer: int, x, x_prev):
        mx = _mx()
        prefix = f"model.layers.{layer}.ffn"
        xx = x_prev - x
        k = x + xx * self._get(f"{prefix}.x_k").reshape(1, self.hidden_size)
        key_weight = f"{prefix}.key.weight"
        key_qlinear = self.quantized_linears.get(key_weight)
        if (
            self.fused_ffn_key_relu2
            and key_qlinear is not None
            and int(key_qlinear.bits) == 4
            and key_qlinear._selected_backend(k) == "metal"
        ):
            k = key_qlinear.relu2(k)
            self.fused_ffn_key_relu2_counts["metal"] = int(
                self.fused_ffn_key_relu2_counts.get("metal", 0)
            ) + 1
        else:
            if self.fused_ffn_key_relu2:
                self.fused_ffn_key_relu2_counts["fallback"] = int(
                    self.fused_ffn_key_relu2_counts.get("fallback", 0)
                ) + 1
            k = mx.maximum(self._linear(k, key_weight), 0)
            k = k * k
        return self._linear(k, f"{prefix}.value.weight"), x


    def _shift_prev_sequence(self, x, x_prev):
        """Return per-token previous activations for a layer-major sequence."""

        mx = _mx()
        B, T, hidden = (int(dim) for dim in x.shape)
        first = x_prev.reshape(B, 1, hidden)
        if T == 1:
            return first
        return mx.concatenate([first, x[:, :-1, :]], axis=1)

    def _attn_sequence(self, layer: int, x, x_prev, v_first_seq, state):
        """Layer-major attention over a full prefill chunk using WKV scan."""

        mx = _mx()
        B, T, hidden = (int(dim) for dim in x.shape)
        H = self.num_heads
        N = self.head_dim
        prefix = f"model.layers.{layer}.attn"
        xp = self._shift_prev_sequence(x, x_prev)
        xx = xp - x
        xr = x + xx * self._get(f"{prefix}.x_r").reshape(1, 1, hidden)
        xw = x + xx * self._get(f"{prefix}.x_w").reshape(1, 1, hidden)
        xk = x + xx * self._get(f"{prefix}.x_k").reshape(1, 1, hidden)
        xv = x + xx * self._get(f"{prefix}.x_v").reshape(1, 1, hidden)
        xa = x + xx * self._get(f"{prefix}.x_a").reshape(1, 1, hidden)
        xg = x + xx * self._get(f"{prefix}.x_g").reshape(1, 1, hidden)

        grouped_rkv = self._grouped_rkv_projection(layer, xr, xk, xv, prefix)
        if grouped_rkv is None:
            r = self._linear(xr, f"{prefix}.r_proj.weight")
            k = self._linear(xk, f"{prefix}.k_proj.weight")
            v = self._linear(xv, f"{prefix}.v_proj.weight")
        else:
            r, k, v = grouped_rkv
        w = self._linear(
            mx.tanh(self._linear(xw, f"{prefix}.w_lora.lora.0.weight")),
            f"{prefix}.w_lora.lora.2.weight",
            f"{prefix}.w_lora.lora.2.bias",
        )
        a = mx.sigmoid(
            self._linear(
                self._linear(xa, f"{prefix}.a_lora.lora.0.weight"),
                f"{prefix}.a_lora.lora.2.weight",
                f"{prefix}.a_lora.lora.2.bias",
            )
        )
        g = self._linear(
            mx.sigmoid(self._linear(xg, f"{prefix}.g_lora.lora.0.weight")),
            f"{prefix}.g_lora.lora.2.weight",
        )

        kk = self._normalize_last_dim(
            (k * self._get(f"{prefix}.k_k").reshape(1, 1, hidden)).reshape(B, T, H, N)
        ).reshape(B, T, hidden)
        k = k * (1 + (a - 1) * self._get(f"{prefix}.k_a").reshape(1, 1, hidden))
        if layer == 0:
            new_v_first_seq = v
        else:
            v_mix = mx.sigmoid(
                self._linear(
                    self._linear(xv, f"{prefix}.v_lora.lora.0.weight"),
                    f"{prefix}.v_lora.lora.2.weight",
                    f"{prefix}.v_lora.lora.2.bias",
                )
            )
            v = v + (v_first_seq - v) * v_mix
            new_v_first_seq = v_first_seq
        w = mx.exp(-EXP_HALF * mx.sigmoid(w.astype(mx.float32)))

        out_heads, state, backend_used = wkv_scan(
            state,
            w.reshape(B, T, H, N),
            v.reshape(B, T, H, N),
            k.reshape(B, T, H, N),
            kk.reshape(B, T, H, N),
            a.reshape(B, T, H, N),
            r.reshape(B, T, H, N),
            backend=self.wkv_backend,
        )
        self.wkv_backend_last = backend_used
        self.wkv_backend_counts[backend_used] = int(self.wkv_backend_counts.get(backend_used, 0)) + 1
        self.wkv_scan_prefill_counts[backend_used] = int(self.wkv_scan_prefill_counts.get(backend_used, 0)) + 1
        out = out_heads.reshape(B, T, hidden)
        out = self._group_norm_heads(out, layer)
        sk = (
            r.reshape(B, T, H, N)
            * k.reshape(B, T, H, N)
            * self._get(f"{prefix}.r_k").reshape(1, 1, H, N)
        ).sum(axis=-1, keepdims=True)
        out = out + (sk * v.reshape(B, T, H, N)).reshape(B, T, hidden)
        out = self._linear(out * g, f"{prefix}.o_proj.weight")
        return out, x[:, -1, :], state, new_v_first_seq

    def _ffn_sequence(self, layer: int, x, x_prev):
        mx = _mx()
        B, T, hidden = (int(dim) for dim in x.shape)
        prefix = f"model.layers.{layer}.ffn"
        xp = self._shift_prev_sequence(x, x_prev)
        xx = xp - x
        k = x + xx * self._get(f"{prefix}.x_k").reshape(1, 1, hidden)
        key_weight = f"{prefix}.key.weight"
        key_qlinear = self.quantized_linears.get(key_weight)
        if (
            self.fused_ffn_key_relu2
            and key_qlinear is not None
            and int(key_qlinear.bits) == 4
            and key_qlinear._selected_backend(k) == "metal"
        ):
            k = key_qlinear.relu2(k)
            self.fused_ffn_key_relu2_counts["metal"] = int(
                self.fused_ffn_key_relu2_counts.get("metal", 0)
            ) + 1
        else:
            if self.fused_ffn_key_relu2:
                self.fused_ffn_key_relu2_counts["fallback"] = int(
                    self.fused_ffn_key_relu2_counts.get("fallback", 0)
                ) + 1
            k = mx.maximum(self._linear(k, key_weight), 0)
            k = k * k
        return self._linear(k, f"{prefix}.value.weight"), x[:, -1, :]

    def _should_scan_prefill(self, tokens: int) -> tuple[bool, str]:
        T = int(tokens)
        mode = self.wkv_scan_prefill_mode
        if T <= 1:
            return False, "single_token"
        if mode == "off":
            return False, "disabled"
        if mode == "on":
            return True, "forced"
        if mode == "auto":
            if self.wkv_backend == "metal" and not metal_wkv_scan_available():
                return False, "metal_unavailable"
            if T < int(self.wkv_scan_prefill_min_tokens):
                return False, "below_min_tokens"
            return True, "auto"
        return False, "disabled"

    def _record_scan_prefill_reason(self, reason: str) -> None:
        self.wkv_scan_prefill_reason_counts[reason] = int(
            self.wkv_scan_prefill_reason_counts.get(reason, 0)
        ) + 1

    def _forward_scan_prefill(
        self,
        input_ids: Iterable[Iterable[int]] | Any,
        state: MLXRWKV7State | None = None,
        *,
        collect_all: bool = False,
        state_only: bool = False,
    ):
        """Layer-major prefill path that calls one multi-token WKV scan per layer."""

        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if ids.ndim != 2:
            raise ValueError("MLXRWKV7Model scan prefill expects input ids shaped [batch, seq]")
        B, T = int(ids.shape[0]), int(ids.shape[1])
        if T <= 0 or B <= 0:
            raise ValueError("MLXRWKV7Model scan prefill requires a non-empty batch and sequence")
        if state is None:
            state = self.init_state(B)
        elif state.batch_size != B:
            raise ValueError(f"state batch size {state.batch_size} does not match input batch size {B}")

        x = self._get("model.embeddings.weight")[ids]
        v_first_seq = None
        for layer in range(self.num_hidden_layers):
            residual = self._layer_norm(x, f"model.layers.{layer}.pre_norm") if layer == 0 else x
            h = self._layer_norm(residual, f"model.layers.{layer}.attn_norm")
            if layer > 0 and v_first_seq is None:
                raise RuntimeError("RWKV-7 scan prefill missing layer-0 v_first sequence")
            a, state.attn_x_prev[layer], state.recurrent_state[layer], v_first_seq = self._attn_sequence(
                layer,
                h,
                state.attn_x_prev[layer],
                v_first_seq if v_first_seq is not None else state.v_first.reshape(B, 1, self.hidden_size),
                state.recurrent_state[layer],
            )
            x = residual + a
            residual = x
            h2 = self._layer_norm(x, f"model.layers.{layer}.ffn_norm")
            f, state.ffn_x_prev[layer] = self._ffn_sequence(layer, h2, state.ffn_x_prev[layer])
            x = residual + f

        state.seen_tokens += T
        if v_first_seq is not None:
            state.v_first = v_first_seq[:, -1, :]
        if state_only:
            self.state_only_prefill_calls += 1
            self.state_only_prefill_tokens += T
            self._eval_step_state(x[:, -1, :], state)
            return state
        if collect_all:
            out = self._logits_from_hidden(x)
        else:
            out = self._logits_from_hidden(x[:, -1, :]).reshape(B, 1, self.vocab_size)
        self._eval_step_state(out, state)
        return out, state

    def _embedding(self, token_ids):
        mx = _mx()
        ids = token_ids.astype(mx.int32).reshape(-1)
        return self._get("model.embeddings.weight")[ids]

    def _eval_step_state(self, x, state: MLXRWKV7State) -> None:
        mx = _mx()
        mx.eval(x, state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev)

    def _step_token(self, token_ids, state: MLXRWKV7State):
        mx = _mx()
        x = self._embedding(token_ids)
        for layer in range(self.num_hidden_layers):
            residual = self._layer_norm(x, f"model.layers.{layer}.pre_norm") if layer == 0 else x
            h = self._layer_norm(residual, f"model.layers.{layer}.attn_norm")
            a, state.attn_x_prev[layer], state.recurrent_state[layer], state.v_first = self._attn_step(
                layer,
                h,
                state.attn_x_prev[layer],
                state.v_first,
                state.recurrent_state[layer],
            )
            x = residual + a
            residual = x
            h2 = self._layer_norm(x, f"model.layers.{layer}.ffn_norm")
            f, state.ffn_x_prev[layer] = self._ffn_step(layer, h2, state.ffn_x_prev[layer])
            x = residual + f
        state.seen_tokens += 1
        if int(self.step_eval_interval) <= 1 or int(state.seen_tokens) % int(self.step_eval_interval) == 0:
            self._eval_step_state(x, state)
        return x, state

    def _logits_from_hidden(self, x):
        x = self._layer_norm(x, "model.norm")
        return self._linear(x, "lm_head.weight")

    def forward(self, input_ids: Iterable[Iterable[int]] | Any, state: MLXRWKV7State | None = None, *, collect_all: bool = True):
        """Run recurrent forward over ``input_ids`` shaped ``[B, T]``.

        Returns ``(logits, state)``.  With ``collect_all=True``, logits are
        shaped ``[B, T, vocab]``.  Otherwise only the last-token logits are
        returned as ``[B, 1, vocab]``.
        """

        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if ids.ndim != 2:
            raise ValueError("MLXRWKV7Model.forward expects input ids shaped [batch, seq]")
        B, T = int(ids.shape[0]), int(ids.shape[1])
        if T <= 0 or B <= 0:
            raise ValueError("MLXRWKV7Model.forward requires a non-empty batch and sequence")
        use_scan, scan_reason = self._should_scan_prefill(T)
        self._record_scan_prefill_reason(scan_reason)
        if use_scan:
            return self._forward_scan_prefill(ids, state=state, collect_all=collect_all)
        if state is None:
            state = self.init_state(B)
        elif state.batch_size != B:
            raise ValueError(f"state batch size {state.batch_size} does not match input batch size {B}")
        logits = []
        last = None
        for t in range(T):
            last, state = self._step_token(ids[:, t], state)
            if collect_all:
                logits.append(self._logits_from_hidden(last))
        if collect_all:
            out = mx.stack(logits, axis=1)
        else:
            out = self._logits_from_hidden(last).reshape(B, 1, self.vocab_size)
        if int(self.step_eval_interval) > 1:
            self._eval_step_state(out, state)
        else:
            mx.eval(out)
        return out, state

    def prefill_state_only(
        self,
        input_ids: Iterable[Iterable[int]] | Any,
        state: MLXRWKV7State | None = None,
    ) -> MLXRWKV7State:
        """Advance recurrent state over ``input_ids`` without producing logits.

        This is the serving/chunked-prefill fast path for non-final chunks:
        intermediate chunks only need to update recurrent state, so running the
        final layer norm and ``lm_head`` on every chunk boundary is wasted work.
        The full ``prefill`` path remains unchanged and final chunks still call
        ``forward(..., collect_all=False)`` to produce comparable last-token
        logits.
        """

        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if ids.ndim != 2:
            raise ValueError("MLXRWKV7Model.prefill_state_only expects input ids shaped [batch, seq]")
        B, T = int(ids.shape[0]), int(ids.shape[1])
        if T <= 0 or B <= 0:
            raise ValueError("MLXRWKV7Model.prefill_state_only requires a non-empty batch and sequence")
        use_scan, scan_reason = self._should_scan_prefill(T)
        self._record_scan_prefill_reason(scan_reason)
        if use_scan:
            return self._forward_scan_prefill(ids, state=state, state_only=True)
        if state is None:
            state = self.init_state(B)
        elif state.batch_size != B:
            raise ValueError(f"state batch size {state.batch_size} does not match input batch size {B}")
        last = None
        for t in range(T):
            last, state = self._step_token(ids[:, t], state)
        self.state_only_prefill_calls += 1
        self.state_only_prefill_tokens += T
        # Force the recurrent cache to materialize at the chunk boundary so the
        # lazy graph does not span an unbounded prompt.  This mirrors
        # ``forward(..., collect_all=False)`` final synchronization without the
        # last-token norm/lm_head projection.
        if last is not None:
            self._eval_step_state(last, state)
        return state

    def prefill(self, input_ids: Iterable[Iterable[int]] | Any, state: MLXRWKV7State | None = None):
        return self.forward(input_ids, state=state, collect_all=False)

    def decode_step(self, token_ids: Iterable[int] | Any, state: MLXRWKV7State):
        mx = _mx()
        ids = mx.array(token_ids, dtype=mx.int32).reshape(-1, 1)
        return self.forward(ids, state=state, collect_all=False)

    def decode_greedy(self, logits: Any, state: MLXRWKV7State, *, max_new_tokens: int):
        """Continue decoding from an existing prefill ``logits`` + ``state``.

        This is the serving-shaped path: callers prefill a prompt once, keep the
        recurrent state cache, and then decode one token at a time without
        recomputing the prompt.
        """

        mx = _mx()
        generated = []
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        for _ in range(int(max_new_tokens)):
            generated.append(next_token)
            logits, state = self.decode_step(next_token, state)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        if not generated:
            return mx.zeros((int(logits.shape[0]), 0), dtype=mx.int32), state
        out = mx.stack(generated, axis=1)
        mx.eval(out)
        return out, state

    def chunked_prefill(self, input_ids: Iterable[Iterable[int]] | Any, *, chunk_size: int):
        mx = _mx()
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        state = self.init_state(int(ids.shape[0]))
        logits = None
        total_tokens = int(ids.shape[1])
        for start in range(0, total_tokens, int(chunk_size)):
            end = min(start + int(chunk_size), total_tokens)
            chunk = ids[:, start:end]
            if end < total_tokens:
                state = self.prefill_state_only(chunk, state=state)
            else:
                logits, state = self.forward(chunk, state=state, collect_all=False)
        if logits is None:
            raise ValueError("chunked_prefill requires non-empty input")
        return logits, state

    def generate_greedy(self, input_ids: Iterable[Iterable[int]] | Any, *, max_new_tokens: int):
        logits, state = self.prefill(input_ids)
        return self.decode_greedy(logits, state, max_new_tokens=max_new_tokens)

    def generate_text(
        self,
        tokenizer: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        skip_special_tokens: bool = False,
    ) -> MLXGenerateOutput:
        """Encode ``prompt`` with an HF tokenizer and greedily decode on MLX.

        This is a lightweight reusable API for Apple-local demos and smoke
        harnesses.  It intentionally returns generated text only (not prompt +
        completion) so callers can decide how to display or postprocess.
        """

        session = MLXGenerationSession.from_prompt(
            self,
            tokenizer,
            prompt,
            skip_special_tokens=skip_special_tokens,
        )
        session.decode(int(max_new_tokens))
        return session.output()


def load_mlx_rwkv7_model(
    model_dir: str | Path,
    *,
    dtype: str | None = "fp16",
    quantization: str | None = None,
    quant_min_params: int = 8_000_000,
    quant_rkv_min_params: int | None = None,
    quant_backend: str = "affine",
    wkv_backend: str = "reference",
) -> MLXRWKV7Model:
    return MLXRWKV7Model.from_hf(
        model_dir,
        dtype=dtype,
        quantization=quantization,
        quant_min_params=quant_min_params,
        quant_rkv_min_params=quant_rkv_min_params,
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )


def load_mlx_generation_session(
    model_dir: str | Path,
    prompt: str,
    *,
    dtype: str | None = "fp16",
    skip_special_tokens: bool = False,
    quantization: str | None = None,
    quant_min_params: int = 8_000_000,
    quant_rkv_min_params: int | None = None,
    quant_backend: str = "affine",
    wkv_backend: str = "reference",
) -> MLXGenerationSession:
    """Load a converted HF directory and prefill a tokenizer-backed MLX session."""

    from transformers import AutoTokenizer

    model = load_mlx_rwkv7_model(
        model_dir,
        dtype=dtype,
        quantization=quantization,
        quant_min_params=quant_min_params,
        quant_rkv_min_params=quant_rkv_min_params,
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return MLXGenerationSession.from_prompt(
        model,
        tokenizer,
        prompt,
        skip_special_tokens=skip_special_tokens,
    )


def generate_text_from_hf(
    model_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int,
    dtype: str | None = "fp16",
    skip_special_tokens: bool = False,
    quantization: str | None = None,
    quant_min_params: int = 8_000_000,
    quant_rkv_min_params: int | None = None,
    quant_backend: str = "affine",
    wkv_backend: str = "reference",
) -> MLXGenerateOutput:
    """Load a converted HF directory and run tokenizer-integrated MLX generate."""

    from transformers import AutoTokenizer

    model = load_mlx_rwkv7_model(
        model_dir,
        dtype=dtype,
        quantization=quantization,
        quant_min_params=quant_min_params,
        quant_rkv_min_params=quant_rkv_min_params,
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model.generate_text(
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
    )
