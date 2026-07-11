# coding=utf-8
"""Tokenizer-backed MLX generation sessions and dynamic batch orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .mlx_bridge import require_mlx
from .mlx_policy import env_choice as _env_choice
from .mlx_policy import env_flag as _env_flag
from .mlx_policy import env_float as _env_float
from .mlx_state import MLXRWKV7State, _as_list

if TYPE_CHECKING:
    from .mlx_model import MLXRWKV7Model


def _mx():
    return require_mlx()


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

    def prepare_compiled_decode(
        self,
        *,
        validation_tokens: int = 32,
        logits_atol: float = 0.0,
        state_atol: float = 1e-6,
        reference_logits_atol: float = 0.25,
        reference_state_atol: float = 0.5,
    ) -> dict[str, Any]:
        """Warm and parity-gate compiled batch-one decode for this prompt."""

        self.model.prepare_compiled_decode(batch_size=1)
        return self.model.validate_compiled_decode(
            self.logits,
            self.state,
            steps=int(validation_tokens),
            logits_atol=float(logits_atol),
            state_atol=float(state_atol),
            reference_logits_atol=float(reference_logits_atol),
            reference_state_atol=float(reference_state_atol),
        )

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

__all__ = [
    "MLXGenerateOutput",
    "MLXGenerationSession",
    "MLXGenerationSessionBatch",
    "MLXSessionStepOutput",
]
