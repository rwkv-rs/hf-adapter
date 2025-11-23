# coding=utf-8
"""Greedy speculative decoding for the Apple MLX RWKV-7 backend.

The verifier uses the sequence-parallel DPLR prefill path, so several draft
tokens are checked by one target-model graph instead of one target decode graph
per token.  Rejections are replayed from shallow immutable MLX state forks,
which preserves exact target-greedy output without copying the recurrent cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .mlx_model import MLXRWKV7Model, MLXRWKV7State


def _mx():
    import mlx.core as mx

    return mx


def fork_state(state: MLXRWKV7State) -> MLXRWKV7State:
    """Fork the state container without copying immutable MLX arrays."""

    return MLXRWKV7State(
        recurrent_state=list(state.recurrent_state),
        attn_x_prev=list(state.attn_x_prev),
        ffn_x_prev=list(state.ffn_x_prev),
        v_first=state.v_first,
        seen_tokens=int(state.seen_tokens),
    )


def _greedy_id(logits: Any) -> int:
    mx = _mx()
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(token)
    values = token.tolist()
    if len(values) != 1:
        raise ValueError("MLX speculative decode currently supports batch size 1")
    return int(values[0])


def _consume(
    model: MLXRWKV7Model,
    token_ids: list[int],
    state: MLXRWKV7State,
):
    mx = _mx()
    if not token_ids:
        raise ValueError("cannot consume an empty token list")
    ids = mx.array([token_ids], dtype=mx.int32)
    if len(token_ids) == 1:
        return model.decode_step(ids.reshape(-1), state)
    return model.forward(ids, state=state, collect_all=False)


@dataclass
class MLXSpeculativeResult:
    generated_ids: list[int]
    target_state: MLXRWKV7State
    draft_state: MLXRWKV7State
    target_logits: Any
    draft_logits: Any
    elapsed_s: float
    first_token_s: float
    draft_tokens: int
    accepted_draft_tokens: int
    target_verify_calls: int
    target_replay_calls: int
    adaptive_fallback: bool = False
    fallback_tokens: int = 0

    @property
    def decode_tok_s(self) -> float:
        return len(self.generated_ids) / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.draft_tokens if self.draft_tokens else 0.0

    def telemetry(self) -> dict[str, Any]:
        return {
            "generated_tokens": len(self.generated_ids),
            "elapsed_s": round(float(self.elapsed_s), 6),
            "first_token_s": round(float(self.first_token_s), 6),
            "decode_tok_s": round(float(self.decode_tok_s), 6),
            "draft_tokens": int(self.draft_tokens),
            "accepted_draft_tokens": int(self.accepted_draft_tokens),
            "acceptance_rate": round(float(self.acceptance_rate), 6),
            "target_verify_calls": int(self.target_verify_calls),
            "target_replay_calls": int(self.target_replay_calls),
            "adaptive_fallback": bool(self.adaptive_fallback),
            "fallback_tokens": int(self.fallback_tokens),
            "generated_preview": self.generated_ids[:16],
        }


@dataclass
class MLXBatchSpeculativeResult:
    """Lockstep batched greedy speculation result."""

    generated_ids: list[list[int]]
    target_state: MLXRWKV7State
    draft_state: MLXRWKV7State
    target_logits: Any
    draft_logits: Any
    elapsed_s: float
    first_token_s: float
    draft_tokens: int
    accepted_draft_tokens: int
    target_verify_calls: int
    target_replay_calls: int

    @property
    def generated_tokens(self) -> int:
        return sum(len(row) for row in self.generated_ids)

    @property
    def decode_tok_s(self) -> float:
        return self.generated_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.draft_tokens if self.draft_tokens else 0.0

    def telemetry(self) -> dict[str, Any]:
        return {
            "batch_size": len(self.generated_ids),
            "generated_tokens": self.generated_tokens,
            "elapsed_s": round(float(self.elapsed_s), 6),
            "first_token_s": round(float(self.first_token_s), 6),
            "decode_tok_s": round(float(self.decode_tok_s), 6),
            "draft_tokens": int(self.draft_tokens),
            "accepted_draft_tokens": int(self.accepted_draft_tokens),
            "acceptance_rate": round(float(self.acceptance_rate), 6),
            "target_verify_calls": int(self.target_verify_calls),
            "target_replay_calls": int(self.target_replay_calls),
            "generated_preview": self.generated_ids[0][:16] if self.generated_ids else [],
        }


def speculative_decode_greedy_batch(
    target_model: MLXRWKV7Model,
    draft_model: MLXRWKV7Model,
    target_logits: Any,
    target_state: MLXRWKV7State,
    draft_logits: Any,
    draft_state: MLXRWKV7State,
    *,
    max_new_tokens: int,
    proposal_tokens: int = 32,
) -> MLXBatchSpeculativeResult:
    """Decode a synchronized batch while preserving each target-greedy row.

    The verifier stops a proposal block at the first sequence mismatch.  Rows
    whose draft token still matches simply emit the same target token at that
    position, so heterogeneous batches remain synchronized without changing
    any row's greedy output.
    """

    mx = _mx()
    total = int(max_new_tokens)
    width = int(proposal_tokens)
    batch = int(target_state.batch_size)
    if total < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if width < 2:
        raise ValueError("proposal_tokens must be at least 2")
    if batch <= 0 or int(draft_state.batch_size) != batch:
        raise ValueError("target and draft speculative states must have the same positive batch size")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match")

    generated: list[list[int]] = [[] for _ in range(batch)]
    proposed_total = 0
    accepted_total = 0
    verify_calls = 0
    replay_calls = 0
    started = time.perf_counter()
    first_token_s = 0.0
    if total:
        first = mx.argmax(target_logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(first)
        first_token_s = time.perf_counter() - started

    while len(generated[0]) < total:
        count = min(width, total - len(generated[0]))
        target_base = fork_state(target_state)
        draft_base = fork_state(draft_state)
        target_base_logits = target_logits
        proposals = []
        proposal_logits = draft_logits
        proposal_state = fork_state(draft_state)
        for _ in range(count):
            token = mx.argmax(proposal_logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(token)
            proposals.append(token)
            proposal_logits, proposal_state = draft_model.decode_step(token, proposal_state)
        proposal_block = mx.stack(proposals, axis=1)
        mx.eval(proposal_block)
        proposed_total += batch * count

        verify_logits, verify_state = target_model.forward(
            proposal_block,
            state=fork_state(target_state),
            collect_all=True,
        )
        mx.eval(verify_logits)
        verify_calls += 1

        mismatch = None
        mismatch_choice = None
        mismatch_matches = None
        for index in range(count):
            source = target_base_logits if index == 0 else verify_logits[:, index - 1 : index, :]
            choice = mx.argmax(source[:, -1, :], axis=-1).astype(mx.int32)
            matches = choice == proposals[index]
            mx.eval(choice, matches)
            matched = [bool(value) for value in matches.tolist()]
            if not all(matched):
                mismatch = index
                mismatch_choice = choice
                mismatch_matches = matched
                break

        if mismatch is None:
            values = proposal_block.tolist()
            for row, tokens in zip(generated, values, strict=True):
                row.extend(int(token) for token in tokens)
            accepted_total += batch * count
            target_state = verify_state
            target_logits = verify_logits[:, -1:, :]
            draft_state = proposal_state
            draft_logits = proposal_logits
            continue

        accepted_total += batch * mismatch + sum(bool(value) for value in mismatch_matches or [])
        emitted_columns = [*proposals[:mismatch], mismatch_choice]
        emitted = mx.stack(emitted_columns, axis=1)
        mx.eval(emitted)
        values = emitted.tolist()
        for row, tokens in zip(generated, values, strict=True):
            row.extend(int(token) for token in tokens)
        if int(emitted.shape[1]) == 1:
            target_logits, target_state = target_model.decode_step(emitted[:, 0], target_base)
            draft_logits, draft_state = draft_model.decode_step(emitted[:, 0], draft_base)
        else:
            target_logits, target_state = target_model.forward(emitted, state=target_base, collect_all=False)
            draft_logits, draft_state = draft_model.forward(emitted, state=draft_base, collect_all=False)
        replay_calls += 1

    mx.eval(target_logits, draft_logits)
    elapsed = time.perf_counter() - started
    return MLXBatchSpeculativeResult(
        generated_ids=generated,
        target_state=target_state,
        draft_state=draft_state,
        target_logits=target_logits,
        draft_logits=draft_logits,
        elapsed_s=elapsed,
        first_token_s=first_token_s,
        draft_tokens=proposed_total,
        accepted_draft_tokens=accepted_total,
        target_verify_calls=verify_calls,
        target_replay_calls=replay_calls,
    )


def speculative_decode_greedy(
    target_model: MLXRWKV7Model,
    draft_model: MLXRWKV7Model,
    target_logits: Any,
    target_state: MLXRWKV7State,
    draft_logits: Any,
    draft_state: MLXRWKV7State,
    *,
    max_new_tokens: int,
    proposal_tokens: int = 4,
    min_acceptance_rate: float = 0.25,
) -> MLXSpeculativeResult:
    """Decode exactly the target model's greedy sequence using an RWKV draft.

    Both models must use the same tokenizer/vocabulary and batch size one.  The
    target's DPLR chunk size should normally equal ``proposal_tokens`` so short
    verification blocks are not padded to a large prefill chunk.
    """

    mx = _mx()
    total = int(max_new_tokens)
    width = int(proposal_tokens)
    if total < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if width < 2:
        raise ValueError("proposal_tokens must be at least 2")
    if not 0.0 <= float(min_acceptance_rate) <= 1.0:
        raise ValueError("min_acceptance_rate must be in [0,1]")
    if target_state.batch_size != 1 or draft_state.batch_size != 1:
        raise ValueError("MLX speculative decode currently supports batch size 1")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match")

    generated: list[int] = []
    proposed_total = 0
    accepted_total = 0
    verify_calls = 0
    replay_calls = 0
    adaptive_fallback = False
    fallback_tokens = 0
    started = time.perf_counter()
    first_token_s = 0.0

    # The first target-greedy token is already available in the prefill
    # logits.  A serving loop can stream it immediately while the first draft
    # block and target verification graph execute.  Recording that point here
    # makes TTFT reflect the actual streaming boundary rather than waiting for
    # the whole first speculative block.
    if total:
        _greedy_id(target_logits)
        first_token_s = time.perf_counter() - started

    while len(generated) < total:
        if adaptive_fallback:
            # Bound the worst case for unrelated or short-context drafts.  We
            # keep both recurrent states synchronized so callers may reuse the
            # result even though the rest of this request is target-only.
            token = _greedy_id(target_logits)
            generated.append(token)
            target_logits, target_state = _consume(target_model, [token], target_state)
            draft_logits, draft_state = _consume(draft_model, [token], draft_state)
            fallback_tokens += 1
            continue
        count = min(width, total - len(generated))
        target_base = fork_state(target_state)
        draft_base = fork_state(draft_state)
        target_base_logits = target_logits

        proposals: list[int] = []
        proposal_logits = draft_logits
        proposal_state = fork_state(draft_state)
        for _ in range(count):
            token = _greedy_id(proposal_logits)
            proposals.append(token)
            proposal_logits, proposal_state = draft_model.decode_step(
                mx.array([token], dtype=mx.int32), proposal_state
            )
        proposed_total += len(proposals)

        verify_logits, verify_state = target_model.forward(
            mx.array([proposals], dtype=mx.int32),
            state=fork_state(target_state),
            collect_all=True,
        )
        mx.eval(verify_logits)
        verify_calls += 1

        mismatch = None
        target_choice = None
        for index, proposal in enumerate(proposals):
            source = target_base_logits if index == 0 else verify_logits[:, index - 1 : index, :]
            choice = _greedy_id(source)
            if choice != proposal:
                mismatch = index
                target_choice = choice
                break

        if mismatch is None:
            generated.extend(proposals)
            accepted_total += len(proposals)
            target_state = verify_state
            target_logits = verify_logits[:, -1:, :]
            draft_state = proposal_state
            draft_logits = proposal_logits
            continue

        accepted = proposals[:mismatch]
        accepted_total += len(accepted)
        emitted = [*accepted, int(target_choice)]
        generated.extend(emitted)
        target_logits, target_state = _consume(target_model, emitted, target_base)
        draft_logits, draft_state = _consume(draft_model, emitted, draft_base)
        replay_calls += 1
        if proposed_total >= width and accepted_total / proposed_total < float(min_acceptance_rate):
            adaptive_fallback = True

    mx.eval(target_logits, draft_logits)
    elapsed = time.perf_counter() - started
    return MLXSpeculativeResult(
        generated_ids=generated,
        target_state=target_state,
        draft_state=draft_state,
        target_logits=target_logits,
        draft_logits=draft_logits,
        elapsed_s=elapsed,
        first_token_s=first_token_s,
        draft_tokens=proposed_total,
        accepted_draft_tokens=accepted_total,
        target_verify_calls=verify_calls,
        target_replay_calls=replay_calls,
        adaptive_fallback=adaptive_fallback,
        fallback_tokens=fallback_tokens,
    )
