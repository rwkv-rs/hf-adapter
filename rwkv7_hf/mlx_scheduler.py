# coding=utf-8
"""Production-shaped ragged dynamic batching for MLX RWKV-7 sessions."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .mlx_session import MLXGenerationSession, MLXGenerationSessionBatch


class MLXBackpressureError(RuntimeError):
    """Raised when the configured in-flight request budget is exhausted."""


def create_cached_mlx_generation_session(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    prefix_cache: Any | None = None,
    skip_special_tokens: bool = False,
) -> MLXGenerationSession:
    """Create a serving session, reusing its longest immutable prefix state.

    Cache-aware construction belongs at the serving boundary rather than in
    :mod:`mlx_session`, keeping tokenizer sessions independent of cache policy
    after the MLX runtime module split.
    """

    encoded = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = [int(token) for token in encoded.input_ids]
    if not prompt_ids:
        raise ValueError("prompt produced no token ids")

    started = time.perf_counter()
    cache_hit = None
    if prefix_cache is not None:
        prefix_cache.assert_compatible(model, tokenizer)
        cache_hit = prefix_cache.find_longest(prompt_ids)

    if cache_hit is None:
        logits, state = model.prefill([prompt_ids])
        prefix_tokens_reused = 0
        prefill_tokens_computed = len(prompt_ids)
        if prefix_cache is not None:
            prefix_cache.put(prompt_ids, logits, state)
    else:
        logits, state = cache_hit.logits, cache_hit.state
        prefix_tokens_reused = int(cache_hit.prefix_tokens)
        suffix = prompt_ids[prefix_tokens_reused:]
        prefill_tokens_computed = len(suffix)
        if suffix:
            logits, state = model.prefill([suffix], state=state)
            prefix_cache.put(prompt_ids, logits, state)

    session = MLXGenerationSession(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_ids=prompt_ids,
        logits=logits,
        state=state,
        prefill_s=time.perf_counter() - started,
        skip_special_tokens=skip_special_tokens,
    )
    # Preserve cache observability without making mlx_session depend on a
    # concrete cache implementation.
    session.prefix_cache_hit = cache_hit is not None
    session.prefix_cache_exact = bool(cache_hit is not None and cache_hit.exact)
    session.prefix_tokens_reused = prefix_tokens_reused
    session.prefill_tokens_computed = prefill_tokens_computed
    return session


@dataclass
class MLXDynamicRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    arrival_tick: int
    submitted_s: float
    deadline_s: float | None = None
    status: str = "queued"
    session: MLXGenerationSession | None = field(default=None, repr=False)
    generated_ids: list[int] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    final_seen_tokens: int | None = None
    completion_tick: int | None = None
    cancellation_reason: str | None = None
    prefix_cache_hit: bool = False
    prefix_cache_exact: bool = False
    prefix_tokens_reused: int = 0
    prefill_tokens_computed: int = 0

    @property
    def generated_tokens(self) -> int:
        if self.session is not None:
            return self.session.generated_tokens
        return len(self.generated_ids)

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.max_new_tokens) - self.generated_tokens)

    def telemetry(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "arrival_tick": self.arrival_tick,
            "submitted_s": round(float(self.submitted_s), 6),
            "deadline_s": round(float(self.deadline_s), 6) if self.deadline_s is not None else None,
            "completion_tick": self.completion_tick,
            "max_new_tokens": self.max_new_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "remaining_tokens": self.remaining_tokens,
            "final_seen_tokens": self.final_seen_tokens,
            "cancellation_reason": self.cancellation_reason,
            "prefix_cache_hit": self.prefix_cache_hit,
            "prefix_cache_exact": self.prefix_cache_exact,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "prefill_tokens_computed": self.prefill_tokens_computed,
            "generated_preview": self.generated_ids[:16]
            if self.session is None
            else self.session.generated_ids[:16],
        }


class MLXDynamicBatchScheduler:
    """FIFO scheduler with true stacked-state decode and dynamic departures.

    Prefill happens at submission. Each scheduler tick advances every active
    request by one token through one :class:`MLXGenerationSessionBatch`, then
    releases completed/cancelled session states and promotes queued requests.
    This makes batch-size changes explicit and testable without pretending a
    Python loop over independent sessions is dynamic batching.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_batch_size: int = 8,
        max_in_flight: int = 128,
        prefix_cache: Any | None = None,
        session_backend: str = "auto",
        prepare_decode_policy: bool = False,
        dtype: str | None = None,
        quantization: str | None = None,
        policy_device: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if int(max_batch_size) <= 0:
            raise ValueError("max_batch_size must be positive")
        if int(max_in_flight) < int(max_batch_size):
            raise ValueError("max_in_flight must be at least max_batch_size")
        if session_backend not in {"auto", "batched", "batched_stable"}:
            raise ValueError("session_backend must be auto, batched, or batched_stable")
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = int(max_batch_size)
        self.max_in_flight = int(max_in_flight)
        self.prefix_cache = prefix_cache
        self.session_backend = session_backend
        self.prepare_policy = bool(prepare_decode_policy)
        self.dtype = dtype if dtype is not None else getattr(model, "loaded_dtype", None)
        self.quantization = (
            quantization
            if quantization is not None
            else getattr(model, "requested_quantization", None)
        )
        self.policy_device = policy_device
        self._clock = clock
        self.tick = 0
        self._next_id = 1
        self._requests: OrderedDict[str, MLXDynamicRequest] = OrderedDict()
        self._queued: OrderedDict[str, MLXDynamicRequest] = OrderedDict()
        self._active: OrderedDict[str, MLXDynamicRequest] = OrderedDict()
        self._prepared_policy_batches: set[int] = set()
        self.policy_telemetry_by_batch: dict[int, dict[str, Any]] = {}
        self.batch_size_history: list[int] = []
        self.batch_backend_history: list[str] = []
        self.batch_backend_reason_history: list[str] = []
        self.completed_count = 0
        self.cancelled_count = 0
        self.timed_out_count = 0
        self.rejected_count = 0

    @property
    def in_flight(self) -> int:
        return len(self._queued) + len(self._active)

    def _promote(self) -> None:
        while self._queued and len(self._active) < self.max_batch_size:
            request_id, request = self._queued.popitem(last=False)
            request.status = "active"
            self._active[request_id] = request

    def submit(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        request_id: str | None = None,
        skip_special_tokens: bool = False,
        timeout_s: float | None = None,
    ) -> str:
        count = int(max_new_tokens)
        if count <= 0:
            raise ValueError("max_new_tokens must be positive")
        if timeout_s is not None and float(timeout_s) <= 0:
            raise ValueError("timeout_s must be positive or None")
        selected_id = request_id or f"mlx-{self._next_id}"
        if selected_id in self._requests:
            raise ValueError(f"duplicate request_id {selected_id!r}")
        self._expire()
        self._promote()
        if self.in_flight >= self.max_in_flight:
            self.rejected_count += 1
            raise MLXBackpressureError(
                f"in-flight request limit {self.max_in_flight} is full"
            )
        if request_id is None:
            self._next_id += 1
        submitted_s = float(self._clock())
        session = create_cached_mlx_generation_session(
            self.model,
            self.tokenizer,
            prompt,
            skip_special_tokens=skip_special_tokens,
            prefix_cache=self.prefix_cache,
        )
        request = MLXDynamicRequest(
            request_id=selected_id,
            prompt=prompt,
            max_new_tokens=count,
            arrival_tick=self.tick,
            submitted_s=submitted_s,
            deadline_s=(submitted_s + float(timeout_s)) if timeout_s is not None else None,
            session=session,
            prompt_tokens=session.prompt_tokens,
            prefix_cache_hit=bool(session.prefix_cache_hit),
            prefix_cache_exact=bool(session.prefix_cache_exact),
            prefix_tokens_reused=int(session.prefix_tokens_reused),
            prefill_tokens_computed=int(session.prefill_tokens_computed),
        )
        self._requests[selected_id] = request
        self._queued[selected_id] = request
        self._promote()
        return selected_id

    def _expire(self) -> list[str]:
        now = float(self._clock())
        expired: list[str] = []
        for collection in (self._active, self._queued):
            for request_id, request in list(collection.items()):
                if request.deadline_s is None or now < request.deadline_s:
                    continue
                collection.pop(request_id)
                request.cancellation_reason = "timeout"
                self._release(request, status="timed_out")
                self.timed_out_count += 1
                expired.append(request_id)
        return expired

    def _release(self, request: MLXDynamicRequest, *, status: str) -> None:
        if request.session is not None:
            request.generated_ids = list(request.session.generated_ids)
            request.text = request.session.text
            request.final_seen_tokens = int(request.session.state.seen_tokens)
        request.session = None
        request.status = status
        request.completion_tick = self.tick

    def cancel(self, request_id: str, *, reason: str = "cancelled") -> bool:
        request = self._active.pop(request_id, None)
        if request is None:
            request = self._queued.pop(request_id, None)
        if request is None:
            return False
        request.cancellation_reason = str(reason)
        self._release(request, status="cancelled")
        self.cancelled_count += 1
        self._promote()
        return True

    def _prepare_batch_policy(self, manager: MLXGenerationSessionBatch) -> None:
        """Retain API telemetry after policy ownership moved out of serving.

        Current model/session backends own eager/compiled and quant safety
        selection. The old exact-device decode policy no longer exists on the
        refactored mainline, so the scheduler records the request without
        mutating model-wide backend state between concurrent batches.
        """

        batch = manager.batch_size
        if batch in self._prepared_policy_batches:
            return
        self._prepared_policy_batches.add(batch)
        self.policy_telemetry_by_batch[batch] = {
            "status": "not_applicable",
            "reason": "decode_policy_owned_by_model_and_session_backends",
            "batch_size": int(batch),
        }

    def step(self) -> list[str]:
        self._expire()
        self._promote()
        if not self._active:
            return []
        active = list(self._active.values())
        sessions = [request.session for request in active]
        if any(session is None for session in sessions):
            raise RuntimeError("active scheduler request has no session")
        manager = MLXGenerationSessionBatch(sessions)  # type: ignore[arg-type]
        if self.prepare_policy:
            self._prepare_batch_policy(manager)
        manager.decode_round(1, backend=self.session_backend)
        self.tick += 1
        self.batch_size_history.append(manager.batch_size)
        telemetry = manager.telemetry()
        self.batch_backend_history.append(str(telemetry["last_round_backend"]))
        self.batch_backend_reason_history.append(str(telemetry["last_round_backend_reason"]))

        completed: list[str] = []
        for request_id, request in list(self._active.items()):
            if request.remaining_tokens == 0:
                self._active.pop(request_id)
                self._release(request, status="completed")
                self.completed_count += 1
                completed.append(request_id)
        self._expire()
        self._promote()
        return completed

    def run_until_idle(self, *, max_ticks: int = 100_000) -> None:
        limit = int(max_ticks)
        if limit <= 0:
            raise ValueError("max_ticks must be positive")
        iterations = 0
        while self.in_flight:
            self.step()
            iterations += 1
            if iterations >= limit and self.in_flight:
                raise TimeoutError(f"scheduler still has {self.in_flight} requests after {limit} ticks")

    def request(self, request_id: str) -> MLXDynamicRequest:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id {request_id!r}") from exc

    def telemetry(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for request in self._requests.values():
            status_counts[request.status] = status_counts.get(request.status, 0) + 1
        return {
            "tick": self.tick,
            "max_batch_size": self.max_batch_size,
            "max_in_flight": self.max_in_flight,
            "in_flight": self.in_flight,
            "queued": len(self._queued),
            "active": len(self._active),
            "completed_count": self.completed_count,
            "cancelled_count": self.cancelled_count,
            "timed_out_count": self.timed_out_count,
            "rejected_count": self.rejected_count,
            "status_counts": status_counts,
            "batch_size_history": list(self.batch_size_history),
            "batch_backend_history": list(self.batch_backend_history),
            "batch_backend_reason_history": list(self.batch_backend_reason_history),
            "prepared_policy_batches": sorted(self._prepared_policy_batches),
            "policy_telemetry_by_batch": dict(self.policy_telemetry_by_batch),
            "prefix_cache": self.prefix_cache.telemetry() if self.prefix_cache is not None else None,
            "requests": [request.telemetry() for request in self._requests.values()],
        }


__all__ = [
    "MLXBackpressureError",
    "MLXDynamicBatchScheduler",
    "MLXDynamicRequest",
    "create_cached_mlx_generation_session",
]
