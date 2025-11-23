from __future__ import annotations

import importlib.util

import pytest

from rwkv7_hf.mlx_cache import MLXPrefixStateCache
from rwkv7_hf.mlx_scheduler import MLXBackpressureError, MLXDynamicBatchScheduler


def _tiny():
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is unavailable")
    from tests.test_apple_silicon_mlx_model_smoke import TinyTokenizer, tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    model.loaded_dtype = "fp16"
    model.requested_quantization = "none"
    return model, TinyTokenizer()


def test_ragged_arrival_departure_true_batch_and_prefix_reuse() -> None:
    model, tokenizer = _tiny()
    cache = MLXPrefixStateCache(
        model,
        max_entries=8,
        max_bytes=64 * 1024**2,
        ttl_s=None,
        tokenizer=tokenizer,
    )
    scheduler = MLXDynamicBatchScheduler(
        model,
        tokenizer,
        max_batch_size=2,
        max_in_flight=4,
        prefix_cache=cache,
        session_backend="auto",
        prepare_decode_policy=True,
        dtype="fp16",
        quantization="none",
    )
    cases = {
        "r1": ("tiny-a", 1),
        "r2": ("tiny-b", 3),
        "r3": ("tiny-c", 2),
    }
    expected = {
        request_id: model.generate_text(tokenizer, prompt, max_new_tokens=count).generated_ids
        for request_id, (prompt, count) in cases.items()
    }
    scheduler.submit(cases["r1"][0], max_new_tokens=1, request_id="r1")
    scheduler.submit(cases["r2"][0], max_new_tokens=3, request_id="r2")
    assert scheduler.step() == ["r1"]
    scheduler.submit(cases["r3"][0], max_new_tokens=2, request_id="r3")
    scheduler.run_until_idle()

    for request_id in cases:
        request = scheduler.request(request_id)
        assert request.status == "completed"
        assert request.generated_ids == expected[request_id]
        assert request.final_seen_tokens == request.prompt_tokens + request.max_new_tokens
        assert request.session is None  # recurrent arrays were released
        timing = request.telemetry()
        assert timing["activated_s"] is not None
        assert timing["first_token_s"] is not None
        assert timing["completed_s"] is not None
        assert timing["prefill_s"] >= 0
        assert timing["queue_s"] >= 0
        assert timing["ttft_s"] >= 0
        assert timing["e2e_s"] >= timing["ttft_s"]
    assert scheduler.batch_size_history == [2, 2, 2]
    assert scheduler.batch_backend_history == ["batched", "batched", "batched"]
    assert set(scheduler.telemetry()["prepared_policy_batches"]) == {2}
    assert scheduler.telemetry()["policy_telemetry_by_batch"][2]["status"] == "not_applicable"

    # The same prompt reuses an immutable cached prefix, then runs as a real
    # batch-one scheduler tick rather than a separate session code path.
    scheduler.submit("tiny-a", max_new_tokens=1, request_id="r4")
    live = scheduler.request("r4")
    assert live.session is not None and live.session.prefix_cache_exact
    scheduler.run_until_idle()
    assert scheduler.batch_size_history[-1] == 1
    assert scheduler.batch_backend_history[-1] == "batched"
    assert scheduler.request("r4").generated_ids == expected["r1"]
    assert scheduler.telemetry()["prefix_cache"]["exact_hits"] >= 1


def test_cancellation_backpressure_and_duplicate_ids() -> None:
    model, tokenizer = _tiny()
    scheduler = MLXDynamicBatchScheduler(
        model,
        tokenizer,
        max_batch_size=1,
        max_in_flight=1,
    )
    scheduler.submit("tiny-a", max_new_tokens=3, request_id="one")
    with pytest.raises(MLXBackpressureError, match="limit"):
        scheduler.submit("tiny-b", max_new_tokens=1, request_id="two")
    assert scheduler.cancel("one", reason="client_disconnect")
    request = scheduler.request("one")
    assert request.status == "cancelled"
    assert request.cancellation_reason == "client_disconnect"
    assert request.session is None
    assert scheduler.cancel("one") is False

    scheduler.submit("tiny-b", max_new_tokens=1, request_id="two")
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.submit("tiny-c", max_new_tokens=1, request_id="two")
    scheduler.run_until_idle()
    telemetry = scheduler.telemetry()
    assert telemetry["cancelled_count"] == 1
    assert telemetry["rejected_count"] == 1
    assert telemetry["completed_count"] == 1


def test_timeout_releases_state_without_decode() -> None:
    model, tokenizer = _tiny()
    now = [100.0]
    scheduler = MLXDynamicBatchScheduler(
        model,
        tokenizer,
        max_batch_size=1,
        max_in_flight=1,
        clock=lambda: now[0],
    )
    with pytest.raises(ValueError, match="timeout_s"):
        scheduler.submit("tiny", max_new_tokens=1, timeout_s=0)
    scheduler.submit("tiny", max_new_tokens=8, request_id="deadline", timeout_s=1.0)
    assert scheduler.request("deadline").session is not None
    now[0] += 2.0
    assert scheduler.step() == []
    request = scheduler.request("deadline")
    assert request.status == "timed_out"
    assert request.cancellation_reason == "timeout"
    assert request.generated_ids == []
    assert request.session is None
    timing = request.telemetry()
    assert timing["first_token_s"] is None
    assert timing["ttft_s"] is None
    assert timing["e2e_s"] == 2.0
    assert scheduler.in_flight == 0
    assert scheduler.telemetry()["timed_out_count"] == 1
