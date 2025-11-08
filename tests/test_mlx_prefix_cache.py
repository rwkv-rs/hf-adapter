from __future__ import annotations

import importlib.util

import pytest

from rwkv7_hf.mlx_cache import MLXPrefixStateCache, mlx_model_cache_fingerprint


def _mlx_model():
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX is unavailable")
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    _, model, _ = tiny_torch_model_to_mlx()
    model.loaded_dtype = "fp16"
    return model


def _max_state_diff(mx, left, right) -> float:
    arrays_left = [left.v_first, *left.recurrent_state, *left.attn_x_prev, *left.ffn_x_prev]
    arrays_right = [right.v_first, *right.recurrent_state, *right.attn_x_prev, *right.ffn_x_prev]
    return max(
        float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        for a, b in zip(arrays_left, arrays_right, strict=True)
    )


def test_prefix_cache_exact_longest_and_state_isolation() -> None:
    model = _mlx_model()
    import mlx.core as mx
    cache = MLXPrefixStateCache(model, max_entries=4, max_bytes=64 * 1024**2, ttl_s=None)
    prefix = [1, 2]
    full = [1, 2, 3, 4]
    prefix_logits, prefix_state = model.prefill([prefix])
    assert cache.put(prefix, prefix_logits, prefix_state)

    exact = cache.get_exact(prefix)
    assert exact is not None and exact.exact and exact.prefix_tokens == 2
    assert exact.state is not prefix_state
    next_token = mx.argmax(exact.logits[:, -1, :], axis=-1).astype(mx.int32)
    model.decode_step(next_token, exact.state)
    untouched = cache.get_exact(prefix)
    assert untouched is not None
    assert int(untouched.state.seen_tokens) == 2

    longest = cache.find_longest(full)
    assert longest is not None and not longest.exact and longest.prefix_tokens == 2
    cached_logits, cached_state = model.prefill([full[2:]], state=longest.state)
    full_logits, full_state = model.prefill([full])
    mx.eval(cached_logits, full_logits)
    assert float(mx.max(mx.abs(cached_logits.astype(mx.float32) - full_logits.astype(mx.float32)))) < 1e-5
    assert _max_state_diff(mx, cached_state, full_state) < 1e-5
    assert cache.put(full, cached_logits, cached_state)
    assert cache.telemetry()["hits"] == 3
    assert cache.telemetry()["exact_hits"] == 2
    assert cache.telemetry()["prefix_hits"] == 1


def test_prefix_cache_lru_ttl_oversize_and_namespace_isolation() -> None:
    model = _mlx_model()
    assert mlx_model_cache_fingerprint(model, namespace="a") != mlx_model_cache_fingerprint(
        model, namespace="b"
    )
    now = [100.0]
    cache = MLXPrefixStateCache(
        model,
        max_entries=2,
        max_bytes=64 * 1024**2,
        ttl_s=10,
        clock=lambda: now[0],
        namespace="tenant-a",
    )

    def put(tokens: list[int]) -> None:
        logits, state = model.prefill([tokens])
        assert cache.put(tokens, logits, state)

    put([1])
    put([2])
    assert cache.get_exact([1]) is not None  # [2] becomes LRU.
    put([3])
    assert cache.get_exact([2]) is None
    assert cache.get_exact([1]) is not None
    assert cache.telemetry()["evictions"] == 1

    now[0] += 11
    assert cache.get_exact([1]) is None
    assert cache.telemetry()["expirations"] >= 1

    tiny_budget = MLXPrefixStateCache(model, max_entries=1, max_bytes=1, ttl_s=None)
    logits, state = model.prefill([[1]])
    assert tiny_budget.put([1], logits, state) is False
    assert tiny_budget.telemetry()["rejected_oversize"] == 1


def test_generation_session_uses_exact_cache_without_state_aliasing() -> None:
    from rwkv7_hf.mlx_scheduler import create_cached_mlx_generation_session
    from tests.test_apple_silicon_mlx_model_smoke import TinyTokenizer

    model = _mlx_model()
    tokenizer = TinyTokenizer()
    cache = MLXPrefixStateCache(
        model,
        max_entries=4,
        max_bytes=64 * 1024**2,
        ttl_s=None,
        tokenizer=tokenizer,
    )
    first = create_cached_mlx_generation_session(
        model,
        tokenizer,
        "tiny-prefix",
        prefix_cache=cache,
    )
    assert first.prefix_cache_hit is False
    assert first.prefill_tokens_computed == first.prompt_tokens
    first.decode(4)

    second = create_cached_mlx_generation_session(
        model,
        tokenizer,
        "tiny-prefix",
        prefix_cache=cache,
    )
    assert second.prefix_cache_hit is True
    assert second.prefix_cache_exact is True
    assert second.prefix_tokens_reused == second.prompt_tokens
    assert second.prefill_tokens_computed == 0
    second.decode(4)
    assert second.generated_ids == first.generated_ids
    assert second.text == first.text
    assert cache.telemetry()["hit_rate"] > 0
    assert cache.telemetry()["tokenizer_fingerprint"] is not None

    other_model = _mlx_model()
    with pytest.raises(ValueError, match="different MLX model"):
        create_cached_mlx_generation_session(
            other_model,
            tokenizer,
            "tiny-prefix",
            prefix_cache=cache,
        )

    class OtherTokenizer(TinyTokenizer):
        pass

    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        create_cached_mlx_generation_session(
            model,
            OtherTokenizer(),
            "tiny-prefix",
            prefix_cache=cache,
        )

    model.prefill_backend = "auto" if model.prefill_backend != "auto" else "recurrent"
    with pytest.raises(ValueError, match="execution configuration"):
        create_cached_mlx_generation_session(
            model,
            tokenizer,
            "tiny-prefix",
            prefix_cache=cache,
        )
