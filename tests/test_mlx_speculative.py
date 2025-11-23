from __future__ import annotations

import importlib.util


def test_mlx_speculative_same_model_matches_greedy_if_available(monkeypatch) -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_dplr_prefill import mlx_dplr_metal_available
    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_speculative import speculative_decode_greedy
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not mlx_dplr_metal_available():
        return
    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "0")
    _, source, cfg = tiny_torch_model_to_mlx()
    arrays = dict(source.arrays)
    greedy = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    target = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    draft = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    target.prefill_backend = "dplr_metal"
    target.dplr_chunk_size = 4
    target.dplr_min_tokens = 2
    target.dplr_layer_eval_interval = 1
    target.dplr_layer_eval_min_tokens = 1
    draft.prefill_backend = "dplr_metal"
    draft.dplr_chunk_size = 4
    draft.dplr_min_tokens = 2

    prompt = [[1, 2, 3, 4]]
    greedy_logits, greedy_state = greedy.prefill(prompt)
    expected, greedy_state = greedy.decode_greedy(greedy_logits, greedy_state, max_new_tokens=8)
    target_logits, target_state = target.prefill(prompt)
    draft_logits, draft_state = draft.prefill(prompt)
    result = speculative_decode_greedy(
        target,
        draft,
        target_logits,
        target_state,
        draft_logits,
        draft_state,
        max_new_tokens=8,
        proposal_tokens=4,
    )

    assert result.generated_ids == [int(value) for value in expected.reshape(-1).tolist()]
    assert result.acceptance_rate == 1.0
    assert result.target_verify_calls == 2
    assert result.target_replay_calls == 0
    assert result.target_state.seen_tokens == 12
    mx.eval(*result.target_state.recurrent_state, *greedy_state.recurrent_state)
    for actual, reference in zip(
        result.target_state.recurrent_state,
        greedy_state.recurrent_state,
        strict=True,
    ):
        # DPLR changes fp32 accumulation order while preserving exact greedy
        # tokens; keep the same bounded-state criterion as the DPLR parity gate.
        assert float(mx.max(mx.abs(actual - reference))) < 1e-2


def test_mlx_batch_speculative_same_model_matches_each_greedy_row_if_available(monkeypatch) -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_dplr_prefill import mlx_dplr_metal_available
    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_speculative import speculative_decode_greedy_batch
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not mlx_dplr_metal_available():
        return
    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "0")
    _, source, cfg = tiny_torch_model_to_mlx()
    arrays = dict(source.arrays)
    greedy = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    target = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    draft = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    for model in (target, draft):
        model.prefill_backend = "dplr_metal"
        model.dplr_chunk_size = 4
        model.dplr_min_tokens = 2
        model.dplr_layer_eval_interval = 1
        model.dplr_layer_eval_min_tokens = 1

    prompt = [[1, 2, 3, 4], [4, 3, 2, 1]]
    greedy_logits, greedy_state = greedy.prefill(prompt)
    expected, greedy_state = greedy.decode_greedy(greedy_logits, greedy_state, max_new_tokens=8)
    target_logits, target_state = target.prefill(prompt)
    draft_logits, draft_state = draft.prefill(prompt)
    result = speculative_decode_greedy_batch(
        target,
        draft,
        target_logits,
        target_state,
        draft_logits,
        draft_state,
        max_new_tokens=8,
        proposal_tokens=4,
    )

    assert result.generated_ids == [
        [int(value) for value in row]
        for row in expected.tolist()
    ]
    assert result.acceptance_rate == 1.0
    assert result.target_verify_calls == 2
    assert result.target_replay_calls == 0
    assert result.target_state.seen_tokens == 12
    assert result.telemetry()["batch_size"] == 2
    mx.eval(*result.target_state.recurrent_state, *greedy_state.recurrent_state)
    for actual, reference in zip(
        result.target_state.recurrent_state,
        greedy_state.recurrent_state,
        strict=True,
    ):
        assert float(mx.max(mx.abs(actual - reference))) < 1e-2


def test_mlx_batch_speculative_rejection_replays_exact_target_if_available(monkeypatch) -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_dplr_prefill import mlx_dplr_metal_available
    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from rwkv7_hf.mlx_speculative import speculative_decode_greedy_batch
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    if not mlx_dplr_metal_available():
        return
    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "0")
    _, source, cfg = tiny_torch_model_to_mlx()
    arrays = dict(source.arrays)
    draft_arrays = dict(arrays)
    draft_arrays["lm_head.weight"] = -draft_arrays["lm_head.weight"]
    greedy = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    target = MLXRWKV7Model.from_arrays(cfg, dict(arrays), wkv_backend="metal")
    draft = MLXRWKV7Model.from_arrays(cfg, draft_arrays, wkv_backend="metal")
    for model in (target, draft):
        model.prefill_backend = "dplr_metal"
        model.dplr_chunk_size = 4
        model.dplr_min_tokens = 2
        model.dplr_layer_eval_interval = 1
        model.dplr_layer_eval_min_tokens = 1

    prompt = [[1, 2, 3, 4], [4, 3, 2, 1]]
    greedy_logits, greedy_state = greedy.prefill(prompt)
    expected, _ = greedy.decode_greedy(greedy_logits, greedy_state, max_new_tokens=8)
    target_logits, target_state = target.prefill(prompt)
    draft_logits, draft_state = draft.prefill(prompt)
    result = speculative_decode_greedy_batch(
        target,
        draft,
        target_logits,
        target_state,
        draft_logits,
        draft_state,
        max_new_tokens=8,
        proposal_tokens=4,
    )

    assert result.generated_ids == expected.tolist()
    assert result.target_replay_calls > 0
    assert result.acceptance_rate < 1.0
    assert result.target_state.seen_tokens == 12
    mx.eval(*result.target_state.recurrent_state)


def test_mlx_identical_prefix_state_coalescing_matches_cold_batch_if_available(monkeypatch) -> None:
    if importlib.util.find_spec("mlx") is None:
        return
    import mlx.core as mx

    from rwkv7_hf.mlx_model import MLXRWKV7Model
    from tests.test_apple_silicon_mlx_model_smoke import tiny_torch_model_to_mlx

    monkeypatch.setenv("RWKV7_MLX_WKV_SCAN_PREFILL", "1")
    _, source, cfg = tiny_torch_model_to_mlx()
    model = MLXRWKV7Model.from_arrays(cfg, dict(source.arrays), wkv_backend="reference")
    prompt = [[1, 2, 3, 4], [1, 2, 3, 4]]
    cold_logits, cold_state = model.prefill(prompt)
    cold_tokens, cold_final = model.decode_greedy(cold_logits, cold_state.clone(), max_new_tokens=8)

    seed_logits, seed_state = model.prefill(prompt[:1])
    cached_logits = mx.repeat(seed_logits, 2, axis=0)
    cached_state = seed_state.select_batch([0, 0])
    cached_tokens, cached_final = model.decode_greedy(cached_logits, cached_state, max_new_tokens=8)
    mx.eval(cold_logits, cached_logits, cold_tokens, cached_tokens)

    assert cached_tokens.tolist() == cold_tokens.tolist()
    assert float(mx.max(mx.abs(cached_logits - cold_logits))) < 1e-3
    for actual, reference in zip(
        [*cached_final.recurrent_state, *cached_final.attn_x_prev, *cached_final.ffn_x_prev],
        [*cold_final.recurrent_state, *cold_final.attn_x_prev, *cold_final.ffn_x_prev],
        strict=True,
    ):
        assert float(mx.max(mx.abs(actual - reference))) < 1e-3
