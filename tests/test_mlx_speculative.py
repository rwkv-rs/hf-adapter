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
