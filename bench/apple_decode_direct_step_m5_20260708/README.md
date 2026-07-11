# Apple M5 decode direct-step experiment (2026-07-08)

This batch tests a decode-path cleanup in `MLXRWKV7Model.decode_step`: instead
of routing one-token decode through `forward(T=1)`, decode now calls `_step_token`
directly and returns logits.  This avoids forcing a full recurrent-state sync at
the end of every decode step; state synchronization is still controlled by
`RWKV7_MLX_STEP_EVAL_INTERVAL` inside `_step_token`.

## Correctness

A tiny MLX model test verifies that direct `decode_step([token], state)` matches
`forward([[token]], state, collect_all=False)` exactly for logits and seen-token
count.

## Performance observations

Files:

- `results_rwkv04_512_64_direct_decode.jsonl`
- `results_rwkv15_512_64_direct_decode.jsonl`
- `results_rwkv15_512_64_direct_decode_warm1.jsonl`

Current Apple M5 rows did **not** show a stable positive speedup versus the best
previous scan-prefill rows.  They are kept as evidence for the decode-barrier
experiment, not as a Qwen3.5 win claim.

| Model | Shape | Prefill tok/s | Decode tok/s | Notes |
|---|---|---:|---:|---|
| RWKV-7 0.4B mm4 | 512 / 64 | 139.16 | 37.94 | no stable speed win |
| RWKV-7 1.5B mm4 | 512 / 64 | 43.62 | 22.54 | no stable speed win |
| RWKV-7 1.5B mm4 warm1 | 512 / 64 | 43.81 | 22.60 | no stable speed win |

## Related rejected fast path

`mx.fast.layer_norm` is available in MLX 0.31.2 and was tested as an opt-in
`RWKV7_MLX_FAST_LAYER_NORM=1` path.  On the current RWKV 1.5B mm4 row it
regressed throughput, so the option remains default-off.
