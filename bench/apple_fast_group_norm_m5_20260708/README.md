# Apple M5 fast group norm experiment (2026-07-08)

This batch tests an opt-in `RWKV7_MLX_FAST_GROUP_NORM=1` path for the attention
per-head group norm.  The implementation calls `mx.fast.layer_norm` over each
head dimension without fused scale/bias, then applies the RWKV per-head
`g_norm.weight` and `g_norm.bias` manually.  The path is correctness-tested on a
tiny MLX model but remains default-off until more devices/models show stable
wins.

## Rows

| Model | Shape | Prefill tok/s | Decode tok/s | Interpretation |
|---|---|---:|---:|---|
| RWKV-7 0.4B mm4 | 512 / 64 | 222.24 | 57.07 | positive vs recent default row; near/better than previous best prefill |
| RWKV-7 1.5B mm4 | 512 / 64 | 52.01 | 24.92 | better than direct-decode experiment row, below previous best scan row |
| RWKV-7 1.5B mm4 warm1 | 512 / 64 | 51.09 | 23.70 | mixed/no stable win |

Files:

- `results_rwkv04_512_64_fast_gn.jsonl`
- `results_rwkv15_512_64_fast_gn.jsonl`
- `results_rwkv15_512_64_fast_gn_warm1.jsonl`

## Decision

Keep `RWKV7_MLX_FAST_GROUP_NORM` default-off.  It is a valid optional seam and
may help smaller models or future MLX versions, but the current 1.5B Apple M5
rows do not justify enabling it by default.
