# Apple M5 fast layer norm experiment (2026-07-08)

MLX exposes `mx.fast.layer_norm`, so RWKV MLX now has an opt-in
`RWKV7_MLX_FAST_LAYER_NORM=1` path.  Tiny-model correctness matched the existing
formula, but the real 1.5B mm4 row regressed on Apple M5.

File:

- `results_rwkv15_512_64_fast_ln.jsonl`

Observed row:

| Model | Shape | Prefill tok/s | Decode tok/s | Result |
|---|---|---:|---:|---|
| RWKV-7 1.5B mm4 | 512 / 64 | 38.84 | 19.38 | regression vs previous scan rows |

Conclusion: keep `RWKV7_MLX_FAST_LAYER_NORM` default-off.  Do not treat this as
a production optimization until future MLX versions or a fused RWKV-specific
norm kernel beat the manual formula.
