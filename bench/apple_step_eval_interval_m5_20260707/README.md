# Apple MLX step eval interval smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This evidence set tests `RWKV7_MLX_STEP_EVAL_INTERVAL`, a runtime knob that
reduces per-token `mx.eval` synchronization in the recurrent MLX loop while the
model default remains `1` for historical behavior.

All rows use RWKV-7 0.4B/mm4 with R/K/V quant enabled
(`--rwkv-quant-min-params 4000000 --rwkv-quant-rkv-min-params 0`) and grouped
R/K/V Metal projection active (`fallback=0`).

| Interval | Prefill tok/s | Decode tok/s | TTFT | Peak memory | Chunked/full max_abs |
|---:|---:|---:|---:|---:|---:|
| 1 baseline from previous evidence | 69.061072 | 50.514476 | 1.926324s | 402156430 B | 0.0 |
| 2 | 76.905622 | 58.361776 | 1.730042s | 402251086 B | 0.0 |
| 4 | 74.066204 | 58.402642 | 1.795990s | 402552924 B | 0.0 |
| 8 | 77.103414 | 58.148376 | 1.725628s | 403124788 B | 0.0 |
| 16 | 74.428961 | 57.318431 | 1.787776s | 405348880 B | 0.0 |

Interval `2` is the best local M5 point in this smoke and is now the default for
the Qwen3.5 Apple acceptance wrapper.  The model default remains `1`; set
`RWKV7_MLX_STEP_EVAL_INTERVAL=1` to recover the historical synchronization
policy.

Against the same Qwen3.5 0.8B MLX-4bit token-only 512 / 64 row, interval `2`
records `decode_ratio_rwkv_over_qwen=0.264835`,
`prefill_ratio_rwkv_over_qwen=0.050648`, `ttft_ratio_rwkv_over_qwen=20.684633`,
and `memory_ratio_rwkv_over_qwen=0.450406`.  This is a clear incremental speed
movement, but the production Qwen speed gap remains open.
