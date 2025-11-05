# Apple M5 Qwen3.5 comparison refresh with RWKV scan-prefill auto (2026-07-08)

This directory records a Qwen3.5 comparison refresh after wiring RWKV MLX
scan-prefill and the `RWKV_WKV_SCAN_PREFILL=auto` policy.

Live Ollama pull for `qwen3.5:0.8b-mlx` was attempted on 2026-07-08 but became
blocked at `pulling manifest` with an empty local Ollama model list.  The
comparison below therefore combines existing same-device Qwen rows collected on
2026-07-07 with the new RWKV scan-auto rows from
`bench/apple_scan_prefill_auto_m5_20260708/`.

## Live pull status

File: `results_pull.jsonl`

- `qwen3.5:0.8b-mlx` HTTP helper reached manifest/model metadata but timed out
  waiting for byte/status progress.
- `ollama pull qwen3.5:0.8b-mlx` stayed at `pulling manifest` for about two
  minutes and was stopped.

## Comparison rows

### Qwen3.5 0.8B vs RWKV-7 0.4B, 4096 chars / 128 decode

Files:

- `combined_08b_4096_128.jsonl`
- `results_compare_08b_4096_128.jsonl`

| Metric | Qwen3.5 0.8B MLX 4bit | RWKV-7 0.4B mm4 scan-auto | RWKV / Qwen |
|---|---:|---:|---:|
| Prefill tok/s | 1902.37 | 247.42 | 0.130 |
| Decode tok/s | 110.86 | 60.14 | 0.542 |
| TTFT s | 0.525 | 5.295 | 10.08x slower |
| Peak memory | 2.27 GB | 1.24 GB | 0.546 |

Status: speed/TTFT fail, memory pass.

### Qwen3.5 2B vs RWKV-7 1.5B, 512 chars / 64 decode

Files:

- `combined_2b_512_64.jsonl`
- `results_compare_2b_512_64.jsonl`

| Metric | Qwen3.5 2B MLX 4bit | RWKV-7 1.5B mm4 scan | RWKV / Qwen |
|---|---:|---:|---:|
| Prefill tok/s | 1205.58 | 65.20 | 0.054 |
| Decode tok/s | 110.63 | 29.25 | 0.264 |
| TTFT s | 0.105 | 2.516 | 23.88x slower |
| Peak memory | 2.19 GB | 1.27 GB | 0.578 |

Status: speed/TTFT fail, memory pass.

## Interpretation

The scan-prefill work materially improves RWKV prefill versus previous RWKV
rows, especially for the 0.4B 4096/128 comparison, but it is not enough to beat
Qwen3.5's highly optimized MLX attention stack.  The next performance priorities
are now clear:

1. Decode path fusion/batching, because decode still runs single-token kernels.
2. Projection + scan deeper fusion, because prefill remains far below Qwen on
   both 0.4B and 1.5B tiers.
3. TTFT reduction through less model load/prefill graph overhead and better
   chunk scheduling.
