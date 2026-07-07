# Apple M5 MLX multi-token WKV scan prototype

This evidence directory records the first Apple-side **big fused WKV** seam.

Previous MLX work already had a single-token Metal WKV update.  This change adds
`rwkv7_hf.mlx_scan.wkv_scan()`, a multi-token recurrent scan kernel that updates
RWKV state across an entire sequence for one layer after projections have
produced `r/w/v/k/kk/a [B,T,H,N]`.

This is not wired into full model prefill yet.  It is the core kernel needed for
the next architecture step: convert MLX prefill from token-major execution to
layer-major execution, then call this scan once per layer/chunk instead of one
WKV Metal kernel per token/layer.

## Device and shape

- Device: Mac17,3 / Apple M5 / 16GB unified memory
- OS: macOS 26.5 arm64
- MLX: local Apple environment
- Batch: 1
- Heads: 16
- Head dim: 64
- Bench target: isolated recurrent WKV update only

## Commands

```bash
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  scripts/mlx_wkv_scan_bench.py \
  --tokens 128 \
  --heads 16 \
  --head-dim 64 \
  --warmup 1 \
  --runs 3 \
  --results bench/apple_mlx_wkv_scan_m5_20260707/results_scan_bench_128.jsonl

PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  scripts/mlx_wkv_scan_bench.py \
  --tokens 32 \
  --heads 16 \
  --head-dim 64 \
  --warmup 1 \
  --runs 3 \
  --results bench/apple_mlx_wkv_scan_m5_20260707/results_scan_bench_32.jsonl
```

## Results

| Tokens | Per-token Metal loop | Multi-token scan | Speedup | Loop tok/s | Scan tok/s |
|---:|---:|---:|---:|---:|---:|
| 32 | `0.005828s` | `0.002338s` | `2.493x` | `5490.9` | `13689.5` |
| 128 | `0.016071s` | `0.003925s` | `4.095x` | `7964.9` | `32613.0` |

Correctness against the existing per-token Metal loop:

| Tokens | `max_abs_out_vs_loop` | `max_abs_state_vs_loop` |
|---:|---:|---:|
| 32 | `0.0625` | `0.0029335` |
| 128 | `0.0625` | `0.00344086` |

The standalone unit test `tests/test_mlx_scan.py` also compares the Metal scan
against the portable MLX reference on a small exactness fixture and passes with a
tight tolerance.  The larger random microbench shows fp16 output accumulation
order drift, so this remains a prototype until full model prefill parity gates
are added.

## Engineering implication

This is the first kernel with the right shape to attack the 20x-30x prefill gap:
sequence scan in one launch instead of per-token WKV launches.  Remaining work:

1. Build a layer-major MLX prefill path that computes per-layer projections over
   a chunk and calls `wkv_scan()`.
2. Add full-model parity against the existing token-major MLX path.
3. Add Qwen3.5 Apple baseline rows showing end-to-end prefill/TTFT improvement.
4. Tune fp32/fp16 output policy if full-model parity needs stricter logits.
