# Apple Qwen3.5 2B MLX-VLM token-only vs RWKV-7 1.5B MLX — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This evidence set extends the Apple/Qwen3.5 matrix from the 0.8B/0.4B row to a
larger Qwen3.5 2B MLX-4bit baseline against RWKV-7 1.5B with MLX `mm4` weight
quantization, RKV projection quantization enabled, and `RWKV7_MLX_STEP_EVAL_INTERVAL=2`.
It is a speed-gap measurement, not a pass claim.

The Qwen3.5 2B snapshot initially stalled in `huggingface_hub.snapshot_download`
after the small metadata/tokenizer files.  The large `model.safetensors` file
was downloaded with resumable parallel HTTP Range requests, which is now captured
by `scripts/hf_parallel_download.py` for repeatable large-file fetches.  The
`results_hf_parallel_download.jsonl` file records the post-download
`already_complete` verification row for the 1,722,271,785-byte shard.

## Commands

Download Qwen3.5 2B MLX-4bit large shard when the normal snapshot path stalls:

```bash
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 \
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  scripts/hf_parallel_download.py \
  --repo-id mlx-community/Qwen3.5-2B-MLX-4bit \
  --filename model.safetensors \
  --output /Users/wangyue/Documents/vllmsp/models/qwen35-2b-mlx-4bit/model.safetensors \
  --jobs 12 \
  --chunk-mib 32
```

Qwen3.5 2B MLX-VLM token-only row:

```bash
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
  --repeat 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models /Users/wangyue/Documents/vllmsp/models/qwen35-2b-mlx-4bit \
  --qwen-mlx-vlm-token-only \
  --rwkv-mlx-models '' \
  --temperature 0.0
```

RWKV-7 1.5B MLX row:

```bash
RWKV7_MLX_STEP_EVAL_INTERVAL=2 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
  --repeat 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 8000000 \
  --rwkv-quant-rkv-min-params 0 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 256
```

## Result

### Cold first measured row

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 2B MLX-4bit | mlx-vlm token-only | 127 | 64 | 0.378899s | 335.181414 | 37.070964 | 2,020,245,484 B | baseline |
| RWKV-7 1.5B mm4 + RKV quant | rwkv7_hf MLX | 133 | 64 | 11.019150s | 12.083573 | 8.979128 | 1,225,111,246 B | memory pass, speed fail |

Cold-row comparison gates:

- `decode_ratio_rwkv_over_qwen=0.242215`
- `prefill_ratio_rwkv_over_qwen=0.036051`
- `ttft_ratio_rwkv_over_qwen=29.082024`
- `memory_ratio_rwkv_over_qwen=0.606417`

### Warmed steady-state row (`--warmup-repeats 1`)

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Kernel evidence | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3.5 2B MLX-4bit | mlx-vlm token-only | 127 | 64 | 0.105343s | 1205.584132 | 110.627117 | 2,193,064,509 B | mlx-vlm token-only | baseline |
| RWKV-7 1.5B mm4 + RKV quant | rwkv7_hf MLX | 133 | 64 | 3.105107s | 42.835321 | 31.695076 | 1,238,026,466 B | WKV Metal 4728, RKV Metal 4728/fallback 0 | memory pass, speed fail |

Warm-row comparison gates:

- `decode_ratio_rwkv_over_qwen=0.286504`
- `prefill_ratio_rwkv_over_qwen=0.035531`
- `ttft_ratio_rwkv_over_qwen=29.476159`
- `memory_ratio_rwkv_over_qwen=0.564519`

Interpretation:

- The 1.5B RWKV MLX path is runnable with W4/mm4 and RKV quantization on this
  16GB Apple device.
- Memory is the one passing gate: the cold row is about 60.6% of Qwen peak; the
  warmed row is about 57.0% of Qwen peak.
- Warmup matters for MLX/Metal rows.  The RWKV warmed row proves WKV and grouped
  R/K/V quant projection are actually on Metal, but Qwen also speeds up after
  warmup, so this does not close the speed gate.
- Main generation telemetry is separated from `chunked_prefill` correctness
  telemetry: the warmed row records main WKV/RKV Metal counts of 4728 and
  chunked-prefill WKV/RKV Metal counts of 3192 with max diff `0.0`.
- On the warmed row, RWKV still needs about `3.49x` decode speedup and `28.14x`
  prefill speedup to match the measured Qwen3.5 2B token baseline.
- The next optimization lane remains fused recurrent/prefill work first, then
  deeper fused quant projection and TTFT reduction.
