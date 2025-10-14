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

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 2B MLX-4bit | mlx-vlm token-only | 127 | 64 | 0.378899s | 335.181414 | 37.070964 | 2,020,245,484 B | baseline |
| RWKV-7 1.5B mm4 + RKV quant | rwkv7_hf MLX | 133 | 64 | 11.019150s | 12.083573 | 8.979128 | 1,225,111,246 B | memory pass, speed fail |

Comparison gates on this expanded token-only smoke:

- `decode_ratio_rwkv_over_qwen=0.242215`
- `prefill_ratio_rwkv_over_qwen=0.036051`
- `ttft_ratio_rwkv_over_qwen=29.082024`
- `memory_ratio_rwkv_over_qwen=0.606417`

Interpretation:

- The 1.5B RWKV MLX path is runnable with W4/mm4 and RKV quantization on this
  16GB Apple device.
- Memory is the one passing gate on this row: RWKV peak is about 60.6% of the
  Qwen3.5 2B MLX-4bit peak.
- Speed is still the blocker: RWKV needs about `4.13x` decode speedup and
  `27.74x` prefill speedup on this row to match the measured Qwen3.5 2B token
  baseline.
- The next optimization lane remains fused recurrent/prefill work first, then
  deeper fused quant projection and TTFT reduction.
