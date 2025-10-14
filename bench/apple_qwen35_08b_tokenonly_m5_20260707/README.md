# Apple Qwen3.5 0.8B MLX-VLM token-only vs RWKV 0.4B expanded smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This evidence set extends the earlier short `128 chars / 4 tokens` smoke to a
larger `512 chars / 64 tokens` row and uses the MLX-VLM token-only Qwen lane.
The token-only lane avoids `mlx-vlm` text detokenizer `UnicodeDecodeError`
failures while still measuring generated-token latency, throughput, and MLX
peak memory.

## Command shape

Qwen3.5 0.8B MLX-4bit token-only row:

```bash
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
  --repeat 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models /Users/wangyue/Documents/vllmsp/models/qwen35-0.8b-mlx-4bit \
  --qwen-mlx-vlm-token-only \
  --rwkv-mlx-models '' \
  --temperature 0.0
```

RWKV-7 0.4B/mm4 group-quant row:

```bash
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
  --repeat 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 256
```

## Result

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 0.8B MLX-4bit | mlx-vlm token-only | 127 | 64 | 0.083639s | 1518.438715 | 220.370403 | 893086209 B | baseline |
| RWKV-7 0.4B mm4 + group quant projection | rwkv7_hf MLX | 133 | 64 | 2.195128s | 60.608566 | 56.477395 | 514793582 B | memory pass, speed fail |

Comparison gates on this expanded token-only smoke:

- `decode_ratio_rwkv_over_qwen=0.256284`
- `prefill_ratio_rwkv_over_qwen=0.039915`
- `ttft_ratio_rwkv_over_qwen=26.245268`
- `memory_ratio_rwkv_over_qwen=0.576421`

So the current RWKV path still wins memory on this row, but the stronger Qwen
MLX token-only baseline exposes the real next bottlenecks: decode kernel /
batching, prefill/chunked prefill, and TTFT reduction.  Treat the earlier short
pass as a smoke-scale pass only, not a production-speed claim.
