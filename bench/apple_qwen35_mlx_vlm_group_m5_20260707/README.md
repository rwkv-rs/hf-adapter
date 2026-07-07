# Apple Qwen3.5 MLX-VLM vs RWKV group-quant pass smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This is a follow-up to `bench/apple_qwen35_mlx_vlm_m5_20260707/`.  The first
same-prompt smoke showed RWKV-7 0.4B/mm4 already winning TTFT, prefill, and
memory but missing the decode gate.  This run enables the existing
`RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1` path, which is also the default in the
Apple/Qwen3.5 acceptance wrapper.

## Command shape

```bash
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 128 \
  --decode-lengths 4 \
  --qwen-models '' \
  --qwen-mlx-vlm-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 64 \
  --store-responses
```

## Result

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 0.8B MLX-4bit | mlx-vlm | 32 | 4 | 1.875480s | 17.074117 | 62.404281 | 766525058 B | baseline |
| RWKV-7 0.4B mm4 + group quant projection | rwkv7_hf MLX | 33 | 4 | 0.496328s | 66.583779 | 65.663789 | 514796662 B | pass |

Comparison gates pass on this short smoke:

- `decode_ratio_rwkv_over_qwen=1.052234`
- `prefill_ratio_rwkv_over_qwen=3.899688`
- `ttft_ratio_rwkv_over_qwen=0.264636`
- `memory_ratio_rwkv_over_qwen=0.671599`

This is still only a smoke-scale pass.  The next acceptance step is to repeat the
same Qwen/RWKV pair on longer prompt/decode lengths and larger model tiers.
