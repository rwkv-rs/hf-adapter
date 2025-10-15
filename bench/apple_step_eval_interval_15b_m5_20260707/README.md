# Apple MLX step-eval interval sweep — 1.5B/mm4 fused FFN, Apple M5, 2026-07-07

This sweep checks `RWKV7_MLX_STEP_EVAL_INTERVAL` after enabling the MM4 FFN key
`relu²` fused Metal seam (`RWKV7_MLX_FUSED_FFN_KEY_RELU2=1`).  The model default
remains correctness-first; this evidence only tunes the Apple acceptance wrapper
policy.

Common command shape:

```bash
RWKV7_MLX_STEP_EVAL_INTERVAL={4|8|16} \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2=1 \
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
  --repeat 1 \
  --warmup-repeats 1 \
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

Observed measured rows:

| interval | prefill tok/s | decode tok/s | TTFT s | peak bytes | correctness |
|---:|---:|---:|---:|---:|---|
| 4 | 28.945080 | 23.222910 | 4.595174 | 1239118508 | preview identical, chunked diff 0 |
| 8 | 29.719766 | 22.600976 | 4.475457 | 1240144500 | preview identical, chunked diff 0 |
| 16 | 29.171174 | 22.175616 | 4.559621 | 1244588692 | preview identical, chunked diff 0 |

Compared with the prior eval2 fused-FFN smoke (`28.830945` prefill tok/s,
`22.103406` decode tok/s, `4.613353` TTFT), interval 8 gives the best
prefill/TTFT and still improves decode.  The Apple acceptance wrapper therefore
uses interval 8 by default, while callers can override `RWKV7_MLX_STEP_EVAL_INTERVAL`.
