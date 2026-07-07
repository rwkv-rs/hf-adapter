# Apple MLX fused FFN key relu² smoke — Apple M5, 2026-07-07

This evidence compares the existing RWKV-7 1.5B/mm4 Apple MLX path against an
opt-in FFN seam that fuses the quantized MM4 key projection with `relu²` in one
Metal kernel:

- baseline: `RWKV7_MLX_FUSED_FFN_KEY_RELU2=0`
- fused: `RWKV7_MLX_FUSED_FFN_KEY_RELU2=1`

Common command shape:

```bash
RWKV7_MLX_STEP_EVAL_INTERVAL=2 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2={0|1} \
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

Observed row pair:

| path | prefill tok/s | decode tok/s | TTFT s | peak bytes | generated preview |
|---|---:|---:|---:|---:|---|
| base | 27.464524 | 21.548159 | 4.843214 | 1238065228 | identical |
| fused | 28.830945 | 22.103406 | 4.613353 | 1238015180 | identical |

Relative fused/base:

- prefill: `1.04975x`
- decode: `1.02577x`
- TTFT reduction: `1.04983x` as base/fused

The fused row records `fused_ffn_key_relu2_counts={"metal":4728,"fallback":0}`
and chunked prefill remains exact (`chunked_prefill_max_abs=0.0`).  This is not
sufficient to close the Qwen3.5 2B gap, but it is an actual positive MLX fused
kernel seam on the current highest-priority FFN bucket.
