# Apple MLX RWKV-7 component profile — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This evidence set profiles the RWKV-7 1.5B MLX path with synchronized component
boundaries.  It is **not** an end-to-end speed gate: inserting `mx.eval` around
components intentionally perturbs MLX scheduling.  The purpose is to rank the
components that should be fused next after the Qwen3.5 2B evidence showed that
memory passes but prefill/decode speed still fails.

## Command

```bash
RWKV7_MLX_STEP_EVAL_INTERVAL=2 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/profile_mlx_components.py \
  --model-dir /Users/wangyue/Documents/vllmsp/models/rwkv7-g1g-1.5b-hf \
  --prompt-target-chars 512 \
  --decode-length 16 \
  --warmup-repeats 1 \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 8000000 \
  --rwkv-quant-rkv-min-params 0 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto
```

The prompt seed matches `bench/run_qwen35_apple_baseline.py`, so the row uses the
same 512-character prompt text family as the Qwen3.5 2B baseline.

## Result

| Component | Calls | Total time | Avg / call | Share of profiled time |
|---|---:|---:|---:|---:|
| FFN step | 3576 | 7.116802s | 1.990157ms | 39.4555% |
| Attention/WKV step | 3576 | 5.979686s | 1.672172ms | 33.1513% |
| Attention layernorm | 3576 | 3.343930s | 0.935104ms | 18.5387% |
| FFN layernorm | 3576 | 1.231262s | 0.344313ms | 6.8261% |
| First-layer pre-norm | 149 | 0.228608s | 1.534280ms | 1.2674% |
| Final norm + lm_head | 17 | 0.078810s | 4.635865ms | 0.4369% |
| Embedding | 149 | 0.054293s | 0.364380ms | 0.3010% |
| Final layernorm | 17 | 0.004160s | 0.244686ms | 0.0231% |

Kernel evidence from the same row:

- WKV backend counts: `{"reference": 0, "metal": 3576}`
- Quantized linear backend counts: `{"reference": 0, "affine": 0, "metal": 17897}`
- Grouped R/K/V quant counts: `{"metal": 3576, "fallback": 0}`

Interpretation:

- The next Apple MLX speed work should prioritize FFN step fusion and attention
  step fusion, not final `lm_head`.
- Layernorm is a material launch/synchronization contributor under synchronized
  profiling, especially `attn_norm`; fusing norm with the following mix/projection
  is likely more valuable than optimizing final logits first.
- WKV and grouped R/K/V quant are already on Metal for this row, so the remaining
  gap is mostly around the surrounding per-token layer structure and FFN path.
