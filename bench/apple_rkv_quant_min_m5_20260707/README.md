# Apple MLX R/K/V quant-min activation smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This evidence set verifies the new separate R/K/V quantization threshold used by
Apple MLX grouped quant projection work:

- general quant threshold remains high (`--rwkv-quant-min-params 4000000`), so
  FFN/lm_head quantization policy stays unchanged;
- `--rwkv-quant-rkv-min-params 0` additionally quantizes attention
  `r_proj`/`k_proj`/`v_proj`;
- `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1` then hits the one-launch MLX/Metal
  grouped R/K/V path with `fallback=0`.

## Key rows

| Row | Mode | Prompt / decode | Prefill tok/s | Decode tok/s | Peak memory | Grouped RKV counts |
|---|---|---:|---:|---:|---:|---|
| 0.4B mm4 | direct | 128 / 4 | 67.020907 | 63.648196 | 402162760 B | `metal=1680`, `fallback=0` |
| 0.4B mm4 | direct | 512 / 64 | 69.061072 | 50.514476 | 402156430 B | `metal=7920`, `fallback=0` |
| 0.4B mm4 | packed | 512 / 64 | 61.261574 | 56.617763 | 440517924 B | `metal=7920`, `fallback=0` |

The previous 512 / 64 row with the same general threshold but without the
separate R/K/V threshold had `group_rkv_quant_projection_counts={"metal":0,
"fallback":7920}` and peak≈514.8MB.  The direct R/K/V quant row lowers peak to
≈402.2MB and improves prefill/TTFT, but decode is slower than the no-RKV row;
packed mode recovers decode while using more memory.  This confirms the grouped
quant projection path is now actually exercised by the acceptance harness, while
showing that the next production work remains deeper decode fusion rather than
claiming a Qwen speed win.

## Qwen3.5 0.8B token-only comparison

Using the earlier Qwen3.5 0.8B MLX-4bit token-only row for the same 512 / 64
shape, the direct R/K/V quant RWKV row records:

- `decode_ratio_rwkv_over_qwen=0.229225`
- `prefill_ratio_rwkv_over_qwen=0.045482`
- `ttft_ratio_rwkv_over_qwen=23.031409`
- `memory_ratio_rwkv_over_qwen=0.450300`

So memory moves further ahead, but speed/latency remain the open gap.
