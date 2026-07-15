# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `72/72` cells.

| Metric | Minimum | Median |
|---|---:|---:|
| Prefill RWKV/Qwen | 1.001x | 1.089x |
| Decode RWKV/Qwen | 1.892x | 3.475x |

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 24 | 1.050x / 1.080x | 1.892x / 1.960x | - | - | - | - |
| w4 | 24 | 1.081x / 1.179x | 2.908x / 3.475x | 1.000x | 1.010x | 0.973x | 0.990x |
| w8 | 24 | 1.001x / 1.014x | 4.615x / 5.100x | 1.001x | 1.016x | 0.981x | 0.985x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
