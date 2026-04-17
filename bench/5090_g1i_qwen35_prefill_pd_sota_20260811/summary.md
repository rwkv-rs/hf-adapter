# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `24/24` cells.

Required Qwen backend: `fla`; verified: `24/24` cells.

Required Qwen full fusion: `true`; verified: `24/24` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.348x | 1.819x | 6.953x | 24/24 |
| Decode RWKV/Qwen | 2.711x | 6.105x | 19.891x | 24/24 |
| Model footprint RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Peak VRAM RWKV/Qwen | 0.591x | 0.792x | 1.073x | 23/24 |
| Runtime working set RWKV/Qwen | 0.556x | 1.543x | 8.019x | 6/24 |
| Active parameters RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Prefill tok/s per active-B | 1.676x | 2.632x | 11.606x | 24/24 |
| Decode tok/s per active-B | 3.372x | 8.244x | 33.201x | 24/24 |
| Prefill active-param work rate | 1.073x | 1.318x | - | 24/24 |
| Decode active-param work rate | 2.180x | 4.331x | - | 24/24 |

Strict speed cells: `24/24`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 24 | 1.348x / 1.819x | 2.711x / 6.105x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
