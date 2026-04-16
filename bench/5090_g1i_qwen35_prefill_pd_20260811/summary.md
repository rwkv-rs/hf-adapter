# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `24/24` cells.

Required Qwen backend: `fla`; verified: `24/24` cells.

Required Qwen full fusion: `true`; verified: `24/24` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.320x | 1.760x | 6.827x | 24/24 |
| Decode RWKV/Qwen | 2.713x | 6.101x | 19.837x | 24/24 |
| Model footprint RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Peak VRAM RWKV/Qwen | 0.611x | 0.831x | 1.070x | 18/24 |
| Runtime working set RWKV/Qwen | 0.637x | 2.482x | 41.114x | 5/24 |
| Active parameters RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Prefill tok/s per active-B | 1.642x | 2.512x | 11.394x | 24/24 |
| Decode tok/s per active-B | 3.375x | 8.240x | 33.110x | 24/24 |
| Prefill active-param work rate | 1.028x | 1.269x | - | 24/24 |
| Decode active-param work rate | 2.182x | 4.328x | - | 24/24 |

Strict speed cells: `24/24`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 24 | 1.320x / 1.760x | 2.713x / 6.101x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
