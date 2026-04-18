# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `24/24` cells.

Required Qwen backend: `fla`; verified: `24/24` cells.

Required Qwen full fusion: `true`; verified: `24/24` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.532x | 2.076x | 9.061x | 24/24 |
| Decode RWKV/Qwen | 2.070x | 4.525x | 11.183x | 24/24 |
| Model footprint RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Peak VRAM RWKV/Qwen | 0.533x | 0.754x | 1.040x | 21/24 |
| Runtime working set RWKV/Qwen | 0.191x | 0.936x | 21.947x | 13/24 |
| Active parameters RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Prefill tok/s per active-B | 1.905x | 2.962x | 15.124x | 24/24 |
| Decode tok/s per active-B | 2.574x | 6.096x | 18.667x | 24/24 |
| Prefill active-param work rate | 1.227x | 1.468x | - | 24/24 |
| Decode active-param work rate | 1.664x | 3.434x | - | 24/24 |

Strict speed cells: `24/24`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 24 | 1.532x / 2.076x | 2.070x / 4.525x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
