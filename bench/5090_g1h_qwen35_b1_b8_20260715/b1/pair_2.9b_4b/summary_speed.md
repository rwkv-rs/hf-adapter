# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.329x | 3.583x | 7.689x | 18/18 |
| Decode RWKV/Qwen | 5.210x | 5.935x | 7.153x | 18/18 |
| Model footprint RWKV/Qwen | 0.252x | 0.382x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.294x | 0.409x | 0.717x | 18/18 |
| Runtime working set RWKV/Qwen | 0.747x | 1.667x | 5.191x | 2/18 |
| Active parameters RWKV/Qwen | 0.661x | 0.701x | 0.701x | 18/18 |
| Prefill tok/s per active-B | 3.322x | 5.421x | 11.633x | 18/18 |
| Decode tok/s per active-B | 7.433x | 8.979x | 10.206x | 18/18 |
| Prefill active-param work rate | 1.632x | 2.368x | - | reported 18/18 |
| Decode active-param work rate | 3.652x | 3.923x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.389x / 2.716x | 5.210x / 5.243x | - | - | - | - | - |
| w4 | 6 | 2.329x / 2.831x | 6.999x / 7.066x | 0.816x | 1.343x | 1.307x | 0.360x | 0.431x |
| w8 | 6 | 3.572x / 4.570x | 5.877x / 5.935x | 1.193x | 1.128x | 1.130x | 0.545x | 0.586x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
