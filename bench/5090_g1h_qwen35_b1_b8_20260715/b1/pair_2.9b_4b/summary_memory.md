# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.309x | 3.583x | 6.262x | 18/18 |
| Decode RWKV/Qwen | 5.006x | 5.930x | 7.150x | 18/18 |
| Model footprint RWKV/Qwen | 0.252x | 0.382x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.294x | 0.409x | 0.717x | 18/18 |
| Runtime working set RWKV/Qwen | 1.063x | 1.715x | 5.191x | 0/18 |
| Active parameters RWKV/Qwen | 0.661x | 0.701x | 0.701x | 18/18 |
| Prefill tok/s per active-B | 3.294x | 5.421x | 9.473x | 18/18 |
| Decode tok/s per active-B | 7.142x | 8.971x | 10.201x | 18/18 |
| Prefill active-param work rate | 1.618x | 2.368x | - | reported 18/18 |
| Decode active-param work rate | 3.508x | 3.920x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.368x / 2.715x | 5.006x / 5.236x | - | - | - | - | - |
| w4 | 6 | 2.309x / 2.838x | 6.937x / 7.048x | - | - | - | 0.360x | 0.431x |
| w8 | 6 | 3.582x / 4.577x | 5.826x / 5.930x | - | - | - | 0.545x | 0.586x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
