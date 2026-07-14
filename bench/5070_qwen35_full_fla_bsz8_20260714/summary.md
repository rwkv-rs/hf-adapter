# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.083x | 1.375x | 1.689x | 18/18 |
| Decode RWKV/Qwen | 1.795x | 2.545x | 3.457x | 18/18 |
| Model footprint RWKV/Qwen | 0.729x | 0.812x | 0.857x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.606x | 0.845x | 0.956x | 18/18 |
| Runtime working set RWKV/Qwen | 0.547x | 1.019x | 1.915x | 8/18 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 1.334x | 1.694x | 2.081x | 18/18 |
| Decode tok/s per active-B | 2.212x | 3.136x | 4.259x | 18/18 |
| Prefill active-param work rate | 0.879x | 1.116x | - | reported 18/18 |
| Decode active-param work rate | 1.457x | 2.066x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.107x / 1.332x | 2.490x / 2.624x | - | - | - | - |
| w4 | 6 | 1.083x / 1.331x | 2.508x / 2.858x | - | - | - | - |
| w8 | 6 | 1.286x / 1.590x | 1.795x / 1.871x | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
