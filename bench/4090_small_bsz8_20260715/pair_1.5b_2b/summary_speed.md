# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `false`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 0.973x | 1.377x | 2.585x | 17/18 |
| Decode RWKV/Qwen | 5.637x | 5.880x | 6.408x | 18/18 |
| Model footprint RWKV/Qwen | 0.455x | 0.758x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.552x | 1.105x | 1.208x | 6/18 |
| Runtime working set RWKV/Qwen | 1.430x | 2.541x | 5.721x | 0/18 |
| Active parameters RWKV/Qwen | 0.740x | 0.740x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 1.284x | 1.860x | 3.492x | 18/18 |
| Decode tok/s per active-B | 6.945x | 7.934x | 8.655x | 18/18 |
| Prefill active-param work rate | 0.721x | 1.019x | - | reported 18/18 |
| Decode active-param work rate | 4.346x | 4.595x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.042x / 1.115x | 5.637x / 5.658x | - | - | - | - | - |
| w4 | 6 | 0.973x / 1.117x | 5.870x / 5.880x | 0.931x | 1.038x | 1.027x | 0.935x | 0.969x |
| w8 | 6 | 1.375x / 1.610x | 6.375x / 6.393x | 1.294x | 1.129x | 1.132x | 0.561x | 0.625x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
