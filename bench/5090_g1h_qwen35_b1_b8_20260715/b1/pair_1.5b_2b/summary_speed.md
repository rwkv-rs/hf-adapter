# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.811x | 4.285x | 8.933x | 18/18 |
| Decode RWKV/Qwen | 6.687x | 7.356x | 9.193x | 18/18 |
| Model footprint RWKV/Qwen | 0.330x | 0.455x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.387x | 0.522x | 0.849x | 18/18 |
| Runtime working set RWKV/Qwen | 1.890x | 2.517x | 8.755x | 0/18 |
| Active parameters RWKV/Qwen | 0.740x | 0.812x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 3.464x | 5.788x | 12.066x | 18/18 |
| Decode tok/s per active-B | 8.238x | 9.936x | 11.326x | 18/18 |
| Prefill active-param work rate | 2.084x | 3.172x | - | reported 18/18 |
| Decode active-param work rate | 5.112x | 5.470x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.811x / 3.258x | 6.687x / 6.737x | - | - | - | - | - |
| w4 | 6 | 2.815x / 3.606x | 6.904x / 8.614x | 0.792x | 1.025x | 1.024x | 0.934x | 0.938x |
| w8 | 6 | 4.252x / 5.704x | 7.332x / 7.386x | 1.141x | 1.096x | 1.096x | 0.561x | 0.641x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
