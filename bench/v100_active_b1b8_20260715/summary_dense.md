# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `2/2` cells.

Required Qwen backend: `fla`; verified: `2/2` cells.

Required Qwen full fusion: `true`; verified: `2/2` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.816x | 4.112x | 5.408x | 2/2 |
| Decode RWKV/Qwen | 5.270x | 5.592x | 5.913x | 2/2 |
| Model footprint RWKV/Qwen | 0.812x | 0.812x | 0.812x | 2/2 |
| Peak VRAM RWKV/Qwen | 0.837x | 0.931x | 1.025x | 1/2 |
| Runtime working set RWKV/Qwen | 0.968x | 4.995x | 9.022x | 1/2 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 2/2 |
| Prefill tok/s per active-B | 3.469x | 5.066x | 6.663x | 2/2 |
| Decode tok/s per active-B | 6.493x | 6.889x | 7.285x | 2/2 |
| Prefill active-param work rate | 2.286x | 3.337x | - | 2/2 |
| Decode active-param work rate | 4.278x | 4.539x | - | 2/2 |

Strict speed cells: `2/2`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 2 | 2.816x / 4.112x | 5.270x / 5.592x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
