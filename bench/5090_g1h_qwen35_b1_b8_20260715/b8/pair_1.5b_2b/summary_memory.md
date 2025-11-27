# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.017x | 1.455x | 3.895x | 18/18 |
| Decode RWKV/Qwen | 4.559x | 4.692x | 6.657x | 18/18 |
| Model footprint RWKV/Qwen | 0.491x | 0.759x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.594x | 0.902x | 1.080x | 12/18 |
| Runtime working set RWKV/Qwen | 1.257x | 1.916x | 2.775x | 0/18 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 1.253x | 1.793x | 4.799x | 18/18 |
| Decode tok/s per active-B | 5.617x | 5.781x | 8.201x | 18/18 |
| Prefill active-param work rate | 0.825x | 1.181x | - | reported 18/18 |
| Decode active-param work rate | 3.700x | 3.809x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.023x / 1.190x | 4.559x / 4.594x | - | - | - | - | - |
| w4 | 6 | 1.017x / 1.184x | 4.648x / 4.690x | - | - | - | 0.935x | 0.959x |
| w8 | 6 | 1.455x / 1.719x | 5.266x / 5.310x | - | - | - | 0.605x | 0.693x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
