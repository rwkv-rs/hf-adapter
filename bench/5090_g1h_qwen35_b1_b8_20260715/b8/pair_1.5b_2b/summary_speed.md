# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.014x | 1.429x | 3.025x | 18/18 |
| Decode RWKV/Qwen | 4.475x | 4.579x | 5.203x | 18/18 |
| Model footprint RWKV/Qwen | 0.491x | 0.759x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.594x | 0.902x | 1.080x | 12/18 |
| Runtime working set RWKV/Qwen | 1.257x | 1.916x | 2.775x | 0/18 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 1.250x | 1.760x | 3.727x | 18/18 |
| Decode tok/s per active-B | 5.514x | 5.642x | 6.411x | 18/18 |
| Prefill active-param work rate | 0.823x | 1.160x | - | reported 18/18 |
| Decode active-param work rate | 3.632x | 3.717x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.019x / 1.190x | 4.475x / 4.486x | - | - | - | - | - |
| w4 | 6 | 1.014x / 1.183x | 4.566x / 4.579x | 0.961x | 1.020x | 1.012x | 0.935x | 0.959x |
| w8 | 6 | 1.427x / 1.716x | 5.162x / 5.175x | 1.398x | 1.153x | 1.155x | 0.605x | 0.693x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
