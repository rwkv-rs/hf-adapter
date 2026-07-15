# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.031x | 1.107x | 1.953x | 18/18 |
| Decode RWKV/Qwen | 2.813x | 2.875x | 4.856x | 18/18 |
| Model footprint RWKV/Qwen | 0.444x | 0.782x | 0.804x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.505x | 0.999x | 1.042x | 9/18 |
| Runtime working set RWKV/Qwen | 1.314x | 2.650x | 6.729x | 0/18 |
| Active parameters RWKV/Qwen | 0.774x | 0.804x | 0.804x | 18/18 |
| Prefill tok/s per active-B | 1.282x | 1.377x | 2.429x | 18/18 |
| Decode tok/s per active-B | 3.499x | 3.575x | 6.039x | 18/18 |
| Prefill active-param work rate | 0.814x | 0.890x | - | reported 18/18 |
| Decode active-param work rate | 2.262x | 2.311x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.031x / 1.041x | 2.813x / 2.830x | - | - | - | - | - |
| w4 | 6 | 1.031x / 1.042x | 2.853x / 2.869x | - | - | - | 0.973x | 0.980x |
| w8 | 6 | 1.808x / 1.902x | 3.849x / 3.872x | - | - | - | 0.553x | 0.543x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
