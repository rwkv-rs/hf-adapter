# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `24/24` cells.

Required Qwen backend: `fla`; verified: `24/24` cells.

Required Qwen full fusion: `true`; verified: `24/24` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.393x | 1.857x | 8.633x | 24/24 |
| Decode RWKV/Qwen | 2.105x | 4.361x | 10.136x | 24/24 |
| Model footprint RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Peak VRAM RWKV/Qwen | 0.533x | 0.754x | 1.040x | 21/24 |
| Runtime working set RWKV/Qwen | 0.191x | 0.936x | 21.947x | 13/24 |
| Active parameters RWKV/Qwen | 0.599x | 0.752x | 0.812x | 24/24 |
| Prefill tok/s per active-B | 1.716x | 2.624x | 14.410x | 24/24 |
| Decode tok/s per active-B | 2.618x | 5.788x | 16.918x | 24/24 |
| Prefill active-param work rate | 1.038x | 1.352x | - | 24/24 |
| Decode active-param work rate | 1.693x | 3.302x | - | 24/24 |

Strict speed cells: `24/24`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 24 | 1.393x / 1.857x | 2.105x / 4.361x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
