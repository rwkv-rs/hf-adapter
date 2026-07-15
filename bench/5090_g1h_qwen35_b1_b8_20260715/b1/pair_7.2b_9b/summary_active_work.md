# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.174x | 1.212x | 4.122x | 6/6 |
| Decode RWKV/Qwen | 2.893x | 2.908x | 2.915x | 6/6 |
| Model footprint RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.812x | 0.814x | 0.817x | 6/6 |
| Runtime working set RWKV/Qwen | 1.444x | 2.067x | 2.336x | 0/6 |
| Active parameters RWKV/Qwen | 0.804x | 0.804x | 0.804x | 6/6 |
| Prefill tok/s per active-B | 1.460x | 1.507x | 5.127x | 6/6 |
| Decode tok/s per active-B | 3.599x | 3.617x | 3.625x | 6/6 |
| Prefill active-param work rate | 0.944x | 0.974x | - | 6/6 |
| Decode active-param work rate | 2.326x | 2.338x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.174x / 1.212x | 2.893x / 2.908x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
