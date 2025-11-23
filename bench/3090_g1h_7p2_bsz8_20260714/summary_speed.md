# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `false`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.052x | 1.111x | 2.163x | 18/18 |
| Decode RWKV/Qwen | 1.788x | 1.856x | 1.988x | 18/18 |
| Model footprint RWKV/Qwen | 0.444x | 0.782x | 0.804x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.504x | 0.858x | 0.995x | 18/18 |
| Runtime working set RWKV/Qwen | 1.842x | 2.047x | 2.239x | 0/18 |
| Active parameters RWKV/Qwen | 0.774x | 0.804x | 0.804x | 18/18 |
| Prefill tok/s per active-B | 1.317x | 1.389x | 2.691x | 18/18 |
| Decode tok/s per active-B | 2.224x | 2.398x | 2.473x | 18/18 |
| Prefill active-param work rate | 0.815x | 0.866x | - | reported 18/18 |
| Decode active-param work rate | 1.423x | 1.454x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.059x / 1.064x | 1.788x / 1.808x | - | - | - | - | - |
| w4 | 6 | 1.052x / 1.062x | 1.839x / 1.856x | 0.989x | 1.026x | 1.015x | 0.972x | 0.981x |
| w8 | 6 | 1.901x / 2.148x | 1.941x / 1.962x | 1.697x | 1.084x | 1.099x | 0.553x | 0.703x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
