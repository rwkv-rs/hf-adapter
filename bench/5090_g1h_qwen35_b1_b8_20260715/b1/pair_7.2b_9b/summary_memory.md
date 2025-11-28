# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.073x | 2.330x | 5.725x | 18/18 |
| Decode RWKV/Qwen | 2.893x | 4.091x | 5.030x | 18/18 |
| Model footprint RWKV/Qwen | 0.264x | 0.429x | 0.804x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.305x | 0.448x | 0.817x | 18/18 |
| Runtime working set RWKV/Qwen | 1.444x | 2.420x | 8.036x | 0/18 |
| Active parameters RWKV/Qwen | 0.774x | 0.804x | 0.804x | 18/18 |
| Prefill tok/s per active-B | 1.334x | 2.957x | 7.396x | 18/18 |
| Decode tok/s per active-B | 3.599x | 5.286x | 6.256x | 18/18 |
| Prefill active-param work rate | 0.863x | 1.837x | - | reported 18/18 |
| Decode active-param work rate | 2.326x | 3.167x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.174x / 1.212x | 2.893x / 2.908x | - | - | - | - | - |
| w4 | 6 | 1.073x / 1.129x | 4.987x / 5.013x | - | - | - | 0.329x | 0.383x |
| w8 | 6 | 2.460x / 2.727x | 4.073x / 4.091x | - | - | - | 0.534x | 0.551x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
