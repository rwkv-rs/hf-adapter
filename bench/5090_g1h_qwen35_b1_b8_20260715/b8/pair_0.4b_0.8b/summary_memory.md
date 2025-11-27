# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.344x | 1.615x | 5.480x | 18/18 |
| Decode RWKV/Qwen | 6.860x | 7.203x | 7.267x | 18/18 |
| Model footprint RWKV/Qwen | 0.398x | 0.555x | 0.599x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.555x | 0.648x | 0.709x | 18/18 |
| Runtime working set RWKV/Qwen | 0.682x | 0.983x | 1.132x | 12/18 |
| Active parameters RWKV/Qwen | 0.510x | 0.599x | 0.599x | 18/18 |
| Prefill tok/s per active-B | 2.320x | 2.764x | 10.602x | 18/18 |
| Decode tok/s per active-B | 11.450x | 12.094x | 14.188x | 18/18 |
| Prefill active-param work rate | 0.685x | 0.968x | - | reported 18/18 |
| Decode active-param work rate | 3.663x | 4.267x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.390x / 1.613x | 7.102x / 7.129x | - | - | - | - | - |
| w4 | 6 | 1.344x / 1.617x | 7.225x / 7.245x | - | - | - | 0.891x | 0.936x |
| w8 | 6 | 1.390x / 1.936x | 6.860x / 7.203x | - | - | - | 0.926x | 0.956x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
