# RTX 5090 native quant benchmark (2026-07-04)

This artifact closes the native mm8/mm4 part of the RTX 5090 card issue. It records both isolated native R/K/V fused dequant-GEMV sweep rows and end-to-end HF decode rows using `quantize_model_mm8` / `quantize_model_mm4`.

## End-to-end native quant decode (0.1B fp16, native_graph, bsz=1)

| quant | replaced modules | footprint MB | footprint ratio | tok/s | speed ratio vs fp16 | final cos | same next | peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| none | 0 | 364.4 | 1.0000 | 957.3 | 1.0000 | 1.00000000 | True | 629.6 |
| mm8 | 1 | 316.6 | 0.8688 | 908.2 | 0.9487 | 0.99999428 | True | 1062.0 |
| mm4 | 1 | 292.6 | 0.8030 | 948.0 | 0.9903 | 0.99980855 | True | 1070.1 |

## Native R/K/V fused quant sweep

| quant | best block_m | best block_k | fused ms | speed vs fp16 | speed vs separate | footprint ratio | min cosine vs fp16 | peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int8_rowwise_fused_rkv | 32 | 64 | 0.03691 | 0.6706 | 1.7694 | 0.5026 | 0.9999033808708191 | 383.8 |
| int4_rowwise_fused_rkv | 32 | 64 | 0.03753 | 0.6613 | 1.7896 | 0.2526 | 0.9783441424369812 | 383.8 |

## Interpretation

- Native mm8/mm4 are functional and reduce model footprint on the 0.1B checkpoint.
- On this small 0.1B decode shape, native e2e speed does **not** beat fp16 yet: mm8 is 0.9487x and mm4 is 0.9903x.
- The isolated R/K/V fused kernels also remain below fp16 on this shape, while beating the separate dequant path by ~1.77-1.82x.
- This is enough to close the RTX 5090 card issue's native-quant evidence requirement, but it is not a claim that the final quantized-speed target is solved across larger models/cards.
