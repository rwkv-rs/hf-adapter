# Native-graph W4 R/K/V integration (RTX 4090)

Date: 2026-07-11

Branch: `wangyue/kernelbench-mega-w4a16-bench`

## Scope

This validates the KernelBench-Mega-inspired projection-axis W4A16 R/K/V path
inside a complete RWKV-7 native-graph decode step. Fused time-mix produces the
`[batch, 3, hidden]` layout directly, and projection-major W4 weights are packed
once outside the timed graph. The experiment remains opt-in through
`RWKV7_NATIVE_GRAPH_W4_RKV=1`.

## Environment and method

- GPU: NVIDIA GeForce RTX 4090 (Ada, SM89)
- Model: RWKV-7 G1D 0.4B HF, 24 layers, hidden size 1024
- dtype: fp16; recurrent state accumulation remains fp32
- prompt state: eight dense-prefilled tokens
- timed region: embedding, all 24 layers, final norm, and lm_head
- measurement: 256 CUDA-graph replays, three independent preheated runs
- quality: dense and W4 start from the same recurrent state; 16-token greedy
  rollout is checked for every sequence

## Card-local tile selection

An isolated microkernel sweep selected `(32,64,4)` for batch 1, but whole-graph
confirmation—not the isolated kernel alone—decided the runtime policy:

| batch | selected `(M,K,warps)` | rejected candidate | selected speedup | candidate speedup |
|---:|---:|---:|---:|---:|
| 1 | `(32,64,4)` | `(8,64,1)` | **1.2062x** | 1.1912x |
| 2 | `(16,128,4)` | `(64,32,2)` | **1.2163x** | 1.1478x |

The batch-2 result shows why microkernel winners must be rechecked in the whole
model. Batch 4 uses `(16,128,4)` and batch 8 uses `(16,128,2)`. Environment
variables continue to override all defaults.

## Stable full-token result

Values are the medians of the three final runs.

| batch | dense ms | W4 ms | W4 speedup | dense tok/s | W4 tok/s | min cosine | greedy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.573159 | **2.145807** | **1.1970x** | 388.63 | **466.03** | 0.999485 | 48/48 |
| 2 | 3.515499 | **2.900080** | **1.2122x** | 568.91 | **689.64** | 0.999484 | 96/96 |
| 4 | 3.546052 | **2.988079** | **1.1867x** | 1128.02 | **1338.65** | 0.999484 | 192/192 |
| 8 | 3.624958 | **3.160197** | **1.1467x** | 2206.92 | **2531.49** | 0.999484 | 384/384 |

The packed R/K/V slice is 36.2812 MiB versus 144 MiB fp16 (`0.252x`). Current
peak model VRAM does not realize that reduction because dense fallback weights
are still retained.

The repo-code HF public-API handoff smoke also passed at batch 1 and 4: native
prefill handed state to `native_graph`, greedy continuation matched, and the
decode-after-prefill maximum absolute logit difference was 0.125.

## Interpretation

The projection-axis W4 integration now clears its end-to-end speed gate on both
V100 and RTX 4090 for batch 1/2/4/8. It is not finished production W4: dense
weight release, serialized prepacking, and group-wise/asymmetric quality remain
open. No result here claims full-model W4 or Q*_K_M-equivalent accuracy.

## Artifacts

- `bench/results_kernelbench_mega_w4a16_rkv_4090_20260711.jsonl`
- `bench/results_native_graph_w4_rkv_4090_config_20260711.jsonl`
- `bench/results_native_graph_w4_rkv_4090_20260711.jsonl`
- `bench/results_native_graph_w4_rkv_public_api_4090_20260711.jsonl`
