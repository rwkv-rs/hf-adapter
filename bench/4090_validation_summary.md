# RTX 4090 HF Fused Backend Validation Summary

Date: 2026-07-02  
GPU: NVIDIA GeForce RTX 4090 24GB, driver 570.124.06  
Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth` / converted HF dir `rwkv7-g1d-0.4b-hf`  
Checkpoint SHA256: `947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb`

This records the first real 4090 validation loop for the HF wrapper + native fused backend line. Albatross is used as the external performance acceptance line, not as our code path.

## Albatross A/B

Albatross backends built/run on the 4090:

- `faster4_cpp`, built with `CMAKE_CUDA_ARCHITECTURES=89`.
- `faster3a`, default fp16 WKV; analyzer picks the fastest Albatross row per case.

### Decode, T=1

| batch | HF native_graph tok/s | Albatross tok/s | HF / Albatross |
|---:|---:|---:|---:|
| 1 | 392.3 | 833.111 | 0.4709 |
| 4 | 1124.9 | 2791.18 | 0.4030 |
| 8 | 2205.1 | 2347.53 | 0.9393 |

Decode conclusion: bsz=8 is close, but bsz=1/4 are still far from Albatross. P1 target is not met: decode min ratio is **0.403x**.

### Prefill, T=512

| batch | HF chunked prefill tok/s | Albatross tok/s | HF / Albatross |
|---:|---:|---:|---:|
| 1 | 4672.7 | 60047.8 | 0.0778 |
| 4 | 19178.6 | 117789.0 | 0.1628 |

Prefill conclusion: this is the largest gap. P1 target is not met: prefill min ratio is **0.0778x**.

## Chunked prefill correctness + speed

Prompt tokens: 512, dtype fp16, attn mode fused_recurrent, fuse_norm=false.

| batch | mode | chunk | tok/s | speed vs full | peak VRAM MB | VRAM vs full | max diff | decode diff | seq match |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | full | - | 9245.7 | 1.0000 | 1162.9 | 1.0000 | - | - | yes |
| 1 | chunked | 64 | 1158.2 | 0.1253 | 1144.3 | 0.9840 | 0.1250 | 0.0625 | yes |
| 1 | chunked | 128 | 2352.8 | 0.2545 | 907.7 | 0.7805 | 0.0625 | 0.0625 | yes |
| 1 | chunked | 256 | 4672.7 | 0.5054 | 917.9 | 0.7893 | 0.0625 | 0.0625 | yes |
| 4 | full | - | 37141.2 | 1.0000 | 1082.2 | 1.0000 | - | - | yes |
| 4 | chunked | 64 | 4735.1 | 0.1275 | 980.2 | 0.9057 | 0.0625 | 0.0625 | yes |
| 4 | chunked | 128 | 9537.3 | 0.2568 | 1023.3 | 0.9456 | 0.0625 | 0.0625 | yes |
| 4 | chunked | 256 | 19178.6 | 0.5164 | 1064.3 | 0.9835 | 0.0625 | 0.0625 | yes |

Chunked prefill conclusion: correctness is good, but current chunked helper is 0.13x-0.52x of full HF prefill and 0.08x-0.16x of Albatross. It is a semantic bridge for future vLLM/sglang-style prefill scheduling, not yet a fast prefill backend. Native graph decode cache hit is not used during chunked prefill; decode/native_graph cache validation separately shows >0.99 graph-cache hit rate in the replay-overhead bench and 1.0 in dynamic-batch runs.

## Decode bottleneck profiling

### End-to-end decode

| batch | HF forward tok/s | native_graph tok/s | native_graph ms/step |
|---:|---:|---:|---:|
| 1 | 28.7 | 392.3 | 2.55 |
| 4 | 111.0 | 1124.9 | 3.56 |
| 8 | 216.3 | 2205.1 | 3.63 |

### Native graph replay overhead

| batch | graph replay ms | copy cache ms | bind cache ms | cache hit rate | copy share |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.6184 | 0.0252 | 0.0023 | 0.9932 | 0.0089 |
| 4 | 3.6462 | 0.0163 | 0.0021 | 0.9932 | 0.0041 |
| 8 | 3.7471 | 0.0150 | 0.0016 | 0.9932 | 0.0037 |

Overhead conclusion: cache copy/bind is not the bottleneck anymore. Most time is inside CUDA graph replay.

### Component-level direction

For bsz=4 component profiling, top components were:

1. `attn_linears_lora`: 8.177 ms/token in the instrumented path
2. `attn_norm_out_proj`: 4.2371 ms/token
3. `attn_recurrent`: 3.2793 ms/token
4. `attn_key_mix_norm`: 2.831 ms/token
5. `ffn_key_relu`: 1.7988 ms/token
6. `attn_shift_mix`: 1.7837 ms/token

Component profiler is slower than native_graph because it instruments the decomposed path, but the relative ranking is useful: next decode work should target projection/LoRA and attention output/recurrent fusion, not wrapper/cache plumbing.

## Native W8/W4 quant route

R/K/V fused quant sweep on layers 0/1/23, batch=1:

| quant | footprint ratio | best fused ms | fp16 baseline ms | fused / fp16 speed | fused / separate quant | accuracy |
|---|---:|---:|---:|---:|---:|---|
| W8 rowwise fused RKV | 0.502 | 0.05052 | 0.03661 | 0.7248x | 2.3359x | min cosine 0.9999377 |
| W4 rowwise fused RKV | 0.252 | 0.05272 | 0.03842 | 0.7288x | 2.1374x | min cosine 0.9790789 |

Quant conclusion: memory reduction works, and fused quant is much faster than separate quant GEMVs, but it is still slower than fp16 cuBLAS. Single fused R/K/V dequant-GEMV is insufficient for the final target. The next quant route should fuse deeper into projection/LoRA/state update or switch to a tensor-core-friendly activation/weight quant path.

## Kernel conclusion

Priority order from this 4090 validation:

1. **Prefill gap is largest**: HF chunked/full prefill is only 0.08x-0.16x of Albatross at T=512. Need native fused prefill/scan or a chunked prefill kernel, not more wrapper changes.
2. **Decode bsz=1/4 gap remains material**: native_graph is 0.40x-0.47x of Albatross for bsz=1/4. Target fused fp16 projection/LoRA + attention output/recurrent integration first.
3. **Cache plumbing is no longer the bottleneck**: native_graph cache hit rate is ~0.993 and copy/bind share is <1%.
4. **Quant is a second-stage kernel problem**: W8/W4 footprint targets are partially met, but speed is only ~0.72x fp16. Do not spend more effort on wrapper-level quant; fuse quant with the hot projection/LoRA/recurrent path.
