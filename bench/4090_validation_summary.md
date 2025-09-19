# RTX 4090 HF Fused Backend Validation Summary

Date: 2026-07-02  
GPU: NVIDIA GeForce RTX 4090 24GB, driver 570.124.06  
Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth` / converted HF dir `rwkv7-g1d-0.4b-hf`  
Checkpoint SHA256: `947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb`

This records the first real 4090 validation loop for the HF wrapper + native fused backend line. Albatross is used as the external performance acceptance line, not as our code path.

## Issue #66 HF adapter validation pass

Final validation artifact:
`bench/results_4090_issue66_final_20260702_113804.jsonl` (also appended to
`bench/results.jsonl`). Full remote log:
`/tmp/issue66_4090_final_20260702_113804.log`.

Environment:

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, sm_89 |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| Transformers | 5.12.1 |
| PEFT / TRL / DeepSpeed / Accelerate | 0.19.1 / 1.7.0 / 0.19.2 / 1.14.0 |
| bitsandbytes | 0.49.2 |
| Base model | `/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf` |
| Effective repo-code model | `/tmp/rwkv7_issue66_repo_model` |

Issue checklist status:

| check | status | notes |
|---|---|---|
| `tests/smoke_hf_generate.py` | PASS | generate uses `native_graph` fast-token backend |
| `tests/test_hf_api_contract.py` | PASS | fp16 and bf16, `fused_recurrent` |
| `tests/test_quantized_inference.py` | PASS | 8-bit and 4-bit; quantized fast-forward safely resolves to FLA |
| `bench/bench_speed.py` | PASS | fp16/bf16 rows appended |
| `bench/bench_batch_sweep.py` | PASS | bsz 1/2/4 rows appended |
| `tests/test_peft_lora.py` | PASS | LoRA gradients non-zero |
| `tests/test_hf_training_smoke.py` | PASS | Trainer + TRL SFT |
| `tests/test_hf_rl_training_smoke.py` | PASS | TRL DPO |

Issue #66 headline rows:

| row | dtype / quant | result |
|---|---|---:|
| speed prefill | fp16 | 22,222.6 tok/s |
| speed decode | fp16 | 376.7 tok/s |
| speed prefill | bf16 | 22,242.0 tok/s |
| speed decode | bf16 | 375.1 tok/s |
| batch sweep bsz=1 | fp16/native_graph decode | 377.0 tok/s |
| batch sweep bsz=2 | fp16/native_graph decode | 549.8 tok/s |
| batch sweep bsz=4 | fp16/native_graph decode | 1,138.0 tok/s |
| quant footprint | W8 | 571.8 MB model footprint, 622.1 MB peak VRAM |
| quant footprint | W4 | 427.8 MB model footprint, 494.8 MB peak VRAM |
| training | Trainer / TRL SFT / TRL DPO | pass, trainable delta ≈ `1e-4` |

Implementation note: bitsandbytes/HF W8/W4 modules have packed int8/int4
weights. The native dense fast-token runners are intentionally bypassed for
externally quantized models even when `RWKV7_FAST_TOKEN_BACKEND=native_graph`
is globally set; quantized fast-forward falls back to the FLA tensor path until
a dedicated native quant kernel is added.

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

## Native recurrent scan prefill prototype

The first native scan prototype isolates the recurrent part of prefill after
R/K/V/W/A projection. It is **not wired into the full HF prefill path yet**;
it is the kernel development target that should replace the slow chunked
helper once projection/output integration is added.

Benchmark command:

```bash
PYTHONPATH=. python bench/bench_fused_recurrent_scan.py \
  --dtype fp16 --device cuda --batch-sizes 1 4 --tokens 128 512 \
  --heads 16 --head-dim 64 --block-n 64 --chunk-size 64 \
  --warmup 4 --steps 32 --results bench/results.jsonl
```

| batch | tokens | native scan ms | native tok/s | FLA chunk ms | FLA tok/s | native / FLA speed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.08458 | 1,513,286.5 | 0.48032 | 266,490.0 | 5.6786x |
| 1 | 512 | 0.32533 | 1,573,771.6 | 0.48062 | 1,065,284.1 | 1.4773x |
| 4 | 128 | 0.08446 | 6,062,324.3 | 0.47362 | 1,081,039.4 | 5.6079x |
| 4 | 512 | 0.32610 | 6,280,254.5 | 0.48457 | 4,226,414.0 | 1.4860x |

Correctness gate:

- Native Triton scan vs native torch reference passed for T=128:
  output max abs diff `0.00390625`, state max abs diff <= `0.0002553`,
  min output cosine >= `0.99999988`.
- FLA `chunk_rwkv7` is tracked as a recurrent-only speed target. Its interface
  uses log-decay and DPLR orientation conventions, so strict correctness is
  checked against the native torch recurrence while FLA cosine is used as a
  sanity check (`0.9998878`-`0.9999724` in these rows).

Scan prototype conclusion: the recurrent scan kernel itself is no longer the
obvious prefill blocker in isolation; the next required work is wiring this
kernel into a full native prefill path and fusing the surrounding projection,
LoRA, groupnorm/output, and state-cache plumbing.

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

### Albatross-inspired R/K/V projection layout sweep

Albatross notes that small B/T performance depends heavily on GPU-specific
linear layout tuning. We added an HF-safe telemetry prototype instead of
copying the standalone engine:

- `single`: existing one-launch Triton R/K/V GEMV prototype.
- `splitk`: new two-launch split-K R/K/V GEMV prototype inspired by
  Albatross-style small-B K parallelism.

Benchmark command:

```bash
PYTHONPATH=. python bench/bench_albatross_projection_layout.py \
  --hf-dir /workspace/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --dtype fp16 --device cuda --batch-sizes 1 4 --layers 0 1 -1 \
  --backends single splitk --block-ms 8 16 32 64 \
  --block-ks 64 128 256 --warmup 4 --steps 32 \
  --results bench/results.jsonl
```

| batch | baseline cuBLAS R/K/V ms | best prototype | best prototype ms | prototype / baseline speed |
|---:|---:|---|---:|---:|
| 1 | 0.05100 | splitk m8 k64 | 0.09470 | 0.5385x |
| 4 | 0.05345 | single m64 k128 | 0.06597 | 0.8102x |

Correctness was good (`min_cosine >= 0.99999964`, max abs diff <= `0.00390625`),
but the simple split-K/layout sweep is still slower than cuBLAS on 4090 for
0.4B hidden=1024. Conclusion: do **not** integrate shallow R/K/V GEMV split-K
into `native_graph`. Borrow the deeper Albatross idea instead: fuse
layernorm + time-mix + projection / LoRA / output so the custom kernel removes
multiple launches and intermediate tensors, not just the dense GEMV.

## Native W8/W4 quant route

R/K/V fused quant sweep on layers 0/1/23, batch=1:

| quant | footprint ratio | best fused ms | fp16 baseline ms | fused / fp16 speed | fused / separate quant | accuracy |
|---|---:|---:|---:|---:|---:|---|
| W8 rowwise fused RKV | 0.502 | 0.05052 | 0.03661 | 0.7248x | 2.3359x | min cosine 0.9999377 |
| W4 rowwise fused RKV | 0.252 | 0.05272 | 0.03842 | 0.7288x | 2.1374x | min cosine 0.9790789 |

Quant conclusion: memory reduction works, and fused quant is much faster than separate quant GEMVs, but it is still slower than fp16 cuBLAS. Single fused R/K/V dequant-GEMV is insufficient for the final target. The next quant route should fuse deeper into projection/LoRA/state update or switch to a tensor-core-friendly activation/weight quant path.

## Kernel conclusion

Priority order from this 4090 validation:

1. **Prefill gap is largest**: HF chunked/full prefill is only 0.08x-0.16x of Albatross at T=512. The new native recurrent scan prototype is promising in isolation (1.48x-5.68x FLA recurrent-only speed), but it is not wired into full HF prefill yet. Next step is full native prefill integration with projection/LoRA/output fusion, not more wrapper changes.
2. **Decode bsz=1/4 gap remains material**: native_graph is 0.40x-0.47x of Albatross for bsz=1/4. The shallow Albatross-inspired R/K/V split-K/layout sweep was slower than cuBLAS, so target deeper fused fp16 layernorm+mix+projection/LoRA and attention output/recurrent integration first.
3. **Cache plumbing is no longer the bottleneck**: native_graph cache hit rate is ~0.993 and copy/bind share is <1%.
4. **Quant is a second-stage kernel problem**: W8/W4 footprint targets are partially met, but speed is only ~0.72x fp16. Do not spend more effort on wrapper-level quant; fuse quant with the hot projection/LoRA/recurrent path.
