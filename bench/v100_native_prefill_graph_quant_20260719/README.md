# V100 Native prefill graph and quant-prefill optimization

Date: 2026-07-19

GPU: Tesla V100-PCIE-32GB (sm70)

Model: RWKV7-G1G 1.5B, fp16
Runtime: PyTorch 2.5.1+cu124

This artifact separates the production speed gate from the full-memory
quantization lane. A quantized result is not called speed-complete merely
because it saves memory or accelerates decode.

## Dense Native prefill

Fixed-shape Native CUDA graphs close the previous pure-Native short-prefill
regression:

| Batch | Prompt | Native graph tok/s | prior wrapper tok/s | Native / wrapper |
|---:|---:|---:|---:|---:|
| 1 | 128 | 6,111.57 | 5,831.4 | 1.048x |
| 8 | 128 | 18,287.22 | 17,495.5 | 1.045x |
| 1 | 512 | 10,874.77 | n/a | n/a |
| 8 | 512 | 20,762.07 | n/a | n/a |

The wrapper reference is
`../v100_pr57_regression_20260718/native_vs_wrapper_15b_exact.jsonl`.
Prompt and continuation top-1 equality pass; minimum cosine is 1.0 in all four
Native graph rows.

## Chunked prefill

Prompt length is 512. Ratios are chunked throughput divided by full-prefill
throughput from the same run.

| Batch | Chunk | tok/s | ratio vs full |
|---:|---:|---:|---:|
| 1 | 64 | 4,774.5 | 0.4190x |
| 1 | 128 | 7,215.9 | 0.6332x |
| 1 | 256 | 9,967.6 | 0.8747x |
| 8 | 64 | 15,776.6 | 0.7584x |
| 8 | 128 | 18,808.5 | 0.9042x |
| 8 | 256 | 20,104.3 | 0.9665x |

This is a large improvement over the prior Native B1 ratios
`0.1241/0.2487/0.4867` and B8 ratios `0.1649/0.3331/0.6637` for chunks
64/128/256. B1 small chunks remain below the production target because each
chunk repeats fixed graph, projection, and state-transfer work.

## W4 production speed profile

The `speed` profile quantizes the language head while retaining dense hot FFN
projections. It is the current V100 policy that satisfies both speed and memory
gates at B1 and B8, prompt 128 and measured decode length 16:

| Batch | prefill / fp16 | decode / fp16 | footprint / fp16 | prompt/final cosine |
|---:|---:|---:|---:|---:|
| 1 | 1.0180x | 1.0293x | 0.9348x | 0.999854 / 0.999816 |
| 8 | 1.0024x | 1.0012x | 0.9348x | 0.999813 / 0.999783 |

Greedy-token equality and repeat determinism pass in both rows.

## Full-memory profiles: improved, not speed-complete

The full-memory routes keep every large FFN projection packed. The new path
uses graph-captured side-stream dequantization into bounded reusable workspaces
and tensor-core dense GEMMs for prompt rows.

| Profile | prefill before | prefill after | decode / fp16 | footprint / fp16 | status |
|---|---:|---:|---:|---:|---|
| W4 memory, B1/T128 | about 0.13x | 0.9166x best clean row | 1.3354x | 0.5395x | prefill gap open |
| W8 memory, B1/T128 | 0.3872x | 0.8514x | 1.0156x | 0.6932x | prefill gap open |

All listed full-memory rows pass logits cosine, same-greedy, and deterministic
repeat checks. They are retained as memory-specialized profiles and are not
used as evidence that V100 quantized prefill is faster than fp16.

The follow-up MM8 transpose sweep replaces the square 32x32 tile with
orientation-aware 32x64 (`[2048,8192]`) and 32x16 (`[8192,2048]`) tiles. The
two exact 1.5B FFN operands improve by `1.0540x` and `1.0403x` respectively,
with bit-identical output. This microkernel win is enabled in code, but the
`0.8514x` table row remains the last clean end-to-end result until a fresh
uncontended B1/B8 rerun is recorded.

V100 has no native INT4 tensor-core MMA. A trial tiled W4 WMMA kernel was
correct but substantially slower than fused dequantization plus cuBLAS, so it
was rejected rather than promoted. Closing the last full-memory prefill gap
requires a materially better packed-weight tensor-core tile schedule; changing
the report threshold or silently selecting the memory profile is not accepted.

A second same-session diagnostic replaced the W8 Triton 32x32
transpose-dequantizer with a hand-written CUDA 64x64 shared-memory tile. It was
bit-identical but slower on both 1.5B FFN orientations (`0.1408/0.1315 ms`
versus `0.0909/0.1058 ms`), so that implementation was also removed rather
than adding a slower architecture-specific branch.

## Files

- `native_prefill_graph_b1b8.jsonl`: dense B1/B8, prompt 128.
- `native_prefill_graph_512_b1b8.jsonl`: dense B1/B8, prompt 512.
- `pr58_chunk_b1.jsonl`, `pr58_chunk_b8.jsonl`: chunk sweep.
- `pr58_w4_speed_b1.jsonl`, `pr58_w4_speed_b8.jsonl`: passing W4 speed profile.
- `pr58_w4_stream_b1.jsonl`: clean W4 full-memory optimization row.
- `pr58_w8_current_b1.jsonl`: W8 full-memory baseline before fused prefetch.
- `pr58_w8_nolook_b1.jsonl`: W8 full-memory row after fused prefetch.
- `mm8_dequant_tile_sweep.jsonl`: orientation-aware V100 W8 dequantizer A/B.
