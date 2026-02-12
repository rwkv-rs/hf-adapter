# AMD gfx1100 HF production close-out (2026-07-30)

This directory records the exact-card promotion of recurrent-scan prefill and
the paired A8W8 FFN work on AMD `gfx1100`.  It complements the output-head
MM8/MM4 evidence in `../amd_gfx1100_quant_20260728/README.md`.

## System and promotion boundary

- AMD Navi 31 / `gfx1100`, 47.98 GiB VRAM.
- ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, Triton 3.5.1.
- RWKV-7 G1D 0.4B and G1H 1.5B/2.9B, fp16 native HF adapter.
- Exact runtime `gcnArchName` dispatch. Other AMD architectures and unmeasured
  model/BxT shapes fail closed to the native compatibility path.
- The speed reference for scan promotion is an explicitly disabled,
  independently executed native recurrence (`RWKV7_NATIVE_PREFILL_FUSED_SCAN=0`).

## Promoted recurrent-scan prefill matrix

The policy allowlist contains all 60 measured rows:

- model shapes: `(1024,24)`, `(2048,24)`, `(2560,32)`;
- batches: 1, 2, 4, 8;
- prompt/chunk lengths: 32, 64, 128, 256, 512;
- launch: `BLOCK_M=64`, 8 warps.

| model | measured rows | speedup vs unfused recurrence | minimum cosine | greedy |
|---|---:|---:|---:|---:|
| 0.4B | 20 | 4.6895x - 57.6341x | >=0.99999994 | pass |
| 1.5B | 20 | 4.4611x - 30.3430x | >=0.99999994 | pass |
| 2.9B | 20 | 4.3664x - 27.8328x | >=0.99999994 | pass |

All 60 rows pass prefill greedy equality and decode-after-prefill greedy
equality. The largest observed fp16 absolute difference is 0.1875.

## End-to-end 0.4B result

Prompt 128, decode 32, native cache and native graph:

| batch | prefill tok/s | decode tok/s | decode ms/step | peak VRAM MiB |
|---:|---:|---:|---:|---:|
| 1 | 3,876.1 | 142.0 | 7.04 | 1,045.3 |
| 2 | 7,588.4 | 281.1 | 7.11 | 1,095.7 |
| 4 | 14,569.0 | 544.4 | 7.35 | 1,168.2 |
| 8 | 23,212.6 | 1,062.8 | 7.53 | 1,310.9 |

The promoted decode policy is 1.7589x at B1 and 1.7521x at B8 versus its
unfused policy baseline, with 32/32 and 256/256 greedy matches respectively.
Cache hit rate is 0.9941 in both runs.

Generation, Transformers API/beam search, PEFT LoRA backward, dynamic batching,
cache reorder/compact, chunked-prefill correctness, and six-step native Trainer
all pass. The generated smoke output is: `Hello! How can I assist you today`.

### Chunked prefill

For B1/P256, steady-state full prefill is 36.83 ms (6,951 tok/s). Chunked
prefill preserves greedy output, next-token decode, and sequence length:

| chunk | latency ms | tok/s | vs full |
|---:|---:|---:|---:|
| 32 | 252.06 | 1,015.6 | 0.1461x |
| 64 | 128.83 | 1,987.1 | 0.2859x |
| 128 | 67.89 | 3,770.7 | 0.5425x |

Chunking is therefore a bounded-memory/cancellation feature, not a throughput
claim against one-shot prefill. Promoting the 32/64/128 scan shapes removes the
old unfused chunk bottleneck.

## Quantization result and remaining boundary

The A8W8 path now uses backend-neutral Triton rounding, gfx1100's padded int8
matrix route, and a paired `key -> ReLU2 -> requant -> value + residual` FFN.
The 0.4B memory-lane run quantizes 49 eligible linears (minimum 4M parameters),
including 24 paired FFNs:

| batch | footprint / fp16 | saved | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7031 | 29.69% | 0.9411x | 0.8632x | 0.99996823 | pass |
| 8 | 0.7031 | 29.69% | 1.2527x | 0.9002x | 0.99992895 | pass |

This is an opt-in **memory lane**, not an all-phase speed promotion. On gfx1100,
`torch._int_mm` rejects 16 rows or fewer and must be padded to 17 rows, while
`torch._scaled_mm` is unavailable (ROCm support is limited to newer Instinct
targets). A 96-configuration MM8 FFN tile sweep found no competitive hidden
route: the best B8 key/value ratios were 0.1052x/0.1672x dense.

The production quantized speed lane remains the already measured output-head
MM8/MM4 route. Full-model W8/W4/A8W8 does lower memory, but is deliberately not
advertised as "no slower than fp16" until the remaining low-row backend/kernel
gap is closed. In particular, the prior 2.9B full-model W4 row also changes the
tested greedy stream and stays diagnostic-only.

## Evidence files

- `prefill_default_paired.jsonl`: T128/T512, 24 paired promotion rows.
- `prefill_chunk_shapes_paired.jsonl`: T32/T64/T256, 36 paired promotion rows.
- `chunked_prefill_promoted.jsonl`: isolated warmed B1/P256 chunk matrix.
- `a8w8_full_fused_ffn_final.jsonl`: paired A8W8 B1/B8 result.
- `mm8_ffn_tile_sweep.json`: 96-config gfx1100 FFN diagnosis.
- `e2e_validation_final/`: final runnable acceptance logs and JSONL results.
- `environment.json`, `model_manifest.json`, `SHA256SUMS`: reproduction
  metadata and artifact integrity.
