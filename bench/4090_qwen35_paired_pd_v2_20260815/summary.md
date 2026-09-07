# RTX 5090 Qwen3.5 best-optimized HF raw results

Status: **PASS, reference-only**. This Qwen-only artifact is not eligible for the unified RWKV/Qwen main table because no same-runtime RWKV candidate was measured.

Prefill and Decode form an `independent_best_prefill_and_decode` performance envelope. They are not one continuous end-to-end request, TTFT result, or DynamicCache-to-StaticCache handoff. Prefill uses the official FLA plus `causal_conv1d` eager DynamicCache path. Raw CUDA Graph, where selected, is a repository benchmark optimization around the official Qwen operators, not an official Qwen Graph path.

Rows are sorted by model size, GPU, B1/B8, prompt and decode. Display values use 0 decimals at >=100 tok/s and 1 decimal below 100; JSONL and JSON retain the original numeric values and all seven samples.

## Correctness contract

**Same-cache hard gate:** StaticCache eager vs the candidate graph route must have finite logits, full-horizon greedy-token equality, and minimum cosine >= 0.9999.

**Cross-cache hard gates:** DynamicCache eager vs StaticCache eager and DynamicCache eager vs the candidate graph route must have finite logits, full-horizon greedy-token equality, and prefill-next-token equality. Cross-cache cosine is informational only and has no acceptance threshold.

| Qwen3.5 | GPU | Route | Cells | Same-cache min cosine | Dynamic/Static min cosine | Dynamic/Candidate min cosine |
|---|---|---|---:|---:|---:|---:|
| 0.8b | NVIDIA GeForce RTX 4090 | static_cache_inductor_cudagraph | 12 | 0.999986 | 0.999986 | 0.999985 |
| 2b | NVIDIA GeForce RTX 4090 | static_cache_inductor_cudagraph | 12 | 0.999988 | 0.999983 | 0.999977 |
| 4b | NVIDIA GeForce RTX 4090 | static_cache_raw_cudagraph | 12 | 0.999987 | 0.996799 | 0.996799 |
| 9b | NVIDIA GeForce RTX 4090 | static_cache_raw_cudagraph | 12 | 0.999985 | 0.998888 | 0.998888 |

## Model / batch medians

| Qwen3.5 | GPU | Batch | Decode route | Cells | Prefill tok/s | Decode tok/s |
|---|---|---:|---|---:|---:|---:|
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | static_cache_inductor_cudagraph | 6 | 9,311 | 407 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | static_cache_inductor_cudagraph | 6 | 65,924 | 2,252 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | static_cache_inductor_cudagraph | 6 | 9,220 | 212 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | static_cache_inductor_cudagraph | 6 | 36,953 | 1,302 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | static_cache_raw_cudagraph | 6 | 6,946 | 84.5 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | static_cache_raw_cudagraph | 6 | 14,458 | 518 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | static_cache_raw_cudagraph | 6 | 6,801 | 50.7 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | static_cache_raw_cudagraph | 6 | 8,434 | 331 |

## Complete 48-cell raw matrix

| Qwen3.5 | GPU | Batch | Prompt | Decode | Route | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---|---:|---:|
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 128 | 128 | static_cache_inductor_cudagraph | 2,799 | 447 |
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 128 | 512 | static_cache_inductor_cudagraph | 2,338 | 420 |
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 512 | 128 | static_cache_inductor_cudagraph | 9,303 | 419 |
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 512 | 512 | static_cache_inductor_cudagraph | 9,319 | 396 |
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 128 | static_cache_inductor_cudagraph | 9,374 | 338 |
| 0.8b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 512 | static_cache_inductor_cudagraph | 9,331 | 322 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 128 | 128 | static_cache_inductor_cudagraph | 18,361 | 2,570 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 128 | 512 | static_cache_inductor_cudagraph | 18,549 | 2,358 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 512 | 128 | static_cache_inductor_cudagraph | 72,737 | 2,352 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 512 | 512 | static_cache_inductor_cudagraph | 72,397 | 2,151 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 128 | static_cache_inductor_cudagraph | 65,965 | 1,677 |
| 0.8b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 512 | static_cache_inductor_cudagraph | 65,882 | 1,554 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 128 | 128 | static_cache_inductor_cudagraph | 2,706 | 221 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 128 | 512 | static_cache_inductor_cudagraph | 2,267 | 215 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 512 | 128 | static_cache_inductor_cudagraph | 9,238 | 215 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 512 | 512 | static_cache_inductor_cudagraph | 9,202 | 209 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 128 | static_cache_inductor_cudagraph | 9,300 | 192 |
| 2b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 512 | static_cache_inductor_cudagraph | 9,320 | 187 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 128 | 128 | static_cache_inductor_cudagraph | 18,079 | 1,411 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 128 | 512 | static_cache_inductor_cudagraph | 18,322 | 1,339 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 512 | 128 | static_cache_inductor_cudagraph | 41,078 | 1,337 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 512 | 512 | static_cache_inductor_cudagraph | 40,903 | 1,267 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 128 | static_cache_inductor_cudagraph | 36,943 | 1,089 |
| 2b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 512 | static_cache_inductor_cudagraph | 36,964 | 1,036 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 128 | 128 | static_cache_raw_cudagraph | 2,037 | 87.9 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 128 | 512 | static_cache_raw_cudagraph | 1,756 | 85.2 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 512 | 128 | static_cache_raw_cudagraph | 6,943 | 85.3 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 512 | 512 | static_cache_raw_cudagraph | 6,949 | 83.8 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 128 | static_cache_raw_cudagraph | 7,002 | 80.0 |
| 4b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 512 | static_cache_raw_cudagraph | 7,018 | 78.4 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 128 | 128 | static_cache_raw_cudagraph | 13,804 | 569 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 128 | 512 | static_cache_raw_cudagraph | 13,732 | 536 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 512 | 128 | static_cache_raw_cudagraph | 15,979 | 535 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 512 | 512 | static_cache_raw_cudagraph | 15,948 | 500 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 128 | static_cache_raw_cudagraph | 14,470 | 420 |
| 4b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 512 | static_cache_raw_cudagraph | 14,445 | 400 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 128 | 128 | static_cache_raw_cudagraph | 1,994 | 52.0 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 128 | 512 | static_cache_raw_cudagraph | 1,724 | 51.0 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 512 | 128 | static_cache_raw_cudagraph | 6,804 | 51.0 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 512 | 512 | static_cache_raw_cudagraph | 6,797 | 50.5 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 128 | static_cache_raw_cudagraph | 6,974 | 49.2 |
| 9b | NVIDIA GeForce RTX 4090 | B1 | 2048 | 512 | static_cache_raw_cudagraph | 6,951 | 48.6 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 128 | 128 | static_cache_raw_cudagraph | 9,272 | 352 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 128 | 512 | static_cache_raw_cudagraph | 9,281 | 339 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 512 | 128 | static_cache_raw_cudagraph | 8,436 | 339 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 512 | 512 | static_cache_raw_cudagraph | 8,432 | 324 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 128 | static_cache_raw_cudagraph | 7,972 | 289 |
| 9b | NVIDIA GeForce RTX 4090 | B8 | 2048 | 512 | static_cache_raw_cudagraph | 7,958 | 279 |
