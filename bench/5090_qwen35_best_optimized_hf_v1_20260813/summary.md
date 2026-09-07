# RTX 5090 Qwen3.5 best-optimized HF raw results

Status: **PASS, reference-only**. This Qwen-only artifact is not eligible for the unified RWKV/Qwen main table because no same-runtime RWKV candidate was measured.

Prefill and Decode form an `independent_best_prefill_and_decode` performance envelope. They are not one continuous end-to-end request, TTFT result, or DynamicCache-to-StaticCache handoff. Prefill uses the official FLA plus `causal_conv1d` eager DynamicCache path. Raw CUDA Graph, where selected, is a repository benchmark optimization around the official Qwen operators, not an official Qwen Graph path.

Rows are sorted by model size, GPU, B1/B8, prompt and decode. Display values use 0 decimals at >=100 tok/s and 1 decimal below 100; JSONL and JSON retain the original numeric values and all seven samples.

## Correctness contract

**Same-cache hard gate:** StaticCache eager vs the candidate graph route must have finite logits, full-horizon greedy-token equality, and minimum cosine >= 0.9999.

**Cross-cache hard gates:** DynamicCache eager vs StaticCache eager and DynamicCache eager vs the candidate graph route must have finite logits, full-horizon greedy-token equality, and prefill-next-token equality. Cross-cache cosine is informational only and has no acceptance threshold.

| Qwen3.5 | GPU | Route | Cells | Same-cache min cosine | Dynamic/Static min cosine | Dynamic/Candidate min cosine |
|---|---|---|---:|---:|---:|---:|
| 0.8b | NVIDIA GeForce RTX 5090 | static_cache_inductor_cudagraph | 12 | 0.999986 | 0.999986 | 0.999986 |
| 2b | NVIDIA GeForce RTX 5090 | static_cache_inductor_cudagraph | 12 | 0.999987 | 0.999988 | 0.999987 |
| 4b | NVIDIA GeForce RTX 5090 | static_cache_raw_cudagraph | 12 | 0.999987 | 0.998650 | 0.998650 |
| 9b | NVIDIA GeForce RTX 5090 | static_cache_raw_cudagraph | 12 | 0.999986 | 0.999215 | 0.999215 |

## Model / batch medians

| Qwen3.5 | GPU | Batch | Decode route | Cells | Prefill tok/s | Decode tok/s |
|---|---|---:|---|---:|---:|---:|
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | static_cache_inductor_cudagraph | 6 | 14,467 | 559 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | static_cache_inductor_cudagraph | 6 | 93,375 | 3,180 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | static_cache_inductor_cudagraph | 6 | 14,177 | 325 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | static_cache_inductor_cudagraph | 6 | 50,778 | 2,058 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | static_cache_raw_cudagraph | 6 | 10,042 | 120 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | static_cache_raw_cudagraph | 6 | 21,808 | 731 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | static_cache_raw_cudagraph | 6 | 10,461 | 79.2 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | static_cache_raw_cudagraph | 6 | 12,199 | 518 |

## Complete 48-cell raw matrix

| Qwen3.5 | GPU | Batch | Prompt | Decode | Route | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---|---:|---:|
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 128 | 128 | static_cache_inductor_cudagraph | 4,150 | 654 |
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 128 | 512 | static_cache_inductor_cudagraph | 3,576 | 586 |
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 512 | 128 | static_cache_inductor_cudagraph | 14,445 | 584 |
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 512 | 512 | static_cache_inductor_cudagraph | 14,703 | 534 |
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 128 | static_cache_inductor_cudagraph | 14,758 | 424 |
| 0.8b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 512 | static_cache_inductor_cudagraph | 14,489 | 396 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 128 | 128 | static_cache_inductor_cudagraph | 29,332 | 3,722 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 128 | 512 | static_cache_inductor_cudagraph | 28,304 | 3,379 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 512 | 128 | static_cache_inductor_cudagraph | 112,490 | 3,371 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 512 | 512 | static_cache_inductor_cudagraph | 111,332 | 2,988 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 128 | static_cache_inductor_cudagraph | 93,260 | 2,322 |
| 0.8b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 512 | static_cache_inductor_cudagraph | 93,490 | 2,142 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 128 | 128 | static_cache_inductor_cudagraph | 4,102 | 352 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 128 | 512 | static_cache_inductor_cudagraph | 3,549 | 333 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 512 | 128 | static_cache_inductor_cudagraph | 14,225 | 334 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 512 | 512 | static_cache_inductor_cudagraph | 14,322 | 317 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 128 | static_cache_inductor_cudagraph | 14,753 | 272 |
| 2b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 512 | static_cache_inductor_cudagraph | 14,128 | 261 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 128 | 128 | static_cache_inductor_cudagraph | 28,335 | 2,261 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 128 | 512 | static_cache_inductor_cudagraph | 28,310 | 2,122 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 512 | 128 | static_cache_inductor_cudagraph | 57,718 | 2,114 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 512 | 512 | static_cache_inductor_cudagraph | 57,761 | 2,003 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 128 | static_cache_inductor_cudagraph | 50,793 | 1,650 |
| 2b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 512 | static_cache_inductor_cudagraph | 50,763 | 1,561 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 128 | 128 | static_cache_raw_cudagraph | 3,108 | 127 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 128 | 512 | static_cache_raw_cudagraph | 2,775 | 122 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 512 | 128 | static_cache_raw_cudagraph | 11,175 | 123 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 512 | 512 | static_cache_raw_cudagraph | 11,149 | 118 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 128 | static_cache_raw_cudagraph | 8,935 | 108 |
| 4b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 512 | static_cache_raw_cudagraph | 11,269 | 105 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 128 | 128 | static_cache_raw_cudagraph | 21,896 | 796 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 128 | 512 | static_cache_raw_cudagraph | 21,721 | 752 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 512 | 128 | static_cache_raw_cudagraph | 23,502 | 751 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 512 | 512 | static_cache_raw_cudagraph | 23,416 | 711 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 128 | static_cache_raw_cudagraph | 20,739 | 618 |
| 4b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 512 | static_cache_raw_cudagraph | 20,681 | 591 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 128 | 128 | static_cache_raw_cudagraph | 3,041 | 82.0 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 128 | 512 | static_cache_raw_cudagraph | 2,674 | 80.3 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 512 | 128 | static_cache_raw_cudagraph | 10,599 | 80.1 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 512 | 512 | static_cache_raw_cudagraph | 10,646 | 78.2 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 128 | static_cache_raw_cudagraph | 10,498 | 73.7 |
| 9b | NVIDIA GeForce RTX 5090 | B1 | 2048 | 512 | static_cache_raw_cudagraph | 10,423 | 72.3 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 128 | 128 | static_cache_raw_cudagraph | 12,191 | 550 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 128 | 512 | static_cache_raw_cudagraph | 12,208 | 529 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 512 | 128 | static_cache_raw_cudagraph | 12,774 | 529 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 512 | 512 | static_cache_raw_cudagraph | 12,722 | 508 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 128 | static_cache_raw_cudagraph | 11,841 | 459 |
| 9b | NVIDIA GeForce RTX 5090 | B8 | 2048 | 512 | static_cache_raw_cudagraph | 11,819 | 444 |
