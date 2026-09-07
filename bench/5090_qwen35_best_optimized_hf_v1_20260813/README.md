# RTX 5090 Qwen3.5 best-optimized HF reference v2

Status: **PASS, reference-only 48/48**.

This artifact records the strongest correctness-passing Qwen3.5 official-HF
operator reference measured on one RTX 5090 for the fixed B1/B8,
P128/P512/P2048 and D128/D512 matrix. It is not a joined RWKV/Qwen result:
[`validation.json`](validation.json) deliberately reports
`unified_main_table_eligible=false`.

## Scope and interpretation

- Model sizes: Qwen3.5 0.8B, 2B, 4B and 9B.
- Dense FP16 only; quantization, MTP and speculative decoding are disabled.
- Batch sizes 1 and 8; prompt lengths 128, 512 and 2048; decode lengths 128
  and 512; prefill chunk size 512.
- Every cell uses 3 warmup iterations and 7 measured iterations. The reported
  statistic is the median.
- Prefill uses the verified official FLA plus Dao-AILab `causal_conv1d` eager
  `DynamicCache` path.
- Decode uses a fixed correctness-passing `StaticCache` CUDA Graph route per
  model. Raw CUDA Graph is a repository benchmark optimization around the
  official Qwen operators, not an official Qwen Graph implementation.
- `independent_best_prefill_and_decode` is an independent-axis performance
  envelope. It is not a continuous end-to-end cache route, TTFT result, or
  DynamicCache-to-StaticCache handoff.
- B8 Decode is aggregate throughput across eight sequences.

The absence of a same-runtime RWKV candidate is intentional. This evidence may
serve as a Qwen reference lane, but it must not be used to calculate RWKV/Qwen
ratios or to mark the RTX 5090 unified main table complete.

## Fixed runtime

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, SM 12.0, 32,607 MiB |
| Driver | 595.58.03 |
| Python | 3.10.12 |
| PyTorch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| Triton | 3.4.0 |
| Transformers | 5.12.1 |
| FLA | 0.5.1 |
| causal-conv1d | 1.6.2.post1 |
| Repository commit | `1e80d3f7af6340c796a01eaae479274949c412dd` |

The FLA and causal-conv1d revisions in
[`runtime-lock.json`](runtime-lock.json) are locked acquisition revisions. The
source snapshots did not retain `.git` metadata, so the artifact does not claim
a local checkout verification. Installed package identity and native binary
hashes are recorded in
[`extension_build_manifest.json`](extension_build_manifest.json); per-row
operator contracts prove the effective runtime path.

## Decode route selection

Each model uses exactly one route for all 12 cells; there is no per-cell
fallback or route cherry-picking.

| Qwen3.5 | Fixed Decode route | Compile mode | Cells |
|---|---|---|---:|
| 0.8B | `static_cache_inductor_cudagraph` | `max-autotune` | 12 |
| 2B | `static_cache_inductor_cudagraph` | `max-autotune` | 12 |
| 4B | `static_cache_raw_cudagraph` | n/a | 12 |
| 9B | `static_cache_raw_cudagraph` | n/a | 12 |

Inductor was selected for 0.8B/2B after paired route probes showed a clear
Decode advantage while passing the v2 correctness contract. The strict
same-cache numerical gate rejected the Inductor candidate for 4B/9B, so those
models use raw CUDA Graph for the complete matrix. Failed diagnostic routes are
not mixed into the formal table.

## Correctness result

The same-cache hard gate compares StaticCache eager with the selected graph
route and requires finite logits, full-horizon greedy equality and minimum
cosine at least 0.9999. DynamicCache-to-StaticCache comparisons retain finite
trace, full greedy and prefill-next-token hard gates; their cosine is recorded
for diagnosis but is not thresholded because the two cache layouts exercise
different attention shapes.

| Qwen3.5 | Route | Cells | Same-cache min | Dynamic/Static min | Dynamic/Candidate min | Peak VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | Inductor Graph | 12 | 0.9999860525 | 0.9999855757 | 0.9999862909 | 2,313.2 |
| 2B | Inductor Graph | 12 | 0.9999872446 | 0.9999880791 | 0.9999868870 | 4,770.3 |
| 4B | raw Graph | 12 | 0.9999874830 | 0.9986497760 | 0.9986497760 | 10,225.8 |
| 9B | raw Graph | 12 | 0.9999860525 | 0.9992154837 | 0.9992154837 | 19,318.0 |

All 48 rows pass the official FLA/causal-conv1d contract, graph evidence,
pointer-stability, full-horizon greedy and finite-logit gates. Inductor rows
have zero graph breaks and zero CUDAGraph skips; raw rows have exactly one
`cudaGraphLaunch` per replay.

## Model / batch medians

| Qwen3.5 | GPU | Batch | Decode route | Cells | Prefill tok/s | Decode tok/s |
|---|---|---:|---|---:|---:|---:|
| 0.8B | RTX 5090 | B1 | `static_cache_inductor_cudagraph` | 6 | 14,467 | 559 |
| 0.8B | RTX 5090 | B8 | `static_cache_inductor_cudagraph` | 6 | 93,375 | 3,180 |
| 2B | RTX 5090 | B1 | `static_cache_inductor_cudagraph` | 6 | 14,177 | 325 |
| 2B | RTX 5090 | B8 | `static_cache_inductor_cudagraph` | 6 | 50,778 | 2,058 |
| 4B | RTX 5090 | B1 | `static_cache_raw_cudagraph` | 6 | 10,042 | 120 |
| 4B | RTX 5090 | B8 | `static_cache_raw_cudagraph` | 6 | 21,808 | 731 |
| 9B | RTX 5090 | B1 | `static_cache_raw_cudagraph` | 6 | 10,461 | 79.2 |
| 9B | RTX 5090 | B8 | `static_cache_raw_cudagraph` | 6 | 12,199 | 518 |

These medians are computed from the unrounded raw tok/s fields. Display
rounding is report-only: values at or above 100 use zero decimals and lower
values use one decimal. Machine-readable tok/s and all seven timing samples are
retained in [`summary.json`](summary.json) and the source JSONL files.

## Complete raw matrix

[`summary.md`](summary.md) contains the sorted 48-cell table in model, GPU,
B1/B8, prompt and decode order. [`qwen_reference.jsonl`](qwen_reference.jsonl)
is the corresponding sorted machine-readable table.

## Artifact map

- `qwen_0p8.jsonl`, `qwen_2b.jsonl`, `qwen_4b.jsonl`, `qwen_9b.jsonl`: the
  four 12-cell resident runs, including raw timing samples and correctness
  telemetry.
- `qwen_reference.jsonl`: validated and sorted 48-row Qwen reference table.
- `validation.json`: fail-closed matrix validation and eligibility decision.
- `summary.json`, `summary.md`: raw medians, correctness minima and complete
  display table.
- `environment.json`, `runtime-lock.json`, `pip-freeze.txt`, `system.csv`:
  fixed runtime and exact-card evidence.
- `extension_build_manifest.json`: installed extension identity and binary
  hashes.
- `model_revisions.jsonl`, `model_hashes.sha256`: pinned Hub revisions and
  checkpoint hashes.
- `logs/`: per-model formal logs plus validator and summarizer logs.
- `formal.log`, `exit_code.txt`: aggregate status.
- `artifact_sha256.txt`: checksum manifest for every other file in this
  directory.

There is deliberately no `candidate.jsonl` or `main_table.jsonl`.

## Reproduce

Run [`../run_5090_qwen35_best_optimized_hf.sh`](../run_5090_qwen35_best_optimized_hf.sh)
once per pinned checkpoint, setting `RESULT_NAME` and one fixed
`QWEN_DECODE_OPTIMIZATION` per model. Then validate and summarize:

```bash
python ../validate_qwen35_best_optimized_hf_v1.py \
  --reference-results qwen_0p8.jsonl qwen_2b.jsonl qwen_4b.jsonl qwen_9b.jsonl \
  --expected-device "NVIDIA GeForce RTX 5090" \
  --summary validation.json \
  --reference-table qwen_reference.jsonl

python ../summarize_qwen35_best_optimized_hf_v1.py \
  qwen_0p8.jsonl qwen_2b.jsonl qwen_4b.jsonl qwen_9b.jsonl \
  --expected-device "NVIDIA GeForce RTX 5090" \
  --json summary.json \
  --markdown summary.md
```

Exact model revisions, package versions, source identities and the repository
commit are recorded in this directory; do not substitute moving model tags or
different extension builds when reproducing the reference.
