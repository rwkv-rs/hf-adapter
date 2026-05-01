# RTX 4090 best-optimized HF vs official Qwen3.5 fast path

Status: **PASS**. This is the promoted RTX 4090 comparison for the
`hf_fast_path_v1` shape matrix. All 48 RWKV candidate rows and all 48 Qwen
reference rows pass the fail-closed runtime contract. Parameter-adjusted
Prefill and Decode both clear Qwen in **48/48** matched cells.

## Scope and interpretation

- GPU: one NVIDIA GeForce RTX 4090 (`sm_89`, 24 GiB).
- Models: RWKV-7 0.4B/1.5B/2.9B/7.2B versus Qwen3.5 0.8B/2B/4B/9B.
- Dense FP16, batch 1/8, prompt 128/512/2048, decode 128/512.
- Prefill chunk 512; warmup 3; measured runs 7; median per cell.
- Quantization, MTP and speculative decoding are disabled.
- Qwen is the official Transformers FLA path with Dao-AILab
  `causal_conv1d`; every row verifies the live Prefill, Decode and convolution
  bindings. It is a strong official HF fast-operator baseline, but the overall
  HF call remains eager rather than a whole-token CUDA Graph.
- RWKV is the exact-card **best optimized HF** lane. Decode uses
  `native_graph`; Prefill uses the promoted native graph/fused path and
  exact-shape FP16 block accumulation. Only 7.2B B8/P2048 disables Prefill
  Graph to fit 24 GiB; its Decode remains on `native_graph`.

The no-Graph `native_jit` matrix is retained separately as diagnostic evidence
in `../4090_hf_fast_path_v1_20260812/`; it is not the primary max-performance
comparison requested here.

## Acceptance result

The active-parameter adjustment is:

```text
adjusted ratio = (RWKV tok/s / Qwen tok/s)
               * (RWKV active parameters / Qwen active parameters)
```

| Metric across all 48 matched cells | Minimum | Median | Cells above 1.0x |
|---|---:|---:|---:|
| Raw Prefill ratio | 1.361373x | 2.315043x | 48/48 |
| Adjusted Prefill ratio | 1.060506x | 1.549011x | 48/48 |
| Raw Decode ratio | 2.275368x | 5.871032x | 48/48 |
| Adjusted Decode ratio | 1.829468x | 4.468521x | 48/48 |

The weakest adjusted Prefill cell is 0.4B/0.8B, B8, P512, D128 at
`1.060506x`. The weakest adjusted Decode cell is 7.2B/9B, B8, P512, D512 at
`1.829468x`.

## Median results by model pair and batch

Throughput columns are medians in tok/s: values at or above 100 use zero
decimal places, while values below 100 use one decimal place. B8 Decode is
aggregate throughput across all eight sequences, not per-sequence throughput.
This rounding is display-only. Full-precision original Prefill and Decode
measurements remain in [candidate.jsonl](candidate.jsonl),
[qwen_reference.jsonl](qwen_reference.jsonl), and the complete joined
[main_table.jsonl](main_table.jsonl); read `prefill_tokps_total` and
`decode_tokps_total`. No raw measurement was rounded or overwritten.

| RWKV / Qwen | Batch | RWKV Prefill / Decode | Qwen Prefill / Decode | Raw Prefill / Decode | Adjusted Prefill / Decode | Adjusted minima P / D |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 63,487 / 585 | 10,779 / 35.5 | 6.259513x / 16.491463x | 3.750151x / 9.880237x | 3.522779x / 9.850599x |
| 0.4B / 0.8B | B8 | 147,413 / 3,845 | 68,760 / 269 | 2.173516x / 14.304123x | 1.302180x / 8.569775x | 1.060506x / 8.543714x |
| 1.5B / 2B | B1 | 36,381 / 251 | 10,557 / 35.1 | 3.506888x / 7.135784x | 2.846405x / 5.791840x | 2.796494x / 5.770237x |
| 1.5B / 2B | B8 | 56,564 / 1,717 | 37,337 / 268 | 1.515276x / 6.400621x | 1.229891x / 5.195136x | 1.104974x / 5.181846x |
| 2.9B / 4B | B1 | 18,773 / 136 | 7,627 / 25.4 | 2.487282x / 5.336797x | 1.743291x / 3.740464x | 1.721029x / 3.733072x |
| 2.9B / 4B | B8 | 28,520 / 953 | 15,026 / 196 | 1.914915x / 4.860912x | 1.342129x / 3.406925x | 1.232327x / 3.388397x |
| 7.2B / 9B | B1 | 10,842 / 61.6 | 7,476 / 25.5 | 1.450215x / 2.416559x | 1.166019x / 1.942990x | 1.132183x / 1.941701x |
| 7.2B / 9B | B8 | 13,836 / 450 | 8,525 / 197 | 1.622909x / 2.276916x | 1.304871x / 1.830713x | 1.190438x / 1.829468x |

## Why RWKV Decode is this fast

This is expected for the verified route rather than a timing shortcut:

1. RWKV Decode updates a fixed-size recurrent state, so work per new token is
   independent of prompt length.
2. The exact-card token step is captured by CUDA Graph, removing most Python,
   Transformers-dispatch and kernel-launch overhead.
3. Recurrent/output/norm-mix work is fused and the recurrent cache is reused.
4. RWKV's 65,536-token vocabulary makes its final projection substantially
   smaller than Qwen3.5's 248,320-token vocabulary.

The runner performs the full Prefill and then 128/512 real Decode steps. Prompt,
cache handoff, greedy tokens and `generate()` are checked; no model work is
skipped.

## Correctness and exact-card optimization evidence

- `correctness/`: graph-route prompt/cache/greedy/generate checks for all four
  models at B1/B8; the 7.2B B8/P2048 continuation path has its own no-Prefill-
  Graph log.
- `7p2_accum_ab_short.jsonl`: 7.2B B1/B8 P128/P512 accumulation A/B rows.
- `7p2_accum_ab_p2048_b1.jsonl`: 7.2B B1/P2048 A/B rows.
- `7p2_accum_ab_p2048_b8_chunked.jsonl`: memory-bounded 7.2B B8/P2048 A/B;
  all six rows pass, block accumulation gains `1.422952x-1.436970x`, prompt
  cosine is at least `0.99999654`, Decode cosine at least `0.99999481`, greedy
  tokens match, and peak allocation is about 18,808 MiB.

## Artifact map

- [candidate.jsonl](candidate.jsonl): 48 full-precision RWKV rows.
- [qwen_reference.jsonl](qwen_reference.jsonl): 48 full-precision official Qwen rows.
- [main_table.jsonl](main_table.jsonl): all 96 validated full-precision rows.
- `validation.json`: fail-closed matrix validator output.
- `qwen_official_fast_path_status.json`: official FLA/causal-conv contract.
- `summary.json` and `summary.md`: cell-level adjusted-P/D acceptance.
- `runtime-lock.json`, `environment.json`, `pip-freeze.txt`, `system.csv`:
  pinned runtime evidence.
- `model_hashes.sha256`, `extension_build_manifest.json`: model and extension
  provenance.
- `formal.log`, `exit_code.txt`: formal-run log and exit status.

The formal benchmark used repository commit
`1dc8be160f44639da18e72a92fbfbc2c6c49f34f`. The checked-in summarizer and
documentation may be from a later commit on the same PR.
