# RTX 3090 latest-checkpoint prefill/decode evidence

This directory is the reproducible evidence bundle for the strict RTX 3090
comparison between the latest RWKV-7 G1 checkpoints and the corresponding
official Qwen3.5 checkpoints. The run completed on 2026-08-12 and passed all
24 required speed cells plus all 15 precision/cache-handoff rows.

## Contract

- GPU: NVIDIA GeForce RTX 3090, `sm_86`, 24 GiB, driver `550.142`, 350 W.
- Candidate: RWKV-7 G1 0.4B / G1i 1.5B / 2.9B / 7.2B, fp16.
- Reference: Qwen3.5 0.8B / 2B / 4B / 9B, fp16, full FLA backend with the
  fused Triton short-convolution route required and verified for every row.
- Matrix: batch `1/8`, prompt `128/512/2048`, decode `128`, chunk `512`,
  two warmups and five timed repetitions.
- Source commit: `4ca152ba88cda1f7b7b08da6f3af1458ee3092a5`.
- Strict adjusted-prefill gate:
  `(RWKV prefill tok/s / Qwen prefill tok/s) *
  (RWKV active parameters / Qwen active parameters) >= 1.0` for every cell.

The comparison is a hardware-local HF execution benchmark. It does not claim
that an engine-speed result establishes model-quality superiority.

## Result

Overall status: **PASS** (`24/24` speed cells, no missing or red cells).

| Pair | Raw prefill minimum | Adjusted prefill minimum | Passing cells |
|---|---:|---:|---:|
| G1d 0.4B / Qwen3.5 0.8B | 1.996429x | 1.196085x | 6/6 |
| G1i 1.5B / Qwen3.5 2B | 1.393044x | 1.130680x | 6/6 |
| G1i 2.9B / Qwen3.5 4B | 1.480804x | 1.037869x | 6/6 |
| G1i 7.2B / Qwen3.5 9B | 1.531508x | 1.231381x | 6/6 |

Across the full matrix, raw prefill ratio is `1.393044x` minimum and
`1.856646x` median. Adjusted prefill ratio is `1.037869x` minimum and
`1.351562x` median. Raw decode ratio is `2.105217x` minimum and `4.361090x`
median; adjusted decode is `1.692661x` minimum and `3.301953x` median.
Adjusted end-to-end ratio is `1.507863x` minimum and `2.973440x` median.

The narrowest cell is G1i 2.9B / Qwen3.5 4B at batch 8, prompt 128:
RWKV `9860.980` prefill tok/s versus Qwen `6659.207` prefill tok/s, for
`1.480804x` raw and `1.037869x` active-parameter-adjusted prefill.

## Correctness and routing

The 15-row fp32-oracle comparison passed with minimum prompt cosine
`0.99999344` and minimum first-decode-after-prefill cosine `0.99999356`.
All 15 prompt greedy tokens and all 15 post-prefill greedy tokens match. The
rows also verify that the optimized accumulation route is active only on its
approved RTX 3090 shapes, including direct and chunked prefill handoff.

The exact-card policy uses the validated row-32 scan shapes and a narrowly
scoped fp16-accumulation allowlist. Other Ampere cards retain their conservative
defaults until equivalent exact-card evidence exists.

## Files

- `results.jsonl`: 24 candidate and 24 reference benchmark rows.
- `summary.json` / `summary.md`: analyzer output and strict gate result.
- `correctness.jsonl` / `correctness_summary.json`: oracle, greedy-token, and
  cache-handoff validation.
- `environment.json`, `rwkv_runtime.json`, `qwen_runtime.json`, `system.csv`:
  exact software and hardware identity.
- `model_hashes.txt`: checkpoint file hashes.
- `source_commit.txt`: benchmarked source revision.

## Reproduce

Run from the repository root after setting local checkpoint and interpreter
paths:

```bash
OUT_DIR=/tmp/3090_latest_pd \
SOURCE_COMMIT="$(git rev-parse HEAD)" \
RWKV_PYTHON_BIN=/path/to/rwkv-python \
QWEN_PYTHON_BIN=/path/to/qwen-python \
RWKV_04B=/path/to/rwkv-g1d-0.4b \
RWKV_15B=/path/to/rwkv-g1i-1.5b \
RWKV_29B=/path/to/rwkv-g1i-2.9b \
RWKV_72B=/path/to/rwkv-g1i-7.2b \
QWEN_08B=/path/to/qwen3.5-0.8b \
QWEN_2B=/path/to/qwen3.5-2b \
QWEN_4B=/path/to/qwen3.5-4b \
QWEN_9B=/path/to/qwen3.5-9b \
bash bench/run_3090_adjusted_prefill_pd.sh
```

The runner fails closed on GPU/runtime mismatch, missing checkpoints, an
unexpected Qwen backend/fusion route, incomplete coverage, correctness failure,
or any adjusted-prefill cell below `1.0x`.
