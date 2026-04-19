# RTX 3090 max-performance prefill/decode evidence

This immutable evidence bundle supersedes the earlier 2026-08-12 RTX 3090
latest-checkpoint artifact for the exact B1/B8 dense-FP16 profiles below. It
adds exact-shape FP16 GEMM accumulation and 2.9B scan-tile tuning, then reruns
every RWKV and Qwen row on the same card. The run passes all 24 speed cells and
all 25 prompt/cache-handoff correctness rows.

## Contract

- GPU: NVIDIA GeForce RTX 3090, `sm_86`, 24 GiB, driver `550.142`, 350 W.
- Candidate: RWKV-7 G1d 0.4B and G1i 1.5B / 2.9B / 7.2B, fp16.
- Reference: Qwen3.5 0.8B / 2B / 4B / 9B, fp16, with FLA Gated DeltaNet,
  Triton causal convolution, live fused bindings, and the full-fused contract
  required for every row.
- Matrix: batch `1/8`, prompt `128/512/2048`, decode `128`, chunk `512`, two
  warmups and five timed repetitions.
- Source commit: `5581072627c761ef18a094a8e314cdcc103dbf16`.
- Strict adjusted-prefill gate:
  `(RWKV prefill tok/s / Qwen prefill tok/s) *
  (RWKV active parameters / Qwen active parameters) >= 1.0` in every cell.

This is a hardware-local HF execution comparison. It does not use an engine
speed result as evidence of model-quality superiority.

## Final result

Overall status: **PASS** (`24/24` speed cells; no missing or red cells).

| Pair | Raw prefill minimum | Adjusted prefill minimum | Passing cells |
|---|---:|---:|---:|
| G1d 0.4B / Qwen3.5 0.8B | 2.048826x | 1.227477x | 6/6 |
| G1i 1.5B / Qwen3.5 2B | 1.644101x | 1.334453x | 6/6 |
| G1i 2.9B / Qwen3.5 4B | 2.056397x | 1.441292x | 6/6 |
| G1i 7.2B / Qwen3.5 9B | 1.531589x | 1.231446x | 6/6 |

Across all 24 cells, raw prefill is `1.531589x` minimum and `2.076170x`
median. Parameter-adjusted prefill is `1.227477x` minimum and `1.467758x`
median. Raw decode is `2.069838x` minimum and `4.524636x` median;
parameter-adjusted decode is `1.664218x` minimum and `3.433680x` median.

The narrowest adjusted cell is G1d 0.4B / Qwen3.5 0.8B at B8/P512: RWKV
reaches `78,949.489 tok/s` versus Qwen `38,534.012 tok/s`, or `2.048826x`
raw and `1.227477x` after active-parameter adjustment. Compared with the
previous formal artifact, the global adjusted-prefill minimum rises from
`1.037869x` to `1.227477x` and the median from `1.351562x` to `1.467758x`.

## Optimization evidence

Profiling G1i 2.9B B8/P128 attributes `52.2%` of instrumented prefill time to
the FFN and about `22%` to dense R/K/V and output projections; fused output
preparation is only `1.3%`. The selected route therefore keeps the recurrent
state in fp32 but enables global fp16 GEMM accumulation only for exact, tested
3090 model/batch/token tuples.

Alternating-order A/B on the former weakest 2.9B B8/P128 shape moves from
`9,876.0/9,881.6 tok/s` with fp32 GEMM accumulation to
`13,559.3/13,463.5 tok/s` with fp16 accumulation, about `1.368x`, with
unchanged `6,493.7 MiB` peak VRAM. A second exact-shape scan sweep promotes
block 64 for 2.9B B8/P128 and B8/P512, and block 32 for B1/P512. Their paired
end-to-end gains are approximately `4.8%-6.5%` for B8 and `2.7%` for B1/P512.
The adjacent 0.4B B1/P128 accumulation probe improves only `1.0079x` and
remains disabled.

## Correctness and isolation

All `25/25` fp32-oracle rows pass. Minimum prompt cosine is `0.99999321` and
minimum first-decode-after-prefill cosine is `0.99999356`; all 25 prompt
greedy tokens and all 25 post-prefill greedy tokens match. The rows cover
every direct or chunk-carried shape that can select the promoted accumulation
route. Policy tests keep adjacent Ampere products and unlisted shapes on their
conservative defaults.

## Files

- `results.jsonl`: 24 candidate plus 24 reference benchmark rows.
- `summary.json` / `summary.md`: analyzer output and strict gate result.
- `correctness.jsonl` / `correctness_summary.json`: fp32-oracle, greedy-token,
  and cache-handoff validation.
- `probes/`: pre-promotion hotspot, alternating A/B, and scan-tile evidence.
- `environment.json`, runtime JSON files, `system.csv`, `model_hashes.txt`, and
  `source_commit.txt`: pinned hardware, software, checkpoint, and source data.

## Reproduce

Run from the repository root after assigning local interpreters and all eight
checkpoint paths:

```bash
OUT_DIR=/tmp/3090_maxperf \
SOURCE_COMMIT="$(git rev-parse HEAD)" \
BENCHMARK_MATRIX=qwen35_3090_g1i_maxperf_20260812 \
RWKV_PYTHON_BIN=/path/to/rwkv-python \
QWEN_PYTHON_BIN=/path/to/qwen-python \
RWKV_04_MODEL=/path/to/rwkv-g1d-0.4b \
RWKV_15_MODEL=/path/to/rwkv-g1i-1.5b \
RWKV_29_MODEL=/path/to/rwkv-g1i-2.9b \
RWKV_72_MODEL=/path/to/rwkv-g1i-7.2b \
QWEN_08_MODEL=/path/to/qwen3.5-0.8b \
QWEN_2_MODEL=/path/to/qwen3.5-2b \
QWEN_4_MODEL=/path/to/qwen3.5-4b \
QWEN_9_MODEL=/path/to/qwen3.5-9b \
bash bench/run_3090_adjusted_prefill_pd.sh
```

The runner fails closed on a GPU/runtime mismatch, missing checkpoint,
unexpected Qwen route, incomplete coverage, any adjusted-prefill cell below
`1.0x`, or any precision/cache-handoff failure.
