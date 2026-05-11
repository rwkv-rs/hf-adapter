# RWKV-7 vs Qwen3.5: Unified HF Fast-Path Benchmark

Updated: **2026-08-16**. [中文版](QWEN35_SPEED_COMPARISON_ZH.md)

## Current status

The previous cross-card Qwen3.5 baselines are **not eligible for the unified
main table**. They mixed causal-convolution implementations, runtime versions,
and RWKV CUDA-Graph settings. In particular, a row that requested FLA could
still use a non-official convolution route or a slow Transformers fallback.
Those artifacts remain reproducibility history only.

The replacement main table is populated card by card. RTX 3090 completed
`qwen35_3090_paired_pd_v2` on 2026-08-16: a SHA-locked 48-row optimized Qwen
reference is joined to 48 fresh exact-card RWKV rows, and raw plus
parameter-adjusted Prefill and Decode pass every cell. RTX 4090 completed the
superseding `qwen35_4090_paired_pd_v2` contract on 2026-08-15: a SHA-locked
48-row optimized Qwen reference is joined to 48 fresh exact-card RWKV rows,
and raw plus parameter-adjusted Prefill and Decode pass every cell. RTX 5090 now has both the
immutable 48-row Qwen best-optimized HF reference and a same-runtime 48-row
RWKV candidate. “Same-runtime” here means the six validator package fields,
GPU and shape protocol match; the repository commits differ and the captures
were not interleaved. Their `qwen35_paired_decode_v1` join passes strict
parameter-adjusted Decode in 48/48 cells. This completes a Decode-only
subtable, not the full Prefill/Decode main table: the frozen reference combines
independently optimized axes and no Prefill or continuous-E2E gate is promoted.
RTX 4080 has also completed a same-runtime 36-cell paired P+D table over the
three model pairs that fit its 16 GiB capacity. Raw and parameter-adjusted
Prefill and Decode pass every cell. No backend fallback or legacy result is
merged into the current table.

Tesla V100 now also has a strict four-pair, 48-cell P+D result. Fresh RWKV
rows are joined to a SHA-locked Qwen reference from the same server and exact
runtime. Raw and parameter-adjusted Prefill and Decode pass all 48 cells; the
commits differ and the measurements were sequential, so this is not an
interleaved A/B or continuous-E2E result.

## Fixed protocol (`hf_fast_path_v1`)

| Axis | Required setting |
|---|---|
| GPU | One V100, RTX 3090, RTX 4080, RTX 4090 or RTX 5090; one card per run |
| Pairs | RWKV 0.4/1.5/2.9/7.2B vs Qwen3.5 0.8/2/4/9B |
| Precision | Dense FP16; quantization and MTP/speculative decode disabled |
| Batch | 1 and 8 |
| Prompt | 128, 512 and 2048 tokens |
| Decode | 128 and 512 tokens |
| Prefill chunk | 512 tokens |
| Timing | 3 warmups, 7 measured runs, median per cell |
| Qwen | Transformers FLA fast path plus official Dao-AILab `causal_conv1d` |
| RWKV performance lane | Exact-card `best_optimized_hf`; CUDA Graph and verified fusions enabled |

The four-pair matrix has `4 × 2 × 3 × 2 = 48` cells per model side. RTX 4080
uses the capacity-safe three-pair subset, `3 × 2 × 3 × 2 = 36` cells per side.

### Unified main-table status

| GPU | Rows | Qwen official fast path | RWKV performance lane | Adjusted Prefill > Qwen | Raw / adjusted Decode > Qwen | Evidence |
|---|---:|---|---|---:|---:|---|
| RTX 4090 | 48 Qwen + 48 RWKV; 48 joined P+D cells | 48/48 pass, fixed per-model Graph route | 48/48 `best_optimized_hf`; native Graph Decode | 48/48 | 48/48 / 48/48 | [strict paired P+D v2 artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 3090 | 48 Qwen + 48 RWKV; 48 joined P+D cells | 48/48 pass, fixed per-model Graph route | 48/48 `best_optimized_hf`; native Graph Decode | 48/48 | 48/48 / 48/48 | [strict paired P+D v2 artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | 36 Qwen + 36 RWKV; 36 joined P+D cells | 36/36 pass, no fallback | 36/36 `best_optimized_hf`; native Graph Decode | 36/36 | 36/36 / 36/36 | [strict paired P+D artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| Tesla V100 | 48 Qwen + 48 RWKV; 48 joined P+D cells | 48/48 pass, no fallback | 48/48 `best_optimized_hf`; native Graph Decode | 48/48 | 48/48 / 48/48 | [strict paired P+D artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 5090 | 48 Qwen + 48 RWKV; 48 joined Decode cells | 48/48 pass, no fallback | 48/48 `best_optimized_hf`; Decode Graph on | not gated | 48/48 telemetry / 48/48 strict | [paired Decode artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |

RTX 4090 clears the current optimized Qwen reference in every matched cell.
Raw Prefill minimum/median/maximum is `1.415206x/2.449410x/12.686421x`, and
parameter-adjusted Prefill is `1.148668x/1.695334x/7.600590x`. Raw Decode is
`1.276285x/1.770640x/3.116990x`, and parameter-adjusted Decode is
`1.026173x/1.323737x/1.867427x`. The weakest adjusted Decode cell is
7.2B/9B B8/P128/D128: `449` versus `352 tok/s`, a `+2.6173%`
margin. Eight 512-token FLA/native checks pass with exact greedy traces,
finite Decode logits and minimum prompt/final cosine `0.999992967`.

RTX 3090 also clears every matched cell. Raw Prefill minimum/median/maximum is
`1.574925x/2.182651x/8.428073x`, and parameter-adjusted Prefill is
`1.208324x/1.535161x/5.049362x`. Raw Decode is
`1.253926x/1.740275x/3.094400x`, and parameter-adjusted Decode is
`1.017763x/1.207730x/1.853893x`. The weakest adjusted Decode cell is
1.5B/2B B1/P128/D128: RWKV reaches `232 tok/s` versus Qwen at `185 tok/s`,
leaving a `+1.7763%` adjusted margin. Eight 512-token FLA/native checks pass
with exact greedy traces, finite Decode logits and minimum prompt/final cosine
`0.999987364`.

The RTX 5090 paired Decode subtable has parameter-adjusted
minimum/median/maximum `1.029966x/1.409279x/2.063849x`. The weakest cell is
0.4B/0.8B B1/P128/D128: RWKV reaches 1,125 tok/s versus Qwen at 654 tok/s and
clears the required RWKV rate by `+2.996552%`. Raw Decode also wins 48/48 at
minimum/median `1.373660x/1.903882x`, but that is subordinate telemetry rather
than the acceptance contract. This does not claim model quality, Prefill,
TTFT, continuous E2E, or cache-handoff latency.

RTX 4080 clears every one of the 36 cells on all four gates. Raw Prefill has
minimum/median/maximum `1.500014x/1.920082x/4.825638x`; raw Decode has
`1.302605x/1.723038x/3.065001x`. Parameter-adjusted Prefill has
`1.051333x/1.313931x/2.891099x`; parameter-adjusted Decode has
`1.022115x/1.190224x/1.836279x`. The weakest adjusted Decode cell is
0.4B/0.8B B8/P128/D128, where RWKV reaches 3,344 tok/s against Qwen's
1,960 tok/s.

Tesla V100 clears every one of the 48 cells on all four gates. Raw Prefill has
minimum/median/maximum `2.249335x/4.744253x/13.714267x`; raw Decode has
`1.393444x/2.398393x/4.662333x`. Parameter-adjusted Prefill has
`1.808536x/3.217214x/8.216385x`; parameter-adjusted Decode has
`1.120373x/1.617469x/2.793261x`. The weakest adjusted Decode cell is
7.2B/9B B8/P128/D128, where RWKV reaches `267 tok/s` against Qwen's
`191 tok/s`.

The corrected Qwen Decode medians on RTX 4090 are:

| Qwen3.5 | B1 | B8 |
|---|---:|---:|
| 0.8B | 407 tok/s | 2,252 tok/s |
| 2B | 212 tok/s | 1,302 tok/s |
| 4B | 84.5 tok/s | 518 tok/s |
| 9B | 50.7 tok/s | 331 tok/s |

The environment is also part of the result: all cards must use the same
Python, PyTorch+CUDA build, Transformers revision, FLA revision,
`causal-conv1d` revision, and repository commit. The artifact records the
runtime lock, `pip freeze`, Docker digest when present, repository commit, and
SHA256 hashes for model configs and safetensors.

### RTX 4090 frozen-reference paired Prefill/Decode v2

The 2026-08-15 artifact uses the current fastest correctness-passing Qwen
Graph route per model: StaticCache Inductor CUDA Graph for 0.8B/2B and raw
CUDA Graph for 4B/9B. RWKV uses Native Graph in every row, with exact-card
small-model B8 projection and compiled-FFN routes gated by full-layer
telemetry. The validator reads unrounded values, requires strict `> 1.0` for
all four throughput ratios in every cell, and reports 48/48 for all gates.
See the [README](../bench/4090_qwen35_paired_pd_v2_20260815/README.md),
[complete joined table](../bench/4090_qwen35_paired_pd_v2_20260815/paired_pd_table.jsonl),
and [validator result](../bench/4090_qwen35_paired_pd_v2_20260815/paired_validation.json).
This is an inference-speed result, not a model-quality or continuous-E2E claim.

### RTX 3090 frozen-reference paired Prefill/Decode v2

The 2026-08-16 artifact fixes one correctness-passing StaticCache Graph route
per Qwen model and uses RWKV Native Graph with the exact
`sm86_qwen_alignment` route profile. The validator reads unrounded values and
requires raw and parameter-adjusted Prefill and Decode to be strictly greater
than Qwen in every cell. All four gates pass 48/48; 8/8 independent
P2048/D512, 512-token FLA/native comparisons also pass. See the
[README](../bench/3090_qwen35_paired_pd_v2_20260816/README.md),
[RWKV candidate](../bench/3090_qwen35_paired_pd_v2_20260816/rwkv_candidate.jsonl),
[Qwen reference](../bench/3090_qwen35_paired_pd_v2_20260816/qwen_reference.jsonl),
[complete joined table](../bench/3090_qwen35_paired_pd_v2_20260816/paired_pd_table.jsonl),
and [validator result](../bench/3090_qwen35_paired_pd_v2_20260816/validation.json).
This is an inference-speed result, not a model-quality or continuous-E2E claim.

### RTX 4080 same-runtime paired Prefill/Decode v1

The 2026-08-14 artifact captures both sides from clean commit
`398277d94e1d1dc441af97dea0578b87fa072f74` in one locked runtime. Qwen 0.8B
and 2B use official fast operators with StaticCache Inductor CUDA Graph;
Qwen 4B uses the strongest 16 GiB-safe official DynamicCache module-call
route. RWKV uses native Graph with fail-closed exact-card route evidence.

| RWKV / Qwen | Batch | RWKV P tok/s | Qwen P tok/s | Raw P | RWKV D tok/s | Qwen D tok/s | Raw D |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 45,562 | 22,984 | `2.066x` | 618 | 310 | `1.998x` |
| 0.4B / 0.8B | B8 | 104,224 | 45,067 | `2.200x` | 3,344 | 1,711 | `1.960x` |
| 1.5B / 2B | B1 | 30,683 | 19,692 | `1.630x` | 207 | 154 | `1.345x` |
| 1.5B / 2B | B8 | 39,234 | 22,896 | `1.649x` | 1,360 | 960 | `1.417x` |
| 2.9B / 4B | B1 | 14,264 | 8,956 | `1.665x` | 108 | 63.4 | `1.702x` |
| 2.9B / 4B | B8 | 19,533 | 9,866 | `1.987x` | 729 | 422 | `1.727x` |

These are six-cell lane medians; the validator gates every unrounded cell.
All six P2048/D512 native-graph-versus-FLA probes preserve 512 greedy tokens
and finite Decode logits, with prompt/final minimum row cosine
`0.999990582/0.999993861`. Formal B8 timing replicates one prompt; only the
correctness probes use distinct prompts. This is not a model-quality or
continuous-E2E claim. See the
[`README`](../bench/4080_qwen35_paired_pd_v1_20260814/README.md),
[`paired table`](../bench/4080_qwen35_paired_pd_v1_20260814/paired_pd_table.jsonl),
and [`validation`](../bench/4080_qwen35_paired_pd_v1_20260814/paired_validation.json).

### Tesla V100 frozen-reference paired Prefill/Decode v1

The 2026-08-14 artifact covers all four model pairs, B1/B8,
P128/P512/P2048 and D128/D512. RWKV uses Native Graph; every B1 lane requires
the real SM70 W/A/G/V extension and proves every eligible layer before graph
capture. Qwen uses the official FLA operator route plus StaticCache raw CUDA
Graph Decode; Qwen9 locks math-only SDPA.

| RWKV / Qwen | Batch | RWKV P tok/s | Qwen P tok/s | Raw P | RWKV D tok/s | Qwen D tok/s | Raw D |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 17,822 | 4,117 | `4.384x` | 434 | 111 | `3.909x` |
| 0.4B / 0.8B | B8 | 56,270 | 4,140 | `13.457x` | 1,784 | 688 | `2.596x` |
| 1.5B / 2B | B1 | 11,167 | 3,672 | `3.055x` | 230 | 83.3 | `2.762x` |
| 1.5B / 2B | B8 | 20,861 | 3,816 | `5.467x` | 841 | 517 | `1.630x` |
| 2.9B / 4B | B1 | 7,066 | 1,382 | `5.109x` | 124 | 46.0 | `2.696x` |
| 2.9B / 4B | B8 | 10,711 | 1,505 | `7.144x` | 536 | 275 | `1.950x` |
| 7.2B / 9B | B1 | 3,758 | 1,174 | `3.142x` | 56.1 | 31.2 | `1.801x` |
| 7.2B / 9B | B8 | 4,708 | 1,283 | `3.653x` | 267 | 164 | `1.632x` |

These are six-cell lane medians; the validator gates every unrounded cell.
All eight P2048/D512 FLA/native probes preserve 512 greedy tokens and finite
Decode logits, with prompt/final minimum row cosine
`0.999994457/0.999991894`. The additional 7.2B/B8/P128 native-eager versus
native-graph closure also passes. Formal B8 timing replicates one prompt; only
the correctness probes use distinct prompts. This is an engine-speed result,
not model quality or continuous E2E. See the
[`README`](../bench/v100_qwen35_paired_pd_v1_20260814/README.md),
[`paired table`](../bench/v100_qwen35_paired_pd_v1_20260814/paired_pd_table.jsonl),
and [`validation`](../bench/v100_qwen35_paired_pd_v1_20260814/paired_validation.json).

### RTX 5090 frozen-reference paired Decode v1

The 2026-08-13 paired artifact joins the unchanged Qwen reference with a clean
commit RWKV capture over all four model pairs, B1/B8,
P128/P512/P2048 and D128/D512. The validator uses unrounded raw throughput and
requires

```text
(RWKV Decode tok/s / Qwen Decode tok/s)
* (RWKV active parameters / Qwen active parameters) > 1.0
```

for every cell. It reports 48/48 strict passes, zero errors and
`paired_decode_table_eligible=true`; `continuous_e2e_eligible=false` remains
explicit.

Reference and candidate were captured separately at different repository
commits, not as an interleaved A/B. The validator proves that the six package
runtime fields, exact GPU and shape protocol match. Formal B8 timing replicates
one prompt eight times; only the independent correctness probes use eight
distinct prompts.

| RWKV / Qwen | Batch | Cells | Adjusted Decode minimum | Adjusted Decode median |
|---|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 6 | `1.029966x` | `1.210827x` |
| 0.4B / 0.8B | B8 | 6 | `1.040730x` | `1.225006x` |
| 1.5B / 2B | B1 | 6 | `1.261697x` | `1.369630x` |
| 1.5B / 2B | B8 | 6 | `1.114947x` | `1.226407x` |
| 2.9B / 4B | B1 | 6 | `1.708151x` | `1.801785x` |
| 2.9B / 4B | B8 | 6 | `1.099272x` | `1.196935x` |
| 7.2B / 9B | B1 | 6 | `1.429633x` | `1.480590x` |
| 7.2B / 9B | B8 | 6 | `1.266346x` | `1.344888x` |

The two narrow SM120 B8 A/B routes improve 0.4B/1.5B Decode by
`1.865301x/1.492719x` with exact 512-token greedy traces. All eight independent
native-graph-versus-FLA checks preserve 512 greedy tokens and finite logits;
their prompt/final cosine minima are `0.999981999/0.999970913`. The formal
candidate rows are in
[`rwkv_candidate.jsonl`](../bench/5090_qwen35_paired_decode_v1_20260813/rwkv_candidate.jsonl),
the complete joined rows in
[`paired_decode_table.jsonl`](../bench/5090_qwen35_paired_decode_v1_20260813/paired_decode_table.jsonl),
and the fail-closed result in
[`paired_validation.json`](../bench/5090_qwen35_paired_decode_v1_20260813/paired_validation.json).

This result is parameter-adjusted Decode only. Raw Decode 48/48 is retained as
supporting telemetry; no model-quality, Prefill, TTFT, continuous-E2E or
cache-handoff-latency conclusion is drawn. Full evidence:
[`bench/5090_qwen35_paired_decode_v1_20260813/`](../bench/5090_qwen35_paired_decode_v1_20260813/README.md).

### RTX 5090 Qwen-only best-optimized HF reference v2

The 2026-08-13 artifact completes all 48 dense-FP16 reference cells for
Qwen3.5 0.8B/2B/4B/9B on one RTX 5090. Every row verifies the official
Transformers FLA operators and Dao-AILab `causal_conv1d`, with no fallback.
Prefill uses the eager DynamicCache official path. Decode uses one fixed
correctness-passing StaticCache CUDA Graph route per model: Inductor
`max-autotune` for 0.8B/2B and raw CUDA Graph for 4B/9B. The raw graph wrapper
is a repository benchmark optimization, not an official Qwen graph path.

This is an `independent_best_prefill_and_decode` reference envelope, not a
continuous end-to-end cache route, TTFT result, or cache-handoff latency. By
itself this source artifact contains no RWKV candidate and cannot produce
ratios. The paired Decode v1 artifact above binds these exact bytes by SHA256
and adds a separately captured runtime-aligned candidate without rewriting the
reference. Same-cache eager-versus-Graph cosine is at least `0.9999860525` with
finite logits and full greedy equality. Cross-cache cosine is informational;
finite traces, full greedy equality and prefill-next-token equality pass.

Rows are ordered by model size, GPU and B1/B8. Throughputs are medians over the
six Prompt/Decode cells in each model/batch slice. B8 Decode is aggregate
throughput across eight sequences. Values at or above 100 tok/s use zero
decimals and lower values use one decimal; the artifact retains full precision
and all seven timing samples.

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

For historical scale only, the closest 2026-08-11 RTX 5090 module-call rows
can be aligned at D128 by taking the median over P128/P512/P2048:

| Qwen3.5 | Batch | Historical module-call tok/s | New optimized tok/s | Historical ratio |
|---|---:|---:|---:|---:|
| 0.8B | B1 | 56.7 | 584.4 | 10.31x |
| 0.8B | B8 | 429.4 | 3,371.2 | 7.85x |
| 2B | B1 | 56.7 | 334.0 | 5.89x |
| 2B | B8 | 434.0 | 2,113.5 | 4.87x |
| 4B | B1 | 41.3 | 122.5 | 2.96x |
| 4B | B8 | 317.4 | 751.2 | 2.37x |
| 9B | B1 | 41.7 | 80.1 | 1.92x |
| 9B | B8 | 318.6 | 528.8 | 1.66x |

This is not a controlled same-runtime A/B. The historical rows used
`module_call` plus `DynamicCache`, the repository `fla_triton` convolution
route, PyTorch 2.11 and 2/5 warmup/runs; the new artifact uses official
`causal_conv1d`, PyTorch 2.8, 3/7 warmup/runs and fixed StaticCache Graph
routes. The ratios describe the practical baseline change, not an isolated
Graph or kernel speedup.

Full evidence:
[`bench/5090_qwen35_best_optimized_hf_v1_20260813/`](../bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md).

### Qwen row acceptance

Every Qwen row must satisfy all of the following:

```text
status=pass
qwen_fast_path_available=true
qwen_fast_path_verified=true
qwen_full_fused_contract_pass=true
qwen_causal_conv1d_importable=true
qwen_conv_backend_effective=causal_conv1d
qwen_force_torch=false
```

The runner now enforces the requested convolution backend against the live
operators in every loaded Qwen GatedDeltaNet layer. Environment, binding and
result-row checks are fail-closed. RTX 5090 is not allowed to switch to the
repository `fla_triton` convolution route: if the official path cannot pass on
SM120, the card is recorded as **SM120 official HF fast path unverified** and
is excluded from the unified main table.

### RWKV best-optimized row acceptance

The main comparison uses:

```bash
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_MODEL_BACKEND=native_graph
```

The runner requires `optimization_lane=best_optimized_hf`,
`rwkv_optimization_contract=exact_card_best_optimized_hf`, and an effective
`native_graph` Decode backend. Exact-card graph/fusion and block-accumulation
routes must be recorded in every row and pass prompt/cache/greedy correctness.
For 7.2B B8/P2048 only, Prefill Graph is disabled to remain within 24 GiB;
Decode remains graphed. The no-Graph `native_jit` result is never mixed into
this primary performance lane.

### Reproduce one card

Set the eight local model directories, establish or consume one runtime lock,
then run the same entry point on each card:

```bash
export GPU_MODEL=4090
export OUT_DIR=/path/to/hf-fast-path-v1-4090
export PYTHON_BIN=/path/to/locked-python
export RUNTIME_LOCK=/path/to/hf-fast-path-v1-runtime-lock.json
export FLA_SOURCE_COMMIT=2e38c1fab332174d056928feaf29f8c5fd5ac550
export CAUSAL_CONV1D_SOURCE_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223

export RWKV_04_MODEL=/models/rwkv-0.4b
export RWKV_15_MODEL=/models/rwkv-1.5b
export RWKV_29_MODEL=/models/rwkv-2.9b
export RWKV_72_MODEL=/models/rwkv-7.2b
export QWEN_08_MODEL=/models/Qwen3.5-0.8B
export QWEN_2_MODEL=/models/Qwen3.5-2B
export QWEN_4_MODEL=/models/Qwen3.5-4B
export QWEN_9_MODEL=/models/Qwen3.5-9B

bash bench/run_hf_fast_path_v1.sh
```

Use `WRITE_RUNTIME_LOCK=/path/to/lock.json` instead of `RUNTIME_LOCK` only on
the first card that establishes the lock. The script runs Qwen first; if its
official fast path fails, RWKV is not run and no `main_table.jsonl` is created.
Build both extensions from these exact revisions with
`bench/build_hf_fast_path_v1_extensions.sh`; it requires a CUDA developer image
and forces `TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"`.

## Parameter accounting

The model names are release tiers. Active parameter counts recorded by
benchmark telemetry are shown in billions, rounded to three decimal places:

| Model pair (RWKV / Qwen3.5) | RWKV active params | Qwen active params |
|---|---:|---:|
| 0.4B / 0.8B | `0.451B` | `0.752B` |
| 1.5B / 2B | `1.527B` | `1.882B` |
| 2.9B / 4B | `2.948B` | `4.206B` |
| 7.2B / 9B | `7.199B` | `8.954B` |

- **Raw speed ratio** = RWKV tok/s ÷ Qwen tok/s. This is the throughput seen
  directly by the user.
- **Parameter-size-adjusted speed ratio** = raw speed ratio × RWKV active
  parameters ÷ Qwen active parameters. The exact counts are retained in the
  artifacts; the table shows them directly.
- Example: the promoted RTX 4090 0.4B/0.8B B8 group has median raw Prefill
  `2.173516x` and `1.302180x` after active-parameter adjustment.

## Historical non-unified NVIDIA evidence

> The following table is retained for audit and regression context. It mixes
> older runtime and backend contracts and is **not** the `hf_fast_path_v1`
> unified main table. Its speed ratios must not be presented as the new
> 3090/4090/5090 comparison.

The table lists every GPU, model pair, and batch combination in the promoted
optimized-Qwen evidence. Rows are ordered by RWKV model size, GPU, then B1/B8.
`RWKV P / D tok/s` and `Qwen P / D tok/s` are separately computed medians over
the declared cells. Throughput at or above 100 tok/s is shown with zero decimal
places; lower throughput is shown with one decimal place. `Raw P / D` and
`Adjusted P / D` remain medians of matched cell-level ratios, so they need not
equal a division of the displayed throughput medians. The RTX 4090 rows below
have been refreshed from the promoted unified artifact; other rows retain their
historical contracts.

These rounded table cells are display values only; no original Prefill or
Decode measurement is rounded or overwritten. Every Evidence link leads to
the corresponding full-precision artifact. For the promoted RTX 4090 matrix,
the original RWKV rows are in [rwkv_candidate.jsonl](../bench/4090_qwen35_paired_pd_v2_20260815/rwkv_candidate.jsonl),
the original Qwen rows are in [qwen_reference.jsonl](../bench/4090_qwen35_paired_pd_v2_20260815/qwen_reference.jsonl),
and the complete 48-cell join is in [paired_pd_table.jsonl](../bench/4090_qwen35_paired_pd_v2_20260815/paired_pd_table.jsonl).
Use the `*_tokps_total_raw` fields for the full-precision
original throughput.

The new RTX 5090 Qwen-only values are not substituted into these historical
joined rows. Their protocols remain immutable; the separately captured,
runtime-aligned candidate is reported only in the Decode-v1 subtable above.

For RTX 4080, the stricter cell-level gate now passes **36/36 adjusted
Prefill cells and 36/36 adjusted Decode cells**; the full-matrix minima are
`1.068520x / 1.140700x`.

For RTX 4090, the latest strict gate passes **48/48 raw and adjusted Prefill
cells and 48/48 raw and adjusted Decode cells**; the adjusted minima are
`1.148668x / 1.026173x`.

| GPU | Model pair | Batch | Scope | RWKV active params | Qwen active params | RWKV P / D tok/s | Qwen P / D tok/s | Raw P / D | Adjusted P / D | Evidence |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 3090 | 0.4B / 0.8B | B1 | 3 cells | 0.451B | 0.752B | **29,368 / 293** | **7,155 / 26.5** | **4.10x / 11.05x** | **2.46x / 6.62x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 0.4B / 0.8B | B8 | 3 cells | 0.451B | 0.752B | **78,949 / 1,692** | **32,678 / 213** | **2.47x / 7.93x** | **1.48x / 4.75x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6 cells, all pass | 0.451B | 0.752B | **45,538 / 492** | **24,889 / 100** | **1.83x / 4.91x** | **1.10x / 2.94x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6 cells, all pass | 0.451B | 0.752B | **103,571 / 3,206** | **50,004 / 768** | **1.98x / 4.17x** | **1.19x / 2.50x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B1 | 6 cells, all pass | 0.451B | 0.752B | **62,950 / 788** | **9,311 / 407** | **7.19x / 1.94x** | **4.31x / 1.16x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6 cells, all pass | 0.451B | 0.752B | **146,273 / 4,839** | **65,924 / 2,252** | **2.25x / 2.15x** | **1.35x / 1.29x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 3 cells | 0.451B | 0.752B | **58,105 / 1,121** | **15,886 / 56.7** | **3.86x / 19.79x** | **2.31x / 11.85x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 3 cells | 0.451B | 0.752B | **206,364 / 3,432** | **93,886 / 429** | **2.24x / 7.99x** | **1.34x / 4.79x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527B | 1.882B | **10,426 / 151** | **3,702 / 25.6** | **2.82x / 5.91x** | **2.29x / 4.80x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527B | 1.882B | **20,729 / 817** | **3,833 / 155** | **5.41x / 5.27x** | **4.39x / 4.28x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 1.5B / 2B | B1 | 3 cells | 1.527B | 1.882B | **17,641 / 164** | **8,529 / 28.5** | **2.12x / 5.75x** | **1.72x / 4.67x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 3 cells | 1.527B | 1.882B | **29,163 / 985** | **16,416 / 220** | **1.66x / 4.47x** | **1.34x / 3.63x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6 cells, all pass | 1.527B | 1.882B | **30,858 / 194** | **19,871 / 102** | **1.55x / 1.90x** | **1.26x / 1.55x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6 cells, all pass | 1.527B | 1.882B | **38,144 / 1,356** | **21,602 / 765** | **1.76x / 1.77x** | **1.43x / 1.44x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 1.5B / 2B | B1 | 6 cells, all pass | 1.527B | 1.882B | **36,222 / 349** | **9,220 / 212** | **4.02x / 1.65x** | **3.26x / 1.34x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6 cells, all pass | 1.527B | 1.882B | **57,929 / 1,909** | **36,953 / 1,302** | **1.57x / 1.47x** | **1.27x / 1.19x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6 cells | 1.527B | 1.882B | **10,770 / 690** | **8,239 / 269** | **1.33x / 2.62x** | **1.08x / 2.13x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 3 cells | 1.527B | 1.882B | **33,698 / 547** | **15,795 / 56.7** | **2.16x / 9.63x** | **1.75x / 7.82x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 3 cells | 1.527B | 1.882B | **82,339 / 2,061** | **50,353 / 434** | **1.43x / 4.77x** | **1.16x / 3.87x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 3090 | 2.9B / 4B | B1 | 3 cells | 2.948B | 4.206B | **11,774 / 88.7** | **5,657 / 19.2** | **2.08x / 4.61x** | **1.46x / 3.23x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 3 cells | 2.948B | 4.206B | **15,776 / 596** | **7,094 / 151** | **2.14x / 3.96x** | **1.50x / 2.78x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6 cells, all pass | 2.948B | 4.206B | **14,276 / 103** | **8,819 / 62.8** | **1.75x / 1.63x** | **1.22x / 1.15x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6 cells, all pass | 2.948B | 4.206B | **19,517 / 729** | **9,824 / 416** | **1.99x / 1.75x** | **1.40x / 1.23x** | [4080 all P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 2.9B / 4B | B1 | 6 cells, all pass | 2.948B | 4.206B | **19,115 / 191** | **6,946 / 84.5** | **2.78x / 2.27x** | **1.95x / 1.59x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6 cells, all pass | 2.948B | 4.206B | **28,183 / 948** | **14,458 / 518** | **1.94x / 1.83x** | **1.36x / 1.28x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 3 cells | 2.948B | 4.206B | **21,787 / 309** | **11,795 / 41.3** | **1.87x / 7.49x** | **1.31x / 5.25x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 3 cells | 2.948B | 4.206B | **37,326 / 1,247** | **22,253 / 317** | **1.69x / 3.92x** | **1.19x / 2.75x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 3090 | 7.2B / 9B | B1 | 3 cells | 7.199B | 8.954B | **5,764 / 46.4** | **3,616 / 19.7** | **1.63x / 2.35x** | **1.31x / 1.89x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 3 cells | 7.199B | 8.954B | **6,633 / 342** | **4,156 / 164** | **1.60x / 2.08x** | **1.28x / 1.67x** | [3090 maxperf](../bench/3090_g1i_qwen35_maxperf_20260812/README.md) |
| RTX 4090 | 7.2B / 9B | B1 | 6 cells, all pass | 7.199B | 8.954B | **10,736 / 85.2** | **6,801 / 50.7** | **1.58x / 1.68x** | **1.27x / 1.35x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6 cells, all pass | 7.199B | 8.954B | **13,658 / 449** | **8,434 / 331** | **1.62x / 1.36x** | **1.30x / 1.09x** | [4090 paired P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 3 cells | 7.199B | 8.954B | **14,876 / 146** | **10,652 / 41.7** | **1.42x / 3.50x** | **1.14x / 2.81x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 3 cells | 7.199B | 8.954B | **19,624 / 867** | **12,262 / 319** | **1.54x / 2.72x** | **1.24x / 2.19x** | [5090 latest](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |

This complete table preserves every promoted GPU/model/batch result and makes
the raw and parameter-size-adjusted ratios directly comparable.

### Historical RTX 3090 latest-checkpoint strict gate (2026-08-12)

The latest RTX 3090 artifact uses RWKV-7 g1d 0.4B and 2026-08-05 g1i
1.5B/2.9B/7.2B against official Qwen3.5 0.8B/2B/4B/9B. It checks every
B1/B8 and P128/P512/P2048 cell independently at D128. All `24/24` Qwen rows
verify FLA, Triton causal convolution, live fused bindings and the full-fused
contract.

The strict prefill gate passes `24/24`: raw Prefill minimum/median is
`1.531589x/2.076170x`, while parameter-adjusted Prefill minimum/median is
`1.227477x/1.467758x`. Raw Decode minimum/median is
`2.069838x/4.524636x`, and adjusted Decode minimum/median is
`1.664218x/3.433680x`. The narrowest adjusted cell is 0.4B/0.8B B8/P512,
where RWKV delivers `78,949 tok/s` versus Qwen `38,534 tok/s`, or
`1.227477x` after parameter adjustment.

The exact-shape FP16-accumulation oracle passes `25/25` direct and
chunk-carried prompt/cache-handoff rows at cosine `>=0.9999` with exact greedy
tokens. The promoted route is restricted to exact RTX 3090 model, batch and
token-block shapes. See the
[immutable evidence](../bench/3090_g1i_qwen35_maxperf_20260812/README.md).

### RTX 4090 current optimized-Qwen strict gate

The promoted RTX 4090 v2 artifact compares RWKV-7 g1d 0.4B and g1i
1.5B/2.9B/7.2B with the current optimized Qwen3.5 0.8B/2B/4B/9B reference
across B1/B8, P128/P512/P2048 and D128/D512. Qwen fixes one
correctness-passing StaticCache Graph route per model; RWKV remains on Native
Graph throughout.

All four cell-level gates pass `48/48`. Adjusted Prefill global
minimum/median is `1.148668x/1.695334x`; adjusted Decode is
`1.026173x/1.323737x`. Raw Prefill/Decode minima are `1.415206x/1.276285x`.
Eight independent P2048/D512, 512-step FLA/native probes preserve exact greedy
tokens and finite Decode logits with cosine `>=0.999992967`. See the
[immutable evidence](../bench/4090_qwen35_paired_pd_v2_20260815/README.md).

### Historical RTX 5090 latest-checkpoint gate (2026-08-11)

The historical 2026-08-11 RTX 5090 rows use RWKV-7 g1d 0.4B plus the
2026-08-05 g1i
1.5B/2.9B/7.2B checkpoints against official Qwen3.5 0.8B/2B/4B/9B. All 24
Qwen reference cells verify FLA, Triton causal convolution, live fused
bindings, and the full-fused contract.

Unlike the row medians above, the strict gate checks every B1/B8 and
P128/P512/P2048 cell independently. All `24/24` cells pass: raw Prefill has
minimum/median `1.347871x/1.819072x`, and parameter-adjusted Prefill has
minimum/median `1.072987x/1.317515x`. Raw Decode has minimum/median
`2.710952x/6.104568x`, while parameter-adjusted Decode has minimum/median
`2.179692x/4.330813x`.

The 0.4B/B1/P2048 candidate reaches `61,344 tok/s`, `2.2495x` its prior
candidate row. The graph-versus-eager P2048 oracle passes `8/8` model/batch
rows with prompt/post-cache-handoff cosine minima
`0.99999988/0.99999994` and exact greedy tokens. Removing the negative 7.2B
stacked-RKV route lowers its candidate peak from `17.4-18.6 GiB` to
`14.3-15.5 GiB`. See the
[immutable evidence](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md).

### Apple M5: complete target-only W4 comparison

Apple MLX W4 is shown separately so that backend and precision remain
consistent within each table. Concrete throughput columns are aggregate tok/s
medians with the same `>=100`: zero decimals, `<100`: one decimal rule.

| Model pair | Batch / shape | RWKV active params | Qwen active params | RWKV P / D tok/s | Qwen P / D tok/s | Raw P / D | Adjusted P / D | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.4B / 0.8B | B8, cold, P512 chars/D64 | 0.451B | 0.752B | **11,650 / 992** | **5,702 / 487** | **2.04x / 2.04x** | **1.22x / 1.22x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| 1.5B / 2B | B1, P512 chars/D64 | 1.527B | 1.882B | **2,126 / 129** | **1,273 / 89.9** | **1.67x / 1.44x** | **1.36x / 1.17x** | [M5 B1](../bench/apple_bsz1_active_m5_20260715/README.md) |
| 1.5B / 2B | B8, cold, P512 chars/D64 | 1.527B | 1.882B | **2,249 / 186** | **1,601 / 132** | **1.41x / 1.40x** | **1.14x / 1.14x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |

## AMD and other hardware

Beyond the NVIDIA and Apple comparisons above, the repository covers AMD
ROCm, Turing/Ampere/Hopper, and dedicated accelerator backends.

| Platform | Implemented and validated capabilities | Entry point |
|---|---|---|
| AMD Navi 31 / `gfx1100` | Native HF, cache, chunked Prefill, PEFT, BF16 Trainer, fused Decode from 0.1B through 13.3B, and 40/40 output-head W8/W4 Decode rows | [AMD ROCm validation](validation/AMD_ROCM_HF_VALIDATION.md) |
| AMD MI series, `gfx1101/gfx1102` | Portable ROCm/HIP path, architecture detection, and portable dispatch | [Hardware matrix](HARDWARE_MATRIX.md) |
| Apple M5 | Target-only W4 comparisons for 0.4B/0.8B and 1.5B/2B, plus MLX, MPS, and CoreML workflows | [Apple guide](APPLE_USAGE.md) |
| NVIDIA T4 | Native HF, quantization, training, and production-close validation | [T4 evidence](../bench/t4_production_close_20260720/README.md) |
| NVIDIA A100/A800 | Ampere CUDA, training, parallel, and HF workflows | [Hardware matrix](HARDWARE_MATRIX.md) |
| NVIDIA H100 | Hopper CUDA, Transformers/HF, and benchmark execution path | [Performance guide](PERFORMANCE.md) |
| Ascend, Biren, MetaX, MUSA | Dedicated backends, runtime integration, and compatibility validation | [Hardware matrix](HARDWARE_MATRIX.md) |

### AMD `gfx1100` measured throughput

FP16, P128, cached decode. `Speedup` compares the fused RWKV route with the
generic RWKV route.

| RWKV-7 | B1 Decode | B8 aggregate Decode | Fused / generic speedup (B1 / B8) |
|---|---:|---:|---:|
| 0.1B | 347 tok/s | 2,667 tok/s | `1.88x / 2.04x` |
| 0.4B | 142 tok/s | 1,073 tok/s | `1.75x / 1.74x` |
| 1.5B | 71.3 tok/s | 514 tok/s | `1.40x / 1.47x` |
| 2.9B | 47.7 tok/s | 353 tok/s | `1.37x / 1.41x` |
| 7.2B | 29.7 tok/s | 214 tok/s | `1.23x / 1.29x` |
| 13.3B | 15.5 tok/s | 113 tok/s | `1.21x / 1.29x` |

The gfx1100 output-head W8/W4 route is faster than matching RWKV FP16 in all
40/40 Decode rows across 0.4B–13.3B and B1/B2/B4/B8. See the
[AMD validation guide](validation/AMD_ROCM_HF_VALIDATION.md),
[fused Decode evidence](../bench/amd_gfx1100_fused_decode_20260728/README.md),
and [0.4B–13.3B regression evidence](../bench/amd_gfx1100_rebase_validation_20260728/README.md).

## Comparison contract

- NVIDIA rows use the same GPU, batch size, prompt/decode lengths, and FP16
  precision. Apple rows use each model's promoted MLX W4 route.
- Every NVIDIA Qwen3.5 row records and verifies the optimized
  **FLA + Triton causal-conv** path and fused operator bindings.
- NVIDIA RWKV uses repository Native prefill and native-graph cached decode.
  Apple uses target-only MLX W4 for both sides.
- RTX 4080 uses each model's validated optimized runtime: PyTorch 2.11 with
  exact-shape FP16 accumulation for RWKV, and PyTorch 2.6 with full FLA for
  Qwen. Versions and backend telemetry are recorded; GPU, shapes, batch, and
  FP16 precision are identical.
- Release tiers are paired directly, such as 7.2B versus 9B. The tables expose
  raw tok/s, exact active parameter counts, and parameter-size-adjusted speed.
- The NVIDIA table is consistently dense FP16. The Apple table is consistently
  MLX W4.

## Live GPU reproduction

The commands below reload RWKV-7 and Qwen3.5 on the GPU, run warmup and timed
measurements, and regenerate Prefill, Decode, parameter-size-adjusted speed,
and backend-binding results.

### 1. Prepare the environment and models

Match PyTorch, CUDA, Triton, Transformers, FLA, and bitsandbytes to the
`environment.json` or `environment.txt` in the corresponding evidence
directory. Use the validated environment for each GPU.

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout 28f724259f8438cfcc71de40cf33889c6cf2396e

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[cuda,fla-reference,quant,torchao]"
```

Prepare two local directories:

1. an HF model converted from an official RWKV-7 `.pth` checkpoint with
   [`scripts/convert_rwkv7_to_hf.py`](../scripts/convert_rwkv7_to_hf.py);
2. the official Qwen3.5 HF model directory.

See the [user guide](USER_GUIDE.md) for conversion instructions, then check
the RWKV directory:

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

The output should contain `RESULT: READY` and `[PASS] Model directory`.

### 2. Complete RTX 4090 example: 1.5B versus 2B, B8

```bash
OUT=/tmp/rwkv-qwen35-4090-b8

PYTHON_BIN=python \
BATCH_SIZES=8 \
PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1h-1.5b-hf \
  /path/to/Qwen3.5-2B \
  "$OUT"

test "$(cat "$OUT/pipeline_exit_code.txt")" = 0
python - "$OUT/summary_active_work.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
speed = summary["speed"]
adjusted = summary["active_parameter_work"]
print(
    "raw prefill/decode median:",
    speed["median_prefill_speedup"],
    speed["median_decode_speedup"],
)
print(
    "parameter-adjusted prefill/decode:",
    adjusted["median_prefill_throughput_ratio"],
    adjusted["median_decode_throughput_ratio"],
)
print("red cells:", len(summary["red_cells"]))
PY
```

A complete run reports exit code 0, `pipeline_exit_code.txt=0`,
`red cells: 0`, and the Qwen full-FLA path.

### 3. Other NVIDIA GPU entry points

| GPU | Live measurement entry point | Scope |
|---|---|---|
| V100 historical active-pair lane | [Commands in the V100 evidence](../bench/v100_active_b1b8_20260715/README.md#reproduce) | 1.5B/2B, B1/B8 |
| V100 strict paired P+D v1 | [`bench/run_v100_rwkv_paired_pd_v1.sh`](../bench/run_v100_rwkv_paired_pd_v1.sh) + [`bench/run_v100_qwen35_paired_pd_v1.sh`](../bench/run_v100_qwen35_paired_pd_v1.sh) + [`bench/validate_qwen35_v100_paired_pd_v1.py`](../bench/validate_qwen35_v100_paired_pd_v1.py) | Frozen-reference 48+48 rows; raw and adjusted Prefill/Decode must all pass 48/48 |
| RTX 3090 current optimized Qwen paired P+D v2 | [`bench/run_3090_rwkv_paired_pd_v2.sh`](../bench/run_3090_rwkv_paired_pd_v2.sh) + [`bench/validate_qwen35_3090_paired_pd_v2.py`](../bench/validate_qwen35_3090_paired_pd_v2.py) | Four model pairs, 48 frozen-reference cells; requires raw and adjusted Prefill/Decode strict `>1.0x` plus 8/8 long-horizon correctness checks |
| RTX 3090 latest checkpoints | [`bench/run_3090_adjusted_prefill_pd.sh`](../bench/run_3090_adjusted_prefill_pd.sh) | Four model pairs, B1/B8, P128/512/2048, D128; strict per-cell adjusted-Prefill gate plus 25 correctness rows |
| RTX 4080 | [`bench/run_4080_adjusted_pd.sh`](../bench/run_4080_adjusted_pd.sh) | Runs all three pairs at B1/B8 and requires adjusted P/D `>1.00x` in every one of the 36 cells |
| RTX 4080 strict paired P+D v1 | [`bench/run_4080_rwkv_paired_pd_v1.sh`](../bench/run_4080_rwkv_paired_pd_v1.sh) + [`bench/run_4080_qwen35_paired_pd_v1.sh`](../bench/run_4080_qwen35_paired_pd_v1.sh) + [`bench/validate_qwen35_paired_pd_v1.py`](../bench/validate_qwen35_paired_pd_v1.py) | Same-runtime 36+36 rows; raw and adjusted Prefill/Decode must all pass 36/36 |
| RTX 4090 current optimized Qwen paired P+D v2 | [`bench/run_4090_rwkv_paired_pd_v2.sh`](../bench/run_4090_rwkv_paired_pd_v2.sh) + [`bench/validate_qwen35_4090_paired_pd_v2.py`](../bench/validate_qwen35_4090_paired_pd_v2.py) | Four model pairs, 48 frozen-reference cells; requires raw and adjusted Prefill/Decode strict `>1.0x` plus 8/8 long-horizon correctness checks |
| Historical RTX 4090 latest checkpoints | [`bench/run_4090_adjusted_pd.sh`](../bench/run_4090_adjusted_pd.sh) | Three model pairs, B1/B8, P128/512/2048, D128/512; requires adjusted P/D `>1.00x` in all 36 cells |
| RTX 5070 Laptop | [`bench/run_5070_qwen35_full_fla_bsz8.ps1`](../bench/run_5070_qwen35_full_fla_bsz8.ps1) | PowerShell with `-RwkvModel`, `-QwenModel`, and `-OutDir` |
| RTX 5090 | [`bench/run_5090_qwen35_full_matrix.sh`](../bench/run_5090_qwen35_full_matrix.sh) | Four model pairs, B1/B8 full matrix |
| RTX 5090 paired Decode v1 | [`bench/run_5090_rwkv_paired_decode_v1.sh`](../bench/run_5090_rwkv_paired_decode_v1.sh) + [`bench/validate_qwen35_paired_decode_v1.py`](../bench/validate_qwen35_paired_decode_v1.py) | Fresh 48-row RWKV capture joined to the SHA-locked Qwen reference; strict adjusted Decode 48/48, no Prefill/E2E gate |
| RTX 5090 historical 2026-08-11 checkpoints | [Commands in the strict-gate evidence](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md#reproduce-the-gate) | Four model pairs, B1/B8, P128/512/2048, D128 |
| RTX 5090 Qwen-only optimized reference v2 | [`bench/run_5090_qwen35_best_optimized_hf.sh`](../bench/run_5090_qwen35_best_optimized_hf.sh) | Run once per Qwen checkpoint; four fixed model routes form the 48-row reference-only matrix |

Each runner verifies the exact GPU, backend bindings, matrix coverage, and
acceptance gates, and writes `pipeline_exit_code.txt`,
`matrix_failures.txt`, `summary*.json`, and full logs.

### 4. Apple M5 live GPU measurement

Use the MLX environment and local W4 model directory:

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/models \
COOLDOWN_SECONDS=30 \
INITIAL_COOLDOWN_SECONDS=60 \
  scripts/run_apple_bsz8_target_only_acceptance.sh
```

The script runs RWKV-7 0.4B/1.5B and Qwen3.5 0.8B/2B on the Apple GPU and
reports raw Prefill/Decode, parameter-size-adjusted ratios, peak memory, and
token consistency.

### 5. AMD `gfx1100` live GPU measurement

Use a PyTorch environment for ROCm 7.2.1 and make `/dev/kfd` and
`/dev/dri/render*` available:

```bash
OUT=/tmp/rwkv7-amd-gfx1100
mkdir -p "$OUT"
set -o pipefail

bash bench/run_amd_rocm_hf_validation.sh \
  HF_DIR=/path/to/rwkv7-g1d-0.1b-hf \
  OUT_DIR="$OUT" |& tee "$OUT/console.log"

grep -F "AMD ROCm HF VALIDATION PASS" "$OUT/console.log"
```

The runner verifies HIP visibility and `gcnArchName`; exact `gfx1100` selects
the promoted fused route and architecture tuning. To reproduce only the fused
Decode A/B, use the [focused evidence command](../bench/amd_gfx1100_fused_decode_20260728/README.md#reproduce).
