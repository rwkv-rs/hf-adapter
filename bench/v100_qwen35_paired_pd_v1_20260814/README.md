# Tesla V100 strict paired Prefill/Decode v1

Status: **PASS — raw and parameter-adjusted Prefill and Decode clear Qwen in
48/48 matched cells**. Every gate uses unrounded throughput and requires a
strict ratio greater than `1.0x`.

This is an inference-engine speed result, not a model-quality result. It does
not claim continuous end-to-end latency, TTFT, cache-handoff latency, or that
RWKV is better than Qwen on reasoning, coding, multilingual, or other quality
evaluations. [`paired_validation.json`](paired_validation.json) therefore
records `continuous_e2e_eligible=false`.

## Scope

- GPU: GPU 0 on a 2 x Tesla V100-PCIE-32GB server (`sm_70`), driver
  `580.173.02`.
- Models: RWKV-7 0.4B/1.5B/2.9B/7.2B versus Qwen3.5 0.8B/2B/4B/9B.
- Dense FP16, batch 1/8, prompt 128/512/2048, Decode 128/512, Prefill chunk
  512: 4 pairs x 2 batches x 3 prompts x 2 Decode lengths = 48 cells.
- Three warmups, seven measured runs, and the unrounded per-cell median.
- B8 throughput is aggregate across eight sequences. Formal timing replicates
  one prompt eight times; the separate correctness probes use eight distinct
  prompts.
- RWKV was captured from clean commit
  `2b4359e43448ed657dfd96bb084385a3c49a8b19`. Qwen is the immutable
  `47ee4104b07e35e85a0a54df2aaa8b4e87db1dc8` reference, copied under an exact
  aggregate SHA256 lock. Both sides have the same GPU and six-field runtime
  signature, but they were measured sequentially rather than as interleaved
  A/B samples.

Qwen is deliberately strong here. All four models use official FLA operators
and StaticCache raw CUDA Graph Decode; the 9B lane additionally locks math-only
SDPA. Every Qwen graph replays as one CUDA Graph launch and passes its
same-cache numerical and complete greedy gates. The benchmark does not weaken
Qwen to make the comparison pass.

## Strict result

For each cell:

```text
raw_ratio = rwkv_tokps_total_raw / qwen_tokps_total_raw

adjusted_ratio = raw_ratio
               * (rwkv_active_parameters / qwen_active_parameters)
```

All four ratios must be strictly greater than `1.0` in all 48 cells. Rounded
display values and aggregate medians cannot make a cell pass.

| Metric across 48 cells | Minimum | Median | Maximum | Cells >1.0x |
|---|---:|---:|---:|---:|
| Raw Prefill ratio | `2.249335x` | `4.744253x` | `13.714267x` | **48/48** |
| Raw Decode ratio | `1.393444x` | `2.398393x` | `4.662333x` | **48/48** |
| Parameter-adjusted Prefill ratio | `1.808536x` | `3.217214x` | `8.216385x` | **48/48** |
| Parameter-adjusted Decode ratio | `1.120373x` | `1.617469x` | `2.793261x` | **48/48** |

The weakest adjusted Decode cell is RWKV 7.2B versus Qwen3.5 9B at
B8/P128/D128:

| RWKV tok/s | Qwen tok/s | Required RWKV tok/s | Raw ratio | Adjusted ratio | Margin |
|---:|---:|---:|---:|---:|---:|
| `266.506455` | `191.257425` | `237.872983` | `1.393444x` | `1.120373x` | `+28.633472 tok/s` |

The weakest adjusted Prefill cell is also the 7.2B/9B pair, at B1/P128/D128:
RWKV `2,232.609399 tok/s`, Qwen `992.564328 tok/s`, raw `2.249335x`,
adjusted `1.808536x`.

## Original throughput comparison

These are medians across each model/batch lane's six P/D cells. They are
descriptive summaries; the strict gate evaluates every individual cell.

| RWKV / Qwen | Batch | RWKV Prefill | Qwen Prefill | Raw P ratio | RWKV Decode | Qwen Decode | Raw D ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 17,822 | 4,117 | `4.384x` | 434 | 111 | `3.909x` |
| 0.4B / 0.8B | B8 | 56,270 | 4,140 | `13.457x` | 1,784 | 688 | `2.596x` |
| 1.5B / 2B | B1 | 11,167 | 3,672 | `3.055x` | 230 | 83.3 | `2.762x` |
| 1.5B / 2B | B8 | 20,861 | 3,816 | `5.467x` | 841 | 517 | `1.630x` |
| 2.9B / 4B | B1 | 7,066 | 1,382 | `5.109x` | 124 | 46.0 | `2.696x` |
| 2.9B / 4B | B8 | 10,711 | 1,505 | `7.144x` | 536 | 275 | `1.950x` |
| 7.2B / 9B | B1 | 3,758 | 1,174 | `3.142x` | 56.1 | 31.2 | `1.801x` |
| 7.2B / 9B | B8 | 4,708 | 1,283 | `3.653x` | 267 | 164 | `1.632x` |

Full precision for every cell is in
[`paired_pd_table.jsonl`](paired_pd_table.jsonl), with a readable 48-cell table
in [`paired_pd.md`](paired_pd.md).

## Routes and correctness

RWKV uses the repository native-graph route. Every B1 lane requires the real
SM70 W/A/G/V extension and fails closed before graph capture if it cannot build
or execute. The extension is selected/effective for all eligible layers:
23/23 in the 24-layer checkpoints and 31/31 in the 32-layer checkpoints. B8
uses its shape-specific policies and explicitly leaves the B1-only extension
off. [`rwkv_candidate_routes.json`](rwkv_candidate_routes.json) binds these
effective routes and the fresh-process matrix.

Independent FLA wrapper-reference versus native-graph oracles cover all four
checkpoints at B1/B8, P2048/D512. All **8/8** comparisons preserve all 512
greedy tokens and finite Decode logits. Across those comparisons, the minimum
prompt/final row cosines are `0.999994457` and `0.999991894`, above the
`0.9999` hard gate. A ninth 7.2B/B8/P128 closure compares native eager with
native graph and also passes 512-token greedy equality, finite logits, and
`0.999993205/0.999993682` prompt/final cosine.

See
[`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json)
and the nine `decode_correctness_*_compare.json` files.

## Fixed runtime

| Component | Value |
|---|---|
| Python | 3.11.15 |
| PyTorch / CUDA runtime | 2.5.1+cu124 / 12.4 |
| Triton | 3.4.0 |
| Transformers | 5.12.1 |
| FLA | 0.5.1 |
| causal-conv1d | not installed; Qwen uses the locked FLA-Triton convolution route |
| GPU / driver | Tesla V100-PCIE-32GB / 580.173.02 |

[`runtime-lock.json`](runtime-lock.json), [`pip-freeze.txt`](pip-freeze.txt),
[`system.csv`](system.csv), and the before/after model manifests bind the
runtime, exact cards, and model bytes. Candidate SHA256 is
`45d90cf900972296ab1ca41185e2beedad543dc4cc3876f39344fc5aaea3c27a`;
Qwen reference SHA256 is
`00b81eb0f80d8204069fc69acc5b802d3192b52386055262776e0c9e9ddab5bd`.

## Artifact map

- [`rwkv_candidate.jsonl`](rwkv_candidate.jsonl): sorted 48-row RWKV matrix.
- [`qwen_reference.jsonl`](qwen_reference.jsonl): sorted frozen 48-row Qwen
  matrix.
- [`paired_pd_table.jsonl`](paired_pd_table.jsonl): joined full-precision cells.
- [`paired_validation.json`](paired_validation.json): fail-closed validation.
- [`rwkv_candidate_routes.json`](rwkv_candidate_routes.json): exact lane route,
  source, environment, and fresh-process contract.
- `rwkv_{0p4,1p5,2p9,7p2}_b{1,8}.jsonl`: six formal cells per RWKV lane.
- `qwen_{0p8,2b,4b,9b}.jsonl`: twelve frozen formal cells per Qwen model.
- [`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json):
  eight 512-token FLA/native comparisons plus the targeted 7.2B closure.
- [`formal.log`](formal.log), [`exit_code.txt`](exit_code.txt), and
  [`runner_exit_code.txt`](runner_exit_code.txt): formal execution evidence.
- [`remote_artifact_sha256.txt`](remote_artifact_sha256.txt): all 86 source
  files in the complete external formal capture.

The complete external audit copy contains eighteen binary `.pt` probe tensors.
They are intentionally excluded from Git for repository-size and binary-hygiene
reasons; their exact names and hashes remain in
[`probe_artifact_sha256.txt`](probe_artifact_sha256.txt). Consequently, a fresh
clone can inspect the recorded text evidence but cannot recompute tensor-level
cosines without the original hash-matching probes or a fresh formal run.

## Reproduce

Use new, empty absolute directories. The runner is append-never and refuses to
overwrite prior evidence. To reuse the exact frozen Qwen reference:

```bash
export OUT_DIR=/home/user/benchmarks/new-v100-paired-pd-v1
export CACHE_ROOT=/home/user/.cache/new-v100-paired-pd-v1
export PYTHON_BIN=/path/to/frozen-runtime/bin/python
export REPOSITORY_COMMIT=$(git rev-parse HEAD)
export CUDA_TOOLKIT_VIEW=/path/to/cuda/toolkit/view
export CUDA_COMPONENT_INCLUDE=/path/containing/cusparse.h
export FLA_TARGET=/path/to/fla/python/target
export TRITON_TARGET=/path/to/triton/python/target
export RWKV_04_MODEL=/models/rwkv7-g1d-0.4b-hf
export RWKV_15_MODEL=/models/rwkv7-g1i-1.5b-hf
export RWKV_29_MODEL=/models/rwkv7-g1i-2.9b-hf
export RWKV_72_MODEL=/models/rwkv7-g1i-7.2b-hf
export FROZEN_QWEN_DIR=/evidence/v100-qwen35-reference
export FROZEN_QWEN_REFERENCE_SHA256=00b81eb0f80d8204069fc69acc5b802d3192b52386055262776e0c9e9ddab5bd

bash bench/run_v100_qwen35_paired_pd_v1.sh
```

Promotion requires exit zero, no validator errors,
`paired_pd_table_eligible=true`, and all four unrounded P/D gates strictly
above `1.0x` in all 48 cells.
