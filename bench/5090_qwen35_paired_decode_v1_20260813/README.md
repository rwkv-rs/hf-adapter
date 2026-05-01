# RTX 5090 frozen-Qwen paired Decode v1

Status: **PASS — parameter-adjusted Decode clears Qwen in 48/48 cells**.
Every ratio is recomputed from unrounded raw throughput and is strictly greater
than `1.0x`.

This is a narrowly scoped Decode result. It is **not** a model-quality,
Prefill, TTFT, continuous end-to-end, or cache-handoff-latency claim. The
frozen Qwen source remains an `independent_best_prefill_and_decode` envelope,
so [`paired_validation.json`](paired_validation.json) deliberately records
`continuous_e2e_eligible=false`.

## Scope

- GPU: one NVIDIA GeForce RTX 5090 (`sm_120`, 32,607 MiB), driver
  `595.58.03`.
- Models: RWKV-7 0.4B/1.5B/2.9B/7.2B versus Qwen3.5 0.8B/2B/4B/9B.
- Dense FP16, batch 1/8, prompt 128/512/2048, Decode 128/512, Prefill chunk
  512.
- Three warmups, seven measured runs and the unrounded per-cell median.
- B8 Decode is aggregate throughput across eight sequences.
- Formal B8 timing follows the benchmark protocol by replicating one prompt
  eight times. The separate correctness probes use eight distinct prompts;
  timing rows must not be interpreted as a distinct-request workload.
- RWKV uses one declared `best_optimized_hf` native-graph route for all six
  cells in each model/batch lane; all eight lanes run in fresh processes.
- Qwen is the immutable 48-row official-operator reference in
  [`../5090_qwen35_best_optimized_hf_v1_20260813/`](../5090_qwen35_best_optimized_hf_v1_20260813/README.md).
  Its SHA256 is
  `b02378fe14d455f52940a3d24e4f515f49c18a06f57c65ad0b461a2330b5f6d1`.

The formal RWKV capture used clean repository commit
`32c054208b4bd9b9b5c8eca6340e2fba1bd4b533`. The 48-row candidate SHA256 is
`e24fda2db1f467eab5cbcac7da17ea3adf5a924c45c04c44e01fcae9af91b7af`.
The frozen Qwen source was captured at commit
`1e80d3f7af6340c796a01eaae479274949c412dd`. Candidate and reference align on
the six validator runtime fields, GPU and shape protocol, but come from
different repository commits and were not measured as an interleaved A/B run.

## Strict acceptance result

The hard gate is:

```text
adjusted_decode_ratio =
    (rwkv_decode_tokps_total_raw / qwen_decode_tokps_total_raw)
    * (rwkv_active_parameters / qwen_active_parameters)

adjusted_decode_ratio > 1.0 for every one of the 48 cells
```

`>=1.0`, a rounded display value, or a median-only win cannot pass. The
validator reports `uses_unrounded_raw_throughput=true`, 48 joined cells, zero
errors and `paired_decode_table_eligible=true`.

| Metric across all 48 matched cells | Minimum | Median | Maximum | Cells above 1.0x |
|---|---:|---:|---:|---:|
| Parameter-adjusted Decode ratio | `1.029966x` | `1.409279x` | `2.063849x` | **48/48** |
| Raw Decode ratio (supporting telemetry) | `1.373660x` | `1.903882x` | `3.017760x` | 48/48 |

Raw Decode also happens to lead in all 48 measured cells, but it is supporting
telemetry rather than the formal acceptance contract.

The weakest adjusted cell is RWKV 0.4B versus Qwen3.5 0.8B at B1/P128/D128:

| RWKV tok/s | Qwen tok/s | Required RWKV tok/s | Adjusted ratio | Margin |
|---:|---:|---:|---:|---:|
| `1,124.579967` | `654.147751` | `1,091.861765` | `1.029966x` | `+32.718202 tok/s` / `+2.996552%` |

## Results by model pair and batch

These minima and medians are computed across the six Prompt/Decode cells in
each fixed route. Full precision remains in
[`paired_decode_table.jsonl`](paired_decode_table.jsonl).

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

The complete sorted 48-cell human-readable table is in
[`paired_decode.md`](paired_decode.md). It includes raw RWKV/Qwen throughput,
active-parameter ratio, required RWKV throughput, margin and per-cell PASS.

## Exact-card optimization and correctness

The two narrow SM120 B8 routes were independently compared with their
exact-card baseline at P2048/D512. Their attached full 512-token correctness
probes use distinct prompts; the throughput timing retains the replicated-
prompt benchmark protocol:

| RWKV | Baseline tok/s | Candidate tok/s | Speedup | Prompt cosine | Final cosine | Greedy |
|---|---:|---:|---:|---:|---:|---|
| 0.4B | 3,442 | 6,421 | `1.865301x` | `0.999994278` | `0.999994159` | exact |
| 1.5B | 2,065 | 3,083 | `1.492719x` | `0.999995232` | `0.999994516` | exact |

Both routes keep the SM120 W/A/G/V grouped projection and shared compiled FFN
effective across all 24 layers. The compiled FFN reports one reused graph and
zero graph breaks. Some `sm120_*_candidate.log` lines show Inductor autotune
rejecting Triton candidates with `out of resource: Required 110592 Hardware
limit 101376`. Those messages concern discarded kernel candidates; they are
not CUDA VRAM OOMs or formal-route failures. The selected route remains 24/24
effective and passes the A/B and final validator. Accordingly, this README
claims zero **validator contract errors**, not an absence of diagnostic error-
level lines from autotune.

An independent FLA-versus-native oracle covers all four RWKV checkpoints at
B1/B8, P2048/D512. All **8/8** comparisons preserve all 512 greedy tokens and
finite Decode logits. The global prompt/final minimum row cosines are
`0.999981999` and `0.999970913`, both above the `0.9999` hard gate. See
[`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json)
and the eight committed `decode_correctness_*_compare.json` files.

## Fixed runtime

The RWKV rows use the same package/runtime signature as the frozen Qwen
reference.

| Component | Value |
|---|---|
| Python | 3.10.12 |
| PyTorch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| Triton | 3.4.0 |
| Transformers | 5.12.1 |
| FLA | 0.5.1 |
| causal-conv1d | 1.6.2.post1 |
| GPU / driver | NVIDIA GeForce RTX 5090 / 595.58.03 |

[`runtime-lock.json`](runtime-lock.json), [`pip-freeze.txt`](pip-freeze.txt),
[`system.csv`](system.csv), and the before/after model manifests bind the
runtime, exact card and checkpoint bytes. The model hash snapshots are
byte-identical across the formal run.

## Artifact map

- [`rwkv_candidate.jsonl`](rwkv_candidate.jsonl): sorted 48-row, full-precision
  RWKV candidate matrix.
- [`paired_decode_table.jsonl`](paired_decode_table.jsonl): 48 joined cells
  with recomputed raw and parameter-adjusted Decode ratios.
- [`paired_validation.json`](paired_validation.json): fail-closed validator
  result, coverage, route, A/B and correctness summaries.
- [`rwkv_candidate_routes.json`](rwkv_candidate_routes.json): exact route and
  fresh-process contract for the eight model/batch lanes.
- `rwkv_{0p4,1p5,2p9,7p2}_b{1,8}.jsonl`: six formal cells per lane.
- [`rwkv_sm120_b8_ab.json`](rwkv_sm120_b8_ab.json): two small-model B8 A/B
  manifests plus the committed source rows and comparison JSON.
- [`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json):
  eight P2048/D512 native-graph-versus-FLA checks plus source rows and
  comparison JSON.
- [`model_hashes.sha256`](model_hashes.sha256) and
  [`model_hashes.after.sha256`](model_hashes.after.sha256): recursive model
  hashes before and after capture.
- [`formal.log`](formal.log), [`validator.log`](validator.log),
  [`exit_code.txt`](exit_code.txt), and [`validator.exit`](validator.exit):
  formal runner/validator output and zero exit codes.
- [`remote_artifact_sha256.txt`](remote_artifact_sha256.txt): checksums from
  the complete remote formal directory.

The complete external audit copy, including twenty binary `.pt` probe tensors,
was used for the local validator rerun. Those tensors are intentionally
excluded from Git for repository-size and binary-hygiene reasons; a fresh clone
cannot rerun the tensor comparison from committed files alone. Their exact
names and SHA256 values remain in
[`probe_artifact_sha256.txt`](probe_artifact_sha256.txt), while all text source
rows, comparison outputs and manifests are committed. The text bundle supports
inspection of the recorded PASS; recomputing tensor-level cosines requires the
original hash-matching external probes or a fresh formal capture.

## Reproduce

Generate a fresh append-only candidate bundle outside the repository:

```bash
export OUT_DIR=/path/to/new-rwkv-paired-decode-v1
export PYTHON_BIN=/path/to/frozen-qwen-runtime/bin/python
export REPOSITORY_COMMIT=32c054208b4bd9b9b5c8eca6340e2fba1bd4b533
export RWKV_04_MODEL=/models/rwkv7-g1d-0.4b-hf
export RWKV_15_MODEL=/models/rwkv7-g1i-1.5b-hf
export RWKV_29_MODEL=/models/rwkv7-g1i-2.9b-hf
export RWKV_72_MODEL=/models/rwkv7-g1i-7.2b-hf

bash bench/run_5090_rwkv_paired_decode_v1.sh
```

Then run the fail-closed join against the immutable checked-in Qwen reference:

```bash
python bench/validate_qwen35_paired_decode_v1.py \
  --qwen-reference bench/5090_qwen35_best_optimized_hf_v1_20260813/qwen_reference.jsonl \
  --rwkv-candidate "$OUT_DIR/rwkv_candidate.jsonl" \
  --sm120-ab-manifest "$OUT_DIR/rwkv_sm120_b8_ab.json" \
  --decode-correctness-manifest "$OUT_DIR/rwkv_native_graph_fla_correctness.json" \
  --candidate-route-manifest "$OUT_DIR/rwkv_candidate_routes.json" \
  --runtime-lock "$OUT_DIR/runtime-lock.json" \
  --model-hashes "$OUT_DIR/model_hashes.sha256" \
  --expected-candidate-commit "$REPOSITORY_COMMIT" \
  --validation "$OUT_DIR/paired_validation.json" \
  --paired-table "$OUT_DIR/paired_decode_table.jsonl" \
  --markdown "$OUT_DIR/paired_decode.md"
```

The run is promotable only when the validator exits zero, reports no errors,
sets `paired_decode_table_eligible=true`, and passes all 48 unrounded adjusted
Decode ratios strictly above `1.0x`.
