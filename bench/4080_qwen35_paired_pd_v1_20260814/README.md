# RTX 4080 strict paired Prefill/Decode v1

Status: **PASS — raw and parameter-adjusted Prefill and Decode clear Qwen in
36/36 matched cells**. Every gate uses unrounded throughput and requires a
strict ratio greater than `1.0x`.

This is a same-runtime inference-speed result, not a model-quality result. It
does not claim continuous end-to-end latency, TTFT, cache-handoff latency, or
that RWKV is better than Qwen on reasoning, coding, multilingual, or other
quality evaluations. [`paired_validation.json`](paired_validation.json)
therefore records `continuous_e2e_eligible=false`.

## Scope

- GPU: one NVIDIA GeForce RTX 4080 (`sm_89`, 16,376 MiB), driver `595.71.05`.
- Models: RWKV-7 0.4B/1.5B/2.9B versus Qwen3.5 0.8B/2B/4B.
- Dense FP16, batch 1/8, prompt 128/512/2048, Decode 128/512, Prefill chunk
  512: 3 pairs x 2 batches x 3 prompts x 2 Decode lengths = 36 cells.
- Three warmups, seven measured runs, and the unrounded per-cell median.
- B8 throughput is aggregate across eight sequences. Formal timing replicates
  one prompt eight times; the separate correctness probes use eight distinct
  prompts.
- Candidate and reference were both captured from clean repository commit
  `398277d94e1d1dc441af97dea0578b87fa072f74` with the same locked runtime.
  The candidate and reference phases are sequential, not interleaved A/B.

Qwen is deliberately strong here: 0.8B and 2B use official fast operators with
StaticCache + Inductor CUDA Graph (`max-autotune`), while 4B uses the strongest
16 GiB-safe official path, eager `module_call` + DynamicCache. The benchmark
does not weaken Qwen to make the comparison pass.

## Strict result

For each cell:

```text
raw_ratio = rwkv_tokps_total_raw / qwen_tokps_total_raw

adjusted_ratio = raw_ratio
               * (rwkv_active_parameters / qwen_active_parameters)
```

All four ratios must be strictly greater than `1.0` in all 36 cells. Rounded
display values and aggregate medians cannot make a cell pass.

| Metric across 36 cells | Minimum | Median | Maximum | Cells >1.0x |
|---|---:|---:|---:|---:|
| Raw Prefill ratio | `1.500014x` | `1.920082x` | `4.825638x` | **36/36** |
| Raw Decode ratio | `1.302605x` | `1.723038x` | `3.065001x` | **36/36** |
| Parameter-adjusted Prefill ratio | `1.051333x` | `1.313931x` | `2.891099x` | **36/36** |
| Parameter-adjusted Decode ratio | `1.022115x` | `1.190224x` | `1.836279x` | **36/36** |

The weakest adjusted Decode cell is RWKV 0.4B versus Qwen3.5 0.8B at
B8/P128/D128:

| RWKV tok/s | Qwen tok/s | Required RWKV tok/s | Raw ratio | Adjusted ratio | Margin |
|---:|---:|---:|---:|---:|---:|
| `3,344.250766` | `1,960.231741` | `3,271.893982` | `1.706049x` | `1.022115x` | `+72.356785 tok/s` |

The weakest adjusted Prefill cell is RWKV 2.9B versus Qwen3.5 4B at
B1/P512/D128: RWKV `14,257.086343 tok/s`, Qwen `9,504.633890 tok/s`, raw
`1.500014x`, adjusted `1.051333x`.

## Original throughput comparison

These are the medians across each model/batch lane's six P/D cells. They are
descriptive summaries; the strict gate still evaluates every individual cell.

| RWKV / Qwen | Batch | RWKV Prefill | Qwen Prefill | Raw P ratio | RWKV Decode | Qwen Decode | Raw D ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4B / 0.8B | B1 | 45,562 | 22,984 | `2.066x` | 618 | 310 | `1.998x` |
| 0.4B / 0.8B | B8 | 104,224 | 45,067 | `2.200x` | 3,344 | 1,711 | `1.960x` |
| 1.5B / 2B | B1 | 30,683 | 19,692 | `1.630x` | 207 | 154 | `1.345x` |
| 1.5B / 2B | B8 | 39,234 | 22,896 | `1.649x` | 1,360 | 960 | `1.417x` |
| 2.9B / 4B | B1 | 14,264 | 8,956 | `1.665x` | 108 | 63.4 | `1.702x` |
| 2.9B / 4B | B8 | 19,533 | 9,866 | `1.987x` | 729 | 422 | `1.727x` |

Full precision for every cell is in
[`paired_pd_table.jsonl`](paired_pd_table.jsonl), with a readable 36-cell table
in [`paired_pd.md`](paired_pd.md).

## Routes and correctness

RWKV uses the repository native-graph route. B1 requires the real Ada W/A/G/V
extension and fails closed before graph capture if it cannot build or execute;
B8 uses its shape-specific route. The RKV policy is locked per lane: `vkwr_auto`
for every B1 lane and the 0.4B/1.5B B8 lanes, and `manual` for 2.9B B8. The
route manifest proves the selected/effective layer coverage rather than merely
recording requested environment flags.

An independent FLA wrapper-reference versus native-graph oracle covers all
three checkpoints at B1/B8, P2048/D512. All **6/6** comparisons preserve all
512 greedy tokens and finite Decode logits. The global prompt/final minimum
row cosines are `0.999990582` and `0.999993861`, above the `0.9999` hard gate.
See
[`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json)
and the six `decode_correctness_*_compare.json` files.

## Fixed runtime

| Component | Value |
|---|---|
| Python | 3.12.2 |
| PyTorch / CUDA runtime | 2.11.0+cu130 / 13.0 |
| Triton | 3.6.0 |
| Transformers | 5.12.1 |
| FLA | 0.5.1 |
| causal-conv1d | 1.6.2.post1 |
| GPU / driver | NVIDIA GeForce RTX 4080 / 595.71.05 |

[`runtime-lock.json`](runtime-lock.json), [`pip-freeze.txt`](pip-freeze.txt),
[`system.csv`](system.csv), and the before/after model manifests bind the
runtime, exact card, and model bytes. Candidate SHA256 is
`4c4359a6cd379d01bf009e503d9f67a5ffb141bb2c4e2c962be0a9029b2edb5a`;
Qwen reference SHA256 is
`4077d0eeb402a3ef20a5e9d3e4247767f07b996456ec26d985ff8cdbdd346efa`.

## Artifact map

- [`rwkv_candidate.jsonl`](rwkv_candidate.jsonl): sorted 36-row RWKV matrix.
- [`qwen_reference.jsonl`](qwen_reference.jsonl): sorted 36-row Qwen matrix.
- [`paired_pd_table.jsonl`](paired_pd_table.jsonl): joined full-precision cells.
- [`paired_validation.json`](paired_validation.json): fail-closed validation.
- [`rwkv_candidate_routes.json`](rwkv_candidate_routes.json): exact lane route,
  source, environment, and fresh-process contract.
- `rwkv_{0p4,1p5,2p9}_b{1,8}.jsonl`: six formal cells per RWKV lane.
- `qwen_{0p8,2b,4b}.jsonl`: twelve formal cells per Qwen model.
- [`rwkv_native_graph_fla_correctness.json`](rwkv_native_graph_fla_correctness.json):
  six 512-token correctness comparisons.
- [`formal.log`](formal.log), [`exit_code.txt`](exit_code.txt), and
  [`runner_exit_code.txt`](runner_exit_code.txt): formal execution evidence.
- [`remote_artifact_sha256.txt`](remote_artifact_sha256.txt): all 78 files in
  the complete remote formal directory.

Committed log copies are line-ending and trailing-whitespace normalized for
repository hygiene. `remote_artifact_sha256.txt` binds the untouched remote
originals; the structured JSON/JSONL result files remain byte-identical.

The complete external audit copy contains twelve binary `.pt` probe tensors.
They are intentionally excluded from Git for repository-size and binary-hygiene
reasons; their exact names and hashes remain in
[`probe_artifact_sha256.txt`](probe_artifact_sha256.txt). Consequently, a fresh
clone can inspect the recorded text evidence but cannot recompute tensor-level
cosines without the original hash-matching probes or a fresh formal run.

## Reproduce

Use new, empty absolute directories; the runner is append-never and refuses to
overwrite prior evidence:

```bash
export OUT_DIR=/home/user/benchmarks/new-4080-paired-pd-v1
export CACHE_ROOT=/home/user/.cache/new-4080-paired-pd-v1
export PYTHON_BIN=/path/to/frozen-runtime/bin/python
export REPOSITORY_COMMIT=$(git rev-parse HEAD)
export CUDA_COMPONENT_INCLUDE=/path/containing/cusparse.h
export RWKV_04_MODEL=/models/rwkv7-g1d-0.4b-hf
export RWKV_15_MODEL=/models/rwkv7-g1i-1.5b-hf
export RWKV_29_MODEL=/models/rwkv7-g1i-2.9b-hf
export QWEN_08_MODEL=/models/qwen3.5-0.8b
export QWEN_2_MODEL=/models/qwen3.5-2b
export QWEN_4_MODEL=/models/qwen3.5-4b

bash bench/run_4080_rwkv_paired_pd_v1.sh
bash bench/run_4080_qwen35_paired_pd_v1.sh

python bench/validate_qwen35_paired_pd_v1.py \
  --candidate "$OUT_DIR/rwkv_candidate.jsonl" \
  --reference "$OUT_DIR/qwen_reference.jsonl" \
  --expected-device "NVIDIA GeForce RTX 4080" \
  --summary "$OUT_DIR/paired_validation.json" \
  --paired-table "$OUT_DIR/paired_pd_table.jsonl" \
  --markdown "$OUT_DIR/paired_pd.md"
```

Promotion requires exit zero, no validator errors,
`paired_pd_table_eligible=true`, and all four unrounded P/D gates strictly
above `1.0x` in all 36 cells.
