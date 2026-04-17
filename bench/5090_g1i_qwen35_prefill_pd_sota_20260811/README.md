# RTX 5090 single-card prefill ceiling

This immutable follow-up tightens the dense-FP16 B1/B8 prefill result from
[`../5090_g1i_qwen35_prefill_pd_20260811/`](../5090_g1i_qwen35_prefill_pd_20260811/README.md).
It keeps the same official full-FLA Qwen3.5 reference rows and reruns every
RWKV candidate cell after two exact-card changes: recurrent-cache continuation
inside the native prefill CUDA graph, and removal of the 7.2B stacked-RKV
weight copy.

## Contract

- Device: one NVIDIA GeForce RTX 5090 (`sm_120`), driver `595.58.03`.
- Runtime: PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, Triton `3.6.0`,
  Transformers `5.12.1`, FLA `0.5.1`, bitsandbytes `0.49.2`.
- Candidate checkpoints: RWKV-7 g1d 0.4B and 2026-08-05 g1i
  1.5B/2.9B/7.2B, converted to HF FP16.
- References: official Qwen3.5 0.8B/2B/4B/9B. The 24 reference rows are reused
  field-for-field from the prior artifact because their models, shapes, runtime,
  and required full-FLA route are unchanged.
- Matrix: B1/B8, prompt 128/512/2048, decode 128, prefill chunk 512,
  warmup 2, measured runs 5.
- Acceptance: every cell must have parameter-adjusted prefill PD `>=1.0`;
  all Qwen references must verify FLA, Triton causal convolution, live fused
  bindings, and the full-fused contract.

The primary metric remains:

```text
prefill PD = (RWKV prefill tok/s / Qwen prefill tok/s)
             * (RWKV active parameters / Qwen active parameters)
```

## Result

The strict analyzer passes all `24/24` cells with no red cells. Raw
RWKV/Qwen prefill is `1.347871x` minimum and `1.819072x` median.
Parameter-adjusted prefill PD is `1.072987x` minimum and `1.317515x` median.
The prior artifact was `1.028427x` minimum and `1.269319x` median.

| RWKV / Qwen pair | Minimum raw prefill | Minimum prefill PD | Passing cells |
|---|---:|---:|---:|
| 0.4B / 0.8B | `1.790961x` | `1.072987x` | 6/6 |
| 1.5B / 2B | `1.386841x` | `1.125645x` | 6/6 |
| 2.9B / 4B | `1.581536x` | `1.108470x` | 6/6 |
| 7.2B / 9B | `1.347871x` | `1.083731x` | 6/6 |

The new weakest cell is 0.4B/B8/P512 at `206,364.2 tok/s` and
`1.072987x` adjusted PD. The largest gain is 0.4B/B1/P2048: throughput rises
from `27,270.0` to `61,343.8 tok/s`, or `2.2495x` the prior candidate row,
while adjusted PD reaches `2.313444x`.

## What changed

### CUDA-graph continuation

`rwkv7_prefill_chunks` marks dense continuation chunks as graph-safe. The
runner owns stable recurrent state, attention/FFN shift state, and an FP16
elapsed-token tensor; the captured graph copies each output state back into
its next input. Consecutive 512-token chunks therefore replay one graph instead
of returning to eager execution after the first chunk. Independent requests
reset the stable inputs, while a returned cache is detached before its storage
is reused by another request.

The elapsed-token input matters for exact RWKV-7 decay semantics. Both graph
and eager continuation now pass the cumulative token offset to the FP16
sequence recurrence rather than restarting it at zero for every chunk.

The dedicated graph-versus-eager chunked oracle passes `8/8` P2048 rows across
all four models and B1/B8. Prompt minimum cosine is `0.99999988`, post-cache
handoff minimum cosine is `0.99999994`, and both greedy checks match in every
row. This is the appropriate graph gate because both sides use identical
512-token GEMM shapes and recurrence math. A direct 2048-token call changes
FP16 GEMM shape/order, so direct-full versus chunked is informative numeric
telemetry rather than evidence about graph replay itself.

### 7.2B stacked-RKV removal

Profiling showed the packed R/K/V copy cost memory without an end-to-end win
on 7.2B. Its exact RTX 5090 allowlist rows are removed. All six final 7.2B
candidate rows report `rwkv_prefill_stacked_rkv_effective=false`, reach
`8,789.6-19,988.8 tok/s`, and peak at `14,334.9-15,484.4 MiB`. The prior
artifact used `17,406.9-18,616.3 MiB`, so the final route releases roughly
`2.4-3.2 GiB` of peak VRAM while remaining faster on every P2048 cell.

The previous weakest-cell route was also challenged with individual fusion
and accumulation toggles. Disabling global FP16 accumulation or substituting
the sequence-FFN path reduced the 0.4B/B8/P512 result to roughly
`157-161k tok/s`; disabling individual established fusions was neutral or
slower. None of those negative A/B variants was promoted.

## Files

- [`results.jsonl`](results.jsonl): 24 final candidate rows plus the 24
  unchanged full-FLA reference rows.
- [`correctness.jsonl`](correctness.jsonl): 8 graph-versus-eager P2048
  continuation and cache-handoff gates.
- [`summary.json`](summary.json): fail-closed machine-readable comparison.
- [`summary.md`](summary.md): rendered comparison table.

## Reproduce the gate

Candidate rows use `bench/bench_cross_model_speed_resident.py` with the same
model pairs and this shared shape contract:

```bash
python bench/bench_cross_model_speed_resident.py \
  --model /path/to/rwkv7-hf \
  --model-kind rwkv --model-role candidate \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 1.5b \
  --benchmark-matrix qwen35_5090_g1i_pd_sota \
  --dtype fp16 --quantization none --device cuda \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
  --decode-tokens 128 --prefill-chunk-size 512 \
  --warmup 2 --runs 5 --rwkv-code-source repo \
  --results /tmp/5090-g1i-pd-sota.jsonl

python bench/compare_qwen35_speed_matrix.py \
  --results /tmp/5090-g1i-pd-sota.jsonl --expected-cells 24 \
  --require-qwen-fast-path --require-qwen-full-fused \
  --required-reference-backend fla \
  --min-prefill-active-parameter-throughput-ratio 1.0 \
  --fail-on-gate
```

Reference reproduction and the full contract are documented in the prior
artifact linked above.
