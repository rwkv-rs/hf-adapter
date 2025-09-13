# RWKV-7 HF Adapter — Benchmark Target

This file is the persistent benchmark contract for the RWKV-7 HF adapter work.
The goal is to iterate until the HF path approaches the official `rwkv` package
and Albatross-style paths in correctness, speed, and memory.

## Hardware currently measured

- Development server: **Tesla V100-PCIE-32GB**, CUDA fp16.
- Local dev box baseline from earlier PR: **NVIDIA RTX 5070 Laptop GPU**, fp16/bf16/fp32.
- Baseline model: **rwkv7-g1d-0.1b-20260129-ctx8192**.

## Acceptance targets for 0.1B smoke baseline

### 1. Precision

| Metric | Target |
|---|---:|
| top-5 token IDs match | 100% for fp32, high stability for fp16/bf16 |
| cosine similarity | >= 0.9999 |
| max abs logit diff | <= 0.05 for fp32 reference; dtype-aware for fp16/bf16 |
| greedy decode equality window | identical for >= 64 tokens |

### 2. Speed

| Metric | Target |
|---|---:|
| prefill tok/s | HF >= 0.9 x official comparable path |
| decode tok/s | HF >= 0.9 x official comparable path |

### 3. Memory

| Metric | Target |
|---|---:|
| peak VRAM | HF <= 1.1 x official comparable path |

## Current V100 status

Latest V100 runs are appended in `bench/results.jsonl`.

### Correctness / precision

Command:

```bash
python tests/test_official_alignment.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --official-strategy 'cpu fp32' \
  --greedy-window 64 \
  --fuse-norm false \
  --results bench/results.jsonl
```

Result on Tesla V100:

| Metric | Result | Status |
|---|---:|---|
| top5_match | 1.0000 | PASS |
| argmax_match | 1.0000 | PASS |
| cosine | 0.9999977 | PASS |
| max_abs_diff | 0.0718 | PASS for fp16 smoke; fp32 reference remains ≈0.030 |
| greedy window | 64 / 64 tokens | PASS |

Earlier fp32 reference on the 5070 Laptop produced `max_abs_diff≈0.030`, proving
that the adapter math and weight mapping are correct when dtype noise is removed.
The V100 optimized path uses `fuse_norm=false`; it preserves top-k/greedy behavior
and improves fp16 max-abs error versus the FLA fused-norm path.

### Save/reload roundtrip

Command:

```bash
python tests/test_reload_roundtrip.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --dtype fp16
```

Result:

| Metric | Result | Status |
|---|---:|---|
| reloaded logits max_abs_diff | 0.0 | PASS |

### High-level speed/memory, serving-style HF prefill

`bench/bench_speed.py` now measures HF prefill with `use_cache=True` and
`logits_to_keep=1`, which matches serving needs and avoids retaining full prompt
logits. The HF path now uses the adapter remote-code class and the lightweight
`RWKV7StateCache` hot path by default (`RWKV7_FAST_CACHE=1`).

Command:

```bash
python bench/bench_speed.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --device cuda \
  --warmup 2 \
  --runs 3 \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true
```

Result on Tesla V100:

| Backend | Prefill tok/s | Decode tok/s | Decode ms/tok | Peak VRAM |
|---|---:|---:|---:|---:|
| HF adapter, `fuse_norm=true` | 11852.0 | 31.5 | 31.70 | 406.4 MB |
| HF adapter, `fuse_norm=false` | 14247.7 | 41.3 | 24.24 | 406.4 MB |
| HF adapter, `fuse_norm=false`, `RWKV7StateCache` | 13801.4 | 41.2 | 24.28 | 406.4 MB |
| HF adapter, `rwkv7_forward_token` | 14055.1 | 59.2 | 16.89 | 406.4 MB |
| HF adapter, `rwkv7_forward_token`, `native_jit` backend | 13755.4 | 92.1 | 10.86 | 406.4 MB |
| HF adapter, `rwkv7_forward_token`, `native_graph` backend | 18386.6 | 255.5 | 3.91 | 643.7 MB |
| official `rwkv` | 225.6 | 92.1 | 10.86 | 406.2 MB |

Interpretation:

- **Memory target is met** for the 0.1B V100 serving-style path: HF is roughly equal to official.
- HF prefill is much faster than the official pure-torch reference path measured here.
- Disabling FLA fused norm for inference improved HF decode from `31.5` to about `41` tok/s (`+31%`).
- The lightweight `RWKV7StateCache` preserves exact logits/cache behavior and keeps the real remote-code `AutoModelForCausalLM` path at the same ~41 tok/s level while avoiding FLA CacheLayer bookkeeping.
- `RWKV7StateCache.select_batch` / `batch_select` now gives serving stacks a
  direct dynamic-batch compact/drop API; `reorder_cache` remains as the HF beam
  compatibility hook. `RWKV7StateCache.rwkv7_cache_metrics()` exposes
  update/select/reorder/offload counters and current cache shape telemetry.
- `RWKV7StateCache.detach()` and `to(device, dtype=None)` cover serving state
  offload/restore. V100 dynamic cache tests now compact active rows, detach the
  cache, move it to CPU, restore it to CUDA, and verify the next logits.
- **bsz=1 decode target is met** with the opt-in `native_jit` fast-token backend:
  standard optimized HF decode is about `0.45x` official, FLA fast-token reaches
  about `0.64x` official, and `RWKV7_FAST_TOKEN_BACKEND=native_jit` reaches
  `1.00x` official on this V100 run.
- `RWKV7_FAST_TOKEN_BACKEND=native_graph` moves the standalone CUDA-graph
  prototype into the HF `rwkv7_forward_token` API for fixed bsz and dynamic
  active-batch serving: bsz=1 reaches `255.5 tok/s` (`2.77x` official), with
  bsz=1/2/4/8 batch sweep rows shown below. Captured graph runners are kept in a
  per-model LRU controlled by `RWKV7_NATIVE_GRAPH_CACHE_SIZE`; serving code can
  call `rwkv7_clear_native_graph_cache()` to release retained graph buffers. The
  formal memory target remains anchored to the lower-memory native-JIT row.
  Native-graph replay overhead rows also record cache requests, hits, misses,
  evictions, retained batch sizes, and hit rate so serving cache reuse is a
  gated metric rather than an undocumented implementation detail.
- `RWKV7_FAST_TOKEN_BACKEND=auto` now resolves the effective fast-token backend
  at runtime as `native_graph` -> `native_jit` -> FLA, gated by CUDA/model
  placement, available native helpers, active batch size, and dense
  non-bitsandbytes weights. Benchmark scripts set the env var even when
  `--fast-token-backend auto` is used and write
  `fast_token_backend_effective` for regression analysis.
- `RWKV7_FAST_FORWARD=1` (default) routes ordinary eval/no-grad HF cached
  one-token `forward()` calls through `rwkv7_forward_token`, so
  `model.generate(..., use_cache=True)` gets the same auto-selected backend.
  Benchmark baseline loops explicitly set `RWKV7_FAST_FORWARD=0` around
  reference forward timing so historical forward-vs-fast comparisons stay
  comparable.

### Decode breakdown

Command:

```bash
python bench/bench_decode_breakdown.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --warmup 2 \
  --runs 3 \
  --attn-modes chunk fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --results bench/results.jsonl
```

Result on Tesla V100:

| Path | Prefill tok/s | Greedy decode tok/s | Fixed-token decode tok/s | Sampling overhead | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| HF `chunk`, `fuse_norm=true` | 11536.2 | 30.4 | 30.4 | 0.05 ms/tok | 439.7 MB |
| HF `chunk`, `fuse_norm=false` | 13343.7 | 38.2 | 38.0 | ≈0 ms/tok | 439.7 MB |
| HF `chunk`, `fuse_norm=false`, `RWKV7StateCache` | 13510.3 | 36.7 | 37.4 | 0.51 ms/tok | 439.7 MB |
| HF `fused_recurrent`, `fuse_norm=false` | 17192.8 | 38.3 | 38.2 | ≈0 ms/tok | 440.2 MB |
| HF `fused_recurrent`, `fuse_norm=false`, `RWKV7StateCache` | 17198.9 | 38.4 | 38.5 | 0.09 ms/tok | 440.2 MB |
| HF `fused_recurrent`, `rwkv7_forward_token` | 16571.8 | 52.9 | 53.0 | ≈0 ms/tok | 440.2 MB |
| official `rwkv` | 222.1 | 91.5 | n/a | n/a | 470.0 MB |

Interpretation:

- Greedy argmax/sampling overhead is negligible.
- `chunk` vs `fused_recurrent` does not materially change single-token decode.
- `fuse_norm=false` removes the expensive FLA `LayerNormFunction` path and improves decode, but does not remove the main gap.
- The fast token API reduces standard HF one-token decode from about `26 ms/token`
  to about `19 ms/token`, but the remaining gap is still inside the HF/FLA model
  + recurrent cache + per-token layer path, not in Python sampling.


### Decode profiler findings

Profiler commands were added via `bench/profile_decode.py`. On V100 fixed-token
decode, the original HF path spent most wall time in CPU dispatch/custom-function
overhead, not GPU math. The most important finding was:

- `fuse_norm=true`: FLA `LayerNormFunction` showed about `54.8 ms` CPU total over 6 active decode tokens.
- `fuse_norm=false`: native `aten::native_layer_norm` path reduced norm overhead to about `6.6 ms` CPU total over 6 active decode tokens.
- Result: high-level HF decode improved from `31.5` tok/s to `41.3` tok/s on V100.

The profile still shows thousands of tiny kernel launches per handful of decode
tokens, so the next optimization has to reduce/fuse the one-token layer path
rather than tune sampling.

## Reproducible V100 fast-decode validation

When the V100 server is reachable, run the committed bundle from the repository root:

```bash
./bench/run_v100_fast_decode_validation.sh
```

It runs `test_fast_decode_api.py`, `bench_speed.py --hf-decode-api rwkv7_forward_token`,
`test_batch_cache.py`, `test_dynamic_batch_cache.py`, `bench_batch_sweep.py`, `bench_dynamic_batch.py`, `bench_decode_breakdown.py --fast-decode-api true`, `bench_decode_micro.py`, `bench_forward_fast_path.py`, `bench_generate_fast_path.py`, `tests/test_device_map_generate.py` when at least two CUDA devices are visible, `bench_fast_token_warmup.py`, `bench_native_graph_overhead.py`, `bench_decode_components.py`, `bench_projection_lora.py`, `bench_fused_projection.py`, `bench_fused_wa_lora.py`, `bench_fused_wag_lora.py`, `bench_fused_shift_mix.py`, `bench_fused_recurrent.py`, `bench_native_graph_fused_recurrent.py`, `bench_native_quant_gemv.py`, `bench_native_quant_w4_gemv.py`, `bench_native_quant_rkv.py`, `bench_native_quant_w4_rkv.py`, `bench_larger_model_smoke.py` when the 0.4B/1.5B/2.9B/7.2B/13.3B paths exist, `bench_speculative_decode.py` when the target/draft HF dirs exist, `profile_decode.py --hf-decode-api rwkv7_forward_token`, `bench/analyze_results.py`, and `bench/check_results.py`,
then writes logs under `bench/logs/`. The bundle now also validates the
`native_jit` backend plus fixed-batch and dynamic `native_graph` fast-token
backends, and appends native HF speed rows before running the target gate. Use
`python bench/summarize_results.py --device V100
--last 12` for a compact view of the latest JSONL rows.

## Fast-token layout A/B harness

The validated fast-token path remains the default `3d` layout.  For candidate
one-token hot-path changes, the repository also includes an opt-in layout switch
and a V100 A/B bundle:

```bash
# Default baseline behavior.
RWKV7_FAST_TOKEN_LAYOUT=3d python bench/bench_speed.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --hf-decode-api rwkv7_forward_token \
  --fast-token-layout 3d

# Run 3d vs experimental 2d correctness + speed + microbench rows.
./bench/run_v100_fast_token_layout_ab.sh

# Resume only the missing candidate side after an interrupted/flaky-SSH run.
LAYOUTS=2d SPEED_BACKEND=hf ./bench/run_v100_fast_token_layout_ab.sh

python bench/compare_fast_token_layouts.py --results bench/results.jsonl --device V100 --dtype fp16 --require-candidate --min-speedup 1.0
```

Rows without `fast_token_layout` are treated as `3d` by
`bench/compare_fast_token_layouts.py`, so older V100 results remain the baseline
until new A/B rows are appended. Candidate rows are not accepted as an
optimization until `tests/test_fast_decode_api.py --fast-token-layouts 2d` passes
and the layout comparison command with `--require-candidate --min-speedup 1.0`
passes on V100.

## Batch-size coverage

The serving path now has a dedicated repeated-prompt batch smoke test:

```bash
python tests/test_batch_cache.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-sizes 1 2 4
```

The benchmark sweep records both aggregate and per-sequence throughput:

```bash
python bench/bench_batch_sweep.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto \
  --batch-sizes 1 2 4 8 \
  --results bench/results.jsonl
```

Latest V100 batch sweep:

| Batch | Forward total tok/s | Fast-token total tok/s | Fast-token per-seq tok/s |
|---:|---:|---:|---:|
| 1 | 40.0 | 56.4 | 56.4 |
| 2 | 79.1 | 111.3 | 55.7 |
| 4 | 156.6 | 221.0 | 55.3 |
| 8 | 312.9 | 441.3 | 55.2 |

## Dynamic-batch coverage

The dynamic-batch smoke test uses heterogeneous prompts, advances both batched
and per-row states, reorders the batched cache, then verifies the reordered next
logits against independently decoded rows:

```bash
python tests/test_dynamic_batch_cache.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-size 3 \
  --prompt-tokens 64
```

The benchmark simulation repeatedly reorders active rows and drops completed
rows from the recurrent state cache:

```bash
python bench/bench_dynamic_batch.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto \
  --decode-apis forward rwkv7_forward_token \
  --batch-size 8 \
  --min-batch-size 2 \
  --results bench/results.jsonl
```

This is not a full scheduler, but it gives a reproducible `axis=dynamic_batch`
signal for the cache operations needed by dynamic batching.

Latest V100 dynamic-batch simulation with native-JIT fast-token enabled:

| Decode API | Fast backend | Initial -> final batch | Reorders | Drops | Total tok/s | ms/token |
|---|---|---:|---:|---:|---:|---:|
| `forward` | n/a | 8 -> 4 | 32 | 4 | 214.8 | 4.6555 |
| `rwkv7_forward_token` | native-JIT | 8 -> 4 | 32 | 4 | 417.9 | 2.3931 |

## Decode microbench coverage

`bench_decode_micro.py` appends `axis=decode_micro` rows with stable per-component timings:

```bash
python bench/bench_decode_micro.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto \
  --steps 128 \
  --results bench/results.jsonl
```

The row records reference HF fixed/greedy one-token decode, ordinary HF
fixed/greedy decode with `RWKV7_FAST_FORWARD=1`, optional direct fast-token API
fixed/greedy decode, and isolated `lm_head`, `norm+lm_head`, `argmax`,
embedding, and empty-loop costs. This gives an easier regression signal than
profiler tables while keeping the profiler for operator-level investigation.

Latest V100 microbench:

| Component | ms/token | tok/s |
|---|---:|---:|
| Reference HF `forward` fixed-token (`RWKV7_FAST_FORWARD=0`) | 25.1180 | 39.8 |
| Ordinary HF `forward` fixed-token (`RWKV7_FAST_FORWARD=1`, auto->native_graph) | 3.9643 | 252.3 |
| Direct `rwkv7_forward_token` fixed-token (auto->native_graph) | 3.9494 | 253.2 |
| `lm_head` only | 0.1388 | 7205.2 |
| argmax only | 0.0249 | 40233.1 |

`bench_forward_fast_path.py` emits a smaller `axis=forward_fast_path` gate row
for the production-facing path. It compares `RWKV7_FAST_FORWARD=0` reference HF
forward, ordinary HF forward with fast-forward enabled, and direct
`rwkv7_forward_token`; `check_results.py` requires the ordinary HF fast path to
be at least `3.0x` faster than reference forward, at least `0.9x` of direct
fast-token speed, and within fp16 diff tolerance.

`bench_generate_fast_path.py` emits `axis=generate_fast_path` for the top-level
HF API. It compares greedy `model.generate(..., use_cache=True)` with
`RWKV7_FAST_FORWARD=0` and `1`; `check_results.py` requires identical generated
tokens, bsz>=2 coverage, a valid effective backend, and at least `2.0x`
end-to-end new-token throughput improvement. The recorded V100 prompt=8/new=16 bsz=2 row is `75.3 tok/s`
aggregate for reference generate vs `303.5 tok/s` aggregate with fast-forward
(`4.03x`), with `generated_equal=true`, `32/32` generated tokens matched,
and effective backend `native_graph`.

`rwkv7_warmup_fast_token()` exposes a public serving preflight API for native
fast-token resources. With `backend="auto"` it follows the same native-graph ->
native-JIT -> FLA resolution as `rwkv7_forward_token`; with
`backend="native_graph"` it raises if graph replay is unavailable. The paired
`rwkv7_native_graph_cache_batch_sizes()` API reports which active batch sizes
are currently retained in the per-model graph-runner LRU, and
`rwkv7_native_graph_cache_stats()` reports requests/hits/misses/evictions plus
hit rate for cache-reuse dashboards.

`bench_fast_token_warmup.py` emits `axis=fast_token_warmup`:

```bash
python bench/bench_fast_token_warmup.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto \
  --batch-sizes 1 2 4 8 \
  --native-graph-cache-size 8 \
  --results bench/results.jsonl
```

`check_results.py` now requires the warmup row to prove bsz=1/2/4/8 resolve to
`native_graph`, fit inside the configured graph cache, and are visible through
the cache-size inspection API before production traffic starts.

The native-graph runner now skips cache copies when the cache is already bound
to the graph runner's own buffers, which is the steady state for continuous
decode. `bench_native_graph_overhead.py` emits
`axis=native_graph_replay_overhead` to keep that wrapper overhead visible:

```bash
python bench/bench_native_graph_overhead.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-sizes 1 2 4 8 \
  --prompt-tokens 64 \
  --steps 32 \
  --fixed-token \
  --results bench/results.jsonl
```

Latest V100 rows for bsz=1/2/4/8: public API `254.9` / `449.8` / `858.5` /
`1546.9` aggregate tok/s, runner-vs-API max diff `0.0` for all rows, graph
replay `3.9375` / `4.4620` / `4.6760` / `5.1876ms`, and cache-copy share
`0.0703` / `0.0376` / `0.0361` / `0.0329`. `check_results.py` gates every
required batch size with a minimum API throughput, runner/API equality
tolerance, and maximum cache-copy share.

## Decode component benchmark

`bench_decode_components.py` instruments the fast-token path itself:

```bash
python bench/bench_decode_components.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fixed-token \
  --results bench/results.jsonl
```

It appends `axis=decode_components` rows with `component_ms`, `top_components`,
and `top_layers`. This bridges the gap between stable microbench rows and raw
profiler tables, and should be used to decide which per-layer operations to fuse
next.

Latest V100 component timing (instrumented, so use relative component weights
rather than the instrumented wall tok/s):

| Component group | ms/token |
|---|---:|
| attention linears + LoRA projections | 9.8695 |
| attention norm/correction/output projection | 4.5735 |
| recurrent kernel | 3.9276 |
| attention key mix/norm | 3.2613 |
| FFN key + ReLU square | 1.8493 |
| attention shift/mix | 1.7954 |

This makes the next optimization target concrete: reduce/fuse the many
one-token attention projection/LoRA calls first, then revisit output projection
and recurrent/norm groups.

## Projection/LoRA benchmark

`bench_projection_lora.py` drills into the largest component group:

```bash
python bench/bench_projection_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 projection/LoRA timing for sampled layers:

| Item | ms/layer |
|---|---:|
| R/K/V current separate projections | 0.0896 |
| R/K/V PyTorch bmm candidate | 0.0836 |
| W/A LoRA current | 0.1424 |
| W/A LoRA PyTorch bmm candidate | 0.2658 |
| Avg current linears+LoRA sum | 0.3502 |
| Avg PyTorch candidate sum | 0.4679 |

Interpretation: simple PyTorch bmm grouping is not enough (`0.75x` of current
overall for this group). R/K/V batched matmul is only a small win, while W/A
LoRA bmm is slower and can introduce larger fp16 numerical differences. The
next real optimization should be a custom fused projection/LoRA path or a
deeper rewrite that reduces launches without adding stack/bmm overhead.

Newer rows also emit `sample_matrix_profile`, `sample_matrix_profile_summary`,
and `fused_kernel_plan`. These fields turn the profiler into the first concrete
step of `FUSED_BACKEND.md`: they record matrix shapes, per-token FLOPs,
fp16/int8/int4 weight sizes, timed members, the first fp16 fusion target, and
the native-quant candidates that should later replace generic bnb kernels.

## Fused projection prototype

`rwkv7_hf/fused_projection.py` contains the first optional fp16 fused projection
prototype. `bench_fused_projection.py` times a single Triton R/K/V GEMV launch
against the current three separate projection linears:

```bash
python bench/bench_fused_projection.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_rkv_gemv` matches the current projections
with max abs diff `0.001953125` and min cosine `0.9999997`, but it is still
slower (`0.8429x` current linears, `0.11798ms` vs `0.09945ms`). This is useful
negative evidence: the first integration target should be a more optimized
shape-specialized/tensor-core-aware projection or a deeper fused time-mix path,
not this initial GEMV kernel.

## Fused W/A LoRA prototype

`rwkv7_hf/fused_lora.py` contains the first custom LoRA fusion probe for the
attention W/A pair. `bench_fused_wa_lora.py` times a grouped Triton
down/activation kernel plus a grouped up/bias kernel against the current W/A
LoRA modules:

```bash
python bench/bench_fused_wa_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 64 \
  --block-r 64 \
  --block-k 64 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_fused_wa_lora` is correctness-clean
(`max_abs_diff=0.015625`, `min_cosine=0.9999999`) but still slower than the
current W/A LoRA modules (`0.8601x`, `0.16883ms` vs `0.14521ms`). This is
negative evidence for standalone two-kernel LoRA grouping; the next LoRA attempt
should fuse deeper with R/K/V and other attention projection work.

## Fused W/A/G LoRA prototype

`bench_fused_wag_lora.py` extends the LoRA grouping probe from W/A to W/A/G.
This covers the larger attention LoRA bucket and supports mixed ranks (`w=64`,
`a=64`, `g=128` on the 0.1B V100 checkpoint):

```bash
python bench/bench_fused_wag_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 64 \
  --block-r 64 \
  --block-k 64 \
  --steps 512 \
  --results bench/results.jsonl
```

Latest stable V100 row: `triton_fused_wag_lora` is correctness-clean
(`max_abs_diff=0.0078125`, `min_cosine=0.99999994`) and is faster than the
current W/A/G LoRA modules (`1.0985x`, `0.26336ms` vs `0.28931ms`). This is the
first profitable LoRA grouping row, but it is still only a sub-kernel win; the
next performance step is to combine W/A/G with R/K/V projection and state/update
work so the full token path can move toward the Albatross ratios.

## Fused shift-mix prototype

`rwkv7_hf/fused_time_mix.py` contains an optional Triton prototype for the six
attention time-mix inputs used before RWKV-7 R/W/K/V/A/G projections. It is
measured separately because the native-graph decode path is launch-sensitive:

```bash
python bench/bench_fused_shift_mix.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --input-rank 2 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_attn_shift_mix` is exact (`max_abs_diff=0`)
with min cosine `0.9999999`, but it is slower than current torch pointwise ops
(`0.7715x`, `0.13416ms` vs `0.10351ms`). This rules out integrating a standalone
shift-mix kernel; the next fused fp16 attempt should combine shift-mix with the
following projection/LoRA/state-update work so one launch does more useful math.

## Fused recurrent prototype

`rwkv7_hf/fused_recurrent_update.py` contains an optional Triton prototype for
the one-token recurrent state update. It avoids materializing the rank-1
transition matrix and fuses state update plus readout:

```bash
python bench/bench_fused_recurrent.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_rank1_recurrent` is profitable versus the
current torch expression (`2.7931x`, `0.07841ms` vs `0.21901ms`) with
`out_max_abs_diff=0.0234375`, `state_max_abs_diff=0.0037985`, and
`out_min_cosine=0.9999998`. This is the first fused fp16 prototype worth
integrating behind the HF native-graph fast-token path, subject to full
end-to-end greedy/cache correctness gates.

## Native-graph fused recurrent integration

Set `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` to capture native-graph decode with
the recurrent Triton prototype. The graph-runner cache key includes this flag so
serving can switch the experiment on/off without accidentally reusing a graph
captured under the other mode.

```bash
python bench/bench_native_graph_fused_recurrent.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --results bench/results.jsonl
```

Latest V100 integration row: first-step logits are identical
(`max_abs_diff_first_step=0`, `min_cosine_first_step=1.0000002`) and greedy
tokens match `32/32`. End-to-end graph replay is currently neutral
(`1.0033x`, `4.2878ms` fused vs `4.3018ms` baseline), so this remains opt-in;
the isolated recurrent kernel is fast, but the captured full-token graph still
needs deeper fusion around the larger projection/LoRA bottleneck.

## Native W8 dequant-GEMV prototype

`rwkv7_hf/native_quant.py` contains the first RWKV-native W8 serving prototype:
row-wise int8 weight packing plus a fused dequant GEMV/GEMM. This is separate
from bitsandbytes; it is intended to become the native quant fast path after the
kernel is fast enough.

```bash
python bench/bench_native_quant_gemv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int8_rowwise_gemv` reduces the sampled
projection/FFN weight footprint to `0.502x` fp16 and keeps output cosine high
(`min_cosine=0.9999172`, `max_abs_diff=0.044921875`), but the first Triton
kernel is still slower (`0.3816x`, `0.05409ms` vs `0.02064ms`). This confirms
the native W8 packing direction while showing the kernel still needs a more
optimized/tensor-core-aware implementation before it can replace bnb or fp16.

## Native W4 dequant-GEMV prototype

`rwkv7_hf/native_quant.py` also contains the first RWKV-native W4 serving
prototype: row-wise signed int4 weight packing with two values per byte plus a
fused nibble-unpack/dequant GEMV/GEMM. This is telemetry-first and separate
from bitsandbytes.

```bash
python bench/bench_native_quant_w4_gemv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int4_rowwise_gemv` reduces the sampled
projection/FFN weight footprint to `0.252x` fp16. Correctness is usable as a
first W4 probe but visibly looser than W8 (`min_cosine=0.9802878`,
`max_abs_diff=0.9287109`). The first Triton kernel remains slower than fp16
cuBLAS (`0.359x`, `0.05773ms` vs `0.02072ms`), so W4 now has working pack,
fallback, fused-kernel telemetry, and analyzer visibility, but still needs a
faster packed reduction / fusion with projection groups before it can satisfy
the final `decode >= fp16` target.

## Native W8 fused R/K/V quant projection prototype

`bench/bench_native_quant_rkv.py` measures the next native quant step: the three
decode-hot attention R/K/V projections are computed from row-wise W8 weights in
one Triton launch, then compared with both fp16 linears and three separate
native W8 GEMVs.

```bash
python bench/bench_native_quant_rkv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int8_fused_rkv_gemv` keeps the R/K/V sampled
weight footprint at `0.5026x` fp16 and is bit-identical to the three separate
native W8 GEMVs (`max_abs_diff_separate_vs_fused=0`). It improves the separate
W8 GEMV path by `1.7628x` (`0.08878ms` fused vs `0.1565ms` separate) while
remaining below fp16 cuBLAS at `0.7847x` (`0.08878ms` fused vs `0.06967ms`
fp16). This shows launch/group fusion is the right direction for native quant,
but the next step must fuse more projection/LoRA work or specialize the packed
reduction further to clear the `>=1.0x fp16` target.

## Native W4 fused R/K/V quant projection prototype

`bench/bench_native_quant_w4_rkv.py` mirrors the W8 R/K/V fusion for packed W4
weights, using one Triton launch to unpack/dequantize and compute R/K/V.

```bash
python bench/bench_native_quant_w4_rkv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int4_fused_rkv_gemv` keeps the R/K/V sampled
weight footprint at `0.2526x` fp16 and is bit-identical to the three separate
native W4 GEMVs (`max_abs_diff_separate_vs_fused=0`). It improves separate W4
GEMVs by `1.7958x` (`0.0912ms` fused vs `0.16378ms` separate), with
`min_cosine_fp16_vs_fused=0.9750665`. Like the W8 fused R/K/V row, it is still
below fp16 cuBLAS (`0.7795x`), so launch fusion works but the final quant
target needs deeper fusion with LoRA/projection groups or a faster packed
reduction.

## Larger converted-model smoke

`bench_larger_model_smoke.py` proves the shape-inferred converter on real
checkpoints beyond the 0.1B development model. It loads each generated HF
directory with AutoConfig/AutoTokenizer/AutoModelForCausalLM, runs cached
forward, runs greedy generation, records config dimensions, checkpoint
provenance, backend selection, and memory.

```bash
python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --model-size-label 0.4b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 4 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-hf \
  --model-size-label 1.5b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-hf \
  --model-size-label 2.9b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-hf \
  --model-size-label 7.2b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-hf \
  --model-size-label 13.3b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend native_jit \
  --max-new-tokens 2 \
  --results bench/results.jsonl
```

Latest V100 larger-model rows:

| Model | hidden | layers | head_dim | value_dim | generated | backend | load s | generate s | footprint | peak VRAM |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| rwkv7-g1d-0.4b-hf | 1024 | 24 | 64 | 1024 | 4 | native_graph | 15.095 | 0.6751 | 859.8 MB | 1124.5 MB |
| rwkv7-g1g-1.5b-hf | 2048 | 24 | 64 | 2048 | 2 | native_graph | 27.991 | 0.6307 | 2913.3 MB | 3178.6 MB |
| rwkv7-g1g-2.9b-hf | 2560 | 32 | 64 | 2560 | 2 | native_graph | 35.589 | 0.7148 | 5622.4 MB | 5888.0 MB |
| rwkv7-g1g-7.2b-hf | 4096 | 32 | 64 | 4096 | 2 | native_graph | 66.292 | 0.7564 | 13731.3 MB | 13997.8 MB |
| rwkv7-g1g-13.3b-hf | 4096 | 61 | 64 | 4096 | 2 | native_jit | 99.107 | 0.7428 | 25309.1 MB | 25575.6 MB |

Checkpoint provenance is recorded in the rows: 0.4B SHA256
`947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb`, size
`901776749` bytes; 1.5B SHA256
`441f70b096ad62442b5c33128bfe717c5d8529915c45a9709d4482016e8a0482`, size
`3055444605` bytes; 2.9B SHA256
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`, size
`5896273469` bytes; 7.2B SHA256
`425fc9bda2d12d4ce3b6bfe5c3b3f355be8b14d85960cf40fcca58a19d632630`, size
`14400007869` bytes; 13.3B SHA256
`0aa686d3ca4bb486e83e3071f4798a210f960e1fc1f5042e6cb418cc463814d6`, size
`26540868485` bytes. The regression gate now requires all five smoke rows so
the converter cannot silently regress to 0.1B-only shape assumptions.

## Quantized inference coverage

`tests/test_quantized_inference.py` checks that the adapter loads and generates
through standard HF `BitsAndBytesConfig` paths:

```bash
python tests/test_quantized_inference.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantization 8bit

python tests/test_quantized_inference.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantization 4bit
```

`bench/bench_quantization.py` records comparable fp16 / 8-bit / 4-bit rows and
can compare the slower cached-HF reference decode against the HF fast-forward
path:

```bash
python bench/bench_quantization.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl
```

Latest short V100 rows:

| Quantization | Model footprint | Peak VRAM | Prefill tok/s | Reference decode tok/s | Fast-forward decode tok/s | Status |
|---|---:|---:|---:|---:|---:|---|
| none/fp16 | 364.4 MB | 636.2 MB | 8370.4 | 41.0 | 217.2 | PASS |
| 8-bit bnb + dense LoRA rank | 283.4 MB | 321.6 MB | 3226.6 | 15.9 | 16.3 | PASS smoke, speed gap |
| 4-bit bnb + dense LoRA rank | 242.9 MB | 286.4 MB | 6075.9 | 32.6 | 32.1 | PASS smoke, speed gap |
| 8-bit bnb `decode_hot` | 310.4 MB | 582.4 MB | 5406.3 | 25.6 | 27.0 | faster hybrid, speed gap |
| 4-bit bnb `decode_hot` | 283.4 MB | 310.0 MB | 7527.1 | 37.5 | 39.1 | faster hybrid, speed gap |

The adapter appends `lm_head` and `.*_lora\.lora\.[02]` to HF/bnb
`llm_int8_skip_modules` so tiny RWKV LoRA rank projections are not replaced
with inefficient quantized kernels, while the large projection/FFN weights
remain W8/W4. `bench_quantization.py` now records `quant_skip_modules`,
`module_counts`, and `selected_decode_path`; the latest row selects fast-forward
for 8-bit and reference cached decode for 4-bit because that path is slightly
faster on V100. The memory direction is correct, but selected W8/W4 decode is
still slower than fp16 native-graph decode, so production quantized serving
still needs a custom fused/native quantized projection path before it can meet
the original "not slower than fp16" target.

`RWKV7_BNB_SKIP_POLICY` / `--quant-skip-policy` adds explicit quantization
speed-memory policies:

- `memory` (default): keep only `lm_head` and tiny LoRA rank projections dense;
  this is the canonical memory-target row used by result gates.
- `decode_hot`: additionally keep attention `r_proj/k_proj/v_proj/o_proj`
  dense while FFN key/value remain quantized. Latest V100 rows improve selected
  decode to `27.0 tok/s` for 8-bit and `39.1 tok/s` for 4-bit while keeping
  footprint below fp16 (`310.4 MB` / `283.4 MB`). It is useful as a hybrid speed
  probe but still far below fp16 native-graph.
- `dense`: keep attention and FFN projections dense; diagnostic upper bound,
  effectively fp16 footprint.

Analyzer/check gates keep canonical quantization status anchored to `memory`
policy rows so hybrid probes do not accidentally overwrite W4 memory-target
evidence. The analyzer now also reports `quantization_best_variants`, selecting
the fastest passing policy per W8/W4 mode and comparing its decode and footprint
ratios against fp16.

### 0.4B V100 quantization sweep

Before refreshing older converted model dirs, run the code-only sync helper so
their remote-code wrappers include the latest quantization skip-policy support:

```bash
python scripts/sync_hf_adapter_code.py \
  /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf
```

Then benchmark the 0.4B model:

```bash
python bench/bench_quantization.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --model-size-label 0.4b \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit \
  --quant-skip-policy memory \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --decode-mode compare \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl
```

Latest V100 0.4B rows:

| Quantization | Policy | Model footprint | Peak VRAM | Prefill tok/s | Selected decode tok/s | Fast backend | Status |
|---|---|---:|---:|---:|---:|---|---|
| none/fp16 | memory | 859.8 MB | 1136.7 MB | 2117.8 | 107.0 | native_graph | PASS |
| 8-bit bnb | memory | 571.8 MB | 629.5 MB | 817.6 | 8.4 | FLA | PASS, speed gap |
| 4-bit bnb | memory | 427.8 MB | 502.6 MB | 1517.3 | 16.3 | FLA | PASS, speed gap |
| 8-bit bnb | `decode_hot` | 667.8 MB | 945.3 MB | 1362.4 | 13.7 | FLA | faster hybrid, speed gap |
| 4-bit bnb | `decode_hot` | 571.8 MB | 624.3 MB | 1882.3 | 19.6 | FLA | faster hybrid, speed gap |

`analyze_results.py` keeps the canonical quantization gate anchored to the 0.1B
baseline, and reports larger-model rows separately under
`quantization_model_sweep`. The 0.4B rows confirm memory decreases
substantially, but V100 decode speed is still far below fp16 native-graph; the
next quantization task remains a fused/native W8/W4 serving path.

## HF speculative decoding smoke

`rwkv7_speculative_generate()` is the initial HF-only speculative decoding
helper. It keeps the target and draft as ordinary HF models, proposes greedy
draft spans, verifies them with block target forwards, and reports acceptance
telemetry:

```bash
python tests/test_speculative_decode.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --max-new-tokens 8 \
  --draft-tokens 4
```

The default smoke uses the same model as target and draft, so every proposed
token should be accepted and the sequence must match greedy `generate()`.
Passing `--draft-model /path/to/smaller-hf-rwkv` exercises the same API with a
real draft model. The real-draft benchmark records the production gate row:

```bash
python bench/bench_speculative_decode.py \
  --target-model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --draft-model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --max-new-tokens 8 \
  --draft-tokens 4 \
  --results bench/results.jsonl
```

Latest V100 row: target greedy and speculative outputs match for 8/8 new
tokens, the 0.1B draft proposes 9 tokens, accepts 7, corrects 1, resyncs once,
replays 3 cache-resync tokens instead of 11 full-prefix tokens, and reports
acceptance `0.7777777777777778`. The short V100 row now reaches `2.1079x`
speedup over target greedy; next work is validating longer prompts and better
draft/block-size choices.

## HF RL / ZeRO training smoke

`tests/test_hf_rl_training_smoke.py` covers one-step LoRA preference/RL training
through common TRL trainers:

```bash
python tests/test_hf_rl_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --backend dpo

python tests/test_hf_rl_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --backend grpo \
  --grpo-max-completion-length 2
```

`configs/deepspeed/zero2.json` and `configs/deepspeed/zero3.json` are
HF Trainer-compatible ZeRO presets with auto micro-batch, gradient accumulation,
fp16/bf16, and bucket sizing. Validate them with:

```bash
python tests/test_deepspeed_configs.py
```

`tests/test_deepspeed_training_smoke.py` is the executable ZeRO training
harness. It loads the HF adapter through `AutoModelForCausalLM`, attaches PEFT
LoRA adapters, runs one or more HF `Trainer` steps with `deepspeed=zero2/zero3`,
checks that loss is finite, checks that trainable parameters changed, and emits
`deepspeed_training_smoke` rows for the analyzer:

```bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:$PYTHONPATH

python tests/test_deepspeed_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --results bench/results.jsonl
```

On machines without DeepSpeed or live GPUs, use `--optional --results
/tmp/rwkv7_zero_optional.jsonl` to record explicit skip rows while keeping local
analyzer/report tests green. Real pass rows remain a GPU follow-up item.

## Benchmark gap report

`bench/analyze_results.py` turns accumulated JSONL rows into a target/gap report:

```bash
python bench/analyze_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16
```

It reports HF-vs-official prefill/decode/memory ratios, best decode-breakdown
rows, fast-token API status, latest correctness row, batch/dynamic rows, decode
microbench rows, fast-token warmup and native-graph overhead rows, larger-model
smoke rows, quantization rows, `fused_backend_targets`, and a short next-focus
list. Current committed
V100 rows show:

| Metric | Current | Target | Status |
|---|---:|---:|---|
| speed_mem fast-token decode ratio (`native_jit`, bsz=1) | 1.00x official | >=0.90x | PASS |
| fast_decode best ratio (`native_graph`, bsz=1) | 2.77x official | >=0.90x | PASS |
| decode_breakdown fast-token ratio | ~0.57x official | >=0.90x | GAP |
| native_graph prototype decode ratio | ~2.76x official | >=0.90x | PASS prototype |
| native_graph warmup bsz=1/2/4/8 | cache contains 1/2/4/8 in 1.389s | preflight complete | PASS |
| native_graph replay overhead bsz=1/2/4/8 | API `255.1` / `449.8` / `857.2` / `1548.1` tok/s, max copy share `0.052`, hit rate `0.9737` | >=150 tok/s, <=0.15 copy share, >=0.80 hit rate | PASS |
| HF device_map generate smoke | 2 x V100, split layer 6, greedy tail matches single-device, fast backend skipped | >=2 CUDA devices, finite logits, greedy equality | PASS |
| speed_mem memory ratio | ~1.00x official | <=1.10x | PASS |
| 8-bit / 4-bit footprint ratio | 0.76x / 0.65x fp16 | lower is better | PASS smoke |
| 8-bit / 4-bit decode ratio | 0.24x / 0.67x fp16 | >=1.00x | GAP |
| Albatross V100 decode ratio | HF native-graph `0.32x`-`0.47x` Albatross faster3a for bsz=1/2/4/8 | approach Albatross | GAP |
| Albatross V100 prefill ratio | HF `0.32x` Albatross faster3a for B=1,T=512 | approach Albatross | GAP |
| Fused backend P1 decode ladder | analyzer target min ratio `>=0.55x` Albatross | `FUSED_BACKEND.md` P1 | GAP |
| Fused backend quant ladder | W8/W4 decode `>=1.0x` fp16 reference with W8 footprint `<=0.75x`, W4 footprint `<=0.55x` | native W8/W4 fused path | GAP |
| 0.4B converted-model smoke | hidden=1024, layers=24, generated=4, backend=native_graph | load + generate | PASS |
| 1.5B converted-model smoke | hidden=2048, layers=24, generated=2, backend=native_graph | load + generate | PASS |
| 2.9B converted-model smoke | hidden=2560, layers=32, generated=2, backend=native_graph | load + generate | PASS |
| 7.2B converted-model smoke | hidden=4096, layers=32, generated=2, backend=native_graph | load + generate | PASS |
| 13.3B converted-model smoke | hidden=4096, layers=61, generated=2, backend=native_jit | load + generate | PASS |

The current next-focus list is: run 13.3B official-alignment/speed sweeps,
validate newer GPUs, and solve the generic bnb quantized decode speed gap. The bsz=1 HF fast-token target is exceeded by `native_graph`;
bsz=2/4/8 native-graph serving now reaches `434.3` / `852.6` / `1539.1`
aggregate tok/s, and preflight warmup confirms graph runners are captured for
bsz=1/2/4/8 before the first serving request. The native-graph overhead rows
confirm the public API scales to `1548.1` aggregate tok/s at bsz=8 while
cache-copy overhead stays below `5.3%` of measured manual replay wall time and
graph-runner cache hit rate stays at `0.9737` for all required batch sizes. The
HF `device_map` row validates the multi-GPU pipeline-parallel direction on
2 x V100 by splitting the 0.1B model at layer 6; normal cached `generate()`
keeps finite logits, bypasses the single-device fast-token backend, and matches
the single-device greedy tail.

### Albatross A/B baseline

`bench/bench_albatross.py` ingests Albatross `RESULT B=... T=...` rows into the
same JSONL report used by the HF benchmarks:

```bash
python bench/bench_albatross.py \
  --engine faster3a \
  --engine-config wkv=fp32io16 \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --model-size-label 0.1b \
  --device-name 'Tesla V100-PCIE-32GB' \
  --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,1x512,2x1,4x1,8x1,16x1,32x1,2x2,4x4,8x8,16x16' \
  --warmup 1 \
  --iters 3 \
  --results bench/results.jsonl \
  -- --wkv fp32io16
```

On V100, Albatross faster3a's default fp16 WKV kernel uses `cp.async` and does
not compile for sm70, so the recorded V100 baseline uses `--wkv fp32io16`.
Latest 0.1B rows show Albatross decode `741.5` / `1354.5` / `2368.9` /
`3300.6` tok/s for bsz=1/2/4/8 and B=1,T=512 prefill `39472.6` tok/s.
The analyzer now reports HF native-graph ratios against those rows: decode
`0.34` / `0.32` / `0.36` / `0.47` for bsz=1/2/4/8 and prefill `0.32` for
B=1,T=512. This makes the Albatross gap explicit and keeps the next wrapper
optimization target measurable.

## Benchmark regression and target gates

`bench/check_results.py` turns the report into an executable gate:

```bash
# Passing regression gate for the current PR/V100 baseline.
python bench/check_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16

# Final acceptance gate for the current V100 0.1B HF fast-token target.
python bench/check_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16 \
  --target
```

Current committed V100 rows pass both the regression gate and the target gate.
The gate now uses the native-JIT HF fast-token speed row (`92.1 tok/s` vs
official `92.1 tok/s`) for the low-memory 0.1B bsz=1 target, while the
`fast_decode` section reports the optional native-graph row at `255.5 tok/s`.
It also requires passing 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B `larger_model_smoke` rows with checkpoint
SHA256 and generated-token evidence.

## Current optimization target

The next optimization work should focus on **HF recurrent decode**:

1. Continue beyond the first cache optimization: `RWKV7StateCache` removes generic
   FLA CacheLayer bookkeeping, but the remaining gap requires reducing per-layer
   tiny kernels and Python dispatch in the one-token path.
2. Inspect FLA `Cache.update`, per-layer state gather/update, token shift, group norm,
   and output projection overhead in the single-token path.
3. Profile one-token decode with `torch.profiler` / Nsight and compare against official
   `rwkv` package layer-by-layer. `profile_decode.py --hf-decode-api rwkv7_forward_token` profiles the fast token API directly.
4. Benchmark the new batched `rwkv7_forward_token` API with `bench_speed.py --hf-decode-api rwkv7_forward_token`, `bench_batch_sweep.py --fast-decode-api true`, and `bench_decode_breakdown.py --fast-decode-api true`; the V100 result is now stable enough that ordinary eval/no-grad HF `forward`/`generate` use the same path by default, while benchmarks can still disable it with `RWKV7_FAST_FORWARD=0` for reference timing.
5. Use `bench_batch_sweep.py` to keep bsz=1/2/4/8 regressions visible while optimizing the batched fast decode path.
6. Use `tests/test_dynamic_batch_cache.py` and `bench_dynamic_batch.py` to keep heterogeneous-row cache reorder/drop behavior correct while approaching serving-style dynamic batching.
7. Use `tests/test_chunked_prefill.py` and `bench_chunked_prefill.py` to keep long-prompt chunked prefill logits/cache compatible with full prefill while measuring the memory/throughput tradeoff.
8. Use `bench_decode_micro.py` to separate recurrent model cost from `lm_head`, argmax, and Python loop overhead before changing the decode implementation.
9. Use `bench_decode_components.py` to choose the next fusion target inside the fast-token layer path.
10. Use `bench_projection_lora.py` to verify projection/LoRA fusion candidates before changing model code.
11. Use `bench/analyze_results.py` after every V100 run to verify target ratios and missing axes before choosing the next optimization.
12. Use `bench/check_results.py` as the regression gate, and `bench/check_results.py --target` as the final performance gate.
13. Use `rwkv7_warmup_fast_token()` and `bench_fast_token_warmup.py` to remove
   first-request native-graph capture from serving latency before measuring
   production traffic.
14. Use `bench_native_graph_overhead.py` to keep cache-copy/bind overhead around
   the captured graph below the gate while optimizing dynamic serving paths.
15. Use `bench_speculative_decode.py` to keep real-draft greedy equality and
   acceptance telemetry gated while optimizing HF speculative decoding.
16. Keep `logits_to_keep=1` as the default serving benchmark path because it already
   fixes the earlier excess-memory measurement.
17. After V100 decode approaches official `rwkv`, rerun on newer GPUs and larger models.

## Loop state

- Correctness tests are now strong enough for 0.1B smoke: prompt logits, greedy 64,
  and save/reload roundtrip.
- Memory for the serving-style HF path is now at parity with official on V100.
- First V100 decode optimizations landed: `fuse_norm=false` plus the exact-match
  `RWKV7StateCache` keep the real remote-code HF path at ~41 tok/s vs official
  ~92 tok/s on V100.
- Batch correctness and sweep harnesses are in place; V100 native-JIT bsz=1/2/4/8
  fast-token decode runs at `91.5` / `195.3` / `374.5` / `647.3` aggregate tok/s.
- HF native-graph fast-token is now integrated for fixed bsz=1/2/4/8; V100
  speed_mem reaches `255.5 tok/s`, batch sweep reaches `253.9` / `434.3` /
  `852.6` / `1539.1` aggregate tok/s, and dynamic reorder/drop reaches
  `1209.3` total tok/s through the explicit cache select API while using the
  normal HF prefill/cache handoff. The graph runner cache is now an LRU over
  active batch sizes instead of a single most recent runner, so dynamic serving
  does not recapture when a retained size reappears.
- Dynamic-batch cache reorder/drop correctness and benchmark harnesses are in
  place; V100 tests now cover non-inplace reorder plus compact/drop through
  `select_batch` / `batch_select`, plus detach and CPU offload/restore before
  continuing decode.
- Chunked prefill helper, correctness test, benchmark, analyzer section, and
  regression gate are in place. V100 bsz=2 prompt=512 chunked prefill matches
  full prefill/decode within fp16 tolerance; chunk sizes 64/128/256 reduce peak
  VRAM to `0.598x` / `0.616x` / `0.633x` of full prefill while reaching
  `0.125x` / `0.252x` / `0.499x` of full-prefill throughput.
- Decode microbench harness is in place; V100 shows `rwkv7_forward_token` at
  `16.8 ms/token` vs HF `forward` at `24.5 ms/token`, while `lm_head` and argmax
  are tiny.
- Decode component harness is in place; V100 shows `attn_linears_lora` is the
  largest remaining fast-token component at about `9.87 ms/token`.
- Projection/LoRA harness is in place; V100 shows naive PyTorch bmm grouping is
  slower overall, so custom fusion is needed.
- Quantization smoke and benchmark harnesses are in place; V100 bnb 8-bit/4-bit
  loads pass and reduce model footprint, but current generic bnb decode is
  slower than fp16.
- Real-draft HF speculative benchmark is in place; V100 0.1B draft -> 0.4B
  target matches target greedy for 8/8 new tokens with 7/9 accepted proposals
  and one correction/resync; cached-prefix resync saves 8 token replays and the
  short V100 row reaches `2.1079x` speedup over target greedy.
- Benchmark gap analysis is in place and currently identifies decode throughput
  as the active optimization gap.
- Benchmark check gate is in place: current regression gate passes, target gate
  now passes after the opt-in HF `native_jit` fast-token backend reached
  `1.00x` official for the bsz=1 V100 speed row.
- Latest `main` added a native RWKV-7 decode experiment for 50-series / Blackwell:
  `rwkv7_hf/native.py`, `rwkv7_hf/native_jit.py`, and `bench/bench_batch.py`.
  This is valuable as a next V100 experiment because it attacks the same tiny
  kernel / dispatch bottleneck with a TorchScript block step and CUDA graph.
- Formal V100 native-decode row is now recorded: native JIT reaches `103.52 tok/s`
  and native CUDA graph reaches `254.33 tok/s` on the 0.1B V100 smoke model, with
  graph-vs-JIT greedy equality `16/16`.
- The active V100 blocker has moved from decode parity to additional
  larger-model/newer-GPU and quantized serving validation: bsz=1 native-graph HF
  is at `255.5 tok/s` vs official `92.1`, bsz=2/4/8 native-graph reaches
  `434.3`, `852.6`, `1539.1` aggregate tok/s in the latest sweep, and the real
  0.4B, 1.5B, 2.9B, 7.2B, and 13.3B converted HF directories now pass load/forward/generate smoke
  on V100; the 13.3B row uses native-JIT to avoid native-graph memory overhead on 32GB V100.

### Batched native-JIT fast-token results

Latest V100 `bench_batch_sweep.py --fast-token-backend native_jit` rows:

| bsz | HF forward total tok/s | native-JIT fast-token total tok/s | per-seq fast tok/s | step ms |
|---:|---:|---:|---:|---:|
| 1 | 41.4 | 91.5 | 91.5 | 10.92 |
| 2 | 84.0 | 195.3 | 97.7 | 10.24 |
| 4 | 167.0 | 374.5 | 93.6 | 10.68 |
| 8 | 331.5 | 647.3 | 80.9 | 12.36 |

Latest V100 `bench_batch_sweep.py --fast-token-backend native_graph` rows:

| bsz | HF forward total tok/s | native-graph fast-token total tok/s | per-seq fast tok/s | step ms |
|---:|---:|---:|---:|---:|
| 1 | 40.5 | 253.9 | 253.9 | 3.94 |
| 2 | 80.8 | 434.3 | 217.1 | 4.61 |
| 4 | 159.3 | 852.6 | 213.2 | 4.69 |
| 8 | 317.7 | 1539.1 | 192.4 | 5.20 |

Dynamic-batch reorder/drop with `RWKV7_FAST_TOKEN_BACKEND=native_graph` now
reaches `1209.3` total tok/s for `832` decoded tokens with active batch dropping
from 8 to 4, compared with the latest forward row at `211.7` total tok/s and
the previous native-graph row at `524.7` total tok/s. Both latest rows report
`cache_select_api=true` and `final_cache_batch_size=4`, so the result is using
the production-facing cache compact/select path rather than only the beam
reorder hook.

### Chunked prefill results

Latest V100 `bench_chunked_prefill.py --batch-size 2 --prompt-tokens 512` rows:

| mode | chunk | prefill tok/s | speed vs full | peak VRAM | VRAM vs full | max diff | decode diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | - | 36447.0 | 1.0000 | 658.9 MB | 1.0000 | - | - |
| chunked | 64 | 4566.4 | 0.1253 | 394.0 MB | 0.5980 | 0.09375 | 0.09375 |
| chunked | 128 | 9185.5 | 0.2520 | 405.8 MB | 0.6159 | 0.046875 | 0.0625 |
| chunked | 256 | 18178.9 | 0.4988 | 417.1 MB | 0.6330 | 0.125 | 0.03125 |

## Latest main native-decode context (50-series / Blackwell)

`rwkv7_hf/native_jit.py` ports the official `RWKV_x070_TMix_one`/`CMix_one`
per-token math natively (no FLA backend at decode time) and captures the whole
fixed-shape decode step in a CUDA graph. On the latest `main` branch, this path
was validated on RTX 5070 Laptop / Blackwell sm_120 and larger smoke models.

Decode speed (0.1B, RTX 5070 Laptop, fp16, single batch):

| path | tok/s | note |
|---|---:|---|
| FLA HF adapter (`generate`) | 37 | original wrapper path |
| native eager | 40 | direct Python native math |
| native + `torch.jit.script` | ~78 | full-block fused |
| native + CUDA graph | ~395 | about 4x official `rwkv` at 99 tok/s |

Correctness claims from the latest `main` branch:

- forward logits vs FLA: cosine 1.000000, max_abs approximately 0 at fp32.
- CUDA-graph greedy decode: 40/40 tokens identical to the JIT path.
- end-to-end vs `model.generate()` greedy: 32/32 generated tokens identical.

Usage:

```python
from rwkv7_hf.native_jit import fast_generate
print(fast_generate(model, tokenizer, "User: Hello!\n\nAssistant:", max_new_tokens=48))
```

Caveats for the HF adaptation target: the imported CUDA-graph path is currently
single-batch / fixed-shape greedy decode. Dynamic batching, PEFT/RL integration,
state-cache serving semantics, and V100 performance still need separate
validation before it can replace or augment the HF `forward` / `generate` path.

### V100 native JIT / CUDA graph validation

Command:

```bash
python bench/bench_native_decode.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 32 \
  --decode-tokens 64 \
  --greedy-check-tokens 16 \
  --results bench/results.jsonl
```

Result on Tesla V100-PCIE-32GB:

| Path | Decode tok/s | ms/token | Status |
|---|---:|---:|---|
| native JIT block step | 103.52 | 9.6596 | 1.12x official V100 baseline |
| native CUDA graph | 254.33 | 3.9319 | 2.76x official V100 baseline |

Correctness checks in the same row:

- native logits vs HF logits: cosine `1.00000024`, max_abs `0.03125`, argmax match.
- native CUDA graph greedy tokens vs native JIT greedy tokens: `16/16` identical.
- peak VRAM: `400.3 MB`, comparable to the official/HF 0.1B smoke rows.

Interpretation: this does not finish the full HF serving target because it is a
single-batch fixed-shape greedy path, but it gives a concrete implementation
direction: move the TorchScript block-step packing / graph-capture idea into the
HF fast-token API while preserving batched state-cache semantics.

### V100 experimental native-model telemetry

The FLA-free `NativeRWKV7ForCausalLM` remains an experimental fallback, not the
production wrapper replacement. Its smoke row is nevertheless tracked because it
is the long-term base for upstream Transformers, AMD/CPU, and small-shared-memory
training fallback work.

Command:

```bash
python tests/test_native_model.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --expect-jit-decode \
  --results bench/results.jsonl
```

Latest V100 row:

| Check | Result |
|---|---:|
| forward min cosine vs FLA wrapper | `0.99999976` |
| forward max abs diff | `0.00003815` |
| prompt argmax match | `3/3` |
| batched forward min cosine | `0.9999994` |
| batched cached-decode argmax match | `3/3` |
| greedy generate match | `16/16` |
| incremental cache exercised | `true` |
| cached decode backend | `native_jit` |

### Larger-model 50-series native results from latest `main`

| model | metric | FLA HF | official | native path |
|---|---|---:|---:|---:|
| 0.4B | decode tok/s | 11.5 | 26.0 | 174.7 CUDA graph, 6.7x official |
| 1.5B | decode tok/s | 13.3 | 30.7 | 26.6 JIT, 87% official |

Interpretation from latest `main`: the native CUDA-graph path wins strongly on
small launch-bound models, while larger models become compute/bandwidth-bound and
need a different serving-oriented fusion strategy. For the V100 branch, the next
useful step is to validate this native JIT/CUDA-graph path on the V100 0.1B model
and then decide whether to integrate its block-step packing into the HF fast-token
API.
