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
| HF adapter, `rwkv7_forward_token` | 13954.7 | 58.0 | 17.25 | 406.4 MB |
| official `rwkv` | 225.0 | 92.5 | 10.81 | 406.2 MB |

Interpretation:

- **Memory target is met** for the 0.1B V100 serving-style path: HF is roughly equal to official.
- HF prefill is much faster than the official pure-torch reference path measured here.
- Disabling FLA fused norm for inference improved HF decode from `31.5` to about `41` tok/s (`+31%`).
- The lightweight `RWKV7StateCache` preserves exact logits/cache behavior and keeps the real remote-code `AutoModelForCausalLM` path at the same ~41 tok/s level while avoiding FLA CacheLayer bookkeeping.
- **Decode is still not met**, but the fast token API improves the serving path:
  standard optimized HF decode is about `0.45x` official, while
  `rwkv7_forward_token` reaches about `0.64x` official on this V100 run.

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
`test_batch_cache.py`, `test_dynamic_batch_cache.py`, `bench_batch_sweep.py`, `bench_dynamic_batch.py`, `bench_decode_breakdown.py --fast-decode-api true`, `bench_decode_micro.py`, `profile_decode.py --hf-decode-api rwkv7_forward_token`, and `bench/analyze_results.py`,
then writes logs under `bench/logs/`. Use `python bench/summarize_results.py --device V100 --last 12` for a compact view of the latest JSONL rows.

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
  --decode-apis forward rwkv7_forward_token \
  --batch-size 8 \
  --min-batch-size 2 \
  --results bench/results.jsonl
```

This is not a full scheduler, but it gives a reproducible `axis=dynamic_batch`
signal for the cache operations needed by dynamic batching.

Latest V100 dynamic-batch simulation:

| Decode API | Initial -> final batch | Reorders | Drops | Total tok/s | ms/token |
|---|---:|---:|---:|---:|---:|
| `forward` | 8 -> 4 | 32 | 4 | 205.2 | 4.8734 |
| `rwkv7_forward_token` | 8 -> 4 | 32 | 4 | 345.7 | 2.8930 |

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

The row records standard HF fixed/greedy one-token decode, optional fast token API fixed/greedy decode, and isolated `lm_head`, `norm+lm_head`, `argmax`, embedding, and empty-loop costs. This gives an easier regression signal than profiler tables while keeping the profiler for operator-level investigation.

Latest V100 microbench:

| Component | ms/token | tok/s |
|---|---:|---:|
| HF `forward` fixed-token | 24.9691 | 40.0 |
| `rwkv7_forward_token` fixed-token | 17.5097 | 57.1 |
| `lm_head` only | 0.1639 | 6103.0 |
| argmax only | 0.0266 | 37525.1 |

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
microbench rows, and a short next-focus list. Current committed V100 rows show:

| Metric | Current | Target | Status |
|---|---:|---:|---|
| speed_mem fast-token decode ratio | ~0.64x official | >=0.90x | GAP |
| decode_breakdown fast-token ratio | ~0.57x official | >=0.90x | GAP |
| speed_mem memory ratio | ~1.00x official | <=1.10x | PASS |

The current next-focus list is: continue reducing tiny kernels/dispatch in the
fast token path. The profiler still shows thousands of launches over a small
active decode window.

## Current optimization target

The next optimization work should focus on **HF recurrent decode**:

1. Continue beyond the first cache optimization: `RWKV7StateCache` removes generic
   FLA CacheLayer bookkeeping, but the remaining gap requires reducing per-layer
   tiny kernels and Python dispatch in the one-token path.
2. Inspect FLA `Cache.update`, per-layer state gather/update, token shift, group norm,
   and output projection overhead in the single-token path.
3. Profile one-token decode with `torch.profiler` / Nsight and compare against official
   `rwkv` package layer-by-layer. `profile_decode.py --hf-decode-api rwkv7_forward_token` profiles the fast token API directly.
4. Benchmark the new batched `rwkv7_forward_token` API with `bench_speed.py --hf-decode-api rwkv7_forward_token`, `bench_batch_sweep.py --fast-decode-api true`, and `bench_decode_breakdown.py --fast-decode-api true`; if the V100 result is stable, use it as the serving-stack fast path while keeping HF `forward`/`generate` compatibility unchanged.
5. Use `bench_batch_sweep.py` to keep bsz=1/2/4/8 regressions visible while optimizing the batched fast decode path.
6. Use `tests/test_dynamic_batch_cache.py` and `bench_dynamic_batch.py` to keep heterogeneous-row cache reorder/drop behavior correct while approaching serving-style dynamic batching.
7. Use `bench_decode_micro.py` to separate recurrent model cost from `lm_head`, argmax, and Python loop overhead before changing the decode implementation.
8. Use `bench/analyze_results.py` after every V100 run to verify target ratios and missing axes before choosing the next optimization.
9. Keep `logits_to_keep=1` as the default serving benchmark path because it already
   fixes the earlier excess-memory measurement.
10. After V100 decode approaches official `rwkv`, rerun on newer GPUs and larger models.

## Loop state

- Correctness tests are now strong enough for 0.1B smoke: prompt logits, greedy 64,
  and save/reload roundtrip.
- Memory for the serving-style HF path is now at parity with official on V100.
- First decode optimizations landed: `fuse_norm=false` plus the exact-match `RWKV7StateCache` keep the real remote-code HF path at ~41 tok/s vs official ~92 tok/s on V100.
- Batch correctness and sweep harnesses are in place; V100 bsz=1/2/4/8 fast-token decode runs at about `55 tok/s` per sequence.
- Dynamic-batch cache reorder/drop correctness and benchmark harnesses are in place; V100 dynamic simulation improves from `205.2` to `345.7` total tok/s with `rwkv7_forward_token`.
- Decode microbench harness is in place; V100 shows `rwkv7_forward_token` at `17.5 ms/token` vs HF `forward` at `25.0 ms/token`, while `lm_head` and argmax are tiny.
- Benchmark gap analysis is in place and currently identifies decode throughput as the active optimization gap.
- The active blocker remains decode throughput: fast-token HF is now ~0.64x official on V100, still below the 0.90x target.
