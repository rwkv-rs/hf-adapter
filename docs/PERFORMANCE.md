# Production performance status

This page contains only promoted current conclusions. Exploratory tuning and
historical rows remain in platform documents and `bench/` artifacts.

## Current promoted lanes

| Platform | Dense fp16/bf16 | Quant speed lane | Quality/correctness | Status |
|---|---|---|---|---|
| RTX 5070 Laptop | 1.5B RWKV vs full-FLA Qwen3.5 2B bsz8 prefill/decode minimum `1.082707x/1.795119x` | fp16/W8/W4 all pass; footprint and peak VRAM lower in 18/18 | Qwen full-FLA bindings; Qwen and RWKV greedy/cosine probes pass | Production-close for measured bsz8 lane |
| V100 | Decode `0.908x–1.248x`, prompt-512 prefill `0.930x–1.047x` same-host Albatross | W8/W4 decode `1.006x–1.128x` fp16; paired prefill `0.996x–1.007x` | Greedy/cache handoff and focused regressions pass | Production-close for canonical matrix |
| RTX 4090 | 0.4B decode bsz1/2/4/8 `1.007x–1.418x` matching Albatross | W8/W4 measured speed lanes are fp16/bf16 equivalent or faster | 32-step greedy and cache handoff pass | Production-close for measured lanes |
| RTX 5090 | 0.4B MATH500 generation `16,925.6 tok/s`, steady decode `19,339.5 tok/s` | 2.9B/7.2B pressure rows all `>=0.99x` paired fp16; combined matrix `>=0.98x` | pass@64 `0.38`, compression ratio `1.0`, same-next all quant rows | Production-close artifact |
| Apple M5 | Tiled DPLR and guarded compiled decode close selected same-device Qwen3.5 gates | W4 lowers memory; selected production pair gates pass | target-greedy oracle and state/session checks pass | Production-close for measured MLX pairs |

## Interpretation rules

1. Compare the same model/checkpoint, dtype, prompt/decode shape and device.
2. Prefer paired same-process timing for quant-vs-fp comparisons.
3. Preserve both current-session and historical high-water references.
4. A load/generate smoke is not a performance result.
5. Aggregate batch throughput and per-sequence latency must not be conflated.
6. MATH500 speed claims must retain shape, seed, rollout count and accuracy gates.

## Remaining performance work

- Fuse the full-memory W8/W4 projection path instead of limiting the fastest
  policy to selected modules such as `lm_head`.
- Extend P2/P3 Albatross matrices to larger models and more hardware.
- Rerun the final Albatross workload on the same RTX 5090 session; the current
  MATH500 comparison uses the committed reference.
- Recover the retained RTX 4090 historical prompt-512 prefill high-water mark.
- Add H100 and AMD/ROCm production evidence.
- Broaden Apple results beyond M5 and complete CoreML/ANE production telemetry.

## Reproduction entrypoints

- General speed: `bench/bench_speed.py`, `bench/bench_batch_sweep.py`
- TTFT/TPOT: `bench/bench_ttft_tpot.py`
- Albatross ingestion/comparison: `bench/bench_albatross.py`
- Native quant matrix: `bench/run_blackwell_quant_matrix.py`
- MATH500 final runner: `bench/run_math500_final_acceptance.py`
- Apple same-device runner: `scripts/run_qwen35_apple_acceptance.sh`

Numeric summary: [`../BENCHMARK.md`](../BENCHMARK.md). Kernel roadmap:
[`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md).
