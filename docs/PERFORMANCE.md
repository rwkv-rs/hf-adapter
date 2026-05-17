# Performance guide

Last audited: **2026-08-17**.

## Current backend

Native fused routes are the performance backend. FLA is retained only as an
explicit compatibility/reference route and correctness oracle.

Use these entry points:

```bash
python bench/bench_batch_sweep.py --help
python bench/bench_native_prefill_scan.py --help
python bench/bench_native_graph_overhead.py --help
python bench/bench_cross_model_speed.py --help
```

## Reporting

- Report Prefill and Decode separately in tok/s.
- B8 values are aggregate.
- Record raw samples, medians, exact model/GPU/runtime identity, peak VRAM and
  effective route telemetry.
- Use unrounded throughput for gates and ratios.
- Do not infer model quality from throughput.
- Do not promote a microbenchmark without end-to-end correctness and speed.

The latest cross-card P/D numbers are in
[`QWEN35_LATEST_P_D_TOKPS.md`](QWEN35_LATEST_P_D_TOKPS.md). Current raw bundles
are in [`../bench/CURRENT_ARTIFACTS.json`](../bench/CURRENT_ARTIFACTS.json).

Key retained kernel evidence includes RTX 4080 grouped B8 projection
([artifact](../bench/4080_b8_projection_bmm_20260809/README.md)), RTX 4080 7.2B
FP16 state ([artifact](../bench/4080_7p2b_fp16_state_20260809/README.md)), RTX
4090 route tuning ([artifact](../bench/4090_4080_routes_20260812/README.md)),
and RTX 5070 exact-card Native performance
([artifact](../bench/5070_max_perf_20260811/README.md)).
