# Benchmark workspace

`bench/` contains current benchmark entry points and the exact evidence bundles
that support promoted repository claims. The retained evidence boundary is
machine-readable in [`CURRENT_ARTIFACTS.json`](CURRENT_ARTIFACTS.json) and
rendered for humans in [`INDEX.md`](INDEX.md).

Superseded tuning snapshots and RWKV FLA performance matrices are not retained.
FLA remains available as a compatibility/reference backend and as a correctness
oracle for current Native comparisons.

## Current workflow

1. Select an existing generic runner or add a reusable entry point.
2. Write results to a new `<line>_<hardware>_<yyyymmdd>/` directory.
3. Include a README, exact environment/model identity, commands, raw rows,
   correctness results, and the final validator output.
4. Promote the new bundle by replacing the older bundle for that line in
   [`CURRENT_ARTIFACTS.json`](CURRENT_ARTIFACTS.json).
5. Update [`INDEX.md`](INDEX.md), [`../BENCHMARK.md`](../BENCHMARK.md), and only
   the platform documents whose accepted state changed.

Evidence bundles are immutable after promotion. Fix explanatory prose only
when necessary; never rewrite raw measurements in place.

## Main entry points

- General Native performance: `bench_batch_sweep.py`,
  `bench_native_prefill_scan.py`, `bench_native_graph_overhead.py`.
- Cross-model matrices: `bench_cross_model_speed.py`,
  `bench_cross_model_speed_resident.py`.
- Quantization: `bench_native_quant_e2e_decode.py`,
  `bench_native_mm_quant_decode.py`, `bench_sm70_w4_bn_tn.py`.
- Training and quality: `bench_train_temp_alignment.py`,
  `run_math500_final_acceptance.py`.
- Current hardware-specific runners and validators are listed in
  [`INDEX.md`](INDEX.md).

## Promotion rules

- Record exact hardware, runtime, model hash, dtype, batch, prompt, decode,
  route, samples, correctness, and peak memory.
- Use fail-closed validators and unrounded values for pass/fail decisions.
- Default-on optimizations require correctness plus non-negative end-to-end
  value across the declared scope.
- Keep negative evidence only when it is part of the current artifact and
  prevents a known regression; do not accumulate historical probe directories.

## Local checks

```bash
python -m pytest -q tests/test_current_benchmark_artifacts.py
python tests/test_markdown_links.py
git diff --check
```
