# Benchmark workspace

`bench/` contains internal benchmark runners and the exact evidence bundles
supporting promoted repository claims. It is intentionally not included in the
`rwkv7-hf` wheel.

The two machine-readable sources of truth are:

- [`CURRENT_ARTIFACTS.json`](CURRENT_ARTIFACTS.json): retained evidence;
- [`BENCH_SCRIPTS.json`](BENCH_SCRIPTS.json): every executable benchmark,
  validator, analyzer, and probe.

[`INDEX.md`](INDEX.md) is the human-readable inventory.

## Layout

```text
bench/
├── benchlib/       shared loading, timing, environment, correctness and I/O
├── runners/        maintained orchestration and reusable benchmark programs
├── probes/         kernel, route and one-purpose microbenchmarks
├── validators/     fail-closed acceptance checks
├── analyzers/      comparison, reporting, selection and plotting tools
├── tools/          dataset, profile and environment construction helpers
├── _runs/          ignored scratch output; never promoted in place
└── *_YYYYMMDD/     immutable promoted evidence bundles
```

Only stable exact-card shell entry points remain at the `bench/` root. Python
programs use their categorized paths, for example:

```bash
python bench/runners/bench_native_prefill_scan.py --help
python -m bench.runners.bench_cross_model_speed --help
python bench/validators/check_dynamic_prefill_matrix.py --help
```

Direct-file and module execution are both supported.

## Current workflow

1. Select an existing runner or add a reusable categorized entry point.
2. Write unpromoted output below `bench/_runs/` or an explicit external path.
3. Create a new `<line>_<hardware>_<yyyymmdd>/` evidence directory only after
   the run is complete.
4. Include a README, exact environment/model identity, commands, raw rows,
   correctness results, and validator output.
5. Promote the bundle through `CURRENT_ARTIFACTS.json` and update `INDEX.md`.
6. Update `../BENCHMARK.md` and platform documentation only when the accepted
   state or numeric claim changed.

Evidence bundles are immutable after promotion. Raw measurements and logs must
never be rewritten. Explanatory prose may be repaired without changing the
recorded result.

## Shared infrastructure

New or actively maintained runners should use `bench.benchlib` rather than
copying harness code:

- `model_loader.py`: lazy `AutoModel` and tokenizer loading;
- `timing.py`: CUDA-event/wall timing, synchronization and environment scopes;
- `environment.py`: Git, Python, package and GPU identity;
- `correctness.py`: tensor, cosine and greedy comparisons;
- `results.py`: JSON/JSONL output;
- `gpu_guard.py`: fail-closed exact-product checks;
- `paths.py`: repository, benchmark and scratch-result paths.

Benchmark-only policy must remain here; production dispatch belongs in
`rwkv7_hf/` and requires exact-card evidence.

## Output contract

Formal runners should accept an explicit output directory or result path and
produce, as applicable:

```text
command.txt
environment.json
rows.jsonl
summary.json
validation.json
```

No runner may default to `bench/results.jsonl` or write raw logs directly into
the benchmark root. The shared fallback stream is
`bench/_runs/results.jsonl`.

## Promotion rules

- Record hardware, runtime, model hash, dtype, batch, prompt, decode, route,
  samples, correctness, and peak memory.
- Use fail-closed validators and unrounded values for pass/fail decisions.
- Default-on optimizations require correctness and non-negative end-to-end
  value across the declared scope.
- Keep negative evidence only when it is part of the current bundle and
  prevents a known regression.
- Delete obsolete scripts instead of accumulating a permanent `legacy/`
  directory; Git retains history.

## Local checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_benchmark_script_inventory.py \
  tests/test_current_benchmark_artifacts.py \
  tests/test_benchlib.py
python tests/test_markdown_links.py
git diff --check
```
