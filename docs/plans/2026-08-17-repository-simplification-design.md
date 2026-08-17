# Repository simplification design

> Lifecycle: **active implementation design**. After the cleanup lands, retain
> this file as a historical record of the evidence-retention boundary.

## Goal

Reduce the repository to the current RWKV-7 Hugging Face implementation,
current benchmark tooling, and one reviewable evidence bundle per active
hardware or feature line. Remove superseded benchmark snapshots, especially
RWKV FLA-era performance artifacts that have been replaced by Native results.

The cleanup must not change model math, public HF APIs, runtime defaults, or
benchmark acceptance thresholds.

## Boundary

This is a repository-maintenance change, not a backend rewrite.

Keep:

- the Native HF implementation and all current correctness/performance routes;
- the FLA compatibility/reference implementation, its tests, and the FLA
  oracle used by current Native correctness probes;
- current benchmark runners, validators, summarizers, and analyzers;
- one latest formal artifact per active hardware, quantization, training,
  parallelism, or correctness line;
- the frozen Qwen3.5 reference bundles needed by current paired comparisons;
- canonical user and reviewer documentation.

Remove:

- dated benchmark directories superseded by a newer formal artifact on the
  same line;
- RWKV FLA performance matrices superseded by Native paired matrices;
- intermediate tuning sweeps, negative probes, one-off regression snapshots,
  stale logs, and legacy aggregate result streams that are not a current gate;
- scripts whose only consumer is a removed historical artifact;
- dead documentation links and duplicated benchmark inventories.

## Evidence manifest

`bench/CURRENT_ARTIFACTS.json` is the source of truth for retained dated
evidence. Every retained directory has a stable line ID and purpose. A test
fails when a dated evidence directory is added without entering the manifest,
or when the manifest points at a missing directory.

The initial retained set is intentionally small but covers every active line:

- RTX 3090: paired Prefill/Decode and native quantization;
- RTX 4080: paired Prefill/Decode, B8 projection, and 7.2B FP16 state;
- RTX 4090: paired Prefill/Decode, exact route tuning, and current quantized B8
  matrices for small and 7.2B models;
- RTX 5070 Laptop: current Native exact-card performance and native quant
  loading;
- RTX 5090: frozen Qwen reference, paired Decode, W4 BN/TN, Native-vs-official
  inference, and real MiniPile training;
- V100: paired Prefill/Decode, dense/Albatross, FP16 state, packed MM4 decode,
  W4 prefill, and Transformers tensor parallelism;
- T4, AMD gfx1100, Apple M5, and Moore Threads S70: their latest accepted
  compatibility/performance evidence;
- MATH500: the final acceptance bundle only.

Apple production-close evidence remains in the existing top-level compact
JSONL set because that is its promoted bundle. Intermediate Apple and MATH500
experiments are removed.

## Documentation model

`BENCHMARK.md` remains the numeric source of truth. `docs/RESULTS_INDEX.md`
becomes the compact reviewer index, while `bench/INDEX.md` is generated from
the manifest and no longer lists every historical experiment. Historical
claims whose artifacts are removed are either deleted or relabeled as
unretained history without a repository-local reproduction claim.

## Code simplification

Benchmark code is pruned only after artifact pruning establishes real
consumers. Generic runners and current paired-matrix tooling stay. Hardware
wrappers that differ only by fixed arguments should share common helpers where
that can be done without changing command-line behavior or result schemas.

No production module is deleted merely because its name contains `fla`.
FLA-reference imports, compatibility loading, and current cross-backend probes
remain explicit supported code.

## Safety and verification

Deletion is allowlist-driven. Every exact target is resolved beneath the
isolated cleanup worktree before removal. The implementation is accepted only
when:

1. all retained artifact manifests and hashes still parse;
2. canonical Markdown links resolve;
3. benchmark contract and document-freshness tests pass;
4. HF API, cache, training, quantization, Native, and FLA-reference unit tests
   remain green;
5. no removed artifact or script is referenced by tracked current docs/tests;
6. the original user worktree remains untouched.
