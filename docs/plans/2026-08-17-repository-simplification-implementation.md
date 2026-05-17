# Repository simplification implementation plan

> Lifecycle: **active implementation plan**. Preserve as historical rationale
> after completion.

**Goal:** Replace the accumulated benchmark archive with a manifest-driven
current evidence set, remove superseded RWKV FLA-era artifacts and dead bench
tooling, and leave all active HF/Native/compatibility lines reproducible.

**Architecture:** A checked-in `bench/CURRENT_ARTIFACTS.json` defines the dated
artifact allowlist. Tests validate the manifest, directory inventory, required
files, and documentation references. Cleanup proceeds from artifacts to docs
to scripts so code is removed only after its consumers are known.

**Tech stack:** Python 3, pytest, PowerShell, Git, Markdown link checks.

---

## Task 1: Add the current-artifact contract

**Files:**

- Create: `bench/CURRENT_ARTIFACTS.json`
- Create: `tests/test_current_benchmark_artifacts.py`
- Modify: `bench/INDEX.md`

1. Encode stable line IDs, platforms, scopes, and exact retained paths.
2. Add a failing test requiring exact agreement between the manifest and every
   dated artifact directory under `bench/`.
3. Add checks for unique IDs/paths, existing README files, and sorted entries.
4. Render the lightweight evidence table in `bench/INDEX.md` from this set.

## Task 2: Remove superseded evidence

**Files:**

- Delete: all dated `bench/` directories not in the manifest
- Delete: unpromoted top-level `bench/results_*.jsonl` and stale logs
- Modify: `docs/RESULTS_INDEX.md`
- Modify: `BENCHMARK.md`
- Modify: hardware and validation docs that link removed artifacts

1. Resolve every deletion target and assert it is a direct child of this
   worktree's `bench/` directory.
2. Remove old RWKV FLA matrices, intermediate tuning sweeps, superseded exact
   card snapshots, and exploratory Apple/MATH500 directories.
3. Preserve current Qwen reference artifacts and all FLA-reference code/tests.
4. Remove or replace every dead link and stale evidence claim.
5. Run artifact-manifest, Markdown-link, comparison-layout, and document
   freshness tests.

## Task 3: Prune dead benchmark scripts

**Files:**

- Modify/Delete: `bench/*.py`, `bench/*.sh`, and `bench/*.ps1` proven to have
  no current artifact, documentation, test, or workflow consumer
- Modify: tests that enumerate benchmark entry points

1. Build a reference graph from retained artifact READMEs, canonical docs,
   tests, and workflows.
2. Keep generic current runners, Native profiling tools, paired Qwen runners,
   validators, analyzers, and compatibility probes.
3. Remove one-off historical runners and analyzers whose evidence was deleted.
4. Consolidate duplicate wrappers only when their external CLI and output
   schema remain unchanged.
5. Run import, CLI-help, benchmark-contract, and targeted unit tests.

## Task 4: Simplify documentation and repository navigation

**Files:**

- Modify: `README.md`
- Modify: `README_ZH.md`
- Modify: `HF_STATUS.md`
- Modify: `HF_TODO.md`
- Modify: `BENCHMARK.md`
- Modify: `docs/RESULTS_INDEX.md`
- Modify: `bench/INDEX.md`

1. Remove duplicated historical inventories and point readers to the manifest.
2. Keep the latest Qwen3.5 Prefill/Decode tables and reproduction commands.
3. State the FLA boundary once: compatibility/reference only for RWKV;
   production performance evidence is Native.
4. Verify English/Chinese links, table ordering, and throughput formatting.

## Task 5: Full verification and publication

**Files:**

- Modify: `CHANGELOG.md` if required by repository release policy

1. Run all fast CPU tests and all document/manifest tests.
2. Run the broader test suite, recording any hardware-only skips.
3. Run `git diff --check`, inspect deletion scope, and verify no unrelated
   user-worktree changes were touched.
4. Commit using the currently authenticated Git identity, push the
   `btlqql/repository-simplification` branch, and open a compliant PR.
