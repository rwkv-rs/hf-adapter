# Qwen3.5 V100 Speed Matrix Implementation Plan

> **Completed historical implementation plan.** The harness described here
> exists, but the original V100 Qwen rows used a Torch fallback and are not the
> current optimized-reference result. Use
> [`../../bench/v100_active_b1b8_20260715/README.md`](../../bench/v100_active_b1b8_20260715/README.md)
> for current V100/full-FLA acceptance and `HF_TODO.md` for remaining work.

**Goal:** Add a resumable HF benchmark that compares RWKV-7 against official text-only Qwen3.5 models and run its complete 216-cell fp16/bnb8/bnb4 matrix on V100.

**Architecture:** A single-row worker loads one model and precision in a fresh process, performs exact-shape prefill and cached decode, and appends one JSONL row.  An orchestrator expands model pairs and shape axes into candidate/reference subprocesses, while a pure-Python comparator joins rows and enforces coverage and minimum speed-ratio gates.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, bitsandbytes, JSONL, pytest-style standalone tests, SSH.

---

### Task 1: Specify row keys and comparison gates

**Files:**
- Create: `tests/test_qwen35_speed_matrix.py`
- Create: `bench/compare_qwen35_speed_matrix.py`

**Step 1: Write the failing test**

Create synthetic RWKV/Qwen rows that cover two cells, assert exact key joins,
coverage counts, minimum ratios, missing-row reporting, and `--fail-on-gate`
exit codes.

**Step 2: Run test to verify it fails**

Run: `python tests/test_qwen35_speed_matrix.py`
Expected: FAIL because the comparator does not exist.

**Step 3: Implement the minimal comparator**

Load JSONL, match `model_pair/prompt/decode/batch/dtype/quantization`, calculate
RWKV-over-Qwen prefill/decode ratios, and write JSON plus Markdown summaries.

**Step 4: Run test to verify it passes**

Run: `python tests/test_qwen35_speed_matrix.py`
Expected: `QWEN35 SPEED MATRIX TESTS PASS`.

### Task 2: Add the generic single-row worker

**Files:**
- Create: `bench/bench_cross_model_speed.py`
- Modify: `tests/test_qwen35_speed_matrix.py`

**Step 1: Add failing unit tests**

Test argument validation, deterministic synthetic prompt construction, model
metadata extraction, and failure-row schema without requiring CUDA or weights.

**Step 2: Implement model adapters**

Load RWKV with `AutoModelForCausalLM` and Qwen with
`Qwen3_5ForCausalLM`.  Support none/bnb8/bnb4, exact prompt length, repeated
batch input, timed prefill, cached greedy decode, memory telemetry, and explicit
failure JSON.

**Step 3: Verify locally**

Run: `python tests/test_qwen35_speed_matrix.py`
Expected: PASS without CUDA.

### Task 3: Add the resumable matrix orchestrator

**Files:**
- Create: `bench/run_qwen35_speed_matrix.py`
- Modify: `tests/test_qwen35_speed_matrix.py`

**Step 1: Add failing dry-run/resume tests**

Assert the default three model pairs produce 432 raw subprocess rows and 216
comparison cells, command lines preserve model kind and pair labels, and
existing pass/fail/skip keys are skipped.

**Step 2: Implement orchestration**

Expand axes, run candidate then reference in fresh subprocesses, append failure
rows, support `--dry-run`, `--skip-existing`, `--max-runs`, and `--fail-fast`.

**Step 3: Verify locally**

Run: `python tests/test_qwen35_speed_matrix.py`
Expected: PASS.

### Task 4: V100 smoke and complete run

**Files:**
- Create remotely: `/home/data/wangyue/projects/rwkv7-hf-adapter-qwen35`
- Create remotely: `/home/data/wangyue/bench/qwen35_v100_20260712/`

**Step 1: Probe the remote environment**

Confirm both V100s are idle, record driver/CUDA/PyTorch/Transformers/bnb/FLA,
and locate converted RWKV model directories.

**Step 2: Install a Qwen3.5-capable Transformers revision**

Use an isolated environment or upgrade only the dedicated benchmark environment.
Verify text-only Qwen3.5 2B load, prefill, cached decode, and no vision parameters.

**Step 3: Run one pair smoke**

Run 1.5B vs 2B, prompt128/decode8/bsz1/fp16 and inspect both JSON rows plus
the comparison summary.

**Step 4: Run the complete matrix**

Launch all 432 raw rows with immediate JSONL append and resume enabled.  Preserve
stdout/stderr logs and do not discard OOM/failure rows.

**Step 5: Generate acceptance artifacts**

Run the comparator for 216 expected cells, write JSON/Markdown summaries, and
report coverage, minimum ratios, red cells, runtime versions, and peak VRAM.

### Task 5: Repository verification and documentation

**Files:**
- Modify: `BENCHMARK.md`
- Modify: `HF_STATUS.md`
- Modify: `tests/test_markdown_links.py` only if new links require coverage changes

**Step 1: Document only measured evidence**

Add the exact V100 environment, commands, artifact paths, coverage, and failed
cells.  Do not claim unmeasured hardware or global Qwen superiority.

**Step 2: Run focused and repository tests**

Run: `python tests/test_qwen35_speed_matrix.py`

Run: `python tests/test_markdown_links.py`

Run: `python -m compileall bench tests rwkv7_hf`

Expected: all commands pass.

**Step 3: Submit the evidence PR**

Commit the generic harness, raw result artifact, fail-closed summary, exact
environment metadata, and canonical documentation updates together.
