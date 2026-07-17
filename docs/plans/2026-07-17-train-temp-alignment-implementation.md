# RWKV-LM train_temp Alignment Implementation Plan

> Historical implementation plan. Current user instructions and accepted
> evidence are in `docs/TRAIN_TEMP_CUDA.md` and the dated benchmark artifact.

**Goal:** Build and run a reproducible single-RTX-5090 train_temp-versus-HF numerical and convergence alignment harness.

**Architecture:** Keep train_temp-specific loss and optimizer semantics in one importable module, while a process-isolated benchmark runner owns official/HF loading, immutable inputs, fail-closed comparison and evidence.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, safetensors, official RWKV-LM train_temp, pytest, Bash, JSONL.

---

### Task 1: Lock Down train_temp Training Semantics

**Files:**
- Create: `rwkv7_hf/train_temp_alignment.py`
- Create: `tests/test_train_temp_alignment.py`

**Steps:**
1. Write failing tests for L2Wrap's argmax-logit gradient contribution.
2. Write failing tests for official-name and HF-name parameter grouping.
3. Implement L2Wrap without an in-place loss mutation.
4. Implement `1x`, `2x`, and decayed matrix parameter groups.
5. Run `python -m pytest -q tests/test_train_temp_alignment.py`.

### Task 2: Add Tensor and Provenance Gates

**Files:**
- Modify: `rwkv7_hf/train_temp_alignment.py`
- Modify: `tests/test_train_temp_alignment.py`

**Steps:**
1. Write failing tests for cosine, relative-L2, max-absolute, missing-key, and non-finite handling.
2. Add deterministic file, batch, and tensor hashing helpers.
3. Add official-to-HF gradient mapping through the converter's translation rules.
4. Run the focused unit tests and `tests/test_convert_config.py`.

### Task 3: Build the Process-Isolated Alignment Runner

**Files:**
- Create: `bench/bench_train_temp_alignment.py`
- Create: `tests/test_train_temp_alignment_runner.py`

**Steps:**
1. Test CLI validation and compare mode with tiny synthetic snapshots.
2. Implement `official`, `hf`, and `compare` backends.
3. Emit atomic JSON plus optional safetensors snapshots for gradients and parameter deltas.
4. Record GPU, driver, CUDA, PyTorch, source commits, checkpoint SHA, batch SHA, precision, seed, and exact optimizer groups.
5. Run runner unit tests without CUDA.

### Task 4: Add the Opt-in Official-Kernel CUDA Backend

**Files:**
- Create: `rwkv7_hf/train_temp_cuda.py`
- Create: `rwkv7_hf/csrc/train_temp/`
- Create: `tests/test_train_temp_cuda.py`

**Steps:**
1. Vendor the pinned Apache-2.0 official CUDA sources with provenance.
2. Compile and register the fused attention, FFN and L2Wrap loss operators lazily.
3. Keep the backend explicit and default-off; reject cache, padding and unsupported shapes.
4. Add a standard causal-LM loss helper and CPU-side contract tests.

### Task 5: Run RTX 5090 Numerical Gates

**Files:**
- Create remotely, then import: `bench/5090_train_temp_alignment_20260717/`

**Steps:**
1. Record the clean exact-card environment and official/HF commits.
2. Generate one official initialization checkpoint and one immutable batch.
3. Run the production bf16 forward, backward, and optimizer-step pairs.
4. Fix implementation mismatches and rerun only failed phases.
5. Keep fp32 checks limited to pure-PyTorch loss, grouping, hashing, and metric
   unit tests because current train_temp production CUDA operators require bf16.
6. Import complete raw evidence and a machine-readable summary.

### Task 6: Run Multi-Seed Short Convergence

**Files:**
- Extend: `bench/5090_train_temp_alignment_20260717/`

**Steps:**
1. Run a 100-step pilot for one seed to estimate runtime and detect instability.
2. If all numerical gates remain passing, run 1,000 steps for three seeds on both backends.
3. Compare final validation loss, curve area, gradient norms, and spikes.
4. Extend to 10,000 steps only when the 1,000-step evidence passes and the rental window permits.

### Task 7: Document and Publish Only Proven Claims

**Files:**
- Modify: `docs/TRAINING.md`
- Modify: `docs/TRAINING_WORKFLOWS.md`
- Modify: `docs/ACCEPTANCE.md`
- Modify: `BENCHMARK.md`

**Steps:**
1. Document copyable reproduction, observable gates, failure recovery, and current limits.
2. Clearly separate compatibility, numerical parity, and convergence status.
3. Run focused tests, the full local CPU suite, and `git diff --check`.
4. Commit as wangyue with DCO, push `wangyue/train-temp-alignment`, and create a draft PR.
