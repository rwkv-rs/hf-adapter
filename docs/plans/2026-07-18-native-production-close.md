# Native Production Close Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining RTX 5090 native decode, prefill, inference-alignment, training-throughput, stability, and reproducibility gaps without weakening the existing correctness gates.

**Architecture:** Keep the HF model and FP32 recurrent-state contract unchanged, and add only fail-closed, exact-card kernel routes behind independent opt-in flags until every claimed shape passes repeated speed and numerical checks. Reuse the pinned official RWKV-Gradio-3 and train_temp implementations as same-checkpoint references, store raw machine-readable evidence, and promote no default unless its full claimed matrix is non-regressing.

**Tech Stack:** Python, PyTorch CUDA extensions, CUDA Graphs, Transformers, DeepSpeed, pytest, JSONL benchmark evidence, RTX 5090 (`sm_120`).

---

### Task 1: Freeze acceptance contracts

**Files:**
- Create: `docs/plans/2026-07-18-native-production-close.md`
- Modify: `tests/test_ada_lora.py`
- Modify: `tests/test_native_decode_bench_unit.py`

**Steps:**
1. Add tests that distinguish the existing Ada row limit from a Blackwell B8 route.
2. Add result-schema assertions for precision mode, official commit, repetitions, trace hash, and matched-shape ratios.
3. Run the focused tests and confirm the new policy test fails before implementation.
4. Implement only the policy surface needed by the failing tests.
5. Re-run the focused tests and commit with Wang Yue DCO metadata.

### Task 2: Close RTX 5090 B1/B8 decode parity

**Files:**
- Create: `rwkv7_hf/native_wkv_fp16.py`
- Modify: `rwkv7_hf/native_graph_runtime.py`
- Modify: `rwkv7_hf/native_jit.py`
- Modify: `bench/bench_native_model_decode.py`
- Create: `tests/test_native_wkv_fp16.py`
- Modify: `tests/test_native_graph_runtime_unit.py`
- Create: `bench/5090_native_decode_production_20260718/`

**Steps:**
1. Record the rejected Blackwell B8 W/A/G/V fusion probe; it regressed end-to-end and must not be promoted.
2. Add an independent default-off FP16 recurrent-state route so the Native and official fp16-state lanes use matched precision.
3. Compile and compare the fused recurrence/state/output kernel directly against the pinned official CUDA operation at B1/B8.
4. Verify graph capture, cache handoff, dynamic-batch reorder, extension activation, logits, and greedy traces on the real 7.2B checkpoint.
5. Run three fresh 512-token processes for B1 and B8 against the same official fp16-state harness.
6. Accept only if both shapes are at least official speed, all greedy traces agree, and numerical thresholds pass; retain FP32 state as the default production contract.

### Task 3: Add matched-precision prefill and cache handoff

**Files:**
- Modify: `bench/bench_native_prefill_scan.py`
- Modify: `rwkv7_hf/native_model.py`
- Modify: `rwkv7_hf/native_graph_runtime.py`
- Modify: `tests/test_native_prefill_scan.py`
- Modify: `tests/test_chunked_prefill.py`
- Create: `bench/5090_native_prefill_production_20260718/`

**Steps:**
1. Add a pinned official/native harness for B1/B8 and prompt 128/512/2048 using identical weights and state precision.
2. Verify prefill logits, final recurrent state, first cached decode token, peak VRAM, latency, and throughput.
3. Profile the failing shape before changing kernels.
4. Optimize only the dominant boundary and retain eager fallback for unsupported shapes.
5. Repeat each accepted shape in fresh processes and store raw rows plus a generated summary.

### Task 4: Direct official inference alignment

**Files:**
- Create: `scripts/compare_official_native_inference.py`
- Create: `tests/test_official_native_inference_alignment.py`
- Create: `bench/5090_native_official_alignment_20260718/`

**Steps:**
1. Instrument the pinned official Space code without changing its math to emit input, per-layer output, recurrent state, logits, and greedy-token digests.
2. Emit the same tensors from the native HF path for identical token batches and initial state.
3. Compare B1/B8 prefill and decode with explicit cosine, maximum-absolute, top-1, and state thresholds.
4. Fail on missing layers, mismatched shapes, non-finite values, or an unpinned official commit.
5. Store the exact command, environment, raw comparison, and concise summary.

### Task 5: Close train_temp throughput without changing training math

**Files:**
- Modify: `rwkv7_hf/train_temp_alignment.py`
- Modify: `rwkv7_hf/train_temp_cuda.py`
- Modify: `scripts/run_train_temp_native_recipe.py`
- Modify: `tests/test_train_temp_alignment.py`
- Modify: `tests/test_train_temp_cuda.py`
- Create: `bench/5090_native_train_temp_production_20260718/`

**Steps:**
1. Add synchronized CUDA timing for data, forward, backward, optimizer, and checkpoint boundaries.
2. Profile official and native B16/T512 recipes from fresh processes.
3. Remove only measured Python or launch overhead while preserving exact loss, all gradient tensors, and all parameter deltas.
4. Repeat throughput measurements and require native median throughput at least official median throughput.
5. Re-run backward and optimizer tensor alignment after every performance change.

### Task 6: Extend training stability and recovery evidence

**Files:**
- Modify: `scripts/run_train_temp_native_recipe.py`
- Modify: `scripts/run_zero_training_smoke.sh`
- Modify: `tests/test_train_temp_native_recipe.py`
- Modify: `tests/test_deepspeed_resume_smoke.py`
- Create: `bench/5090_native_training_stability_20260718/`

**Steps:**
1. Use a deterministic real-token corpus slice with recorded tokenizer and dataset digests.
2. Run at least three seeds with held-out validation, periodic state hashes, and finite-loss checks.
3. Run a longer uninterrupted job and matched interrupted/resumed job; compare model, optimizer, scheduler, RNG, and validation metrics.
4. Run ZeRO-2 and ZeRO-3 base plus resume on available multi-GPU hardware; keep exact-card status explicit if hardware is unavailable.
5. Record throughput, peak VRAM, convergence curves, and recovery equivalence in machine-readable outputs.

### Task 7: Expand exact-card matrix and publish evidence

**Files:**
- Modify: `AGENTS.md`
- Modify: `BENCHMARK.md`
- Modify: `HF_STATUS.md`
- Modify: `docs/performance/FUSED_BACKEND.md`
- Modify: `docs/TRAINING.md`
- Modify: `README.md`
- Modify: `README_ZH.md`

**Steps:**
1. Run the accepted B1/B8 paths on every locally available checkpoint that fits, then run other available exact cards without reusing card-local defaults blindly.
2. Generate summaries from raw rows and label unavailable hardware as unvalidated rather than passed.
3. Document commands, observable pass criteria, recovery, current limitations, and the single AI setup entry.
4. Run focused tests, the full local suite with known platform limitations separated, and remote CUDA tests.
5. Commit and push each independently reviewable patch with `Signed-off-by: wangyue <wangyue20060908@gamil.com>`.
6. Update PR #57 as Wang Yue, wait for CI, and mark ready only after all claimed gates are green.
