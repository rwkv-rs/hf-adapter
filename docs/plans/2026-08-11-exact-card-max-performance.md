# Exact-Card Maximum Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Maximize reproducible RWKV-7 HF prefill and cached-decode performance on the exact RTX 5070 Laptop and V100 profiles while preserving HF, cache, state, logits, and greedy correctness.

**Architecture:** Use the existing native/no-FLA model, native graph runtime, benchmark harnesses, and fail-closed kernel policy. Establish an immutable baseline, select one measured hotspot at a time, benchmark every candidate through opt-in overrides, and promote only exact-card/model/batch shapes whose repeated full-model A/B rows pass.

**Tech Stack:** Python 3.12, PyTorch/CUDA, Triton, Transformers, native CUDA graph runtime, pytest, JSONL benchmark artifacts, RTX 5070 Laptop sm120, Tesla V100 sm70.

---

### Task 1: Freeze the Clean Current-Main Contract

**Files:**
- Verify: `rwkv7_hf/kernel_policy.py`
- Verify: `tests/test_kernel_policy.py`
- Verify: `tests/test_native_graph_cache.py`
- Verify: `tests/test_native_jit_graph_dispatch_split.py`

**Step 1: Record repository and environment identity**

Run:

```powershell
git status --short --branch
python examples/check_environment.py --model /path/to/rwkv7-g1d-0.4b-hf
nvidia-smi
```

Expected: clean branch, environment check PASS, exact RTX 5070 Laptop name and
`sm_120` visible.

**Step 2: Run the focused offline policy suite**

Run:

```powershell
python -m pytest tests/test_kernel_policy.py tests/test_native_jit_graph_dispatch_split.py tests/test_native_graph_runtime_unit.py -q
```

Expected: PASS.

**Step 3: Run the focused GPU cache and generation contracts**

Run:

```powershell
python -m pytest tests/test_native_graph_cache.py tests/test_hf_api_contract.py -q
```

Expected: PASS, allowing documented hardware-specific skips only.

### Task 2: Capture the RTX 5070 Current-Policy Baseline

**Files:**
- Create: `bench/5070_max_perf_20260811/README.md`
- Create: `bench/5070_max_perf_20260811/environment.json`
- Create: `bench/5070_max_perf_20260811/baseline.jsonl`
- Reuse: `bench/bench_batch_sweep.py`

**Step 1: Capture hardware and software metadata**

Record the exact GPU, driver, power limit, clocks, temperature, WDDM mode,
PyTorch, CUDA, Triton, Transformers, free VRAM, repository commit, and command
line in `environment.json`. Do not record private absolute model paths.

**Step 2: Run the 0.4B baseline**

Run three independent processes with repository code, FP16, native graph,
B1/B2/B4/B8, prompt 128, decode 128, synchronized warmup, and JSONL output.

Expected: all rows select `native_graph`, pass greedy checks, and report
prefill/decode/peak-memory/policy telemetry.

**Step 3: Extend to 1.5B and 2.9B**

Run the same matrix where memory permits. Reduce only the batch cells that do
not fit; do not change prompt/decode shapes to hide a capacity failure.

**Step 4: Validate result structure**

Run:

```powershell
python tests/test_result_tools.py
python bench/analyze_results.py --input bench/5070_max_perf_20260811/baseline.jsonl
```

Expected: analyzer PASS and all required hardware/policy fields present.

### Task 3: A/B the Grouped W/A/G/V Candidate on RTX 5070

**Files:**
- Reuse: `bench/bench_native_graph_ada_wagv_lora.py`
- Create: `bench/5070_max_perf_20260811/grouped_wagv_ab.jsonl`
- Test: `tests/test_ada_lora.py`
- Test: `tests/test_native_jit_graph_dispatch_split.py`

**Step 1: Run the existing kernel tests**

Run:

```powershell
python -m pytest tests/test_ada_lora.py tests/test_native_jit_graph_dispatch_split.py -q
```

Expected: PASS.

**Step 2: Run 0.4B opt-in A/B**

Use `bench_native_graph_ada_wagv_lora.py --axis ada_wagv_lora` at B1 and B8,
prompt 64, 16 warmups, 512 fixed-token timing steps, and 128 greedy correctness
steps. Run three independent model loads. Confirm route telemetry and extension
load state so an environment-enabled fallback cannot be counted as a candidate.

Expected: first-step logits pass, greedy `3072/3072`, and complete baseline and
candidate timing/memory/policy telemetry.

**Step 3: Extend only after the 0.4B gate passes**

Repeat for 1.5B and 2.9B. Reject the candidate for any model whose median is
below `1.01x` or any repeated run is below `1.00x`.

### Task 4: Promote a Passing RTX 5070 Grouped W/A/G/V Route

**Files:**
- Modify: `rwkv7_hf/kernel_policy.py`
- Modify: `tests/test_kernel_policy.py`
- Modify: `tests/test_native_jit_graph_dispatch_split.py`
- Modify: `bench/5070_max_perf_20260811/README.md`

**Step 1: Write failing exact-card isolation tests**

Add tests that expect the route only for exact `NVIDIA GeForce RTX 5070 Laptop
GPU`, FP16, B8, and the passing measured hidden/rank shapes. Assert false for
5070 desktop variants, 5070 Ti, RTX 4080-adjacent names, B1/B2/B4, BF16, and
unmeasured hidden sizes.

**Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_kernel_policy.py tests/test_native_jit_graph_dispatch_split.py -q
```

Expected: FAIL only on the new 5070 assertions.

**Step 3: Implement the minimal fail-closed policy**

Add only the exact-card/model-shape gate proven by Task 3. Preserve environment
override precedence and existing RTX 4080 behavior.

**Step 4: Run focused and cross-card tests**

Run:

```powershell
python -m pytest tests/test_kernel_policy.py tests/test_ada_lora.py tests/test_cross_card_runtime_isolation.py tests/test_native_jit_graph_dispatch_split.py tests/test_native_graph_cache.py -q
```

Expected: PASS.

**Step 5: Commit the atomic policy change**

Commit only implementation, tests, and its exact evidence summary.

### Task 5: Rebaseline and Profile the New RTX 5070 Default

**Files:**
- Create: `bench/5070_max_perf_20260811/promoted_baseline.jsonl`
- Create: `bench/5070_max_perf_20260811/profile.jsonl`
- Reuse: `bench/bench_batch_sweep.py`
- Reuse: native prefill/decode breakdown tools under `bench/`

**Step 1: Repeat the full Task 2 matrix**

Expected: promoted cells reproduce within 3%, no non-B8 regression, and all
correctness gates remain passing.

**Step 2: Capture a synchronized component breakdown**

Measure projection groups, recurrent update, norm/mix, FFN, output preparation,
graph overhead, and prefill kernels on 0.4B/1.5B/2.9B at the dominant B1 and B8
cells.

**Step 3: Select exactly one next hotspot**

Choose the largest reusable component that is not already a vendor GEMM at its
measured roofline. Document rejected candidates before implementing another
kernel.

### Task 6: Iterate Decode, Prefill, and Quantized Candidates

**Files:**
- Modify: the single native kernel/runtime module owning the selected hotspot
- Modify: the matching focused test module
- Create: one candidate JSONL file under `bench/5070_max_perf_20260811/`

**Step 1: Write a numerical oracle test**

Compare the candidate with the readable native/reference path across supported
and fallback shapes, including state and long greedy behavior.

**Step 2: Implement an opt-in candidate**

Keep default behavior unchanged and expose route telemetry.

**Step 3: Run microbench screening**

Reject candidates that do not improve the isolated hotspot or materially raise
memory/register pressure.

**Step 4: Run three-process end-to-end A/B**

Promote only at `>=1.01x` median with no repeated regression and complete
correctness/memory evidence.

**Step 5: Repeat Tasks 5-6 until the plateau rule is met**

Expected: the final artifact identifies the remaining hardware-bound component
and records all rejected candidates.

### Task 7: Execute the V100 Exact-Card Loop

**Files:**
- Create: `bench/v100_max_perf_20260811/README.md`
- Create: `bench/v100_max_perf_20260811/environment.json`
- Create: `bench/v100_max_perf_20260811/baseline.jsonl`
- Modify: V100 kernel policy/tests only after exact-card evidence passes

**Step 1: Establish remote execution without storing credentials**

Use an already configured SSH host alias. Do not commit hostnames, usernames,
private paths, tokens, or keys.

**Step 2: Reproduce current V100 production rows**

Run FP16 and accepted MM4 matrices for 1.5B/2.9B/7.2B at B1/B2/B4/B8 where
memory permits. Confirm the exact V100 B8 WAVG launch remains selected.

**Step 3: Profile the largest remaining V100 gap**

Start with full-memory prefill/dequant traffic and projection/recurrent
breakdown; do not port Blackwell schedules.

**Step 4: Iterate through the same correctness and A/B gates**

Promote only exact V100 shapes. Rerun training/cache regressions required by
the V100 baseline before changing a default.

### Task 8: Final Verification and Evidence Closeout

**Files:**
- Modify: `BENCHMARK.md` only for promoted conclusions
- Modify: `docs/PERFORMANCE.md` only for promoted conclusions
- Modify: `docs/HARDWARE_MATRIX.md` only if exact-card status changes
- Finalize: both dated evidence directories

**Step 1: Run focused CPU and documentation contracts**

Run:

```powershell
python tests/test_convert_config.py
python tests/test_batch_convert_manifest.py
python tests/test_sync_hf_adapter_code.py
python tests/test_result_tools.py
python tests/test_markdown_links.py
python tests/test_document_freshness.py
python tests/test_repository_docs_layout.py
git diff --check
```

Expected: PASS.

**Step 2: Run exact-card final matrices**

Expected: correctness, prefill, decode, footprint, peak VRAM, policy route, and
reproduction commands are complete for every promoted profile.

**Step 3: Commit small atomic closeout changes**

Keep kernel, evidence, and documentation commits reviewable and do not mix
unrelated hardware profiles.
