# Native Fused MM8/MM4 FFN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fuse the MM8/MM4 FFN key projection and ReLU-square epilogue behind an opt-in native-graph route.

**Architecture:** Extend the existing quantized GEMV kernels with a compile-time epilogue and expose it through the quantized Linear modules. Route only non-dense FFN-up operands through the new method when an explicit environment flag is enabled, and include that flag in CUDA graph cache identity.

**Tech Stack:** Python, PyTorch, Triton, CUDA C++ extension, Hugging Face remote code, pytest-style script tests.

---

### Task 1: Add the behavioral contract tests

**Files:**
- Create: `tests/test_native_quant_fused_ffn.py`
- Modify: `tests/test_kernel_policy.py`

1. Add CPU tests for the MM8/MM4 `rwkv7_forward_relu2` fallback.
2. Add a probe module test showing native graph dispatch uses the fused method only when the flag is enabled.
3. Assert the policy default remains disabled.
4. Run `python tests/test_native_quant_fused_ffn.py` and verify it fails before implementation.

### Task 2: Implement MM8 and MM4 fused epilogues

**Files:**
- Modify: `rwkv7_hf/native_quant_mm8.py`
- Modify: `rwkv7_hf/native_quant_mm4.py`
- Modify: `rwkv7_hf/sm70_quant.py`
- Modify: `tests/test_native_quant_mm8.py`
- Modify: `tests/test_native_quant_mm4.py`

1. Add a compile-time `RELU2` epilogue to single-row, batched GEMV, and Blackwell dot kernels.
2. Add an sm70 W4 CUDA extension entry point for the same epilogue.
3. Expose module methods with bias-safe portable fallbacks.
4. Extend exact-card correctness tests to compare fused and separate outputs.
5. Run CPU tests and Python compilation.

### Task 3: Route native graph and protect cache identity

**Files:**
- Modify: `rwkv7_hf/kernel_policy.py`
- Modify: `rwkv7_hf/native_jit.py`
- Modify: `rwkv7_hf/modeling_rwkv7.py`

1. Add a conservative `fused_quant_ffn=False` policy field.
2. Add runtime flag parsing in native JIT and modeling code.
3. Route quantized FFN-up operands through `rwkv7_forward_relu2` only under the flag.
4. Add the flag to the native graph runner key.
5. Run focused policy, cache, and quant tests.

### Task 4: Add benchmark telemetry

**Files:**
- Create: `bench/bench_native_quant_fused_ffn.py`
- Modify: `bench/bench_native_quant_e2e_decode.py`
- Modify: `docs/performance/FUSED_BACKEND.md`

1. Add synthetic hidden-to-intermediate MM8/MM4 separate-vs-fused rows.
2. Add an end-to-end CLI flag and row metadata for the opt-in route.
3. Document that the route remains telemetry-only.
4. Run script help, compile, and CPU unit coverage.

### Task 5: Run exact V100 A/B after the active matrix releases GPU1

**Files:**
- Create after evidence exists: `bench/v100_native_fused_quant_ffn_20260712/README.md`
- Create after evidence exists: `bench/v100_native_fused_quant_ffn_20260712/results.jsonl`

1. Run synthetic bsz 1/2/4/8 MM8/MM4 rows on Tesla V100-PCIE-32GB.
2. Run 1.5B end-to-end memory-policy decode with the flag off/on.
3. Check logits/greedy alignment and peak VRAM.
4. Keep the default off unless all claimed batch sizes are correct and non-negative end to end.
