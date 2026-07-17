# Native-Default RTX 5070 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository's RWKV-7 Hugging Face runtime native and FLA-free by default, connect the official `train_temp` CUDA training path directly to the native model, and validate the complete supported surface on one RTX 5070 Laptop GPU without regressing the existing accelerated decode path.

**Architecture:** `NativeRWKV7Config`, `NativeRWKV7Model`, and `NativeRWKV7ForCausalLM` become the canonical Auto* classes. The graph/fused inference runtime is extracted from the FLA-backed wrapper into FLA-free modules shared by the canonical native model; the old FLA implementation remains an explicit reference backend only. Official `train_temp` operators run a vectorized native full-sequence path rather than patching FLA layer classes.

**Tech Stack:** Python 3.12, PyTorch 2.11, Transformers, CUDA 12.8, Triton, official RWKV-LM `train_temp` CUDA extensions, DeepSpeed FusedAdam/ZeRO-2, pytest, RTX 5070 Laptop GPU (`sm_120`, 8GB).

---

## Non-Negotiable Acceptance Contract

- A base install with `flash-linear-attention` blocked must import the package and load converted checkpoints through `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` without setting `RWKV7_NATIVE_MODEL`.
- Production Auto* metadata must point directly to native classes. FLA may remain only as an explicit reference/benchmark backend; Qwen full-FLA comparison remains unchanged.
- The official training recipe is the current RWKV-LM `RWKV-v7/train_temp/demo-training-prepare.sh` plus `demo-training-run.sh`: x070, L12/D768, head64, vocab65536, T512, Minipile binidx with matching `magic_prime`, B16, BF16, LR `6e-4 -> 6e-5`, betas `0.9/0.99`, eps `1e-18`, weight decay `0.001`, warmup 10, gradient checkpointing enabled, one GPU, DeepSpeed ZeRO-2, kernel `@rwkv3`.
- The native training implementation must pass exact single-step backward and FusedAdam update gates against official code, then complete a predeclared multi-seed official-recipe cohort. A B1 custom harness is diagnostic evidence only.
- The native default must preserve HF load/generate, recurrent cache operations, dynamic batching, chunked prefill, save/reload, Trainer, PEFT, TRL SFT/DPO/GRPO, checkpoint resume, native W8/W4, and speculative decoding contracts.
- RTX 5070 exact-card tests must cover B1/B2/B4/B8 where memory permits. New defaults require logits/greedy parity and no material regression against the current wrapper `native_graph` lane.
- Baseline captured before implementation: 0.4B fp16 prompt32/decode16 B1 native eager `31.35 tok/s`, pure-native JIT `41.68 tok/s`, wrapper-hosted native graph `226.3 tok/s`. Moving the default while leaving this 5.43x gap open is forbidden.

## Task 1: Freeze the Architecture and Acceptance Rules

**Files:**
- Create: `docs/architecture/NATIVE_DEFAULT_BACKEND.md`
- Modify: `docs/ACCEPTANCE.md`
- Modify: `docs/performance/FUSED_BACKEND.md`
- Create: `tests/test_native_default_contract.py`

1. Write a failing contract test that checks the documented official shell recipe, native-default Auto* metadata, and the allowed FLA-reference boundary.
2. Run `python -m pytest -q tests/test_native_default_contract.py` and confirm it fails.
3. Add the architecture decision and acceptance rows, including the exact RTX 5070 baseline above.
4. Run the test and Markdown-link tests.
5. Commit with Wang Yue identity and DCO.

## Task 2: Make Converted Checkpoints Native by Default

**Files:**
- Modify: `scripts/convert_rwkv7_to_hf.py`
- Modify: `scripts/sync_hf_adapter_code.py`
- Modify: `rwkv7_hf/__init__.py`
- Modify: `rwkv7_hf/native_model.py`
- Modify: `tests/test_convert_config.py`
- Modify: `tests/test_sync_hf_adapter_code.py`
- Modify: `tests/test_native_fla_free_import.py`

1. Add failing tests requiring `auto_map` to target `native_model.NativeRWKV7*` and requiring native loading without `RWKV7_NATIVE_MODEL`.
2. Make conversion construct the native template first and emit native Auto* metadata.
3. Keep an explicit migration/sync path for existing converted directories without rewriting weights.
4. Block `fla` imports in a clean subprocess and test config/model load from a tiny saved checkpoint.
5. Run conversion, packaging, and remote-code tests; commit with DCO.

## Task 3: Isolate FLA as a Reference Backend

**Files:**
- Create: `rwkv7_hf/fla_reference.py`
- Modify: `rwkv7_hf/modeling_rwkv7.py`
- Modify: `rwkv7_hf/configuration_rwkv7.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_runtime_compat_and_bench_contracts.py`
- Modify: `tests/test_native_default_contract.py`

1. Add an AST/import test that rejects direct production FLA imports outside the explicit reference module.
2. Move FLA class imports and reference-only dispatch behind lazy functions.
3. Ensure base and CUDA-native extras do not install FLA; retain a separate `fla-reference` extra for A/B and Qwen comparison environments.
4. Verify import, wheel, clean-install, and reference-backend selection behavior; commit with DCO.

## Task 4: Move Native Graph/Fused Decode into the Canonical Native Model

**Files:**
- Create: `rwkv7_hf/native_graph_runtime.py`
- Modify: `rwkv7_hf/modeling_rwkv7.py`
- Modify: `rwkv7_hf/native_model.py`
- Modify: `rwkv7_hf/native_jit.py`
- Modify: `tests/test_native_graph_cache.py`
- Modify: `tests/test_native_model_generate_unit.py`
- Modify: `bench/bench_native_model_decode.py`

1. Add failing native-model tests for graph runner creation, B1/B2/B4/B8 cache reuse, dynamic select/reorder/drop, fused-output policy, and telemetry.
2. Extract graph runner/cache ownership from the FLA wrapper without importing FLA.
3. Bind the same runtime to `NativeRWKV7ForCausalLM`; preserve eager and JIT fallbacks.
4. Run CPU contract tests and RTX 5070 0.4B B1/B2/B4/B8 A/B rows.
5. Require greedy parity and at least 0.95x of the recorded wrapper-native-graph speed before continuing; tune only exact-card policy when needed.
6. Commit implementation and machine evidence with DCO.

## Task 5: Connect Official train_temp CUDA Directly to Native Layers

**Files:**
- Modify: `rwkv7_hf/train_temp_cuda.py`
- Modify: `rwkv7_hf/native_model.py`
- Modify: `rwkv7_hf/train_temp_alignment.py`
- Modify: `bench/bench_train_temp_alignment.py`
- Modify: `tests/test_train_temp_cuda.py`
- Modify: `tests/test_train_temp_alignment.py`
- Modify: `tests/test_train_temp_alignment_runner.py`

1. Add failing tests that enable train_temp on `NativeRWKV7ForCausalLM` while `fla` imports are blocked.
2. Implement a vectorized dense no-cache native training forward over `[B,T,C]` using the vendored Mix6, KkPre, ClampW, LnxOutput, VResGate, AGate, CMix, and fused CE/L2Wrap operators.
3. Preserve native module parameter names, LayerNorm placement, `v_first`, residuals, gradient-checkpoint hooks, and HF CausalLM output semantics.
4. Remove FLA layer-type requirements from the public train_temp enable/disable API.
5. Run tiny CPU contracts plus RTX 5070 BF16 T512 compile, forward, backward, and optimizer-step comparisons; commit with DCO.

## Task 6: Reproduce the Official Shell Recipe in WSL

**Files:**
- Create: `configs/train_temp_official_x070_12x768_b16.json`
- Create: `scripts/run_train_temp_official_recipe.py`
- Modify: `bench/bench_train_temp_alignment.py`
- Modify: `tests/test_train_temp_alignment_runner.py`

1. Clone/pin official RWKV-LM on D: and record both shell files plus commit hashes.
2. Create a D:-backed WSL virtual environment with PyTorch CUDA, DeepSpeed, Ninja, and the pinned dependencies.
3. Download/verify Minipile `.bin/.idx` on D: and validate `magic_prime=2926181` for T512.
4. Run official prepare once and verify `rwkv-init.pth` identity and output log.
5. Run a bounded official B16 training acceptance slice, preserving the exact command and environment.
6. Run the native HF path with the identical checkpoint, serialized samples, optimizer groups, LR schedule, and step count.
7. Emit fail-closed JSON comparisons and commit only compact evidence/logs, not checkpoints or data.

## Task 7: Complete the RTX 5070 Native-Default Matrix

**Files:**
- Create: `bench/5070_native_default_20260717/README.md`
- Create: `bench/5070_native_default_20260717/summary.json`
- Modify: `bench/results.jsonl`
- Modify: `bench/INDEX.md`
- Modify: `docs/HARDWARE_MATRIX.md`
- Modify: `docs/ACCEPTANCE.md`

1. Run native-default load/generate, HF API, cache, dynamic batch, chunked prefill, save/reload, and speculative decode rows.
2. Run B1/B2/B4/B8 prefill/decode/VRAM rows on 0.4B and the largest models that fit; compare against pre-migration exact-shape baselines.
3. Run native W8/W4 footprint, logits, greedy, and speed rows without converting memory evidence into speed claims.
4. Run Trainer, PEFT, SFT, DPO, GRPO, and checkpoint-resume rows; record any 8GB-bound lanes explicitly.
5. Run the official B16 train_temp acceptance and a repeated/multi-seed stability cohort.
6. Generate machine-readable summary and human report only from committed raw evidence.

## Task 8: Final Cleanup and Publication

**Files:**
- Modify: `README.md`
- Modify: `docs/BACKENDS.md`
- Modify: `docs/TRAIN_TEMP_CUDA.md`
- Modify: `docs/AI_ASSISTED_SETUP.md`
- Modify: `HF_STATUS.md`
- Modify: `BENCHMARK.md`

1. Make native the documented default and FLA reference opt-in.
2. Include prerequisites, copyable commands, observable pass criteria, failure recovery, limits, and the single AI setup entry.
3. Run focused tests, the full local suite, wheel/clean-install tests, Markdown links, and `git diff --check`.
4. Verify a clean environment with FLA absent and a separate reference environment with FLA present.
5. Push `wangyue/native-default-5070`, open a draft PR as Wang Yue, attach DCO and evidence links, then wait for CI and review.

