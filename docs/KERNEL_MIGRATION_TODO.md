# RWKV7 optional-kernel migration TODO

> **Working rule:** read this file before every implementation, benchmark, or
> validation session. Update the checkboxes, evidence paths, actual route,
> commands, code SHA, and blockers before ending that session.

## Fixed decisions

- Clean base: `4bbd911e4dcb446e8c21fb795e373b4a59775ff3`.
- Working branch: `perf/optional-kernels-v1`.
- `rwkv7_hf/` remains the readable HF source of truth and contains only the
  canonical model modules.
- CLI/conversion/smoke stay in the sibling `rwkv7_hf_tools/` package.
- Optimized code is built as a separate `rwkv7-kernels` wheel from `kernels/`.
- Model weights, `config.json`, public cache ABI, and HF forward/generation
  signatures do not select hardware or kernel policy.
- Public recurrent state remains canonical `[B,H,K,V]`.
- Kernel-policy `auto` routes one-token FP16 decode to Triton and multi-token
  FP16 prefill to the exact CUDA-graph implementation. Explicit `triton` and
  `graph` modes remain available for isolated evidence; requested policy is
  never reported as the actual route.
- Unsupported device/dtype/shape, missing wheel, autograd, or probe failure
  falls back to `rwkv7_recurrent_reference` in `auto` mode.
- No old model wrapper, compatibility module, monkey patch, or performance
  policy may be copied from `perf/native-kernels-v0.8` or
  `perf/optional-native-backend-v0.10` into `rwkv7_hf/`.
- FLA comparison is pinned to commit
  `80e494f6c588e091fc8316b612870df29375c5b8`.
- RTX 4080 is the first release device. V100 and RTX 4090 follow only after the
  4080 evidence bundle is internally consistent.

## Phase 0 — clean layout

- [x] Move CLI/converter/manifest/smoke to `rwkv7_hf_tools/`.
- [x] Remove `model_cache.py`, `model_config.py`, `native_model.py` and old
      `NativeRWKV7*` aliases.
- [x] Remove package-backed `thin` conversion and duplicate console scripts.
- [x] Build and load a real converted 0.1B model package-free on V100.
- [x] Confirm the v1 conversion produces byte-identical safetensors.

## Phase 1 — recurrent plugin v1

### Layout

- [x] Create `kernels/pyproject.toml` for distribution `rwkv7-kernels`.
- [x] Create `kernels/rwkv7_kernels/protocol.py`.
- [x] Port exact CUDA-graph recurrence into
      `kernels/rwkv7_kernels/recurrent/graph.py`.
- [x] Port Triton rank-1 scan into
      `kernels/rwkv7_kernels/recurrent/triton.py`.
- [x] Keep implementation selection and environment parsing inside the kernel
      wheel, not in `rwkv7_hf/`.
- [x] Expose only `RWKV7_KERNEL_API_VERSION`, `probe_recurrent_v1`, and
      `recurrent_v1` as the v1 public kernel protocol.

### Core boundary

- [x] Split `ops_rwkv7.py` into visibly separate
      `rwkv7_recurrent_reference(...)` and `rwkv7_recurrent(...)` functions.
- [x] Add one lazy optional-package call in `rwkv7_recurrent(...)`.
- [x] Preserve package-free Hub loading when `rwkv7-kernels` is absent.
- [x] Record the actual route and implementation for validation without adding
      hardware fields to the model config.

### Local tests

- [x] Core model package still contains only canonical HF files.
- [x] Core never imports `rwkv7_hf_tools`.
- [x] Missing kernel package uses reference.
- [x] `auto` falls back on unsupported inputs and autograd.
- [x] `optimized` fails clearly rather than silently falling back.
- [x] API version mismatch fails clearly.
- [x] Broken probe/kernel is contained in `auto` and surfaced in `optimized`.
- [x] Kernel wheel and HF wheel build independently.
- [x] Package-free converted directory loads without either installed wheel.

## Phase 2 — RTX 4080 recurrent acceptance

### Environment and artifact identity

- [x] Record GPU, driver, CUDA, Python, Torch, Transformers, Triton and FLA.
- [x] Record source SHA, HF wheel SHA256, kernel wheel SHA256, model SHA256,
      tokenizer SHA256, command, seed, dtype and environment variables.
- [x] Verify JSON reports the real implementation route; filenames or requested
      environment variables are not accepted as route evidence.

### Correctness matrix

Run reference, optimized Graph, optimized Triton, and pinned FLA with:

```text
B = 1 / 4 / 8
T = 1 / 17 / 128 / 512
Dtype = FP32 / FP16 / BF16 where supported
```

- [x] Output parity.
- [x] Final recurrent-state parity.
- [x] Attention-mask and unequal-length batch parity.
- [x] Input and state gradients for training-capable routes.
- [x] All outputs and states finite.
- [x] No state update at masked positions.

### HF model matrix

Use 0.1B, 0.4B, and 1.5B:

- [x] AutoConfig/AutoTokenizer/AutoModel/AutoModelForCausalLM.
- [x] No-cache logits.
- [x] Prefill state.
- [x] Teacher-forced cached decode.
- [x] Left/right padding.
- [x] 64-token greedy equality.
- [x] Beam generation.
- [x] Save/reload.
- [x] Training/autograd reference fallback.

## Phase 3 — fair RTX 4080 speed comparison

Produce separate result tables. Do not mix these modes.

### Eager/operator table

Disable model-level CUDA Graph and `torch.compile` for all lanes:

```text
B = 1 / 4 / 8
T = 1 / 17 / 128 / 512 / 2048
```

- [x] Reference vs optimized recurrent vs FLA fused recurrent.
- [x] Reference vs optimized prefill vs FLA chunk where semantically matched.
- [x] Forward latency and tokens/s.
- [ ] Forward+backward latency for training-capable routes.
- [x] Peak VRAM, warmup count, measured iterations, median and p95.

### Whole-model table

Use 0.4B and 1.5B:

- [x] Prefill B1/B4/B8 × T128/T512/T2048.
- [x] Cached decode B1/B4/B8 for 256 generated tokens.
- [x] Separate compile/capture time from steady-state latency.

### Production table

- [x] Our best validated Graph/Triton/CUDA route.
- [x] FLA best supported official route.
- [x] End-to-end prefill and generation, including framework overhead.

## Phase 4 — three-way lm_eval equivalence

Lanes:

```text
hf-reference / hf-optimized / fla-rwkv7
```

Models and batches:

```text
0.1B / 0.4B / 1.5B
batch 1 / 8
```

Tasks:

```text
wikitext, lambada_openai, piqa, hellaswag, winogrande,
arc_easy, arc_challenge, openbookqa
```

Total: `3 lanes × 3 models × 8 tasks × 2 batches = 144` units.

- [ ] All 144 commands exit zero without NaN/Inf.
- [ ] Classification/LAMBADA per-sample selected answers match across lanes.
- [ ] Aggregate discrete metrics match exactly.
- [ ] Wikitext mean NLL is recorded and perplexity relative difference is
      `<=0.1%`.
- [ ] Batch 1/8 discrete predictions match and continuous Wikitext metrics stay
      within `0.1%`.
- [ ] Store raw per-sample outputs outside Git; commit only compact summaries,
      manifests, hashes, commands, and validators.

## Phase 5 — full fast decode and prefill

- [ ] Define `probe_decode_step_v1` / `decode_step_v1`.
- [ ] Port fused token, W/A/G/V, state pool and CUDA Graph replay without
      replacing `modeling_rwkv7.py`.
- [ ] Define `probe_prefill_v1` / `prefill_v1`.
- [ ] Port chunk/fused prefill and shape routing behind the protocol.
- [ ] Keep canonical cache visible to HF; internal layouts never escape the
      kernel package.

## Phase 6 — training kernels

- [ ] Define versioned forward/backward operator ABI.
- [ ] Compare outputs, states and all gradients against reference and FLA.
- [ ] Run Trainer/Accelerate/PEFT/TRL SFT, DPO and GRPO.
- [ ] Distinguish optimized training from reference fallback in every report.
- [ ] Benchmark forward+backward only after numerical gates pass.

## Release gate

- [ ] `rwkv7_hf/` remains clean after kernel installation.
- [ ] `rwkv7-hf` and `rwkv7-kernels` install independently.
- [ ] RTX 4080 correctness, HF, FLA speed and three-way lm_eval gates pass.
- [ ] Equivalent V100 and RTX 4090 gates pass.
- [ ] Six Hub repositories contain only self-contained reference code and
      unchanged weights where hashes match.
- [ ] GitHub, Hub and PyPI versions/tags agree.

## Session log

### 2026-08-27 — checklist created

- Base commit: `4bbd911e4dcb446e8c21fb795e373b4a59775ff3`.
- Branch: `perf/optional-kernels-v1`.
- Next action: port only the existing recurrent kernel wheel and protocol, run
  local fallback/package gates, then sync exact wheels to RTX 4080.

### 2026-08-27 — recurrent-v1 local gate

- Added sibling distribution layout:
  `kernels/rwkv7_kernels/{protocol,dispatcher,recurrent/*}.py`.
- The public model/config/cache remain free of hardware policy. The only model
  change is the semantic `training=self.training` hint at the operator call.
- `rwkv7_hf/ops_rwkv7.py` now keeps the complete readable reference recurrence
  and one lazy versioned optional-package boundary.
- Local test command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
- Local result: `46 passed`.
- HF wheel: `rwkv7_hf-1.0.0-py3-none-any.whl`, SHA256
  `07b4f6668c3123a3e996e33d4fab8230c468db23bbd7249c3454a93e2f04338f`.
- Kernel wheel: `rwkv7_kernels-1.0.0.dev0-py3-none-any.whl`, SHA256
  `31c0892a5284a26f89790567dbbdf4f6255b996cf5f7a32c14fa2406c15e24c9`.
- Both wheels passed `twine check --strict` and independent target-directory
  imports. A saved local model loaded through AutoModelForCausalLM while
  top-level `rwkv7_hf` and `rwkv7_kernels` imports were explicitly blocked.
- Next action: sync this exact commit and these wheel hashes to RTX 4080; record
  the actual Graph/Triton routes before accepting any benchmark result.

### 2026-08-27 — first RTX 4080 acceptance slice

- Compact evidence:
  `results/kernel-migration/4080-7d8df0c1/` with verified
  `MANIFEST.sha256`.
- Environment: RTX 4080, driver 595.84, CUDA 13.0, Torch 2.11.0+cu130,
  Transformers 5.8.0, Triton 3.6.0, pinned FLA
  `80e494f6c588e091fc8316b612870df29375c5b8`.
- Graph actual route `torch-cuda-graph-reference-v1`: 12/12 FP16 operator
  cases passed; 0.1B/0.4B/1.5B model/cache/64-token greedy gates passed.
- Triton actual route `native-triton-rank1-scan-v1`: 12/12 operator cases,
  finite/state/cache/greedy passed. Strict aggregate remains failed because
  the 0.4B B1/T17 logits max-abs is `0.15625`, above the fixed `0.15` gate.
- Both optional routes passed AutoConfig/AutoTokenizer/AutoModel,
  AutoModelForCausalLM, greedy, beam, save/reload, and training reference
  fallback. A separate no-wheel environment passed package-free 0.1B loading.
- Eager operator matrix: Graph is 1.35x-3.18x faster than the readable
  recurrence. Triton is 1.08x-1.49x faster than pinned FLA fused recurrent in
  all 12 measured B/T cases. FLA chunk remains faster at T=512, so no
  whole-model or long-prefill claim is made.
- Clean reference vs FLA 0.4B retained `outside_thresholds`: operator, state,
  and 64-token greedy passed, but B4/T128 logits max-abs reached `0.1875`.
- Still open: FP32/BF16 expansion, explicit left/right padding, whole-model
  prefill/decode, backward speed, and the three-way 144-unit lm_eval matrix.

### 2026-08-27 — promoted RTX 4080 auto route

- Production policy is now shape-based inside the separate kernel wheel:
  `RWKV7_KERNEL_IMPL=auto` selects actual route
  `native-triton-rank1-scan-v1` for `T=1` and
  `torch-cuda-graph-reference-v1` for `T>1`.
- Final route-traced kernel wheel SHA256:
  `22c3ef0fb0af1743261efed7ed23cfc2185982b8d03ea3c13bf3864b27dc932f`.
- RTX 4080 full validation evidence root:
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/4080-auto-v1`.
- FP16, BF16, and FP32 validation all exit zero. FP16 covers 12 operator
  B/T cases, 0.1B/0.4B/1.5B model B=`1/4/8` T=`17/128`, cached teacher
  decode, 64-token greedy, regrouping, explicit left/right padding, cache
  equality, and training fallback. All six multi-token model cases per model
  take the Graph route with exact logits; cached decode takes the Triton route
  and stays within the fixed FP16 gate with identical greedy tokens.
- Auto HF smoke passes all three models for AutoConfig, AutoTokenizer,
  AutoModel, AutoModelForCausalLM, greedy, beam, save/reload, and finite
  training gradients with actual reference fallback.
- Whole-model auto versus readable reference speed evidence is stored under
  `4080-auto-v1/speed`: Graph prefill is 2.28x–2.78x faster on the measured
  0.4B/1.5B cases, and Triton cached decode is 1.05x–1.32x faster.
- Explicit long-sequence Triton remains an experimental operator lane. Attempts
  to match CUTLASS's final FP16 readout reduction reached exact FP32 state
  updates and up to 99.98% elementwise readout equality, but either retained
  a few full-model logits above 0.15 or removed the FLA speed advantage. Those
  failed experiment bundles remain outside Git and are not release evidence.
- Final-wheel FP16 route trace records 7,623 actual
  `native-triton-rank1-scan-v1` calls and 789 actual
  `torch-cuda-graph-reference-v1` calls; validation and HF smoke both passed.
- The final eager/operator matrix covers B=`1/4/8`,
  T=`1/17/128/512/2048`. T=1 Triton is 1.26x–1.33x faster than pinned FLA
  fused recurrent. Exact Graph prefill remains slower than FLA chunk/fused.
- The production whole-model matrix covers 0.4B/1.5B prefill B=`1/4/8`,
  T=`128/512/2048`, plus 256-step cached decode B=`1/4/8`. Production `auto`
  beats the readable reference but is currently 1.36x–1.49x slower than FLA
  on cached decode and roughly 3.9x–13.5x slower on prefill. These results are
  reported directly; no FLA speed advantage is hidden.
- Reference/optimized/FLA PIQA smoke passed with identical selected answers.
  The optimized manifest records 96 actual Graph calls. A provenance bug that
  could select `kernel-route.json` instead of lm_eval's result JSON was found,
  fixed, and regression-tested. FLA is run with one TorchInductor compile
  worker so each subprocess exits deterministically.
- The formal 144-unit three-way matrix is running sequentially on the RTX 4080
  under `/home/wzu/codex-run/results/rwkv7-kernels-v1/4080-auto-v1/lm-eval`.
  Do not check Phase 4 until all three 48-unit manifests and the validator pass.
