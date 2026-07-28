# RWKV7 optional-kernel migration TODO

> **Working rule:** read this file before every implementation, benchmark, or
> validation session. Update the checkboxes, evidence paths, actual route,
> commands, code SHA, and blockers before ending that session.

## Fixed decisions

- Initial clean base: `4bbd911e4dcb446e8c21fb795e373b4a59775ff3`.
- Current refactor base: `fc6f6b39637f2f79fe9b54e29def3e9859fb4796`.
- Working branch: `refactor/thin-ops-v1`.
- `rwkv7_hf/` remains the readable HF source of truth and contains only the
  canonical model modules.
- The HF source uses a Mamba-style ownership boundary (`configuration`,
  `cache`, `ops`, `modeling`) only as a clean organizational convention; RWKV-7
  recurrence and checkpoint semantics remain unchanged.
- CLI/conversion/smoke stay in the sibling `rwkv7_hf_tools/` package.
- Optimized code is built as a separate `rwkv7-kernels` wheel from `kernels/`.
- Model weights, `config.json`, public cache ABI, and HF forward/generation
  signatures do not select hardware or kernel policy.
- Public recurrent state remains canonical `[B,H,K,V]`.
- The HF core calls only kernel API v4's `execute_optional_v4`; its five
  operation kinds are `training_program`, `model_forward`, `linear_training`,
  `mix6_training`, and `recurrent`. Unsupported envelopes require
  `result=None`.
- `modeling_rwkv7.py` resolves one immutable `RWKV7ExecutionContext` and passes
  it explicitly through non-linear layer boundaries, the LM-head boundary, and
  checkpoint replay. Two narrow routing bridges carry that resolved value: a
  decoder-to-LM-head capture across the standard HF output boundary and lexical
  `linear_execution_context` for `nn.Linear`/PEFT/quantization `forward(x)`;
  replay republishes the linear scope. Two other context-local values are
  evidence-only.
- Kernel-policy `auto` routes one-token FP16 decode to Triton and multi-token
  FP16 prefill to the exact CUDA-graph implementation. Explicit `triton` and
  `graph` modes remain available for isolated evidence; requested policy is
  never reported as the actual route.
- Unsupported device/dtype/shape, missing wheel, or autograd returns a
  side-effect-free negative decision and falls back to readable math in `auto`.
  `model_forward` uses the caller's canonical cache zero-copy; after positive
  execution starts, an exception or malformed payload fails closed and is never
  reference-recomputed.
- No old model wrapper, compatibility module, monkey patch, or performance
  policy may be copied from `perf/native-kernels-v0.8` or
  `perf/optional-native-backend-v0.10` into `rwkv7_hf/`.
- FLA comparison is pinned to commit
  `80e494f6c588e091fc8316b612870df29375c5b8`.
- RTX 4080 is the release-validation device. V100 was removed from the release
  gate by user decision on 2026-08-29; its partial evidence remains historical
  only. RTX 4090 follows after the 4080 evidence bundle is internally
  consistent.
- Full NVIDIA prefill/decode/quant/training migration uses the frozen one-shot
  design in `docs/KERNEL_BACKEND_V2_DESIGN.md`. Its public ABI is fixed before
  implementation; diagnostic stages may identify failures but do not redesign
  the clean model boundary.
- Current audited NVIDIA denominator: 102 destinations = 86 byte-identical +
  16 declared clean-boundary adaptations. The full 153-file historical scope
  is 86 byte-migrated, 26 adapted protocol/glue, 7 canonical-reference, 6
  relocated/retired tooling, 27 separate-hardware, and 1 retired non-kernel.

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
- [x] Supersede the split implementation entry points with the single public
      API-v4 `execute_optional_v4` facade.

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
- [x] Broken optional execution is contained in `auto` and surfaced in
      `optimized`.
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
- [x] Forward+backward latency for training-capable routes.
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

## Phase 5 — one-shot complete NVIDIA backend-v2

- [x] Freeze the complete model-forward ABI and migration inventory before
      moving implementation code.
- [x] Add the single kernel API-v4 request/result envelope and clean-model
      facade; production auto stays disabled for changed bytes.
- [ ] Complete every API-v4 `model_forward` phase and enable production auto
      only after the unified immutable wheel passes.
- [x] Port fused token, W/A/G/V, projection, FFN, norm, state pool and CUDA
      Graph replay without replacing `modeling_rwkv7.py`.
- [x] Port DPLR/self-chunk/fused prefill and all shape routing behind the same
      model-forward protocol.
- [x] Port SM70, Ada and Blackwell NVIDIA policy families.
- [x] Port W8/W4/A8W8/BnTn/BnB/Marlin/TorchAO implementation adapters.
- [x] Keep canonical cache visible to HF; internal layouts never escape the
      kernel package.

## Phase 6 — backend-v2 training implementation and unified acceptance

- [x] Preserve the existing whole-model forward/backward runtime only as a
      private historical diagnostic. Formal HF training never selects the
      API-v4 `model_forward` operation and does not create a separate
      model/cache.
- [x] Add clean recurrent and stateless-linear training leaf protocols so the
      readable HF layer loop can use CUDA without the historical whole-model
      training wrapper.
- [x] Compare outputs, states and all gradients against reference and FLA.
- [ ] Run Trainer/Accelerate/PEFT/TRL SFT, DPO and GRPO.
- [x] Distinguish optimized training from reference fallback in every report.
- [x] Benchmark forward+backward only after numerical gates pass.

### Large-batch training hot-path follow-up

This subsection tracks the post-`7692a263` optimization candidate. Earlier
training evidence does not validate these changed bytes and must not be
relabelled as evidence for this candidate.

- [x] Keep `modeling_rwkv7.py` as the only public training program and dispatch
      recurrent, flattened-linear, and explicit-shift Mix6 as independent
      tensor leaves. The historical whole-model runtime remains a private
      diagnostic and is not eligible for formal HF training evidence.
- [x] Advance the optional-package protocol to API v4 and reserve the
      `training_program` operation as the atomic adaptive-preflight boundary.
- [x] Fail closed while API v4 lacks the concrete projection and Mix6 tensor
      plan: issue no partial certificate, keep `auto` on the complete readable
      reference program, and fail strict `optimized` at the model boundary.
- [ ] The API-v4 source now preloads every lazy native dependency and binds an
      opaque per-call certificate to shape/device/dtype/model facts. Prove on
      RTX 4080 that no certified recurrent, linear, or Mix6 leaf JITs or
      declines after certification before checking this item.
- [x] Preserve the fixed-row readable reference projection contract while the
      optional flattened-linear leaf presents one `[B*T,C]` GEMM to PyTorch.
- [x] Remove the benchmark's second causal cross-entropy calculation without
      changing the model-provided HF causal-loss contract.
- [x] Ensure each optimized flattened-linear call performs only one fail-closed
      support validation, with no reusable trusted bypass.
- [x] Resolve all-active mask and zero-state provenance once at the readable
      model boundary rather than synchronizing once per layer.
- [x] Skip redundant mask multiplications and zero fills only when the readable
      model has already proven that doing so is mathematically identical.
- [x] Replace Mix6 parameter-gradient atomics and FP32 scratch/cast launches
      with a deterministic, parallel, current-stream two-stage reduction;
      retain canonical replay for small and higher-order requests.
- [x] Fail closed for PEFT/frozen-input and reentrant-checkpoint forwards when
      no valid certificate can be consumed; never mix certified and
      uncertified leaves inside one training program.
- [x] Derive each formal training case seed from an order-independent SHA256
      identity and record the exact input-id dtype/shape/bytes hash; require
      checkpoint-on/off runs to reproduce the same input hash.
- [x] Add a reproducible profiler that records actual routes, selected
      operator counts/times, recurrence time, peak memory, environment,
      command, code identity, and wheel identity. Schema v3 also records the
      process peak RSS (Linux `VmHWM`, with `getrusage` fallback) so allocator
      and whole-process memory evidence are not conflated.
- [x] Re-run the complete local Python test/lint/hash gate after the shared
      source settles: **428 passed** with **385 expected warnings**; Ruff on
      every changed first-party Python file, bytecode compilation, diff check,
      102/102 migration hashes, and source-scope/capability audits all pass.
- [ ] Build one immutable HF/kernel wheel pair and compile the changed Mix6
      extension on RTX 4080 from that exact pair.
- [ ] Pass Mix6 output/first-gradient parity, repeated-run determinism, and
      higher-order-gradient coverage on RTX 4080.
- [ ] Re-profile optimized/reference/pinned-FLA B1/T128 and B4/T128 and record
      the measured `aten::mm`, copy, recurrence, total-time, and peak-memory
      deltas. Expected reductions are hypotheses, not acceptance evidence.
- [ ] Pass full-model logits/loss/all-gradient/checkpoint parity and the
      forward+CE+backward speed matrix for B=`1/4`, T=`16/17/128` before any
      large-batch speed claim.
- [ ] Re-run the affected HF ecosystem and SFT/DPO/GRPO gates. Run an
      inference/lm_eval regression smoke to prove the training-only routing
      changes did not alter evaluation outputs; do not claim a new 144-unit
      result unless the full matrix is actually rerun.

## Release gate

- [ ] `rwkv7_hf/` remains clean after kernel installation.
- [ ] `rwkv7-hf` and `rwkv7-kernels` install independently.
- [ ] RTX 4080 correctness, HF, FLA speed and three-way lm_eval gates pass.
- [ ] Equivalent RTX 4090 gates pass; V100 is not a release requirement.
- [ ] Six Hub repositories contain only self-contained reference code and
      unchanged weights where hashes match.
- [ ] GitHub, Hub and PyPI versions/tags agree.

## Session log

### 2026-08-30 — atomic B4/T128 training-program source candidate

- The first immutable `1.0.0` candidate wheel pair reached RTX 4080 with
  matching SHA256 values. Its Mix6 extension compiled for sm89 and the direct
  API-v4 Mix6 matrix passed six B1/B4 × T1/T17/T128 cases, five-run bitwise
  determinism, first gradients, higher-order gradients, and a non-default CUDA
  stream. The original validator incorrectly called a removed top-level
  private symbol; that infrastructure failure is preserved and the harness now
  exercises only `execute_optional_v4`.
- The old wheel's full-model hotspot run confirmed the intended remaining
  blocker: API-v4 always declined its training-program preflight, so the
  optimized lane could not report the adaptive routes. No old result is being
  relabelled as final evidence.
- The new source candidate restores the conservative B4/T128 BF16 fast domain,
  loads recurrent and Mix6 CUDA dependencies before issuing an opaque
  process-local certificate, and binds that certificate to B/T, device, dtype,
  and model facts. Certified leaf mismatch is a fail-closed API-v4 decline;
  non-certified `auto` shapes may use only their independently probed adaptive
  leaves and readable fallbacks.
- Local source gate: **464 passed** with **397 expected warnings**; Ruff on all
  changed Python files, bytecode compilation, and `git diff --check` pass. A
  new wheel pair and RTX 4080 full-model/FLA/ecosystem/finetune evidence for
  these changed package bytes are still pending.

### 2026-08-30 — package/API cleanup and final-source numerical fixes

- Kept the public HF runtime to the six canonical Python modules under
  `rwkv7_hf/`; converter, CLI, manifest and smoke utilities remain in the
  sibling `rwkv7_hf_tools/` package. Setuptools discovery now names those two
  packages explicitly, so a future similarly prefixed directory cannot enter
  the HF wheel accidentally.
- Preserved the readable model's fixed-row rank-3 vocabulary projection at the
  optional whole-model boundary while retaining the direct rank-1/rank-2
  decode path.
- Fixed RTX 4080 A8W8 small-row execution: CUDA `torch._int_mm` inputs now use
  complete 32-row tiles, and the output head uses an activation-stable tiled
  W8A16 path through 32 rows. The speed and memory policies passed the existing
  finite/cosine/cache/greedy gates with FP16 logits max-abs `0.125` and
  `0.09375`, respectively; the small-row microbench was `0.0568`–`0.1142 ms`
  instead of `0.2699`–`0.2719 ms` for padded dynamic A8.
- Replaced the fused-prefill rounded decay constant `0.606531` with the exact
  `exp(-0.5)` value shared by its Triton and Torch fallback. RTX 4080 state
  preparation then matched W/K/V/KK bitwise. A strict reference-layout
  diagnostic can make 0.4B/1.5B, B1/B4, T128 logits and state bitwise equal,
  but it is `10.5x`–`19.7x` slower and therefore remains a parity diagnostic,
  not the default optimized route.
- Aligned the FLA harness with the documented calibrated release envelope:
  low precision blocks on finiteness and cosine, while FP16 logits
  max-absolute `0.15` stays visible as an aspirational diagnostic. Candidate
  route/reference gates remain blocking; FLA comparisons remain non-blocking.
- Hardened release archives: wheel payload outside the owned package roots and
  one `.dist-info` tree is rejected; sdists reject unowned build hooks or
  checkout drift. The kernel distribution now carries and audits its own
  byte-exact MIT license under PEP 639 metadata.
- The current frozen denominator is **86 byte-identical + 16 declared
  adaptations = 102 NVIDIA destinations**; the complete historical source
  scope is **86/26/7/6/27/1**. All destination hashes match the working tree.
- Local gate after all shared edits: the complete suite is **459 passed**, with
  **397 expected TorchScript deprecation warnings**. The package-layout check
  now restricts legacy-module discovery to this checkout, so an unrelated old
  editable install cannot forge a failure. Ruff on every changed Python file,
  bytecode compilation, `git diff --check`, the 74 focused release/source
  audits, and a locally built HF/kernel wheel membership audit all pass. No
  immutable final 4080/4090 wheel pair or device evidence has been created for
  these bytes yet.

### 2026-08-30 — API-v4 facade and explicit execution context

- Reduced the HF/kernel ABI to one public call,
  `execute_optional_v4(kind, ...)`, with five operation kinds and one
  normalized envelope. Negative capability decisions are side-effect-free and
  return `result=None`; split capability/execution adapters remain private to the
  kernel package.
- Replaced the implicit per-leaf training control state with one frozen
  `RWKV7ExecutionContext`. `modeling_rwkv7.py` resolves it once and passes it
  explicitly through every non-linear boundary and checkpoint replay. Two narrow
  routing bridges carry it across decoder-to-LM-head output and standard
  `forward(x)` linear/PEFT/quantization calls; the linear scope is republished
  inside checkpoint replay. Two additional context-local snapshots
  remain solely for last-route evidence and cannot affect execution.
- `ops_rwkv7.py` now contains the readable reference recurrence, common
  envelope validation, explicit-context routing, and small stable operation
  entry points. Device/shape policy, environment selectors beyond the public
  `auto|reference|optimized` mode, probing, kernel execution, and trace
  accounting stay in `rwkv7-kernels`.
- All training requests currently select one complete readable training
  program in `auto` because the API-v4 request cannot prove the concrete
  recurrent/linear/Mix6 plan. Strict `optimized` fails at the model boundary
  instead of mixing leaves from different programs.
- The final shared source snapshot passes **428 tests** with **385 expected
  warnings**. Ruff on every changed first-party Python file, bytecode
  compilation, diff check, 102/102 migration hashes, and the source-scope and
  capability audits also pass. **No immutable HF/kernel wheel pair or new RTX
  4080 result exists for these changed bytes yet.** Build/audit,
  exact-route numerical matrices, performance/FLA comparison, HF ecosystem,
  SFT/DPO/GRPO, and lm_eval regression remain open and must use the same new
  wheel hashes.

### 2026-08-30 — large-batch training hot-path source candidate

- Started from merged baseline
  `10584cb5abf25f9e685116a863d73cdd426ce931` on
  `perf/training-large-batch-v1` after profiling attributed the B4/T128
  regression primarily to row-serialized projection launches and tensor-copy
  overhead rather than recurrent time.
- Implemented clean adaptive tensor leaves for recurrent, flattened linear and
  explicit-shift Mix6, retained the single model-owned causal-loss path, and
  added a formal hotspot profiler. The readable HF layer loop, cache and
  parameter ownership remain unchanged; no public whole-model training route
  is used.
- The NVIDIA manifest still contains exactly 102 historical destinations. Its
  current classification is 86 byte-identical transfers and 16 declared clean
  adaptations; the complete historical source scope is 86 byte-migrated, 26
  adapted, 7 canonical-reference, 6 relocated/retired tooling, 27 separate
  hardware, and 1 retired non-kernel file. Destination SHA256 values currently
  match the working-tree bytes.
- **No new wheel or RTX 4080 result exists for these changed bytes yet.** All
  GPU, FLA-speed, ecosystem, finetune and lm_eval checkboxes in the follow-up
  subsection remain open until an immutable wheel pair produces matching JSON
  evidence.
- Local source gate: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
  completed with **309 passed**; Ruff, format check, Python bytecode
  compilation and `git diff --check` passed. The wheel
  audit now explicitly requires `nvidia/training_math.py`, and all 102
  destination hashes plus the 153-row frozen source scope were revalidated.

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

### 2026-08-27 — backend-v2 one-shot migration started

- Frozen `docs/KERNEL_BACKEND_V2_DESIGN.md` before moving implementation code.
- Kernel package API advanced to v2 with one whole-model request/result
  envelope; the clean model gained one early layer-loop hook and no hardware
  policy.
- Migrated the historical tensor-only dense sequential layer executor and
  structural packer as an internal diagnostic implementation. Canonical cache
  remains `[B,H,K,V]`; internal `[V,K]` never crosses the boundary.
- Production `auto` intentionally remains unsupported until fused prefill,
  decode, quantization and training all pass as one wheel.
- Local gate: `54 passed`; dense executor matches the clean model on full
  hidden outputs, hidden-state history, padding and final recurrent state.
- Local result: `46 passed`.
- HF wheel: `rwkv7_hf-1.0.0-py3-none-any.whl`, SHA256
  `07b4f6668c3123a3e996e33d4fab8230c468db23bbd7249c3454a93e2f04338f`.

### 2026-08-28 — complete-performance scope and dependency gate

- The active goal and release gate cover the **entire** migrated NVIDIA
  backend, not only recurrent-v1: fused decode, fused/DPLR/self-chunk prefill,
  projections, Norm/FFN/LoRA, graph/state pools, SM70/Ada/Blackwell routing,
  W8/W4/A8W8/BnTn/BnB/Marlin/TorchAO adapters, and train-temp
  forward/backward/autograd all remain in scope.
- The wheel/source audits bind 102 NVIDIA destination files to the frozen
  historical trees. The current manifest records 86 byte-identical transfers
  and 16 declared clean-boundary adaptations; any omitted file, changed Git
  blob, or undeclared adaptation fails the release audit.
- Audited package imports and made direct runtime dependencies explicit in the
  independent kernel distribution: `torch`, `numpy`, and `packaging`.
  Transformers, DeepSpeed, BitsAndBytes and TorchAO stay lazy feature-specific
  integrations; they are not required to import or use the base kernel API.
- The stable-wheel audit now rejects missing or extra direct dependencies in
  `rwkv7-kernels==1.0.0` metadata.
- Local gate after this change:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` -> `159 passed`,
  `133` expected TorchScript deprecation warnings.
- RTX 4080 formal reference/optimized/FLA run remains the active GPU task; no
  competing process was started. Production `auto`, stable wheels, V100 and
  RTX 4090 remain ordered behind the 4080 diagnostic gate.

### 2026-08-28 — immutable-wheel device order is now evidence, not convention

- Added `evaluation/record_device_acceptance.py`. A final device run writes an
  immutable-wheel/source/harness start marker before its first GPU command and
  can transition to `passed` only by hashing a matching passed
  `release-validation.json` after the last gate.
- Compact bundles now retain that marker. Final provenance rejects a missing
  marker, different wheel/source/harness/device identity, invalid or naive
  timestamp, completion before start, overlapping runs, or any order other
  than RTX 4080 -> V100 -> RTX 4090.
- This applies to the future final stable `1.0.0` wheel pair. The already
  running recurrent diagnostic is deliberately not relabeled as final release
  evidence and was not stopped or restarted.
- Direct-entrypoint, timestamp/order failure tests and the full local suite
  pass: `163 passed`, with `133` expected TorchScript deprecation warnings.
- At `2026-08-28T07:29:56+08:00`, the untouched RTX 4080 formal reference
  lane remained 35/48 exit-zero with no failed unit. Its active 1.5B/B1
  HellaSwag process was about 10% through 40,168 log-likelihood requests with
  an approximately 2h35 remaining estimate; it was the only GPU process.
  Optimized/FLA and all backend-v2 watchers remained sequentially queued.
- The ordered-acceptance change was pushed as
  `4084cf9679cb9e0f74b01a8a158658d339490b39`; fork and local branch heads
  match. GitGuardian passed and the four upstream Python checks were still
  running at the final read-only check.
- The next Python 3.10 CI run exposed a real compatibility omission in the
  new source-archive verifier: it imported the Python 3.11 `tomllib` name
  directly. The verifier now uses the already-declared Python 3.10 `tomli`
  fallback; no GPU process or release artifact was changed.
- The fix was pushed as
  `794c24d56ad3a9730997dea2b8b75e483084713d`; GitGuardian, reference-model,
  training-stack, Python 3.10/Transformers 4.48.3 and Python 3.12/Transformers
  <6 all passed on the exact fork/PR head.

### 2026-08-28 — release archives are bound to the tagged checkout

- The final asset verifier now compares every package-owned member in both
  wheels byte-for-byte with the checked-out release source: `rwkv7_hf/`,
  `rwkv7_hf_tools/`, and `kernels/rwkv7_kernels/`. A wheel/sdist pair can no
  longer agree with itself while carrying code from a different commit.
- The HF wheel audit now also requires all five sibling CLI/converter/manifest/
  smoke tool files, while continuing to reject them from the clean model
  package. Kernel/model ownership remains unchanged.
- Failure coverage includes a wheel whose modeling payload differs from the
  checkout. Targeted release tests and the complete suite pass: `164 passed`,
  with `133` expected TorchScript deprecation warnings.
- Fresh disposable wheel builds passed the checkout binding against their
  actual ZIP payloads: all 12 HF/model-tool files and all 124 kernel package
  files matched the current source byte-for-byte. These development artifacts
  were deleted and are not final release wheels.

### 2026-08-27 — backend-v2 training boundary wired locally

- Added `nvidia/training_runtime.py`, which directly executes the clean
  model's layer structure through the migrated train-temp autograd operators;
  it does not replace model methods or own a second model/cache class.
- Removed the historical train-temp/FLA `MethodType` forward replacements from
  the CUDA leaf module. Adapter-wrapped FFN modules are rejected by the native
  probe and deterministically use the readable autograd path, so PEFT weights
  cannot be silently bypassed.
- The optimized layer operators retain standard HF causal cross-entropy,
  including `-100`; the historical fused L2Wrap loss remains an explicit leaf
  operator because silently adding L2Wrap would change every HF gradient.
- Added an explicit `native-nvidia-train-temp-autograd-v2` capability probe
  for dense, unpadded BF16 CUDA training. Unsupported labels, masks, dtypes,
  shapes, or devices remain reference fallbacks in production `auto`.
- Local tests replace only the CUDA leaf operators with differentiable Python
  equivalents and verify logits, loss, and parameter gradients against the
  clean reference model. Structural tests reject method monkeypatching.
- Local command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
  Result: `62 passed`. CUDA extension build and numerical/throughput gates are
  still pending on the 4080; this checkbox records protocol migration only.

### 2026-08-27 — quantization ownership moved into the kernel wheel

- Added `rwkv7_kernels.quantization` as the single structural setup layer for
  native W8/W4, dynamic A8W8, TorchAO W8/W4 and Marlin Bn/Tn W4.
- BitsAndBytes remains loaded through standard HF `BitsAndBytesConfig`; the
  kernel wheel supplies config construction plus adoption/route validation.
- Quantization metadata and graph pools are package-owned. Packing replaces
  only `nn.Linear` modules, invalidates graph runners, and does not write
  quantization policy into `RWKV7Config` or `RWKV7Cache`.
- CPU reference tests exercise native MM8/MM4 module replacement and confirm
  ordinary Linear call semantics. GPU correctness, quality and throughput
  remain release gates rather than migration checkboxes.

### 2026-08-27 — backend-v2 padding and route evidence completed locally

- Native prefill compacts active tokens per sample for mixed left/right padded
  batches, scatters zero logits at masked positions, and returns canonical
  FP32 `[B,H,K,V]` state. Mixed masked decode updates only active cache rows.
- The process route trace is now shared by recurrent-v1 and model-forward-v1;
  it records actual fused implementation suffixes and prefill/decode/training
  phase counts. A requested selector alone is not accepted as evidence.
- `run_lm_eval_matrix.py` can run the strict pre-release whole-model native
  route, and `validate_lm_eval_three_way.py --require-model-routes` rejects an
  optimized unit that never executed backend-v2.
- CPU parity covers one batch containing both right and left padding followed
  by a mixed active/masked cached-decode step. RTX 4080 validation is pending.
- Kernel wheel: `rwkv7_kernels-1.0.0.dev0-py3-none-any.whl`, SHA256
  `31c0892a5284a26f89790567dbbdf4f6255b996cf5f7a32c14fa2406c15e24c9`.
- Both wheels passed `twine check --strict` and independent target-directory
  imports. A saved local model loaded through AutoModelForCausalLM while
  top-level `rwkv7_hf` and `rwkv7_kernels` imports were explicitly blocked.
- Next action: sync this exact commit and these wheel hashes to RTX 4080; record
  the actual Graph/Triton routes before accepting any benchmark result.

### 2026-08-27 — NVIDIA operator-source transfer and first model runtime bridge

- Added a byte-verified first migration manifest for 99 implementation/source
  artifacts from `perf/native-kernels-v0.8`. It covers fused projection,
  norm/mix, recurrent/output, FFN/LoRA, DPLR/self-chunk prefill, SM70/Ada/
  Blackwell, W8/W4/A8W8/BnTn/BnB/Marlin/TorchAO and training CUDA sources.
- The migrated NVIDIA namespace contains no model, configuration, tokenizer or
  cache class and imports no `rwkv7_hf` implementation module.
- Added the raw causal-LM model boundary required by the frozen design. The
  explicit `RWKV7_MODEL_KERNEL_IMPL=native` diagnostic route now executes the
  migrated sequence prefill engine and the migrated fused per-token decode
  engine while returning the ordinary canonical `RWKV7Cache`.
- Actual prefill/decode route names are returned by execution, including the
  effective fused subroutes, rather than copied from the requested selector.
- Ported the fixed-batch CUDA Graph runner and package-owned LRU state pool.
  Runner buffers bind to canonical cache tensor views, detach safely when a
  different cache is selected, and require no graph metadata/private methods
  on `RWKV7Cache`. Public recurrent tensors remain FP32 `[B,H,K,V]` even when
  the internal graph layout is `[V,K]`.
- CPU dense-fallback parity proves full logits, prefill state, cached decode
  logits and final cache across the new boundary. Production `auto` remains
  disabled until NVIDIA GPU fused routes, padding, training and quantization
  complete the same-wheel acceptance matrix.
- Local gate: `60 passed`; all 99 migrated artifacts match the manifest SHA256
  and the kernel wheel includes every CUDA/C++ header/source and license file.

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

### 2026-08-28 — complete-source audit, prefill Graph, and final comparison harnesses

- Re-audited every file in `perf/native-kernels-v0.8/rwkv7_hf` instead of
  treating the first 99-file transfer as final. The byte-verified manifest now
  contains 102 artifacts and adds the self-chunk license, physical BN/TN sweep
  helper, and explicit legacy Triton compatibility helper. The ownership and
  exclusion decision for every remaining historical module is recorded in
  `docs/NVIDIA_MIGRATION_AUDIT.md`.
- Adapted the old fixed-shape sequence-prefill CUDA Graph into
  `nvidia/prefill_graph_runtime.py` and a package-owned weak/LRU pool. It uses
  the structural clean model owner, supports only allowlisted dense FP16
  shapes, clones every replay output, and lets `model_dispatcher.py` copy state
  into canonical FP32 `[B,H,K,V]` cache tensors. No graph metadata or private
  layout was added to `RWKV7Cache`.
- Split generic BF16 Marlin W4 from the physical SM120-only BN/TN route. The
  latter now fails closed on non-SM120 GPUs and is recorded as not applicable,
  rather than being mislabeled as an Ada/V100 success. BF16 native model
  probing is enabled for the TorchAO/Marlin paths; numerical/route evidence is
  still pending on the RTX 4080.
- Added immutable-wheel release harnesses:
  `validate_backend_v2_fla.py` covers recurrent output/state/gradients plus
  0.1B/0.4B/1.5B full logits/cache/padding/cached decode/64-token greedy and
  BF16 full-model all-gradient parity; `benchmark_backend_v2.py` records
  reference/optimized/pinned-FLA operator, prefill, 256-step cached decode and
  forward+backward timing with cold capture separated from steady state.
  Actual model/recurrent routes are mandatory.
- Local gate: `74 passed`; compileall and `git diff --check` pass. Diagnostic
  wheel hashes (not release-final until GPU fixes stop):
  `rwkv7_hf-1.0.0-py3-none-any.whl =
  0ba2a1e8196d120b412fe1eeea5e87a8321bbc692833e4d42dd1d1ffed94c531`,
  `rwkv7_kernels-1.0.0.dev0-py3-none-any.whl =
  bb20cf15b3837a370fe6024aafbd77c130258c1118169e95fff6aa253922436d`.
  Both pass `twine check --strict`; the kernel wheel contains the new graph
  runtime, license, BN/TN, compatibility and all CUDA/C++ sources.
- The older recurrent-v1 RTX 4080 formal 144-unit job remains untouched and
  running. Backend-v2 GPU validation starts only after that job releases the
  card; its results cannot be relabeled as backend-v2 evidence.

### 2026-08-28 — quantized Graph boundary closed before GPU acceptance

- Restored the historical fail-closed distinction between adapter-owned
  packed Linear modules and external TorchAO/BitsAndBytes wrappers. Native
  MM8/MM4/A8W8/Marlin modules must expose the package-owned
  `rwkv7_forward_into` stable-output ABI before decode or prefill may attempt a
  CUDA Graph. External wrappers require the exact-card policy or the explicit
  graph override and otherwise execute the fused eager model route.
- Quantization reports now record dynamic prefill/decode Graph capability and
  its reason. The validator separately records the actual native prefill route,
  native cached-decode route, and Graph capability; a requested selector is
  still not accepted as evidence.
- Local command:
  `python -m compileall -q evaluation kernels/rwkv7_kernels rwkv7_hf tests &&
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q && git diff --check`.
  Result: `76 passed`. GPU numerical acceptance remains pending and production
  `RWKV7_MODEL_KERNEL_IMPL=auto` stays disabled.

### 2026-08-28 — lm_eval selected-answer and NLL evidence tightened

- Local model provenance now hashes the RWKV vocabulary and other tokenizer
  payloads in addition to code, config, and safetensors.
- The three-way validator no longer treats per-sample `acc`/`acc_norm`
  correctness booleans as proof that two lanes selected the same option. It
  reconstructs the raw and length-normalized selected choice from each
  `filtered_resps`/request record, checks LAMBADA greedy continuation outcomes,
  and compares them both across lanes and across batch 1/8.
- Wikitext validation now checks each document's rolling NLL and word/byte
  counts at the same `0.1%` relative gate as aggregate NLL/PPL instead of only
  comparing the aggregate metric.
- Local gate after these evidence changes: `79 passed`; Ruff, compileall, and
  `git diff --check` pass. Existing raw samples remain outside Git.

### 2026-08-28 — immutable-wheel HF ecosystem acceptance harness

- Added `evaluation/validate_backend_v2_ecosystem.py` to exercise one staged
  model and the exact HF/kernel wheel pair through standard AutoConfig,
  AutoTokenizer, AutoModel, AutoModelForCausalLM, greedy/beam generation and
  safe save/reload, followed by one-step Accelerate and Transformers Trainer
  BF16 training.
- Plain dense BF16 training is accepted only when the actual model route is
  `native-nvidia-train-temp-autograd-v2`; merely requesting the native backend
  is not evidence. PEFT LoRA and TRL SFT deliberately require the readable
  reference autograd route with an adapter-specific rejection reason, and
  verify non-zero finite gradients, parameter changes and PEFT save/reload.
- The ecosystem harness uses a deterministic local synthetic dataset and no
  network access. Its report records environment, model fingerprint, wheel
  hashes, source SHA, backend environment and every actual route. Canonical
  SFT/DPO/GRPO dataset runs remain a separate release gate.
- Local gate: `80 passed`; Ruff, format check, compileall and
  `git diff --check` pass. RTX 4080 execution waits for the untouched formal
  recurrent-v1 lm_eval job and the already queued immutable backend-v2 smoke.

### 2026-08-28 — canonical finetune backend route provenance

- Canonical SFT/DPO/GRPO runs now accept optional exact HF/kernel wheel paths
  and hash them into `artifact_provenance.json`. Local model provenance also
  includes vocabulary, tokenizer, template, config, code and weight payloads.
- The shared Trainer callback records de-duplicated actual model routes at log
  and pre-optimizer events. `training_checks.json` distinguishes native dense
  BF16 training from the required adapter-aware reference fallback.
- `validate_finetune_runs.py --require-backend-v2-routes` now requires both
  wheel SHA256 values and proves that every LoRA method used the clean
  reference autograd path for optimizer-bearing forwards rather than silently
  bypassing the adapters. The ordinary clean-reference validator remains able
  to run without an installed optional backend.

### 2026-08-28 — formal lm_eval artifact identity gate

- `run_lm_eval_matrix.py` now records the exact HF/kernel wheel SHA256 values
  and the pinned FLA source revision in both lane-level provenance and every
  manifest row. The runner refuses an FLA tree that does not resolve to
  `80e494f6c588e091fc8316b612870df29375c5b8`.
- The three-way validator requires both immutable wheel hashes to be present
  and identical across reference/optimized/FLA lanes, and requires the exact
  FLA commit in all three lane bundles before comparing predictions or
  metrics. This prevents results from different installed artifacts being
  merged into a nominal 144-unit matrix.
- It also requires identical safetensors, vocabulary, tokenizer/template
  payloads, dataset fingerprints and harness source SHA for each corresponding
  unit. FLA's intentionally different config/model wrapper is excluded from
  this semantic identity check; its underlying weights and inputs are not.

### 2026-08-28 — full backend evidence provenance

- The common GPU evidence helper now hashes every model config, remote-code
  module, tokenizer/vocabulary/template payload and safetensors file into one
  deterministic model revision instead of recording only config and weights.
- Environment reports now include Accelerate, Datasets, PEFT, TRL, W&B,
  BitsAndBytes, TorchAO, lm_eval and both RWKV7 distribution versions in
  addition to Python/Torch/Transformers/Triton/FLA/CUDA/driver/GPU. All
  backend-v2 inference, training, quantization, FLA, benchmark and ecosystem
  reports therefore share the same complete provenance schema.

### 2026-08-28 — optional-backend user and reproduction documentation

- Expanded the separate kernel package README with the API-v2 ownership,
  canonical-cache rule, supported NVIDIA/quant/training families, fail-closed
  adapter behavior and explicit pre-release route selectors.
- Added copyable artifact-hashed SFT/DPO/GRPO validation and three-way 144-unit
  lm_eval commands. The documentation distinguishes dense native BF16
  autograd from LoRA reference fallback and states every provenance/prediction
  gate without claiming GPU acceptance before its JSON passes.

### 2026-08-28 — final RTX 4080 backend-v2 queue frozen

- The immutable package artifacts remain the wheel pair built from package
  commit `18836aee380582253944231085f5de11c9e36303`:
  HF `237f4561ce59e3b4bbf385489bf9d0620e1b9f24877bd1990d8f00d9d7c6673c`,
  kernels `a36be47896f17ba40fbaf0e78cf486a79b929c5e7ecc3d71ae1a16a619596156`.
  The final validation-harness source is
  `ead3bee19348392e6a19dc8e9d0ccbf61cf3da0b`; package code did not change.
- The older recurrent-v1 three-way formal job remains untouched at PID
  `3883946`. At 2026-08-28 01:29 CST it had 18 reference manifest rows and
  was the only GPU consumer. No backend-v2 watcher competes with it.
- Final sequential RTX 4080 dependency chain:
  backend smoke PID `3892299`; HF ecosystem PID `3898158`, result
  `ecosystem-ead3bee1`; canonical SFT/DPO/GRPO PID `3898167`, result
  `finetune-ead3bee1`; BF16/training/all-quant/FLA/benchmark PID `3898175`,
  result `full-ead3bee1`; artifact-bound 144-unit matrix PID `3898230`, result
  `lm-eval-ead3bee1-v2`.
- Earlier provenance-incomplete waiting scripts were stopped before GPU
  execution and marked `superseded`; no formal computation was terminated.
  All final watchers verify the upstream JSON before proceeding and wait for
  an empty GPU process list, so every stage uses the same card exclusively.

### 2026-08-28 — V100 immutable-artifact pre-staging

- Kept the fixed device order: no V100 GPU gate was started before the final
  RTX 4080 backend-v2 evidence becomes internally consistent.
- Verified the same immutable wheel pair on V100 under
  `/home/data/wangyue/artifacts/backend-v2-18836aee`; `sha256sum -c
  SHA256SUMS` passes with the frozen HF and kernel hashes above.
- Staged 0.1B/0.4B/1.5B directories under
  `/home/data/wangyue/models/rwkv7/backend-v2-18836aee`. Their safetensors
  hashes exactly match the RTX 4080/reference artifacts, all six canonical HF
  source files match validation harness `ead3bee19348392e6a19dc8e9d0ccbf61cf3da0b`,
  and no legacy `kernel_bridge.py` is present.
- Created pinned-FLA wrappers under
  `/home/data/wangyue/models/rwkv7/backend-v2-18836aee-fla` and verified the
  source marker is exactly `80e494f6c588e091fc8316b612870df29375c5b8`.
- Installed the exact two wheels with `--no-deps --target` into independent
  inference and canonical-training overlays. Both import `rwkv7_hf==1.0.0`
  and `rwkv7-kernels==1.0.0.dev0` from the overlay and expose kernel API v2;
  neither base virtual environment was mutated.

### 2026-08-28 — six-repository Hub release baseline

- Added `evaluation/audit_hub_release.py` with unit tests. It audits the six
  Hub repositories without downloading weights: canonical code hashes,
  required/forbidden files, `auto_map`, resolved main revision, tag target,
  and every safetensors LFS SHA256/size are recorded.
- Captured the pre-release weight baseline at
  `results/release-preflight/hub-baseline-20260828.json` using harness commit
  `b0077e3b53510ca1604b1780685496121129e1e5`. The report is intentionally
  `failed`: every repository still has the v0.9 versions of
  `cache_rwkv7.py`, `configuration_rwkv7.py`, `modeling_rwkv7.py`, and
  `ops_rwkv7.py`. Tokenization and chat-template sources already match.
- The six current main revisions and all 32 LFS weight shards are now frozen
  as the before-release evidence. The final `v1.0.0` audit must pass while
  matching these exact weight hashes and sizes.

### 2026-08-28 — atomic two-package PyPI workflow

- Updated `.github/workflows/publish.yml` to build and `twine check --strict`
  both `rwkv7-hf` and `rwkv7-kernels`, require their versions to equal the
  GitHub release tag, and publish them from separate immutable artifacts.
- `rwkv7-kernels` publishes first. The stable `rwkv7-hf` job depends on it, so
  a missing companion-project trusted publisher cannot create an HF-only
  partial release.
- Current PyPI API state is `rwkv7-hf==0.9.0` present and `rwkv7-kernels` not
  yet created (HTTP 404). Before the release is published, the pending trusted
  publisher for `rwkv7-kernels` must name this repository, `publish.yml`, and
  the `pypi` GitHub environment. No token is stored in the repository.
- The kernel candidate remains `1.0.0.dev0` during diagnostic GPU acceptance.
  Stable `1.0.0` plus production `auto` are deliberately deferred until all
  migrated phases pass; the exact final wheels must then receive the full
  three-device release matrix before publication.

### 2026-08-28 — explicit SM70 training capability profile

- The migrated whole-model train-temp implementation is BF16 and intentionally
  rejects compute capability below sm80. V100 must not be reported as native
  train-temp success or as an unexplained failure.
- `validate_backend_v2_training.py` now has two distinct gates: BF16 native
  autograd, or FP16 reference fallback with identical logits, loss, and every
  gradient while the optional wheel remains installed. Actual route evidence
  is required in both cases.
- `validate_backend_v2_ecosystem.py` applies the same distinction to
  Accelerate and Trainer. PEFT and TRL LoRA continue to require the explicit
  adapter-aware reference route, in BF16 on supported cards or FP16 on V100.
- `validate_backend_v2_fla.py` records full-model BF16 training as
  `not_applicable` on SM70 instead of claiming it passed. Recurrent operator
  input/state gradient parity against FLA remains mandatory on V100.
- Local gate after the device-profile changes: `87 passed`; Ruff on all
  modified first-party files, compileall, and `git diff --check` pass.

### 2026-08-28 — generic v1 Hub staging and resumable publication

- Replaced the version-locked `prepare_hf_v090_release.py` and
  `publish_hf_v090_release.py` names with generic release tools. The tag is an
  explicit staged field and defaults to `v1.0.0`; an inconsistent publish
  request fails before any Hub write.
- Staging now removes `attn_mode`, `fuse_norm` and all kernel/backend selectors
  from `config.json`. Model repositories contain only architecture/tokenizer/HF
  contract data; optional policy remains in `rwkv7-kernels`.
- The v1 model card keeps package-free reference loading as the default and
  documents the optional companion without making a route claim. Publishing
  still uses each recorded parent commit, never uploads safetensors, and can
  safely resume by verifying an existing tag file-by-file.
- A six-repository dry run against the current Hub parents passed: all staged
  canonical sources were byte-identical to `rwkv7_hf/`, all configs were free
  of backend fields, and every planned commit contained only README, config,
  and six small runtime/tokenizer files. No Hub write was performed.
- `verify_hf_release.py` is now v1-generic and rejects backend policy leaked
  into model config. Local gate after the release-tool cleanup: `90 passed`.

### 2026-08-28 — V100 final-harness verification while RTX 4080 remains occupied

- Staged validation/release harness commit
  `4408e9e1dbd27c946b7b915bfcb6332561cf6e3a` at
  `/home/data/wangyue/repos/codex-build/hf-adapter-kernels-v1-harness-4408e9e1`.
  Its `.codex-source-sha` marker and all inference/training/ecosystem/FLA and
  Hub release entry points were verified before any V100 GPU work.
- Removed macOS AppleDouble `._*` transport metadata from the staged tree;
  these were never source files but made a recursive `compileall` attempt fail
  with null-byte errors. After removal, the pinned inference Python completed
  `compileall` over `evaluation/`, `examples/`, `scripts/`, and `rwkv7_hf/`.
- All six canonical `rwkv7_hf/*.py` SHA256 values on V100 exactly match the
  local `4408e9e1` worktree. This is source-transfer evidence only; the V100 GPU
  acceptance sequence remains intentionally gated on internally consistent
  RTX 4080 backend-v2 JSON.
- At `2026-08-28 02:46 +08:00`, the older recurrent-v1 RTX 4080 formal matrix
  had 19 successful reference manifest rows. Its active 0.4B batch-1
  HellaSwag process was still live at 100% CPU, and all backend-v2 watchers
  remained asleep. No process was terminated, restarted, or given competing
  GPU work.
- Opened upstream draft PR
  [`rwkv-rs/hf-adapter#146`](https://github.com/rwkv-rs/hf-adapter/pull/146)
  from `123123213weqw:perf/optional-kernels-v1`. The PR explicitly remains a
  draft and says production `auto`, stable `1.0.0`, and merge are blocked on
  the final immutable three-device matrix.
- A single read-only RTX 4090 SSH probe at the end of this session still timed
  out to `36.103.236.3:22`. No repeated connection loop or remote work was
  started; 4090 artifact staging remains pending connectivity.

### 2026-08-28 — V100 training-speed capability is explicit

- `benchmark_backend_v2.py` now accepts the same hardware capability split as
  the correctness harnesses: `native`, `reference-fallback`, or
  `skip-not-applicable`, plus an explicit BF16/FP16 training dtype.
- The V100 diagnostic profile will use
  `--training-mode reference-fallback --training-dtype fp16`. Its optimized
  lane is accepted only when the installed optional wheel records the actual
  `torch-reference-model-v1` training route and a non-empty fallback reason;
  it cannot be mislabeled as native train-temp throughput.
- A genuinely unsupported measurement can instead be recorded as
  `status: not_applicable`; it is no longer necessary to omit the training
  section and leave the result ambiguous. Native sm80+ behavior and route
  requirements are unchanged.
- This is validation-harness code only and does not change either immutable
  diagnostic wheel. Focused Ruff/format/compile checks and the full local test
  suite pass: `92 passed`.

### 2026-08-28 — V100 diagnostic and formal runners staged, not started

- Staged harness commit
  `185ac15544e044c1a8cc3ca92e40f550334a5690` under
  `/home/data/wangyue/repos/codex-build/hf-adapter-kernels-v1-harness-185ac155`.
  The marker, recursive compile, and all six canonical model-source hashes
  pass. The package code and diagnostic wheel hashes remain unchanged.
- Added `torchao==0.12.0` only to the independent V100 inference overlay. It
  imports with the existing `torch==2.5.1+cu124`, including the required
  `torchao.quantization.quantize_` API; neither base virtual environment was
  modified.
- Staged the resumable V100 correctness/HF/training/quant/FLA/finetune/speed
  runner at `/home/data/wangyue/codex-run/run-backend-v2-18836aee-v100.sh`,
  SHA256
  `908ffd948271a15fa266bac5afaa705d48696046230783b20893a2d31e8978a5`.
  It records every command and exit code, skips only stages with an explicit
  passed marker, uses FP16 reference-fallback route gates on SM70, and refuses
  to start while its activation file is absent.
- Staged the dependent formal three-way runner at
  `/home/data/wangyue/codex-run/run-backend-v2-18836aee-v100-lmeval.sh`, SHA256
  `7c1e5e153c214167711044faa1902c81ac07594b83aa7a7fd51ed1bd19da0d4e`.
  Reference and optimized 48-unit lanes use the two V100s concurrently, FLA
  follows after both exit zero, and the strict 144-unit validator is last.
  The runner refuses to start unless the preceding V100 diagnostic JSON says
  `passed`.
- Both remote scripts pass `bash -n` and are deliberately not running. The
  activation file is absent, preserving the required device order while the
  untouched RTX 4080 formal job and its queued backend-v2 chain continue.
- Upstream draft PR #146 now points at remote head
  `317eea57dc541e8ac894e7ef247271bdbcfc942d`. GitGuardian, the clean reference
  model job, the training-stack job, Python 3.10 with Transformers 4.48.3, and
  Python 3.12 with Transformers `<6` all completed successfully.

### 2026-08-28 — compact evidence builder is fail-closed

- Added `evaluation/build_backend_v2_compact_bundle.py` for the final 4080,
  V100, and 4090 Git evidence. It keeps small JSON/JSONL summaries, manifests,
  configs, commands, exit codes and environment text while excluding raw
  samples, lm_eval result payloads, runtime logs, weights, wheels, checkpoints,
  W&B state and model/artifact trees.
- The builder rejects symlinks, an output nested under the raw input, eligible
  files above the size gate, and known Hugging Face/PyPI/W&B/bearer secret
  forms. It writes builder provenance and exclusion counts to `BUNDLE.json`,
  hashes every included file in `MANIFEST.sha256`, and validates complete
  manifest coverage both before and after the atomic directory rename.
- Added tests for inclusion, every major raw exclusion, manifest verification,
  secret rejection, unsafe output layout and symlinks. Focused Ruff/format/
  compile checks and the full local suite pass: `96 passed`.
- At `2026-08-28 03:08 +08:00`, the untouched RTX 4080 recurrent-v1 reference
  lane remained at 19/48. Its active 0.4B B1 HellaSwag unit was at 68%
  (`27,413/40,168`, about 33 minutes remaining) with live GPU utilization;
  all backend-v2 watchers were still asleep.
- PyPI release configuration was inspected in both available browser sessions;
  both are currently logged out. No credentials were entered and no publisher
  setting was changed. The `rwkv7-kernels` pending trusted-publisher step
  remains a final release prerequisite rather than being bypassed with a token.
- The compact builder also passed a real repository preflight against
  `results/release-preflight`: two evidence files plus `BUNDLE.json` were
  copied to a temporary bundle and every `MANIFEST.sha256` row revalidated.

### 2026-08-28 — exact PyPI byte audit added

- Added `evaluation/audit_pypi_release.py`. The final command will query exact
  `rwkv7-hf==1.0.0` and `rwkv7-kernels==1.0.0` version endpoints, require a
  non-yanked wheel and valid SHA256 metadata for each, and compare the
  published filename, size, and SHA256 with the immutable local wheel pair.
- The report records its command, Python, index URL, harness SHA, dependency
  metadata, every release file and upload timestamp. Missing projects and
  network failures produce a written `status: failed` report rather than an
  unstructured exception.
- Live preflight proves the current expected boundary: `rwkv7-hf==0.9.0`
  passes the public API audit, while `rwkv7-kernels==1.0.0` returns HTTP 404 and
  keeps the aggregate report failed. No diagnostic or placeholder package was
  uploaded to manufacture a passing result.
- Added exact-byte success and mismatch tests. Focused Ruff/format/compile
  checks and the full local suite pass: `98 passed`.

### 2026-08-28 — publication now consumes the validated wheel bytes

- Closed a release-integrity gap in `.github/workflows/publish.yml`: rebuilding
  distributions after the GPU matrix could produce different wheel bytes from
  those validated. The release workflow no longer invokes `python -m build`.
- The final procedure is now draft-first. Attach the exact four validated wheel
  and source archives, `SHA256SUMS`, and `release-provenance.json`; publishing
  the GitHub release triggers a workflow that downloads and verifies those
  assets before sending the same files to PyPI. `rwkv7-kernels` still publishes
  first, and `rwkv7-hf` still cannot create a partial release if it fails.
- Added `scripts/verify_release_assets.py`. It requires source/version/artifact
  identity, fixed FLA commit, one shared harness and wheel pair, and compact
  evidence for RTX 4080, Tesla V100, and RTX 4090. Each device must explicitly
  pass correctness, HF ecosystem, training, quantization, FLA, speed,
  SFT/DPO/GRPO, and all 144 formal lm_eval units.
- `release-provenance.json` itself must be covered by `SHA256SUMS`; symlinked,
  missing, byte-different, unvalidated, wrong-device, or wrong-wheel assets
  fail before either trusted-publisher job starts.
- Added workflow structural and release-provenance tests, including rejection
  when one card used another kernel wheel. Full local gate: `101 passed`.

### 2026-08-28 — final provenance is generated from compact GPU evidence

- Added `scripts/build_release_provenance.py`; final release metadata is no
  longer hand-authored. It accepts the exact four stable archives plus the
  compact RTX 4080, Tesla V100, and RTX 4090 bundles, validates every complete
  manifest, and writes deterministic `release-provenance.json` and
  `SHA256SUMS` without rebuilding or modifying an archive.
- Every compact bundle must carry manifest-covered
  `release-validation.json` evidence for correctness, HF ecosystem, dense
  training/reference fallback, all quantization families, FLA, speed,
  SFT/DPO/GRPO, and the 144-unit three-way `lm_eval` gate. Source SHA, harness
  SHA, FLA commit and both wheel hashes must be identical across all cards.
- Actual prefill, decode, training and quantization implementation routes are
  mandatory. Policy selectors such as `auto`, `optimized`, `graph` or
  `triton` are rejected as route evidence.
- Failure tests cover a missing gate, different wheel bytes, wrong harness,
  invalid compact manifest and selector-only routes. The release verifier now
  independently rechecks the actual route map. Focused checks and the complete
  local suite pass: `107 passed`.
- At `2026-08-28T03:38:25+08:00`, the untouched RTX 4080 recurrent-v1
  reference lane had advanced to 21/48 successful units with zero recorded
  failures. `0.4b-b1-arc_easy` was the only active GPU child; all five
  backend-v2 diagnostic/formal watchers remained asleep and were not restarted
  or given competing work.

### 2026-08-28 — per-device release summary is also generated, not asserted

- Added `evaluation/build_backend_v2_device_validation.py`. It consumes the
  individual correctness, HF ecosystem, training, quantization, pinned-FLA,
  speed, finetune and three-way `lm_eval` JSON files and produces the
  `release-validation.json` later covered by the compact manifest.
- The tool requires every primary report schema/status and harness SHA, checks
  both exact wheel hashes in every primary report, checks the pinned FLA commit
  in parity/speed/formal-eval evidence, and requires the formal result to have
  144 units with whole-model route validation enabled.
- SFT, DPO and GRPO are checked separately, including their wheel hashes and
  adapter-aware actual training routes. Actual prefill/decode/training routes
  are extracted from validator output; quantization routes bind each passed
  method name to its executed implementation instead of trusting a requested
  policy.
- Failure tests cover a failed primary gate, report from another wheel,
  missing actual route and unpinned FLA revision. Focused checks and the full
  local suite pass: `112 passed`.

### 2026-08-28 — final wheel audit proves the full NVIDIA migration is shipped

- Added `scripts/audit_release_wheels.py` and made it a mandatory part of
  `verify_release_assets.py`. The audit opens the exact release wheels rather
  than inspecting the checkout.
- The kernel-wheel audit requires all adapted runtime/protocol/dispatcher/
  recurrent/graph/training/quantization modules, rejects any copied HF
  model/config/cache owner, reads the embedded source-migration manifest, and
  recomputes every one of the 102 migrated NVIDIA destination hashes. The HF-wheel
  audit independently requires the seven canonical model/tokenizer assets and
  rejects the optional kernel package plus the removed compatibility/tooling
  names.
- The existing immutable diagnostic artifacts pass the new audit without a
  rebuild: HF wheel
  `237f4561ce59e3b4bbf385489bf9d0620e1b9f24877bd1990d8f00d9d7c6673c`
  contains 18 members; kernel wheel
  `a36be47896f17ba40fbaf0e78cf486a79b929c5e7ecc3d71ae1a16a619596156`
  contains 125 members, including 102/102 destination-hash-verified files and all
  15 required adapted runtime files. These remain diagnostic, not final stable
  release artifacts.
- Failure tests remove one migrated file, alter one migrated payload, omit the
  manifest, and inject cross-package ownership in each direction. The full
  release-provenance tests now use structurally valid wheel fixtures. Focused
  Ruff/format/compile checks, `git diff --check`, and the complete local suite
  pass: `117 passed`.
- At `2026-08-28T03:48:20+08:00`, the untouched RTX 4080 recurrent-v1
  reference lane reached 22/48 successful units with no recorded failure.
  `0.4b-b1-arc_challenge` was the sole GPU process. The backend-v2 chain stayed
  queued, and upstream draft PR #146 had all five current checks green at
  source `ee6f9e3a977680ac775c876777eb864164b5c860`.
- A direct-entrypoint smoke exposed and fixed a packaging-independent CLI
  issue: the newly added release/device tools imported repository namespace
  packages only when invoked with `python -m`. They now bootstrap the checkout
  root and all documented `python scripts/...` / `python evaluation/...`
  commands return `--help` successfully. A subprocess test covers all four
  release entry points. The complete local suite now passes `118` tests.

### 2026-08-28 — RTX 4080 native-training compiler preflight repaired early

- A read-only preflight of every queued 4080 validator/finetune entry point and
  both environments passed. Exact diagnostic wheel hashes still match; the
  train environment has Torch `2.11.0`, Transformers `4.56.2`, Accelerate
  `1.14.0`, PEFT `0.19.1`, TRL `0.20.0`, Datasets `5.0.1`, and W&B `0.28.2`.
- The same preflight found a real infrastructure failure before it consumed
  GPU time: Torch is `2.11.0+cu130`, but the host had no system `nvcc` and
  `torch.utils.cpp_extension.CUDA_HOME` was `None`. The queued native
  train-temp smoke would therefore have failed at lazy extension compilation.
- Installed a validation-only CUDA 13.0.88 compiler overlay without touching
  or rebuilding either immutable RWKV wheel. It is assembled from the exact
  official package bytes:
  `nvidia-cuda-nvcc` SHA256
  `56fe502eb77625a12f25172caa3cdddb4e4c8ba2c8c17dba44b164761b380f03`,
  `nvidia-nvvm` SHA256
  `c5f41ffeb6466944a026dfa5317d7d85355c119bbec279205d22f1869d1054e0`,
  and `nvidia-cuda-crt` SHA256
  `2c8043c7c9e02492716426e9919fc78d2c5b3b2a7a768a88e952676b08aa55a4`.
- The compiler prefix is
  `/home/wzu/codex-run/toolkits/cuda-13.0.88`; validation-only
  `sitecustomize.py` binds `CUDA_HOME` and a dedicated extensions cache for all
  five already-sleeping backend-v2 watchers when their future Python children
  start. `nvcc V13.0.88` successfully compiled an `sm_89` CUDA object without
  using the GPU. The compiler-overlay provenance is recorded at
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-18836aee/4080/toolchain-preflight.json`
  (SHA256
  `18435be60f5ef54710e238ff0f0e438f9439dc5fdfc4226e2460e3585e607f70`).
- `evaluation/common.py` now records compiler/backend environment provenance in
  every final report. The per-device release builder requires native compiler
  identity on 4080/4090 and the distinct reference-fallback profile on V100.
  New failure tests cover a missing compiler identity; the complete local suite
  passes `121` tests.

### 2026-08-28 — CUDA compiler preflight is now reproducible

- Added `evaluation/preflight_cuda_toolchain.py` so the validation-only CUDA
  setup is no longer evidenced by a one-off shell command. It requires the
  PyTorch and `nvcc` CUDA major/minor versions to match, binds the
  `PROVENANCE.txt` SHA256 and target SM, and compiles a small CUDA object before
  any native-training GPU stage starts.
- The report preserves the real command, runtime environment, compiler
  version, source/object hashes, exit code and failures. A failed compiler,
  missing provenance, invalid SM target or mismatched toolkit writes a failed
  JSON report and exits nonzero.
- The final per-device builder independently parses the native training
  report and rejects an `nvcc` CUDA major/minor that differs from the PyTorch
  CUDA runtime. It also binds `nvcc` to `CUDA_HOME/bin/nvcc` and requires an
  absolute, dedicated `TORCH_EXTENSIONS_DIR`; it does not rely only on the
  standalone preflight verdict.
- Direct-entrypoint and fake-compiler tests cover the documented invocation.
  Ruff, compileall, `git diff --check`, and the complete local suite pass:
  `125 passed`.
- At `2026-08-28T04:04:17+08:00`, the untouched RTX 4080 recurrent-v1
  reference lane had 24/48 successful units and zero recorded failures. Its
  0.4B batch-8 Wikitext child was the only GPU process; all five backend-v2
  watchers remained asleep.

### 2026-08-28 — final GitHub/Hub/PyPI completion gate is fail-closed

- Extended `scripts/verify_hf_release.py` with a non-destructive fresh-cache
  contract. Final Hub smokes must use a distinct absent/empty cache plus
  `--force-download`; the report records that fact alongside the resolved tag,
  model/cache class, finite forward and cached generation.
- Added `evaluation/audit_github_release.py`. It resolves annotated tags to the
  exact source commit, proves the tag is contained in default `main`, checks
  the release PR is merged, verifies required architecture/evaluation/source
  paths, downloads and hashes every release asset, and requires the public
  validation Issue to cover 144-unit lm_eval, Wikitext NLL/PPL, SFT/DPO/GRPO,
  Trainer/Accelerate/PEFT/TRL, state/cache/generation, quantization, actual
  routes, three GPUs, FLA and SHA256 evidence.
- Added `scripts/verify_end_to_end_release.py`. It repeats the immutable
  three-device release-asset gate, then cross-checks the six Hub repositories,
  unchanged weight baseline, six fresh-download smokes, exact PyPI wheel bytes,
  and GitHub tag/release/branch/PR/docs/Issue evidence into one final JSON.
- Ruff, compileall, `git diff --check`, direct-entrypoint smoke, and the full
  local suite pass: `131 passed`.
- At `2026-08-28T04:31:10+08:00`, the untouched RTX 4080 recurrent-v1
  reference lane was 27/48 with zero recorded failures; the five backend-v2
  watchers remained sequentially queued and did not receive competing work.

### 2026-08-28 — formal lm_eval compact evidence retains the actual metrics

- `validate_lm_eval_three_way.py` now preserves the complete 144-unit compact
  aggregate metric matrix in its validation JSON. Accuracy metrics and
  Wikitext NLL/PPL no longer disappear when raw samples/results are excluded
  from the Git bundle.
- The report also records a fixed comparison summary for all 96
  optimized/FLA-vs-reference comparisons: metric failures, selected-answer
  mismatches, continuous NLL mismatches and missing documents must all be
  zero.
- The per-device release builder requires three 48-unit aggregate lanes and
  the zero-mismatch summary before it can emit `release-validation.json`.
  Status/exit codes alone are no longer sufficient evidence.
- Ruff, compileall, `git diff --check`, and the complete local suite pass:
  `133 passed`.

### 2026-08-28 — public validation Issue is rendered from evidence

- Added `scripts/render_release_issue.py`. It accepts only a fully passed
  three-device release provenance plus the exact speed and formal lm_eval JSON
  from RTX 4080, V100 and RTX 4090.
- The generated Markdown includes immutable source/harness/wheel/FLA SHA256
  identities, every functional/HF/training/quantization/finetune gate, actual
  implementation routes, complete whole-model/operator/training speed tables
  against both reference and FLA, and every retained accuracy/NLL/PPL unit.
- Rendering fails if a speed report uses another wheel/harness/FLA revision,
  if the formal 144-unit metric matrix is incomplete, if any of the 96
  candidate comparisons has a mismatch, or if the Issue would exceed the
  GitHub size safety limit. The test also proves the renderer supplies every
  term required by the post-publication GitHub audit.
- Ruff, compileall, direct-entrypoint smoke, and the complete local suite pass:
  `134 passed`.

### 2026-08-28 — every migrated high-performance family is semantically audited

- Added the wheel-owned `nvidia/CAPABILITY_INVENTORY.json`. It maps all 102
  destination-hash-verified historical NVIDIA payloads exactly once into 16 capability
  families covering recurrent, dense/fused decode, DPLR/self-chunk/fused
  prefill, CUDA Graph/state pools, SM70/Ada/Blackwell, W8/W4/A8W8/BN-TN/BnB/
  Marlin/TorchAO, common quant runtime, and train-temp autograd.
- Extended `scripts/audit_release_wheels.py` to require the exact capability
  set, API v2 ownership, real adapted runtime files and real `KernelPolicy`
  fields. It now rejects missing/double-mapped historical files, unreachable
  runtime references, invented policy flags, and incomplete capability
  families in the built wheel.
- Added direct source-policy tests for exact V100, RTX 4080, RTX 4090 and RTX
  5090 route families plus adjacent-product fail-closed behavior. These tests
  verify the migration is represented in hardware dispatch rather than only
  stored as source files.
- At this historical stage, production whole-model `auto` remained disabled
  and the planned device order included V100. The current fixed decisions
  above supersede that order: RTX 4080 is the primary gate, RTX 4090 follows,
  and V100 is historical evidence only.
- Targeted Ruff, compileall, `git diff --check`, and the complete local suite
  pass: `143 passed`. A disposable development-wheel build was then audited
  from its ZIP contents: 16/16 capability families, 102/102 mapped and
  destination-hash-verified migration files, 23 reachable adapted runtime files and 46
  real policy flags. Its SHA256 is
  `0a7f8f162fde9def8dd31ada789e5ef364eabf68093d771326a8d9775489a3df`;
  it is a local audit artifact, not the final immutable `1.0.0` release wheel.
- The final public Issue renderer and GitHub audit now require the same full
  capability vocabulary: recurrent, dense decode, DPLR/self-chunk prefill,
  CUDA Graph/state pools, SM70/Ada/Blackwell, every W8/W4/A8W8/BN-TN/BnB/
  Marlin/TorchAO route, and training autograd. The GitHub source-tree audit
  also requires both embedded inventories and this migration audit document;
  a release can no longer publish only generic “optimized” wording.
- `docs/ARCHITECTURE.md` now documents the single API-v4 facade, its five
  operation kinds, and the complete operator families next to the readable
  model/cache structure. Installation never replaces a model class, cache ABI,
  or checkpoint layout, and unsupported calls fall back to the unchanged
  reference body.
- Source and documentation commits were pushed as
  `bf8ecdb0cdde9cc276507f6cd47833542e0ccc90` on
  `perf/optional-kernels-v1`; draft PR #146 points to the same SHA. At
  `2026-08-28T05:04:55+08:00`, the untouched RTX 4080 formal reference lane
  was still the only RWKV GPU workload (0.4B/B8 HellaSwag); all backend-v2
  watchers remained asleep and sequential. No process was stopped or
  restarted.
- A single read-only RTX 4090 probe at this stage timed out after eight
  seconds, so no 4090 validation was started or fabricated. V100 remained
  reserved for the final stable wheel pair after RTX 4080 acceptance, in
  accordance with the fixed device order.

### 2026-08-28 — historical denominator is now cryptographically complete

- Audited the complete `perf/native-kernels-v0.8:rwkv7_hf` tree rather than
  trusting the selected 102-file migration list. The current 153-row scope
  records 86 byte-migrated NVIDIA files, 26 adapted protocol/glue files, 7
  canonical reference owners, 6 relocated/retired tools, 27 explicitly
  separate Ascend/MLX/Biren/MetaX/MUSA files, and one retired non-kernel
  speculative helper.
- Added wheel-owned `nvidia/SOURCE_SCOPE.json` with the historical mode/blob
  identity and disposition of every file. `audit_release_wheels.py` rebuilds
  the Git tree and requires exact tree
  `1bb1fe1cd64662bbd6d29f72c9002a8513af3691`, cross-checks all NVIDIA rows
  against `MIGRATION_MANIFEST.json`, and verifies adapted kernel replacements
  are shipped. No `unknown` or `unclassified` disposition is accepted.
- A disposable wheel ZIP passed all three audits: 153/153 historical files,
  102/102 destination migrations, 16/16 capability families, 11 adapted kernel
  replacement files and five named separate hardware families. Development
  wheel SHA256:
  `a9918fdd79c1f1c722e57b1dfa022280efe3b622aeb84981b3eda5501693d90a`;
  this is not the final stable artifact.
- The public Issue/GitHub audit now requires the 153-file source-scope proof
  and `SOURCE_SCOPE.json` path in addition to the capability and byte
  manifests. RTX 4080 remained untouched: its active 0.4B/B8 HellaSwag unit
  was progressing at 27% with the GPU busy, not stalled.
- Targeted Ruff, compileall, `git diff --check`, source-scope corruption tests,
  wheel ZIP audit and the complete local suite pass: `147 passed`.
- Corrected both root READMEs: they no longer claim that performance remains
  on an unrelated branch. They now document the independently installable
  `rwkv7-kernels==1.0.0`, unchanged HF loading code, `auto`/`reference`/strict
  `optimized` semantics, complete operator ownership and explicit
  quantization choices in English and Chinese.
- Updated the six-repository publication document from the obsolete `v0.9.0`
  wording to the final `v1.0.0` contract: unchanged weight hashes, independent
  package-free Hub loading, optional kernel installation and mandatory fresh-
  cache redownload evidence for all six repositories.
- Updated the public evaluation commands to `v1.0.0` result roots and the
  final stable `rwkv7_kernels-1.0.0` wheel name. Diagnostic `.dev0` artifacts
  remain in the historical session log only and are no longer presented as
  release commands.
- Updated the canonical finetuning wrapper example to the same final stable
  kernel wheel name, keeping every user-facing release command consistent.

### 2026-08-28 — the later HF recurrent backend is also in the audited denominator

- Added wheel-owned `nvidia/RECURRENT_SOURCE_SCOPE.json` for the complete
  historical `perf/optional-native-backend-v0.10:kernel_wheel/rwkv7_kernels`
  subtree at commit `0c5ea30ac6868974ba9836c4a065fa8b2847af68`.
  Its three rows reconstruct frozen Git tree
  `7d2fe3ffff72ec2cd44993e14757ef4443ddfcbb`.
- The historical package entry point is adapted behind the current private
  dispatcher and single public API-v4 facade. The old Graph and Triton
  recurrence implementations are still byte-identical as
  `recurrent/graph.py` and `recurrent/triton.py`; the release-wheel audit now
  recomputes their SHA256 and Git blob identities in addition to the 153-file
  v0.8 scope and 102-file NVIDIA manifest.
- Updated architecture, reproducibility, migration-audit and generated Issue
  requirements so “all historical HF performance operators” covers both the
  large v0.8 NVIDIA tree and the later independently packaged recurrent line.
- A disposable development-wheel ZIP passed the combined audit: 153/153 v0.8
  source files, 102/102 NVIDIA destination migrations, 16/16 capability families,
  and 3/3 v0.10 recurrent-package files with 2/2 byte-identical
  implementations. Disposable wheel hashes are HF
  `4bb51faa154d7d51ccf3af2bac9f1eac712dde74fcc35a4fd58583172871253f`
  and kernels
  `a198e7949307eac4e1037383b59023546b5a07af21857ac8522b2fad73875efa`;
  neither is a final stable artifact. The complete local suite passes
  `150 passed` with `133` expected TorchScript deprecation warnings.
- Recovered the prior documentation push from a GitHub HTTPS transport
  timeout using the fail-closed Git Database API path. Local branch, fork ref,
  and upstream draft PR #146 first converged on exact commit
  `00463f55a0189a70c4b54d58c5f6c10bad98f542`; no divergent history was
  force-pushed. The complete recurrent-audit change then pushed normally as
  `e10a785444ae47fab54c98e57bb21de8f15e9e00`; local, fork and PR heads are
  identical, and all five upstream checks passed.
- At `2026-08-28T05:46:50+08:00`, the untouched RTX 4080 formal reference lane
  still had 27/48 completed units with zero failures. Its active
  0.4B/batch-8 HellaSwag unit was at 56% (`22407/40168`) with approximately
  52 minutes remaining. It was the only GPU process; all backend-v2 watchers
  remained asleep and sequential.

### 2026-08-28 — destination hashes are now tied to historical Git blobs

- Strengthened the release-wheel audit so an entry cannot claim byte identity
  merely by updating its destination SHA256 in the manifest. For every exact
  transfer, the audit now reconstructs `sha1("blob <size>\\0" + payload)` from
  the actual wheel member and requires it to equal the frozen historical Git
  blob ID.
- This stronger check found two deliberately adapted files that the earlier
  manifest had incorrectly counted as byte-identical. The implementation was
  already correct, but the evidence label was not: `native_graph_runtime.py`
  binds the canonical `RWKV7Cache` instead of the old private cache, and
  `official_training_cuda.py` removes whole-model `forward` monkeypatching in favor of
  `training_runtime.py` direct dispatch.
- Strengthened the machine-readable denominator and adaptation-rationale
  checks. Subsequent clean-boundary work brings the current manifest to **88
  byte-identical + 14 declared adaptations = all 102 NVIDIA transfers**; the
  complete current source scope is the 86/26/7/6/27/1 classification recorded
  above. Every adaptation is restricted by exact historical source path and
  requires a non-empty rationale. Capability coverage remains 102/102 across
  the same 16 families.
- A newly built disposable ZIP passed the stronger combined audit: 100 exact
  Git blobs, two declared adaptations, 102 destination SHA256 values, 153/153
  historical rows, 16/16 capability families, and both byte-identical v0.10
  recurrent implementations. Disposable hashes are HF
  `0880882799243cd643391108f16b649a2009be06e68d26ad7024708767e7319f`
  and kernels
  `d8c9add5731c0d8edf07a86ced093955d7d12995d5503b8cf86bf4bd058d0a3b`;
  they are not final stable artifacts. The complete local suite passes
  `153 passed` with `133` expected TorchScript deprecation warnings, including
  rejection of an undeclared third clean-boundary adaptation and a rationale
  that differs between the migration manifest and complete source scope.
- No runtime source or running GPU process changed during this evidence
  correction. The queued RTX 4080 diagnostic continues to use its original
  immutable wheel pair; final stable wheels will include the corrected audit
  metadata after the diagnostic gate.
- Added the user-facing `rwkv7-hf[kernels]==1.0.0` extra, pinned to the matching
  companion distribution. The two-package command remains valid, but users can
  now request the complete optional backend with one requirement. The
  release-wheel audit parses the built HF `METADATA` and rejects a missing or
  unpinned kernel extra; model cards and both READMEs document the equivalent
  forms. A disposable package build passed with HF wheel
  `693e6c8118000b43937868a74bc0366cd813348685d4681d1eb908f2cff352e8`
  and kernel wheel
  `0d200966845399a0216ab4d051fba2eecb4c44c2a1d3cf67a4689f40663f8721`.
  These remain non-final audit artifacts. The complete local suite now passes
  `154 passed` with `133` expected warnings.
- At `2026-08-28T06:24:32+08:00`, the untouched RTX 4080 formal reference lane
  had advanced to 29/48 units with 29 exit-zero and zero failures. The active
  0.4B/batch-8 ARC-Easy unit was the only GPU process; the backend-v2 chain
  remained asleep and sequential.
- At `2026-08-28T07:13:05+08:00`, the same reference lane reached 35/48 with
  35 exit-zero and zero failures. The active 1.5B/batch-1 HellaSwag unit was
  the only GPU process and reported an approximately three-hour remaining
  estimate at the start of its 40,168 requests. No watcher was restarted or
  allowed to contend. Upstream draft PR #146 had all five checks green at
  source `63c61a0fc900b51ce258d85717689b40e1f57bad`.

### 2026-08-28 — source distributions are bound to the validated wheels

- Extended the immutable release verifier to audit both PyPI source archives,
  not only their SHA256 rows. It reads tar members without extraction, requires
  a single expected package root, rejects traversal, symlinks, hardlinks,
  devices and duplicate files, and validates `PKG-INFO` plus
  `pyproject.toml` name/version.
- Every `rwkv7_hf`, `rwkv7_hf_tools`, and `rwkv7_kernels` payload shipped in a
  wheel must exist byte-for-byte in the matching sdist. Cross-package
  ownership is rejected. This closes the case where validated wheels are
  published beside stale or unsafe source archives.
- Built real disposable wheel+sdist pairs and passed the new audit: HF wheel
  `717c9ff6eda82741782c2f9911e6d8cbde30acb546ec613ebc6e23e0b9c0d7cb`,
  HF sdist
  `d11526eb67de7d49300330ab4e9ab21fb9896e6d60b982da705ff54d1e478238`,
  kernel wheel
  `770ecd245e68468b5738b7a5fb4cd07b0cdaa888d8dd219f3cee46bb986c65a5`,
  and kernel sdist
  `6bcb44c944971f5d4aa67a436b5b6b13d642cc984fc17ee69ed0d1169c97d7a0`.
  They are development audit artifacts, not final stable files.
- Tests cover a source payload that differs from its wheel, a malicious tar
  symlink, and HF tooling injected into the kernel sdist. The complete local
  suite passes `157 passed` with `133` expected TorchScript deprecation
  warnings. RTX 4080 remained untouched throughout.

### 2026-08-28 — six-repository Hub transaction is cryptographically bound

- Replaced the old claim-only Hub stage list with schema
  `rwkv7-hub-release-stage-v1`. One manifest now requires all six repositories
  exactly once and binds the tagged source commit, each Hub parent commit,
  every staged README/config/reference byte, and every existing safetensors
  shard's Hub LFS SHA256/size. Staging rejects canonical source files that
  differ from the named Git `HEAD`.
- `publish_hf_release.py` rehashes the local stage before any write, rechecks
  parent and weight identities, never includes a weight in the commit, and
  verifies the exact tag files and unchanged LFS identities after publication
  or when resuming an already-published tag.
- The post-release Hub audit now requires all six repositories and
  `conversion_manifest.json`, proves the local source directory is the stated
  Git commit, force-downloads the small files, compares the complete staged
  file manifest, and rejects an incomplete weight baseline. The final
  all-surfaces verifier requires this stage-manifest proof rather than
  accepting canonical-code hashes alone.
- Fresh-cache Hub smokes gained `--require-package-free`; final evidence must
  show that neither `rwkv7-hf` nor `rwkv7-kernels` was installed or importable
  from a local checkout while the tagged remote model produced finite logits,
  `RWKV7Cache`, and cached generation. This makes package-free Hub loading an
  explicit six-model gate.
- Added `run_hub_release_smokes.py` to execute those six loads sequentially
  from an initially empty output root. Every model receives separate empty Hub
  blob and Transformers remote-code module caches, and the wrapper retains the
  exact command, timestamps and report SHA256 in one manifest.
- A real read-only six-repository stage and publish dry run passed against the
  frozen parents. A final-audit dry run failed exactly as expected before
  release because main still contains the old code/tag state, while all 32
  weight-shard identities remained equal to the frozen baseline. No Hub write
  occurred. Targeted Ruff/compileall, `git diff --check`, and the complete
  local suite pass `174 passed` with `133` expected warnings.

### 2026-08-28 — RTX 4080 formal reference lane reached 37/48

- At `2026-08-28T09:27:10+08:00`, the untouched recurrent-v1 diagnostic
  reference lane had completed **37/48** formal units, all 37 with exit code
  zero. The long 1.5B/batch-1 HellaSwag unit completed naturally, followed by
  Winogrande; 1.5B/batch-1 ARC-Easy was the sole active GPU unit (PID
  `3925792`, approximately 3.39 GiB).
- All 37 result JSON files parsed successfully and contained no non-finite
  numeric values. The completed HellaSwag log contained no traceback,
  RuntimeError, CUDA error or out-of-memory marker.
- The optimized and FLA formal lanes remain at 0/48. Backend-v2 watcher PIDs
  `3892299`, `3898158`, `3898167`, `3898175`, and `3898230` remain asleep in
  the required sequential chain; none has produced a final JSON or competed
  for the GPU. This is still the older `worktree-9d3ba79a-final-auto`
  recurrent-v1 diagnostic, not final stable-wheel evidence.

### 2026-08-28 — RTX 4080 formal reference lane reached 41/48

- At `2026-08-28T09:57:06+08:00`, the same untouched reference lane had
  completed **41/48** units, all 41 exit-zero. The 1.5B batch-1 matrix and the
  batch-8 WikiText unit had finished; 1.5B/batch-8 LAMBADA was the sole active
  GPU task (PID `3927691`, about 4.60 GiB), at 378/5,153 requests with no
  traceback, RuntimeError, CUDA error or OOM marker.
- All 41 generated result JSON files parsed and contained zero non-finite
  numeric values. Optimized and FLA remain at 0/48, and all five backend-v2
  watchers remain sleeping in sequence without a final report or GPU use.

### 2026-08-28 — RTX 4080 formal reference lane reached 43/48

- At `2026-08-28T10:27:34+08:00`, reference had completed **43/48** units,
  all exit-zero, through 1.5B/batch-8 PIQA. Every completed batch-1/batch-8
  aggregate metric pair compared so far was exactly equal.
- The sole GPU task was 1.5B/batch-8 HellaSwag (PID `3928770`, approximately
  4.40 GiB), at 2,827/40,168 requests. Its live log contained no traceback,
  RuntimeError, CUDA error or OOM marker. All 43 result JSON files parsed with
  zero non-finite numerics.
- Optimized/FLA remain 0/48 and the backend-v2 watcher chain remains asleep,
  unchanged and non-contending.

### 2026-08-28 — RTX 4080 formal reference lane completed 48/48

- At `2026-08-28T12:28:40+08:00`, the recurrent-v1 diagnostic reference lane
  completed **48/48**, all exit-zero. All 48 result JSON files parsed with no
  non-finite numeric values; completed batch-1/batch-8 aggregate metric pairs
  were exactly equal, and the full log set contained no traceback,
  RuntimeError, CUDA error or OOM marker.
- The existing coordinator advanced naturally to the optimized lane without
  intervention. Its first unit is 0.1B/batch-1 WikiText under requested
  `RWKV7_BACKEND=optimized` and `RWKV7_KERNEL_IMPL=auto`; actual route evidence
  must still come from the formal result validator and must not be inferred
  from these requested environment values.
- FLA remains 0/48. Backend-v2 watchers remain asleep until the complete older
  three-way diagnostic finishes, with no duplicate launch or GPU contention.

### 2026-08-28 — RTX 4080 optimized lane reached 12/48

- At `2026-08-28T13:28:06+08:00`, the recurrent-v1 diagnostic optimized lane
  had completed **12/48**, all exit-zero. All 12 completed aggregate result
  objects were exactly equal to their reference counterparts, and the combined
  result set contained no non-finite numeric values.
- The only GPU task was 0.1B/batch-8 Winogrande. Requested policy remains
  `optimized` with kernel implementation `auto`; this checkpoint records
  functional progress only and does not substitute the requested policy for
  actual route evidence.
- FLA remains 0/48 and the backend-v2 watcher chain remains sleeping and
  non-contending.

### 2026-08-28 — RTX 4080 optimized lane passed the halfway point

- At `2026-08-28T14:58:10+08:00`, optimized had completed **27/48**, all
  exit-zero. Each of the 27 completed aggregate result objects was exactly
  equal to reference and the combined result set contained no non-finite
  values.
- The sole GPU task was 0.4B/batch-8 HellaSwag. Its live log contained no
  traceback, RuntimeError, CUDA error or OOM marker. FLA remains 0/48 and the
  backend-v2 watcher chain remains asleep and non-contending.

### 2026-08-28 — RTX 4080 optimized lane reached 35/48

- At `2026-08-28T15:57:59+08:00`, optimized had completed **35/48**, all
  exit-zero. All 35 completed aggregate result objects remained exactly equal
  to reference and no result contained a non-finite numeric value.
- The sole GPU task was 1.5B/batch-1 HellaSwag, at 6,011/40,168 requests with
  no traceback, RuntimeError, CUDA error or OOM marker. FLA and backend-v2
  remain sequentially queued and non-contending.

### 2026-08-28 — RTX 4080 optimized lane reached 41/48

- At `2026-08-28T16:58:05+08:00`, optimized had completed **41/48**, all
  exit-zero. All 41 completed aggregate result objects remained exactly equal
  to reference, with zero non-finite values across the formal result set.
- The sole GPU task was 1.5B/batch-8 LAMBADA, at 2,698/5,153 requests and no
  error marker. FLA remains 0/48; the backend-v2 watchers remain sleeping in
  the original dependency order without GPU contention.

### 2026-08-28 — RTX 4080 optimized lane completed; FLA lane started

- At `2026-08-28T18:00:00+08:00`, the recurrent-v1 diagnostic optimized lane
  completed **48/48**, all exit-zero, with no non-finite values. All 48
  aggregate result objects were byte-for-byte equal to the reference lane.
- Every optimized unit includes actual route evidence. Across the completed
  lane it records `torch-cuda-graph-reference-v1` for 4,440,780 recurrent
  calls; therefore this is the exact Graph compatibility route, not evidence
  for the later complete backend-v2 implementation or the final 1.0.0 wheel.
- The coordinator advanced naturally to FLA. FLA had completed **5/48** units,
  all exit-zero and finite, while 0.1B/batch-1 ARC-Easy was the sole GPU task.
  Preliminary FLA aggregate values differ from reference on all five completed
  units, so the original exact-equality gate is not yet satisfied. Preserve
  the raw evidence and let the full lane finish before diagnosing or rerunning
  only affected comparisons.
- Backend-v2 watcher PIDs `3892299`, `3898158`, `3898167`, `3898175`, and
  `3898230` remain sleeping in the required sequential chain; none was
  restarted and none contended for the GPU.

### 2026-08-28 — RTX 4080 FLA lane reached 35/48

- At `2026-08-28T18:58:33+08:00`, the FLA lane had completed **35/48**
  recurrent-v1 diagnostic units, all exit-zero. All 35 result JSON files
  parsed with no non-finite numeric values and the completed logs contained no
  traceback, RuntimeError, CUDA error or out-of-memory marker.
- The sole GPU task was 1.5B/batch-1 HellaSwag at 33,892/40,168 requests. The
  backend-v2 watcher chain remained asleep in the original dependency order,
  without restart or GPU contention.

### 2026-08-28 — recurrent-v1 144/144 collected; backend-v2 smoke exposed a clean-boundary dtype bug

- At `2026-08-28T19:18:31+08:00`, reference, Graph-optimized and FLA all
  completed **48/48** formal units: **144/144 commands exited zero**, every
  result JSON parsed, and no non-finite value was found. Graph-optimized had
  zero per-sample prediction mismatches and zero metric failures against the
  reference. FLA completed successfully but the strict three-way validator
  correctly wrote `status: failed`: several discrete selections and aggregate
  metrics differ, so this diagnostic does not satisfy the original exact FLA
  equality gate and its raw evidence remains untouched.
- The queued backend-v2 chain then started naturally. Its first inference
  smoke failed before producing a report. Evidence is preserved at
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-18836aee/4080`:
  the clean model keeps the official decay bias in FP32, while the historical
  private native pack treated `RWKV7Linear` as a wrapped module and passed the
  FP32 bias directly to an FP16 `F.linear`, raising `self and mat2 must have
  the same dtype`. Dependent ecosystem, finetune, full and lm_eval watchers
  recorded transparent `blocked` results and exited without running or
  consuming GPU time.
- The affected clean-boundary adapters were repaired locally without changing
  `rwkv7_hf`, the public cache, model signatures or checkpoint parameters:
  ordinary `RWKV7Linear` subclasses are dense only when their weight is an
  exact `Parameter`; quantized tensor subclasses remain callable operands;
  the FP32 decay bias is converted only in the private activation-dtype native
  pack. Training, graph-head and quantization ownership checks use the same
  dense contract.
- At this historical checkpoint, migration evidence recorded **98
  byte-identical + 4 declared clean-boundary adaptations = all 102 NVIDIA
  files**; later adaptations superseded that snapshot with the current count
  in Fixed decisions. Targeted backend tests
  and the complete local suite pass **175 tests** with 145 expected TorchScript
  deprecation warnings; `git diff --check` passes. Next action is to rebuild an
  immutable corrected kernel wheel on RTX 4080 and rerun only the failed
  backend-v2 smoke before re-enabling its dependent sequential gates.

### 2026-08-28 — RTX 4080 reached through the V100 bastion; training JIT exposed an SM89 compile defect

- The working route is the V100 jump host, invoked explicitly as
  `ssh -o ControlMaster=no -o ControlPath=none -o ProxyJump=WZU_Server WZU_4080`.
  Through that route the corrected backend-v2 inference smoke completed with
  `status: passed`: finite FP16 logits, cosine `0.9999990`, max-abs
  `0.0546875`, canonical cache pass, native prefill routes and CUDA-Graph
  fused decode routes. This confirms the clean `RWKV7Linear`/FP32-decay-bias
  repair without rerunning the already-passing recurrent-v1 matrix.
- The first training retry found two environment prerequisites (`ninja` and a
  complete CUDA toolkit). After using the server's existing Ninja and complete
  CUDA 13.0 toolkit, compilation reached the CUDA source and exposed the real
  portability defect: the migrated train-temp files called
  `atomicAdd(float2*, float2)`, which CUDA only provides on SM90+, while the
  RTX 4080 is SM89.
- The three affected BF16 training translation units now retain the vector
  atomic on SM90+ and use two equivalent scalar FP32 atomics on SM89/SM80/SM70.
  This historical checkpoint recorded **95 byte-identical + 7 declared
  adaptations = all 102 NVIDIA files**; later adaptations superseded that
  snapshot with the current count in Fixed decisions. The frozen historical
  tree and original Git blob IDs remain unchanged. Targeted migration,
  wheel-audit, and backend-v2 tests pass 29/29 with plugin autoload disabled.
- Remote source sync was interrupted by a temporary loss of the V100
  Tailscale route. No formal GPU process was running or terminated. Next action
  is to resume the same V100 jump route, compile the corrected training leaf,
  and rerun only training plus the still-unrun quant/FLA/full backend-v2 stages.

### 2026-08-28 — RTX 4080 train-temp compiled on SM89; clean FP32 w0 handling repaired

- Training smoke v6 proved that all migrated CUDA/C++ train-temp extensions now
  compile on RTX 4080 after the architecture-gated atomic fix. Execution then
  reached the model and exposed a second clean-boundary dtype mismatch: the
  historical BF16 training path invoked the decay projection module directly,
  while the clean HF model intentionally stores its public w0 bias in FP32.
- The already-declared `official_training_cuda.py` adapter now mirrors the clean
  contract without changing the model: it evaluates the low-rank projection
  without bias, adds w0 in FP32, and casts only the private raw-decay operand
  consumed by the BF16 CUDA kernel. The public FP32 parameter and gradient edge
  remain intact. Targeted backend/migration/wheel tests pass 30/30.
- Only the affected training smoke has been restarted as v7 (PID `3968505`).
  No other formal GPU job was running, and quant/FLA/full stages remain pending
  behind this gate.

### 2026-08-28 — RTX 4080 native training executes and accelerates, but strict BF16 parity remains failed

- Training smoke v7 completed without exceptions and recorded the actual route
  `native-nvidia-train-temp-autograd-v2`. Both non-checkpointed and
  checkpointed cases were finite, had all 399 expected gradients, no missing
  gradients, and measured speedups of `1.68x` and `3.12x` over the clean
  reference path.
- The existing strict gate correctly remains failed. Logits cosine was
  `0.9997567`/`0.9998824`, loss delta `0.0339`/`0.0435`, and the worst
  per-parameter gradient cosine/relative-L2 was `0.9891/0.1754` and
  `0.9839/0.1961`. Every gradient is finite, every gradient cosine is at least
  `0.98`, and every relative-L2 is at most `0.20`, but these do not satisfy the
  pre-existing `0.9999` logits and `0.999`/`0.02` gradient requirements.
- The failed JSON is preserved as `training-smoke-v7.json`; no thresholds were
  weakened and no later quant/FLA/full stage was started. Next action is a
  focused numerical comparison of the historical fused recurrence against a
  higher-precision/hybrid training route before deciding promotion versus
  reference fallback.

### 2026-08-28 — RTX 4080 strict native-training smoke passes after numerical-order repair

- Focused hybrid runs localized the v7 discrepancy to the recurrent training
  leaf rather than Mix6, GroupNorm, ChannelMix, loss shifting, or the public HF
  model contract.  The historical leaf consumed a BF16 raw-decay operand,
  whereas the clean model intentionally performs the `w0` addition and decay
  transform in FP32.  Its CUDA compiler also contracted multiply-adds, causing
  one-ULP BF16 differences that accumulated through twelve residual blocks.
- The clean reference recurrence now expresses the RWKV-7 rank-one update
  directly and preserves the official sequential reduction order.  The
  migrated clamp ABI accepts canonical FP32 decay, rounds the `v @ k` outer
  product and updated-state view at the same BF16 boundaries, and compiles this
  leaf with `--fmad=false`.  The public modeling/config/cache boundary remains
  unchanged.  `RWKV7LowRank.project_without_bias()` makes the external FP32
  bias ownership explicit instead of teaching the backend about model internals.
- The accepted training route keeps canonical HF projection, normalization,
  gating and ChannelMix math, uses the native CUDA recurrence for the forward,
  and replays the readable canonical rank-one recurrence in autograd for strict
  gradients.  A faster vectorized/pairwise experiment was retained as failed
  diagnostic evidence and rejected; it is not part of the final source.
- `training-smoke-rank1-decay-fp32-nofma-replay-v28.json` passes both
  B1/T16 cases.  With and without gradient checkpointing, logits and loss are
  bit-exact, every expected gradient is finite and present, worst gradient
  cosine is at least `0.9999924`, worst relative-L2 is at most `0.00414`, and
  actual route evidence is `native-nvidia-train-temp-autograd-v2`.  Measured
  speedups are `1.03x` and `1.25x`; these are correctness-smoke timings, not the
  final performance matrix.
- The scalar metric validator now evaluates reductions in FP64 so large BF16
  gradient tensors cannot report a false zero-norm/cosine failure.  A direct
  CPU unit test also proves that the accepted recurrent backward replay matches
  the clean recurrence for every input gradient.  The complete local suite
  passes **179 tests** with 145 expected TorchScript deprecation warnings, both
  migration manifests verify all 102 destination hashes, and `git diff
  --check` passes.
- Remaining RTX 4080 gate: rebuild a fresh immutable wheel from this exact
  source and run the formal B=`1/4`, T=`16/128`, checkpointing on/off training
  matrix.  Only after that passes may the queued quantization, FLA, complete
  backend-v2 and formal lm_eval stages advance.

### 2026-08-28 — RTX 4080 formal native-training matrix passes 8/8

- Re-synced the accepted source to the idle RTX 4080, deleted the diagnostic
  extension cache, and rebuilt every train-temp extension in the fresh
  `/home/wzu/codex-run/torch-extensions/backend-v2-final-decay-fp32-nofma-v4`
  directory.  The recurrent leaf's recorded Ninja command contains
  `--fmad=false`, omits `--use_fast_math`, and produced clamp shared-object
  SHA256 `4530a76e4ec1a4dd6be22bf9a9152fac0f9784780f588a73ab7ddd8eb71e4343`.
- The formal B=`1/4`, T=`16/128`, checkpointing on/off matrix completed **8/8**
  with exit code zero and no failures.  Every case records actual route
  `native-nvidia-train-temp-autograd-v2`; logits and loss are bit-exact in all
  eight cases.  Across the complete matrix, worst gradient cosine is
  `0.9999917` and worst gradient relative-L2 is `0.00414`, both inside the
  unchanged `0.999` / `0.02` gates.
- Native training materially reduces non-checkpointed B4/T128 peak allocation
  from `4.12 GB` to `1.28 GB`.  Steady cases range from `0.98x` to `1.25x` in
  this correctness harness.  The first B1/T16 candidate timing includes the
  one-time clean JIT build and is therefore not a throughput result; final
  forward+backward numbers remain assigned to the dedicated warmed benchmark.
- Evidence:
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-fix-20260828/4080/training-formal-final-src-6677ec04-v30.json`.
  The next step is to commit this accepted source, build an immutable two-wheel
  pair from that commit, then use only that pair for the remaining 4080
  ecosystem, quantization, FLA, benchmark and formal lm_eval gates.

### 2026-08-28 — candidate-wheel inference exposed a reference-boundary regression

- Built candidate wheels from commit `c956fa274e1b238a6c6cca2753ab253b4387463b`:
  HF SHA256 `f0d107de31b08a4438447ce0e29bc3d1eb4ac0408adb49a2d491e72d6f9acea0`
  and kernel SHA256
  `b0dc3bb07be536d09ecb77dc30e4b1601cc458f603d36456bb787c6b20df9b0e`.
  Installation from those wheels and all three staged model-code hashes were
  verified before GPU execution.
- The candidate inference smoke correctly failed.  Native prefill/decode stayed
  finite, cache lifecycle and 16-token greedy/beam generation passed, and the
  reported routes were the intended native implementations, but BF16 logits
  exceeded the existing max-abs gate.  The failure was caused by changing the
  *public* clean recurrence's reduction order to match the private train-temp
  leaf: all existing inference/Graph/prefill kernels and the earlier 144-unit
  reference baseline implement the original readable `state @ (a @ b)` and
  `state @ r` boundary.
- A controlled diagnostic restored only the original clean recurrence while
  keeping the repaired native training leaf.  Dense BF16 native training then
  failed its unchanged gate (`logits max_abs=0.375`, worst gradient cosine
  `0.9871`, worst relative-L2 `0.1649`).  Therefore the apparent 8/8 result
  above is valid evidence for the train-temp leaf's own sequential numerical
  contract, but it is **not** a release acceptance result for the established
  HF reference contract.
- Release decision: restore the original clean/vectorized recurrence and keep
  it as the single public source of truth.  Do not weaken inference or gradient
  thresholds and do not redefine the HF model around one private training
  kernel.  Native train-temp remains explicit diagnostic capability until it
  can match the clean reference; production training continues through the
  already-defined reference fallback.  Rebuild the candidate wheels after the
  restoration and rerun only the affected inference/training gates.
- Preserved evidence:
  `backend-v2-c956fa27-candidate/4080/inference-smoke.json` and
  `backend-v2-c956fa27-candidate/4080/training-diag-old-reference-v31.json`.

### 2026-08-29 — candidate inference numerical localization and gate audit

- Commit `8da42fc14802b0f848c9b59db88b242f7bbd47f4` preserves the clean
  FP32 decay-bias contract in private native inference packs and was pushed to
  `origin/perf/optional-kernels-v1`.  Its RTX 4080 candidate wheels are:
  `rwkv7_hf-1.0.0 =
  210068449dd8626a02e8fb965b121e86c66fd5b434cdfec20bc1599b9bf31df3`
  and `rwkv7_kernels-1.0.0.dev0 =
  655fc228e720b5a16867ea9a10a98e78a47708a0261fbcf7c3539ce31aeb6733`.
- The repair is active: public w0, packed w0 and native decay remain FP32.
  Layer-zero direct comparison shows recurrent output max-abs
  `2.18e-11`, recurrent state bit-exact, and attention output bit-exact.
  Disabling DPLR, self-chunk and optional fused scans did not remove the
  cross-layer drift, so the recurrent rank-one update is not the root cause.
- FP16 candidate smoke keeps every tensor finite and all greedy/beam sequences
  equal.  B1/T17 logits max-abs for 0.1B/0.4B/1.5B is
  `0.046875/0.1484375/0.09375`; teacher-forced decode is
  `0.046875/0.03125/0.0703125`.  The strict legacy report failed only on a
  few `0.15` absolute ceilings for padding or state.  State cosine remains at
  least `0.9999988` in the affected 1.5B cases.
- The wider B=`1/4`, T=`17/128` diagnostic confirms that long-prefill
  FP16-accumulation policy is the largest numerical contributor.  Disabling
  global/block FP16 accumulation reduces 0.4B B1/T128 logits max-abs from
  `0.390625` to `0.109375` and 1.5B B4/T128 from `1.1875` to `0.125`.
  Fixed-row projection experiments and disabling all other prefill fusions do
  not remove the residual difference, which accumulates gradually across
  layers rather than beginning in the recurrent leaf.
- The backend-v2 inference validator had silently diverged from the calibrated
  release contract already documented in `docs/EVALUATION.md`: it applied an
  undocumented BF16 `0.30` max-abs ceiling and applied the FP16 `0.15` logits
  target to recurrent state.  The validator now records two explicit results
  per tensor: the calibrated finite/cosine release gate and the original
  stricter aspirational diagnostic.  Max-abs, mean-abs and tokenwise argmax
  remain visible; greedy and beam equality stay separate mandatory model
  cases.  No failed value is deleted or relabeled.
- The complete local suite passes **181 tests** with 169 expected TorchScript
  deprecation warnings.  Production whole-model `auto` remains disabled.
  Next action: build one new immutable candidate from the gate-audited source,
  rerun the formal FP16/BF16 inference matrix, then run strict native lm_eval
  against the already-exact recurrent/reference lane before considering any
  production promotion.

### 2026-08-29 — gate-audited candidate formal inference and strict-native lm_eval

- Re-synced commit `5c467e2e3857e75880ab126cf34a01ef14a2ddca` through the
  V100 jump host to the idle RTX 4080.  The first wheel-build attempt failed
  only because the remote environment did not contain the optional `build`
  frontend; the preserved source was then built without isolation using pip.
  The resulting candidate pair is immutable for this validation round:
  `rwkv7_hf-1.0.0 =
  fa42443ecec6a02d0bb0542b76438ace0443c0ee6c46f531b1afa30af98ef4b9`
  and `rwkv7_kernels-1.0.0.dev0 =
  18edad2782f9420dcace52080e92f809d7b21d986391c0b2d4e566c6cdae6c37`.
- The FP16 formal inference matrix passed all 0.1B/0.4B/1.5B cases for
  B=`1/4`, T=`17/128`, padding, cached decode and 64-token greedy/beam
  generation.  All tensors are finite, the calibrated state gate passes
  360/360 comparisons, minimum logits cosine is `0.9999710`, and minimum state
  cosine is `0.9999886`.  The report still exposes six long-prefill cases that
  miss the separate aspirational `0.15` max-abs target; none is removed or
  relabeled.  Evidence:
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-5c467e2e-candidate/4080/inference-formal-fp16.json`.
- The BF16 tensor/state/cache checks and the 0.4B/1.5B generation checks pass,
  but 0.1B greedy generation diverges at generated token six and therefore the
  BF16 report remains `failed`.  Stepwise diagnosis shows an exact clean-model
  BF16 tie at that step (tokens 47 and 21265 both have logit `3.765625`), while
  the candidate selects token 21265 with logits `3.828125` versus `3.8125`.
  This is preserved as a genuine near-tie failure; BF16 whole-model native is
  not promoted.  Evidence: `inference-formal-bf16.json` and
  `inference-diag-bf16-01b-greedy-stepwise.txt` in the same result root.
- A strict whole-model native PIQA smoke (0.1B, batch 1/8, 16 samples) completed
  2/2 with exit code zero.  Both manifests record the requested `native` model
  policy and actual `native-nvidia-prefill-v2[dense_fallback]` calls; wheel,
  source, model and dataset provenance are present.  The no-limit 48-unit FP16
  formal matrix was then started as the only RTX 4080 GPU workload (runner PID
  `3996222`) at
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-5c467e2e-candidate/4080/lm-eval-native-formal`.
  It must finish naturally before any FLA, V100 or RTX 4090 work is scheduled.
- The first complete formal model block (0.1B, 16/16 units) has all commands
  exiting zero, finite metrics and the intended native route, but it exposed a
  real batch-stability gate miss.  ARC-Easy `acc_norm` differs by
  `0.0012626263` between batch 1/8 and PIQA `acc` differs by `0.0010881393`,
  both just above the unchanged `0.001` ceiling.  The corresponding clean
  reference results are bit-stable across batch sizes.  Sample-level analysis
  localizes the metric changes to near-tied choices (three ARC-Easy normalized
  decisions and four PIQA accuracy decisions, including exact FP16 score
  ties); the native routes for both batch sizes are
  `native-nvidia-prefill-v2[dense_fallback]`, so this is not a Graph-route
  substitution.  The 48-unit collector remains running to gather the complete
  model/task evidence; these failures are preserved and only the affected
  units may be rerun after a numerical fix.
- The second complete model block (0.4B, another 16/16 units) also has zero
  command, route, finite-value or infrastructure failures.  Its only batch
  gate miss is Winogrande accuracy: batch 1=`0.5777426993`, batch
  8=`0.5793212313`, absolute difference `0.0015785320`.  The clean reference
  is identical across batch sizes (`0.5753749013` for both).  Exactly two of
  1,267 native samples change correctness, and both are FP16 ties at batch 1
  that batch 8 resolves by `0.0078125` or `0.03125`; both traces again use the
  dense native prefill route.  This evidence is retained while the 1.5B block
  continues.
- The collector completed **48/48** units with runner exit zero.  Every task is
  finite and every manifest records a real `native-nvidia-prefill-v2[...]`
  route; no OOM, traceback or infrastructure failure occurred.  The corrected
  formal validator nevertheless exits 1 with six unique batch-stability
  failures: the two 0.1B metrics and one 0.4B metric above, plus 1.5B
  ARC-Challenge `acc` (`0.4812286689` vs `0.4829351536`), PIQA `acc_norm`
  (`0.7823721436` vs `0.7807399347`) and Winogrande `acc`
  (`0.6874506709` vs `0.6890292028`).  The 1.5B misses correspond to only
  two, three and two near-tied samples respectively; the clean reference is
  batch-stable for all three tasks.
- Formal validation initially selected the newer `kernel-route.json` instead
  of lm_eval's aggregate report and then duplicated each failure once per
  batch.  Commit `4159695c` fixes both validator bugs, adds a regression test,
  and preserves the corrected six-failure `validation.json`; the collector
  results themselves were not changed or rerun.
- A controlled equal-length diagnostic reproduces the cause at the model
  boundary: the same 0.1B prompt run as B1 versus an unmasked repeated B8 batch
  differs by logits max-abs `0.125`/mean-abs `0.01350`, whereas a padded batch
  that takes the existing per-row compact path is bit-exact.  The clean HF
  linears deliberately project fixed 128-row blocks, but the native dense
  fallback currently feeds the full variable B×T row count to GEMM.  Next,
  add the same stable row contract at the native projection boundary (without
  changing readable modeling), verify B1/B8 exactness, and rerun only the six
  affected task pairs before any V100 promotion.
- The stable-row diagnostic now also isolates the remaining batch-dependent
  drift to the recurrent scan launch.  Matching the clean model's fixed
  128-row projection contract alone reduces the repeated-prompt difference to
  max-abs `0.09375` but does not eliminate it.  With the same projection
  contract plus row-serialized B>1 recurrent scan, the identical unmasked B1
  and repeated-B8 native prompt is bit-exact (`max_abs=0`, `mean_abs=0`, no
  argmax mismatch).  The native result still has the separately tracked
  clean-reference drift (`max_abs=0.09375`, `mean_abs=0.01199`, cosine
  `0.9999803`, no argmax mismatch).  The default-on correctness boundary and
  unit coverage are implemented locally; the complete local suite passes
  **183 tests**.  The temporary RTX 4080 overlay is diagnostic only and must
  not be cited as wheel evidence.  Next, build a new immutable wheel, repeat
  the direct B1/B8 smoke from that wheel, and rerun only the six affected task
  pairs.
- Commit `2baaf4b46bdd2e26ca2e8ce979d35065d834f55b` makes that
  batch-invariant projection/scan boundary immutable.  The new RTX 4080
  candidate wheels are `rwkv7_hf-1.0.0 =
  93b5de31e27b8b1777ab6ad782e40441e02f493dd116bd3bae98be36117bb071`
  and `rwkv7_kernels-1.0.0.dev0 =
  affbd3c1b8f769eabd0361448c7f52ddf5f05b8597c64f4048f168b5b9e130ec`.
  A wheel-only install (no source/diagnostic overlay) reproduces the 0.1B
  equal-length B1/repeated-B8 result bit-exactly and records two actual
  `native-nvidia-prefill-v2[dense_fallback]` calls.  Evidence is under
  `backend-v2-2baaf4b4-candidate/4080/batch-invariance-wheel*.json`.
  The six affected metric pairs are now being rerun as exactly 12 formal units
  from the same wheel pair (runner PID `4016444`) at
  `backend-v2-2baaf4b4-candidate/4080/lm-eval-native-affected`; no unaffected
  task is being repeated and no V100/RTX 4090 job may begin before this gate
  finishes.
- The 12-unit RTX 4080 repair shard completed 12/12 with exit code zero.  All
  six B1/B8 task pairs now have exactly equal aggregate metrics, every route is
  an actual `native-nvidia-prefill-v2[...]` implementation, and no NaN, Inf,
  OOM or traceback was found.  A transparent composite replaces only those 12
  units in the original 48-unit run: 36 units retain source `5c467e2e` and 12
  repaired units use `2baaf4b4`; `validation.json` reports 48 units, `passed`,
  with no failures.  The compact evidence and verified manifest are committed
  under `results/kernel-migration/backend-v2-4080-2baaf4b4/`; samples and large
  logs remain outside Git.
- The same two immutable candidate wheels were copied byte-for-byte to the V100
  host.  Its wheel-only 0.1B equal-length B1/repeated-B8 smoke is bit-exact and
  records two actual `native-nvidia-prefill-v2[cuda_graph_prefill]` calls.  The
  authorized temporary `qwen38-27b` Docker service was stopped to release both
  V100s.  A full 48-unit strict-native lm_eval run is active on GPU 1 (PID
  `3890548`).  The first GPU-0 inference invocation exposed only an evaluator
  compatibility bug: Transformers 4.52 treats the newer `dtype=` load keyword
  as an unserializable config field.  Evaluation loaders now use the
  backward-compatible `torch_dtype=` spelling; the immutable model/kernel
  wheel bytes are unchanged.  Only that failed inference gate is being rerun
  on GPU 0 as `inference-v2` (PID `3900773`); the original failed log remains
  preserved.
- The V100 `inference-v2` rerun completed with exit zero from the unchanged
  candidate wheels.  All three models (0.1B/0.4B/1.5B) pass the release
  precision/state/cache/greedy gates and record only real native prefill,
  masked-prefill and fused-decode routes; the separately published strict
  low-precision max-abs targets remain diagnostic.  The first V100 ecosystem
  invocation is preserved as a failed infrastructure/evaluator run: two
  remaining `dtype=` loader spellings were incompatible with Transformers
  4.52, the shared environment exposed a broken DeepSpeed/CUDA_HOME import,
  and its TRL 1.7/Transformers 4.52 pair was incompatible.  The evaluator now
  uses backward-compatible `torch_dtype=` at every load boundary and accepts
  the nested base-model reference route produced after PEFT's adapter-aware
  causal-LM fallback; functional adapter parameter-change and save/reload
  checks remain mandatory.  The complete local suite still passes **183
  tests**.  Only the affected ecosystem gate is running again as
  `ecosystem-v2` (PID `4010758`) on GPU 0 in the existing isolated
  Transformers 4.56/TRL 0.20 environment, using a fresh wheel-only target with
  the exact candidate SHA256 pair.  GPU 1 continues the untouched 48-unit
  strict-native lm_eval run (PID `3890548`); both jobs are progressing without
  NaN/Inf, OOM or traceback and neither may be restarted or duplicated.
- The isolated V100 ecosystem rerun was refined without touching either wheel.
  `ecosystem-v2` preserved the original FP16-parameter/GradScaler failure;
  `ecosystem-v3` proved AutoModel/save/reload/generation, Trainer and PEFT but
  exposed skipped first-step updates in raw Accelerate and TRL.  The final
  `ecosystem-v4` uses the standard HF mixed-precision contract (FP32 master
  parameters plus FP16 autocast/scaling), a deterministic low initial scale
  for raw Accelerate and an eight-step TRL scaler warmup.  It exits zero and
  passes AutoModel, native prefill/decode route trace, Accelerate, Trainer,
  PEFT adapter save/reload and TRL SFT; every training route is explicitly the
  SM70 PyTorch reference fallback and is not claimed as native training.
- A dedicated reproducible SM70 FLA gate now records the only supported pinned
  FLA comparison instead of pretending its chunk/backward path works on V100.
  `/v100/fla-sm70-fused` exits zero for B=1/4 and T=1/17/128, verifies pinned
  FLA commit `80e494f6c588e091fc8316b612870df29375c5b8`, the exact candidate wheel
  hashes, finite output/state cosine gates and actual optimized routes.  The
  one-token Triton route is 1.158x (B1) and 1.128x (B4) the FLA fused-recurrent
  throughput by the report's `FLA median / optimized median` ratio; the
  multi-token CUDA-Graph reference recurrence is intentionally slower than
  FLA fused recurrence and is not advertised as the whole-model native
  prefill benchmark.  The report explicitly records that pinned FLA chunk
  lowering and fused-recurrent backward are unavailable on SM70.  Local tests
  now pass **185 tests**.  GPU 1 continues the untouched 48-unit strict-native
  lm_eval run; 13 unit results are complete with zero command/error failures.
- V100 GPU 0 is now running the canonical same-wheel finetune chain (orchestrator
  PID `4099419`) after ecosystem and FLA completed: 100-step SFT, DPO and GRPO,
  followed by SFT checkpoint resume, W&B offline smoke and strict artifact
  validation.  Datasets are the already cached pinned revisions; the model and
  both wheel SHA256s are recorded in every run. AutoModel remote-code modules
  own a separate evidence namespace, so the callback resolves the accessor
  from the actual model class before falling back to the installed-package
  accessor. The nested causal-LM reference route is accepted only alongside
  the existing nonzero-gradient, changed-parameter and adapter-save/reload
  proof.  Local coverage for this namespace case raises the suite to **186
  tests**.  SFT has entered the 100-step loop normally on GPU 0 while the lm_eval
  collector continues independently on GPU 1; neither task is duplicated or
  restarted.
- The V100 canonical SFT stage completed at 100/100 with exit zero.  It records
  finite loss, nonzero gradients, 144 changed LoRA tensors, explicit SM70
  reference fallback (never native adapter bypass), exact adapter save/reload
  logits (`max_abs=0`), four checkpoints, pinned dataset fingerprints, model
  revision and the exact candidate wheel hashes.  DPO is now the only GPU-0
  finetune workload and is progressing normally (10/100 at inspection).  The
  independent strict-native lm_eval collector is at 22/48 on GPU 1: every
  completed command exits zero and the aggregated actual route count contains
  only `native-nvidia-prefill-v2[cuda_graph_prefill]`.  RTX 4080 remains idle;
  its repaired composite still validates 48/48 with zero failures and its
  wheel-only B1/B8 smoke remains bit-exact.  The three pending local commits
  were successfully pushed to `origin/perf/optional-kernels-v1` at
  `cd1e64d75e38f9f6960fe5a08c7eb163f701ade4`.

### 2026-08-29 — V100 removed from the release gate

- By explicit user decision, V100 is no longer required.  The active V100
  strict-native lm_eval and canonical finetune process groups were terminated
  without deleting their outputs.  At termination, lm_eval had completed
  33/48 units with zero command failures and DPO had reached 73/100; these are
  retained as partial historical evidence and must not be presented as a full
  V100 acceptance bundle.
- RTX 4080 remains the release-validation device.  Its repaired strict-native
  FP16 lm_eval composite passes 48/48 with exact B1/B8 aggregate metrics and
  real `native-nvidia-prefill-v2[...]` routes.  This does not yet close every
  release item: the 0.1B BF16 greedy near-tie remains failed, the complete
  three-lane reference/optimized/FLA 144-unit equivalence gate is unchecked,
  native backend-v2 training parity is unchecked, and the kernel artifact is
  still a `1.0.0.dev0` candidate rather than the final stable wheel.

### 2026-08-29 — native-vs-FLA evidence audit

- The RTX 4080 is idle and no formal watcher was active.  The previous
  three-way bundle contains 48/48 successful commands in each lane, but its
  nominal optimized lane actually records
  `torch-cuda-graph-reference-v1`; it is not evidence for backend-v2 native.
- A read-only diagnostic compared the repaired native 48-unit composite with
  the existing pinned-FLA 48-unit outputs.  All commands exited zero and the
  maximum aggregate classification-accuracy difference is
  `0.0016322089`, but only 60/156 classification metrics are bit-exact and 41
  units have at least one selected-outcome mismatch.  The diagnostic is
  explicitly marked non-release because native is the transparent 36+12
  composite while FLA was produced by the earlier harness.  Evidence:
  `/home/wzu/codex-run/results/rwkv7-kernels-v1/backend-v2-2baaf4b4-candidate/4080/native-vs-fla-existing-diagnostic.json`
  (`sha256=09cf19fe57e8963d46a32f8c386c01c6ecb65c71c2691de57b3c748384eacfd8`).
- The audit exposed a harness provenance bug: `dataset_fingerprint` included
  `lm_eval`'s runtime/model-path metadata, so the same cached documents looked
  different across native and FLA even though their sample-hash fingerprints
  were identical.  Dataset fingerprints now exclude only this runtime
  metadata while preserving the full task config for audit.  The complete
  local suite passes **187 tests**.  The next formal comparison must regenerate
  all three manifests with this corrected harness and the same final wheel;
  old Graph or mixed-SHA results must not be promoted.

### 2026-08-29 — canonical leaf-level CUDA training boundary

- Added an additive recurrent-training adapter without changing the readable
  `modeling_rwkv7.py` layer structure. Its implementation route is
  `native-nvidia-rwkv7-recurrent-training-v1` and its implementation module is
  `recurrent/training_cuda.py`; the adapter is now private behind API v4's
  `recurrent` operation. FLA remains an evaluation dependency only and no
  project or contributor nickname appears in the runtime API.
- `RWKV7_TRAINING_KERNEL_IMPL=auto|cuda` is independent from inference policy.
  Production `auto` stays on reference autograd until the full-gradient release
  gate passes; explicit `cuda` is the only way to compile and exercise the
  candidate.  Unsupported dtype, device, mask, state or sequence shape fails
  closed through the normal HF fallback boundary.
- The migrated CUDA recurrence now returns the canonical final state together
  with the output.  Standard detached-state training uses the native CUDA
  forward and backward.  If a caller consumes the returned state or requests
  the zero initial-state gradient, custom autograd adds only those state
  contributions through the canonical recurrence.  The initial candidate is
  intentionally restricted to dense, zero-state BF16, T divisible by 16, head
  size 64, and sm80+.
- Runtime comments describe only tensor layout, dtype ownership, autograd and
  fallback contracts.  Historical `train_temp` naming remains private to the
  vendored source adapter.  Migration hashes, capability inventory and wheel
  audit requirements were updated.  Ruff passes for every modified first-party
  file, `git diff --check` passes, and the complete local suite passes **191
  tests**.
- Next gate: build a fresh candidate on RTX 4080 and compare reference, pinned
  FLA and the explicit CUDA training route with
  `evaluation/validate_recurrent_training.py`.  The gate covers output, final
  state, all six vector gradients at every shape, the initial-state gradient at
  the shortest sequence for each batch, warmed ordinary-training
  forward+backward time, peak memory and actual route.  Auto must remain
  reference if any numerical gate fails.

### 2026-08-29 — recurrent training acceptance and clean linear leaf

- The first explicit recurrent candidate (`a79d6ff468b2`) passed the RTX 4080
  leaf matrix for B=`1/4`, T=`16/128`, H=`2`: output, final canonical state,
  all six recurrent-vector gradients and the selected initial-state gradients
  passed.  Its native forward+backward route was `1.53x`–`2.11x` the pinned
  FLA recurrent route.  The actual 0.1B model shape B4/T128/H12 also passed all
  recurrent gradients and measured `0.7184 ms` native versus `1.1776 ms` FLA
  (`1.639x`).  Evidence remains outside Git under
  `/home/wzu/codex-run/results/rwkv7-training-a79d6ff468b2/`.
- Full-model BF16 training with the readable HF layer loop and only the CUDA
  recurrent leaf passed 8/8 B=`1/4`, T=`16/128`, checkpointing off/on under
  the declared optimizer-update-vector gate.  CUDA was numerically closer to
  reference than FLA in every case: minimum logits cosine `0.9999625`, maximum
  loss delta `0.00708`, minimum global-gradient cosine `0.9997274`, and maximum
  global-gradient relative L2 `0.02353`.  All 399 named gradient rows remain in
  the report.  The separate strict per-parameter diagnostic is intentionally
  retained as failed for both CUDA and FLA; it is not hidden or relabeled.
- The full-model timing localized the remaining B4/T128 slowdown outside the
  recurrence: CUDA recurrence is faster than FLA, but the clean reference
  model splits every projection into fixed 128-row GEMMs.  On RTX 4080 a
  flattened PyTorch/cuBLAS projection was `2.17x`–`4.46x` faster for the model's
  representative matrix shapes.  This justifies a second **stateless leaf**;
  it does not justify copying a fused model or parameter ownership into the
  kernel package.
- Added the flattened-linear implementation adapter and route
  `native-nvidia-cublas-linear-training-v1`; it is now private behind API v4's
  `linear_training` operation. Route tracing remains at the existing
  `RWKV7_TRAINING_KERNEL_IMPL=auto|cuda` policy.  `RWKV7Linear` remains the
  visible checkpoint-compatible layer; the optional leaf only flattens
  `[B,T,C]` and calls PyTorch `F.linear`, so PyTorch/PEFT still owns autograd,
  adapters, parameters and optimizer state.  Production `auto` remains
  reference until the new whole-model gate passes.
- Recurrent training now accepts arbitrary sequence lengths and left/right or
  unequal padding.  Each sample is compacted in token order, padded only with
  no-op recurrent updates to the CUDA chunk size, and scattered back.  An
  all-padding sample preserves explicit zero gradients for every public
  recurrent operand rather than returning `None` gradients.
- Finetune provenance now records the model plus recurrent, linear, and Mix6
  tensor leaves. A high-performance training claim requires the readable model
  route and every optional leaf expected for that certified program. Runtime
  names and comments use only mathematical RWKV7 terminology; FLA is confined
  to evaluator and documentation text.
- Local quality gate passes: Ruff on every changed first-party file,
  `git diff --check`, and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` -> **197 passed** with
  253 expected TorchScript deprecation warnings.  Candidate source identity is
  `7f5d0b1f679a`; locally built immutable diagnostic wheels are
  `rwkv7_hf-1.0.0` SHA256
  `01cdaaf4412e269bf91564d41d0e50c38e860b6cf7d08c21b901b486a65682d1`
  and `rwkv7_kernels-1.0.0.dev0` SHA256
  `0db18a8ac77226433ed42697077d35bb6ca95ce26cd20401517fedcafe3384f0`.
- RTX 4080 was confirmed idle before deployment.  The V100 jump route then
  became unreachable on both ports 9022 and 9023, so the new candidate has not
  yet produced GPU evidence.  This is a transport blocker only: no local GPU
  fallback was used and no old wheel result is being attributed to the new
  source.  Once connectivity returns, run masked recurrence first, then the
  three-way full-model matrix and decide whether the linear leaf can remain in
  explicit CUDA mode without loosening the published numerical thresholds.

### 2026-08-29 — standardized training API and exact wheel candidate

- Public training terminology is now limited to RWKV7 mathematical operations.
  The recurrent, linear, and Mix6 routes are documented as operation kinds
  under the single API-v4 facade; the pinned FLA name appears only in
  evaluator/documentation comparison text. The readable model remains the
  only owner of modules, parameters, adapters, cache, loss, and checkpoints.
- `docs/EVALUATION.md` now contains the exact leaf and full-model three-way
  commands. Full-model acceptance requires the readable
  `torch-reference-model-v1` loop plus the recurrent, linear, and Mix6 routes
  expected for the certified program; requested environment variables are not
  accepted as evidence.
- Re-ran the complete local quality gate after the masked evaluator and
  documentation changes: Ruff, both evaluator syntax checks and
  `git diff --check` pass; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
  -q` remains **197 passed** with 253 expected TorchScript deprecation
  warnings.
- Built a fresh exact candidate from content identity `ac4437cc3c2a` and
  verified every changed runtime member in each wheel is byte-identical to its
  source.  `rwkv7_hf-1.0.0` SHA256 is
  `94abbadc244953906ad46f6863181aba6521b3d8d0bdb11ef5cefa61efc6ef91`;
  `rwkv7_kernels-1.0.0.dev0` SHA256 is
  `49bd9c5c487b6541f94d1d3407e524052175beeeb4e710f5c5021a6b0f52126f`.
  A clean local wheel-only environment imports API v2, exposes all four
  versioned training functions and confirms CPU/`auto` fails closed to
  `torch-reference-linear-v1` with ordinary gradients.  The compact source
  archive SHA256 is
  `0bb3ede722737701624f5edcd73b040eb164c71cbdd3e15b7cd97c92064eb49a`.
- RTX 4080 deployment is still blocked by transport: both V100 jump ports,
  direct 4080 Tailscale and the independent RTX 4090 endpoint timed out in the
  same inspection window.  No GPU result is attributed to this candidate and
  no old formal job was restarted.  Once either jump route returns, transfer
  the exact archive and wheels above, confirm the GPU is idle, then run the
  masked recurrent smoke before the full-model matrix.

### 2026-08-29 — RTX 4080 masked-training audit and reference-boundary decision

- Restored the RTX 4080 route through the V100 jump host, confirmed the GPU was
  idle, and deployed the exact `03940da37c89` source candidate.  Wheel SHA256
  values are `0590975d1d12464d4478e76cf1c7f9ea4d168f4564c19a2b15642a48f7839fc2`
  for `rwkv7_hf-1.0.0` and
  `c606eec3139f68ee507408ee8ee6eb031f249bb7dd1efe083d0d82373e94aba4`
  for `rwkv7_kernels-1.0.0.dev0`.  All seven HF runtime files and 128 kernel
  runtime/source files were byte-compared against the built wheels before
  deployment.
- Recurrent mask compaction now packs all samples into one no-op-padded CUDA
  batch rather than launching the leaf once per sample.  The B4/T17/H12
  left/right leaf audit passed output, final state, all vector gradients,
  route and auto-fallback gates.  Native forward+backward measured `2.635x`
  and `2.625x` the pinned FLA recurrent lane (`17.15x` and `17.07x` the
  readable recurrence).  Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-03940da37c89/4080/recurrent-masked-formal.json`.
- The full-model FLA comparison no longer passes a mask that the pinned FLA
  RWKV-7 model ignores.  Its evaluator-only lane now compacts each sample in
  token order and scatters logits back before applying the shared causal loss.
  Runtime packages still do not import FLA.  This repaired the misleading
  masked FLA logits cosine from roughly `0.85`–`0.89` to at least `0.99992` and
  makes its slower per-sample masked timing explicit as
  `per-sample-compact-scatter` in JSON.
- The exact-wheel masked model matrix passed 3/4 cases.  Both ordinary
  left/right runs and the right-padding checkpointed run passed actual route,
  logits, loss and complete optimizer-vector gates.  Native was `2.24x` and
  `2.20x` faster than the semantically correct masked FLA lane.  The
  left-padding checkpointed seed remained failed without threshold changes:
  logits cosine `0.999961`, loss delta `0.00866`, gradient cosine `0.999511`,
  and gradient relative L2 `0.03135` versus the fixed `0.025` ceiling.
  Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-03940da37c89/4080/model-masked-formal.json`.
- A four-token FP32 checkpoint experiment reduced that gradient relative L2
  only from `0.03135` to `0.03130`, so the extra recurrent-state memory was
  rejected and the source was restored.  A separate diagnostic using the
  factorized CUDA recurrence as the *public* reference passed (`0.01856`
  gradient relative L2), confirming that the residual is the already-known
  BF16 association difference between the established readable
  `state @ (a @ b)` contract and the historical factorized CUDA leaf.  That
  diagnostic is not accepted: the earlier candidate-wheel audit already
  proved that redefining the public recurrence regresses the established
  inference boundary and 144-unit baseline.
- Decision: keep the clean HF recurrence unchanged, keep production `auto` on
  reference autograd, and do not promote the recurrent CUDA training leaf yet.
  The stateless flattened cuBLAS linear leaf remains mathematically exact and
  independently useful, but it is not sufficient by itself to define a useful
  whole-model training backend.
- A follow-up linear-only diagnostic used the reference recurrence with the
  native linear leaf on B4/T128.  It passed both ordinary cases and the
  unpadded checkpoint case, but the left-padded checkpoint case remained over
  the fixed full-gradient ceiling (`0.025242` relative L2 versus `0.025`).  Its
  ordinary end-to-end speed was only `0.992x` without padding and `1.012x` with
  left padding because the readable recurrence still dominates.  The
  temporary public `linear` policy was therefore removed rather than adding a
  slow near-duplicate route.  Diagnostic evidence is retained at
  `/home/wzu/codex-run/results/rwkv7-training-a75becf53f99/4080/model-linear-only-smoke.json`.
- The next admissible implementation step is an exact matrix-order CUDA
  recurrence matching the unchanged public `state @ (a @ b)` mixed-precision
  contract.  Until that exists and passes the unchanged model gate, both CUDA
  leaves remain explicit diagnostics and no failed result may be relabelled as
  a release pass.

### 2026-08-29 — exact batched-matrix training source diagnostic

- Added `recurrent/training_matrix.py` as a separate, exact training leaf.  It
  preserves the public mixed-precision program: model-dtype outer products,
  FP32 canonical `[B,H,K,V]` state, and the unchanged `state @ (a @ b)`
  multiplication order.  It batches independent samples and heads into each
  PyTorch CUDA matrix multiplication and leaves ordinary autograd, parameters,
  adapters, cache and optimizer state under PyTorch/HF ownership.
- Standardized the explicit selector as
  `RWKV7_TRAINING_KERNEL_IMPL=matrix`; its final actual route name is
  `torch-cuda-rwkv7-batched-matrix-recurrent-training-v1`.  The earlier
  factorized custom-CUDA diagnostic is now named
  `native-nvidia-rwkv7-factorized-recurrent-training-v1`, and its optional
  flattened projection is
  `torch-cuda-rwkv7-flattened-linear-training-v1`.  Production `auto` remains
  reference until immutable-wheel acceptance completes.
- An RTX 4080 source diagnostic passed ordinary and checkpointed B4/T17 left
  padding.  Candidate/reference logits and loss were exact; the complete
  gradient cosine/relative-L2 pairs were `0.9999906/0.004358` and
  `0.9999883/0.004843`.  The ordinary run was `3.006x` faster than the readable
  reference model and `1.097x` faster than the semantically matched pinned-FLA
  lane.  FLA gradient relative L2 was `0.04127` ordinary and `0.11760` with
  checkpointing.  Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-matrix-3f092defb25c/4080/model-matrix-masked-smoke.json`
  (`sha256=d2c81222b25ae16264b0f9978cf7b651ae19776cf51be9a6d3d5d6aae9ed8219`).
- That file is intentionally **source diagnostic only**: it records the
  pre-standardization route string
  `torch-cuda-rwkv7-batched-matrix-training-v1`.  It cannot be cited as final
  wheel evidence.  The next gate is a clean dev-wheel build followed by the
  complete recurrent and full-model matrix on RTX 4080, with the standardized
  actual route verified from JSON before any `auto` promotion.
- Local gate after adding the exact candidate and wheel membership audit:
  targeted Ruff, evaluator bytecode compilation, JSON parsing and
  `git diff --check` pass; the complete suite is **200 passed** with 253
  expected TorchScript deprecation warnings.

### 2026-08-29 — exact matrix immutable-wheel acceptance and speed limit

- Built and installed the exact dev-wheel pair identified by content
  `625fc202ba7f`.  HF wheel SHA256 is
  `98f9da614a87a84c95031e66a7b74ea962d2d84ffa33653a9e761a1afccea8e6`;
  kernel wheel SHA256 is
  `948030b439d80d9d8ad34281f6de0d5136e67caf4ef2b13ce98d63374b1c0828`.
  The clean RTX 4080 environment imported both packages from `site-packages`
  and reported the standardized actual matrix route from the installed kernel
  wheel.
- Recurrent formal passed **18/18** B=`1/4`, T=`16/17/128`, H=`12`, padding
  `none/left/right` cases plus the production-auto reference fallback.  Output
  and final-state max-abs were exactly zero versus the readable reference;
  minimum named-gradient cosine was `0.9999999973`, maximum relative L2 was
  `7.37e-05`, and maximum gradient max-abs was `3.73e-09`.  JSON:
  `/home/wzu/codex-run/results/rwkv7-training-matrix-625fc202ba7f/4080/recurrent-matrix-formal.json`
  (`sha256=2ef263db37855d000d2ba4f96804c965a7c1ca969e6728af61905438a6f0c4a3`,
  exit `0`).
- Full 0.1B readable-model formal passed **36/36** B=`1/4`,
  T=`16/17/128`, padding `none/left/right`, checkpointing `off/on` cases.
  Candidate/reference logits and causal loss were exact in every case;
  minimum complete-gradient cosine was `0.9999467` and maximum relative L2 was
  `0.012453`, below the unchanged `0.025` gate.  The actual routes were exact
  matrix recurrent, reference linear and readable reference-model loop.  JSON:
  `/home/wzu/codex-run/results/rwkv7-training-matrix-625fc202ba7f/4080/model-matrix-formal.json`
  (`sha256=4fb0fcea01de842a8f0444e82e22095121e2d2d5cc43b9cf303e44b9e607ea87`,
  exit `0`).
- The exact route is a successful compatibility accelerator, not yet the final
  dense-training performance route.  Recurrent speedup versus the readable
  reference was median `2.809x` and up to `4.297x` at B4, but only
  `0.0197x`–`0.6409x` versus pinned FLA.  Whole-model speedup versus reference
  was median `1.981x` and up to `3.871x`; versus FLA it was median `0.304x`,
  with only short B4 padded cases reaching `1.073x`–`1.131x`.  These slower
  rows remain first-class evidence and must not be described as an FLA speed
  win.
- Decision: do not promote matrix to production `auto`.  The next performance
  candidate must keep this exact route as the safe masked/stateful fallback
  while using a separately reported fused/factorized route only where its
  unchanged full-gradient gate passes, or replace it with a new exact fused
  implementation.  No threshold or public recurrence change is permitted.

### 2026-08-29 — adaptive training route and RTX 4080 dense rerun

- Standardized the public selector to `auto|adaptive|matrix|factorized` and
  removed the temporary compatibility names. `adaptive` selects
  `native-nvidia-rwkv7-factorized-recurrent-training-v1` only for fully active
  zero-state BF16 CUDA batches; masked or unsupported requests use the exact
  `torch-cuda-rwkv7-batched-matrix-recurrent-training-v1` leaf. The flattened
  linear leaf is selected only for fully active projections with at least 128
  rows. The readable HF model remains the sole owner of layers, parameters,
  loss, cache, padding semantics and gradient checkpointing.
- Renamed the first-party leaf modules to
  `recurrent/training_factorized.py` and `linear/training_flattened.py` so file,
  protocol and route names describe the actual mathematics. The upstream
  `nvidia/official_training_cuda.py` name remains private because it identifies the
  pinned vendored implementation. Comments now distinguish the explicit
  factorized/adaptive request from production `auto`.
- Built and deployed immutable candidate `08c2bf82a012`. HF wheel SHA256 is
  `59af7bc10aa5e8d261ca2669685f919734ce2fb54dfcbac207db9a78621593d0`;
  kernel wheel SHA256 is
  `00793e20eaae99e9df82537acb0b80799607cdd382d479d421bae1e8b01fd66b`.
  The complete local gate is **202 passed**, with targeted Ruff and
  `git diff --check` clean.
- The first recurrent formal retained 12/12 exact masked cases but failed its
  six dense route assertions because the clean RTX 4080 environment had no
  discoverable CUDA toolkit or Ninja, so `adaptive` correctly failed closed to
  the matrix leaf. This is preserved as failed infrastructure evidence at
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-08c2bf82a012/4080/recurrent-adaptive-formal.json`;
  it is not reported as factorized-kernel evidence.
- Assembled an isolated CUDA 13.0 development prefix from the server's cached
  NVIDIA packages, exposed Ninja, and built only the requested recurrent
  extension. The affected dense rerun passed **6/6** with the actual
  factorized route. Minimum named-gradient cosine was `0.9999947`, maximum
  relative L2 was `0.003265`, and maximum gradient max-abs was `5.96e-08`.
  Recurrent speedup was `6.59x`–`321.66x` versus the readable loop and
  `0.622x`–`2.181x` versus pinned FLA (median `1.517x`). Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-08c2bf82a012/4080/recurrent-adaptive-dense-rerun.json`
  (`sha256=e0b829b3d571130ca4578b83451fcc594315d8d4a5510bf94a12ea245b50cd3f`,
  exit `0`).
- The `08c2bf82a012` full-model run reached the last dense checkpointed shape,
  B4/T128, then preserved a real `torch.utils.checkpoint.CheckpointError`
  instead of generating a partial pass bundle. Its first forward used the
  512-row flattened linear program, while autograd checkpoint replay did not
  receive the outer implicit execution decision and returned to 128-row
  reference chunks. The different saved-tensor metadata made the
  failure deterministic; it was a route-replay bug, not a tolerance issue.
- That intermediate repair republished normalized mask semantics during
  checkpoint replay. The current design replaces the implicit mechanism with
  an explicit immutable `RWKV7ExecutionContext`, so forward and recomputation
  receive the same program decision without passing hardware policy,
  parameters, cache, or optimizer state into `modeling_rwkv7.py`. The kernel
  distribution also declares
  Ninja as a direct runtime dependency; factorized JIT still requires an
  explicit matching `nvcc` toolkit and fails closed when it is absent.
- Local acceptance is now **203 passed** with 253 expected TorchScript
  deprecation warnings. Targeted Ruff and `git diff --check` pass. Rebuilt
  candidate `2916f29a8e8f`: HF wheel SHA256 is
  `e27fd675aed636fdbb5681d987b45e23986674cc4276d8e7157d5bdb937cbcef`;
  kernel wheel SHA256 is
  `45330e966a066100a1a25c5d51ab6d5cb1e24964860621f982aca845091d9348`.
  Runtime members were byte-compared with source, and the kernel metadata
  contains `torch`, `numpy`, `packaging`, and `ninja`. This new pair, not
  `08c2bf82a012`, must pass the complete recurrent/model/finetune gate before
  any production `auto` promotion.

### 2026-08-29 — unaligned adaptive fallback correction

- Candidate `2916f29a8e8f` passed the checkpoint route-replay smoke and the
  complete recurrent matrix. The recurrent report passed **18/18** with six
  dense factorized routes and twelve masked exact-matrix routes. Minimum
  named-gradient cosine was `0.9999947` and maximum relative L2 was
  `0.003265`. Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-2916f29a8e8f/4080/recurrent-adaptive-formal.json`
  (`sha256=00a791465eeae2a78b40be7475f006acc32afd1062b2e5b4006a2575495b6457`,
  exit `0`).
- Its full-model report was intentionally retained as **failed 35/36**. The
  sole failure was the fully active B1/T17 checkpointed case: logits cosine
  `0.9999625`, complete-gradient cosine `0.9997805`, and gradient relative L2
  `0.021196` passed, but causal-loss delta `0.014821` exceeded the unchanged
  `0.01` gate. No tolerance was changed. The factorized implementation is a
  16-token-chunk program, so padding T17 to T32 changed BF16 association even
  though it preserved the recurrent equation.
- Adaptive policy now states that contract directly: only fully active token
  lengths divisible by 16 use the factorized recurrence. Masked, unaligned,
  stateful, or unsupported requests use the exact batched-matrix recurrence.
  The optional flattened linear observes the same fully-active and
  token-alignment context, so a model forward cannot mix an exact recurrent
  fallback with the flattened accumulation program. Explicit `factorized`
  remains an isolation/diagnostic selector; production `auto` remains the
  readable reference path.
- Consolidated the historical mask/batch helper into the current
  `RWKV7ExecutionContext` contract. Mask activity and token alignment are
  resolved once and passed explicitly to every layer and checkpoint replay.
  Unit coverage includes fully active unaligned recurrent and linear requests.
- The corrected source passes the complete local gate: **203 passed**, with
  253 expected TorchScript deprecation warnings; targeted Ruff and
  `git diff --check` also pass. Corrected immutable candidate
  `453e29a29e1a` has HF wheel SHA256
  `7189c8b5615628c19befc6076bea0e80af83fd5b7c522d64ba82e297353a48f9`
  and kernel wheel SHA256
  `c814b13f2e82020c6418f7c5fbad8cab5a1310b586b2a256264679f64c84b278`.
  It must pass the affected T17 smoke, complete recurrent/model matrices, and
  SFT/DPO/GRPO validation before any promotion.
- The affected B1/T17 checkpoint smoke now passes exactly through the intended
  exact fallback: matrix recurrent, reference linear, readable model loop,
  zero logits/loss/full-gradient difference versus reference. Evidence:
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-453e29a29e1a/4080/model-adaptive-t17-smoke.json`
  (`sha256=800e861f56a7cbb08776da73671495c827f6d35af3368dcb2ade6571c17e9f77`,
  exit `0`).
- Corrected recurrent formal passes **18/18**. Actual routing is four aligned
  factorized rows and fourteen masked/unaligned exact-matrix rows; no requested
  policy name is counted as execution evidence. JSON:
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-453e29a29e1a/4080/recurrent-adaptive-formal.json`
  (`sha256=a47ebf365ef441153f0ab4adfbdd0f27ab9b8bfca361c5c35d9cc8312ab9de1a`,
  exit `0`).
- Corrected full-model formal passes **36/36**. Actual recurrent routes are
  eight aligned factorized and twenty-eight exact matrix; actual linear routes
  are four flattened and thirty-two reference. Maximum loss delta is
  `0.007085`, minimum complete-gradient cosine `0.9997274`, and maximum
  complete-gradient relative L2 `0.023528`, all within unchanged gates. The
  four non-checkpointed aligned performance rows are `1.008x`, `1.365x`,
  `1.007x`, and `1.216x` versus pinned FLA; exact fallback rows are retained
  even when slower. JSON:
  `/home/wzu/codex-run/results/rwkv7-training-adaptive-453e29a29e1a/4080/model-adaptive-formal.json`
  (`sha256=1227108af5b805c6a90c6981fdac85197e5b73b84cae296324f27cd20cd3b806`).
  SFT/DPO/GRPO remains pending until the formal process has exited naturally;
  no running validation process is terminated for scheduling convenience.

### 2026-08-29 — BF16 LoRA route-proof smoke on RTX 4080

- The direct TRL examples now load the requested model dtype explicitly,
  disable a second Trainer AMP context, and use non-reentrant gradient
  checkpointing. This keeps LayerNorm/projection tensors in the declared BF16
  contract and lets the optional probe observe the real autograd request.
- Route evidence now combines the last optimizer-boundary evidence snapshot
  with the optional package's process-wide actual-call counter. This is
  necessary for
  DPO because its differentiable policy forward is followed by a no-grad
  reference forward; the latter must not erase the earlier optimized call.
- Harness `27b21bf41b73` ran one-step SFT, DPO and GRPO with the unchanged
  runtime wheel pair `453e29a29e1a`. All three exited `0`, produced finite
  loss and nonzero gradients, changed LoRA parameters, and reloaded adapters
  with `max_abs=0`. SFT and DPO each recorded 24 actual factorized recurrent
  calls plus 315/333 flattened-linear calls. GRPO's sampled length was not
  16-token aligned, so adaptive correctly recorded 24 exact-matrix recurrent
  calls and retained reference linears. No NaN, Inf, traceback, or CUDA error
  occurred. This is route-proof smoke only; the canonical 100-step bundle,
  resume check and W&B-offline check remain pending.

- The first formal resume attempt was retained as failed: PEFT promoted LoRA
  matrices from BF16 to FP32 while restoring `checkpoint-100`, but the reload
  checker forced the fresh-run BF16 adapter dtype. Parameters were numerically
  identical after conversion, yet the two different GEMM dtypes produced a
  `0.625` logits max-abs difference. The checker now discovers the actual LoRA
  dtype from the trained model, recreates that runtime explicitly, and reports
  adapter dtypes as part of the artifact. Only the affected resume unit is to
  be rerun; the three completed formal methods remain immutable.

### 2026-08-29 — canonical RTX 4080 adaptive finetune gate passed

- Canonical BF16 LoRA SFT, DPO and GRPO each completed 100 optimizer steps
  with seed 42, length 512, 1024/128 deterministic train/eval samples, the
  pinned dataset revisions, and the immutable runtime wheel pair
  `453e29a29e1a`. Every method exited `0`, logged 102 metric records, changed
  all 144 trainable adapter parameters, produced finite loss and nonzero
  gradients, and reloaded the saved adapter with logits `max_abs=0`.
- Actual process-wide leaf counts prove adaptive execution rather than a
  requested environment name. SFT recorded 2,184 factorized and 216 exact
  matrix recurrent calls plus 28,665 flattened linears. DPO recorded 480
  factorized and 1,920 exact matrix recurrent calls plus 6,660 flattened
  linears. GRPO recorded 216 factorized and 2,184 exact matrix recurrent calls
  plus 2,835 flattened linears. Masked and unaligned requests retained the
  exact matrix/reference-linear pair; all three kept the readable HF model
  loop because LoRA owns target projections.
- The original failed resume artifact remains at
  `sft-resume-failed-60a5be18edb3`. Only that unit was corrected with harness
  `cf65bfdf7d21`: checkpoint 100 resumed to global step 101, the trained and
  reloaded LoRA matrices were both FP32, and reload logits matched exactly.
  W&B offline also exited `0` with local run id `0b0qvajc`.
- Final validation status is `passed`; JSON SHA256 is
  `c856730cc2e908b6fc23353dcf5387a9b1298844580a29b009cffff4905d61e9`.
  The three canonical methods use harness `60a5be18edb3`; the transparent
  affected-only resume correction uses `cf65bfdf7d21`. Runtime wheel hashes
  remain HF `7189c8b5615628c19befc6076bea0e80af83fd5b7c522d64ba82e297353a48f9`
  and kernels
  `c814b13f2e82020c6418f7c5fbad8cab5a1310b586b2a256264679f64c84b278`.
- The post-validation source gate passes **206 tests** with 253 expected
  TorchScript deprecation warnings. Targeted Ruff, byte-manifest verification,
  Python bytecode compilation, and `git diff --check` also pass. Repository-wide
  Ruff is not used as a release gate because historical vendored and preserved
  native-source files are intentionally outside the project lint scope.

### 2026-08-29 — stable-version RTX 4080 training candidate passed

- Built and audited the stable-version package pair from `4bbe5f9e`:
  `rwkv7-hf==1.0.0` SHA256
  `cddc9f16f16a7fcf62ef607b98b06f26f57406d1de9a55e63bf1cb39a6cc2cd0`
  and `rwkv7-kernels==1.0.0` SHA256
  `617aa2ab4834bd53cf4a9380cb6d5e5d2dd28d1e6fa63d0084a265a38e4aa84c`.
  Both wheels were force-installed without source-checkout imports into the
  canonical Transformers 5.8 RTX 4080 environment.
- The unchanged recurrent report passes **18/18** cases. The affected-only
  full-model rerun passes **36/36** cases over batch 1/4, token lengths
  16/17/128, no/left/right padding, and checkpointing off/on. Its actual route
  counts are eight factorized recurrent rows, twenty-eight exact matrix
  recurrent rows, four flattened linear rows, and thirty-two reference linear
  rows; the readable reference model loop remains active in all 36 rows.
- Full-model numerical extrema remain within the declared gate: maximum loss
  delta `0.007085`, minimum logits cosine `0.9999625`, minimum complete-gradient
  cosine `0.9997274`, and maximum complete-gradient relative L2 `0.023528`.
  The model report is SHA256
  `c84068beef64f3ad2496141782a694a2ca8eb9985d9b4f9941931ae1e2cd8a59`.
- Two earlier affected-only model attempts remain preserved as failures. They
  used a Transformers 4.56 environment and either exhausted compiler/CUDA
  memory or failed `cublasCreate`. The successful rerun changed only the
  environment to the previously accepted Transformers 5.8 stack and restored
  the canonical two measured iterations; the wheel bytes did not change and
  the already-passing recurrent matrix was not rerun.
- Compact evidence is committed under
  `results/kernel-migration/4080-training-stable-4bbe5f9e`. It is explicitly
  labeled a stable-version candidate, not immutable final-release evidence;
  final archives are built only after the RTX 4080 and RTX 4090 gates close.

### 2026-08-29 — native training nested-dispatch correction

- Stable candidate `8397dec0` proved that the reproducible CUDA overlay now
  compiles every train_temp extension, then retained a second strict-runtime
  failure: the native whole-model path called `RWKV7Linear.forward` for its
  internal projections. Small LoRA/projection shapes correctly decline the
  independent flattened-linear leaf, but global strict mode promoted that
  deliberate leaf fallback into a failure of the otherwise supported native
  request. The failed artifact remains at
  `/home/wzu/codex-run/results/rwkv7-v1.0.0-8397dec0/4080/train-temp-diagnostic`.
- Added the private, structurally typed
  `rwkv7_kernels.nvidia.training_math` helpers. The whole-model runtime now
  evaluates TMix projections, low-rank gates, ChannelMix, and the LM head with
  the same fixed-128-row canonical PyTorch contract without importing model
  classes, mutating environment state, monkeypatching forwards, or recursively
  entering an optional leaf dispatcher. Parameters and their autograd edges
  remain owned by the clean HF model.
- Naming and comments describe mathematical ownership rather than a device or
  model duplicate. Focused tests prove byte-exact agreement with the clean
  fixed-row linear/ChannelMix contract and explicitly fail on any nested
  `RWKV7Linear.forward` call. The complete local gate passes **209 tests** with
  289 expected TorchScript deprecation warnings; targeted Ruff and
  `git diff --check` pass. RTX 4080 strict train_temp confirmation is pending a
  rebuilt immutable wheel; the `8397dec0` wheel must not be promoted.

### 2026-08-31 — 1.0 core and plugin boundary frozen

- The canonical HF distribution is now structurally closed: `rwkv7_hf/`
  contains exactly the six Python modules `__init__.py`,
  `configuration_rwkv7.py`, `cache_rwkv7.py`, `ops_rwkv7.py`,
  `modeling_rwkv7.py`, and `tokenization_rwkv7.py`, plus the chat-template
  asset. CLI, conversion, manifest, and smoke utilities remain exclusively in
  the sibling `rwkv7_hf_tools/` package. The duplicate source-checkout
  conversion wrappers under `scripts/` were removed; `rwkv7-hf convert` is the
  single conversion command.
- The only optional-backend import boundary is frozen as
  `rwkv7_kernels.execute_optional_v4`. The shipped
  `KERNEL_PLUGIN_API.json` fixes API version 4, its five operation names, exact
  envelope fields, canonical `[B,H,K,V]` cache layout, and fail-closed policy.
  The HF model never imports an NVIDIA/private implementation module. A new
  backend can therefore be installed or removed without replacing config,
  cache, tokenizer, model classes, checkpoint keys, or HF outputs.
- Kernel source naming is normalized around mathematical ownership. The
  official RWKV-LM training modules are
  `official_training_cuda.py`, `official_training_alignment.py`, and
  `official_training_checkpoint.py`; their vendored sources live under
  `nvidia/csrc/training/rwkv_lm/`. Historical `train_temp` function names inside
  byte-provenance code remain only where they identify the upstream recipe and
  are not import paths or public plugin names. The stable execution identity is
  `native-nvidia-official-training-autograd-v2`.
- `RELEASE_SOURCE_FREEZE.json` records SHA-256 for all 155 executable and
  distribution-input files in both packages. The freeze test rejects any
  addition, removal, or byte change. Altering the 1.0.0 source now requires an
  explicit thaw, version/manifest change, and the full hardware matrix again;
  documentation and validation evidence can continue to be appended without
  silently changing the wheel inputs.
- The complete local suite passes **470/470** with 409 expected TorchScript
  deprecation warnings. The package-tree, migration, byte-identity, plugin
  contract, source-freeze, HF roundtrip, cache, conversion, ecosystem, release,
  and evidence tests all pass. A clean wheel install also passed a tiny cached
  forward under Torch 2.13 and Transformers 5.15.
- A pre-commit reproducibility build produced and audited both universal wheels
  and source distributions. The audit verified 7 canonical HF files, 5 tool
  files, 102 migrated NVIDIA files, all 153 historical-source dispositions,
  16 capability families, API v4, the JSON contract, dependencies, RECORD, and
  license metadata. These timestamped artifacts are build proof only. The
  immutable validation pair will be rebuilt from the freeze commit with a fixed
  `SOURCE_DATE_EPOCH`; its hashes replace every earlier candidate hash.
- Per the user's final gate decision, V100 remains historical and is not
  restarted. The deterministic frozen pair must next pass RTX 4080 and then RTX
  4090, including sharded inference, FLA parity/speed, HF/PEFT/TRL training, and
  the formal three-way 144-unit `lm_eval` matrix before GitHub, PyPI, or the six
  Hub repositories are published.
