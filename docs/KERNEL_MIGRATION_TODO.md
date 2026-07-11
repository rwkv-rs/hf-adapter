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
- Full NVIDIA prefill/decode/quant/training migration uses the frozen one-shot
  design in `docs/KERNEL_BACKEND_V2_DESIGN.md`. Its public ABI is fixed before
  implementation; diagnostic stages may identify failures but do not redesign
  the clean model boundary.

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

## Phase 5 — one-shot complete NVIDIA backend-v2

- [x] Freeze the complete model-forward ABI and migration inventory before
      moving implementation code.
- [x] Add the kernel API v2 request/result envelope, explicit diagnostic probe,
      and the single early clean-model hook; production auto stays disabled.
- [ ] Complete every `probe_model_forward_v1` / `model_forward_v1` phase and
      enable production auto only after the unified wheel passes.
- [x] Port fused token, W/A/G/V, projection, FFN, norm, state pool and CUDA
      Graph replay without replacing `modeling_rwkv7.py`.
- [x] Port DPLR/self-chunk/fused prefill and all shape routing behind the same
      model-forward protocol.
- [x] Port SM70, Ada and Blackwell NVIDIA policy families.
- [x] Port W8/W4/A8W8/BnTn/BnB/Marlin/TorchAO implementation adapters.
- [x] Keep canonical cache visible to HF; internal layouts never escape the
      kernel package.

## Phase 6 — backend-v2 training implementation and unified acceptance

- [x] Port the existing versioned forward/backward autograd operators behind
      `model_forward_v1`; do not create a separate model class or cache.
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
  recomputes every one of the 102 migrated NVIDIA payload hashes. The HF-wheel
  audit independently requires the seven canonical model/tokenizer assets and
  rejects the optional kernel package plus the removed compatibility/tooling
  names.
- The existing immutable diagnostic artifacts pass the new audit without a
  rebuild: HF wheel
  `237f4561ce59e3b4bbf385489bf9d0620e1b9f24877bd1990d8f00d9d7c6673c`
  contains 18 members; kernel wheel
  `a36be47896f17ba40fbaf0e78cf486a79b929c5e7ecc3d71ae1a16a619596156`
  contains 125 members, including 102/102 byte-verified migrated files and all
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
  byte-verified historical NVIDIA payloads exactly once into 16 capability
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
- Production whole-model `auto` remains disabled. `migrated` means the
  implementation and route ownership are complete; promotion still waits for
  the immutable-wheel RTX 4080 -> V100 -> RTX 4090 acceptance and 144-unit
  three-way `lm_eval` gate.
- Targeted Ruff, compileall, `git diff --check`, and the complete local suite
  pass: `143 passed`. A disposable development-wheel build was then audited
  from its ZIP contents: 16/16 capability families, 102/102 mapped and
  byte-verified migration files, 23 reachable adapted runtime files and 46
  real policy flags. Its SHA256 is
  `0a7f8f162fde9def8dd31ada789e5ef364eabf68093d771326a8d9775489a3df`;
  it is a local audit artifact, not the final immutable `1.0.0` release wheel.
- The final public Issue renderer and GitHub audit now require the same full
  capability vocabulary: recurrent, dense decode, DPLR/self-chunk prefill,
  CUDA Graph/state pools, SM70/Ada/Blackwell, every W8/W4/A8W8/BN-TN/BnB/
  Marlin/TorchAO route, and training autograd. The GitHub source-tree audit
  also requires both embedded inventories and this migration audit document;
  a release can no longer publish only generic “optimized” wording.
