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

### 2026-08-28 — full backend evidence provenance

- The common GPU evidence helper now hashes every model config, remote-code
  module, tokenizer/vocabulary/template payload and safetensors file into one
  deterministic model revision instead of recording only config and weights.
- Environment reports now include Accelerate, Datasets, PEFT, TRL, W&B,
  BitsAndBytes, TorchAO, lm_eval and both RWKV7 distribution versions in
  addition to Python/Torch/Transformers/Triton/FLA/CUDA/driver/GPU. All
  backend-v2 inference, training, quantization, FLA, benchmark and ecosystem
  reports therefore share the same complete provenance schema.
