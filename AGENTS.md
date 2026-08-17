<!--
project_provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  machine_metadata:
    - docs/reference/provenance.yaml
    - docs/reference/rwkv7_serving_contract.yaml
  rule: Preserve upstream attribution and canonical adapter provenance in derived implementations.
-->

# AGENTS.md

## Mission and scope

This repository delivers the **RWKV-7 Hugging Face / Transformers adapter**.
The public surface includes conversion, loading, generation, recurrent state
cache helpers, PEFT/Trainer/TRL/DeepSpeed training compatibility, quantized
inference, hardware dispatch, and reproducible HF acceptance evidence.

Native vLLM, SGLang, and DFlash runtimes are separate projects. This repository
may document shared operator, checkpoint, state-cache, and quantization
contracts under [`docs/integrations/`](docs/integrations/), but must not add a
serving-engine dependency or report an external integration as implemented.

## Current source of truth

Read current documents in this order:

1. [`HF_STATUS.md`](HF_STATUS.md) — current supported scope.
2. [`HF_TODO.md`](HF_TODO.md) — remaining actionable work.
3. [`BENCHMARK.md`](BENCHMARK.md) — promoted numeric results.
4. [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — completion criteria.
5. [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md) — card coverage.
6. [`bench/CURRENT_ARTIFACTS.json`](bench/CURRENT_ARTIFACTS.json) - retained exact evidence.

Historical plans and milestone prose do not override newer accepted evidence.
Superseded benchmark history is intentionally not retained; use [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md).

## Ordinary-user requests

For installation, first inference, or troubleshooting:

Start with the **smallest safe example** and expand only after it passes. Each
action must end in an **observable PASS gate** before reporting success.

1. Read [`docs/AI_ASSISTED_SETUP.md`](docs/AI_ASSISTED_SETUP.md).
2. Use [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) or
   [`docs/USER_GUIDE_ZH.md`](docs/USER_GUIDE_ZH.md).
3. Run `python examples/check_environment.py` before and after conversion.
4. Do not report success until `python examples/generate.py` exits successfully
   with generated text.

For training, quantization, speculative decoding, Apple, or multi-GPU usage,
route through [`docs/COMPLETE_ADAPTER_GUIDE.md`](docs/COMPLETE_ADAPTER_GUIDE.md)
and its topical tutorial. A public capability is not documented merely because
source code, a test, or a benchmark row exists.

Never request or commit passwords, tokens, private keys, private model URLs, or
machine-specific credentials. Use placeholders such as `/path/to/model`.

## Architecture boundaries

### Stable public and remote-code contract

Keep these interfaces backward compatible:

- `rwkv7_hf.NativeRWKV7Config`
- `rwkv7_hf.NativeRWKV7Model`
- `rwkv7_hf.NativeRWKV7ForCausalLM`
- `rwkv7_hf.NativeRWKV7Cache`
- `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and tokenizer `auto_map`
- converted checkpoint parameter names
- `scripts/adapter_manifest.py` remote-code closure

`rwkv7_hf/native_model.py` and `rwkv7_hf/tokenization_rwkv7.py` are stable
converted-model entry points. Do not move or rename them without compatibility
shims, converter/sync updates, old-model tests, and save/reload validation.

### Model mathematics

- Align with official BlinkDL/RWKV-LM RWKV-7 mathematics.
- Preserve independent residual width `D` and attention width `A=H*N`.
- Recurrent state is not Transformer KV cache.
- Preserve state dtype, update order, clamp/decay, `v_first`, residual, and
  output semantics before optimizing.
- Compare state and long greedy traces, not only one-step logit cosine.

The readable native oracle is `rwkv7_hf/native.py`. The runtime-independent
contract is [`docs/architecture/RWKV7_OPERATOR_SPEC.md`](docs/architecture/RWKV7_OPERATOR_SPEC.md).

### Performance and hardware

- Performance work belongs in fused/native kernels, not wrapper
  micro-optimization.
- Dispatch by capability, dtype, shape, row count, backend, and validated
  policy. Exact GPU names belong only in policy, tests, docs, and evidence.
- Unknown devices must retain a conservative correct fallback.
- Never widen a V100, T4, 4090, 5090, Apple, or other exact-card schedule
  without same-card correctness, memory, and speed evidence.
- Keep card-specific routes isolated so one card cannot regress another.

Current exact-card dispatch additions:

- Cross-card benchmark tables must be ordered by model size, then GPU, then
  B1/B8. Displayed throughput uses zero decimal places at `>=100 tok/s` and
  one decimal place below `100 tok/s`; raw artifacts retain full precision.

- Tesla V100-PCIE-32GB (`sm_70`): FP16 recurrent state is default only for
  `(hidden=1024, layers=24, batch=8)` and
  `(hidden=2048, layers=24, batch=8)`. Other batches, model shapes, and Volta
  product names remain on FP32 state. Prefill tile and decode warp probes from
  2026-08-11 did not clear the promotion gate and must remain unchanged.
- NVIDIA GeForce RTX 3090 (`sm_86`): the 2026-08-12 max-performance
  dense-FP16 B1/B8 P128/P512/P2048 matrix may use only the exact scan tiles
  and scoped full-prefill FP16 GEMM-accumulation shapes allowlisted in
  `kernel_policy.py`. The strict parameter-adjusted prefill gate passes 24/24
  against full-FLA/Triton-conv Qwen3.5 at minimum/median
  `1.227477x/1.467758x`, and all 25 prompt/cache-handoff correctness rows must
  remain green. Other Ampere products, model shapes, batches, and token blocks
  must not inherit these routes.
- NVIDIA GeForce RTX 4090 (`sm_89`): the 2026-08-12 RTX-4080-route transfer
  may use block-scoped FP16 GEMM accumulation only for exact latest-checkpoint
  0.4B/1.5B/2.9B/7.2B B1/B8 P128/P512/P2048 shapes. The final norm and
  vocabulary head retain FP32 accumulation. B8 grouped tensor-core W/A/V BMM is enabled
  only by its existing hidden-size/rank gate for 1024/2048/2560-wide models;
  the generic grouped fallback remains rows<=4 so 7.2B/hidden=4096 is not
  implicitly promoted.  The paired screen passes 108/108 accumulation rows
  and 9/9 BMM rows in `bench/4090_4080_routes_20260812/`; the 7.2B extension
  passes 36/36 A/B rows with gains of about `1.0100x-1.4370x`. Exact
  1.5B/B1/P2048 may additionally use self-chunk tile 16 plus stacked R/K/V;
  three interleaved process runs measure `1.2539x` median versus the prior
  route and its forward/reverse Prompt/cache-handoff gate passes. The promoted
  exact-card best-optimized HF matrix uses official FLA plus Dao-AILab
  causal-conv1d Qwen and passes all 48/48 adjusted Prefill plus 48/48 adjusted
  Decode cells, with minima `1.060506x/1.829468x`, in
  `bench/4090_qwen35_paired_pd_v2_20260815/`. Decode remains `native_graph`
  throughout; only 7.2B/B8/P2048 disables Prefill Graph to fit 24 GiB. Other
  batches, model shapes, RTX 4090 variants, and adjacent Ada cards must not
  inherit these routes without exact-card evidence.
- NVIDIA GeForce RTX 5070 Laptop GPU (`sm_120`): the 2026-08-11 policy may use
  exact 0.4B/1.5B prefill graph+scan, raw recurrent, shape-gated norm/mix, and
  B8 FP16-state routes recorded in
  `bench/5070_max_perf_20260811/`. RTX 5070 desktop, RTX 5070 Ti Laptop, RTX
  5070 SUPER Laptop, unlisted models, and unlisted batches must not inherit
  those defaults.
- NVIDIA GeForce RTX 5090 (`sm_120`): exact dense-FP16 0.4B/1.5B/2.9B/7.2B
  B1/B8 P128/P512/P2048 rows may use the graph, scoped full-prefill FP16 GEMM
  accumulation, and fused boundaries allowlisted in `kernel_policy.py`. The
  latest exact-shape paired evidence and route contract live in
  `bench/5090_qwen35_paired_decode_v1_20260813/`. Dense chunk continuation
  may carry recurrent state inside its exact-shape CUDA graph, but quantized
  continuation must remain eager until it has a matching correctness route.
  Stacked RKV must remain disabled for 7.2B because exact-card rows show lower
  latency and roughly 2.4-3.2 GiB lower peak VRAM without it. The global
  accumulation switch must remain exact-shape, scoped, single-GPU by default,
  and empty for other cards; do not promote it by Blackwell family alone. The
  SM120 compiled dense FFN route is explicit-only through
  `RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=1` and restricted to FP16/B8,
  hidden 1024/2048, and exactly 24 layers. It must compile and warm all layer
  weights before raw CUDA-graph capture and fail without fallback if any
  selected/effective layer or correctness gate is missing; it is not a 5090
  default until full-model long-horizon and paired-matrix evidence passes.

### Quantization

- Distinguish physical packed footprint from end-to-end peak VRAM.
- Distinguish `memory` policy from a production `speed` policy.
- A production speed claim needs lower footprint, aligned logits/tokens, and
  matching-shape prefill/decode no slower than W16 on the exact card.
- Marlin is a packed compute backend, not proof of GPTQ/OBQ calibration.
- Fused FFN ReLU-square applies only to FFN-key and exactly once.

See [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) and
[`docs/quantization/VLLM_QUANTIZATION_PORTING.md`](docs/quantization/VLLM_QUANTIZATION_PORTING.md).

### Training

Keep standard HF signatures and return types compatible with:

- PEFT LoRA;
- Trainer;
- TRL SFT/DPO/GRPO;
- DeepSpeed ZeRO-2/3 and resume;
- gradient checkpointing;
- save/reload and adapter merge.

Changes to training math, optimizer grouping, scheduler, data order,
checkpointing, or CUDA training kernels invalidate the corresponding accepted
training evidence and require rerunning its exact gate.

## Change routing

| Change | Required references and tests |
|---|---|
| Config/weight mapping | converter tests, old/new checkpoint load, save/reload |
| Cache/generation | HF API contract, dynamic batch/reorder, long generate |
| Model math | official/native parity, per-state error, greedy trace |
| Kernel | fallback parity, exact-card benchmark, cross-card isolation |
| Quantization | pack/dequant unit test, full-model quality, footprint and speed |
| Training | forward/backward/update, PEFT/TRL, checkpoint resume as applicable |
| Documentation | Markdown links, document freshness, user entry contracts |
| Remote-code file list | sync/converter manifest and clean-install import tests |

Do not mix unrelated documentation, kernels, training, and hardware results in
one PR.

## Validation ladder

Run the smallest applicable layer first.

### Documentation and metadata

```bash
python tests/test_markdown_links.py
python tests/test_document_freshness.py
python tests/test_repository_docs_layout.py
python tests/test_serving_porting_docs.py
git diff --check
```

### CPU/offline contract

```bash
python tests/test_convert_config.py
python tests/test_batch_convert_manifest.py
python tests/test_sync_hf_adapter_code.py
python tests/test_result_tools.py
```

The clean-install CI entry is:

```bash
RWKV7_CPU_ONLY=1 scripts/run_clean_install_tests.sh smoke
```

### GPU changes

Use the exact model/card command from the related issue or accepted artifact.
At minimum record:

```text
correctness and greedy behavior
prefill and decode separately
batch and prompt/decode lengths
physical footprint and peak VRAM
selected/fallback kernel route
GPU, driver, CUDA/ROCm, PyTorch, Transformers, dtype
```

Do not generalize a smoke test into a production claim.

## Evidence and documentation rules

- Raw dated benchmark artifacts are immutable evidence records.
- A new experiment does not become a default until correctness, shape,
  environment, and reproduction data are accepted.
- Update `BENCHMARK.md` only for promoted numeric conclusions.
- Update `HF_STATUS.md` and `HF_TODO.md` only when accepted status changes.
- Completion is always reported for a named scope; there is no universal
  repository percentage.
- Preserve third-party license and provenance headers in FLA/self-chunk,
  Marlin, train_temp, and other derived/vendored files.

## Repository hygiene

- Do not commit model weights, checkpoints, generated binaries, caches, or
  local benchmark noise.
- Keep public modules importable without optional CUDA/MLX/FLA dependencies.
- Avoid machine-specific absolute paths in committed files.
- Use focused compatibility shims when reorganizing stable modules.
- New documentation belongs in the existing lifecycle directories described by
  [`docs/README.md`](docs/README.md).
- Repository layout and staged refactor rules are documented in
  [`docs/architecture/REPOSITORY_LAYOUT.md`](docs/architecture/REPOSITORY_LAYOUT.md).

## Current development direction

The canonical model is the Native/no-FLA Transformers implementation. Continue
production work along:

```text
correct native graph
-> fused FP16 kernels
-> fused W8/W4 kernels
-> chunked DPLR/WY prefill
-> exact-card policy and acceptance
```

Keep wrapper work limited to compatibility, API behavior, telemetry, and safe
fallbacks. Use current `HF_TODO.md` rather than historical “next” sections when
choosing work.
