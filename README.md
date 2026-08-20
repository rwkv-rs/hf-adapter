# RWKV-7 HF Adapter

[**English**](README.md) | [中文](README_ZH.md)
Native Hugging Face / Transformers support for official RWKV-7 checkpoints.
The repository provides conversion, standard HF loading and generation,
recurrent-state cache helpers, PEFT/Trainer/TRL/DeepSpeed compatibility,
quantized inference, hardware-aware fused backends, and reproducible acceptance
evidence.

The canonical model path is Native/no-FLA. FLA remains an optional developer
reference for dedicated comparison work.

The supported HF ecosystem range is Transformers `>=5.12.1,<6`, PEFT
`>=0.19.1,<1`, and TRL `>=1.7,<2`. CI tests the exact lower edge and the newest
resolver result inside those major-version bounds.

## Five-minute quick start

Normal users do **not** need to clone this repository or convert a `.pth`:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install "rwkv7-hf==0.8.0"
rwkv7-hf-doctor
rwkv7-hf-smoke --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 --device auto --output rwkv7-smoke.json
```

The smoke must end with `RESULT: PASS`. Linux NVIDIA users can optionally run:

```bash
rwkv7-hf-kernels status
rwkv7-hf-kernels recommend
rwkv7-hf-kernels install  # only when one exact compatible wheel is listed
```

The base package never guesses a GPU binary. After optional installation, `auto` selects a compatible prebuilt extension, then JIT where allowed, then a
portable fallback. See [published models](docs/PUBLISHED_MODELS.md), the
[full user guide](docs/USER_GUIDE.md), and [kernel wheels](docs/KERNEL_WHEELS.md).
Clone the repository only for conversion, development, repository tests, or
benchmark reproduction.

- [Published models and direct Hub loading](docs/PUBLISHED_MODELS.md)
- [English step-by-step guide](docs/USER_GUIDE.md)
- [中文零基础逐步指南](docs/USER_GUIDE_ZH.md)
- [Windows and CPU guide](docs/WINDOWS_CPU.md)
- [Complete feature guide](docs/COMPLETE_ADAPTER_GUIDE.md)
- [AI-assisted setup and troubleshooting](docs/AI_ASSISTED_SETUP.md)
- [Prebuilt CUDA kernel wheels and one-command smoke](docs/KERNEL_WHEELS.md)
- [Advanced training, speculative decoding, and multi-GPU](docs/ADVANCED_USAGE.md)
- [Apple MPS, MLX, and CoreML](docs/APPLE_USAGE.md)
- RWKV-7 vs Qwen3.5: [latest Prefill/Decode table](docs/QWEN35_LATEST_P_D_TOKPS_EN.md) and [standalone GPU reproduction tutorial](docs/QWEN35_SPEED_REPRODUCTION.md) ([中文](docs/QWEN35_SPEED_REPRODUCTION_ZH.md))
- [Huawei Ascend NPU / torch-npu](docs/hardware/HUAWEI_ASCEND.md)
- [Biren BR106M / BIRENSUPA](docs/hardware/BIREN_BR106M.md)
- [MetaX C500 / MXMACA](docs/hardware/METAX_C500.md)

## Convert an official checkpoint

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/rwkv7-model.pth \
  --output /path/to/rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 --adapter-layout thin \
  --low-memory
```

Then verify the produced directory:

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
python examples/generate.py \
  --model /path/to/rwkv7-model-hf \
  --prompt "The future of language models is" \
  --max-new-tokens 32
```

See [inference workflows](docs/INFERENCE_WORKFLOWS.md) for batch conversion,
cache continuation, dynamic batching, chunked prefill, `device_map`, and
save/reload.

## Supported adapter surface

| Area | Public path |
|---|---|
| Config/model/tokenizer | `AutoConfig`, `AutoModelForCausalLM`, `AutoTokenizer` |
| Public naming | explicit causal-LM `forward` parameters; `num_heads` / `num_attention_heads` config aliases |
| Generation | greedy, sampling, beam-compatible cache operations, `generate()` |
| Recurrent cache | select, reorder, repeat, reset, offload/restore helpers |
| Training | PEFT LoRA, Trainer, TRL SFT/DPO/GRPO, gradient checkpointing |
| Distributed training | DeepSpeed ZeRO-2/3 and checkpoint resume gates |
| Inference parallelism | `device_map` pipeline-style placement plus Transformers-native dense fp16 `tp_plan="auto"` |
| Quantization | BnB fallback, native MM8/MM4, A8W8, TorchAO, Marlin, MLX |
| Hardware | CUDA/ROCm, Biren BR106M/SUPA, MetaX C500/MXMACA, Huawei Ascend NPU, Moore Threads MUSA, CPU fallback, Apple MPS/MLX/CoreML |
| Serving references | runtime-independent vLLM/SGLang implementation contracts |

The repository does not contain a native vLLM or SGLang runtime. Their model,
operator, state-cache, checkpoint, and quantization implementation references
are under [`docs/integrations/`](docs/integrations/README.md).

The public model surface follows Transformers naming without renaming RWKV
checkpoint or kernel internals. Both native and optional FLA-reference configs
accept `num_heads` or `num_attention_heads`, expose the same value through both
attributes, serialize both fields, and reject conflicting values. The optional
FLA reference wrapper exposes named `forward` parameters for signature
inspection; extra version-specific arguments remain accepted through
`**kwargs`. See the [English](docs/USER_GUIDE.md#public-argument-and-config-names)
or [Chinese](docs/USER_GUIDE_ZH.md#公开参数与配置命名) user guide for the
public compatibility contract.

## Current status

The published **RWKV-7 HF `v0.8.0` release is complete** for its declared,
evidence-backed scope. It retains the accepted v0.6 adapter milestone and adds
the integrated Ascend, Biren, MetaX and MUSA boundaries plus the latest
exact-card NVIDIA performance routes.

Production readiness is scoped to exact models, cards, dtypes, batches, and
shapes. Promoted evidence currently includes V100, T4, RTX 3090/4080/4090/5090,
selected Ampere validation, and bounded Apple M5 paths. Huawei Ascend 910B3
HF/NPUGraph/W8 support is ported from the dedicated validated Ascend repository;
future-main hardware reruns extend that accepted integration. MetaX C500 native eager HF
compatibility is likewise ported from its pinned exact-card evidence repository,
without inheriting NVIDIA Ampere policy from the CUDA-compatible device API.
The Biren BR106M path similarly ports BF16/FP32-state eager compatibility for
all released model sizes; broader performance and quantization profiles are
post-release extensions. Unbounded all-card/all-shape matrices, broader task
quality, and distributed-training breadth are separate expansion projects. HF layer-split
PP and the TP/PP porting contracts are complete for this repository; native
serving-engine executors remain separate projects.

The release additionally includes exact-card B8 decode tuning for V100 and
RTX 4080, grouped RTX 4080 W/A/V projections, the complete adjusted RTX 4080
and RTX 4090 Prefill/Decode matrices, and the latest RTX 3090/5090 Prefill
routes. These are promoted, evidence-backed profiles rather than universal
claims for unmeasured products or shapes.

Representative promoted evidence:
> **Current Qwen3.5 comparison:** use the retained paired matrices in the [latest P/D table](docs/QWEN35_LATEST_P_D_TOKPS_EN.md). RWKV uses Native for performance; FLA remains a compatibility/reference oracle.

| Scope | Evidence |
|---|---|
| RTX 5090 Native vs official/Albatross | [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md) |
| RTX 5090 Qwen3.5 comparison | [`bench/5090_qwen35_paired_decode_v1_20260813/`](bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| RTX 5090 frozen Qwen3.5 reference | [`bench/5090_qwen35_best_optimized_hf_v1_20260813/`](bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md) |
| RTX 5090 Tensor Core W4 | [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| RTX 5090 train_temp alignment | [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md) |
| V100 B1/B8 active-parameter comparison | [`bench/v100_qwen35_paired_pd_v1_20260814/`](bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| V100 production close | [`bench/v100_production_close_20260711/`](bench/v100_production_close_20260711/README.md) |
| RTX 4080 B8 decode tuning | [`bench/4080_b8_projection_bmm_20260809/`](bench/4080_b8_projection_bmm_20260809/README.md) |
| RTX 4080 B8 grouped projections | [`bench/4080_b8_projection_bmm_20260809/`](bench/4080_b8_projection_bmm_20260809/README.md) |
| RTX 4080 7.2B/B8 FP16 state | [`bench/4080_7p2b_fp16_state_20260809/`](bench/4080_7p2b_fp16_state_20260809/README.md) |
| RTX 4090 B8 matrices | [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md) |
| Apple M5 bounded production result | [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |

For exact numbers and caveats use [`BENCHMARK.md`](BENCHMARK.md), not this
landing page.

Completion is reported by **named scope**, not as a single repository-wide
percentage. `v0.8.0` is complete; exact hardware and benchmark claims remain
limited to their promoted profiles.

Canonical project state:

- [Current status](HF_STATUS.md)
- [Post-release expansion projects](HF_TODO.md)
- [Acceptance criteria](docs/ACCEPTANCE.md)
- [Hardware matrix](docs/HARDWARE_MATRIX.md)
- [Benchmark summary](BENCHMARK.md)
- [Raw benchmark inventory](bench/INDEX.md)

## Installation profiles

Normal-use PyPI profiles (`cuda`, `train`, `quant`, `torchao`, `mlx`, and
vendor entrypoint extras) are listed in the [user guide](docs/USER_GUIDE.md).
Vendor extras never guess or install hardware-specific PyTorch wheels.

Editable profiles below are for a source checkout, not the ordinary model
installation path:

```bash
python -m pip install -e .                    # core native HF path
python -m pip install -e ".[cuda]"            # CUDA build helper
python -m pip install -e ".[ascend]"          # install CANN-matched torch-npu first
python -m pip install -e ".[biren]"           # install the matched torch_br stack first
python -m pip install -e ".[metax]"           # install the MXMACA PyTorch stack first
python -m pip install -e ".[train]"           # PEFT/TRL/DeepSpeed
python -m pip install -e ".[quant]"           # bitsandbytes fallback
python -m pip install -e ".[torchao]"         # supported Linux TorchAO path
python -m pip install -e ".[mlx]"             # Apple Silicon MLX
python -m pip install -e ".[fla-reference]"   # optional comparison only
```

Optional backends must not be required merely to import the base package.

## Architecture and repository map

```text
rwkv7_hf/     installable model, runtime, kernels, quantization and backends
examples/     small user-facing entry points
scripts/      conversion, sync, acceptance and specialized runners
tests/        API, unit, integration, policy and artifact verification
docs/         guides, architecture, hardware, validation and history
bench/        benchmark tools and immutable dated evidence
configs/      reproducible training/runtime configurations
```

Important stable files:

```text
rwkv7_hf/native_model.py
rwkv7_hf/tokenization_rwkv7.py
scripts/adapter_manifest.py
scripts/convert_rwkv7_to_hf.py
scripts/sync_hf_adapter_code.py
```

Converted checkpoints depend on these remote-code entry points. Structural
refactors must preserve them through compatibility facades. See
[`docs/architecture/REPOSITORY_LAYOUT.md`](docs/architecture/REPOSITORY_LAYOUT.md)
and the [operator specification](docs/architecture/RWKV7_OPERATOR_SPEC.md).

## Training and quantization

Training tutorials:

- [Training workflows](docs/TRAINING_WORKFLOWS.md)
- [Official train_temp CUDA alignment](docs/TRAIN_TEMP_CUDA.md)
- [Training compatibility/status](docs/TRAINING.md)

Quantization tutorials:

- [How to run W8/W4](docs/QUANTIZATION_USAGE.md)
- [Current quantization status and limits](docs/QUANTIZATION.md)
- [BN/TN tuning contract](docs/performance/BN_TN_TUNING.md)

Quantized weight footprint and runtime peak VRAM are different measurements.
A production speed claim requires matching-shape correctness, lower footprint,
and prefill/decode evidence on the exact card.

## Development and validation

For documentation or metadata changes:

```bash
python tests/test_markdown_links.py
python tests/test_document_freshness.py
python tests/test_repository_docs_layout.py
python tests/test_serving_porting_docs.py
git diff --check
```

For portable contract changes:

```bash
python tests/test_convert_config.py
python tests/test_batch_convert_manifest.py
python tests/test_sync_hf_adapter_code.py
python tests/test_result_tools.py
```

Clean-install smoke:

```bash
RWKV7_CPU_ONLY=1 scripts/run_clean_install_tests.sh smoke
```

GPU/card changes must use the exact-model validation command and benchmark
contract from the related issue or accepted artifact. Record correctness,
prefill, decode, physical footprint, peak VRAM, environment, and selected
kernel route together.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the PR workflow and
[`docs/contributing/APPLE_VALIDATION.md`](docs/contributing/APPLE_VALIDATION.md)
for the specialized Apple evidence command catalog.

## Documentation map

- [`docs/README.md`](docs/README.md) — complete document lifecycle/index
- [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) — one-page project overview
- [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) — cross-platform evidence index
- [`CHANGELOG.md`](CHANGELOG.md) — release and current-main history
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — performance boundaries
- [`docs/BACKENDS.md`](docs/BACKENDS.md) — backend and hardware isolation
- [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) — HF criteria
- [`docs/integrations/README.md`](docs/integrations/README.md) — serving-engine contracts
- [`bench/CURRENT_ARTIFACTS.json`](bench/CURRENT_ARTIFACTS.json) - retained benchmark evidence

## Attribution and license

The project is MIT licensed; see [`LICENSE`](LICENSE). Machine-readable project
and implementation provenance is in [`CITATION.cff`](CITATION.cff),
[`docs/reference/provenance.yaml`](docs/reference/provenance.yaml),
[`CONTRIBUTORS.md`](CONTRIBUTORS.md), and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

RWKV-7 mathematics and official checkpoints originate from BlinkDL/RWKV-LM.
Vendored or derived FLA/self-chunk, Marlin, and train_temp components retain
their own copyright and license notices.
