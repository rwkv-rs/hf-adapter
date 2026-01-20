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

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .                # portable native backend
# Linux NVIDIA optimized path: python -m pip install -e ".[cuda]"
python examples/check_environment.py
```

Run a converted RWKV-7 model directory:

```bash
python examples/generate.py \
  --model /path/to/rwkv7-model-hf \
  --prompt "User: Hello! Assistant:" \
  --max-new-tokens 64
```

Start with a 0.1B or 0.4B checkpoint when validating a new installation.
Converted model directories use `trust_remote_code=True`; load only repositories
or local model directories you trust.

User guides:

- [English step-by-step guide](docs/USER_GUIDE.md)
- [中文零基础逐步指南](docs/USER_GUIDE_ZH.md)
- [Windows and CPU guide](docs/WINDOWS_CPU.md)
- [Complete feature guide](docs/COMPLETE_ADAPTER_GUIDE.md)
- [AI-assisted setup and troubleshooting](docs/AI_ASSISTED_SETUP.md)
- [Advanced training, speculative decoding, and multi-GPU](docs/ADVANCED_USAGE.md)
- [Apple MPS, MLX, and CoreML](docs/APPLE_USAGE.md)

## Convert an official checkpoint

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/rwkv7-model.pth \
  --output /path/to/rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
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
| Generation | greedy, sampling, beam-compatible cache operations, `generate()` |
| Recurrent cache | select, reorder, repeat, reset, offload/restore helpers |
| Training | PEFT LoRA, Trainer, TRL SFT/DPO/GRPO, gradient checkpointing |
| Distributed training | DeepSpeed ZeRO-2/3 and checkpoint resume gates |
| Inference parallelism | `device_map` pipeline-style placement plus Transformers-native dense fp16 `tp_plan="auto"` |
| Quantization | BnB fallback, native MM8/MM4, A8W8, TorchAO, Marlin, MLX |
| Hardware | CUDA capability policies, CPU fallback, Apple MPS/MLX/CoreML |
| Serving references | runtime-independent vLLM/SGLang implementation contracts |

The repository does not contain a native vLLM or SGLang runtime. Their model,
operator, state-cache, checkpoint, and quantization implementation references
are under [`docs/integrations/`](docs/integrations/README.md).

## Current status

Production readiness is scoped to exact models, cards, dtypes, batches, and
shapes. Promoted evidence currently includes V100, T4, RTX 3090/4080/4090/5090,
selected Ampere validation, and bounded Apple M5 paths. Universal all-card,
all-shape quantized speed, every hardware family, broader task quality, and
distributed-training breadth remain separate acceptance items. HF layer-split
PP and the TP/PP porting contracts are complete for this repository; native
serving-engine executors remain separate projects.

Representative promoted evidence:

| Scope | Evidence |
|---|---|
| RTX 5090 Native vs official/Albatross | [`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md) |
| RTX 5090 Qwen3.5 comparison | [`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md) |
| RTX 5090 Tensor Core W4 | [`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md) |
| RTX 5090 train_temp alignment | [`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md) |
| V100 B1/B8 active-parameter comparison | [`bench/v100_active_b1b8_20260715/`](bench/v100_active_b1b8_20260715/README.md) |
| V100 production close | [`bench/v100_production_close_20260711/`](bench/v100_production_close_20260711/README.md) |
| RTX 4090 B8 matrices | [`bench/4090_small_bsz8_20260715/`](bench/4090_small_bsz8_20260715/README.md) |
| Apple M5 bounded production result | [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md) |

For exact numbers and caveats use [`BENCHMARK.md`](BENCHMARK.md), not this
landing page.

Completion is reported by **named scope**, not as a single repository-wide
percentage. A completed milestone does not imply universal hardware, shape,
training, or quantization completion.

Canonical project state:

- [Current status](HF_STATUS.md)
- [Remaining work](HF_TODO.md)
- [Acceptance criteria](docs/ACCEPTANCE.md)
- [Hardware matrix](docs/HARDWARE_MATRIX.md)
- [Benchmark summary](BENCHMARK.md)
- [Raw benchmark inventory](bench/INDEX.md)

## Installation profiles

```bash
python -m pip install -e .                    # core native HF path
python -m pip install -e ".[cuda]"            # CUDA build helper
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
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — performance boundaries
- [`docs/BACKENDS.md`](docs/BACKENDS.md) — backend and hardware isolation
- [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) — HF criteria
- [`docs/integrations/README.md`](docs/integrations/README.md) — serving-engine contracts
- [`docs/archive/`](docs/archive/) — superseded plans and milestone history

## Attribution and license

The project is MIT licensed; see [`LICENSE`](LICENSE). Machine-readable project
and implementation provenance is in [`CITATION.cff`](CITATION.cff),
[`docs/reference/provenance.yaml`](docs/reference/provenance.yaml),
[`CONTRIBUTORS.md`](CONTRIBUTORS.md), and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

RWKV-7 mathematics and official checkpoints originate from BlinkDL/RWKV-LM.
Vendored or derived FLA/self-chunk, Marlin, and train_temp components retain
their own copyright and license notices.
