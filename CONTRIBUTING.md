# Contributing to the RWKV-7 HF Adapter

Thanks for helping with the RWKV-7 Hugging Face / Transformers adapter. This
repository is focused on the **HF adapter track**: loading, conversion,
generation, PEFT, Trainer, TRL, DeepSpeed, HF state-cache helpers, quantized HF
inference, hardware/card validation, and production-readiness evidence.

vLLM, SGLang, DFlash, and standalone serving-engine integrations are separate
projects. Do not mix them into HF adapter PRs unless an issue explicitly asks
for shared helper code or documentation.

## Start here

1. Read [`HF_STATUS.md`](HF_STATUS.md) to understand what is already done.
2. Read [`HF_TODO.md`](HF_TODO.md) to pick a current task.
3. For performance or hardware work, read [`BENCHMARK.md`](BENCHMARK.md).
4. For the current V100 training/quant/ZeRO evidence, read [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md).
5. For kernel/performance experiments, also read [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md).
6. For Apple Silicon work, read [`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md).
7. Pick an issue, comment that you are working on it, then open a focused PR.

## Current issue map

The open card-adaptation issues are designed to make it easy for contributors
with different hardware to help.

| Issue | Target | Main contribution |
|---|---|---|
| #66 | RTX 4090 / Ada | Consumer Ada smoke, speed, quant, and training rows. |
| #67 | RTX 5090 / 50-series / Blackwell | Blackwell decode/prefill/quant regression and 5090 rows. |
| #68 | A100 / Ampere | Production batch sweeps, bf16, int8, ZeRO-2/3 rows. |
| #69 | H100 / Hopper | High-end bf16/fp8-aware validation, large-model rows. |
| #70 | Pascal / Turing | Older-card fallback behavior and fp16/quant constraints. |
| #71 | AMD / ROCm | Native/no-FLA compatibility first, ROCm gaps second. |
| #72 | CPU fallback | No-CUDA import, tiny native forward/generate, API tests. |
| Apple Silicon / MPS | Apple native/no-FLA load/generate first, MLX/Metal backend later. |
| #73 | Jetson AGX Thor | aarch64/Jetson Linux unified-memory validation. |
| #74 | DGX Spark / GB10 | Grace Blackwell unified-memory validation. |

If your card is not listed, open a new `[card] ...` issue using the same shape:
status, checklist, card-specific risks, and definition of done.

## What a good contribution looks like

A good PR is small, reproducible, and tied to one acceptance gap.

Examples:

- Add A100 benchmark rows and update `BENCHMARK.md`.
- Add a ZeRO checkpoint-resume smoke test.
- Add a one-click acceptance script.
- Fix a `generate()` / `attention_mask` / cache behavior bug with a regression
  test.
- Add AMD/CPU fallback coverage to the native/no-FLA path.
- Add 8-bit/4-bit quantized inference telemetry for a new card.

Avoid large PRs that mix unrelated tasks such as docs, kernels, training, and
serving changes at the same time.

## Local setup

Typical environment variables for GPU work:

```bash
export PYTHONNOUSERSITE=1
export RWKV_V7_ON=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/path/to/flash-linear-attention:/path/to/rwkv7-hf-adapter:${PYTHONPATH:-}
```

For DeepSpeed smoke on machines without a full CUDA toolkit setup, some tests
also support:

```bash
export DS_IGNORE_CUDA_DETECTION=1
```

Use the project-specific environment and model paths from your issue or PR body.
Do not hardcode private local paths in committed scripts unless they are examples
with `/path/to/...` placeholders.

## Minimal no-GPU checks

For docs, conversion, and API-contract changes that do not require a live GPU,
run the relevant subset:

```bash
python tests/test_convert_config.py
python tests/test_batch_convert_manifest.py
python tests/test_result_tools.py
python tests/test_sync_hf_adapter_code.py
git diff --check
```

If dependencies are missing, mention the skip reason in the PR body.

## Minimal GPU card validation

For a card-adaptation issue, prefer the one-click wrapper first:

```bash
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=cuda DTYPE=fp16 \
RESULTS=bench/results.jsonl \
bash scripts/run_hardware_smoke.sh
```

If the wrapper fails or you need to bisect, run the underlying commands:

```bash
python tests/smoke_hf_generate.py \
  --model /path/to/rwkv7-g1d-0.1b-hf

python tests/test_hf_api_contract.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --dtype fp16

python tests/test_quantized_inference.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --quantization 8bit \
  --optional

python tests/test_quantized_inference.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --quantization 4bit \
  --optional
```

Then add speed rows:

```bash
python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --backend hf \
  --dtype fp16 \
  --device cuda \
  --results bench/results.jsonl

python bench/bench_batch_sweep.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --results bench/results.jsonl
```

For training-capable cards, add:

```bash
python tests/test_peft_lora.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent

python tests/test_hf_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent \
  --backend both \
  --results bench/results.jsonl

python tests/test_hf_rl_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent \
  --backend dpo \
  --results bench/results.jsonl
```

For multi-GPU cards/nodes, add ZeRO smoke through the wrapper:

```bash
NPROC_PER_NODE=2 ZERO_STAGE=both \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results.jsonl \
bash scripts/run_zero_training_smoke.sh
```

Equivalent raw command for debugging:

```bash
torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 32 \
  --results bench/results.jsonl
```

## Minimal Apple Silicon validation

Apple Silicon does not use the CUDA/FLA path. Use the native backend and record
MPS availability:

```bash
python -m pip install -e .
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=auto DTYPE=fp32 \
RESULTS=bench/results_apple_silicon.jsonl \
bash scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b SKIP_TINY=1 MAX_NEW_TOKENS=1 \
DEVICE=auto DTYPE=fp32 \
RESULTS=bench/results_apple_silicon.jsonl \
bash scripts/run_apple_silicon_smoke.sh

REQUIRE_PEFT=1 \
DEVICE=auto DTYPE=fp32 \
RESULTS=bench/results_apple_silicon_training.jsonl \
bash scripts/run_apple_silicon_training_smoke.sh

REQUIRE_PEFT=1 \
DEVICE=auto DTYPE=fp32 \
RESULTS=bench/results_apple_silicon_trainer.jsonl \
bash scripts/run_apple_silicon_trainer_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=auto DTYPE=fp32 MAX_LENGTH=8 MAX_STEPS=1 REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_model_training.jsonl \
bash scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=auto DTYPE=fp32 MAX_LENGTH=8 MAX_STEPS=1 REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_trl_sft.jsonl \
bash scripts/run_apple_silicon_model_trl_sft_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=auto DTYPE=fp32 MAX_LENGTH=8 MAX_STEPS=1 REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_rl.jsonl \
bash scripts/run_apple_silicon_model_rl_smoke.sh
```

If the model dir has stale remote-code files, sync them first:

```bash
python scripts/sync_hf_adapter_code.py /path/to/rwkv7-g1d-0.1b-hf
```


Apple native MM8/MM4 quant smoke (bitsandbytes-free):

```bash
# Tiny only. Add MODEL=/path/to/rwkv7-g1d-0.1b-hf for a converted-model row.
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 \
RESULTS=bench/results_apple_silicon_quant.jsonl \
bash scripts/run_apple_silicon_quant_smoke.sh
```

Apple MLX bridge/export smoke:

```bash
python -m pip install -e '.[mlx]'

# Tiny MLX save/load/matmul smoke. Add MODEL for a real HF projection row.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
bash scripts/run_apple_silicon_mlx_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
bash scripts/run_apple_silicon_mlx_smoke.sh

python scripts/convert_hf_to_mlx.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  /tmp/rwkv7-g1d-0.1b-mlx \
  --dtype fp16 \
  --include model.layers.0.attn.r_proj.weight \
  --copy-metadata

# Full MLX recurrent reference smoke: tiny parity/cache only.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_model_smoke.sh

# Full MLX recurrent reference smoke on converted 0.1B.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_model_smoke.sh

# Larger MLX rows should start short on 16GB machines.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
SKIP_TINY=1 DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_model_smoke.sh

python scripts/mlx_generate.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --dtype fp16

# Prompt/decode length sweep with MLX memory telemetry.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16,64 \
DECODE_LENGTHS=2,4 \
CHUNK_SIZE=32 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Serving-shaped MLX session smoke: prefill once, decode in chunks, compare with one-shot.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_smoke.sh
```

Include the `torch_mps_built` / `torch_mps_available` lines printed by the
wrapper. On 16GB machines, start with tiny / 0.1B first, then short 0.4B
generate, `scripts/run_apple_silicon_model_sweep.sh`, and 0.4B PEFT/Trainer/TRL
one-step smoke before longer sweeps. For 1.5B on 16GB machines, start with
fp16 load/forward/short-generate and a prompt-length sweep through 512 tokens;
then add prompt512/new8 or 10-step Trainer/TRL rows only after closing other
memory-heavy apps, and confirm the result has finite positive
trainable-gradient or trainable-update totals. Treat non-finite fp16 PEFT
gradients/updates as a failed row, not as evidence.

## Reporting hardware results

Every hardware/card PR should include this information in the PR body or in a
linked issue comment:

````markdown
## Environment

- GPU(s):
- Driver:
- CUDA or ROCm:
- OS:
- Python:
- PyTorch:
- Transformers:
- PEFT:
- TRL:
- DeepSpeed:
- flash-linear-attention:
- Model path / size:
- dtype:

## Commands

```bash
# paste exact commands
```

## Results

- Smoke status:
- Prefill tok/s:
- Decode tok/s:
- Peak VRAM / memory:
- Quantized footprint:
- Quantized speed:
- Training loss / trainable delta, if applicable:

## Known limits

- Unsupported dtype/backend:
- Compile or kernel issues:
- Fallback path used:
````

If a benchmark writes rows to `bench/results.jsonl`, commit only rows that are
relevant to the PR. Do not mix unrelated local experiments into the same results
change.

## Documentation updates

Update docs when the PR changes public behavior, card support, or known gaps.

Common docs to update:

- `HF_STATUS.md` — if a status changes from open/partial to done.
- `HF_TODO.md` — if a TODO is completed, split, or reprioritized.
- `BENCHMARK.md` — if you add benchmark or hardware rows.
- `README.md` — if contributor-facing entry points or quickstart commands change.
- `docs/performance/FUSED_BACKEND.md` — if you change fused/native performance routes.

## Pull request checklist

Before opening a PR:

- [ ] The PR is scoped to one issue or one clear gap.
- [ ] Tests or benchmark commands are listed in the PR body.
- [ ] Hardware/software versions are listed for GPU work.
- [ ] `bench/results.jsonl` rows, if changed, are relevant and reproducible.
- [ ] Docs are updated if support status changed.
- [ ] The PR does not start vLLM/SGLang work in this HF adapter repository.

## Issue completion checklist

A card issue can usually be closed when:

- the required smoke commands pass or skips are explicitly justified;
- benchmark rows are recorded where applicable;
- `BENCHMARK.md` or the PR body summarizes the card result;
- the issue is updated with the final supported dtype/backend/model range;
- known limitations and fallback paths are documented.
