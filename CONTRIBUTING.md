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
6. For backend or hardware work, read [`docs/BACKENDS.md`](docs/BACKENDS.md).
7. For Apple Silicon work, read [`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md).
8. Pick an issue, comment that you are working on it, then open a focused PR.

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

## Backend boundary rule

Cards are validation rows, not code branches. Keep exact card/chip names in
docs, tests, scripts, benchmark JSONL, and `rwkv7_hf/kernel_policy.py`. Core
model code should branch on capabilities such as backend availability,
`device.type`, dtype support, graph-capture support, or the normalized policy
family. See [`docs/BACKENDS.md`](docs/BACKENDS.md) for the full contract.

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
# Tiny only.
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 \
RESULTS=bench/results_apple_silicon_quant.jsonl \
bash scripts/run_apple_silicon_quant_smoke.sh

# Converted-model sweep. MIN_PARAMS_LIST lowers the replacement threshold so
# contributors can prove more than lm_head-only quantization on Apple MPS.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=8000000,1000000,500000 \
RESULTS=bench/results_apple_silicon_quant.jsonl \
bash scripts/run_apple_silicon_quant_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=4000000 \
RESULTS=bench/results_apple_silicon_quant.jsonl \
bash scripts/run_apple_silicon_quant_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=8000000 \
SKIP_TINY=1 MAX_NEW_TOKENS=1 \
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

# Longer 0.1B matrix for pressure rows on 16GB machines.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Longer 0.4B matrix; keep REPEAT=1 on 16GB machines unless memory is quiet.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Initial MLX/Metal WKV custom-kernel seam smoke.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Initial MLX packed W8/W4 quant path: affine dequant-matmul projection smoke.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# 1.5B W8/W4 path: close memory-heavy apps first on 16GB machines.
MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=4 \
CHUNK_SIZE=16 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=4 \
CHUNK_SIZE=16 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=affine \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Initial MLX/Metal packed W8/W4 fused dequant-projection seam. Pair it with
# WKV_BACKEND=metal when comparing against the current fp16 Metal WKV row.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=32 \
DECODE_LENGTHS=2 \
CHUNK_SIZE=16 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
DECODE_LENGTHS=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Longer MLX/Metal W8/W4 quant pressure matrix. These rows pair the fused
# dequant-projection seam with the Metal WKV seam and chunked-prefill equality.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Long-context MLX/Metal quant matrix: prompt512/1024 + decode16.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Same-shape fp16 Metal baselines for W8/W4 ratio gates. Report both the
# quant rows above and these baseline rows before claiming speed parity.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=none \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=512,1024 \
DECODE_LENGTHS=16 \
CHUNK_SIZE=256 \
REPEAT=1 \
QUANTIZATION=none \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Extended long-decode ratio gate. Run the same shape for QUANTIZATION=none,
# mm8, and mm4 before claiming W8/W4 speed parity. For mm8/mm4 set
# QUANT_BACKEND=metal and the model-specific QUANT_MIN_PARAMS shown below.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=2048 \
DECODE_LENGTHS=128 \
CHUNK_SIZE=512 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=2048 \
DECODE_LENGTHS=128 \
CHUNK_SIZE=512 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Longer 1.5B matrix; close memory-heavy apps first on 16GB machines.
MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Serving-shaped MLX session smoke: prefill once, decode in chunks, compare with one-shot.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_smoke.sh

# Interleaved multi-session smoke: prefill multiple prompts and advance them round-by-round.
# SESSION_BACKEND defaults to sequential; use batched/auto to exercise equal-round MLX batching.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
ROUNDS=2,2 \
REPEAT=2 \
SESSION_BACKEND=batched \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Repeat the same 3-session pressure row on 0.4B / 1.5B after tiny/0.1B pass.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
ROUNDS=2,2 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
ROUNDS=2,2 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Quant+Metal interleaved session-batch pressure. Use QUANTIZATION=mm4 with
# SESSION_BACKEND=batched for the current batch-exact path. For QUANTIZATION=mm8
# use SESSION_BACKEND=auto; it records auto_mm8_metal_batch_exactness_guard and
# falls back to sequential until strict W8/Metal batched decode is fixed.
# This validates prefill-once/session state reuse while both WKV and quantized
# projections use the opt-in Metal paths.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=4 \
ROUNDS=4,4 \
REPEAT=2 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=batched \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=4 \
ROUNDS=4,4 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Higher-concurrency quant+Metal session pressure. Run both QUANTIZATION=mm8 and
# QUANTIZATION=mm4 when collecting the full W8/W4 matrix.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=6 \
ROUNDS=4,4 \
REPEAT=3 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=5 \
ROUNDS=4,4 \
REPEAT=2 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Longer equal-round session pressure. Use mm4 + SESSION_BACKEND=batched for
# exact batched W4; use mm8 + SESSION_BACKEND=auto to keep W8 on the safe
# sequential fallback until strict W8/Metal batched exactness is fixed.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=8 \
ROUNDS=8,8 \
REPEAT=2 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=batched \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=5 \
ROUNDS=8,8 \
REPEAT=2 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=batched \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=5 \
ROUNDS=8,8 \
REPEAT=2 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=8000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=auto \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Strict scheduler comparison: compare sequential vs batched without hiding
# mismatches. Use REQUIRE_SESSION_BACKEND_MATCH=1 for W4 gates. For W8 gap
# capture, omit REQUIRE_SESSION_BACKEND_MATCH so the row records the mismatch
# instead of aborting; current 0.4B W8/Metal strict batched decode diverges at
# token index 6 on the short prompt while SESSION_BACKEND=auto remains safe.
# Set TRACE_MISMATCH_LOGITS=1 to append a top-k logit trace for the first
# divergent token.
# Use QUANT_BACKEND=auto to validate the conservative quant backend router:
# W4 normal prefill/decode rows select Metal, while W8 defaults to affine until W8/Metal
# batch exactness is fixed. The result rows expose
# quantized_linear_last_backend_counts so reviewers can verify the route used.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=6 \
ROUNDS=4,4 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=sequential \
COMPARE_SESSION_BACKEND=batched \
COMPARE_ONLY=1 \
REQUIRE_SESSION_BACKEND_MATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=6 \
ROUNDS=4,4 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=metal \
WKV_BACKEND=metal \
SESSION_BACKEND=sequential \
COMPARE_SESSION_BACKEND=batched \
COMPARE_ONLY=1 \
TRACE_MISMATCH_LOGITS=1 \
MISMATCH_TOPK=5 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Conservative quant backend auto route. W4 should report Metal calls in
# quantized_linear_last_backend_counts; W8 should report affine calls by default.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=3 \
ROUNDS=4,4 \
REPEAT=1 \
QUANTIZATION=mm4 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=auto \
WKV_BACKEND=metal \
SESSION_BACKEND=sequential \
COMPARE_SESSION_BACKEND=batched \
COMPARE_ONLY=1 \
REQUIRE_SESSION_BACKEND_MATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=3 \
ROUNDS=4,4 \
REPEAT=1 \
QUANTIZATION=mm8 \
QUANT_MIN_PARAMS=4000000 \
QUANT_BACKEND=auto \
WKV_BACKEND=metal \
SESSION_BACKEND=sequential \
COMPARE_SESSION_BACKEND=batched \
COMPARE_ONLY=1 \
REQUIRE_SESSION_BACKEND_MATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Extended 0.4B / 1.5B long-decode matrix: prompt1024 + decode64.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=1024 \
DECODE_LENGTHS=64 \
CHUNK_SIZE=256 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=1024 \
DECODE_LENGTHS=64 \
CHUNK_SIZE=256 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_generation_sweep.sh

# Higher-pressure session matrix: 4 concurrent sessions, longer rounds, repeat=4.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
PROMPT_D="Fourth concurrent MLX session for pressure." \
ROUNDS=4,4 \
REPEAT=4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
PROMPT_A="The quick brown fox" \
PROMPT_B="User: Apple Silicon RWKV test. Assistant:" \
PROMPT_C="Repeat pressure prompt for MLX sessions." \
PROMPT_D="Fourth concurrent MLX session for pressure." \
ROUNDS=4,4 \
REPEAT=4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

# Higher-concurrency session matrix: SESSION_COUNT lets the wrapper synthesize
# enough extra prompts to reach the requested concurrent-session count. You can
# also use PROMPTS_FILE (one prompt per line) or EXTRA_PROMPTS (newline-separated).
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
SESSION_COUNT=6 \
ROUNDS=4,4 \
REPEAT=5 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
DTYPE=fp16 \
SESSION_COUNT=5 \
ROUNDS=4,4 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
bash scripts/run_apple_silicon_mlx_session_batch_smoke.sh
```

Include the `torch_mps_built` / `torch_mps_available` lines printed by the
wrapper. On 16GB machines, start with tiny / 0.1B first, then short 0.4B
generate, `scripts/run_apple_silicon_model_sweep.sh`, and 0.4B PEFT/Trainer/TRL
one-step smoke before longer sweeps. For 1.5B on 16GB machines, start with
fp16 load/forward/short-generate and a prompt-length sweep through 512 tokens;
then add prompt1024/decode64 or 12-step Trainer/TRL rows only after closing other
memory-heavy apps, and confirm the result has finite positive
trainable-gradient or trainable-update totals. Treat non-finite fp16 PEFT
gradients/updates as a failed row, not as evidence.

When adding or extending Apple hardware scripts, reuse
`tests/apple_silicon_utils.py` for common environment probes, JSONL result
writing, model-size labels, MPS availability checks, dtype/device selection, and
MPS memory telemetry. This keeps MPS/MLX smoke rows comparable across native
generate, training, quantization, and MLX session harnesses.

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
