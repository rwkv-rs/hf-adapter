# Apple Silicon / MPS / MLX adaptation plan

This document tracks the HF adapter work needed for Apple Silicon. It is a
separate hardware-adaptation lane from the CUDA / Albatross performance route:
CUDA fused kernels remain the production-speed target for NVIDIA cards, while
Apple Silicon uses the FLA-free native PyTorch path today plus an initial
optional MLX tensor bridge. Full recurrent MLX/Metal kernels remain the next
Apple performance layer.

## Current status

| Area | Status | Evidence / entry point |
|---|---|---|
| Install without CUDA/FLA | supported by packaging | Base dependencies no longer require `flash-linear-attention`; CUDA users can install `.[fla]` / `.[cuda]`. |
| Tiny Apple smoke | pass on local M-series | `tests/test_apple_silicon_smoke.py` passes on MacBook Air / Apple M5 / 16GB / macOS 26.5 / PyTorch 2.12.1 MPS; see `bench/results_apple_silicon_m5_20260704.jsonl`. |
| Converted-model Apple smoke | 0.1B, 0.4B, and 1.5B pass on local M-series | `scripts/run_apple_silicon_smoke.sh` loads `rwkv7-g1d-0.1b-hf`, `rwkv7-g1d-0.4b-hf`, and `rwkv7-g1g-1.5b-hf` through `RWKV7_NATIVE_MODEL=1` on MPS; 0.4B has fp32/fp16 short-generate rows and 1.5B has fp16 short-generate + prompt sweep rows. |
| HF API coverage | partial | Load + forward + `generate(use_cache=True)` through the native backend; tiny native backward and Trainer paths pass; real 0.1B and 0.4B PEFT LoRA, HF Trainer, and TRL SFT/DPO/GRPO paths on MPS are covered. 0.4B also has fp32/fp16 generation length sweep rows and 2-step Trainer/TRL rows. 1.5B has fp16 inference/sweep rows through prompt 512 / decode 8 plus fp32 PEFT LoRA manual, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10-step rows with finite trainable updates. |
| Quantization | functional native smoke | `bitsandbytes` W8/W4 is CUDA-oriented and is not the Apple path. Native MM8/MM4 config-driven module replacement now runs on MPS for tiny and 0.1B smoke rows with packed-footprint telemetry; production-speed Apple quant still needs MLX/Metal kernels. |
| Production speed | not claimed | PyTorch MPS is a compatibility path, not the final Apple performance backend. |
| MLX bridge / Metal backend | initial MLX bridge, Metal TODO | Optional `.[mlx]` install, `rwkv7_hf.mlx_bridge`, `scripts/convert_hf_to_mlx.py`, and `scripts/run_apple_silicon_mlx_smoke.sh` now validate HF safetensor → MLX array/export and a real 0.1B projection matmul. Full RWKV recurrent forward, state cache, fused quant, and Metal WKV kernels are still TODO. |

## Why the Apple path is native / no-FLA by default

The default optimized CUDA wrapper depends on `flash-linear-attention` and
Triton/CUDA kernels. Those are not the right baseline for macOS. Apple Silicon
smoke should use:

```bash
export RWKV7_NATIVE_MODEL=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export RWKV7_FAST_FORWARD=0
export RWKV7_FAST_CACHE=0
export RWKV7_FAST_TOKEN_BACKEND=native_jit
```

`RWKV7_NATIVE_MODEL=1` routes `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` into the FLA-free native PyTorch backend. MPS fallback
keeps unsupported individual ops from aborting the run while the dedicated
Metal/MLX backend is still future work.

## Current local evidence

Local smoke on 2026-07-04:

| Machine | Memory | macOS | PyTorch | Device | Test | Result |
|---|---:|---|---|---|---|---|
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | tiny native RWKV-7 `generate()` | PASS (`elapsed_s=0.1121`, 2 generated tokens) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.1b-hf` load + forward + `generate()` | PASS (`elapsed_s=0.2406`, 11 prompt tokens + 2 generated tokens) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | tiny native MM8/MM4 quant smoke | PASS (config-driven from_pretrained; MM8 footprint ratio=0.391615, MM4 footprint ratio=0.267734; decode backend=eager) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.1b-hf` native MM8/MM4 quant smoke | PASS (lm_head replacement; MM8 footprint ratio=0.252635, MM4 footprint ratio=0.127635; 1-token generate) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | tiny MLX tensor save/load + matmul smoke | PASS (`axis=apple_silicon_mlx_tiny`, `elapsed_s=0.032803`, output shape `[1, 24]`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` HF safetensor → MLX projection matmul | PASS (`axis=apple_silicon_mlx_projection_smoke`, tensor `model.layers.0.attn.r_proj.weight`, fp16 `[1, 768]`, selected tensor bytes=1179648) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` selected HF safetensor → MLX safetensors export | PASS (`axis=mlx_hf_export`, tensor count=1, fp16 bytes=1179648, manifest `mlx_manifest.json`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` load + forward + `generate()` | PASS (`elapsed_s=0.4699`, 11 prompt tokens + 1 generated token, MPS driver memory≈2171MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` fp16 load + forward + `generate()` | PASS (`elapsed_s=1.2837`, 11 prompt tokens + 1 generated token, MPS driver memory≈1083MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` fp32/fp16 prompt-length sweep | PASS (fp32 prompt tokens 16/64/128; fp16 prompt tokens 16/64/128/256/512; 4 generated tokens; fp16 peak driver_mem≈1219MiB, fp32 peak driver_mem≈2203MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` fp16 load + forward + `generate()` | PASS (`elapsed_s=1.6407`, 11 prompt tokens + 1 generated token, MPS driver memory≈3283MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` fp16 prompt-length/decode sweep | PASS (prompt tokens 16/64/128/256/512; 2/4/8 generated tokens; peak driver_mem≈3547MiB; prompt512/new8 prefill 29.753 tok/s, decode 0.453 tok/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / PEFT 0.19.1 | MPS | tiny native train + PEFT LoRA train | PASS (`loss=3.870411`, LoRA trainable params=1792) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | tiny native Trainer + PEFT LoRA Trainer | PASS (`training_loss=3.877832`, native `changed_l1=6.063786`, LoRA `changed_l1=0.891996`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1d-0.1b-hf` PEFT LoRA train + HF Trainer | PASS (`loss=2.70401`, LoRA params=663552, Trainer `changed_l1=26.63627`, driver_mem≈2466MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1d-0.4b-hf` PEFT LoRA train + HF Trainer | PASS (`loss=2.22734`, LoRA params=1769472, Trainer `changed_l1=67.165365`, driver_mem≈4651MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1g-1.5b-hf` fp32 PEFT LoRA manual backward | PASS (`loss=1.976301`, LoRA params=3538944, `grad_l1=10386.947289`, `changed_l1=137.084609`, driver_mem≈6843MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 | MPS | `rwkv7-g1g-1.5b-hf` fp32 PEFT LoRA HF Trainer | PASS (1/2/3/5/10-step rows; 10-step `training_loss=1.574907`, `changed_l1=769.378162`, 0.565 steps/s, driver_mem≈6875MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL SFTTrainer + PEFT LoRA | PASS (`training_loss=2.70401`, `changed_l1=26.620176`, 1.446 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL GRPOTrainer + PEFT LoRA | PASS (`training_loss=0.0`, `changed_l1=10.454315`, 3.098 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.1b-hf` TRL DPOTrainer + PEFT LoRA | PASS (`training_loss=0.693147`, `changed_l1=28.315877`, 1.518 steps/s) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1d-0.4b-hf` TRL SFT/DPO/GRPO + PEFT LoRA | PASS (1-step and 2-step rows; 2-step SFT `training_loss=3.140634`, DPO `training_loss=0.692913`, GRPO `training_loss=0.0`, peak driver_mem≈4980MiB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 / PEFT 0.19.1 / TRL 1.7.0 | MPS | `rwkv7-g1g-1.5b-hf` TRL SFT/DPO/GRPO + PEFT LoRA | PASS (1/2/3/5/10-step rows; 10-step SFT `training_loss=1.569469`, DPO `training_loss=0.422009`, GRPO `training_loss=0.0`, peak driver_mem≈8604MiB) |

Memory-pressure note after the latest 1.5B prompt512/new8 and 10-step
Trainer/TRL runs: `vm_stat` reported free pages≈410k, inactive pages≈152k,
speculative pages≈74k, wired pages≈183k, compressor pages≈26k. Swap counters
are cumulative macOS counters, so the table relies on per-row MPS memory for
run-local memory evidence.

Commands:

```bash
PYTHONPATH=. python tests/test_apple_silicon_smoke.py \
  --device auto \
  --dtype fp32 \
  --max-new-tokens 2 \
  --results bench/results_apple_silicon_m5_20260704.jsonl

PYTHONPATH=. python tests/test_apple_silicon_smoke.py \
  --device auto \
  --dtype fp32 \
  --max-new-tokens 2 \
  --skip-tiny \
  --model /path/to/rwkv7-g1d-0.1b-hf

REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_training_smoke.sh

REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_trainer_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 REQUIRE_TRL=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_trl_sft_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
REQUIRE_PEFT=1 REQUIRE_TRL=1 RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
SKIP_TINY=1 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
SKIP_TINY=1 \
DTYPE=fp16 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
PROMPT_LENGTHS=16,64,128 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT_LENGTHS=256,512 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_trl_sft_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
MAX_LENGTH=16 MAX_STEPS=2 DATASET_REPEATS=3 \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_rl_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
SKIP_TINY=1 \
DTYPE=fp16 \
MAX_NEW_TOKENS=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=16 \
MAX_NEW_TOKENS=2 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=64,128,256,512 \
MAX_NEW_TOKENS=4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT_LENGTHS=512 \
MAX_NEW_TOKENS=8 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_sweep.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=1 \
DATASET_REPEATS=2 \
BACKEND=manual \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trainer \
REQUIRE_PEFT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trl_sft \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
BACKEND=trl_dpo \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp32 \
MAX_LENGTH=8 \
BATCH_SIZE=1 \
MAX_STEPS=10 \
DATASET_REPEATS=12 \
GRPO_MAX_COMPLETION_LENGTH=1 \
BACKEND=trl_grpo \
REQUIRE_PEFT=1 REQUIRE_TRL=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_model_training_smoke.sh

# Apple native MM8/MM4 quant, tiny-only.
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# Apple native MM8/MM4 quant on converted 0.1B; default MIN_PARAMS=8000000 replaces lm_head.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# Apple MLX bridge, tiny-only.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_smoke.sh

# Apple MLX bridge on one real 0.1B projection tensor.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_smoke.sh

# Export selected HF safetensors into an MLX-readable bundle.
python scripts/convert_hf_to_mlx.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  /tmp/rwkv7-g1d-0.1b-mlx \
  --dtype fp16 \
  --include model.layers.0.attn.r_proj.weight \
  --copy-metadata \
  --results bench/results_apple_silicon_m5_20260704.jsonl
```

The Trainer wrapper calls `tests/test_apple_silicon_trainer_smoke.py` directly. The 0.1B/0.4B/1.5B model-training, TRL SFT, and TRL RL wrappers call `tests/test_apple_silicon_model_training_smoke.py`. The generation sweep wrapper calls `tests/test_apple_silicon_model_sweep.py`. The native quant wrapper calls `tests/test_apple_silicon_quant_smoke.py`. The MLX wrapper calls `tests/test_apple_silicon_mlx_smoke.py`, and the HF→MLX exporter is `scripts/convert_hf_to_mlx.py`.

Recorded rows: [`../../bench/results_apple_silicon_m5_20260704.jsonl`](../../bench/results_apple_silicon_m5_20260704.jsonl).

## Minimal Apple environment

Use an isolated environment. On Apple Silicon the base package should install
without FLA:

```bash
cd /path/to/rwkv7-hf-adapter
python3 -m venv .venv-apple-torch
source .venv-apple-torch/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install accelerate
```

For MLX bridge/export validation on Apple Silicon, install the optional MLX
extra:

```bash
python -m pip install -e '.[mlx]'
```

If `pip install -e .` is not desired, the lightweight fallback is:

```bash
python -m pip install torch torchvision torchaudio transformers safetensors accelerate
export PYTHONPATH=/path/to/rwkv7-hf-adapter:${PYTHONPATH:-}
```

CUDA users who want the optimized default backend should install the optional
extra instead:

```bash
python -m pip install -e '.[fla]'
# or, for CUDA/Triton development helpers:
python -m pip install -e '.[cuda]'
```

## Verify PyTorch MPS

```bash
python - <<'PY'
import platform
import torch
print('platform', platform.platform())
print('machine', platform.machine())
print('torch', torch.__version__)
print('mps built', torch.backends.mps.is_built())
print('mps available', torch.backends.mps.is_available())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(torch.ones(1, device=device))
PY
```

## Smoke commands

Tiny native model only, safe to run before downloading model weights:

```bash
PYTHONPATH=. python tests/test_apple_silicon_smoke.py --device auto --dtype fp32
```

Converted model smoke, records a JSONL row:

```bash
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results_apple_silicon.jsonl \
DEVICE=auto \
DTYPE=fp32 \
MAX_NEW_TOKENS=2 \
scripts/run_apple_silicon_smoke.sh
```

MLX bridge smoke, safe tiny row plus optional converted-model projection row:

```bash
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
scripts/run_apple_silicon_mlx_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx.jsonl \
scripts/run_apple_silicon_mlx_smoke.sh

python scripts/convert_hf_to_mlx.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  /tmp/rwkv7-g1d-0.1b-mlx \
  --dtype fp16 \
  --include model.layers.0.attn.r_proj.weight \
  --copy-metadata
```

Tiny native training + optional PEFT LoRA training smoke:

```bash
RESULTS=bench/results_apple_silicon_training.jsonl \
DEVICE=auto \
DTYPE=fp32 \
REQUIRE_PEFT=1 \
scripts/run_apple_silicon_training_smoke.sh
```

The underlying test is [`../../tests/test_apple_silicon_training_smoke.py`](../../tests/test_apple_silicon_training_smoke.py).

If the converted model directory was produced by an older checkout, sync the
current remote-code files first:

```bash
python scripts/sync_hf_adapter_code.py /path/to/rwkv7-g1d-0.1b-hf
```

## Validation matrix to fill

| Machine | Memory | Model | Dtype | Device | Required result |
|---|---:|---|---|---|---|
| M1 / M2 Air | 16GB | tiny native | fp32 | mps or cpu | `APPLE SILICON SMOKE PASS` |
| M1 / M2 Air | 16GB | 0.1B HF | fp32 | mps | load + forward + 2-token generate + PEFT LoRA/Trainer/SFT/DPO/GRPO smoke |
| M-series 16GB+ | 16GB+ | tiny + selected 0.1B HF tensors | fp16 | MLX GPU | `scripts/run_apple_silicon_mlx_smoke.sh` tiny save/load/matmul + HF projection matmul, and optional `scripts/convert_hf_to_mlx.py` export manifest |
| M-series 16GB+ | 16GB+ | 0.4B HF | fp32 / fp16 | mps | load + forward + generate + prompt-length sweep through 512 tokens + PEFT LoRA/Trainer/SFT/DPO/GRPO 1-step/2-step smoke + memory note |
| M-series 16GB+ | 16GB+ | tiny + 0.1B HF | fp32 native MM8/MM4 | mps | bitsandbytes-free native quant smoke + packed-footprint ratio + finite forward/generate |
| M-series 16GB+ | 16GB+ | 1.5B HF | fp16 inference / fp32 LoRA smoke | mps | load/generate + prompt sweep through 512 tokens / decode 8 + PEFT manual + Trainer/SFT/DPO/GRPO 1/2/3/5/10-step + peak memory + finite trainable update |
| M-series Max / Ultra | 64GB+ | 1.5B+ HF | fp16 / bf16 | mps | longer decode, 10+ step Trainer/TRL rows, peak memory, tok/s |

For every Apple result, include:

- macOS version, chip, memory size;
- Python / PyTorch / Transformers versions;
- `torch.backends.mps.is_built()` and `is_available()`;
- command line and JSONL result row;
- Activity Monitor or `memory_pressure` notes if the run swaps heavily.

## Known limitations

- This is not an Albatross-speed path. PyTorch MPS validates HF compatibility on
  Apple hardware but does not replace CUDA fused kernels.
- `bitsandbytes` quantization is not an Apple path. Native MM8/MM4 now has
  an MPS functional smoke path and packed-footprint telemetry, but production
  Apple W8/W4 still needs MLX/Metal packing and fused kernels.
- The MLX path is currently a bridge/export and projection-matmul proof, not a
  full RWKV recurrent inference backend. It verifies selected HF safetensors can
  become MLX arrays/bundles; full state cache, sequence prefill/decode, and
  WKV/quant kernels are still open.
- Long-running full-size training on MPS is not claimed yet. Tiny native Trainer
  and tiny PEFT LoRA Trainer pass; 0.1B and 0.4B PEFT LoRA backward, HF Trainer,
  TRL SFT, DPO, and GRPO one-step and 2-step smoke pass on a 16GB M5. 1.5B
  fp32 PEFT LoRA manual backward, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10-step
  smoke now pass. Longer 1.5B decode beyond 8 tokens, >10-step training, and larger Apple machines
  are still open. Native MM8/MM4 functional smoke and initial MLX bridge/export
  smoke are present; full MLX/Metal acceleration and production quant speed are still open.
- 1.5B fp16 PEFT LoRA on the 16GB M5 produced non-finite gradient/update values
  in one local trial. The training smoke now rejects non-finite or zero
  trainable-gradient/update totals instead of recording false-positive rows.
- 16GB machines should start with tiny / 0.1B, then short 0.4B generate
  before longer sweeps. Close browsers and IDEs before running converted-model
  smoke.

## MLX / Metal references

RafaelUI's Apple-focused RWKV work is the most relevant starting point for the
next backend layer:

- [RafaelUI/metal-wkv7](https://github.com/RafaelUI/metal-wkv7): custom Metal WKV-7 forward/backward kernel.
- [RafaelUI/rwkv-metal](https://github.com/RafaelUI/rwkv-metal): Apple Silicon RWKV-7 training / LoRA / QLoRA direction.
- [RafaelUI/rwkv-mlx](https://github.com/RafaelUI/rwkv-mlx): MLX RWKV-7 pretraining / conversion / LoRA direction.
- [RafaelUI/SwiftRWKV](https://github.com/RafaelUI/SwiftRWKV): Swift + MLX / Apple platform inference direction.

## Next engineering steps

1. Extend 0.4B Apple rows beyond 2 training steps.
2. Extend 1.5B beyond 10-step Trainer/TRL and prompt512/new8 sweep to longer
   decode, >10-step Trainer/TRL, and memory-pressure notes.
3. Extend Apple native MM8/MM4 from 0.1B lm_head smoke to larger models and more projection groups.
4. Extend the initial MLX tensor bridge into a full recurrent forward path:
   full-weight export, state-cache layout, chunked prefill/decode parity, and
   MLX-native generate smoke.
5. Decide whether the Metal WKV-7 kernel belongs in this repo as an optional
   backend or in a sibling `rwkv7-mlx` / `rwkv7-metal` package.
