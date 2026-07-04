# Apple Silicon / MPS / MLX adaptation plan

This document tracks the HF adapter work needed for Apple Silicon. It is a
separate hardware-adaptation lane from the CUDA / Albatross performance route:
CUDA fused kernels remain the production-speed target for NVIDIA cards, while
Apple Silicon uses the FLA-free native PyTorch path today plus an optional MLX
reference backend. Full fused MLX/Metal kernels remain the next Apple
performance layer.

## Current status

| Area | Status | Evidence / entry point |
|---|---|---|
| Install without CUDA/FLA | supported by packaging | Base dependencies no longer require `flash-linear-attention`; CUDA users can install `.[fla]` / `.[cuda]`. |
| Tiny Apple smoke | pass on local M-series | `tests/test_apple_silicon_smoke.py` passes on MacBook Air / Apple M5 / 16GB / macOS 26.5 / PyTorch 2.12.1 MPS; see `bench/results_apple_silicon_m5_20260704.jsonl`. |
| Converted-model Apple smoke | 0.1B, 0.4B, and 1.5B pass on local M-series | `scripts/run_apple_silicon_smoke.sh` loads `rwkv7-g1d-0.1b-hf`, `rwkv7-g1d-0.4b-hf`, and `rwkv7-g1g-1.5b-hf` through `RWKV7_NATIVE_MODEL=1` on MPS; 0.4B has fp32/fp16 short-generate rows and 1.5B has fp16 short-generate + prompt sweep rows. |
| HF API coverage | partial | Load + forward + `generate(use_cache=True)` through the native backend; tiny native backward and Trainer paths pass; real 0.1B and 0.4B PEFT LoRA, HF Trainer, and TRL SFT/DPO/GRPO paths on MPS are covered. 0.4B also has fp32/fp16 generation length sweep rows and 2-step Trainer/TRL rows. 1.5B has fp16 inference/sweep rows through prompt 512 / decode 8 plus fp32 PEFT LoRA manual, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10-step rows with finite trainable updates. |
| Quantization | broader functional native smoke | `bitsandbytes` W8/W4 is CUDA-oriented and is not the Apple path. Native MM8/MM4 config-driven module replacement now runs on MPS for tiny, 0.1B, 0.4B, and 1.5B smoke rows, including lower `min_params` sweeps that replace FFN/projection groups rather than only `lm_head`; production-speed Apple quant still needs MLX/Metal kernels. |
| Production speed | not claimed | PyTorch MPS is a compatibility path, not the final Apple performance backend. |
| MLX recurrent backend / Metal backend | initial MLX recurrent reference + session helper, Metal TODO | Optional `.[mlx]` install, `rwkv7_hf.mlx_bridge`, `rwkv7_hf.mlx_model`, `scripts/convert_hf_to_mlx.py`, `scripts/mlx_generate.py`, `scripts/mlx_session_smoke.py`, `scripts/mlx_session_batch_smoke.py`, `scripts/mlx_generation_sweep.py`, `scripts/run_apple_silicon_mlx_smoke.sh`, `scripts/run_apple_silicon_mlx_model_smoke.sh`, `scripts/run_apple_silicon_mlx_session_smoke.sh`, `scripts/run_apple_silicon_mlx_session_batch_smoke.sh`, and `scripts/run_apple_silicon_mlx_generation_sweep.sh` now validate HF safetensor → MLX array/export, tiny torch/MLX recurrent parity, MLX state-cache select/chunked-prefill/session behavior, tokenizer-backed prompt smoke, dynamic-batch state select, reusable MLX text generate, prefill-once/session decode, interleaved multi-session decode with repeat-pressure summary rows, prompt/decode length sweeps plus repeat/pressure rows with chunked-prefill checks, and 0.1B/0.4B/1.5B MLX recurrent rows. Fused quant and Metal WKV kernels are still TODO. |

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
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.1b-hf` native MM8/MM4 quant smoke | PASS (`MIN_PARAMS=8000000` lm_head replacement; `MIN_PARAMS=1000000` replaces 25 FFN/lm_head modules; `MIN_PARAMS=500000` replaces 73 attention/FFN/lm_head modules; MM8 footprint ratio≈0.253433, MM4 footprint ratio≈0.128433; 1-token generate) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1d-0.4b-hf` native MM8/MM4 quant sweep | PASS (`MIN_PARAMS=4000000` replaces 49 FFN/lm_head modules; MM8 footprint ratio≈0.252327, MM4 footprint ratio≈0.127327; 1-token generate) |
| MacBook Air / Apple M5 | 16GB | 26.5 | 2.12.1 / Transformers 5.13.0 | MPS | `rwkv7-g1g-1.5b-hf` native MM8/MM4 quant smoke | PASS (`MIN_PARAMS=8000000` replaces 49 FFN/lm_head modules; MM8 footprint ratio≈0.251190, MM4 footprint ratio≈0.126190; 1-token generate; MM4 driver memory≈14.8GB, so 16GB is functional smoke only) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | tiny MLX tensor save/load + matmul smoke | PASS (`axis=apple_silicon_mlx_tiny`, `elapsed_s=0.032803`, output shape `[1, 24]`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` HF safetensor → MLX projection matmul | PASS (`axis=apple_silicon_mlx_projection_smoke`, tensor `model.layers.0.attn.r_proj.weight`, fp16 `[1, 768]`, selected tensor bytes=1179648) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` selected HF safetensor → MLX safetensors export | PASS (`axis=mlx_hf_export`, tensor count=1, fp16 bytes=1179648, manifest `mlx_manifest.json`) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 / PyTorch 2.12.1 | MLX GPU | tiny full recurrent MLX vs native PyTorch parity | PASS (`axis=apple_silicon_mlx_recurrent_tiny_parity`, batch=2, seq=4, max_abs=0.00282228, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | tiny MLX recurrent state cache + chunked prefill + session decode | PASS (`axis=apple_silicon_mlx_state_cache_tiny`, chunked/full max_abs=0.0, select-batch decode max_abs=0.0014168; `axis=apple_silicon_mlx_session_tiny`, step_sizes=2,2, one-shot token/text match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` full MLX recurrent prefill + greedy decode | PASS (`axis=apple_silicon_mlx_recurrent_model_smoke`, fp16 full 399 tensors, prompt=4, generated=1, chunked/full max_abs=0.0, bytes=382069248) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 / PyTorch 2.12.1 | MLX GPU + CPU compare | `rwkv7-g1d-0.1b-hf` MLX recurrent vs HF native PyTorch | PASS (`axis=apple_silicon_mlx_recurrent_model_smoke`, fp32, torch_compare_max_abs=0.01374531, argmax match, bytes=764138496) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.1b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (`prompt="The quick brown fox"`, fp16, prompt=4, generated=2, prefill≈132.05 tok/s, decode≈167.99 tok/s, dynamic select max_abs=0.046875, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1d-0.4b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (fp16 full 795 tensors, bytes=901535744, prompt=4, generated=1, prefill≈62.95 tok/s, decode≈83.74 tok/s, dynamic select max_abs=0.03125, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `rwkv7-g1g-1.5b-hf` tokenizer prompt + dynamic-batch MLX recurrent smoke | PASS (fp16 full 795 tensors, bytes=3054809088, prompt=4, generated=1, prefill≈10.38 tok/s, decode≈29.33 tok/s, dynamic select max_abs=0.046875, argmax match) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `scripts/mlx_generate.py` reusable text-generate API | PASS (`rwkv7-g1d-0.1b-hf` 8 tokens decode≈95.29 tok/s, peak≈389MB; `rwkv7-g1d-0.4b-hf` 8 tokens decode≈53.02 tok/s, peak≈914MB; `rwkv7-g1g-1.5b-hf` 4 tokens decode≈28.97 tok/s, peak≈3080MB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `scripts/mlx_generation_sweep.py` prompt/decode sweep + chunked-prefill/repeat check | PASS (`rwkv7-g1d-0.1b-hf` prompt16/64 decode2/4 chunk=32 peak≈397MB, repeat pressure prompt32/decode2 x3 peak≈397MB, and longer prompt128/256 decode4/8 repeat=2 chunk=64 peak≈397MB / min prefill≈187.23 tok/s / min decode≈153.35 tok/s; `rwkv7-g1d-0.4b-hf` prompt16/64 decode2 peak≈934MB and longer prompt128/256 decode4/8 repeat=1 chunk=64 peak≈934MB / min prefill≈49.77 tok/s / min decode≈33.51 tok/s; `rwkv7-g1g-1.5b-hf` prompt16/64, decode2, peak≈3119MB; all chunked/full max_abs=0.0) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `MLXGenerationSession` prefill-once + chunked decode smoke | PASS (`rwkv7-g1d-0.1b-hf` step_sizes=4,4 token/text match vs one-shot, decode≈60.43 tok/s, peak≈392MB; `rwkv7-g1d-0.4b-hf` step_sizes=4,4 match, decode≈54.13 tok/s, peak≈921MB; `rwkv7-g1g-1.5b-hf` step_sizes=2,2 match, decode≈27.50 tok/s, peak≈3093MB) |
| MacBook Air / Apple M5 | 16GB | 26.5 | MLX 0.31.2 | MLX GPU | `MLXGenerationSessionBatch` interleaved multi-session decode smoke | PASS (`rwkv7-g1d-0.1b-hf` 2 sessions rounds=2,2 match peak≈394MB; 3 sessions rounds=2,2 repeat=2 match, summary peak≈397MB / cache≈7.4MB / min decode≈94.56 tok/s; `rwkv7-g1d-0.4b-hf` 2 sessions rounds=1,1 match peak≈927MB; `rwkv7-g1g-1.5b-hf` 2 sessions rounds=1,1 match peak≈3106MB) |
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

# Apple native MM8/MM4 quant on converted 0.1B.
# MIN_PARAMS_LIST sweeps from lm_head-only into broader projection groups.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=8000000,1000000,500000 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_quant_smoke.sh

# 0.4B quant sweep: MIN_PARAMS=4000000 covers FFN key/value + lm_head modules.
MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DEVICE=auto DTYPE=fp32 QUANTIZATIONS=mm8,mm4 MIN_PARAMS_LIST=4000000 \
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

# Full MLX recurrent reference backend: tiny parity/cache plus optional 0.1B row.
DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=2 \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
DYNAMIC_BATCH=1 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1g-1.5b-hf \
MODEL_SIZE_LABEL=1.5b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
DYNAMIC_BATCH=1 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh

# Reusable tokenizer-integrated MLX text generation API / CLI.
python scripts/mlx_generate.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --dtype fp16 \
  --results bench/results_apple_silicon_m5_20260704.jsonl

# Prompt/decode sweep with chunked-prefill correctness and memory telemetry.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16,64 \
DECODE_LENGTHS=2,4 \
CHUNK_SIZE=32 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_generation_sweep.sh

# Serving-shaped MLX session: prefill once, decode in chunks, compare with one-shot.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_session_smoke.sh

# Optional stronger real-checkpoint parity against HF native PyTorch on CPU.
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp32 \
TOKENS=1,2,3,4 \
CHUNK_SIZE=2 \
MAX_NEW_TOKENS=1 \
COMPARE_TORCH=1 \
TORCH_COMPARE_TOLERANCE=0.05 \
SKIP_TINY=1 \
RESULTS=bench/results_apple_silicon_m5_20260704.jsonl \
  scripts/run_apple_silicon_mlx_model_smoke.sh
```

The Trainer wrapper calls `tests/test_apple_silicon_trainer_smoke.py` directly. The 0.1B/0.4B/1.5B model-training, TRL SFT, and TRL RL wrappers call `tests/test_apple_silicon_model_training_smoke.py`. The generation sweep wrapper calls `tests/test_apple_silicon_model_sweep.py`. The native quant wrapper calls `tests/test_apple_silicon_quant_smoke.py`. The MLX bridge wrapper calls `tests/test_apple_silicon_mlx_smoke.py`; the full recurrent MLX wrapper calls `tests/test_apple_silicon_mlx_model_smoke.py`; the reusable MLX generation CLI is `scripts/mlx_generate.py`; the MLX prompt/decode sweep CLI is `scripts/mlx_generation_sweep.py` with wrapper `scripts/run_apple_silicon_mlx_generation_sweep.sh`; the serving-style prefill-once/session-decode CLI is `scripts/mlx_session_smoke.py` with wrapper `scripts/run_apple_silicon_mlx_session_smoke.sh`; the interleaved multi-session CLI is `scripts/mlx_session_batch_smoke.py` with wrapper `scripts/run_apple_silicon_mlx_session_batch_smoke.sh`; and the HF→MLX exporter is `scripts/convert_hf_to_mlx.py`.

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

DTYPE=fp16 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
MODEL_SIZE_LABEL=0.1b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
MODEL_SIZE_LABEL=0.4b \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
SKIP_TINY=1 \
DYNAMIC_BATCH=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_model_smoke.sh

python scripts/mlx_generate.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  --prompt "The quick brown fox" \
  --max-new-tokens 8 \
  --dtype fp16

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=16,64 \
DECODE_LENGTHS=2,4 \
CHUNK_SIZE=32 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=2 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.4b-hf \
DTYPE=fp16 \
PROMPT_LENGTHS=128,256 \
DECODE_LENGTHS=4,8 \
CHUNK_SIZE=64 \
REPEAT=1 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_generation_sweep.sh

MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DTYPE=fp16 \
PROMPT="The quick brown fox" \
STEP_SIZES=4,4 \
RESULTS=bench/results_apple_silicon_mlx_recurrent.jsonl \
scripts/run_apple_silicon_mlx_session_smoke.sh
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
| M-series 16GB+ | 16GB+ | tiny + 0.1B/0.4B/1.5B HF | fp16 / fp32 | MLX GPU | `scripts/run_apple_silicon_mlx_model_smoke.sh` tiny MLX/Torch recurrent parity, state-cache select, chunked prefill, tokenizer prompt, dynamic-batch state select, 0.1B/0.4B/1.5B full MLX recurrent prefill/generate, optional HF native PyTorch compare, `scripts/mlx_generate.py` reusable text generation, `scripts/run_apple_silicon_mlx_generation_sweep.sh` prompt/decode sweep plus repeat pressure with chunked-prefill checks, `scripts/run_apple_silicon_mlx_session_smoke.sh` prefill-once/session decode equality vs one-shot, and `scripts/run_apple_silicon_mlx_session_batch_smoke.sh` interleaved multi-session equality vs one-shot plus repeat-pressure summary telemetry |
| M-series 16GB+ | 16GB+ | 0.4B HF | fp32 / fp16 | mps | load + forward + generate + prompt-length sweep through 512 tokens + PEFT LoRA/Trainer/SFT/DPO/GRPO 1-step/2-step smoke + memory note |
| M-series 16GB+ | 16GB+ | tiny + 0.1B/0.4B/1.5B HF | fp32 native MM8/MM4 | mps | bitsandbytes-free native quant smoke + min-params sweep + packed-footprint ratio + finite forward/generate; 1.5B on 16GB is memory-tight evidence only |
| M-series 16GB+ | 16GB+ | 1.5B HF | fp16 inference / fp32 LoRA smoke | mps | load/generate + prompt sweep through 512 tokens / decode 8 + PEFT manual + Trainer/SFT/DPO/GRPO 1/2/3/5/10-step + peak memory + finite trainable update |
| M-series Max / Ultra | 64GB+ | 1.5B+ HF | fp16 / bf16 | mps | longer decode, 10+ step Trainer/TRL rows, peak memory, tok/s |

For every Apple result, include:

- macOS version, chip, memory size;
- Python / PyTorch / Transformers versions;
- `torch.backends.mps.is_built()` and `is_available()`;
- command line and JSONL result row;
- Activity Monitor or `memory_pressure` notes if the run swaps heavily.

Apple harness maintenance note: shared script helpers live in
`tests/apple_silicon_utils.py`. New MPS/MLX smoke rows should use it for
hardware probes, JSONL output, package versions, model-size labels, device/dtype
selection, and MPS memory telemetry instead of copying helper blocks into each
test script.

## Known limitations

- This is not an Albatross-speed path. PyTorch MPS validates HF compatibility on
  Apple hardware but does not replace CUDA fused kernels.
- `bitsandbytes` quantization is not an Apple path. Native MM8/MM4 now has
  MPS functional smoke plus 0.1B/0.4B/1.5B min-params sweeps with
  packed-footprint telemetry, but production Apple W8/W4 still needs MLX/Metal
  packing and fused kernels.
- The MLX path is now a correctness-first recurrent reference backend, not a
  production-speed backend. It verifies HF safetensor loading/export, full
  recurrent prefill/decode equations, tokenizer prompt handling/API, state-cache
  select, chunked prefill, dynamic-batch row selection, prefill-once/session
  decode equality vs one-shot, interleaved multi-session equality vs one-shot with repeat-pressure summary rows, prompt/decode sweeps through 256-token prompts on 0.1B/0.4B plus repeat pressure rows,
  and 0.1B/0.4B/1.5B short greedy decode. Fused
  WKV, fused quant/dequant, longer prompts/decodes, and production serving
  integration are still open.
- Long-running full-size training on MPS is not claimed yet. Tiny native Trainer
  and tiny PEFT LoRA Trainer pass; 0.1B and 0.4B PEFT LoRA backward, HF Trainer,
  TRL SFT, DPO, and GRPO one-step and 2-step smoke pass on a 16GB M5. 1.5B
  fp32 PEFT LoRA manual backward, HF Trainer, and TRL SFT/DPO/GRPO 1/2/3/5/10-step
  smoke now pass. Longer 1.5B decode beyond 8 tokens, >10-step training, and larger Apple machines
  are still open. Native MM8/MM4 functional/min-params smoke through 1.5B and
  initial MLX recurrent reference smoke are present; full MLX/Metal acceleration
  and production quant speed are still open.
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
3. Extend Apple native MM8/MM4 beyond 1.5B one-token min-params smoke to longer
   prompts/decodes, memory-pressure rows, and real fused MLX/Metal W8/W4 speed paths.
4. Extend the MLX recurrent reference and `MLXGenerationSession` beyond the current
   0.1B/0.4B prompt256/decode8 matrices and 3-session batch repeat row to 1.5B longer prompt/decode matrices,
   stronger memory-pressure telemetry, and longer production-style concurrent session reuse.
5. Replace the correctness-first MLX recurrent inner loop with a fused MLX or
   Metal WKV-7 kernel, then add packed W8/W4 dequant/fused kernels.
6. Decide whether the Metal WKV-7 kernel belongs in this repo as an optional
   backend or in a sibling `rwkv7-mlx` / `rwkv7-metal` package.
