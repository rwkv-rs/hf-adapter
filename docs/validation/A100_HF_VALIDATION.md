# A100 HF validation matrix

Validation date: 2026-07-02  
Base commit: `c09227a` (`docs: record 13.3B V100 official alignment + decode speed (#83)`)  
Server: `8 x NVIDIA A100-PCIE-40GB` on `gpu03`; validation used 1 GPU for inference and single-card training, 2 GPUs for DeepSpeed.  
Runtime: Python `3.12.8`, PyTorch `2.8.0+cu126`, Transformers `4.57.1`, PEFT `0.19.1`, TRL `1.7.0`, DeepSpeed `0.19.2`, bitsandbytes `0.49.2`, FLA `0.5.1`.

This file records the A100 40GB extension for issue #68 after the initial 0.1B A100 baseline was merged in #82. It covers the larger 0.4B, 1.5B, 2.9B, and 7.2B checkpoints. Cross-card comparison with V100 / 4090 / H100 is intentionally out of scope for this validation pass.

## Summary

| Model | Generate smoke | fp16/bf16 batch sweep | Trainer | SFT | DPO | HF checkpoint resume | ZeRO2 | ZeRO2 resume | ZeRO3 | Quant 8bit/4bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |
| 1.5B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |
| 2.9B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |
| 7.2B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |

Notes:

- All training rows use PEFT LoRA trainable parameters through the HF Trainer / TRL harnesses and assert finite loss plus a non-zero trainable-parameter delta.
- ZeRO3 base training smoke passes for all four sizes on 2 x A100 40GB. ZeRO3 checkpoint resume remains a follow-up: a direct ZeRO3 resume attempt reproduced a DeepSpeed/PyTorch dtype mismatch in `all_gather_into_tensor` during checkpoint epilogue.
- A100 80GB was not available in the current cluster. This pass records A100 40GB evidence only.
- Quantized W8/W4 rows reduce memory for all four sizes. Their decode-speed fields are marked interim pending the native-fused packed-quant / tensor-core-aware kernel work; generic bitsandbytes decode is still slower than fp16/native-graph.

## Model assets

All checkpoints were converted to HF format with `--precision fp16 --attn-mode fused_recurrent --no-fuse-norm`.

| Model | Source checkpoint sha256 | Source bytes | HF dir |
|---|---|---:|---|
| 0.4B | `947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb` | 901776749 | `/home/wenhongli/workspace/rwkv/models/rwkv7/rwkv7-g1d-0.4b-hf` |
| 1.5B | `441f70b096ad62442b5c33128bfe717c5d8529915c45a9709d4482016e8a0482` | 3055444605 | `/home/wenhongli/workspace/rwkv/models/rwkv7/rwkv7-g1g-1.5b-hf` |
| 2.9B | `3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798` | 5896273469 | `/home/wenhongli/workspace/rwkv/models/rwkv7/rwkv7-g1g-2.9b-hf` |
| 7.2B | `425fc9bda2d12d4ce3b6bfe5c3b3f355be8b14d85960cf40fcca58a19d632630` | 14400007869 | `/home/wenhongli/workspace/rwkv/models/rwkv7/rwkv7-g1g-7.2b-hf` |

## Environment

The GPU worker used the shared virtual environment:

```bash
export PATH=/home/wenhongli/workspace/rwkv/.venv-rwkv7-a100/bin:$PATH
export PYTHON_BIN=/home/wenhongli/workspace/rwkv/.venv-rwkv7-a100/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/wenhongli/workspace/rwkv/rwkv7-hf-adapter
export RWKV_V7_ON=1
export TORCHDYNAMO_DISABLE=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR=/tmp/wenhongli/triton-cache
export HF_MODULES_CACHE=/tmp/wenhongli/hf-modules-rwkv7-a100-ext
```

Transformers remote-code relative imports were preseeded into `HF_MODULES_CACHE` before each model load because the GPU nodes run isolated from the public network.

## Commands

Representative commands for the extended A100 pass:

```bash
python bench/bench_larger_model_smoke.py \
  --hf-dir "$MODEL" \
  --model-size-label "$MODEL_SIZE_LABEL" \
  --checkpoint-path "$CHECKPOINT" \
  --device cuda \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_batch_sweep.py \
  --hf-dir "$MODEL" \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 128 \
  --decode-tokens 16 \
  --batch-sizes 1 2 4 8 \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl

python bench/bench_quantization.py \
  --hf-dir "$MODEL" \
  --device cuda \
  --dtype fp16 \
  --quantizations none 8bit 4bit \
  --prompt-tokens 128 \
  --decode-tokens 16 \
  --results bench/results.jsonl

RUN_PEFT=1 RUN_TRAINER=1 RUN_RL=0 RUN_RESUME=1 TRAIN_DTYPE=bf16 \
  MAX_STEPS=2 RESUME_FIRST_STEPS=1 RESUME_STEPS=2 \
  BATCH_SIZE=1 DATASET_REPEATS=4 DEVICE=cuda \
  bash scripts/run_hf_training_matrix.sh "$MODEL"

python tests/test_hf_rl_training_smoke.py \
  --model "$MODEL" \
  --device cuda \
  --backend dpo \
  --train-dtype bf16 \
  --max-steps 1 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ZERO_STAGE=both TRAIN_DTYPE=bf16 \
  MAX_LENGTH=16 MAX_STEPS=2 BATCH_SIZE=1 DATASET_REPEATS=4 \
  bash scripts/run_zero_training_smoke.sh "$MODEL"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL" \
  --zero-stage 2 \
  --train-dtype bf16 \
  --first-steps 1 \
  --resume-steps 2 \
  --max-length 16 \
  --results bench/results.jsonl
```

The batch-size upper bound was reduced for larger checkpoints to stay within the A100 40GB memory envelope: 0.4B used up to batch 8, 1.5B up to batch 4, and 2.9B/7.2B up to batch 2.

## Inference smoke

| Model | Layers | Hidden | Footprint MB | Peak VRAM MB | Generate tok/s |
|---|---:|---:|---:|---:|---:|
| 0.4B | 24 | 1024 | 859.8 | 1124.5 | 2.7 |
| 1.5B | 24 | 2048 | 2913.3 | 3178.6 | 2.7 |
| 2.9B | 32 | 2560 | 5622.4 | 5888.0 | 2.5 |
| 7.2B | 32 | 4096 | 13731.3 | 13997.8 | 2.6 |

All four smoke rows used `RWKV7_FAST_TOKEN_BACKEND=auto` and resolved the cached decode path to `native_graph`.

## Batch sweep

The table reports the `rwkv7_forward_token` rows. The same JSONL block also keeps the ordinary cached `forward` rows for comparison.

| Model | Dtype | Batches | Batch-1 decode tok/s | Max-batch decode tok/s | Max-batch peak VRAM MB |
|---|---|---|---:|---:|---:|
| 0.4B | fp16 | 1,2,4,8 | 147.0 | 1539.8 | 2867.4 |
| 0.4B | bf16 | 1,2,4,8 | 146.5 | 1530.8 | 2867.4 |
| 1.5B | fp16 | 1,2,4 | 164.5 | 578.2 | 4904.1 |
| 1.5B | bf16 | 1,2,4 | 164.9 | 552.6 | 4904.1 |
| 2.9B | fp16 | 1,2 | 101.8 | 189.1 | 7261.5 |
| 2.9B | bf16 | 1,2 | 78.5 | 166.2 | 7261.5 |
| 7.2B | fp16 | 1,2 | 59.2 | 117.2 | 16336.1 |
| 7.2B | bf16 | 1,2 | 58.5 | 117.1 | 16336.1 |

## Quantized inference

| Model | Quantization | Footprint MB | Peak VRAM MB | Decode tok/s | Speed status | Load s |
|---|---|---:|---:|---:|---|---:|
| 0.4B | fp16 none | 859.8 | 921.9 | 144.8 | baseline | 2.53 |
| 0.4B | 8bit | 571.8 | 629.6 | 12.3 | interim | 2.18 |
| 0.4B | 4bit | 427.8 | 502.6 | 25.3 | interim | 1.06 |
| 1.5B | fp16 none | 2913.3 | 3012.4 | 119.5 | baseline | 3.19 |
| 1.5B | 8bit | 1761.3 | 1853.2 | 11.5 | interim | 3.39 |
| 1.5B | 4bit | 1185.3 | 1345.7 | 25.0 | interim | 2.58 |
| 2.9B | fp16 none | 5622.4 | 5770.4 | 73.5 | baseline | 4.06 |
| 2.9B | 8bit | 3222.4 | 3358.5 | 8.9 | interim | 4.75 |
| 2.9B | 4bit | 2022.4 | 2301.2 | 19.2 | interim | 4.83 |
| 7.2B | fp16 none | 13731.3 | 13953.0 | 61.4 | baseline | 6.46 |
| 7.2B | 8bit | 7587.3 | 7887.7 | 7.0 | interim | 12.37 |
| 7.2B | 4bit | 4515.3 | 5195.6 | 15.3 | interim | 10.11 |

All quantized rows are functional and reduce memory. W8/W4 decode-speed rows
are also tagged with `quant_speed_status=interim` in `bench/results.jsonl`.
They do not close the production speed target; W8/W4 still need a native
packed/fused serving path.

## Single-GPU training and resume

| Model | Backend | Max length | Steps | Loss | Runtime s | Trainable delta |
|---|---|---:|---:|---:|---:|---:|
| 0.4B | HF Trainer | 32 | 2 | 2.5358 | 23.7025 | 0.00015 |
| 0.4B | TRL SFT | 32 | 2 | 2.3997 | 0.5884 | 0.00015 |
| 0.4B | TRL DPO | 32 | 1 | 0.6931 | 1.7628 | 0.00010 |
| 1.5B | HF Trainer | 32 | 2 | 2.2910 | 138.5073 | 0.00015 |
| 1.5B | TRL SFT | 32 | 2 | 2.1203 | 0.5836 | 0.00015 |
| 1.5B | TRL DPO | 32 | 1 | 0.6931 | 1.6910 | 0.00010 |
| 2.9B | HF Trainer | 32 | 2 | 2.2153 | 276.5354 | 0.00015 |
| 2.9B | TRL SFT | 32 | 2 | 2.0503 | 0.7614 | 0.00015 |
| 2.9B | TRL DPO | 32 | 1 | 0.6931 | 1.7851 | 0.00010 |
| 7.2B | HF Trainer | 16 | 2 | 2.2522 | 24.0003 | 0.00015 |
| 7.2B | TRL SFT | 16 | 2 | 2.0409 | 0.8109 | 0.00015 |
| 7.2B | TRL DPO | 16 | 1 | 0.6931 | 1.9104 | 0.00010 |

HF Trainer checkpoint resume:

| Model | Max length | Global step | First loss | Resume loss | Resume trainable delta |
|---|---:|---:|---:|---:|---:|
| 0.4B | 32 | 2 | 1.9162 | 1.5719 | 0.06250 |
| 1.5B | 32 | 2 | 1.9227 | 1.3307 | 0.04419 |
| 2.9B | 32 | 2 | 1.9395 | 1.2430 | 0.03958 |
| 7.2B | 16 | 2 | 1.6358 | 1.4197 | 0.03125 |

## DeepSpeed on 2 x A100 40GB

Rank-0 rows are shown below; `bench/results.jsonl` stores rank-local rows for both ranks.

| Model | ZeRO stage | Max length | Steps | Loss | Runtime s | Trainable delta |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 2 | 16 | 2 | 5.0234 | 16.0444 | 0.00015 |
| 0.4B | 3 | 16 | 2 | 5.0156 | 20.9012 | 0.00015 |
| 1.5B | 2 | 16 | 2 | 4.3984 | 55.3342 | 0.00015 |
| 1.5B | 3 | 16 | 2 | 4.4141 | 12.9105 | 0.00015 |
| 2.9B | 2 | 16 | 2 | 4.3320 | 113.2216 | 0.00015 |
| 2.9B | 3 | 16 | 2 | 4.3438 | 14.1363 | 0.00015 |
| 7.2B | 2 | 16 | 2 | 4.3516 | 17.8671 | 0.00015 |
| 7.2B | 3 | 16 | 2 | 4.3594 | 9.1463 | 0.00015 |

ZeRO2 checkpoint resume:

| Model | ZeRO stage | Max length | Global step | First loss | Resume loss | Resume trainable delta |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 2 | 16 | 2 | 5.1406 | 2.4336 | 0.06250 |
| 1.5B | 2 | 16 | 2 | 4.6484 | 2.0898 | 0.04419 |
| 2.9B | 2 | 16 | 2 | 4.4922 | 2.0898 | 0.03958 |
| 7.2B | 2 | 16 | 2 | 4.5703 | 2.0664 | 0.03125 |

## Result rows

After this validation pass, `bench/results.jsonl` contains 134 A100 rows:

- 68 `batch_sweep` rows, including both ordinary cached `forward` and `rwkv7_forward_token` rows.
- 20 `deepspeed_training_smoke` rows.
- 16 `training_smoke` rows.
- 12 `quantization` rows.
- 8 `deepspeed_resume_smoke` rows.
- 4 `larger_model_smoke` rows.
- 4 `checkpoint_resume_smoke` rows.
- 2 legacy 0.1B `speed_mem` rows from #82.

Remaining A100-specific gaps after this pass:

- A100 80GB validation is not available on the current cluster.
- ZeRO3 checkpoint resume needs a DeepSpeed/PyTorch dtype-mismatch fix.
- Quantized decode needs fused/native W8/W4 kernels to meet the "not slower than fp16" target; current W8/W4 speed rows are interim.
- Longer production training sweeps remain useful, but the requested larger-model smoke, batch sweep, quantized functional/memory evidence, interim quantized speed telemetry, HF checkpoint resume, ZeRO base, and ZeRO2 resume evidence is now present for A100 40GB.
