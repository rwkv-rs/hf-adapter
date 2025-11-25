# A800 HF validation matrix

> **Exact-card snapshot dated 2026-07-04.** Preserve the measured A800 rows and
> their card-local gaps. Current repository-wide quant/training status lives in
> [`../HARDWARE_MATRIX.md`](../HARDWARE_MATRIX.md), [`../QUANTIZATION.md`](../QUANTIZATION.md)
> and [`../TRAINING.md`](../TRAINING.md).

Validation date: 2026-07-04
Base branch: `issue98-a800-validation-work` based on `v0.4.0`
Server: `NVIDIA A800-SXM4-80GB`; inference used one GPU, DeepSpeed used one and two GPUs.
Runtime: Python `3.12.12`, PyTorch `2.10.0+cu128`, Transformers `5.0.0`, PEFT `0.19.1`, TRL `1.6.0`, DeepSpeed `0.18.5`, bitsandbytes `0.49.2`, FLA `0.5.0`.
Driver / CUDA: NVIDIA driver `535.129.03`, `nvidia-smi` CUDA `12.8`.

This file records the A800 extension for issue #98. It complements the earlier
#97 A800 rows for 0.4B / 1.5B / 2.9B batch sweep, bnb quantization, and basic
training smoke. Paths in committed result rows are placeholders; reproduce by
substituting local converted HF checkpoint directories.

## Summary

| Area | Result |
|---|---|
| 0.1B generate / HF API / PEFT | pass |
| 0.1B official alignment | pass; top5 and argmax 1.0, cosine 0.9999957, greedy 64/64 |
| 0.1B Trainer / SFT / DPO / GRPO | pass with nonzero trainable deltas |
| 0.1B bnb W8/W4 functional quant | pass; footprint 283.4 MB / 242.9 MB |
| 7.2B fp16 larger-model smoke | pass; footprint 13731.3 MB, peak 13998.8 MiB |
| 13.3B bnb W8/W4 smoke | pass; footprint 13597.1 MB / 7741.1 MB |
| 0.4B single-GPU ZeRO-2 / ZeRO-3 | pass |
| 0.4B single-GPU ZeRO-2 / ZeRO-3 resume | pass; resumed to global step 2 |
| 0.4B 2-GPU ZeRO-2 / ZeRO-3 | pass |
| 0.4B 2-GPU ZeRO-2 / ZeRO-3 resume | pass; resumed to global step 2 |
| 0.4B / 1.5B / 2.9B / 7.2B / 13.3B native mm8/mm4 | pass; footprint drops, but 1.5B+ decode is slower than fp16 |

## Commands

Representative commands from this pass:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tests/test_official_alignment.py \
  --hf-dir "$MODEL_0P1B_HF" \
  --pth "$MODEL_0P1B_PTH" \
  --dtype fp16 \
  --device cuda \
  --official-strategy "cuda fp16" \
  --greedy-window 64 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tests/test_hf_rl_training_smoke.py \
  --model "$MODEL_0P1B_HF" \
  --model-size-label 0.1b \
  --device cuda \
  --attn-mode fused_recurrent \
  --train-dtype bf16 \
  --backend both \
  --max-steps 1 \
  --batch-size 1 \
  --dataset-repeats 2 \
  --max-length 32 \
  --grpo-max-completion-length 2 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python bench/bench_larger_model_smoke.py \
  --hf-dir "$MODEL_7P2B_HF" \
  --model-size-label 7.2b \
  --checkpoint-path "$MODEL_7P2B_PTH" \
  --device cuda \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 4 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python bench/bench_native_mm_quant_decode.py \
  --hf-dir "$MODEL_HF" \
  --model-size-label "$MODEL_SIZE_LABEL" \
  --dtype fp16 \
  --device cuda \
  --quantizations none mm8 mm4 \
  --min-params 8000000 \
  --prompt-tokens 128 \
  --decode-tokens 64 \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python bench/bench_quantization.py \
  --hf-dir "$MODEL_13P3B_HF" \
  --model-size-label 13.3b \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantizations 8bit 4bit \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --decode-mode fast \
  --warmup 0 \
  --runs 1 \
  --quant-skip-policy memory \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tests/test_deepspeed_training_smoke.py \
  --model "$MODEL_0P4B_HF" \
  --model-size-label 0.4b \
  --zero-stage both \
  --attn-mode fused_recurrent \
  --train-dtype bf16 \
  --max-steps 1 \
  --batch-size 1 \
  --max-length 64 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL_0P4B_HF" \
  --model-size-label 0.4b \
  --zero-stage both \
  --attn-mode fused_recurrent \
  --train-dtype bf16 \
  --first-steps 1 \
  --resume-steps 2 \
  --batch-size 1 \
  --max-length 16 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 \
  tests/test_deepspeed_training_smoke.py \
  --model "$MODEL_0P4B_HF" \
  --model-size-label 0.4b \
  --zero-stage both \
  --attn-mode fused_recurrent \
  --train-dtype bf16 \
  --max-steps 1 \
  --batch-size 1 \
  --max-length 64 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 \
  tests/test_deepspeed_resume_smoke.py \
  --model "$MODEL_0P4B_HF" \
  --model-size-label 0.4b \
  --zero-stage both \
  --attn-mode fused_recurrent \
  --train-dtype bf16 \
  --first-steps 1 \
  --resume-steps 2 \
  --batch-size 1 \
  --max-length 16 \
  --results bench/results.jsonl
```

## Native mm8/mm4 quantization

Rows use `bench_native_mm_quant_decode.py`, prompt128/decode64, fp16 load, and
`min_params=8_000_000`.

| Model | Quantization | Replaced modules | Footprint MB | Decode tok/s | vs fp16 |
|---|---|---:|---:|---:|---:|
| 0.4B | none | 0 | 859.8 | 185.2 | 1.00x |
| 0.4B | native mm8 | 1 | 796.0 | 187.9 | 1.01x |
| 0.4B | native mm4 | 1 | 764.0 | 185.8 | 1.00x |
| 1.5B | none | 0 | 2913.3 | 172.7 | 1.00x |
| 1.5B | native mm8 | 49 | 2019.4 | 27.5 | 0.16x |
| 1.5B | native mm4 | 49 | 1571.4 | 27.1 | 0.16x |
| 2.9B | none | 0 | 5622.4 | 110.7 | 1.00x |
| 2.9B | native mm8 | 65 | 3865.7 | 20.5 | 0.19x |
| 2.9B | native mm4 | 65 | 2985.7 | 19.5 | 0.18x |
| 7.2B | none | 0 | 13731.3 | 36.1 | 1.00x |
| 7.2B | native mm8 | 193 | 7340.5 | 17.0 | 0.47x |
| 7.2B | native mm4 | 193 | 4140.5 | 15.9 | 0.44x |
| 13.3B | none | 0 | 25309.1 | 10.2 | 1.00x |
| 13.3B | native mm8 | 367 | 13358.5 | 7.7 | 0.75x |
| 13.3B | native mm4 | 367 | 7374.5 | 8.6 | 0.84x |

The conclusion is deliberately conservative: native mm8/mm4 works on A800 and
reduces model footprint through 13.3B, but the current larger-row decode path is
not a speed path on this card. This does not promote quantized-speed defaults.

Diagnosis:

- With the default `min_params=8_000_000`, 1.5B quantizes 49 modules, 2.9B
  quantizes 65 modules, 7.2B quantizes 193 modules, and 13.3B quantizes 367
  modules. For larger rows this includes every per-layer FFN `key`/`value`
  matrix plus `lm_head`, not just the final vocabulary projection.
- The current native path is a Triton dequant-GEMV wrapper around each replaced
  `nn.Linear`. It reads packed weights and scale tensors, dequantizes in the
  kernel, and accumulates in fp32. It does not use an int8 tensor-core GEMM path
  and is not fused with the surrounding RWKV FFN/decode work.
- A800 fp16 cuBLAS is already very fast for these decode shapes. Real 2.9B
  weights measured `ffn.key (10240,2560)` at `0.0389 ms` fp16 versus
  `0.0980 ms` naive mm8 and `0.0687 ms` split-K mm8; `ffn.value (2560,10240)`
  at `0.0417 ms` fp16 versus `0.2829 ms` naive mm8 and `0.0696 ms` split-K
  mm8; `lm_head (65536,2560)` at `0.1996 ms` fp16 versus `0.2909 ms` naive
  mm8 and `0.2208 ms` split-K mm8. mm4 is slower still because nibble unpack
  and dequant overhead dominate.
- Raising the replacement gate to `min_params=50_000_000` leaves only
  `lm_head` quantized. That makes 1.5B roughly neutral (`166.9` fp16 tok/s,
  `171.7` mm8 tok/s, `163.9` mm4 tok/s) and 2.9B roughly neutral (`107.4`
  fp16 tok/s, `110.3` mm8 tok/s, `109.0` mm4 tok/s), but the footprint saving
  is much smaller.

So the slow larger-model rows are not a bnb artifact and not a broken benchmark.
They are real evidence that the current A800 native mm8/mm4 implementation is a
memory-saving compatibility path, not the final quantized-speed path. A speed
claim needs a native fused quant kernel that beats fp16 end to end, especially
for the FFN key/value decode shapes.

## 80GB VRAM matrix

| Model | Precision / quantization | Footprint MB | Peak VRAM MiB | Decode / generate tok/s | Status |
|---|---|---:|---:|---:|---|
| 0.4B | fp16 | 859.8 | 1147.5 | 185.2 decode | pass |
| 0.4B | bnb 8bit | 571.8 | 1200.9 | 11.5 decode | pass; slow path |
| 0.4B | bnb 4bit | 427.8 | 1116.4 | 23.4 decode | pass; slow path |
| 0.4B | native mm8 | 796.0 | 1781.7 | 187.9 decode | pass |
| 0.4B | native mm4 | 764.0 | 1789.8 | 185.8 decode | pass |
| 1.5B | fp16 | 2913.3 | 3197.4 | 172.7 decode | pass |
| 1.5B | bnb 8bit | 1761.3 | 3155.2 | 10.9 decode | pass; slow path |
| 1.5B | bnb 4bit | 1185.3 | 2882.5 | 22.7 decode | pass; slow path |
| 1.5B | native mm8 | 2019.4 | 3965.1 | 27.5 decode | pass; slow on current kernels |
| 1.5B | native mm4 | 1571.4 | 3581.1 | 27.1 decode | pass; slow on current kernels |
| 2.9B | fp16 | 5622.4 | 5911.2 | 110.7 decode | pass |
| 2.9B | bnb 8bit | 3222.4 | 5727.6 | 8.0 decode | pass; slow path |
| 2.9B | bnb 4bit | 2022.4 | 4250.6 | 16.7 decode | pass; slow path |
| 2.9B | native mm8 | 3865.7 | 6292.3 | 20.5 decode | pass; slow on current kernels |
| 2.9B | native mm4 | 2985.7 | 5998.7 | 19.5 decode | pass; slow on current kernels |
| 7.2B | fp16 | 13731.3 | 13830.8 | 36.1 decode | pass |
| 7.2B | fp16 smoke | 13731.3 | 13998.8 | 6.52 generate | pass |
| 7.2B | native mm8 | 7340.5 | 14588.9 | 17.0 decode | pass; slow on current kernels |
| 7.2B | native mm4 | 4140.5 | 14556.9 | 15.9 decode | pass; slow on current kernels |
| 13.3B | bnb 8bit | 13597.1 | 20108.6 | 3.9 decode | pass; slow path |
| 13.3B | bnb 4bit | 7741.1 | 18998.6 | 8.4 decode | pass; slow path |
| 13.3B | fp16 | 25309.1 | 25461.7 | 10.2 decode | pass |
| 13.3B | native mm8 | 13358.5 | 26167.1 | 7.7 decode | pass; slow on current kernels |
| 13.3B | native mm4 | 7374.5 | 26135.1 | 8.6 decode | pass; slow on current kernels |

## Large model smoke

| Model | Dtype | Load s | Forward s | Generate tok/s | Footprint MB | Peak VRAM MiB | Fast token backend |
|---|---|---:|---:|---:|---:|---:|---|
| 7.2B | fp16 | 95.715 | 3.9357 | 6.52 | 13731.3 | 13998.8 | native_graph |
| 13.3B | bnb 8bit | 264.568 | - | 3.9 decode | 13597.1 | 20108.6 | FLA fast-forward |
| 13.3B | bnb 4bit | 149.822 | - | 8.4 decode | 7741.1 | 18998.6 | FLA fast-forward |

## Training and ZeRO

| Model | Backend | World size | Dtype | Steps | Loss | Runtime s | Trainable delta |
|---|---|---:|---|---:|---:|---:|---:|
| 0.1B | Trainer | 1 | bf16 | 1 | 1.7344 | 329.7817 | 0.000100 |
| 0.1B | TRL SFT | 1 | bf16 | 1 | 1.8281 | 0.2033 | 0.000100 |
| 0.1B | TRL DPO | 1 | bf16 | 1 | 0.6914 | 1.3241 | 0.000100 |
| 0.1B | TRL GRPO | 1 | bf16 | 1 | 0.0000 | 19.7648 | 0.000100 |
| 0.4B | ZeRO-2 | 1 | bf16 | 1 | 1.9297 | 1.7344 | 0.000100 |
| 0.4B | ZeRO-3 | 1 | bf16 | 1 | 1.9297 | 1.5553 | 0.000100 |
| 0.4B | ZeRO-2 | 2 | bf16 | 1 | 5.1328 | 1.7852 | 0.000100 |
| 0.4B | ZeRO-3 | 2 | bf16 | 1 | 5.1328 | 1.9022 | 0.000100 |

Checkpoint resume rows:

| Model | ZeRO stage | World size | First steps | Resume steps | Global step | First loss | Resume loss | Resume delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4B | 2 | 1 | 1 | 2 | 2 | 1.9297 | 1.5781 | 0.0624 |
| 0.4B | 3 | 1 | 1 | 2 | 2 | 1.9297 | 1.5938 | 0.000100 |
| 0.4B | 2 | 2 | 1 | 2 | 2 | 5.1328 | 2.4336 | 0.0625 |
| 0.4B | 3 | 2 | 1 | 2 | 2 | 5.1328 | 2.4453 | 0.000100 |

Rank-0 rows are shown in the tables and appended to `bench/results.jsonl`.

## Cross-card comparison

| Card | Relevant committed evidence | A800 interpretation |
|---|---|---|
| V100 32GB | HF compatibility baseline, 0.4B/1.5B/2.9B training ecosystem, ZeRO resume rows, quantized memory rows | A800 extends the same HF/Trainer/TRL/ZeRO path to Ampere 80GB and adds larger 7.2B/13.3B memory evidence. |
| A100 40GB | A100 issue #68 covers 0.4B/1.5B/2.9B/7.2B smoke, batch, quant, Trainer/SFT/DPO, ZeRO base, and ZeRO2 resume | A800 is the same Ampere family but validates the 80GB card separately, including 13.3B quantized smoke and 13.3B native mm8/mm4 footprint rows. |
| RTX 4090 | Ada rows focus on fast native prefill and fused-kernel telemetry | A800 does not promote Ada prefill/projection defaults; its current contribution is Ampere compatibility, ZeRO, 80GB memory, and native quant telemetry. |
| H100 | No exact-card H100 rows in this repository yet | A800 should not be treated as Hopper validation; H100 remains a separate follow-up target. |

## Remaining A800 gaps

- Quantized speed remains unsolved on A800 until native/fused W8/W4 beats fp16
  end to end on larger rows.
- The current 80GB matrix has fp16 inference, bf16 training, bnb int8/int4, and
  native mm8/mm4 rows. A dedicated bf16 inference memory/speed sweep would still
  be useful before changing Ampere defaults.
- More batch/prefill and long-context rows would be useful before changing any
  Ampere defaults. The current policy remains conservative: output fusions stay
  allowed, prefill-scan/projection/LoRA/quant-speed fusions stay opt-in.
