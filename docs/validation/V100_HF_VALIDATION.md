# V100 HF validation matrix

Validation date: 2026-07-02  
Base commit: `4528756` (`tests: record DeepSpeed ZeRO smoke passes (#64)`)  
Server: `2 x Tesla V100-PCIE-32GB`  
Main runtime: `torch 2.5.1+cu124`, `deepspeed 0.19.2`, `fused_recurrent` unless noted.

This file records the additional V100 validation pass for the HF-only RWKV-7 adapter work.  The goal was to close the remaining HF ecosystem evidence gap with the hardware currently available.

## Summary

| Model | Trainer/LoRA | SFT | DPO | GRPO | Trainer resume | PEFT save/load/merge | ZeRO2 | ZeRO2 resume | ZeRO3 | Quant 8bit/4bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |
| 1.5B | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass/pass |
| 2.9B | pass | pass native | pass native | pass native | pass | pass | not required for base smoke | pass | pass | pass/pass |
| 7.2B | pass PEFT | V100 limit | V100 limit | V100 limit | V100 limit | reload/generate pass, fp16 merge strict diff open | not run | not run | not run | pass/pass |

Notes:

- 2.9B FLA-backed SFT on a V100 hit the current FLA/Triton path limits (`fp32` autotune OOM; `fp16` produced finite loss but no LoRA update). The native/no-FLA SFT/DPO/GRPO path passed and is the correct HF compatibility route for this size on V100.
- 7.2B native Trainer resume in fp16 hit V100 32GB memory limits. Manual PEFT LoRA training still works, and quantized inference works.
- DeepSpeed ZeRO3 base training smoke passes, but ZeRO3 checkpoint-resume still needs a dedicated fix for DeepSpeed parameter-partition re-entry around fresh model construction. ZeRO2 resume is now validated through 2.9B.

## New V100 passes from this run

### Checkpoint resume

- 0.4B native Trainer resume: pass (`first_steps=1`, `resume_steps=2`, `length=32`, fp32).
- 1.5B native Trainer resume: pass (`first_steps=1`, `resume_steps=2`, `length=16`, fp32).
- 2.9B native Trainer resume: pass after releasing the first model before loading the resumed model (`first_steps=1`, `resume_steps=2`, `length=8`, fp32).

### Native TRL/RL at 2.9B

- 2.9B native SFT: pass, `train_loss=2.6676`, `max_trainable_delta=0.000100`.
- 2.9B native DPO: pass, `train_loss=0.6931`, `max_trainable_delta=0.000100`.
- 2.9B native GRPO: pass, `train_loss=0.0000`, `max_trainable_delta=0.000019`.

### PEFT save/load/merge

- 0.4B fp32: pass, `reload_diff=0.0`, `merge_diff=0.00008774`.
- 1.5B fp32: pass, `reload_diff=0.0`, `merge_diff=0.00005722`.
- 2.9B fp32: pass, `reload_diff=0.0`, `merge_diff=0.00003815`.
- 7.2B fp16: adapter reload and generation passed (`reload_diff=0.0`, generated tail ok), but strict merge equivalence remains open (`merge_diff=0.125` with fp16 merge).

### DeepSpeed resume

- 0.4B ZeRO2 resume on 2 x V100: pass, `global_step=2`, `resume_loss=2.416867`.
- 1.5B ZeRO2 resume on 2 x V100: pass, `global_step=2`, `resume_loss=2.682713`.
- 2.9B ZeRO2 resume on 2 x V100: pass, `global_step=2`, `resume_loss=2.671991`.

The resume harness saves a full DeepSpeed checkpoint, then removes RNG state files to avoid the torch 2.5 `weights_only=True` numpy RNG incompatibility. It only bypasses the torch-load safety guard for checkpoints created locally inside this smoke.

### Native bitsandbytes quantized inference

| Model | 8bit footprint / peak VRAM | 4bit footprint / peak VRAM | Status |
|---|---:|---:|---:|
| 0.4B | 558.3 MB / 586.2 MB | 407.6 MB / 453.2 MB | pass/pass |
| 1.5B | 1713.5 MB / 1780.0 MB | 1113.7 MB / 1228.9 MB | pass/pass |
| 2.9B | 3132.7 MB / 3238.8 MB | 1887.8 MB / 2101.8 MB | pass/pass |
| 7.2B | 7380.0 MB / 7649.6 MB | 4204.4 MB / 4737.7 MB | pass/pass |

All quantized rows exercised HF `quantization_config`, native model load, forward, decode with cache, and `generate(max_new_tokens=2)` on V100.

## Harness changes made for V100-scale validation

- `tests/test_native_trainer_resume_smoke.py`: release the first Trainer/model and clear CUDA cache before loading the resumed model. This avoids holding two fp32 2.9B models on one 32GB V100.
- `tests/test_native_peft_save_load_merge.py`: keep only one full base model resident at a time between train, reload, merge, and generation checks. This allows 2.9B fp32 PEFT save/load/merge on V100.
- `tests/test_deepspeed_resume_smoke.py`: new DeepSpeed ZeRO checkpoint-resume smoke. It validates checkpoint creation, fresh model load, resume to the target global step, finite loss, and trainable LoRA updates.

## Remaining V100-bounded gaps

- 7.2B Trainer/SFT/DPO/GRPO on a single V100 32GB is memory-bound; use larger GPU or more aggressive offload for full training proof.
- ZeRO3 resume needs a follow-up harness fix around DeepSpeed parameter partition re-entry. ZeRO3 base training smoke is already passing.
- Quantized inference is functionally validated with lower VRAM. Quantized speed is still not the final Albatross-level performance target; fused quant kernels are still required.
