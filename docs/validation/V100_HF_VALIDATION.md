# V100 HF validation matrix

Validation date: 2026-07-02; ZeRO3 resume addenda: 2026-07-03 and 2026-07-22;
performance addenda: 2026-07-10, 2026-07-11 and 2026-07-15
Base commit: `4528756` (`tests: record DeepSpeed ZeRO smoke passes (#64)`)
Server: `2 x Tesla V100-PCIE-32GB`
Main runtime: `torch 2.5.1+cu124`, `deepspeed 0.19.2`, `fused_recurrent` unless noted. ZeRO3 resume addendum runtime: `torch 2.8.0+cu126`, Transformers `4.57.1`, PEFT `0.19.1`, TRL `1.7.0`, DeepSpeed `0.19.2`.

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
- DeepSpeed ZeRO3 base training smoke passes, and the 2026-07-03 addendum validates ZeRO3 checkpoint-resume on 2×V100 with the 0.1B native/HF path. ZeRO2 resume is validated through 2.9B; ZeRO3 resume validated through 2.9B (0.1B/0.4B/1.5B/2.9B all pass on 2×V100 fp32 native/HF path, 2026-07-05).
- 13.3B is inference-only on a single V100-32GB: official alignment + decode speed are validated (see [13.3B inference validation](#133b-inference-validation)). Full training needs >32GB / multi-card / offload.

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
- 0.1B ZeRO3 resume on 2 x V100: pass, `global_step=2`, `resume_loss=2.542516`, `first_max_trainable_delta=9.999999e-05` on rank 0, `resume_max_trainable_delta=0.0719312`, fp32 native/HF path, `max_length=8` (`bench/results_v100_zero3_resume_2gpu_20260703.jsonl`, log `bench/v100_zero3_resume_2gpu_20260703.log`).
- 0.1B variable-rank-length resume on 2 x V100: ZeRO2 and ZeRO3 both pass,
  `global_step=2`, `first_loss=4.857278`, `resume_loss=2.093085`, and both
  trainable deltas are `9.9999997e-05` on both ranks. The two rank-local
  examples tokenize to 11 and 14 tokens with `max_length=16`, directly
  exercising global-length normalization rather than equal-length truncation.
  Evidence: `bench/results_v100_zero23_resume_variable_length_20260722.jsonl`.
  The post-fix V100 suite is `615 passed, 8 skipped`.

ZeRO3 resume addendum command:

```bash
RWKV7_NATIVE_MODEL=1 DS_IGNORE_CUDA_DETECTION=1 DS_BUILD_OPS=0 \
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/test_deepspeed_resume_smoke.py \
  --model /home/wzu/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --zero-stage 3 --attn-mode fused_recurrent --train-dtype fp32 \
  --first-steps 1 --resume-steps 2 --max-length 8 --batch-size 1 \
  --gradient-accumulation-steps 1 --dataset-repeats 2 \
  --results bench/results_v100_zero3_resume_2gpu_20260703.jsonl
```

The resume harness saves a full DeepSpeed checkpoint, then removes RNG state files to avoid the torch 2.5 `weights_only=True` numpy RNG incompatibility. It only bypasses the torch-load safety guard for checkpoints created locally inside this smoke.

### Native bitsandbytes quantized inference

| Model | 8bit footprint / peak VRAM | 4bit footprint / peak VRAM | Status |
|---|---:|---:|---:|
| 0.4B | 558.3 MB / 586.2 MB | 407.6 MB / 453.2 MB | pass/pass |
| 1.5B | 1713.5 MB / 1780.0 MB | 1113.7 MB / 1228.9 MB | pass/pass |
| 2.9B | 3132.7 MB / 3238.8 MB | 1887.8 MB / 2101.8 MB | pass/pass |
| 7.2B | 7380.0 MB / 7649.6 MB | 4204.4 MB / 4737.7 MB | pass/pass |

All quantized rows exercised HF `quantization_config`, native model load, forward, decode with cache, and `generate(max_new_tokens=2)` on V100.

### 13.3B inference validation

`rwkv7-g1g-13.3b-hf` (hidden=4096, 61 layers, head_dim=64, vocab=65536, `use_l2warp=true`) is inference-validated on a single Tesla V100-PCIE-32GB in fp16. Training is out of scope on one 32GB card — weights alone are ~25.6GB fp16.

Run against a clean checkout at `7cb1049` (#51); the server could not reach GitHub for the latest `main` during the run, and converted-weight correctness is independent of adapter commit. Env: `torch 2.5.1+cu124`, `fla 0.5.2`, `attn_mode=fused_recurrent`, `fuse_norm=false`.

Official alignment — `tests/test_official_alignment.py`, HF fp16 on GPU vs official `rwkv` `cpu fp32`:

| metric | result | target |
|---|---|---|
| cosine (5-prompt mean) | 0.9999976 | >=0.99 |
| top5_match | 1.0 | >=0.9 |
| argmax_match | 1.0 | — |
| max_abs_diff (worst prompt) | 0.0813 | <=0.15 |
| greedy window | 16/16 matched, 0 mismatch | 16 |

`use_l2warp=true` is baked into the converted weights by the converter (the adapter has no runtime l2warp code); HF output still matches the official model bit-for-bit (cos~1.0), confirming the conversion is correct.

Decode speed — `bench/bench_speed.py`, prompt=128, decode=64, fp16, single V100-32GB, `rwkv7_forward_token` + fast cache:

| fast-token backend | prefill tok/s | decode tok/s | decode ms/tok | peak VRAM |
|---|---:|---:|---:|---:|
| fla (fused_recurrent) | 893 | 11.6 | ~86 | — |
| native_jit | 907 | 18.4 | ~54 | — |
| native_graph | 893 | 17.1 | 58.4 | 25594 MB |

`native_jit` is the best 13.3B decode backend (1.58x over fla). `native_graph` fits the 32GB card (25.6GB peak) but is slightly slower than `native_jit` at this scale: 13.3B decode is memory-bound (~54 ms/tok reading ~27GB of weights), so the graph launch-overhead win that dominates small launch-bound models inverts here under graph-replay overhead. This is why the larger-model smoke defaults 13.3B to `native_jit`.

## Harness changes made for V100-scale validation

- `tests/test_native_trainer_resume_smoke.py`: release the first Trainer/model and clear CUDA cache before loading the resumed model. This avoids holding two fp32 2.9B models on one 32GB V100.
- `tests/test_native_peft_save_load_merge.py`: keep only one full base model resident at a time between train, reload, merge, and generation checks. This allows 2.9B fp32 PEFT save/load/merge on V100.
- `tests/test_deepspeed_resume_smoke.py`: new DeepSpeed ZeRO checkpoint-resume smoke. It validates checkpoint creation, fresh model load, resume to the target global step, finite loss, and trainable LoRA updates.
- Native ZeRO3 hook fix: the batched native loop now calls attention/FFN modules through `Module.__call__`, so DeepSpeed ZeRO3 pre-forward hooks gather raw TMix/CMix parameters (`x_r`, `r_k`, `g_norm.weight`, `ffn.x_k`) before backward.

## Remaining V100-bounded gaps

- 7.2B Trainer/SFT/DPO/GRPO on a single V100 32GB is memory-bound; use larger GPU or more aggressive offload for full training proof.
- ZeRO3 resume is validated through 2.9B on 2×V100; larger-model resume and
  corresponding A100 large-model coverage remain open.
- Native W8/W4 `speed` policy passes the promoted card-local B1/B2/B4/B8
  matrix, but full-memory quantization with the larger footprint reduction is
  still not universally fp16-or-faster.

## 2026-07-10 fused-decode addendum

The Volta-native graph path now combines norm/mix kernels, shape-routed sm70
projection/FFN kernels, and raw recurrent-output preparation. On the idle
V100 GPU1, matching-checkpoint FP16 decode passes P1 (`>=0.55x` Albatross) for
all 0.1B/0.4B/1.5B and bsz=1/2/4/8 rows. All three bsz=8 rows pass P3; 0.4B
and 1.5B bsz=8 exceed the measured Albatross throughput. The 0.4B and 1.5B
raw-recurrent A/B windows retain 32-step greedy equality. See the raw evidence
and ratio table in
[`bench/v100_sm70_decode_gap_20260710/README.md`](../../bench/v100_sm70_decode_gap_20260710/README.md).

This addendum closed the V100 decode P1 floor at the time. The subsequent
2026-07-11 production-close artifact closes the selected native W8/W4 speed
lane; cross-card promotion, full-memory quant and universal P3 parity remain
separate requirements.

## 2026-07-15 full-FLA Qwen3.5 active-parameter addendum

The current optimized-reference comparison uses RWKV-7 g1g 1.5B and official
Qwen3.5-2B on one Tesla V100-PCIE-32GB (`sm_70`), dense fp16, prompt 512,
decode 64 and batch 1/8. It is target-only: no draft model, speculative
acceptance, prefix-state reuse or hidden cache warm-start.

Qwen fails closed unless all 18 linear-attention layers bind FLA chunk prefill,
FLA fused-recurrent decode, fused gated RMS norm and the repository Triton
causal-convolution prefill/update path. Both rows report
`qwen_full_fused_contract_pass=true` and effective backend
`qwen_fla_gated_delta_rule_fla_triton_conv`.

The acceptance metric is `aggregate tok/s * active text parameters`. Active
counts are 1,527,404,544 for RWKV and 1,881,825,088 for Qwen (`0.811661x`), so
the raw break-even ratio is `1.232041x`.

| Bsz | Prefill RWKV/Qwen | Prefill active work | Decode RWKV/Qwen | Decode active work | Peak VRAM RWKV/Qwen |
|---:|---:|---:|---:|---:|---:|
| 1 | `2.815921x` | `2.285574x` | `5.913307x` | `4.799514x` | `1.024885x` |
| 8 | `5.407762x` | `4.389270x` | `5.270432x` | `4.277804x` | `0.837248x` |

All four raw and normalized phase gates pass. Qwen full-FLA/Triton-conv versus
its oracle and RWKV native graph versus its FLA-backed HF route each preserve
32/32 greedy tokens and pass cosine gates. Static RWKV footprint is
`0.811662x` Qwen, but the B1 peak-VRAM ratio is explicitly a loss and must not
be presented as a universal memory win.

Canonical evidence and reproduction commands:
[`bench/v100_active_b1b8_20260715/README.md`](../../bench/v100_active_b1b8_20260715/README.md).
