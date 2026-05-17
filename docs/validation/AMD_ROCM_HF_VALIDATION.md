# AMD ROCm HF validation

This document records the AMD compatibility matrix and exact-card decode and
output-head W8/W4 promotions for the fully native HF, FLA-free RWKV-7 adapter.
It is a **validated compatibility and named-lane gfx1100 performance** result.
It is not an Albatross-parity or full-model quantized-speed claim.

## Validated system (2026-07-28)

- GPU: AMD Navi 31 / `gfx1100`, PCI device `0x744b`, 47.98 GiB VRAM. The cloud
  image reports the generic marketing name `AMD Radeon Graphics`.
- OS: Ubuntu 24.04, Linux 6.8.
- ROCm: 7.2.1.
- PyTorch: `2.9.1+rocm7.2.1.gitff65f5bc`.
- Triton: `3.5.1+rocm7.2.1.gita272dfa8`.
- Transformers: 5.12.1; PEFT 0.19.1; TRL 1.9.1.
- Models: converted RWKV-7 G1D 0.1B/0.4B and G1H
  1.5B/2.9B/7.2B/13.3B HF checkpoints.
- Adapter: the native model module split plus exact-GCN-architecture policy on
  branch `wangyue/amd-full-production-close`.

The converter/sync tool writes direct native remote-code metadata:
`native_model.NativeRWKV7ForCausalLM`. The validation runner fails if the
legacy `modeling_rwkv7.py` FLA wrapper remains in the converted model. Runtime
policy separately classifies the live HIP device as `amd_hip` and preserves
`gcnArchName=gfx1100`. Four existing Triton decode fusions and measured
output-head MM8/MM4 launch profiles are enabled only for that exact
architecture. Unmeasured AMD architectures, fused prefill and full-model quant
speed paths remain off. PyTorch's CUDA-compatible graph API is available under
ROCm, and generation selected the repository `native_graph` route successfully
on this system.

The public native entry point remains small. Cache, generation, fast/chunked
prefill APIs, layers/backbone, prefill graph, quantization and speculative
generation live in their decoupled `model_*.py` modules. AMD support does not
add card-specific methods back into `native_model.py` or modify the legacy FLA
wrapper.

## Passing compatibility gates

- `AutoTokenizer` and `AutoModelForCausalLM(..., trust_remote_code=True)`.
- fp16 forward and greedy `generate(use_cache=True)`.
- HF API contract, beam cache reorder and fixed-vocabulary behavior.
- PEFT LoRA forward/loss/backward: 144 non-zero adapter-gradient tensors in
  the final 0.4B replay.
- Dynamic batching, select/reorder/in-place reorder/compact and CPU
  offload/restore of `NativeRWKV7Cache`.
- Native chunked prefill through `model_fast_api.py`. Full and chunked prompt state
  lengths match; fp16 next-decode max absolute difference was at most `0.0625`
  in the tested matrix.
- HF Trainer + LoRA in bf16: six steps passed, all 144 trainable tensors
  updated, and sampled loss moved from `3.0271` to `0.9417` (minimum `0.7272`)
  in the final post-rebase run.

On this ROCm/PyTorch combination, the fp16 Trainer smoke produced a non-finite
gradient norm and no optimizer update. Use bf16 (preferred) or fp32 for AMD
training until a separately gated fp16 mixed-precision recipe is added. fp16
inference and the direct PEFT backward smoke pass.

## Initial conservative 0.1B baseline

Prompt 128, decode 32, one warmup, two measured prefill runs:

| dtype | batch | prefill tok/s total | decode tok/s total | peak VRAM MiB |
|---|---:|---:|---:|---:|
| fp16 | 1 | 593.4 | 184.1 | 542.3 |
| fp16 | 2 | 1057.1 | 344.4 | 582.1 |
| fp16 | 4 | 1950.2 | 683.7 | 630.2 |
| fp16 | 8 | 3861.4 | 1304.2 | 712.1 |

These are fully native HF baseline rows. No same-card Albatross result exists
yet, so they are not an Albatross-parity claim. Card-local fused decode and
output-head quant tuning are now validated through 13.3B; a pinned same-card
official/Albatross reference remains separate.

## Exact-gfx1100 decode promotion (2026-07-28)

The recurrent-output, raw recurrent, output-preparation and norm/shift-mix
Triton paths all compiled and passed isolated B1/B8 A/B tests. The combined
policy was then compared against a graph with all four fusions disabled:

| model | batch | conservative tok/s | gfx1100 policy tok/s | speedup | greedy |
|---|---:|---:|---:|---:|---:|
| 0.1B | 1 | 184.7 | 347.1 | 1.8795x | 32/32 |
| 0.1B | 8 | 1307.0 | 2666.5 | 2.0402x | 256/256 |
| 0.4B | 1 | 81.2 | 141.8 | 1.7458x | 32/32 |
| 0.4B | 8 | 615.8 | 1073.2 | 1.7428x | 256/256 |
| 1.5B | 1 | 51.0 | 71.3 | 1.3987x | 32/32 |
| 1.5B | 8 | 350.1 | 514.2 | 1.4685x | 256/256 |
| 2.9B | 1 | 35.0 | 47.7 | 1.3658x | 32/32 |
| 2.9B | 8 | 250.3 | 353.0 | 1.4102x | 256/256 |
| 7.2B | 1 | 24.1 | 29.7 | 1.2291x | 16/16 |
| 7.2B | 8 | 165.6 | 213.9 | 1.2911x | 128/128 |
| 13.3B | 1 | 12.8 | 15.5 | 1.2131x | 8/8 |
| 13.3B | 8 | 87.7 | 113.2 | 1.2909x | 64/64 |

All rows stayed on `native_graph`; minimum cosine was at least `1.0` within
floating-point reporting and maximum first-step logit difference was `0.0625`.
The policy uses four warps for norm/mix. Generic Radeon names are not trusted:
the runtime reads `gcnArchName`, strips feature suffixes, and fails closed unless
it is exactly `gfx1100`.

For fp16 prompt 256, chunk sizes 32/64/128 passed cache handoff with final-logit
and next-decode maximum absolute difference `0.0625`, minimum cosine
`0.99999994`, and exact greedy agreement. The cold full-prefill row was 209.3
tok/s; chunked rows were 488.9-563.0 tok/s. Because this separate run used no
warmup, its ratio is correctness/cold-start telemetry rather than a promoted
speed comparison.

Raw logs, JSONL and checksums:
[`bench/amd_gfx1100_full_close_20260730/`](../../bench/amd_gfx1100_full_close_20260730/README.md).
The fused-decode promotion evidence is in
[`bench/amd_gfx1100_full_close_20260730/`](../../bench/amd_gfx1100_full_close_20260730/README.md).

## Exact-gfx1100 output-head W8/W4 decode

The native MM8/MM4 dispatcher now uses measured gfx1100 batched-dot launches
for B2-B8 while retaining the fused GEMV route at B1. Across 0.4B, 1.5B,
2.9B, 7.2B and 13.3B, all 40 output-head rows at B1/B2/B4/B8:

- reduce model footprint;
- beat the paired fp16 cached-decode baseline;
- preserve the tested greedy stream;
- remain on `native_graph`.

The final post-rebase decode speedup ranges were `1.0797x-1.0922x` on 0.4B,
`1.0384x-1.0491x` on 1.5B, `1.0226x-1.0273x` on 2.9B,
`1.0272x-1.0362x` on 7.2B, and `1.0123x-1.0191x` on 13.3B. This is deliberately
an output-head speed policy. Full-model memory quantization remains open: the
2.9B 8M-threshold W8 row fell to `0.6712x` fp16 decode at B8, while W4 also
missed the intended quality gate. Raw rows, launch signatures and the negative
FFN sweep are in
[`bench/amd_gfx1100_quant_20260728/`](../../bench/amd_gfx1100_quant_20260728/README.md).
The complete post-rebase native-HF replay, 40-cell matrix, large-checkpoint
conversion provenance and checksums are in
[`bench/amd_gfx1100_full_close_20260730/`](../../bench/amd_gfx1100_full_close_20260730/README.md).

## Reproduce

Install the matching ROCm PyTorch wheel, ensure the user can open `/dev/kfd`
and `/dev/dri/render*`, then run:

```bash
source /workspace/rwkv-rocm/bin/activate
cd /workspace/rwkv7-hf-adapter
bash bench/run_amd_rocm_hf_validation.sh \
  HF_DIR=/workspace/models/rwkv7-g1d-0.1b-hf
```

If PyTorch reports `No HIP GPUs are available` while `rocminfo` works, add the
runtime user to the numeric groups owning `/dev/kfd` and `/dev/dri/render*`,
then start a new login session.

The larger-model validation set is pinned by source size and SHA256 and can be
prepared sequentially without retaining duplicate `.pth` files:

```bash
python scripts/prepare_rwkv7_g1_validation_models.py \
  --models all \
  --checkpoint-dir /workspace/checkpoints \
  --output-root /workspace/models \
  --vocab-file /workspace/models/rwkv7-g1d-0.1b-hf/rwkv_vocab_v20230424.txt
```

## Open AMD work

1. Add repeated same-card Albatross or official RWKV-LM comparisons.
2. Tune and gate a production HIP/ROCm prefill path; decode has an exact-gfx1100
   promotion but no family-wide AMD claim.
3. Fuse quantized FFN/block execution so full-model W8/W4 can retain its memory
   reduction without losing all-phase speed; close the larger-model W4 quality
   gate independently.
4. Validate MI-series and other AMD architectures independently; they must not
   inherit the exact-`gfx1100` launch policy.
5. Add longer bf16 training, TRL and distributed ROCm evidence.
