# AMD ROCm HF validation

This document records the first real AMD compatibility matrix for the fully
native HF, FLA-free RWKV-7 adapter. It is a **validated compatibility and
baseline-performance** result, not an Albatross-parity or quantized-speed claim.

## Validated system (2026-07-27)

- GPU: AMD Navi 31 / `gfx1100`, PCI device `0x744b`, 47.98 GiB VRAM. The cloud
  image reports the generic marketing name `AMD Radeon Graphics`.
- OS: Ubuntu 24.04, Linux 6.8.
- ROCm: 7.2.1.
- PyTorch: `2.9.1+rocm7.2.1.gitff65f5bc`.
- Triton: `3.5.1+rocm7.2.1.gita272dfa8`.
- Transformers: 5.12.1; PEFT 0.19.1; TRL 1.9.1.
- Model: converted RWKV-7 g1d 0.1B HF checkpoint.
- Adapter baseline: main `6f7737f68e01a9a2a587d0b7a59a8719ecd68084`,
  including the native model module split.

The converter/sync tool writes direct native remote-code metadata:
`native_model.NativeRWKV7ForCausalLM`. The validation runner fails if the
legacy `modeling_rwkv7.py` FLA wrapper remains in the converted model. Runtime
policy separately classifies the live HIP device as `amd_hip`. CUDA-only
compiled extensions, fused kernels and quant speed paths remain off. PyTorch's
CUDA-compatible graph API is available under ROCm, and generation selected the
repository `native_graph` route successfully on this system.

The public native entry point remains small. Cache, generation, fast/chunked
prefill APIs, layers/backbone, prefill graph, quantization and speculative
generation live in their decoupled `model_*.py` modules. AMD support does not
add card-specific methods back into `native_model.py` or modify the legacy FLA
wrapper.

## Passing compatibility gates

- `AutoTokenizer` and `AutoModelForCausalLM(..., trust_remote_code=True)`.
- fp16 forward and greedy `generate(use_cache=True)`.
- HF API contract, beam cache reorder and fixed-vocabulary behavior.
- PEFT LoRA forward/loss/backward: 72 non-zero adapter-gradient tensors.
- Dynamic batching, select/reorder/in-place reorder/compact and CPU
  offload/restore of `NativeRWKV7Cache`.
- Native chunked prefill through `model_fast_api.py`. Full and chunked prompt state
  lengths match; fp16 next-decode max absolute difference was at most `0.0625`
  in the tested matrix.
- HF Trainer + LoRA in bf16: six steps passed, all 72 trainable tensors updated,
  and sampled loss moved from `3.8036` to `1.1910` (minimum `0.4411`) in the
  canonical run.

On this ROCm/PyTorch combination, the fp16 Trainer smoke produced a non-finite
gradient norm and no optimizer update. Use bf16 (preferred) or fp32 for AMD
training until a separately gated fp16 mixed-precision recipe is added. fp16
inference and the direct PEFT backward smoke pass.

## Initial 0.1B baseline

Prompt 128, decode 32, one warmup, two measured prefill runs:

| dtype | batch | prefill tok/s total | decode tok/s total | peak VRAM MiB |
|---|---:|---:|---:|---:|
| fp16 | 1 | 593.4 | 184.1 | 542.3 |
| fp16 | 2 | 1057.1 | 344.4 | 582.1 |
| fp16 | 4 | 1950.2 | 683.7 | 630.2 |
| fp16 | 8 | 3861.4 | 1304.2 | 712.1 |

These are fully native HF baseline rows. No same-card Albatross result exists
yet, so they are not a production-performance claim. Production AMD promotion
still requires a pinned same-card reference, larger models and card-local
tuning.

For fp16 prompt 256, chunk sizes 32/64/128 passed cache handoff with final-logit
and next-decode maximum absolute difference `0.0625`, minimum cosine
`0.99999994`, and exact greedy agreement. The cold full-prefill row was 209.3
tok/s; chunked rows were 488.9-563.0 tok/s. Because this separate run used no
warmup, its ratio is correctness/cold-start telemetry rather than a promoted
speed comparison.

Raw logs, JSONL and checksums:
[`bench/amd_gfx1100_native_20260727/`](../../bench/amd_gfx1100_native_20260727/README.md).

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

## Open AMD work

1. Add repeated same-card Albatross or official RWKV-LM comparisons.
2. Compare the current graph prefill/decode route against a production HIP/ROCm
   fused path and promote only the same-card winner.
3. Add HIP-native W8/W4 kernels with footprint, speed and quality gates.
4. Validate larger models and MI-series cards.
5. Add longer bf16 training, TRL and distributed ROCm evidence.
