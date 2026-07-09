# Changelog

## 1.0.0

- Limited `rwkv7_hf/` to canonical HF model modules named `*_rwkv7.py`.
- Moved CLI, conversion, manifest, and smoke-test code into the sibling
  `rwkv7_hf_tools/` package.
- Removed the `NativeRWKV7*` and `RWKV7StateCache` aliases and their legacy
  compatibility modules.
- Removed package-backed `thin` conversion and deprecated backend/fusion CLI
  flags; conversion now always writes one self-contained reference layout.
- Reduced the installed console interface to `rwkv7-hf convert` and
  `rwkv7-hf smoke`.

## 0.9.0

- Replaced the default runtime with a readable pure-PyTorch HF reference model.
- Added `RWKV7Cache` with canonical `[B,H,K,V]` state and no persisted
  `v_first`.
- Made the self-contained `reference` conversion layout the default.
- Added standard cache, padding, generation, loss, gradient-checkpointing and
  package-free AutoModel tests.
- Added non-blocking FLA backend diagnostics, formal lm_eval matrix tooling, and direct LoRA
  SFT/DPO/GRPO examples with local reproducibility artifacts.
- Preserved `NativeRWKV7*` names as compatibility aliases.
- Moved CUDA/JIT/graph/quantization/hardware-specialized work to
  `perf/native-kernels-v0.8`.

Older release history is retained in Git tags.
