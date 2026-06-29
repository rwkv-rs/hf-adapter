# Changelog

## 0.9.0

- Replaced the default runtime with a readable pure-PyTorch HF reference model.
- Added `RWKV7Cache` with canonical `[B,H,K,V]` state and no persisted
  `v_first`.
- Made the self-contained `reference` conversion layout the default.
- Added standard cache, padding, generation, loss, gradient-checkpointing and
  package-free AutoModel tests.
- Added pinned FLA comparison, formal lm_eval matrix tooling, and direct LoRA
  SFT/DPO/GRPO examples with local reproducibility artifacts.
- Preserved `NativeRWKV7*` names as compatibility aliases.
- Moved CUDA/JIT/graph/quantization/hardware-specialized work to
  `perf/native-kernels-v0.8`.

Older release history is retained in Git tags.
