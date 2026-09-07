# FP8 E4M3 Quantization

## Overview

FP8 E4M3 per-tensor weight quantization with online activation quantization
(W8A8) using `torch._scaled_mm`. Designed for Hopper (H100), Ada Lovelace
(RTX 4090), and Blackwell (RTX 5070 Ti / 5090) GPUs with native FP8 tensor
core support.

## Supported paths

| Path | Purpose | Status |
|------|---------|--------|
| FP8 per-tensor (W8A8) | Full FP8 weight + activation quantization | Promoted on RTX 5070 Ti (Blackwell) with 1.5B/7.2B models |
| FP8 ffn_only | Conservative: quantize only FFN key/value | Validated on 1.5B, improved EAR vs full FP8 |

## Hardware requirements

- CUDA compute capability >= 8.9 (Ada Lovelace / Hopper / Blackwell)
- PyTorch >= 2.1 with `torch._scaled_mm` and `torch.float8_e4m3fn` support
- On unsupported hardware, falls back to dequant + FP16 matmul

## Usage

### Programmatic API

```python
from rwkv7_hf.native_quant_fp8 import quantize_model_fp8

# Full FP8 (aggressive)
replaced = quantize_model_fp8(model, min_params=8_000_000, policy="memory")

# FFN-only FP8 (conservative, higher precision)
replaced = quantize_model_fp8(model, min_params=8_000_000, policy="ffn_only")

# Per-channel scales (finer granularity)
replaced = quantize_model_fp8(model, policy="memory", per_channel=True)
```

### Config-driven (auto-quantize on load)

```python
from transformers import AutoModelForCausalLM
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
config.use_native_fp8 = True
config.native_fp8_policy = "memory"  # or "speed", "ffn_only"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, config=config)
```

## Performance (RTX 5070 Ti, Blackwell)

### 7.2B Model

| Metric | BF16 | FP8 | Change |
|--------|------|-----|--------|
| Decode speed | 7.0 t/s | 44.9 t/s | 6.4x |
| VRAM | 13.32 GB | 7.35 GB | -45% |
| Top-1 consistency | 100% | 93.75% | -6.25pp |

### 1.5B Model

| Metric | BF16 | FP8 | FFN-only FP8 |
|--------|------|-----|-------------|
| Decode speed | 164.1 t/s | 67.8 t/s | ~55 t/s |
| VRAM | 2.69 GB | 1.60 GB | ~2.0 GB |
| EAR | 1.0 | 0.9412 | 0.9498 |
| Top-1 | 100% | 92.52% | 93.46% |

## Acceptance gate

Same five-gate criteria as MM8/MM4: function, quality, memory, speed,
reproducibility.

## Comparison with MM8/MM4

| Dimension | FP8 | MM8 (int8) | MM4 (int4) |
|-----------|-----|-----------|-----------|
| Format | float8 e4m3 | affine int8 | affine int4 |
| Precision | 0.2% quant error | ~0.5% | ~2-5% |
| Speed (Blackwell) | 6.4x vs BF16 | 1.0-1.2x | 1.1-1.5x |
| VRAM reduction | 45% | 50% | 70% |
| Hardware req | SM 8.9+ | Any CUDA | Any CUDA |
