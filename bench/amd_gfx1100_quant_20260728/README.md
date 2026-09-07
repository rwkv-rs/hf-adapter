# AMD gfx1100 native W8/W4 validation (2026-07-28)

This directory records the exact-card tuning and fail-closed promotion of the
native MM8/MM4 output-head decode route on AMD `gfx1100`. It also records the
remaining full-model quantization gap. The promoted result is a **decode speed
lane**, not a claim that every quantized projection is faster than fp16.

## System and method

- AMD Navi 31 / `gfx1100`, 47.98 GiB VRAM.
- ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, Triton 3.5.1.
- Converted RWKV-7 G1D 0.4B and G1H 1.5B, fp16, `native_graph`.
- Prompt 128, decode 128, three timing repeats.
- Every quant row measured its dense baseline in the same fresh process.
- `speed` replaces only `lm_head`; `memory` replaces every eligible Linear at
  or above the stated parameter threshold.

The dispatch key is the runtime `gcnArchName`, not the generic marketing name
`AMD Radeon Graphics`. Only exact `gfx1100` receives these launch parameters:

| format | rows | dot threshold | tile | warps |
|---|---:|---:|---|---:|
| MM8 | <=16 | >=2 | B16 x M256 x N16 | 8 |
| MM4 | <=16 | >=2 | B16 x P64 x N32 | 2 |

Other HIP architectures fail closed and cannot inherit NVIDIA `sm_120` policy
from ROCm's CUDA-compatible capability API.

## Promoted output-head decode lane

### 0.4B

| format | batch | footprint / fp16 | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.9258 | 0.9938x | **1.0888x** | 0.99999517 | pass |
| MM4 | 1 | 0.8886 | 1.0042x | **1.0827x** | 0.99975604 | pass |
| MM8 | 2 | 0.9258 | 0.9947x | **1.0846x** | 0.99998975 | pass |
| MM4 | 2 | 0.8886 | 1.0006x | **1.0895x** | 0.99975157 | pass |
| MM8 | 4 | 0.9258 | 0.9965x | **1.0887x** | 0.99997991 | pass |
| MM4 | 4 | 0.8886 | 0.9995x | **1.0902x** | 0.99974054 | pass |
| MM8 | 8 | 0.9258 | 0.9919x* | **1.0803x** | 0.99998951 | pass |
| MM4 | 8 | 0.8886 | 1.0002x | **1.0843x** | 0.99975175 | pass |

`*` The first matrix process recorded a cold/noisy MM8 B8 prefill ratio of
`0.9408x`; an isolated same-process repeat recorded `0.9919x` while decode was
stable (`1.0808x` and `1.0803x`). The promotion gate is cached decode.

### 1.5B

| format | batch | footprint / fp16 | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.9562 | 1.0060x | **1.0487x** | 0.99999452 | pass |
| MM4 | 1 | 0.9342 | 0.9963x | **1.0463x** | 0.99979413 | pass |
| MM8 | 2 | 0.9562 | 0.9957x | **1.0443x** | 0.99999350 | pass |
| MM4 | 2 | 0.9342 | 1.0052x | **1.0442x** | 0.99979329 | pass |
| MM8 | 4 | 0.9562 | 0.9970x | **1.0443x** | 0.99999052 | pass |
| MM4 | 4 | 0.9342 | 1.0062x | **1.0434x** | 0.99979091 | pass |
| MM8 | 8 | 0.9562 | 1.0009x | **1.0396x** | 0.99997830 | pass |
| MM4 | 8 | 0.9342 | 1.0112x | **1.0423x** | 0.99977881 | pass |

### 2.9B

| format | batch | footprint / fp16 | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.9716 | 1.0020x | **1.0252x** | 0.99999583 | pass |
| MM4 | 1 | 0.9573 | 1.0000x | **1.0210x** | 0.99985766 | pass |
| MM8 | 2 | 0.9716 | 0.9991x | **1.0216x** | 0.99999911 | pass |
| MM4 | 2 | 0.9573 | 0.9983x | **1.0216x** | 0.99986076 | pass |
| MM8 | 4 | 0.9716 | 0.9996x | **1.0224x** | 0.99999475 | pass |
| MM4 | 4 | 0.9573 | 0.9924x | **1.0240x** | 0.99985695 | pass |
| MM8 | 8 | 0.9716 | 1.0055x | **1.0261x** | 0.99996513 | pass |
| MM4 | 8 | 0.9573 | 1.0010x | **1.0258x** | 0.99982536 | pass |

All 24 promoted rows lower the model footprint, beat dense cached decode, keep
the tested greedy stream exactly, and retain high logit cosine. The 0.4B old
dispatcher had fallen to `0.8006x` MM8 and `0.6970x` MM4 at B8; the exact-card
batched dot route closes that dispatch gap.

## Full-model memory lane remains open

The initial 0.4B all-Linear baseline quantized 335 modules:

| format | batch | footprint / fp16 | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.5782 | 0.9435x | 0.8352x | 0.99998629 | pass |
| MM4 | 1 | 0.3657 | 0.8010x | 0.5895x | 0.99785161 | pass |
| MM8 | 8 | 0.5782 | 0.9358x | 0.3122x | 0.99998021 | pass |
| MM4 | 8 | 0.3657 | 0.8068x | 0.2161x | 0.99784702 | pass |

At 1.5B, quantizing the 49 linears with at least 8M parameters gives:

| format | batch | footprint / fp16 | prefill / fp16 | decode / fp16 | final cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 | 1 | 0.6932 | 0.9563x | 1.1178x | 0.99998248 | pass |
| MM4 | 1 | 0.5394 | 0.9551x | 0.9299x | 0.99789113 | pass |
| MM8 | 8 | 0.6932 | 0.9765x | 0.7429x | 0.99996948 | pass |
| MM4 | 8 | 0.5394 | 0.9540x | 0.8836x | 0.99788284 | pass |

At 2.9B, the same threshold quantizes 65 linears. MM8 remains accurate but
falls to `0.6712x` dense decode at B8. MM4 reaches a `0.5310x` footprint but
only `0.8950x/0.8272x` B1/B8 decode; final-logit cosine is about `0.9842` and
the tested greedy stream diverges. This larger-model row therefore fails both
the speed and the intended quality gate and is retained as diagnosis only.

Footprint and quality pass, but all-phase speed does not. Threshold sweeps in
`hybrid_threshold_0p4b.jsonl` show the same trade-off rather than a hidden
module-count sweet spot.

`ffn_mm4_sweep_0p4b.jsonl` explains why tile-only tuning is insufficient. The
best isolated MM4 kernel was still slower than rocBLAS dense for each tested
FFN projection:

| projection | B1 best / dense | B8 best / dense |
|---|---:|---:|
| `ffn.key` | 0.6717x | 0.4280x |
| `ffn.value` | 0.7261x | 0.3478x |

Hundreds of separate unpack/dequant launches dominate these small projections.
The remaining production route is a fused quantized FFN/block kernel (for
example key -> ReLU2 -> value) or a measured shape policy that retains those
FFNs dense. Blindly enabling every quantized Linear is explicitly not promoted.

## Evidence files

- `baseline_0p4b.jsonl`: pre-tuning dense/quant paired baseline.
- `head_kernel_sweep_0p4b.json`: exact launch sweep for the 65,536 x 1,024 head.
- `tuned_speed_0p4b.jsonl`: promoted B1/B2/B4/B8 output-head matrix.
- `hybrid_threshold_0p4b.jsonl`: 1M/4M module-threshold diagnosis and repeat.
- `matrix_1p5b.jsonl`: 1.5B output-head and 8M memory-lane matrix.
- `matrix_2p9b.jsonl`: 2.9B output-head and 8M memory-lane matrix.
- `ffn_mm4_sweep_0p4b.jsonl`: isolated FFN tile/route diagnosis.

Promotion beyond the named output-head lane requires the larger-model matrix
and full-model fused-quant all-phase gate. No family-wide AMD claim is implied.
