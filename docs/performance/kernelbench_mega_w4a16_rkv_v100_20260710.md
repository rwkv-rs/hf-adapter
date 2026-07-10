# KernelBench-Mega-style W4A16 R/K/V experiment (V100)

Date: 2026-07-10

Branch: `wangyue/kernelbench-mega-w4a16-bench`

## Question

Can the KernelBench-Mega strategy of changing the whole timed-path layout,
rather than tuning three isolated W4 GEMVs, close the RWKV-7 quantized decode
gap?

The existing prototype already does direct packed-int4 unpack/dequant inside the
GEMV and fuses R/K/V into one launch. However, every Triton program keeps three
activation tiles, three weight tiles, and three fp32 accumulators live. The
experiment changes R/K/V into a projection grid dimension:

```text
existing:  one program -> R accumulator + K accumulator + V accumulator
candidate: one program -> one projection accumulator; projection is grid axis
```

The candidate consumes pre-stacked activations `[batch, 3, hidden]` and weights
`[3, hidden, packed_hidden]`. This is the layout a later native shift/mix
producer should write directly.

## Environment

- GPU: Tesla V100-PCIE-32GB (SM70)
- Model: RWKV-7 G1D 0.4B HF
- Layers sampled: 0, 1, 11
- Hidden size: 1024
- Activation/accumulation path: fp16 / fp32 accumulator
- Weight format: symmetric row-wise signed int4, two values per byte
- PyTorch: 2.5.1+cu124
- Triton: 3.3.0
- Primary gate: exact equality with the existing W4 fused R/K/V output

Raw artifact:
`bench/results_kernelbench_mega_w4a16_rkv_v100_20260710.jsonl`

## Batch-1 confirmation

Three independent 1,024-step runs were made after the initial sweep. Medians:

| path | latency | vs existing fused W4 | vs fp16 |
|---|---:|---:|---:|
| fp16, three linears | 0.071476 ms | 1.255x | 1.000x |
| existing fused W4 | 0.089683 ms | 1.000x | 0.797x |
| projection-axis W4, pre-stacked | **0.053694 ms** | **1.670x** | **1.326x** |
| projection-axis W4, runtime `torch.stack` | about 0.076 ms | **1.177x** | 0.935x |

The pre-stacked candidate is bit-identical to the existing W4 result. Minimum
cosine versus fp16 across sampled projections is about 0.981-0.982; the kernel
layout itself adds no numerical error.

## Batch sweep

| batch | fp16 ms | existing W4 ms | pre-stacked W4 ms | pre-stacked vs existing | pre-stacked vs fp16 | runtime-copy vs fp16 | min cosine vs fp16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (median) | 0.071476 | 0.089683 | **0.053694** | **1.670x** | **1.326x** | 0.935x | ~0.981 |
| 2 | 0.073920 | 0.089155 | **0.053119** | **1.678x** | **1.392x** | 0.976x | 0.9810 |
| 4 | 0.073742 | 0.089273 | **0.053546** | **1.667x** | **1.377x** | 0.932x | 0.9803 |
| 8 | 0.074241 | 0.089107 | **0.053489** | **1.666x** | **1.388x** | 0.960x | 0.9789 |

## Interpretation

The kernel change is a real improvement, not merely a block-size fluctuation:

1. It is consistently about 1.67x faster than the previous single-launch W4
   kernel across batch sizes 1/2/4/8.
2. With producer-ready pre-stacked inputs it is already faster than the three
   fp16 linears on V100 while retaining the 0.252x packed-weight footprint.
3. Paying `torch.stack` in the timed path erases the win versus fp16, although
   it remains faster than the existing W4 kernel. Production integration must
   therefore fuse the activation producer or write directly into the stacked
   layout; adding another wrapper-level stack is not acceptable.
4. The batch-8 cosine dip is inherited from row-wise W4 quantization, not from
   this kernel. Group-wise/asymmetric or Q*_K_M-quality quantization remains a
   separate accuracy task.

## Next integration gate

Do not replace the production native graph yet. First:

1. make the native shift/mix producer emit `[batch, 3, hidden]` directly;
2. pack R/K/V weights and scales into projection-major storage once at model
   load time;
3. connect the projection-axis kernel without `torch.stack` or per-token weight
   copies;
4. compare complete layer and complete model decode, not only the R/K/V slice;
5. repeat on RTX 4090 before selecting per-architecture defaults.

The experiment validates the KernelBench-Mega direction: changing fusion and
data-layout boundaries can beat fp16 on old hardware, while isolated W4 kernel
tuning had remained below fp16.
