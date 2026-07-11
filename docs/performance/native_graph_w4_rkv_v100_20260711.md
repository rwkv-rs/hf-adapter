# Native-graph W4 R/K/V integration (V100)

Date: 2026-07-11

Branch: `wangyue/kernelbench-mega-w4a16-bench`

## Change

The first KernelBench-Mega-inspired experiment proved that a projection-axis
W4A16 R/K/V kernel can beat fp16 when its input is already stacked. This stage
connects that layout to complete RWKV-7 native-graph decode:

1. fused time-mix writes R/K/V activations directly to `[B,3,H]`;
2. R/K/V weights are quantized and packed projection-major once during native
   pack extraction;
3. the projection-axis W4 kernel consumes that layout without `torch.stack`;
4. the remaining W/A/G/V-gate, recurrent update, output, FFN, final norm, and
   lm_head paths remain unchanged;
5. the feature is opt-in through `RWKV7_NATIVE_GRAPH_W4_RKV=1`.

Dense/native defaults are unchanged.

## Validation environment

- GPU: Tesla V100-PCIE-32GB (SM70)
- Model: RWKV-7 G1D 0.4B HF, 24 layers, hidden 1024
- dtype: fp16, recurrent state accumulation fp32
- prompt state: eight dense-prefilled tokens
- timed path: complete CUDA-graph token step, including embedding, all layers,
  final norm, and lm_head
- measurement: 64 graph replays; three independent A/B runs for each batch
- quality: dense and W4 start from identical dense-prefilled recurrent state

Both physical V100s had unrelated long-running co-tenant jobs using about 22 GB
each during validation. GPU1 was selected because its sampled utilization was
normally idle, but absolute latency/tok/s is still provisional. Relative values
below are same-process dense/W4 A/B medians and were stable across three runs.

## Full token-step result

| batch | dense ms | W4 ms | W4 speedup | dense tok/s total | W4 tok/s total | min logit cosine | 16-token greedy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.195514 | **4.974629** | **1.2456x** | 161.41 | **201.02** | 0.999483 | 16/16 |
| 2 | 7.068582 | **6.023812** | **1.1734x** | 282.94 | **332.02** | 0.999482 | 32/32 |
| 4 | 7.242679 | **6.321930** | **1.1456x** | 552.28 | **632.72** | 0.999482 | 64/64 |
| 8 | 8.049176 | **7.522601** | **1.0704x** | 993.89 | **1063.46** | 0.999484 | 128/128 |

The R/K/V slice itself stores 144 MiB in fp16 and 36.2812 MiB in the current
packed row-wise W4 format, a `0.252x` ratio. The experiment still retains the
dense model parameters for fallback, so total model VRAM does not yet fall by
that full amount.

## Correctness gates

- stacked shift/mix Triton output vs torch fallback: exact max difference 0 for
  batch 1/2/4/8 at hidden 1024;
- projection-axis W4 output vs previous W4 projection: exact max difference 0;
- full graph one-step logit cosine vs dense: at least about 0.99948 in the
  recorded batch sweep;
- full graph greedy rollout: every generated token matched dense for 16 steps
  across batch 1/2/4/8;
- repo-code HF public API smoke after native prefill: native-graph decode and
  cache continuation passed for batch 1 and 4;
- dense `native_jit`/`native_graph` regression remained green after extending
  layer packs to carry optional W4 storage.

## Interpretation

This closes the previous integration gap: the microkernel win survives a full
24-layer token step and remains positive through batch 8. The declining gain at
larger batch is expected because dense GEMM utilization improves and the W4
dequant arithmetic becomes a larger fraction of the projection.

This is still an experimental hybrid path, not completed production W4:

- only attention R/K/V projections use W4;
- dense R/K/V weights remain resident for fallback;
- row-wise symmetric W4 is not yet Q*_K_M-quality quantization;
- 4090/H100/Blackwell need independent tile and accuracy validation;
- idle-card reruns are required before publishing absolute tok/s.

## Next work

1. repeat the same A/B on an idle V100 and RTX 4090;
2. add load/save support for prepacked R/K/V buffers instead of repacking at
   process startup;
3. allow an inference-only mode to release dense R/K/V weights after packing;
4. replace row-wise symmetric W4 with group-wise/asymmetric calibration while
   preserving the projection-axis kernel boundary;
5. extend the same direct-layout approach to `o_proj` and FFN only after
   complete-token A/B proves another positive gain.

## Artifacts

- `bench/bench_native_graph_w4_rkv.py`
- `bench/results_native_graph_w4_rkv_v100_20260711.jsonl`
- `bench/results_native_graph_w4_rkv_public_api_v100_20260711.jsonl`
