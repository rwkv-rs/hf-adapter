# V100 production-close evidence — 2026-07-11

This bundle is the canonical same-host acceptance record for the exact NVIDIA
V100 (`sm_70`) HF native path. It covers RWKV-7 0.1B, 0.4B and 1.5B at batch
sizes 1/2/4/8. Dense fp16 is compared with the recorded same-host Albatross
high-water rows; repository-native W8/W4 `speed` policy is paired against the
same-process fp16 model.

## Result

`python bench/check_v100_production_close.py` returns **PASS**:

- dense decode: 12/12 rows, `0.908x–1.248x` Albatross (gate `>=0.90x`);
- dense prefill, prompt 512: 12/12 rows, `0.930x–1.047x` Albatross;
- W8/W4 decode: 24/24 rows strictly faster than fp16, `1.006x–1.128x`;
- W8/W4 prefill: 24/24 rows production-equivalent to fp16, `0.996x–1.007x`;
- quantized payload: `0.803x–0.956x` fp16;
- all quant rows preserve the fp16 next token; dense prefill preserves greedy
  output and the following decode token.

The prefill gate uses a `0.99x` equivalence floor because only the final
`logits_to_keep=1` projection is quantized and the measured delta is below 1%.
Rows are interleaved, same-process CUDA-event medians over 101 samples to cancel
clock/thermal drift. Decode uses the strict `>=1.0x` gate.

## Dense fp16

### Decode tok/s (HF / Albatross)

| Model | B1 | B2 | B4 | B8 |
|---|---:|---:|---:|---:|
| 0.1B | 767.4 (`0.974x`) | 1,449.3 (`0.963x`) | 2,553.2 (`0.977x`) | 4,181.5 (`1.158x`) |
| 0.4B | 437.7 (`0.933x`) | 749.9 (`0.925x`) | 1,221.6 (`0.954x`) | 1,953.5 (`1.247x`) |
| 1.5B | 232.9 (`0.974x`) | 377.0 (`0.908x`) | 610.0 (`1.026x`) | 892.2 (`1.037x`) |

### Prefill tok/s, prompt 512 (HF / Albatross)

| Model | B1 | B2 | B4 | B8 |
|---|---:|---:|---:|---:|
| 0.1B | 39,613 (`1.007x`) | 73,346 (`1.028x`) | 114,149 (`1.047x`) | 151,573 (`0.988x`) |
| 0.4B | 18,562 (`1.005x`) | 32,220 (`1.031x`) | 47,347 (`1.030x`) | 57,232 (`0.969x`) |
| 1.5B | 11,447 (`0.961x`) | 16,608 (`1.017x`) | 19,809 (`0.983x`) | 20,277 (`0.930x`) |

## Native quant speed policy

### W8

| Model | Decode B1/B2/B4/B8 | Prefill B1/B2/B4/B8 | Payload |
|---|---|---|---:|
| 0.1B | `1.052/1.077/1.019/1.028x` | `1.007/1.001/0.999/0.999x` | `0.869x` |
| 0.4B | `1.046/1.035/1.032/1.015x` | `1.001/1.001/1.002/1.000x` | `0.926x` |
| 1.5B | `1.032/1.032/1.044/1.006x` | `1.001/1.001/1.002/1.007x` | `0.956x` |

### W4

| Model | Decode B1/B2/B4/B8 | Prefill B1/B2/B4/B8 | Payload |
|---|---|---|---:|
| 0.1B | `1.048/1.128/1.044/1.060x` | `0.996/1.007/1.000/1.001x` | `0.803x` |
| 0.4B | `1.051/1.045/1.030/1.015x` | `1.003/1.003/1.003/0.999x` | `0.888x` |
| 1.5B | `1.060/1.045/1.044/1.020x` | `1.001/1.000/1.002/1.000x` | `0.934x` |

## Serving, multi-GPU and training gates

- Dynamic batch B8→B2, 128 decode steps: `2,291.7 tok/s`, 18 reorders,
  6 drops, seven CUDA Graph batch shapes coexisting, graph cache hit rate `1.0`.
- Chunked prefill, 0.1B/B4/prompt2048: chunk512 retains `0.851x` full
  throughput while reducing peak VRAM to `0.794x`; logits and following decode
  stay aligned. Chunk1024 retains `0.944x` throughput.
- Two-V100 manual HF `device_map`/pipeline split (6+6 layers): load, forward
  and 8-token `generate` pass, with exact equality to the single-device tail.
- V100 fp16 HF Trainer and TRL SFT one-step regressions pass with finite loss
  and non-zero trainable deltas.
- Two-V100 DeepSpeed ZeRO-2 and ZeRO-3 checkpoint resume pass from step 1 to
  step 2 on both ranks. The gate reduces partition-local trainable deltas across
  ranks before evaluating updates.

## Implemented V100 paths

- fixed-shape native prefill CUDA Graph cache with four coexisting batch shapes;
- fused shift mix and V100 policy-tuned WAVG LoRA;
- Albatross-derived, Apache-2.0-attributed `sm_70` WAGV/RKV kernels;
- V100 sparse FFN routing for B1/B2/B4;
- direct-output graph heads (no post-head copy);
- W8/W4 rowwise quantized storage;
- B1 weight-only W8/W4 warp kernels;
- B2/B4/B8 activation-quantized DP4A kernels that read each weight row once and
  accumulate all batch rows in registers.

All routes are gated to exact `sm_70`. Other architectures retain their existing
policy and fallback implementations.

## Reproduce and gate

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  bench/bench_native_quant_prefill_paired.py \
  --model /path/to/rwkv7-g1d-0.4b-hf --model-size-label 0.4b \
  --quantization a8w8 --batch-sizes 1,2,4,8 \
  --prompt-tokens 512 --warmup 12 --steps 101 \
  --results /tmp/v100_quant_prefill.jsonl

python bench/check_v100_production_close.py
```

Canonical evidence:

- `dense_decode.jsonl`
- `dense_prefill.jsonl`
- `quant_decode_acceptance.jsonl`
- `quant_prefill_acceptance.jsonl`
- `summary.json`
- serving, multi-GPU and training JSONL plus `pytest_relevant.log`
