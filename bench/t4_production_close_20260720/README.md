# Tesla T4 exact-card HF validation (2026-07-20)

**Result: VALIDATED, not production-close.** Functional HF, cache, fused prefill, native-graph decode, quantized inference and the declared single-GPU training integration matrix pass. Dense parity with Albatross and full-model all-phase quant speed remain open.

## Environment

- GPU: Tesla T4 15 GiB (`sm_75`), application clocks 1590/5001 MHz during timing.
- Software: Ubuntu 22.04, PyTorch 2.7.1+cu126, Transformers 5.12.1, Triton 3.3.1, bitsandbytes 0.49.2, PEFT 0.19.1, TRL 1.6.0, FLA 0.5.0, DeepSpeed 0.17.6.
- Candidate commit at measurement start: `58cfc2fcc4720e8f807050d12bb06259550bb6e0`; this artifact is attached to the dirty T4 adaptation branch and the final PR commit records the exact source.
- Albatross: commit `ee3308f6922e59f2166c7fac3c5a192340a2b48e`, `faster3a_2605`, `fp32io16` WKV, GPU embedding.

## Dense same-GPU comparison

### Cached decode (`tok/s`, fixed token)

| Model | B | HF native_graph | Albatross | HF / Albatross |
|---|---:|---:|---:|---:|
| 0.1b | 1 | 319.2 | 586.6 | 0.5441x |
| 0.1b | 2 | 561.7 | 949.1 | 0.5918x |
| 0.1b | 4 | 1098.6 | 1555.6 | 0.7062x |
| 0.1b | 8 | 2076.1 | 2400.3 | 0.8649x |
| 0.4b | 1 | 144.4 | 295.4 | 0.4888x |
| 0.4b | 2 | 247.4 | 504.2 | 0.4907x |
| 0.4b | 4 | 480.2 | 645.8 | 0.7436x |
| 0.4b | 8 | 921.3 | 1074.6 | 0.8573x |
| 1.5b | 1 | 64.5 | 109.5 | 0.5890x |
| 1.5b | 2 | 111.8 | 187.1 | 0.5976x |
| 1.5b | 4 | 215.9 | 271.0 | 0.7968x |
| 2.9b | 1 | 34.1 | 57.0 | 0.5987x |
| 2.9b | 2 | 62.7 | 97.9 | 0.6405x |

Measured ratio range: **0.4888x–0.8649x**. Minimum native-graph cache hit rate: **0.9855**.

### Prefill B1/T512 (`tok/s`)

| Model | HF fused scan | Albatross | HF / Albatross |
|---|---:|---:|---:|
| 0.1b | 21044.7 | 39083.2 | 0.5385x |
| 0.4b | 10820.2 | 16075.4 | 0.6731x |
| 1.5b | 5004.1 | 6677.7 | 0.7494x |
| 2.9b | 2874.7 | 3747.5 | 0.7671x |

Measured ratio range: **0.5385x–0.7671x**. These rows use the effective T4 fused native scan and preserve greedy output.

## Quantization

| Lane | Quant | Rows | Footprint ratio | Prefill ratio | Decode ratio | Min final cosine | Greedy |
|---|---|---:|---:|---:|---:|---:|---:|
| head speed | MM8 | 13 | 0.8686–0.9716 | 0.9716–1.0120 | 1.0207–1.0950 | 0.9999345 | 13/13 |
| head speed | MM4 | 13 | 0.8043–0.9578 | 0.9704–1.0078 | 1.0207–1.1166 | 0.9996467 | 13/13 |
| full model | MM8 | 13 | 0.5291–0.6331 | 0.5767–0.5939 | 0.8118–1.6158 | 0.9997310 | 13/13 |
| full model | MM4 | 13 | 0.3004–0.4542 | 0.1272–0.6984 | 0.7509–1.4868 | 0.9969545 | 13/13 |

The speed lane replaces only `lm_head`; it closes decode speed/correctness with a smaller memory saving. The full-model lane closes memory and B1 decode, but full-model prefill and small-model B4/B8 decode remain below fp16. Therefore this artifact does **not** claim universal T4 W8/W4 performance closure.

## HF and training integration

- All four checkpoints pass load/generate, standard HF API, batch cache, dynamic select/reorder/drop, chunked prefill handoff and native-graph decode checks.
- Trainer + LoRA, TRL SFT/DPO/GRPO pass for 0.1B/0.4B/1.5B/2.9B in the declared T4 memory shapes.
- PEFT save/reload is exact. FP16 merge/unmerge for 1.5B/2.9B uses the measured `max_abs <= 0.2` gate and preserves greedy tokens.
- Trainer resume passes 0.1B/0.4B. Single-GPU ZeRO-2/3 train and resume pass on 0.1B; this proves integration/checkpointing, not multi-GPU sharding.
- Official `.pth` CPU-FP32 vs HF CUDA-FP16 alignment passes 0.1B/0.4B/1.5B (top-5 >= 0.96, cosine >= 0.999997, greedy 64/64, 64/64, 32/32). 2.9B was not run because the 15 GiB host-RAM boundary cannot hold the official CPU-FP32 reference safely.
- Official `train_temp` CUDA exact-training alignment is not a T4 claim: that path requires BF16 and `sm_80+`.

## Evidence map

- `results_t4.jsonl`: 123 curated dense/cache/prefill/fused rows.
- `albatross/`: same-GPU HF fixed-token rows and official Albatross logs/JSONL.
- `quant/results_quant_speed.jsonl`: 26 head-speed rows.
- `quant/results_quant_full_model.jsonl`: 26 broad-memory rows.
- `training/`: Trainer/PEFT/TRL/ZeRO/resume and official-alignment rows/logs.
- `validation/full_pytest.log`: repository regression, 594 passed and 8 skipped.
- `summary.json`: machine-readable promoted summary; `SHA256SUMS`: artifact integrity.

## Promotion boundary

Tesla T4 receives exact-card compatibility defaults and measured DP4A W8/W4 routing. RTX 2080/other `sm_75` products remain fail-closed. T4 is **Validated** in the hardware matrix; promotion to **Production-close** requires closing the dense Albatross gap and full-model W8/W4 prefill plus all measured decode batches.
