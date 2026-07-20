# Tesla T4 / Turing HF validation

Exact-card validation snapshot for the Hugging Face adapter on one Tesla T4.

Last updated: **2026-07-20**.

## Status

**Validated, not production-close.** The four-checkpoint HF/API/cache matrix,
T4 fused prefill, native-graph decode, W8/W4 functionality, and the declared
single-GPU training integrations pass. Dense Albatross parity and broad
full-model quantized speed do not yet pass.

The final repository regression on this card reports **594 passed, 8 skipped**;
the raw output is retained in the promoted evidence artifact.

Canonical raw evidence:
[`bench/t4_production_close_20260720/`](../../bench/t4_production_close_20260720/README.md).

## Exact environment

- Tesla T4 15 GiB, `sm_75`, driver `580.159.03`;
- PyTorch `2.7.1+cu126`, Transformers `5.12.1`, Triton `3.3.1`;
- bitsandbytes `0.49.2`, PEFT `0.19.1`, TRL `1.6.0`, FLA `0.5.0`,
  DeepSpeed `0.17.6`;
- fp16 inference on RWKV-7 0.1B, 0.4B, 1.5B and 2.9B;
- same-card Albatross commit
  `ee3308f6922e59f2166c7fac3c5a192340a2b48e`, `faster3a_2605`,
  `fp32io16` WKV and GPU embedding.

Checkpoint SHA-256 values are recorded in
[`model_hashes.sha256`](../../bench/t4_production_close_20260720/model_hashes.sha256).

## Card-local runtime policy

The exact device names `Tesla T4` and `NVIDIA T4` use:

- default-on: fast recurrent cache, fused recurrent-output, fused output,
  native fast prefill, and the measured T4 fused prefill scan;
- default-off: output-project, projection, WAG/WAVG LoRA fusion;
- PyTorch 2.7 / Triton 3.3 `torch.compile` compatibility fallback because the
  measured Inductor worker imports the removed legacy `AttrsDescriptor` path;
- exact-T4 DP4A W8/W4 kernels. The extension builds a Volta/Turing fat binary
  and locates `ninja` next to the active Python executable even when the venv
  was not shell-activated.

Routing is fail-closed. `sm_75` alone is insufficient: RTX 2080, NVIDIA T400,
and other Turing products do not inherit the T4 quant or prefill promotion.

## Dense performance versus Albatross

Same-GPU fixed-token cached decode:

| Model | Batches | HF / Albatross range |
|---|---|---:|
| 0.1B | 1/2/4/8 | `0.5441x–0.8649x` |
| 0.4B | 1/2/4/8 | `0.4888x–0.8573x` |
| 1.5B | 1/2/4 | `0.5890x–0.7968x` |
| 2.9B | 1/2 | `0.5987x–0.6405x` |

The minimum measured native-graph cache hit rate is `0.9855`. B1/T512 fused
prefill reaches `0.5385x`, `0.6731x`, `0.7494x`, and `0.7671x` Albatross for
0.1B/0.4B/1.5B/2.9B. These are current gaps, not promoted parity claims.

## Quantization

Two separately named lanes are retained:

| Lane | Quant | Footprint / fp16 | Prefill / fp16 | Decode / fp16 | Correctness |
|---|---|---:|---:|---:|---|
| head speed | W8 | `0.8686x–0.9716x` | `0.9716x–1.0120x` | `1.0207x–1.0950x` | greedy `13/13`, cosine `>=0.9999345` |
| head speed | W4 | `0.8043x–0.9578x` | `0.9704x–1.0078x` | `1.0207x–1.1166x` | greedy `13/13`, cosine `>=0.9996467` |
| full model | W8 | `0.5291x–0.6331x` | `0.5767x–0.5939x` | `0.8118x–1.6158x` | greedy `13/13`, cosine `>=0.9997310` |
| full model | W4 | `0.3004x–0.4542x` | `0.1272x–0.6984x` | `0.7509x–1.4868x` | greedy `13/13`, cosine `>=0.9969545` |

The head-only lane closes the measured decode-speed gate but saves less memory.
The full-model lane closes large memory reduction and B1 decode, but not every
prefill or B4/B8 decode cell. Universal T4 quant performance therefore remains
open.

## HF training integration

- Trainer + LoRA and TRL SFT/DPO/GRPO pass on all four checkpoints in the
  declared memory-safe shapes.
- PEFT adapter save/reload is exact. FP16 merge/unmerge on 1.5B/2.9B uses
  `max_abs <= 0.2` and preserves greedy tokens.
- Trainer resume passes on 0.1B/0.4B.
- Single-GPU ZeRO-2/3 train and resume pass on 0.1B. This is an integration and
  checkpoint contract, not multi-GPU sharding evidence.
- Official CPU-FP32 `.pth` versus HF CUDA-FP16 alignment passes 0.1B/0.4B/1.5B.
  The 2.9B official reference was not run because the host has only 15 GiB RAM.
- Official `train_temp` CUDA exact-training alignment is not a T4 claim; that
  route requires BF16 and `sm_80+`.

## Reproduction

```bash
bash bench/run_t4_hf_validation.sh \
  MATRIX_MODE=full \
  MODEL_ROOT=/opt/models \
  PYTHON_BIN=/path/to/venv/bin/python
```

The runner rejects a non-T4 GPU by default, selects fp32 training for
0.1B/0.4B and fp16 memory-safe shapes for 1.5B/2.9B, records per-model metadata,
and keeps RTX 20 routing conservative.

## Remaining promotion gates

1. Close dense native-graph decode and fused-prefill parity with Albatross.
2. Fuse broad quantized projection/activation work so full-model W8/W4 is
   fp16-or-faster for every declared prefill/decode batch.
3. Add task-quality evaluation beyond short logits/greedy gates.
4. Validate each other Turing product independently; do not promote by SM alone.
