# RTX 3090 Ampere HF acceptance close (2026-07-12)

This artifact records an end-to-end Hugging Face adapter validation on a
24 GiB Ampere consumer GPU. It covers standard Transformers APIs, recurrent
cache behavior, dynamic/chunked execution, PEFT/Trainer/TRL/DeepSpeed training,
dense batch scaling, full-memory bnb W8/W4 functionality, native MM8/MM4
speed-policy rows, MATH500, and larger-checkpoint boundaries.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB, `sm_86`.
- Driver: 550.142; PyTorch: `2.6.0+cu124`; CUDA runtime: 12.4.
- Python 3.10; Triton 3.2.0; FLA 0.5.1. Core HF/training acceptance
  used Transformers 4.57.6; the official Qwen3.5 comparison lane uses
  Transformers 5.12.1 and causal-conv1d 1.6.2.post1.
- PEFT 0.19.1; TRL 1.8.0; DeepSpeed 0.19.2; bitsandbytes 0.49.2.
- Repository base: `77aef8a`; cache telemetry fix: `b3000b3`.

The node has one RTX 3090 and 31 GiB host RAM. Model conversion and every GPU
measurement were performed on this node. Exact package and driver output is
retained in `environment.log` after the final regression phase.

## Cache handoff fix

The first fast-token decode step converts an FLA/HF cache to
`RWKV7StateCache`. The old conversion retained layer state but reset externally
stored sequence-length telemetry to zero. That did not change logits in the
measured case, but made cache length incorrect after a completed prefill.

`RWKV7StateCache.from_legacy_cache` now preserves `get_seq_length()` when no
explicit `seen_tokens` override is supplied. The regression converts a source
cache with length 17 and the live fast-cache test now preserves the complete
prefill/decode length.

## Transformers, cache and serving-like behavior

The 0.4B checkpoint passes remote-code `AutoConfig`, `AutoTokenizer`,
`AutoModelForCausalLM`, cached forward, greedy `generate`, save/load contract,
finite logits and attention-mask/labels behavior. Cache validation covers:

- batch sizes 1/2/4/8;
- dynamic selection/compaction and fast-token decode;
- chunked prefill sizes 8/16/32/64;
- cache length, greedy-token and bounded-logit agreement;
- source-cache to `RWKV7StateCache` handoff.

Raw evidence is under `raw/`; final fast-cache and regression logs are promoted
when the chained run completes.

## Dense batch scaling

All rows use fp16, prompt 128, decode 32 and the production HF wrapper. The
fast-token row uses `native_graph`; the generic forward row is retained in the
raw JSONL but is not the promoted decode path.

| Model | Bsz | Prefill tok/s | Fast decode tok/s | Peak MiB |
|---|---:|---:|---:|---:|
| 0.4B | 1 | 1,933.9 | 243.3 | 1,133.6 |
| 0.4B | 2 | 3,879.6 | 413.3 | 934.2 |
| 0.4B | 4 | 7,507.2 | 817.5 | 998.5 |
| 0.4B | 8 | 14,965.9 | 1,597.9 | 1,119.0 |
| 1.5B | 1 | 1,900.7 | 156.2 | 3,196.9 |
| 1.5B | 2 | 3,597.6 | 261.9 | 3,037.5 |
| 1.5B | 4 | 7,273.0 | 501.6 | 3,157.4 |
| 1.5B | 8 | 14,567.3 | 997.9 | 3,389.2 |

These are hardware characterization rows, not a same-session Albatross
microbenchmark. MATH500 supplies the fail-closed external reference gate.

## Training ecosystem

The 0.4B adapter passes PEFT LoRA forward/loss/backward, HF Trainer, TRL SFT,
TRL DPO, TRL GRPO and Trainer checkpoint resume with finite loss and nonzero
trainable-parameter deltas. DPO/GRPO and resume use bf16 on RTX 3090:

- DPO loss `0.69140625`, max trainable delta about `1e-4`;
- GRPO max trainable delta about `1e-4`;
- resumed Trainer reaches global step 2 with a nonzero post-resume delta.

An fp16 DPO probe produced non-finite chosen logits and no parameter update;
bf16 is therefore the promoted RL-training dtype on this card. ZeRO-2 and
ZeRO-3 both pass the two-rank Trainer smoke. ZeRO-2 and ZeRO-3 checkpoint
resume also reach global step 2 with finite loss and a nonzero trainable
parameter delta. Raw launch, rank and result telemetry is retained in `zero/`.

## RTX 3090 native prefill route

The conservative generic Ampere policy originally used FLA prefill. Exact-card
testing promotes fixed-shape native prefill graphs with the fused recurrent
scan only for an RTX 3090. At 1.5B, bsz1/prompt128, the median dense prefill row
improves from about 1,799 tok/s to 6,887 tok/s. The corresponding optimized
Qwen3.5-2B HF row is 2,436 tok/s.

Correctness is checked against ordinary HF prefill at bsz1/prompt128 and
bsz8/prompt512. Both shapes preserve the greedy token, the following decode
token, chunked-prefill behavior and greedy generation; minimum logit cosine is
at least 0.99999994. Other sm86 cards retain the conservative Ampere default
until exact-card evidence exists.

The generic full-memory bnb W4 route needs a different speed/memory balance on
this card. The promoted `prefill_hot` policy leaves all attention projections,
all FFN up projections and seven of every eight FFN down projections dense.
At 1.5B, bsz1/prompt128, the three-run median is 1,711.9 prefill tok/s and
32.06 decode tok/s versus optimized Qwen3.5-2B W4 at 1,652.5/24.59 tok/s. The
RWKV footprint is 2,841.3 MiB versus its 2,913.3 MiB fp16 footprint. This is a
strict nonnegative-speed W4 route with a modest 2.5% footprint reduction, not
a minimum-memory W4 claim. W8 retains the more memory-efficient `decode_hot`
route because it already clears the speed gate.

## Quantized inference

### Full-memory bnb functionality

The 0.4B dense fp16 footprint is about 859.8 MiB. Generic HF quantized loading
and generation pass with:

| Mode | Footprint MiB | Dense ratio | Peak MiB | Result |
|---|---:|---:|---:|---|
| bnb W8 | 571.8 | 0.665 | 855.8 | PASS |
| bnb W4 | 427.8 | 0.498 | 728.9 | PASS |

This proves the large memory reduction and HF compatibility lane; bnb forward
latency is not promoted as a speed lane.

### Native MM8/MM4 speed policy

The 1.5B pressure matrix contains fp16/MM8/MM4, prompt 128/2048, decode 128/512,
batch 8, with in-process paired fp16 baselines and three timing repeats. Raw
rows are in `quant_pressure_15.jsonl`; the generated table is
`quant_pressure_15_summary.md`.

| Quant | Rows | Min decode ratio | Median ratio | Footprint ratio | Same next |
|---|---:|---:|---:|---:|---:|
| MM8 | 4 | 0.9832 | 1.0027 | 0.9562 | 4/4 |
| MM4 | 4 | 0.9966 | 1.0003 | 0.9342 | 4/4 |

Every quant row lowers footprint, preserves the fp16 greedy next token, and has
prompt/final cosine above 0.9997. The declared conservative `>=0.98x` paired
speed-equivalence gate passes. One W8 long-pressure row is `0.9832x`, so this
artifact does not claim universal strict `>=1.00x` quant speed. These are
selected-module `speed` policy rows, not full-memory fused-quant claims.

## MATH500 / Albatross gate

The first live `500 x 64` run uses seed 43, a bsz64/96/128 sweep, native
prefill, native-graph decode, deferred text decode and four-worker answer
verification. It compares against the repository's committed Albatross
full-run reference.

| Metric | RTX 3090 HF | Albatross reference | Result |
|---|---:|---:|---|
| pass@64 | 0.368 | 0.370 | strict gate MISS by 1/500 tasks |
| rollout accuracy | 0.143188 | 0.145938 | delta -0.002750 |
| generation token/s | 8,822.8 | 3,903.6 | PASS, 2.260x |
| steady decode token/s | 10,051.2 | 3,970.1 | PASS, 2.532x |

Shape and both 2x performance gates pass. The exact pass@64 gate misses by one
task, so this first run is retained as a strict overall FAIL rather than being
rounded or relabeled. A second full same-seed run independently reproduced
pass@64 `0.368`; its generation/steady-decode ratios were `2.247x/2.516x`.
This confirms a stable one-task quality boundary rather than a transient speed
or process-state failure.

The second run also completed the 500-text external compression check: 43,865
tokens, `1.9240783 bits/token`, candidate/reference ratio exactly `1.0`.
The comparison is against a committed reference, not a fresh same-card
Albatross process.

## Larger checkpoints

The official 2.9B checkpoint was downloaded, SHA256-verified and converted on
this 31 GiB host through the low-memory mmap/meta-template path. Its SHA256 is
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`.
Load, finite forward and four-token generation pass with a 5,622.4 MiB model
footprint and 5,888.0 MiB peak; the decoded continuation is `Hello! How can`.

For prompt128/decode32, native-graph decode scales from 89.1 tok/s at bsz1 to
596.2 tok/s at bsz8; bsz8 prefill is 9,444.2 tok/s. Full-memory bnb W8/W4
generation passes at 3,222.4/2,022.4 MiB footprint.

The first Ampere W4 speed-policy run exposed a real dispatch gap: bsz8 was
only about `0.649x` fp16 because sm86 materialized the dequantized weight above
four rows. The new capability-routed fp16 tensor-core batch kernel closes it:

| Prompt | fp16 tok/s | W4 tok/s | W4/fp16 | Footprint ratio | Same next |
|---:|---:|---:|---:|---:|---:|
| 128 | 619.6 | 631.9 | 1.0208x | 0.9573 | yes |
| 2048 | 616.9 | 629.8 | 1.0216x | 0.9573 | yes |

Both rows preserve prompt/final cosine above 0.9998. The bsz8 fused dispatcher
matches the torch reference within 0.010742 max-abs and the end-to-end cosine
is 0.998872. The measured route is capability-gated to exact sm86 within the Ampere
family; unmeasured sm80/Ada defaults are unchanged.

The 7.2B dense prompt-2048 batch-1/2 sequence-fused rows were subsequently
closed against the same-card Qwen3.5-9B fast path; see
[`../3090_self_fused_20260713/README.md`](../3090_self_fused_20260713/README.md).
The remaining full 3090 matrix and strict W8/W4 speed gates are still open.

The 13.3B fp16 model has a known footprint above this card's 24 GiB physical
VRAM and is not presented as a 3090 fp16 support row. Its conversion/load gate
is already covered by the promoted RTX 5090 artifact.

## Reproduction

```bash
python bench/run_math500_final_acceptance.py \
  --hf-dir /models/rwkv7-g1d-0.4b-hf \
  --dataset /data/MATH500.jsonl --out-dir /results/math500-3090 \
  --tokenizer-dir /models/rwkv7-g1d-0.4b-hf \
  --bsz-list '64 96 128' --full-rollout 64 --full-max-new-tokens 1500 \
  --dtype fp16 --device cuda --prefill-backend native \
  --decode-backend fast_token --rng-mode global

python bench/run_blackwell_quant_matrix.py \
  --model 1.5b=/models/rwkv7-g1g-1.5b-hf \
  --prompt-tokens 128 2048 --decode-tokens 128 512 \
  --batch-sizes 8 --quantizations none mm8 mm4 \
  --policy speed --warmup 16 --timing-repeats 3 \
  --paired-baseline --results /results/quant_pressure_15.jsonl
```

## Claim boundary

This is a production-close artifact for the measured RTX 3090 HF lanes. It
does not convert speed-policy quantization into a full-memory fused-quant claim,
does not treat a committed-reference MATH gate as a fresh same-card Albatross
microbenchmark, and does not claim that a 24 GiB card can load 13.3B fp16.
