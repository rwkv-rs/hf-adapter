# HF adapter acceptance status

This is the canonical mapping between the public RWKV-7 Hugging Face adapter
requirements and repository evidence. `PASS` means the named, profile-bounded
gate has a reproducible artifact. A new card, shape, or dataset extends that
matrix instead of retroactively reopening an accepted release gate.

Last updated: **2026-08-12**. The released baseline was audited at `main`
`045bac1b769240facd290e1ac8232e8b1ca39778` after the `v0.6.0` release and
the merged RTX 4080/V100 B8 optimization series.

This page reports status. For ordinary-user commands and PASS gates for every
implemented capability below, start with
[`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md).

## Accepted Native-default and official train_temp promotion gate

The accepted backend promotion replaced RWKV's implicit FLA runtime with the
canonical native model. It was not accepted by merely defaulting
`RWKV7_NATIVE_MODEL`: the native model now owns the graph/fused performance
path, loads with FLA imports blocked, and preserves the HF ecosystem matrix.
FLA remains an explicitly selected RWKV reference backend. Qwen's optimized
full-FLA benchmark route remains unchanged.

Training acceptance is pinned to official RWKV-LM commit `e6f74b6` and both
entry scripts: `RWKV-v7/train_temp/demo-training-prepare.sh` and
`RWKV-v7/train_temp/demo-training-run.sh`. The scripts define two different
phases and must not be conflated. `prepare.sh` creates the initialization on
CPU with `micro_bsz=1`, `adam_eps=1e-8`, and zero weight decay. `run.sh` is the
single-GPU training recipe: `x070`, L12/D768, effective FFN3072, head64,
vocab65536, `ctx_len=512`, Minipile binidx, `magic_prime=2926181`,
`micro_bsz=16`, BF16, `lr_init=6e-4`, `lr_final=6e-5`, betas 0.9/0.99,
`adam_eps=1e-18`, `weight_decay=0.001`, warmup 10, `grad_cp=1`, implicit
`grad_clip=1.0`, kernel `@rwkv3`, and `deepspeed_stage_2` on one GPU. The
machine-readable contract is
[`../configs/train_temp_official_x070_12x768_b16.json`](../configs/train_temp_official_x070_12x768_b16.json).

Promotion requires the native path to use the same initialized checkpoint,
serialized sample order, optimizer grouping, FusedAdam update, schedule and
bounded training steps. Exact backward/step comparison and a predeclared
multi-seed cohort are mandatory. `train.py` reports its unused generic 3.5x
default as `dim_ffn=2688`, while the pinned fast `RWKV_CMix_x070`
implementation and generated checkpoint use 4x FFN3072. The earlier B1/T512
FFN3072 cohort matches the official kernel shape; the separate Native B16/T512
artifact now closes the shell-shape tensor, multi-seed, resume and bounded
memory-stability gates.

The first RTX 5070 inference migration checkpoint passes on 0.4B/fp16:
Native Graph B1 reaches `223.47 tok/s` versus the retained wrapper-hosted
`226.3 tok/s` row (`0.9875x`), with logits cosine `0.99999988` and 32/32
greedy alignment. B1/B2/B4/B8 prompt-32 probes pass prefill and decode greedy
alignment. This is inference evidence only; the shell-recipe training gate
above remains mandatory.

## Executive result

| Requirement | Status | Current evidence | Profile boundary / extensions |
|---|---|---|---|
| HF adapter release scope | **COMPLETE** | Published `v0.6.0`, current-main CI, canonical Native/no-FLA implementation, and the evidence-linked gates below | New profiles extend the release; they are not retroactive blockers |
| RWKV-LM / Albatross correctness and performance | **PASS for declared exact-card profiles** | Existing V100/4080/4090/5090 reference lanes remain; V100 adds exact 0.4B/1.5B B8 FP16 state; RTX 5070 Laptop adds exact Native 0.4B/1.5B prefill/decode schedules while retaining its 18/18 full-FLA Qwen matrix | More Albatross/Qwen cards, batches and shapes are post-release expansion |
| Transformers API | **PASS** | Auto classes, save/reload, generation, labels/loss, attention mask and recurrent cache tests | Upstreaming and long-term Transformers-version maintenance |
| PEFT and RL ecosystem | **PASS for published compatibility and exact-training profiles** | LoRA lifecycle, Trainer, SFT, DPO and GRPO smoke; RTX 5090 BF16 12x768 B1 plus Native B16/T512 exact tensors, paired real-MiniPile 3-seed x 1,000-step cohort, continuous 5,000-step run, 2,500+2,500 resume and steady-memory evidence | Larger models, multi-day runs, additional cards and distributed convergence extend the matrix |
| Dynamic batching, chunked prefill and state cache helpers | **PASS in HF adapter scope** | State select/reorder/drop/compact, chunked-prefill parity, serving-like cache telemetry | Native vLLM/SGLang integration remains a separate repository/project |
| Hardware support | **PASS for declared release policy and exact-card matrix** | NVIDIA, AMD/ROCm, Apple, Ascend, Biren, MetaX, MUSA and CPU fallback boundaries are represented; promotion stays exact-product and fail-closed | Additional products receive independent post-release evidence |
| W8/W4 inference and lower memory | **PASS for functionality, footprint, quality, and promoted speed profiles** | bnb compatibility plus native MM8/MM4; V100/T4/4080/4090/5090 and Apple profiles preserve their exact speed/memory boundaries | Wider full-memory profiles are independent optimizations |
| PP/TP boundary | **PASS for dense HF inference scope** | Two-V100 layer-split `device_map` matches the single-device reference; separate Transformers-native B1/B8 `tp_plan="auto"` shards embedding/attention/FFN/head matrices with exact greedy parity, minimum logits cosine `0.99999821`, and `0.52031x/0.611611x` local peak VRAM | Recurrent state is replicated; quantized TP, TP training, and native serving-engine execution require separate gates |
| ZeRO-2/3 training | **PASS for current smoke matrix** | ZeRO-2/3 base and resume evidence on V100/A100/A800/A6000 combinations | Longer training and larger ZeRO-3 resume matrix |
| Initial speculative decoding | **PASS as experimental HF/Apple path** | HF-compatible target/draft harness and Apple target-greedy oracle evidence | Serving integration and broader quality/speed gates |

## How to report completion

The **current HF milestone is complete**, and `v0.6.0` is the published
HF-adapter release for the boundaries below. The conversion, Transformers,
training ecosystem, cache, hardware-policy, quantization, PP/TP, and
profile-based performance requirements are accepted.

An unbounded combination of every future card, model, batch, sequence length,
quality suite, and serving runtime is not a finite release gate. Those
additions are post-release projects and must preserve their own exact evidence.

There is no official repository-wide completion percentage. Report the named
scope and its status instead; do not estimate a percentage from TODO checkboxes
or by counting the table rows above. Report `RWKV-7 HF adapter v0.6.0` as
**COMPLETE**, then name the exact hardware or benchmark profile when making a
more specific claim.

## Official requirement mapping

### 1. Performance, speed, accuracy and memory

- **V100:** 0.1B/0.4B/1.5B × bsz1/2/4/8 production-close matrix is
  promoted. Dense decode is `0.908x–1.248x` and prompt-512 prefill is
  `0.930x–1.047x` of same-host Albatross references. Separately, target-only
  RWKV-7 1.5B versus full-FLA/Triton-conv Qwen3.5-2B passes B1/B8 raw
  prefill/decode minima `2.815921x/5.270432x` and active-parameter work minima
  `2.285574x/4.277804x`; the B1 peak-VRAM loss remains disclosed. Evidence:
  [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md).
  Exact 0.4B/1.5B B8 FP16 state separately passes opposite A/B orders at
  `1.0216x-1.0288x`, saves `16.875-58.125 MiB`, and preserves the recorded
  greedy traces. Evidence:
  [`../bench/v100_exact_card_20260811/README.md`](../bench/v100_exact_card_20260811/README.md).
- **RTX 5070 Laptop:** the accepted B8 full-FLA-Qwen lane remains unchanged.
  The exact Native/no-FLA expansion promotes 0.4B/1.5B P128/P512 graph+scan,
  raw recurrent, shape-gated norm/mix, and B8 FP16 state. Its rejected
  projection/LoRA/warp candidates remain disabled. Evidence:
  [`../bench/5070_max_perf_20260811/README.md`](../bench/5070_max_perf_20260811/README.md).
- **RTX 4080:** 0.4B/1.5B/2.9B versus full-FLA Qwen3.5 0.8B/2B/4B passes
  all 36/36 parameter-adjusted Prefill and 36/36 Decode cells. Full-matrix
  minima are `1.068520x/1.140700x`; all six group medians and adjusted E2E
  medians also remain ahead. The exact-B8 grouped W/A/V projection route improves
  those RWKV checkpoints by `1.1267x/1.0942x/1.0809x` with exact first-step
  logits and greedy `4,608/4,608`. The separate 7.2B/B8 FP16-state decode
  route reaches `344.39 tok/s`, `1.0301x` its FP32-state route, saves
  `123.88 MiB`, and matches greedy `12,288/12,288`. Evidence:
  [`../bench/4080_adjusted_pd_20260811/README.md`](../bench/4080_adjusted_pd_20260811/README.md),
  [`../bench/4080_b8_projection_bmm_20260809/README.md`](../bench/4080_b8_projection_bmm_20260809/README.md)
  and
  [`../bench/4080_7p2b_fp16_state_20260809/README.md`](../bench/4080_7p2b_fp16_state_20260809/README.md).
- **RTX 3090:** latest g1d 0.4B and 2026-08-05 g1i 1.5B/2.9B/7.2B versus
  official Qwen3.5 0.8B/2B/4B/9B passes all `24/24` B1/B8,
  P128/P512/P2048, D128 dense-FP16 cells at the strict parameter-adjusted
  prefill gate. Minimum/median adjusted prefill PD is
  `1.227477x/1.467758x`; all Qwen rows verify full FLA plus Triton causal
  convolution. The exact-shape FP16-accumulation oracle passes `25/25` prompt
  and cache-handoff rows with cosine `>=0.9999` and exact greedy tokens.
  Evidence:
  [`../bench/3090_g1i_qwen35_maxperf_20260812/README.md`](../bench/3090_g1i_qwen35_maxperf_20260812/README.md).
- **RTX 4090:** 0.4B dense decode bsz1/2/4/8 reaches
  `1.007x/1.016x/1.008x/1.418x` of matching Albatross rows. Prompt-512 bsz4 is
  `1.007x` the same-session reference and `0.916x` the retained historical
  high-water reference. Separately, all published 0.4B/1.5B/2.9B/7.2B pairs
  pass the batch-8 dense/W8/W4 Qwen3.5 contract: `54/54` small-model cells and
  `18/18` 7.2B cells, with full-FLA, dense decode active-work, quant speed and
  quant-local physical-memory gates. A 2026-08-12 exact-card transfer of the
  recent RTX 4080 work adds block-scoped FP16 accumulation for latest
  0.4B/1.5B/2.9B B1/B8 P128/P512/P2048 Prefill and grouped W/A/V BMM for
  those checkpoints at B8. The paired gates pass `108/108` accumulation rows,
  `18/18` default-policy oracle rows, and `9/9` BMM rows; median BMM gains are
  `1.2002x/1.1426x/1.1259x`. The follow-up latest-checkpoint B1/B8 matrix
  passes all `36/36` parameter-adjusted Prefill and `36/36` Decode cells
  against verified full-FLA/Triton-conv Qwen3.5; minima are
  `1.108265x/4.158943x`. Its only initial red shape, 1.5B/B1/P2048, now uses
  exact-card self-chunk tile16 plus stacked R/K/V at `1.2539x` the prior route
  with Prompt/cache-handoff cosine `>=0.9999940` and greedy equality. Evidence:
  [`../bench/4090_4080_routes_20260812/README.md`](../bench/4090_4080_routes_20260812/README.md)
  and
  [`../bench/4090_adjusted_pd_20260812/README.md`](../bench/4090_adjusted_pd_20260812/README.md).
- **RTX 5090:** the full-FLA Qwen3.5 matrix passes 8/8 B1/B8 batch-pairs,
  144/144 cells and 32/32 correctness reports from 0.4B/0.8B through 7.2B/9B;
  raw prefill/decode minima are `1.0226x/2.8130x`; RWKV-7 7.2B versus
  Qwen3.5-9B B1/B8 minima are `1.1739x/1.0309x` prefill and
  `2.8934x/2.8130x` decode. The Native/no-FLA g1h 7.2B same-precision v3a
  comparison passes B1/B8 decode at `1.0010x/1.0104x`, while g1h 2.9B/13.3B
  prefill passes 12/12 B1/B8 prompt128/512/2048 cells at
  `1.0029x–1.5690x`. Full 0.4B MATH500 `500×64` reaches pass@64 `0.38` and
  committed Albatross summary/decode ratios `4.336x/4.871x`. The latest
  official g1h 13.3B checkpoint passes conversion and load/generate. The
  exact-model BN/TN W4 matrix passes official g1h 1.5B/2.9B/7.2B/13.3B at
  B1/B8 with minimum `1.0010x/1.1854x` prefill/decode,
  `0.5298x–0.6250x` footprint, cosine `>=0.9995`, same-next 8/8 and 280/280
  group-128 grid checks. Evidence:
  [`../bench/5090_g1h_qwen35_b1_b8_20260715/README.md`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
  [`../bench/5090_g1h_13p3_20260715/README.md`](../bench/5090_g1h_13p3_20260715/README.md),
  [`../bench/5090_bntn_all_models_20260716/README.md`](../bench/5090_bntn_all_models_20260716/README.md),
  and [`../bench/5090_native_official_fp16_production_20260718/README.md`](../bench/5090_native_official_fp16_production_20260718/README.md).
- Correctness gates include official/HF alignment, cosine/top-k/greedy checks,
  cache handoff, save/reload, MATH500 shape/accuracy gates and logit-compression
  alignment.

Canonical numbers: [`../BENCHMARK.md`](../BENCHMARK.md) and
[`PERFORMANCE.md`](PERFORMANCE.md).

### 2. Transformers, PEFT and RL libraries

Validated interfaces include:

- `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`;
- `generate(use_cache=True)`, labels/loss, attention masks and save/reload;
- PEFT LoRA forward/backward, adapter save/load/merge;
- HF Trainer and checkpoint resume;
- TRL SFTTrainer, DPOTrainer and GRPOTrainer;
- opt-in RTX 5090 BF16 `train_temp_cuda` lanes with exact official backward and
  FusedAdam-step parity; the Native B16/T512 lane also passes paired real-
  MiniPile three-seed convergence, continuous 5,000-step, 2,500+2,500 resume
  and steady-memory gates at `1.00049x` paired-seed median and `1.00255x`
  continuous-run throughput;
- native/no-FLA fallback for compatibility-focused environments.

Training details: [`TRAINING.md`](TRAINING.md). Exact official-kernel usage and
scope: [`TRAIN_TEMP_CUDA.md`](TRAIN_TEMP_CUDA.md).

### 3. HF state-cache and serving-like behavior

The HF adapter exposes recurrent state-cache operations, chunked-prefill
correctness tests, dynamic batch select/reorder behavior and telemetry. These
are the HF compatibility primitives required by serving adapters; they do not
replace native vLLM or SGLang scheduler implementations.

### 4. Hardware support

See the canonical [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md). A card is marked
production-close only when commands, environment, correctness and performance
rows are preserved; load-only smoke is not promoted to that status.

### 5. Quantization

W8/W4 loading and generation work and lower stored/model footprint. Native
speed and memory policies are deliberately separate. The speed lane is closed
on promoted V100/T4/4080/4090/5090 and Apple profiles; RTX 4090 has batch-8
evidence for every published 0.4B–7.2B pair. Wider full-memory profiles are
post-release optimization work. See [`QUANTIZATION.md`](QUANTIZATION.md).

## Release decision

`v0.6.0` is the completed public HF adapter milestone: API, training ecosystem,
cache helpers, conversion, quantized functionality, Native/no-FLA execution,
parallelism boundaries, and reproducible hardware evidence are present.

Completion does not authorize an unbounded claim that every W8/W4 shape on
every future card is faster than fp16, or that every hardware product has run
the same Albatross matrix. Claims remain limited to the promoted profiles.
