# HF adapter acceptance status

This is the canonical mapping between the public RWKV-7 Hugging Face adapter
requirements and repository evidence. `PASS` means the named gate has a
reproducible artifact; `PARTIAL` means the interface works but the complete
hardware/performance matrix is not closed.

Last updated: **2026-07-16**.

## Executive result

| Requirement | Status | Current evidence | Remaining boundary |
|---|---|---|---|
| RWKV-LM / Albatross correctness and performance | **PARTIAL / production-close on measured V100, 4090 and 5090 lanes** | V100 Albatross/native-quant matrix plus 1.5B/full-FLA-Qwen B1/B8 active-work close; 4090 Albatross lane plus 0.4B–7.2B bsz8 Qwen3.5 matrices; 5090 full-FLA Qwen B1/B8, MATH500, quant pressure, and latest g1h 13.3B artifacts | Same-card final Albatross reruns on every target, broader optimized-Qwen shapes/cards, larger-model P2/P3 matrix, historical 4090 prefill high-water mark |
| Transformers API | **PASS** | Auto classes, save/reload, generation, labels/loss, attention mask and recurrent cache tests | Upstreaming and long-term Transformers-version maintenance |
| PEFT and RL ecosystem | **PASS for smoke/compatibility** | LoRA lifecycle, Trainer, SFT, DPO and GRPO smoke across CUDA and Apple/MPS | Longer production training and broader model/card combinations |
| Dynamic batching, chunked prefill and state cache helpers | **PASS in HF adapter scope** | State select/reorder/drop/compact, chunked-prefill parity, serving-like cache telemetry | Native vLLM/SGLang integration remains a separate repository/project |
| Common professional and consumer cards | **PARTIAL** | V100, A100, A800, A6000, 4090, 5090, GTX 1080 Ti and Apple M5 evidence | H100, AMD/ROCm, Turing and broader Apple/50-series coverage |
| W8/W4 inference and lower memory | **PASS functionally; PARTIAL for universal speed** | bnb compatibility plus native MM8/MM4; 5090 speed-policy pressure close; Apple MLX W4 | Full-memory quantized projections must become fp16-or-faster across cards and shapes |
| PP/TP inference | **PARTIAL** | HF `device_map`/multi-device smoke and pipeline-oriented path | Production TP and a complete PP matrix are not closed in this HF repository |
| ZeRO-2/3 training | **PASS for current smoke matrix** | ZeRO-2/3 base and resume evidence on V100/A100/A800/A6000 combinations | Longer training and larger ZeRO-3 resume matrix |
| Initial speculative decoding | **PASS as experimental HF/Apple path** | HF-compatible target/draft harness and Apple target-greedy oracle evidence | Serving integration and broader quality/speed gates |

## How to report completion

The **current HF milestone is complete**, and the repository is suitable for a
public HF-adapter milestone under the boundaries below. The broader universal
requirement remains `PARTIAL`: full-memory quant speed, every target hardware
family, wider Albatross matrices, and production PP/TP are not all closed.

There is no official repository-wide completion percentage. Report the named
scope and its status instead; do not estimate a percentage from TODO checkboxes
or by counting the table rows above.

## Official requirement mapping

### 1. Performance, speed, accuracy and memory

- **V100:** 0.1B/0.4B/1.5B × bsz1/2/4/8 production-close matrix is
  promoted. Dense decode is `0.908x–1.248x` and prompt-512 prefill is
  `0.930x–1.047x` of same-host Albatross references. Separately, target-only
  RWKV-7 1.5B versus full-FLA/Triton-conv Qwen3.5-2B passes B1/B8 raw
  prefill/decode minima `2.815921x/5.270432x` and active-parameter work minima
  `2.285574x/4.277804x`; the B1 peak-VRAM loss remains disclosed. Evidence:
  [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md).
- **RTX 4090:** 0.4B dense decode bsz1/2/4/8 reaches
  `1.007x/1.016x/1.008x/1.418x` of matching Albatross rows. Prompt-512 bsz4 is
  `1.007x` the same-session reference and `0.916x` the retained historical
  high-water reference. Separately, all published 0.4B/1.5B/2.9B/7.2B pairs
  pass the batch-8 dense/W8/W4 Qwen3.5 contract: `54/54` small-model cells and
  `18/18` 7.2B cells, with full-FLA, dense decode active-work, quant speed and
  quant-local physical-memory gates.
- **RTX 5090:** the full-FLA Qwen3.5 matrix passes 8/8 B1/B8 batch-pairs,
  144/144 cells and 32/32 correctness reports from 0.4B/0.8B through 7.2B/9B;
  raw prefill/decode minima are `1.0226x/2.8130x`. Full 0.4B MATH500 `500×64`
  reaches pass@64 `0.38`; against the committed Albatross reference,
  summary/decode throughput ratios are `4.336x/4.871x`. The latest official
  g1h 13.3B checkpoint also passes conversion, load/generate, and selected
  speed-policy MM8/MM4 gates. The MATH500 reference is not a fresh same-card
  Albatross rerun. Evidence:
  [`../bench/5090_g1h_qwen35_b1_b8_20260715/README.md`](../bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
  and [`../bench/5090_g1h_13p3_20260715/README.md`](../bench/5090_g1h_13p3_20260715/README.md).
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
- native/no-FLA fallback for compatibility-focused environments.

Training details: [`TRAINING.md`](TRAINING.md).

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
on selected V100/4090/5090 shapes; RTX 4090 now has batch-8 evidence for every
published 0.4B–7.2B pair. The full-memory lane remains the main kernel work
item. See [`QUANTIZATION.md`](QUANTIZATION.md).

## Release decision

The repository is suitable for a public HF adapter milestone: API, training
ecosystem smoke, cache helpers, conversion, quantized functionality and
reproducible hardware evidence are present. It must not yet claim that every
W8/W4 shape on every supported card is faster than fp16, or that every hardware
family has completed the same Albatross matrix.
