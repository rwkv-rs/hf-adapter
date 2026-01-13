<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  benchmark_methodology: BENCHMARK.md
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# RWKV-7 vLLM production acceptance

## Status meaning

This file defines the gate for a future vLLM/SGLang implementation. Its
presence does not claim that such an integration exists in this repository.

Use:

- **PASS** — command, raw artifact, environment, and threshold all present;
- **FAIL** — measured and below threshold;
- **UNTESTED** — no qualifying evidence;
- **N/A** — architecture genuinely does not require the feature.

Do not convert feature existence into a production PASS.

## Evidence bundle

Every accepted row needs:

```text
commit SHA
engine/runtime versions
model/checkpoint hash
tokenizer hash
GPU name, compute capability, driver, CUDA/ROCm
dtype and quant format
batch/request distribution
prompt/decode lengths
warmup and measured iterations
raw latency/throughput samples
peak VRAM and packed footprint
correctness metrics
exact reproduction command
```

Separate cold compile/load, warm TTFT, prefill throughput, inter-token latency,
and aggregate decode throughput.

## Gate 1: configuration and loading

- [ ] Native converted `config.json` parses without remote code.
- [ ] `A == H*N`; `A != D` test model is included.
- [ ] Embedding and output head remain untied.
- [ ] Every safetensors key is consumed exactly once or explicitly ignored.
- [ ] Missing/unexpected keys fail with a useful error.
- [ ] Official `.pth` → HF conversion → vLLM load produces the same tensor
      shapes.
- [ ] Sharded safetensors load without an extra full-model host/GPU copy.
- [ ] Tokenizer IDs match the HF tokenizer on a multilingual/adversarial set.

## Gate 2: operator correctness

Test FP32 first, then target FP16/BF16:

- [ ] time-mix vectors and six mixed inputs;
- [ ] W/A/G/V low-rank paths;
- [ ] key normalize/adaptation;
- [ ] decay;
- [ ] recurrent `S_new`;
- [ ] recurrent readout, group norm, and skip term;
- [ ] FFN ReLU-square;
- [ ] complete block residuals;
- [ ] final norm/head.

Record max absolute error and relative L2 for intermediates and every state
component. Logit cosine alone is not sufficient.

## Gate 3: model parity

Required references:

1. official RWKV-LM/NumPy for a small deterministic case;
2. `NativeRWKV7ForCausalLM` at matching checkpoint/dtype;
3. vLLM implementation.

Cases:

| Case | Required |
|---|---|
| B1 prompt 1/16/128/512/2048 | yes |
| B8 equal length | yes |
| ragged B8 | yes |
| Unicode/multilingual tokenizer inputs | yes |
| long greedy decode >=512 tokens | yes |
| state resume after prefill | yes |

Report:

```text
prompt-final logit cosine/maxabs
same-next-token
greedy trace equality or first divergence
final state error per component/layer
```

Thresholds must be set per dtype against matching native HF behavior. Any
relaxation requires a quality evaluation, not only a larger tolerance.

## Gate 4: dynamic batching

- [ ] New requests join while old requests continue.
- [ ] Completed and cancelled requests leave without affecting survivors.
- [ ] Request row order changes every step.
- [ ] Non-contiguous state slots work.
- [ ] Duplicate/forked requests diverge independently after the fork.
- [ ] Mixed prompt/decode phases can coexist according to engine policy.
- [ ] Results equal independent B1 runs for every request.
- [ ] Slot reuse after cancellation has no prior-request leakage.
- [ ] Stale generation handles cannot write to reused slots.

Stress test at least thousands of allocate/free/reorder cycles.

## Gate 5: chunked prefill

Compare sequential prefill against chunk sizes:

```text
1, 16, 32, 64, 128, 256
```

Unsupported sizes may use fallback but must remain correct.

- [ ] final state parity;
- [ ] final-token logit parity;
- [ ] requested prompt logprobs parity;
- [ ] partial last chunk;
- [ ] ragged packed requests;
- [ ] prompt shorter than chunk;
- [ ] resumed prefill from an existing state;
- [ ] no cross-request prefix scan;
- [ ] memory scales with chosen workspace contract.

## Gate 6: prefix-state cache

- [ ] exact hit equals full prefill;
- [ ] miss executes full prefill;
- [ ] partial-prefix hit processes only suffix;
- [ ] two requests hitting one entry receive independent mutable states;
- [ ] model/tokenizer/adapter/quant mismatch rejects the entry;
- [ ] hash collision handling is defined;
- [ ] CPU-offloaded restore waits for copy completion;
- [ ] eviction during live use is safe;
- [ ] metrics report hits, misses, saved tokens, bytes, and restore latency.

Measure hit rate on a declared workload; no universal target is meaningful
without a workload distribution.

## Gate 7: CUDA Graph/state pool

- [ ] graph buckets cover target decode batch sizes;
- [ ] padded rows use scratch slots, not duplicate live slots;
- [ ] inactive rows do not mutate state or counters;
- [ ] graph replay survives request reorder;
- [ ] graph runner invalidates on pool relocation/layout change;
- [ ] eager and graph outputs/states match;
- [ ] simultaneous buckets do not share corruptible scratch state.

## Gate 8: quantization

For every W8/W4 format and card:

- [ ] pack/dequant unit parity;
- [ ] packed checkpoint/load round-trip;
- [ ] exact module allowlist recorded;
- [ ] no unintended dense shadow weights;
- [ ] physical packed footprint lower than W16;
- [ ] end-to-end peak VRAM recorded;
- [ ] prompt/decode logits and same-next pass;
- [ ] long greedy trace or quality suite passes;
- [ ] prefill speed >= matching W16 for a universal speed claim;
- [ ] decode speed >= matching W16;
- [ ] unsupported shape/card fails closed;
- [ ] fused epilogues apply exactly once.

Do not label Marlin packing as GPTQ/OBQ calibration.

## Gate 9: performance

Measure at minimum:

```text
batch/concurrency: 1, 2, 4, 8, 16, 32 where memory allows
prompt lengths:    1, 128, 512, 2048
decode lengths:    1, 64, 256
models:            small correctness model plus representative production sizes
```

Metrics:

```text
cold load/compile
TTFT p50/p90/p99
prefill tokens/s
ITL/TPOT p50/p90/p99
aggregate decode tokens/s
per-request output tokens/s
scheduler queue time
state gather/scatter time and bytes
prefix saved tokens
packed footprint
idle, prefill, and decode peak VRAM
```

Compare on the same host/process policy against:

- native HF adapter;
- current RWKV-LM/Albatross reference where available;
- optimized engine baseline for scheduler overhead.

Do not compare different GPUs, clocks, precisions, prompt shapes, or active
parameter counts as if they were the same row.

## Gate 10: hardware matrix

Minimum target families:

| Family | Representative |
|---|---|
| Pascal fallback | GTX 10/P100 class where supported by runtime |
| Volta | V100 |
| Turing | T4 |
| Ampere consumer | RTX 3090 |
| Ampere datacenter | A100/A800 |
| Ada | RTX 4080/4090 |
| Hopper | H100 |
| Blackwell | RTX 5070/5090 and datacenter when available |
| AMD ROCm | at least one supported datacenter/consumer target |

One card does not validate an entire family when exact schedules are used.
Every promoted card needs capability-gated dispatch and an unknown-device
fallback.

## Gate 11: TP/PP

### TP

- [ ] TP1 reference;
- [ ] TP2 and TP4 where shape allows;
- [ ] local recurrent state head partition;
- [ ] low-rank collective correctness;
- [ ] FFN and output-head collectives;
- [ ] quantized TP checkpoint loading;
- [ ] mixed-length dynamic batching;
- [ ] output/state parity with TP1;
- [ ] no full-weight replication outside declared modules.

### PP

- [ ] PP2 minimum;
- [ ] layer-local state ownership;
- [ ] current-token `v_first` transfer;
- [ ] per-request token order;
- [ ] cancellation frees slots on every stage;
- [ ] dynamic batching and chunked prefill;
- [ ] output/state parity with single device.

## Gate 12: reliability

- [ ] one-hour mixed workload without state corruption;
- [ ] repeated model load/unload without leaked extensions/workspaces;
- [ ] cancellation during prefill/decode;
- [ ] out-of-memory recovery;
- [ ] malformed checkpoint/quant metadata rejection;
- [ ] deterministic replay under fixed sampling seed where supported;
- [ ] no request data survives slot reset/reuse;
- [ ] metrics and error messages identify fallback/selected kernel.

## Production completion rule

A production claim requires all applicable correctness, scheduler, memory,
reliability, and target-hardware rows. Performance claims are scoped to the
exact recorded model, shape, dtype, quantization, card, and engine version.

Passing a feature smoke test means only that the feature runs. It does not
establish production performance or universal hardware support.
