# RTX 4080 native-JIT split regression

Date: 2026-07-28

This artifact validates the complete `native_jit.py` structural split at
candidate `39502af` against the unchanged code at repository anchor `0e85048`
(`15f4bf8` is the effective baseline code commit). It is a refactor gate, not
a new kernel-performance claim.

The split reduces the runtime facade from 4,635 to 794 lines. Dense steps,
packing, graph dispatch, prefill policy, prefill execution, recurrent math and
decode execution now have separate owners while historical facade symbols
remain available.

## Environment

- GPU: NVIDIA GeForce RTX 4080, `sm_89`, 16,376 MiB
- driver: `595.71.05`
- Python 3.11.15
- PyTorch `2.6.0+cu124`, CUDA runtime `12.4`
- Transformers `5.12.1`, Triton `3.2.0`, bitsandbytes `0.49.2`
- model: RWKV-7 g1g 1.5B, fp16
- prompt: 128 tokens; decode: 128 tokens; batch sizes: 1 and 8

Candidate runs bracketed the baseline run. The regression floor was `-2%`,
with exact greedy traces, effective backends and peak-VRAM samples required.

## Results

Real-CUDA split and fused BnB tests pass: **35 passed**. The HF remote-code BnB
smoke also passes for both W8 and W4 through load, forward, cached decode and
greedy generation.

| Path | Batch | Baseline tok/s | Candidate median tok/s | Delta |
|---|---:|---:|---:|---:|
| native-JIT decode | 1 | 154.055 | 152.135 | -1.25% |
| native-JIT decode | 8 | 924.135 | 923.935 | -0.02% |
| native-graph decode | 1 | 201.465 | 201.535 | +0.03% |
| native-graph decode | 8 | 1,212.620 | 1,210.945 | -0.14% |
| native-graph prefill | 1 | 14,230.6 | 14,238.6 | +0.06% |
| native-graph prefill | 8 | 25,696.2 | 25,668.2 | -0.11% |

All decode greedy trace hashes are identical. Prefill and the first decode
after prefill have `max_abs_diff=0`, the effective backends are unchanged, and
the repeated peak-VRAM samples match exactly. Native-graph cache hit rate is
`99.24%` with no eviction.

The 0.4B BnB smoke records W8/W4 footprints of 571.8/427.8 MiB and peak VRAM of
614.4/485.8 MiB, with an identical generated tail for both modes.

## Files

- `summary.json`: machine-readable A/B medians, deltas and gates.
- `environment.json`: hardware, software and commit provenance.
- `baseline_decode.jsonl`, `candidate_decode.jsonl`: bracketed B1/B8 decode.
- `baseline_prefill.jsonl`, `candidate_prefill.jsonl`: 25-sample prefill rows.
- `candidate_cuda_split_tests.log`: real-CUDA test result.
- `candidate_bnb_smoke.log`: HF W8/W4 smoke.
- `ab_integration.log`: complete raw integration transcript.
- `run_*.sh`: exact remote commands.
- `SHA256SUMS`: artifact integrity hashes.
