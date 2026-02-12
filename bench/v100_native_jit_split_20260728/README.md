# Tesla V100 native-JIT split regression

Date: 2026-07-28

This artifact validates the complete `native_jit.py` structural split at
candidate `39502af` against repository anchor `0e85048` on a Tesla V100. The
effective baseline code is `15f4bf8`; the later anchor adds evidence only.

## Environment

- GPU: Tesla V100-PCIE-32GB, `sm_70`, device 0
- driver: `580.173.02`
- Python 3.11.15
- PyTorch `2.5.1+cu124`, CUDA runtime `12.4`
- Transformers `5.12.1`, Triton `3.4.0`, bitsandbytes `0.49.2`
- model: complete RWKV-7 g1d 0.1B checkpoint, fp16
- prompt: 128 tokens; batch sizes: 1 and 8

The available server-side 0.4B safetensors file was rejected by `safe_open`
before model construction because its metadata did not cover the complete
file. The acceptance run therefore uses the server's complete 0.1B checkpoint;
`server_0p4b_checkpoint_probe.log` preserves that independent data issue.

## Results

Real-CUDA split and fused BnB tests pass: **35 passed**. HF remote-code W8 and
W4 both pass load, forward, cached decode and generation. W8/W4 footprints are
283.4/242.9 MiB and peak VRAM is 309.6/273.9 MiB.

The standard candidate-baseline-candidate run produced:

| Path | Batch | Baseline tok/s | Candidate median tok/s | Delta |
|---|---:|---:|---:|---:|
| native-JIT decode | 1 | 99.845 | 100.895 | +1.05% |
| native-JIT decode | 8 | 755.415 | 762.385 | +0.92% |
| native-graph decode | 8 | 4,220.965 | 4,221.350 | +0.01% |
| native-graph prefill | 1 | 27,286.6 | 27,277.8 | -0.03% |
| native-graph prefill | 8 | 108,613.0 | 108,436.9 | -0.16% |

The first short native-graph B1 sample showed `-2.66%`, so it was not accepted
as-is. A 1,024-token, 5-baseline/10-candidate bracketed rerun gave:

| Path | Baseline median tok/s | Candidate median tok/s | Delta |
|---|---:|---:|---:|
| native-graph decode B1 | 775.730 | 777.085 | +0.17% |

All decode greedy trace hashes and effective backends are identical. Peak-VRAM
sample sequences match exactly. Native-graph hit rate is `99.62%` in the
standard run and `99.90%` in the long rerun, with no eviction. Prefill and the
first decode after prefill retain greedy agreement with the HF reference and
the same fp16 numerical envelope before and after the split.

## Files

- `summary.json`: machine-readable standard and long-rerun gates.
- `environment.json`: hardware, software, checkpoint and commit provenance.
- `baseline_decode.jsonl`, `candidate_decode.jsonl`: standard B1/B8 decode.
- `baseline_graph_b1_long.jsonl`, `candidate_graph_b1_long.jsonl`: long B1
  native-graph rerun.
- `baseline_prefill.jsonl`, `candidate_prefill.jsonl`: 50-sample prefill rows.
- `candidate_cuda_split_tests.log`: real-CUDA result.
- `candidate_bnb_smoke_0p1b.log`: HF W8/W4 smoke.
- `ab_integration.log`, `graph_b1_long.log`: raw transcripts.
- `run_*.sh`: exact remote commands.
- `SHA256SUMS`: artifact integrity hashes.
