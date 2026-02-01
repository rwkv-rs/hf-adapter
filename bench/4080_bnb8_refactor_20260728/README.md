# RTX 4080 BnB W8 refactor regression

Date: 2026-07-28

This artifact compares the BnB W8 helper split at candidate commit `15f4bf8`
against its direct parent `89bc457` on one NVIDIA GeForce RTX 4080 (`sm_89`).
It is a refactor regression gate, not a new BnB speed-promotion result.

## Environment

- Python 3.11 on Linux
- PyTorch `2.6.0+cu124`
- CUDA runtime `12.4`
- Transformers `5.12.1`
- bitsandbytes `0.49.2`
- Triton `3.2.0`
- model: RWKV-7 g1g 1.5B, fp16 loading dtype

The direct BnB W8 inference, ReLU-squared activation quantization, R/K/V mix
quantization and FFN mix quantization flags were enabled explicitly, with
`RWKV7_BNB_INT8_THRESHOLD=0`, so the extracted helpers were eligible in the
native prefill path.

## Results

Real-CUDA helper and split tests pass: `10 passed`. The 0.4B HF remote-code
smoke also passes before and after the split with identical next token,
generated tail, module counts, model footprint and peak VRAM.

For the 1.5B BnB8 path, all B1/B8 rows have finite logits, exact greedy-token
agreement, `max_abs_diff=0`, and unchanged memory:

| Batch | Metric | Parent | Candidate median | Delta |
|---:|---|---:|---:|---:|
| 1 | prefill tok/s | 3,993.9 | 4,035.8 | +1.05% |
| 8 | prefill tok/s | 31,756.1 | 31,619.6 | -0.43% |
| 1 | decode tok/s | 48.287 | 47.960 | -0.68% |
| 8 | decode tok/s | 374.117 | 372.321 | -0.48% |
| 1 | peak VRAM MiB | 1,814.3 | 1,814.3 | 0.00% |
| 8 | peak VRAM MiB | 2,115.4 | 2,115.4 | 0.00% |

The candidate was sampled on both sides of the parent run. All observed
throughput movement is within 1.1%, with no directional regression across
prefill and decode. The model payload is exactly `1761.3 MiB` for both commits.

The standard full-model BnB8 decode remains on the existing fail-safe eager
backend. Relative to the candidate fp16/native-graph control, BnB8 reduces the
model footprint from `2913.3 MiB` to `1761.3 MiB` (`-39.54%`) but reaches only
`0.2372x` fp16 decode throughput at B1 and `0.3065x` at B8. This is an existing
full-model BnB performance gap; the structural split neither introduces nor
claims to solve it.

## Files

- `environment.json`: software, GPU and commit provenance.
- `cuda_tests.log`: real-CUDA helper/split test result.
- `parent_bnb8_smoke.log`, `candidate_bnb8_smoke.log`: HF BnB8 load and
  generation smoke.
- `parent_bnb8_prefill_long.jsonl`, `candidate_bnb8_prefill_long.jsonl`:
  25-sample prefill correctness and timing rows.
- `parent_bnb8_decode.jsonl`, `candidate_bnb8_decode.jsonl`: B1/B8 cached
  decode rows, with the candidate bracketing the parent run.
- `candidate_fp16_decode.jsonl`: same-card fp16 memory and throughput control.
- `SHA256SUMS`: artifact integrity hashes.
