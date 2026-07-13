# V100 full-memory native MM8/MM4 matrix

Date: 2026-07-13

Hardware: two `Tesla V100-PCIE-32GB` cards, exact `sm_70`.

This artifact is the complete fresh-process matrix for the default-off native
quant FFN experiments on V100. It covers:

- models: 1.5B, 2.9B, and 7.2B;
- seven cells per model: bsz1/2/4/8 prompt128/decode128, bsz1
  prompt512/2048/decode128, and bsz1 prompt128/decode512;
- fp16, MM8 off/up/deep, and MM4 off/up;
- three timing repeats, paired or cached exact-shape fp16 baselines, footprint,
  prompt/final logits cosine, and greedy-token comparison.

Both shards completed without infrastructure failures: `66 + 60 = 126/126`
rows. This means the execution matrix is complete; it does not mean every
quant path passes the performance and correctness acceptance gate.

## Acceptance result

| Model | Path | Speed >= fp16 | Decode / fp16 range | Footprint / fp16 | Greedy |
|---|---|---:|---:|---:|---:|
| 1.5B | MM4 up | 7/7 | `1.0553x-1.1951x` | `0.5389x` | 6/7 |
| 2.9B | MM4 up | 7/7 | `1.0415x-1.2564x` | `0.5306x` | 6/7 |
| 7.2B | MM4 up | 7/7 | `1.2238x-1.9110x` | `0.3010x` | 4/7 |
| 1.5B | MM8 up | 0/7 | `0.1824x-0.4143x` | `0.6932x` | 6/7 |
| 2.9B | MM8 up | 0/7 | `0.1746x-0.4394x` | `0.6876x` | 6/7 |
| 7.2B | MM8 up | 0/7 | `0.1123x-0.4288x` | `0.5346x` | 5/7 |

MM4 off also beats fp16 in all `21/21` model/cell pairs. The fused FFN-up
epilogue wins 18/21 comparisons versus MM4 off and stays within `0.9964x` in
the remaining cells. MM4 is therefore a complete V100 speed/footprint result,
but not a promoted quant path because greedy diverges at 1.5B/2.9B bsz8 and
7.2B bsz2/4/8. Off and up produce the same mismatches, so the fusion epilogue
is not the source of the quality gap.

MM8 fails the speed gate in all `63/63` MM8 rows. FFN-up fusion is only a
small local improvement, and the deeper down+residual route is negative:
deep versus up has median `0.9470x` and only one win in 21 cells. Keep both
MM8 fusion flags disabled on Volta.

## Rejected W4A16 batch probe

The exact-sm70 MM4 runtime uses W4A16 for bsz1 and dynamic A8xW4 DP4A for
bsz2/4/8. A default-off batched W4A16 prototype matched the dequantized linear
reference, but did not restore end-to-end greedy equality and lost the speed
gate: 1.5B bsz8 reached `0.9210x` fp16 and 7.2B bsz8 reached `0.8949x`.
The code was removed; the two negative rows are retained in
`w4a16_batch_rejected.jsonl`.

The remaining MM4 quality work requires a better weight format, such as
groupwise/K-style W4, rather than replacing activation quantization alone.
The remaining MM8 performance work requires a true Volta W8A16 or a deeper
fused projection path; the existing A8W8 prototype is not a useful fallback.

All fused quant flags remain default-off. No V100 quant path is promoted by
this artifact.
