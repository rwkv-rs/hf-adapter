# RTX 4080 dynamic prefill full matrix (G1i 1.5B)

- Card: NVIDIA GeForce RTX 4080 (exact-card gate)
- Model: RWKV-7 G1i 1.5B HF FP16
- Grid: B=1..8 and T={31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049}
- Correctness reference: `native-direct`

`baseline_b1348.jsonl` and `baseline_b1348.log` retain the preliminary
B1/B3/B4/B8 P128/P512 route screen that motivated the continuous-shape
matrix. They are supporting evidence only; the acceptance decision is based on
the full matrix below.

`dynamic_prefill.jsonl` is the immutable 168-shape first pass. It found one
bounded-profile error: B8/T2049 was eight total rows above the former 16384
limit and reverted to the unfused path. `summary-before-fix.json` records that
failure.

The policy limit was corrected to cover the complete B<=8,T<=4096 rectangle
(32768 total rows). `patch_b8_t2049.jsonl` is the isolated post-fix rerun.
`summary.json` overlays that rerun on the immutable full pass and is the final
acceptance result: 168/168 shapes, zero failures.
