# RTX 3090 max-performance promotion probes

These JSONL files are the raw screening and alternating-order A/B evidence
used before the full matrix was promoted:

- `2p9_b8p128_breakdown.jsonl`: per-component 2.9B B8/P128 hotspot profile.
- `0p4_remaining_ab.jsonl`, `1p5_remaining_ab.jsonl`, and
  `2p9_b1p128_remaining_ab.jsonl`: fp32/fp16 GEMM-accumulation shape probes.
- `2p9_b8p128_accum_ab.jsonl`: alternating off/on/on/off confirmation for the
  former weakest strict cell.
- `scan_2p9_focus.jsonl`: recurrent-scan microbenchmark tile sweep.
- `2p9_b8p128_scan_ab.jsonl` and `2p9_p512_scan_probe.jsonl`: full-model
  paired confirmation for the selected 2.9B scan tiles.

These probes inform routing, but the parent directory's complete 24-cell
matrix and 25-row correctness gate are the promotion authority.
