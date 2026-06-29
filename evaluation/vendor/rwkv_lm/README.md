# Pinned official RWKV-7 reference sources

These files are unmodified snapshots from
[`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM) commit
`524481d5099b38d9bc8ef1e89209161b86c8011b` (2026-08-21).

| file | upstream blob | SHA256 |
|---|---|---|
| `rwkv_v7_numpy.py` | `RWKV-v7/rwkv_v7_numpy.py` | `dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6` |
| `rwkv_v7_demo_rnn.py` | `RWKV-v7/rwkv_v7_demo_rnn.py` (`bc8e6b5974c1e90005e7091b7414be79f29ea887`) | `a61f35716b2ef81fa1c97bfd7f67bccd78d3a8968d0570748d4631fecf885500` |

`evaluation/official_rwkv7_oracle.py` adapts the token-wise equations to a
batched, provenance-recording harness. The original snapshots and Apache-2.0
license remain here so the oracle can be audited against upstream.
