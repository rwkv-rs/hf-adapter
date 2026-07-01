# RTX 4080 reference-vs-FLA speed diagnostic — 2026-08-26

This is a non-blocking FP16 throughput diagnostic. It compares the final
reference model code from `559a07f2faa458be11fa67a8b36f636ec5b626ba`
against FLA commit `80e494f6c588e091fc8316b612870df29375c5b8` using
the same converted 0.4B checkpoint and inputs.

Environment: RTX 4080, PyTorch 2.11.0+cu130, Transformers 5.8.0, Triton
3.6.0. CUDA graphs and `torch.compile` were disabled. Each number is the
median of five synchronized measurements after two warmups; Triton compilation
is therefore excluded. The checkpoint weight SHA256 is
`8aa0fb580a0c5d442b28f63b9b08c2c60a81821f7c60e79307c11a1a3a2693e0`.

## End-to-end model

| case | reference | FLA | FLA speedup |
|---|---:|---:|---:|
| prefill B1 T128 | 225.95 ms / 567 tok/s | 23.28 ms / 5,497 tok/s | **9.70x** |
| prefill B1 T512 | 846.00 ms / 605 tok/s | 24.45 ms / 20,944 tok/s | **34.61x** |
| prefill B1 T2048 | 3,342.54 ms / 613 tok/s | 104.49 ms / 19,600 tok/s | **31.99x** |
| prefill B4 T128 | 292.71 ms / 1,749 tok/s | 24.15 ms / 21,203 tok/s | **12.12x** |
| prefill B4 T512 | 1,122.47 ms / 1,825 tok/s | 103.06 ms / 19,871 tok/s | **10.89x** |
| cached decode B1 | 21.44 ms/step / 46.6 tok/s | 15.28 ms/step / 65.5 tok/s | **1.40x** |
| cached decode B4 | 21.89 ms/step / 182.7 tok/s | 15.87 ms/step / 252.0 tok/s | **1.38x** |

## WKV operator only

| case | reference | FLA | FLA speedup |
|---|---:|---:|---:|
| B1 T128 | 5.911 ms | 0.236 ms | **25.00x** |
| B1 T512 | 23.615 ms | 0.230 ms | **102.79x** |
| B1 T2048 | 94.330 ms | 0.314 ms | **300.51x** |
| B4 T128 | 8.201 ms | 0.225 ms | **36.51x** |
| B4 T512 | 32.628 ms | 0.225 ms | **144.98x** |

The operator gap is much larger than the end-to-end gap because embeddings,
projections, normalization and the LM head are common model costs. FLA's
parallel chunk algorithm is most advantageous for prefill. Cached decode is
already recurrent and dominated by the rest of the model, so its measured
gain is about 1.4x.

These numbers are not a correctness verdict. The pinned FLA implementation
emits its own potentially-buggy warning, and the separate numerical bundle
records full-model deviations. Official RWKV checkpoint parity remains the
correctness oracle.

Raw synchronized samples, memory measurements and the exact environment are
in [`fp16.json`](fp16.json).
