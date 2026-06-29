# RWKV-7 HF v0.9 status

The main line is the readable pure-PyTorch reference implementation. Historical
performance backends and evidence are preserved on `perf/native-kernels-v0.8`.

Release gates:

- [x] clean config, cache, operator and modeling structure
- [x] package-free save/load and reference conversion layout
- [x] tiny cache, padding, generation, loss and gradient tests
- [x] first RTX 4080 0.4B FP16 clean-vs-FLA run archived
- [ ] pinned-FLA full V100 and RTX 4080 matrix
- [ ] official checkpoint oracle matrix
- [ ] formal 48-unit lm_eval run
- [ ] canonical SFT, DPO and GRPO runs
- [x] clean wheel build, metadata check, isolated install and CLI/model smoke
- [ ] six model repositories updated and tagged v0.9.0
- [ ] PyPI 0.9.0 published

A release is not complete until every unchecked item passes.

The first RTX 4080 bundle is preserved in
[`results/4080-reference-20260825`](results/4080-reference-20260825/README.md).
The operator matrix, cached teacher decode, and 64-token greedy comparison
passed, but the full-model FP16 B=4/T=1 and B=4/T=128 cases exceeded the fixed
0.15 max-absolute tolerance. This is evidence from a **failed pre-release
gate**, not release evidence. The tolerance was not relaxed.
