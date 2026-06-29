# RWKV-7 HF v0.9 status

The main line is the readable pure-PyTorch reference implementation. Historical
performance backends and evidence are preserved on `perf/native-kernels-v0.8`.

Release gates:

- [x] clean config, cache, operator and modeling structure
- [x] package-free save/load and reference conversion layout
- [x] tiny cache, padding, generation, loss and gradient tests
- [x] first RTX 4080 0.4B FP32/FP16/BF16 clean-vs-FLA runs archived
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
All three dtype gates failed: FP16 exceeded the fixed logit error bound, BF16
missed cosine/greedy parity, and pinned FLA's unsupported FP32 path missed the
strict operator/model tolerances. This is evidence from a **failed pre-release
gate**, not release evidence. No tolerance was relaxed.
