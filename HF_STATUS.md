# RWKV-7 HF v0.9 status

The main line is the readable pure-PyTorch reference implementation. Historical
performance backends and evidence are preserved on `perf/native-kernels-v0.8`.

Release gates:

- [x] clean config, cache, operator and modeling structure
- [x] package-free save/load and reference conversion layout
- [x] tiny cache, padding, generation, loss and gradient tests
- [x] first RTX 4080 0.4B FP16 clean-vs-FLA smoke
- [ ] pinned-FLA full V100 and RTX 4080 matrix
- [ ] official checkpoint oracle matrix
- [ ] formal 48-unit lm_eval run
- [ ] canonical SFT, DPO and GRPO runs
- [x] clean wheel build, metadata check, isolated install and CLI/model smoke
- [ ] six model repositories updated and tagged v0.9.0
- [ ] PyPI 0.9.0 published

A release is not complete until every unchecked item passes.
