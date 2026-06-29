# RWKV-7 HF v0.9 status

The main line is the readable pure-PyTorch reference implementation. Historical
performance backends and evidence are preserved on `perf/native-kernels-v0.8`.

Release gates:

- [x] clean config, cache, operator and modeling structure
- [x] package-free save/load and reference conversion layout
- [x] tiny cache, padding, generation, loss and gradient tests
- [x] non-blocking RTX 4080 FLA backend diagnostics archived
- [ ] official checkpoint oracle matrix
- [ ] formal 48-unit lm_eval run
- [ ] canonical SFT, DPO and GRPO runs
- [x] clean wheel build, metadata check, isolated install and CLI/model smoke
- [ ] six model repositories updated and tagged v0.9.0
- [ ] PyPI 0.9.0 published

A release is not complete until every unchecked item passes.

The optional optimized-backend comparison is preserved in
[`benchmarks/fla/results/4080-reference-20260825`](benchmarks/fla/results/4080-reference-20260825/README.md).
It records numerical differences but is not a correctness oracle and does not
block release. Official RWKV checkpoints, HF invariants, formal evaluation,
and training reproducibility are the release gates.
