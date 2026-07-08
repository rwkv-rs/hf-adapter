# RWKV-7 HF v0.9 status

The main line is the readable pure-PyTorch reference implementation. Optional
CUDA Graph/Triton development is isolated on
`perf/optional-native-backend-v0.10`; older native-kernel experiments remain
preserved on `perf/native-kernels-v0.8`.

Release gates:

- [x] clean config, cache, operator and modeling structure
- [x] package-free save/load and reference conversion layout
- [x] tiny cache, padding, generation, loss and gradient tests
- [x] non-blocking RTX 4080 FLA backend diagnostics archived
- [x] official checkpoint oracle matrix (V100 6/6; RTX 4080 9/9)
- [x] formal 48-unit lm_eval run (42 retained units plus six affected 1.5B
      batch-pair reruns; merged validator passed)
- [x] canonical SFT, DPO and GRPO runs, exact resume, and W&B offline smoke
- [x] clean wheel build, metadata check, isolated install and CLI/model smoke
- [x] six model repositories updated and tagged v0.9.0; exact-revision Hub
      metadata passed for all six, direct Hub loading passed for
      0.1B/0.4B/1.5B, and V100 local-source smoke passed for
      2.9B/7.2B/13.3B with hashes matched to each conversion manifest
- [x] PyPI 0.9.0 published through the upstream trusted-publishing workflow;
      the wheel was downloaded back from PyPI and passed CLI plus V100 Hub
      smoke in the clean Python 3.12/CUDA 12.6 environment

A release is not complete until every unchecked item passes.

The optional optimized-backend comparison is preserved in
[`benchmarks/fla/results/4080-reference-20260825`](benchmarks/fla/results/4080-reference-20260825/README.md).
It records numerical differences but is not a correctness oracle and does not
block release. Official RWKV checkpoints, HF invariants, formal evaluation,
and training reproducibility are the release gates.

The compact six-model release evidence is archived in
[`results/release/hf-v0.9.0-v100`](results/release/hf-v0.9.0-v100/README.md).
