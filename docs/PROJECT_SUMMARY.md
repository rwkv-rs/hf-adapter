# Project summary

Last audited: **2026-08-17**.

This repository provides the RWKV-7 Hugging Face / Transformers adapter. The
current release exposes native Auto classes, generation with recurrent cache,
dynamic batching, chunked prefill, PEFT/Trainer/TRL workflows, quantized
loading, and reproducible exact-card benchmarks.

Native is the current performance backend. The wrapper/reference FLA path is
kept for compatibility and correctness comparisons, not as the promoted RWKV
performance route.

Current review surfaces:

- [Release status](../HF_STATUS.md)
- [Acceptance contract](ACCEPTANCE.md)
- [Hardware matrix](HARDWARE_MATRIX.md)
- [Latest Qwen3.5 P/D table](QWEN35_LATEST_P_D_TOKPS.md)
- [Current results index](RESULTS_INDEX.md)
- [Machine-readable artifact manifest](../bench/CURRENT_ARTIFACTS.json)

Historical benchmark accumulations were removed on 2026-08-17. Each active
hardware or feature line now keeps one latest reviewable bundle.
