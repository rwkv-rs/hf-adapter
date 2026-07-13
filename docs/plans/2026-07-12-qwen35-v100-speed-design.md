# Qwen3.5 V100 Speed Matrix Design

The benchmark compares RWKV-7 HF adapter models with the nearest larger
official Qwen3.5 dense text models on one Tesla V100-PCIE-32GB.  It is an HF
Transformers comparison, not an Albatross or serving-engine comparison.  The
declared pairs are RWKV 1.5B vs Qwen3.5 2B, RWKV 2.9B vs Qwen3.5 4B, and RWKV
7.2B vs Qwen3.5 9B.

Each comparison cell is keyed by model pair, prompt length, decode length,
batch size, dtype, and quantization.  The initial V100 contract uses prompt
lengths 128/512/2048, decode lengths 128/512, batch sizes 1/2/4/8, fp16, and
the common HF precision lanes none/bnb8/bnb4.  This yields 216 comparison
cells.  Qwen uses the text-only `Qwen3_5ForCausalLM` path so the unused vision
encoder is not charged to the baseline.  Both sides use cache-aware one-token
decode, `logits_to_keep=1`, identical tensor shapes, warmup outside timing,
and fresh processes per raw row.

The raw JSONL records environment versions, effective backend, prefill and
decode throughput, per-sequence latency, model footprint, peak VRAM, and row
status.  A separate comparator joins candidate/reference rows and reports
coverage plus minimum/median speed ratios.  A row only passes the speed claim
when both sides pass and RWKV reaches the configured ratio.  Failures and OOMs
remain explicit rows; they are never silently dropped.  The orchestrator can
resume with `--skip-existing`, which is required for flaky SSH and long V100
runs.

The defensible claim is limited to the declared matrix: "RWKV-7 HF is faster
than Qwen3.5 on every completed row of the V100 HF Transformers matrix."  Model
quality is evaluated separately and must not be inferred from these synthetic
throughput rows.
