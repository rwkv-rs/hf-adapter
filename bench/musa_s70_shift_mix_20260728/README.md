# MTT S70 attention shift-mix opt-in experiment (2026-07-28)

This directory records narrow, exact-card evidence for the optional MUSA
attention shift-mix fusion. It is **not** a generic MUSA promotion and does not
enable the route by default.

## Scope

- Device: one legacy first-generation Moore Threads MTT S70, 7 GiB, MUSA SDK 4.2.0.
- Runtime: PyTorch 2.5.0, torch_musa 2.5.0, Transformers 5.12.1.
- Model: converted RWKV-7 G1D 0.1B, FP16, chunk attention, unfused norm.
- Candidate: `RWKV7_MUSA_ATTN_SHIFT_MIX=1`.
- Baseline: the canonical eager shift-mix expressions.
- Both sides retain the validated MUSA WKV inference kernel.

The route is constrained to exact MTT S70 product tokens, FP16 storage/IO,
inference mode, and matching `[B,D]` tensors. S70 has no Tensor Core and its
fp16 compute is extremely slow; this tiny elementwise kernel is an exact-card
launch-reduction experiment, not evidence for a general fp16 compute policy.
Build or runtime failures return to eager. The fusion uses a separate lazy
module so it cannot disable the validated MUSA WKV extension.

## Correctness

- Six-output micro tests at B1/B2/B8 and D768 are bitwise exact.
- A 36-token greedy generate trace is identical with the fusion off/on.
- Final logits for that trace are bitwise exact.
- Direct 4-token forward comparison is bitwise exact for logits and all cache
  groups: 12 recurrent tensors, 12 attention previous-state tensors, 12 FFN
  previous-state tensors, and `v_first`.
- Route calls are zero when disabled and positive when enabled.

See `state-compare.json`, `e2e-off.json`, and `e2e-on.json`.

## Paired performance

Each row uses prompt 128 or 512, batch 1/2/4/8, decode 128, three warmups and
five timed prefill repeats. Two fresh-process rounds reverse candidate/baseline
order. Decode is measured through `rwkv7_forward_token`; route metadata is
retained in every raw row.

Across 16 paired cells:

- prefill ratio median: `1.0508x`;
- prefill range: `1.0423x` to `1.0552x`;
- fast-token decode ratio median: `1.0008x`;
- decode range: `0.9967x` to `1.0050x`;
- peak allocated memory: identical in every pair;
- route telemetry: valid in every pair.

The result supports a narrow MTT S70 prefill experiment. Decode is neutral
within measurement noise and is not claimed as accelerated.

## Decision and boundary

Retain the route as an exact-MTT-S70, FP16, inference-only, explicit opt-in
experiment. Do not enable it by default and do not update `HF_STATUS.md` or
`BENCHMARK.md` from this single-card, single-model matrix. Other MUSA cards,
model sizes, BF16, training kernels, quantization, graphs, and multi-device
execution remain unvalidated. Later MUSA generations, including S4000/S5000,
may have substantially different capabilities and must not inherit the S70
compute profile; they still require their own exact-device evidence before use.

Run `python summarize.py` to regenerate `summary.json`. Raw JSONL rows are under
`raw/`; `SHA256SUMS` covers the committed evidence files.
