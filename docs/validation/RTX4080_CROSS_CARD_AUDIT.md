# RTX 4080 changes: RTX 4090/5090 isolation audit

Date: 2026-07-21

## Scope

This audit compares the last pre-4080 implementation commit
`8945395a165c497c2e3eb5f1b6e9284176b48872` with the two RTX 4080 commits
`b2078fc` and `f7209e5`, then checks the resulting runtime policy for RTX 4080,
RTX 4090 and RTX 5090. Runtime files in scope are:

- `pyproject.toml`
- `rwkv7_hf/fused_time_mix.py`
- `rwkv7_hf/kernel_policy.py`
- `rwkv7_hf/native_jit.py`
- `rwkv7_hf/native_model.py`
- `rwkv7_hf/triton_compat.py`

Benchmark, documentation and test artifacts were also scanned for installation
or acceptance commands that could silently change another card's software
stack.

## Findings and fixes

| Finding | Cross-card risk | Resolution |
|---|---|---|
| Ada `prefill_self_chunk_size=32` was assigned to the whole family | RTX 4090 kept self-chunk off by default, but an explicit self-chunk run inherited the RTX 4080 tile instead of its measured tile 16 | Set 32 only for exact RTX 4080; RTX 4090 and other Ada cards retain 16 |
| Exact cards were selected with substring tests such as `"4080" in name` | Similar names such as Laptop/SUPER/Ti or adjacent numeric products could inherit unmeasured routes | Centralized token-scoped desktop RTX matching and added negative policy tests |
| The RTX 4080 Triton 3.2 workaround replaced one six-output strict-FP16 PTX expression with six one-output expressions globally | RTX 5090 uses strict FP16 shift-mix on Triton 3.6, so its previously validated code generation was changed even though the compiler defect is specific to the older stack | Triton `<3.3` uses the safe six-call lowering; Triton `>=3.3` restores the original one-delta/six-output lowering through a compile-time gate |
| The 4080 commit changed the global TorchAO extra to `torchao>=0.16.0` | RTX 4090 evidence uses TorchAO 0.9.0 while RTX 5090 uses a newer stack; installing the extra could force an unvalidated upgrade | Removed the global TorchAO minimum and made the exact 4080 acceptance script require TorchAO 0.16.0 |
| Generic CUDA/quant extras independently required Triton 3.3+ | The measured RTX 4080 and RTX 4090 PyTorch 2.6 stacks use bundled Triton 3.2; pip could replace PyTorch's matched compiler | Extras no longer install/upgrade Triton; the PyTorch-compatible Triton is authoritative |

## Resulting card policy

The machine-readable dataclass diff now has the following result:

- **RTX 4080:** retains its exact shape allowlists, graph cache size 4,
  self-chunk tile 32, row-4 scan selections, and disabled regressing Ada linear
  and sparse-FFN routes.
- **RTX 4090:** no executable policy value differs from the pre-4080 policy.
  Only later-added inert schema fields (`None`, `False`, or empty tuples) and
  explanatory notes differ. Self-chunk remains off by default and its fallback
  tile is 16.
- **RTX 5090:** no executable policy value differs from the pre-4080 policy.
  Only later-added inert schema fields differ. No 4080 shape, tile, graph-cache,
  quantization, or Ada route is inherited.

The exact-card helper also fails closed for `RTX 40800`, `RTX 40900`,
`RTX 50900`, `RTX 4080 Laptop GPU`, `RTX 4080 SUPER`, and `RTX 4090 Ti`.

## Generic code added during the 4080 work

The large `native_model.py` change is not a card-tile promotion. It adds native
prefill/chunk-continuation/speculative APIs and mirrors the existing HF model's
BnB loader policy into the fully native model. Dense single-token decode is
unchanged. Optimized continuation is guarded by CUDA inference, no mask, no
adapter, no gradient, one device, sequence length greater than one, and an
available native prefill implementation. CUDA graphs are not used for an
existing continuation cache.

The `triton_compat.py` change only annotates Triton 3.2's existing descriptor
class for PyTorch/DeepSpeed dataclass introspection. Triton 3.6 follows the
pre-existing missing-class shim path. It does not choose a model kernel or tile.

## Evidence boundary

Historical hardware artifacts establish the software stacks being protected:

- RTX 4080: PyTorch 2.6.0+cu124, Triton 3.2.0, TorchAO 0.16.0 in
  `bench/4080_full_model_ladder_20260719/environment.json`.
- RTX 4090: PyTorch 2.6.0+cu124, Triton 3.2.0, TorchAO 0.9.0 in
  `bench/4090_g1h_7p2_bsz8_20260715/environment.json`.
- RTX 5090: PyTorch 2.11.0+cu128 and Triton 3.6.0 in
  `bench/5090_native_decode_fused_20260718/environment.json`.

Policy/source tests prove dispatch isolation without requiring those GPUs. A
fresh physical RTX 4090 and RTX 5090 benchmark is still required before claiming
post-audit throughput parity; historical throughput is not relabeled as a new
run.

## Reproduction

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_kernel_policy.py \
  tests/test_native_prefill_scan.py \
  tests/test_native_quant_mm4_policy.py \
  tests/test_fused_sequence_mix.py \
  tests/test_qwen35_speed_matrix.py
```

For physical no-regression, run each exact card's existing acceptance entrypoint
with the environment recorded above and compare route metadata as well as
prefill/decode throughput. A source-only policy pass is necessary but not a
substitute for that hardware gate.
