# RTX 5090 Blackwell smoke (2026-07-04)

> Superseded note (2026-07-04): the native prefill pack-unpack blocker documented here has been fixed and revalidated in [`../5090_blackwell_native_prefill_smoke_20260704`](../5090_blackwell_native_prefill_smoke_20260704/README.md). This directory is kept as the original first-smoke record.


Purpose: confirm the HF adapter can load and run on a real Blackwell RTX 5090 (`sm_120`) node, and record the current environment caveats before doing any full MATH500 acceptance run on 50-series hardware.

## Environment

See `env.txt` for the exact capture. Key facts:

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 610.43.02
- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8
- Triton: `3.3.1`
- FLA: `0.5.1`
- Transformers: `5.13.0`
- bitsandbytes: `0.49.2`

A local `triton_compat_shim` was installed in the venv so FLA/Torch can import the removed legacy `triton.compiler.compiler.AttrsDescriptor` path while keeping Triton 3.3's `triton.set_allocator` required by Blackwell paths. This is an environment workaround, not a repo code change.

## Smoke results

Model used for smoke only: `/workspace/models/rwkv7-g1d-0.1b-hf` (local 0.1B adapter copy). Dataset: 2 synthetic MATH-style rows. This is **not** an accuracy acceptance run.

| run | status | bsz | rollout | decoded tokens | prefill s | decode s | decode tok/s | summary tok/s | backend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `forward_prefill` | pass | 4 | 2 | 128 | 13.811 | 0.885 | 144.7 | 8.71 | native_graph |
| `warm_bsz8` | pass | 8 | 4 | 512 | 13.793 | 0.980 | 522.6 | 34.66 | native_graph |

Important: the prefill time includes first-use compile/import overhead in a fresh process, so the useful signal here is that the 5090 can execute load + generate + dynamic batching + deferred verification + deferred text decode without crashing. The final 4090 MATH500 acceptance artifact remains the production acceptance number.

## Caveats found on 5090

1. `torch.compile`/Inductor + Triton 3.3 + legacy `AttrsDescriptor` is fragile on this PyTorch 2.6 NVIDIA image. Plain generation works after setting `TORCH_COMPILE_DISABLE=1`.
2. The native prefill path on this 0.1B local smoke model failed with `ValueError: too many values to unpack (expected 40)`, so the smoke used `--prefill-backend forward --decode-backend forward`.
3. This does **not** replace a full MATH500 avg@64 comparison. It only confirms a current RTX 5090 path and records the exact environment blockers.

## Repro command used

```bash
source /workspace/venvs/rwkv7-5090/bin/activate
cd /workspace/projects/rwkv7-hf-adapter
export PYTHONPATH=.
export PYTHONNOUSERSITE=1
export TORCH_COMPILE_DISABLE=1
python bench/eval_math500_hf.py \
  --hf-dir /workspace/models/rwkv7-g1d-0.1b-hf \
  --dataset /workspace/data/math500_smoke.jsonl \
  --out-dir bench/5090_blackwell_smoke_20260704_warm \
  --rollout 4 --limit 2 --max-new-tokens 64 --ctx-limit 512 \
  --dynamic-batching --bsz 8 \
  --prefill-backend forward --decode-backend forward \
  --dtype fp16 --device cuda \
  --defer-verification --verify-workers 1 \
  --summary-speed-timing generation --defer-text-decode
```
