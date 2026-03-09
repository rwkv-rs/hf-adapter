# MTT S70 MUSA HF adapter validation

Date: 2026-07-28

This artifact records exact-card validation of the optional MUSA WKV forward
kernel in the RWKV-7 Hugging Face adapter. The conclusion is limited to one MTT
S70, the converted 0.1B model, fp16, batch 1, prompt length 128 and decode length
32. It does not generalize to another MUSA architecture, model, shape or
capability.

## Scope and correctness gates

- Device: MTT S70 (`mp_21`, 7,427,821,568 bytes reported memory).
- Runtime: MUSA SDK 4.2.0, torch 2.5.0, torch_musa 2.5.0+0dbf6f1.
- HF layer: Transformers 5.12.1 and NumPy 1.26.4 from an isolated target; the
  installed torch and torch_musa packages were not changed.
- Model: `rwkv7-g1d-0.1b-hf-b3d93fe-tf5-np126`, 12 layers, hidden width 768,
  12 heads of width 64.
- Correctness prerequisites passed before timing: standalone WKV output/state
  parity, HF load/forward/cache/reorder/generate, eager/WKV greedy-token equality,
  a 64-token trace with logits cosine `0.9999946355819702`, and differentiable
  eager fallback under autograd. A counter around `try_musa_wkv` observed zero
  calls during the passing backward run, directly confirming kernel bypass.
- Three separate paired processes were run in alternating order:
  eager, WKV, eager, WKV, eager, WKV.
- A separate batch 1/2 smoke confirmed model/input placement, synchronization,
  peak-memory reporting and explicit eager/WKV route telemetry. Its eager and
  WKV processes ran concurrently, so those smoke throughput values are omitted
  and are not performance evidence.

## Fixed benchmark shape

- dtype: fp16
- batch: 1
- prompt: 128 tokens
- decode: 32 tokens
- attention mode: chunk
- `fuse_norm=false`
- fast cache enabled
- HF decode API: ordinary `forward`
- three warmups and five measured prefill runs per process

Every eager row asserts that `RWKV7_MUSA_WKV` was disabled and the extension was
not loaded. Every WKV row asserts that the route was enabled, the extension was
loaded, and no module error was recorded.

## Result

| Metric | Eager min / median / max | WKV min / median / max | Paired WKV/eager min / median / max |
|---|---:|---:|---:|
| Prefill tok/s | `64.4 / 66.8 / 66.9` | `81.0 / 81.0 / 81.1` | `1.210762x / 1.214072x / 1.257764x` |
| Decode tok/s | `64.6 / 65.9 / 66.2` | `65.9 / 66.0 / 66.1` | `0.998489x / 1.000000x / 1.021672x` |
| Peak allocated MiB | `406.0 / 406.0 / 406.0` | `406.0 / 406.0 / 406.0` | `1.000000x / 1.000000x / 1.000000x` |

For this exact shape, the optional WKV route improved median prefill throughput
by about `1.21x`. Decode was effectively unchanged, so this artifact does not
support a decode-acceleration claim. Peak allocated MUSA memory was also
unchanged, so it does not support a memory-reduction claim.

## Non-claims

This artifact does not validate bf16, quantization, graph capture, multi-device,
training kernels, PEFT package integration, larger models, or other batch and
sequence shapes. The imported WKV kernel remains inference-only; training uses
the differentiable pure-PyTorch recurrence.

## Files

- `environment.json`: exact hardware, software, model, source and shape record;
  workstation identifiers and local installation paths are intentionally redacted.
- `bench_speed_executed.py.txt` and `bench_batch_sweep_executed.py.txt`: exact
  source snapshots used on the validation host, retained for independent audit.
- `raw_{eager,wkv}_run{1,2,3}.jsonl`: performance outputs with only the local
  absolute `hf_model_dir` replaced by the stable model identifier.
- `batch_smoke.jsonl`: B1/B2 device, memory and route contract rows; throughput
  is deliberately omitted because the two modes ran concurrently.
- `speed_paired.jsonl`: normalized copies of the six rows with explicit pair
  indices and the batch size, warmup and run count restored from the command.
- `summary.json`: statistics, route assertions and narrow conclusion.
- `SHA256SUMS`: integrity hashes for this evidence package. Verify from this
  directory with `sha256sum --check SHA256SUMS`.
