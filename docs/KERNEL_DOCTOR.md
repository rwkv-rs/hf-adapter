# Runtime and kernel-policy doctor

Chinese version: [`KERNEL_DOCTOR_ZH.md`](KERNEL_DOCTOR_ZH.md)

Run the read-only doctor after installing the current source tree and before a
large checkpoint load:

```bash
python -m rwkv7_hf.doctor
```

An installed wheel also exposes the equivalent console command:

```bash
rwkv7-hf-doctor
```

The report identifies the Python, PyTorch, Transformers, CUDA/ROCm, Triton,
NVCC, Ninja, visible accelerators, exact hardware profile, policy defaults,
and cache locations. A successful inspection ends with `RESULT: READY`.

Policy features are reported as **candidates**, not as a claim that a kernel
has executed. The final route also depends on model shape, dtype, batch size,
sequence length, optional packages, and environment overrides. The doctor does
not download weights, compile an extension, capture a graph, or benchmark the
device.

## One device

By default every visible CUDA device is reported. Select one explicitly with:

```bash
python -m rwkv7_hf.doctor --device cuda:1
```

`--device mps` and other devices understood by the installed PyTorch runtime
are also accepted.

## Machine-readable evidence

```bash
python -m rwkv7_hf.doctor \
  --json \
  --output rwkv7-doctor.json
```

Attach this JSON before reporting that an optimized route is missing. It does
not intentionally collect model weights or authentication tokens, but it does
include local compiler and cache paths; review it before sharing publicly.

## Interpreting warnings

- **Triton unavailable:** Triton candidates cannot run; compatible Torch paths
  can still work.
- **CUDA extension toolchain incomplete:** NVCC and Ninja were not both found.
  JIT extensions that require them cannot build, while Torch and Triton routes
  may remain available.
- **PyTorch CUDA binaries do not support the device capability:** install a
  PyTorch build whose CUDA architecture list covers the reported GPU. This is
  a hard `RESULT: FAIL`, because CUDA tensor execution would fail later.
- **No exact validated hardware profile:** the device stays on conservative
  defaults until exact-card acceptance evidence promotes a route.

Use [`examples/check_environment.py`](../examples/check_environment.py) for
the broader dependency and model-directory gate. The doctor is the narrower
accelerator and kernel-policy report.
