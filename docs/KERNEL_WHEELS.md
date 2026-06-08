# Prebuilt CUDA kernel wheels

Chinese version: [`KERNEL_WHEELS_ZH.md`](KERNEL_WHEELS_ZH.md)

`rwkv7-hf` remains the portable Transformers package. Its wheel contains the
model implementation, dispatch policy, Triton code, conservative PyTorch
fallbacks, and CUDA source for the historical lazy-JIT path. An optional
`rwkv7-kernels` wheel contains already compiled CUDA extension modules for one
exact runtime lane.

## Ordinary-user workflow

Install the adapter and inspect the current environment:

```bash
python -m pip install "rwkv7-hf==0.8.1"
rwkv7-hf kernels status
rwkv7-hf kernels recommend
rwkv7-hf doctor
```

The base PyPI installation intentionally does not install a GPU wheel. On a
supported Linux NVIDIA lane, `recommend` lists one exact build; then install
the hash-pinned wheel selected from the public release index:

```bash
rwkv7-hf kernels install
rwkv7-hf doctor
```

The installer matches all of these fields before invoking pip:

- Python major/minor and CPython ABI;
- platform and machine architecture;
- PyTorch major/minor and libstdc++ ABI;
- PyTorch CUDA runtime;
- device compute capability;
- compatible `rwkv7-hf` release series.

It installs by an HTTPS asset URL with a `sha256` fragment. It never chooses a
"nearby" wheel. If there is no exact match, the adapter remains usable through
the existing JIT, Triton, or PyTorch fallback.

No model setting is required after installation. The default
`RWKV7_KERNELS_MODE=auto` revalidates the installed manifest in each new
process and automatically chooses compatible prebuilt extensions before JIT
or portable fallbacks. A stale or incompatible package is never imported.

## Initial verified binary matrix

| Build lane | Verified device | Included extensions |
|---|---|---|
| CPython 3.11, Torch 2.5, CUDA 12.4, `sm_70` | Tesla V100-PCIE-32GB | FP16 recurrence, SM70 linear, SM70 W/A/G/V, SM7x quant, sparse FFN |
| CPython 3.11, Torch 2.6, CUDA 12.4, `sm_89` | NVIDIA GeForce RTX 4080 | FP16 recurrence, Ada W/A/G/V, sparse FFN |

The build system also describes SM75, SM80, SM86, SM90 and SM120 source lanes,
but a binary is not published as validated merely because NVCC can compile it.
Each new lane requires same-card import, correctness, route, and public-install
evidence.

## Selection and fallback

`RWKV7_KERNELS_MODE` controls the native extension boundary:

| Value | Behavior |
|---|---|
| `auto` | Prefer a compatible prebuilt wheel, then retain lazy JIT fallback. This is the default. |
| `prebuilt` | Require a compatible prebuilt extension and forbid JIT fallback. Acceptance tests use this mode. |
| `jit` | Ignore the prebuilt package and use the historical lazy-JIT route. |
| `portable` | Disable both binary and JIT CUDA extensions; Triton/tensor/eager fallbacks remain available. |

Every binary loader follows the same order:

```text
compatible rwkv7-kernels module
  -> lazy JIT extension (auto/jit only)
  -> existing correct Triton/tensor/PyTorch fallback
```

Use the model-level report after inference to see what actually ran:

```python
report = model.rwkv7_runtime_report()
print(report["last_prefill_backend"])
print(report["last_decode_backend"])
print(report["kernels"]["extensions"])
```

The Doctor reports compatibility without importing a binary. The model report
records successful binary or JIT selection after a route has executed.

## Public-model acceptance

The smallest public checkpoint provides the one-command end-to-end gate:

```bash
rwkv7-hf smoke \
  --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 \
  --device cuda \
  --output rwkv7-smoke.json
```

It downloads and loads the model, performs prefill and greedy decode, checks
finite logits, reports the effective prefill/decode and kernel routes, records
peak CUDA allocation, and prints `RESULT: PASS`. Timing fields are a smoke
measurement, not a promoted benchmark claim.

## Building a wheel

Use the Python/PyTorch/CUDA environment that the wheel will target:

```bash
export CUDA_HOME=/path/to/cuda-12.4
export PYTHON_BIN=/path/to/target/python
export RWKV7_KERNEL_ARCH_LIST=8.9
export RWKV7_KERNEL_SOURCE_COMMIT="$(git rev-parse HEAD)"
export OUT_DIR=/path/to/output
bash scripts/build_kernel_wheel.sh
```

The build extracts the canonical C++/CUDA source strings from `rwkv7_hf`,
compiles architecture-specific binary modules, embeds `_manifest.json`, and
runs `scripts/inspect_kernel_wheel.py`. Generated sources and binaries are not
committed.

The manually dispatched `kernel-wheels` workflow runs the same build on
labelled SM70 and SM89 hardware runners, installs the adapter and wheel into an
isolated target, sets `RWKV7_KERNELS_MODE=prebuilt`, executes the exact-card
kernel tests, generates the hash index, and can upload verified assets to an
existing GitHub release.
