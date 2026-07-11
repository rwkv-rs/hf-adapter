# NVIDIA performance migration audit

This audit answers a narrow question: whether the NVIDIA implementation from
`perf/native-kernels-v0.8` has been preserved behind the clean optional-kernel
boundary, without copying its duplicate Hugging Face model stack back into
`rwkv7_hf`.

## Exact source transfer

`kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json` records **102** files
from `perf/native-kernels-v0.8`. Each row contains the old path, Git blob, new
path, destination SHA256, and transfer kind. The wheel test recomputes every
destination hash. For the **100 byte-identical** rows it also reconstructs the
Git blob directly from the wheel bytes. Two files require explicit
clean-boundary adaptation rather than a false byte-identity claim:

- `native_graph_runtime.py` now binds only canonical `RWKV7Cache` views rather
  than the historical private `NativeRWKV7Cache` ABI;
- `train_temp_cuda.py` now owns only the leaf forward/backward autograd
  operators; whole-model/layer dispatch lives in `training_runtime.py` and no
  model `forward` method is replaced.

Both adapted destinations retain their old Git blob identity, new SHA256 and a
machine-checked rationale. The complete 102-file set includes:

- CUDA/Triton fused projection, norm/mix, W/A/G/V LoRA, recurrent output,
  FFN, DPLR, self-chunk, and sequence-prefill implementations;
- SM70, Ada, and Blackwell policies and kernels;
- W8, W4, A8W8, BitsAndBytes, TorchAO, Marlin, and physical BN/TN code;
- decode graph, state-layout, native packing, and runtime policy code;
- the complete train-temp CUDA/C++ forward/backward leaf operators;
- Marlin and self-chunk licenses;
- explicit BN/TN sweep helpers and the opt-in legacy Triton compatibility
  utility.

Source-tree presence is not enough: `scripts/audit_release_wheels.py` opens the
actual kernel wheel, rejects unsafe/cross-package members, reads the embedded
manifest, and recomputes all 102 destination hashes plus all 100 applicable
source Git blobs. It also requires the adapted
dispatcher, dense/prefill/decode, graph/state-pool, quantization, recurrent and
training runtime modules that were intentionally not byte-copied from the old
duplicate model stack. The same command checks that the `rwkv7-hf` wheel keeps
only the canonical model package and does not contain `rwkv7_kernels`.

`kernels/rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json` is the second,
semantic half of this audit. It assigns every one of those 102 files exactly
once to one of 16 executable capability families: recurrent, dense decode,
fused/DPLR/self-chunk prefill, graph/state pools, SM70, Ada, Blackwell, native
W8/W4/A8W8, BN/TN, BitsAndBytes, Marlin, TorchAO, common quantization runtime,
and training autograd. Each family also names its adapted runtime entry files,
phases, activation status, devices and `KernelPolicy` fields. The wheel audit
fails if a migrated byte is unmapped or double-mapped, a runtime file is
absent, a policy flag is invented, or a required family is missing. Thus an
opaque directory of preserved sources cannot satisfy the release gate.

The later recurrent-only performance line is audited separately rather than
being hidden inside the larger v0.8 count. The wheel embeds
`RECURRENT_SOURCE_SCOPE.json` for
`perf/optional-native-backend-v0.10` commit
`0c5ea30ac6868974ba9836c4a065fa8b2847af68`. It reconstructs the complete
three-file historical `kernel_wheel/rwkv7_kernels` Git subtree
(`7d2fe3ffff72ec2cd44993e14757ef4443ddfcbb`): the old package entry point is
adapted to API v2, while `recurrent_graph.py` and `recurrent_triton.py` remain
byte-identical as `recurrent/graph.py` and `recurrent/triton.py`. Release
auditing recomputes both SHA256 and Git blob identity, so the earlier HF
high-performance recurrent implementation is covered in addition to all 102
v0.8 NVIDIA files.

The inventory deliberately distinguishes `migrated` from `production auto`.
All implementation families are present behind API v2, but full-model v2
families remain diagnostic until the same immutable wheel passes the complete
three-device gate. Native quantization stays an explicit user opt-in. This
prevents a source-completeness statement from being mistaken for unmeasured
device/shape promotion.

## Complete historical scope, including every deliberate exclusion

The number 102 is not a hand-selected denominator. The wheel also embeds
`SOURCE_SCOPE.json`, which classifies the complete 153-file `rwkv7_hf` tree at
historical commit `1014acf1a52fa4dee1e4d2b46e6059275c1d3bea`:

| disposition | files |
|---|---:|
| byte-identical NVIDIA implementation | 100 |
| adapted to the clean model-forward protocol | 12 |
| replaced by canonical reference ownership | 7 |
| tooling relocated or retired | 6 |
| separate non-NVIDIA hardware distribution | 27 |
| non-kernel speculative helper retired | 1 |

Every row retains its historical Git mode and blob ID. The wheel audit rebuilds
the Git tree object from all 153 rows and requires the result to equal frozen
tree `1bb1fe1cd64662bbd6d29f72c9002a8513af3691`. It then cross-checks all 102
NVIDIA rows (100 exact transfers and two clean adaptations) against
`MIGRATION_MANIFEST.json` and requires every adapted kernel
replacement to exist in the wheel. An omitted historical file, an
`unclassified` row, or a relabelled blob changes the reconstructed tree and
fails the release.

The 27 separate-hardware files are explicitly identified as Ascend, Apple/MLX,
Biren, MetaX or MUSA. They are not silently dropped and are not NVIDIA
operators; combining their runtimes and licenses into `rwkv7-kernels` would
violate the distribution boundary. The retired speculative helper is likewise
recorded as a higher-level HF feature rather than misrepresented as a kernel.

## Adapted rather than copied

The following old modules mixed performance code with a second model/config/
cache implementation.  Their functionality is represented by new
package-owned modules instead of byte-copying the duplicate classes:

| old module family | new owner |
|---|---|
| `model_prefill_graph.py` | `nvidia/prefill_graph_runtime.py` and `nvidia/prefill_graph_pool.py`; replay clones are copied into canonical `RWKV7Cache` |
| `model_runtime_policy.py` | `nvidia/kernel_policy.py` and runtime policy helpers |
| `model_quantization.py` | `rwkv7_kernels/quantization.py` plus the migrated quant operators |
| `model_backbone.py`, `model_layers.py`, `native.py`, `native_model.py`, `model_runtime.py` | the versioned model-forward protocol, `model_dispatcher.py`, `native_jit.py`, decode/prefill runtimes, and the clean structural owner passed at call time |
| `model_generation.py` | canonical `RWKV7Cache` batch reorder/select/repeat and standard Transformers generation methods |
| `model_fast_api.py` | standard HF `forward()`/`generate()` plus route trace and validation tools; no methods are injected into the model |
| `model_speculative.py` | not a kernel or required HF contract; it remains preserved on the historical branch and is intentionally not injected into the reference model |

The last row is an explicit scope decision, not a missing operator.  A future
speculative-decoding helper can consume the same public HF forward/cache API
without moving an implementation into `modeling_rwkv7.py`.

## Clean/reference and tooling ownership

These historical paths are not NVIDIA operators and therefore are not copied
to the kernel wheel:

- `configuration_rwkv7.py`, `modeling_rwkv7.py`, `model_config.py`,
  `model_cache.py`, `tokenization_rwkv7.py`: replaced by the canonical clean
  HF reference modules;
- `cli.py`, `converter.py`, `doctor.py`, `kernels_cli.py`, `smoke.py`,
  `adapter_manifest.py`: conversion/release tooling lives outside the model
  package in `rwkv7_hf_tools` or evaluation scripts;
- package `__init__` and old remote-code glue: replaced by the two small,
  independent package entry points.

## Deliberately separate hardware distributions

Ascend, MUSA, Biren, MetaX, MLX, and CoreML files are not NVIDIA backend-v2.
They remain intact on `perf/native-kernels-v0.8` and require their own optional
distributions.  They are excluded from `rwkv7-kernels` so installing an NVIDIA
wheel does not import unrelated runtimes or combine incompatible licenses.

## Acceptance rule

Source presence alone is not acceptance.  Production `auto` remains disabled
until one immutable HF wheel and one immutable kernel wheel pass:

1. reference/optimized/pinned-FLA operator and full-model parity;
2. cache, padding, cached decode, greedy/beam, loss and all-gradient gates;
3. dense and quantized route gates, HF ecosystem, SFT/DPO/GRPO;
4. the 144-unit three-way `lm_eval` matrix;
5. honest prefill/decode/forward-backward speed matrices on RTX 4080, V100,
   and RTX 4090.

Requested environment selectors are never accepted as proof.  Result bundles
must contain the actual `native-nvidia-*-v2[...]` or recurrent implementation
route together with code, model, environment, and wheel hashes.
