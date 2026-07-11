# MLX runtime module boundaries

The Apple runtime keeps the public imports from `rwkv7_hf.mlx_model` stable,
but implementation ownership is split so state/cache, serving orchestration,
and environment policy can evolve independently of model math.

## Module map

| Module | Owns | Must not own |
|---|---|---|
| `mlx_state.py` | `MLXRWKV7State`, cache clone/select/reorder/compact, small array/list conversion | model weights, tokenizer sessions, backend policy |
| `mlx_policy.py` | dependency-free environment parsing and backend-choice helpers | MLX arrays, model/session objects, kernels |
| `mlx_session.py` | generation outputs, single-session decode, dynamic session batching, stable argmax/repair policy | RWKV layer math and weight loading |
| `mlx_model.py` | RWKV-7 layer math, weight loading, prefill/decode dispatch, compiled-decode gates | session scheduling implementation |
| `mlx_quant.py` | W8/W4 layouts and projection kernels | model/session lifecycle |
| `mlx_dplr_prefill.py`, `mlx_scan.py`, `mlx_wkv.py` | focused recurrent/prefill kernels | tokenizer and serving policy |
| `mlx_speculative.py` | draft/target verification orchestration | target model implementation |

Dependency direction:

```text
mlx_policy      mlx_state
     \             /
      \           /
       mlx_session
            ^
            |
        mlx_model ----> quant / DPLR / scan / WKV kernels
            ^
            |
      mlx_speculative
```

`mlx_session` refers to `MLXRWKV7Model` only under `TYPE_CHECKING`, preventing a
runtime import cycle. `mlx_model` re-exports the historical state/output/session
classes, so existing imports continue to work:

```python
from rwkv7_hf.mlx_model import (
    MLXRWKV7State,
    MLXGenerationSession,
    MLXGenerationSessionBatch,
)
```

## Remote-code manifest

`scripts/adapter_manifest.py` is the only source of truth for Python files
copied into converted HF checkpoints. Both conversion and code-only sync tools
consume the same list. The closure test recursively checks relative imports, so
adding a runtime module without shipping it fails CI.

## Refactor invariants

1. No model math, policy default, environment variable, telemetry key, or
   benchmark threshold changes in boundary-only PRs.
2. Keep old imports as compatibility re-exports for at least one release.
3. Every extracted module must be included in the remote-code manifest.
4. Run state/session, quant, DPLR, speculative, converter closure, CPU/no-CUDA,
   and full pytest gates before merging.
5. Delete old paths only in a separate PR with telemetry/evidence proving they
   are no longer selected.

## Next safe splits

1. Move prefill/DPLR dispatch from `mlx_model.py` into `mlx_prefill.py` behind a
   narrow model-context protocol.
2. Move compiled/eager decode promotion into `mlx_decode.py`.
3. Replace repeated CLI `choices=[...]` literals with exported backend choice
   constants.
4. Split the Qwen/Apple benchmark runner by engine while preserving its JSONL
   schema and comparison gates.
