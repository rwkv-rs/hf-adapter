# RTX 4080 stable-version training candidate

This compact record validates the `rwkv7-hf==1.0.0` and
`rwkv7-kernels==1.0.0` stable-version candidate built from `4bbe5f9e`.
It is not labeled as the immutable final release pair: the final archives are
rebuilt only after both RTX 4080 and RTX 4090 acceptance gates close.

The recurrent matrix passed 18/18 cases. The full 0.1B model matrix passed
36/36 cases across batch 1/4, token lengths 16/17/128, no/left/right padding,
and gradient checkpointing off/on. Actual route counters distinguish the
factorized CUDA recurrent, exact matrix fallback, flattened CUDA linear, and
readable reference model loop. See `validation.json` for hashes, environment,
toolchain provenance, numerical extrema, and pinned-FLA speed rows.

Two earlier affected-only model attempts are retained remotely as failed
infrastructure/environment evidence. The successful retry uses the same wheel
bytes, the canonical Transformers 5.8 environment, and the original two
measured iterations. The already-passing recurrent matrix was not rerun.
