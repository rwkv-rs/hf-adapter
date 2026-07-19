# Configurable Native train_temp training

## Goal

Keep the pinned official runner unchanged while exposing the proven Native
`train_temp_cuda` convergence path as a configurable user command.

## Design

`scripts/train_native.py` resolves configuration in this order: built-in preset,
optional JSON file, then explicit CLI flags. The `native-default` preset starts
with batch size 1; `official-x070-12x768-b16` reproduces the retained official
shell hyperparameters without pinning the user's model dimensions or run length.

The command accepts either an existing packed training sequence plus validation
batch, or a MiniPile-style `.bin/.idx` prefix. The latter reuses the exact cubic
sampler already validated by the train_temp benchmark. It writes the resolved
configuration before launching and delegates training, checkpointing, RNG
restore, curves, and memory telemetry to `bench_train_temp_alignment.converge_hf`.

Only implementation constraints remain fixed: Linux CUDA, BF16, sm80+, head64,
dense labels, and sequence length divisible by 16. Model size, batch size,
sequence length, optimizer hyperparameters, schedule, seed, steps, evaluation,
and checkpoint cadence are user-configurable.

## Verification

CPU tests cover preset and override precedence, existing-sequence shape
inference, invalid kernel shapes, command construction, deterministic model
provenance, and a subprocess dry run. Existing train_temp alignment and user
documentation tests remain required.
