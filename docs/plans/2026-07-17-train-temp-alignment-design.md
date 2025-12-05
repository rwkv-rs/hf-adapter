# RWKV-LM train_temp Alignment Design

## Goal

Establish reproducible, single-GPU evidence for whether the Hugging Face RWKV-7
training path matches the official RWKV-LM `RWKV-v7/train_temp` reference. The
result must distinguish interface compatibility, one-step numerical parity,
and multi-step convergence instead of treating a finite-loss smoke as training
effect alignment.

## Scope

- Hardware: one exact RTX 5090, with both implementations run sequentially.
- Reference: a pinned RWKV-LM commit and the production `train_temp` model, loss,
  initialization, data order, and optimizer recipe.
- First model: the smallest production-shaped RWKV-7 configuration that can run
  both implementations with the same vocabulary and head size. Expand to the
  available 0.4B checkpoint only after the one-step gates pass.
- Precision: the current production train_temp CUDA operators require bf16, so
  bf16 is the end-to-end oracle. Fp32 is limited to unit tests for loss,
  parameter grouping, hashing, and metric semantics; it is not reported as a
  production train_temp parity row.
- Multi-GPU ZeRO, throughput promotion, and large-model convergence are separate
  follow-up validations.

## Architecture

`rwkv7_hf/train_temp_alignment.py` contains training semantics that are reusable
from HF Trainer and the benchmark harness: the train_temp L2Wrap cross-entropy
reference, parameter classification, optimizer groups, and tensor comparison
metrics. Unit tests lock down the extra L2Wrap gradient and every parameter
group before any GPU run.

`bench/bench_train_temp_alignment.py` is a process-isolated evidence runner. It
accepts a pinned official checkout, one initial checkpoint, one serialized token
batch, one backend (`official` or `hf`), and one phase (`forward`, `backward`, or
`step`). Each run writes provenance, scalar metrics, and mapped tensor snapshots.
A compare mode verifies official-to-HF name translation, transpose rules, loss,
gradient, and optimizer-delta gates.

`scripts/run_train_temp_alignment.sh` orchestrates the exact-card run without
hard-coding a GPU index. It creates immutable input hashes, resumes completed
phases, stops before convergence when a numerical gate fails, and appends JSONL
rows. Long runs use the same serialized sample order in both implementations.

## Acceptance

The first production claim requires all of the following on the recorded card:

1. Forward: matching batch/checkpoint hashes, finite loss, and logits cosine at
   least 0.9999 in the production bf16 lane.
2. Backward: every mapped trainable parameter is present; gradient cosine is at
   least 0.999 in bf16, with bounded relative L2 error.
3. Optimizer step: matching parameter groups, learning rates, weight decay,
   clipping order, and parameter-delta cosine under the same thresholds.
4. Convergence: at least three sequential seeds; no non-finite loss or gradient
   spike; final validation loss and loss-curve area remain within the declared
   tolerance and observed seed variance.

Passing a Trainer, PEFT, TRL, or DeepSpeed smoke remains compatibility evidence
only. Passing one-step parity is numerical evidence only. Neither can be used as
a convergence claim without the multi-seed curve artifact.

## Failure Recovery

Every phase writes to a temporary file and renames it only after validation.
The runner records completed phase keys, so interrupted work resumes from the
same checkpoint, batch, backend, precision, and seed. A changed source commit,
checkpoint hash, or batch hash invalidates reuse instead of merging unlike rows.
