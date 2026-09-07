# Paused 1.0 candidate: documentation closeout

Prepared 2026-09-08. **This is a historical stop/handoff record, not a new GPU
validation bundle. `release_ready` is false.**

## Provenance and limits

The session directly read the RTX 4080 manifests, process tree and logs on
2026-09-01, then stopped the formal run and its retry watchers at the user's
request. The counts below reproduce those retained tool observations. Raw
manifests, result JSON, samples and logs were **not** available locally and
could not be retrieved during this closeout: both direct SSH and the existing
V100 jump route to RTX 4080 timed out on 2026-09-08.

Consequently `status.json` is explicitly typed as a historical observation,
not a validator result. It records no invented per-unit result hashes or
metric values and makes no claim about current server processes. Before
publication or resumption, retrieve the original data and verify the actual
loaded model code as well as wheel hashes. The installed package version alone
cannot exclude old adapter code loaded from a model directory/module cache.

## Last observed run

Source: `5f4a4a25018f7567fd1b35b93fb1edf57ff86ac7`.
FLA: `80e494f6c588e091fc8316b612870df29375c5b8`.
Matrix: 0.1B/0.4B/1.5B × B1/B8 × eight tasks × three lanes = 144 units.

| Lane | Successful exits | Failed exits | Interrupted | Not started |
|---|---:|---:|---:|---:|
| Optimized | 47 | 1 | 0 | 0 |
| FLA | 48 | 0 | 0 | 0 |
| Reference | 43 | 0 | 1 | 4 |
| Total | 138 | 1 | 1 | 4 |

There were 139 completed process outcomes, not 144 accepted units. Numerical,
batch-invariance and cross-lane validation must be audited separately from
process exits. No additional FLA accuracy/speed numbers are re-derived here
without the underlying results.

- Optimized `1.5b-b8-wikitext` failed with CUDA OOM while lm_eval allocated
  full-vocabulary log-softmax output; graph private-pool residency was reported
  in the error. A graph-disabled retry was queued but stopped before execution.
  This is neither a measured retry success nor a completed memory fix.
- Reference `1.5b-b8-hellaswag` was interrupted by the user stop. Its remaining
  B8 tasks were Winogrande, ARC Easy, ARC Challenge and OpenBookQA.
- The matrix runner, current child, priority orchestrator and OOM retry watcher
  were all absent after the stop check. The observed GPU utilization was 0%.
  This describes 2026-09-01 only, not a live 2026-09-08 probe.

The root needed for later retrieval is
`/home/wzu/codex-run/results/rwkv7-freeze-5f4a4a25/4080`.
Do not edit the original failed/interrupted attempts when selecting reruns.

## Scope of this closeout

- Stable/candidate user entrypoints and the capability matrix are clarified.
- All frozen executable/distribution inputs and existing result bundles remain
  untouched; both English/package READMEs are also frozen inputs.
- No GPU task, retry watcher, automation or publication was started.
- Pending hardware, CI, Hub and PyPI gates remain pending, as listed in
  [HF_STATUS.md](../../../HF_STATUS.md).
- Packed variable-length prefill and graph memory management remain future
  backend work, requiring a new versioned artifact if implemented.

`MANIFEST.sha256` checks only the integrity of this small handoff bundle. It
cannot authenticate missing remote results or turn the historical snapshot
into final acceptance. `checks.json` records local documentation/structure
checks only; it is not a model accuracy or GPU performance report.
