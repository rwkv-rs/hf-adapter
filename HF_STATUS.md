# RWKV-7 HF: stable release and 1.0 candidate

**Status reviewed: 2026-09-08. The published reference release is 0.9.0;
the frozen 1.0 source is a candidate, not a completed release.** Merging source
into GitHub `main` does not publish a wheel or update the model repositories.

## Acceptance resumed on 2026-09-08

The user has asked to finish the existing acceptance criteria. Local and
publication preflight work has resumed; no recurring monitor was created.
GPU execution cannot yet resume: the RTX 4080 Tailscale peer is offline
(last seen 2026-09-07 22:40 Asia/Shanghai), and RTX 4090 SSH times out.

- The complete local suite passes **470/470**, with 409 deprecation warnings.
  All 155 frozen inputs remain byte-identical.
- Upstream `main` is verified at
  `2d0426e02401975606a17828ff335a349dd9871d`; its tree matches the frozen
  source. Current-main [full-tests](https://github.com/rwkv-rs/hf-adapter/actions/runs/34100068161),
  [HF ecosystem CI](https://github.com/rwkv-rs/hf-adapter/actions/runs/33492973905)
  and [CPU smoke](https://github.com/rwkv-rs/hf-adapter/actions/runs/33388562778)
  all have successful completed runs. These are CPU/framework checks, not
  final GPU acceptance.
- PyPI is rechecked at 0.9.0, with no 1.0.0 release files; GitHub's latest
  published release is also v0.9.0.
- The original frozen HF/kernel wheel pair has been reconstructed with
  **exactly matching SHA256**. Wheel inventory, source/archive consistency
  and strict Twine checks passed; no source byte or acceptance threshold was
  changed. The matching artifacts are retained outside Git for hardware use.
- CPU checks from the installed wheels passed forward/cache, save/reload,
  greedy/beam generation and loss/backward. A synthetic tiny checkpoint also
  passed the installed conversion CLI and package-free local-directory load.
  These environments share existing dependencies; they are not fresh GPU
  environments or six-model Hub redownloads.

The compact [September 8 preflight evidence](results/release-preflight/20260908/README.md)
records those checks and their scope, including failed infrastructure/build
attempts. Its `release_ready` remains `false`.

The hardware and publication checklist below remains open. The September 1
counts are still historical observations until original remote results can
be retrieved. No missing GPU result is substituted by this CPU preflight.

## Start here

### Use the published reference release

Install the appropriate PyTorch build for your device first. Then, if you need
the converter/CLI:

```bash
python -m pip install "rwkv7-hf==0.9.0"
rwkv7-hf --version
rwkv7-hf convert --help
```

For Hub inference, use a published model and pin `revision="v0.9.0"` in both
`AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`, with
`trust_remote_code=True`. The published reference model does not require the
optional kernel package. See [the model list](docs/PUBLISHED_MODELS.md) and
the [Chinese quickstart](README_ZH.md).

### Inspect or reproduce the 1.0 candidate

The candidate source is frozen at
`5f4a4a25018f7567fd1b35b93fb1edf57ff86ac7`. Developers can install that exact
checkout, preferably in a separate environment:

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout --detach 5f4a4a25018f7567fd1b35b93fb1edf57ff86ac7
# Install the appropriate PyTorch build first.
python -m pip install -e .
# Optional, for candidate backend development only:
python -m pip install -e ./kernels
```

An editable/source installation is **not** the immutable wheel used in formal
validation. Candidate API-v4 testing also requires model directories converted
with this candidate; do not assume a `v0.9.0` Hub model gains API v4 just by
installing a kernel package.

The `==1.0.0` PyPI commands in the frozen English README are future-release
instructions, not current installation instructions. `README.md` and
`kernels/README.md` are distribution inputs in the source freeze and are left
byte-identical. This status page is the current entrypoint while the candidate
is held. Rebuilding the candidate after a documentation commit does not make
the new archives identical to the previously validated artifacts.

## Capability and evidence are separate

- **Implemented:** clean HF config/cache/ops/modeling/tokenizer; conversion;
  standard generation/loss/training surfaces; optional API-v4 backend;
  SFT/DPO/GRPO and evaluation tooling.
- **Conditional acceleration:** inference and training depend on the actual
  device, dtype, shape, mask and autograd request. `auto` may select reference
  execution after a negative capability decision; strict `optimized` raises
  for unsupported requests.
- **Not established:** universal HF-framework compatibility, universal
  training acceleration, a completed packed variable-length prefill frontend,
  or a speed advantage over FLA on every workload.

The [capability matrix](docs/NVIDIA_MIGRATION_AUDIT.md#user-facing-capability-matrix)
separates migrated source, active routes, fallback and missing work. The
[plugin contract](docs/KERNEL_PLUGIN_API.md) keeps those decisions out of the
public HF model and canonical `[B,H,K,V]` cache.

## Paused formal evaluation

The last directly observed RTX 4080 snapshot was taken on **2026-09-01**,
before the user requested that all current evaluation and rerun watchers stop:

| Lane | Successful exits | Failed exits | Interrupted | Not started |
|---|---:|---:|---:|---:|
| Optimized | 47 | 1 | 0 | 0 |
| FLA | 48 | 0 | 0 | 0 |
| Reference | 43 | 0 | 1 | 4 |

The failed optimized unit was `1.5b-b8-wikitext` (CUDA OOM). Its planned
graph-disabled retry was stopped before execution. The interrupted reference
unit was `1.5b-b8-hellaswag`; the four remaining tasks were Winogrande, ARC
Easy, ARC Challenge and OpenBookQA. Successful process exits do not by
themselves establish the numerical, batch-stability or three-way release gate.

**No GPU tasks were restarted for this closeout.** The RTX 4080 was unreachable on
2026-09-08, so this is a historical stop record, not a freshly retrieved
validation bundle. Its provenance and limitations are recorded in
[the closeout snapshot](results/closeout/20260908/README.md). Raw results remain
to be retrieved and audited; there is no completed 144-unit acceptance report.

## Remaining release checklist

| Item | Current disposition | Required to close |
|---|---|---|
| Source/package boundary | Frozen candidate | Keep all 155 frozen inputs byte-identical; no hidden rebuild |
| RTX 4080 matrix | Resume requested; device offline | After connectivity and source/provenance checks, retry the one OOM unit and finish the five incomplete reference units; validate the combined evidence |
| Actual loaded model code | Final provenance audit pending | Match wheel, model-directory code, config and loaded remote-code modules; reject legacy/stale adapter files |
| RTX 4080 HF/training/finetune | Historical evidence exists | Audit final-artifact identity and actual leaf routes; do not substitute older-SHA or fallback results for optimized acceptance |
| RTX 4090 | Final matching-artifact bundle pending | Use the same immutable artifacts after RTX 4080 acceptance |
| V100 | Excluded by user decision | Historical evidence only; do not restart |
| Six Hub repositories | v0.9.0 history retained | Stage/audit candidate code and unchanged weight hashes, then publish/tag and redownload-smoke all six |
| GitHub/PyPI 1.0 | Main CI verified; release not closed | Verify final release provenance, publish the exact accepted artifacts, then verify downloads |

Packed variable-length prefill and bounded CUDA Graph memory management are
next-version backend work, not completed features of the frozen candidate.
They require a separate versioned artifact and affected-route validation;
they must not silently replace the wheel under existing results.

## Published history

Version 0.9.0 has its own six-model Hub and PyPI evidence under
[`results/release/hf-v0.9.0-v100`](results/release/hf-v0.9.0-v100/README.md)
and [`results/release/wheel-0.9.0`](results/release/wheel-0.9.0/README.md).
Those records establish that release only. The 1.0 release workflow still
requires matching checksummed artifacts, both required hardware bundles,
formal evaluation, training/finetune, six tagged Hub models and download
audits. This closeout does not bypass or lower those gates.
