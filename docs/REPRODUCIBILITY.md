# Reproducibility artifacts

Local files are the source of truth even when W&B reporting is enabled.

Each evaluation or training run writes:

- resolved configuration and exact command;
- source Git SHA, model revision and dataset revision;
- Python, PyTorch, Transformers, TRL and PEFT versions;
- JSONL metrics and final evaluation;
- checkpoint inventory and SHA256;
- stdout/stderr paths and exit status;
- optional W&B run ID and URL, never a token.

GPU reports also record the requested backend selectors separately from the
actual route, plus `CUDA_HOME`, `TORCH_EXTENSIONS_DIR`, `nvcc --version`, and a
hash of the external CUDA toolchain provenance when lazy native extensions are
built. RTX 4080/4090 release summaries reject native training evidence that
does not identify its compiler. The V100 summary instead requires the explicit
SM70 `reference-fallback` training profile.

Before a native-training run, create its compiler gate with:

```bash
python evaluation/preflight_cuda_toolchain.py \
  --output results/toolchain-preflight.json
```

The command requires the PyTorch and `nvcc` CUDA major/minor versions to match,
detects the active GPU SM target, and compiles a small CUDA object without
launching GPU work. The report retains the compiler/provenance identity and
object hash.

Release bundles live below `results/`. Large task sample logs can be
attached to a release artifact while the manifest and summary stay in Git.
Committed result summaries must be immutable and must identify the exact GPU.
Use `evaluation/build_backend_v2_compact_bundle.py` to apply the checked-in
exclusion, secret-scan and complete-manifest policy instead of copying result
files by hand.

After PyPI publication, verify the uploaded bytes against the exact locally
validated release artifacts:

```bash
python evaluation/audit_pypi_release.py \
  --distribution rwkv7-hf=1.0.0 \
  --distribution rwkv7-kernels=1.0.0 \
  --artifact rwkv7-hf=/artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --artifact rwkv7-kernels=/artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl \
  --output results/release/pypi-v1.0.0.json \
  --harness-sha "$(git rev-parse HEAD)"
```

The audit requires both versions, a non-yanked wheel for each distribution,
valid PyPI SHA256 metadata, and exact filename, size and SHA256 equality with
the immutable local wheels.

Run every Hub smoke from a distinct empty cache so an earlier checkout cannot
be mistaken for a release redownload:

```bash
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v1.0.0 \
  --device cuda \
  --cache-dir /results/hub-smoke/rwkv7-g1d-0.1b-hf/cache \
  --require-empty-cache \
  --force-download \
  --output /results/hub-smoke/rwkv7-g1d-0.1b-hf.json
```

Repeat for all six repositories. The smoke report retains the resolved tag
commit, weight metadata, `RWKV7ForCausalLM`, `RWKV7Cache`, finite logits, and
cached generation.

The GitHub release is prepared as a draft after the final wheel pair completes
all three device gates. The exact wheel/source archives, `SHA256SUMS`, and
`release-provenance.json` are attached before that draft is published. The
release-triggered PyPI workflow downloads those assets, verifies their hashes,
source SHA, fixed FLA commit, shared harness/wheel identities and every required
device gate, and publishes the downloaded bytes. It deliberately does not
rebuild either distribution in GitHub Actions.

Before any GPU run, audit that the candidate wheel bytes contain the clean HF
model and every migrated NVIDIA payload:

```bash
python scripts/audit_release_wheels.py \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl
```

The final release verifier repeats this audit, including all 102 embedded
migration-manifest hashes, so a locally complete source tree cannot hide an
incomplete wheel.

Generate the final provenance from the three compact bundles rather than
writing it by hand:

```bash
# Run once per device before compacting the raw result directory.
python evaluation/build_backend_v2_device_validation.py \
  --device rtx-4080 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --harness-sha "$FINAL_HARNESS_SHA" \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl \
  --correctness-report /results/4080/inference.json \
  --hf-ecosystem-report /results/4080/hf-ecosystem.json \
  --training-report /results/4080/training.json \
  --quantization-report /results/4080/quantization.json \
  --fla-report /results/4080/fla.json \
  --speed-report /results/4080/speed.json \
  --finetune-report /results/4080/finetune/validation.json \
  --lm-eval-report /results/4080/lm-eval/validation-three-way.json \
  --output /results/4080/release-validation.json

python evaluation/build_backend_v2_compact_bundle.py \
  --input-dir /results/4080 \
  --output-dir /results/4080-final-compact \
  --device rtx-4080 \
  --harness-sha "$FINAL_HARNESS_SHA"

# Run after all three device summaries and compact bundles pass.
python scripts/build_release_provenance.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$(git rev-parse HEAD)" \
  --harness-sha "$FINAL_HARNESS_SHA" \
  --device-evidence rtx-4080=/results/4080-final-compact \
  --device-evidence tesla-v100=/results/v100-final-compact \
  --device-evidence rtx-4090=/results/4090-final-compact

python scripts/verify_release_assets.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$(git rev-parse HEAD)" \
  --require-validation-passed
```

Each compact bundle must contain a manifest-covered
`release-validation.json`. The generator rejects a missing gate, selector-only
route name, invalid compact manifest, different wheel byte hash, different
harness/source revision, or an FLA revision other than the pinned commit. It
then deterministically writes `release-provenance.json` and `SHA256SUMS` for the
four already-built archives; it never builds or alters those archives.
`build_backend_v2_device_validation.py` creates that device summary directly
from the individual validator outputs. It checks the report schemas, exact
wheel bytes, shared harness, pinned FLA revision, 144-unit result, actual
prefill/decode/training/quantization routes, and separate SFT/DPO/GRPO results.

After the GitHub release, validation Issue, Hub repositories, and PyPI files
exist, audit the GitHub tag/branch/PR/source tree and download every release
asset again before creating the final all-surfaces verdict:

```bash
# Render the Issue body from the passed release/speed/lm_eval JSON first.
python scripts/render_release_issue.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --speed rtx-4080=/results/4080/speed.json \
  --speed tesla-v100=/results/v100/speed.json \
  --speed rtx-4090=/results/4090/speed.json \
  --lm-eval rtx-4080=/results/4080/lm-eval/validation-three-way.json \
  --lm-eval tesla-v100=/results/v100/lm-eval/validation-three-way.json \
  --lm-eval rtx-4090=/results/4090/lm-eval/validation-three-way.json \
  --output /results/release/validation-issue-v1.0.0.md \
  --report /results/release/validation-issue-v1.0.0.json

python evaluation/audit_github_release.py \
  --repo rwkv-rs/hf-adapter \
  --tag v1.0.0 \
  --version 1.0.0 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --release-dir /artifacts/rwkv7-v1.0.0 \
  --pull-request 146 \
  --issue "$VALIDATION_ISSUE" \
  --output /results/release/github-v1.0.0.json

python scripts/verify_end_to_end_release.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --hub-audit /results/release/hub-v1.0.0.json \
  --pypi-audit /results/release/pypi-v1.0.0.json \
  --github-audit /results/release/github-v1.0.0.json \
  --hub-smoke wangyue114514/rwkv7-g1d-0.1b-hf=/results/hub-smoke/rwkv7-g1d-0.1b-hf.json \
  --hub-smoke wangyue114514/rwkv7-g1d-0.4b-hf=/results/hub-smoke/rwkv7-g1d-0.4b-hf.json \
  --hub-smoke wangyue114514/rwkv7-g1g-1.5b-hf=/results/hub-smoke/rwkv7-g1g-1.5b-hf.json \
  --hub-smoke wangyue114514/rwkv7-g1g-2.9b-hf=/results/hub-smoke/rwkv7-g1g-2.9b-hf.json \
  --hub-smoke wangyue114514/rwkv7-g1g-7.2b-hf=/results/hub-smoke/rwkv7-g1g-7.2b-hf.json \
  --hub-smoke wangyue114514/rwkv7-g1g-13.3b-hf=/results/hub-smoke/rwkv7-g1g-13.3b-hf.json \
  --output /results/release/end-to-end-v1.0.0.json
```

This final verifier repeats the three-device release-asset gate, requires the
six Hub tags and unchanged weight baselines, exact PyPI wheel bytes, a GitHub
tag contained in `main`, merged release PR, required architecture/evaluation
documentation, a comprehensive public validation Issue, and six genuine
empty-cache Hub redownload/load/cache/generation smokes.

The historical fused implementation remains on `perf/native-kernels-v0.8`.
The clean optional-package migration is reviewed separately on
`perf/optional-kernels-v1`; neither branch changes the readable reference
contract before its release gates pass.
