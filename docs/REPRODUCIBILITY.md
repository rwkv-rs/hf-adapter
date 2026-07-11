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

The GitHub release is prepared as a draft after the final wheel pair completes
all three device gates. The exact wheel/source archives, `SHA256SUMS`, and
`release-provenance.json` are attached before that draft is published. The
release-triggered PyPI workflow downloads those assets, verifies their hashes,
source SHA, fixed FLA commit, shared harness/wheel identities and every required
device gate, and publishes the downloaded bytes. It deliberately does not
rebuild either distribution in GitHub Actions.

Generate the final provenance from the three compact bundles rather than
writing it by hand:

```bash
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

The historical fused implementation remains on `perf/native-kernels-v0.8`.
The clean optional-package migration is reviewed separately on
`perf/optional-kernels-v1`; neither branch changes the readable reference
contract before its release gates pass.
