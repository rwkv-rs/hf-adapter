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

The historical fused implementation remains on `perf/native-kernels-v0.8`.
The clean optional-package migration is reviewed separately on
`perf/optional-kernels-v1`; neither branch changes the readable reference
contract before its release gates pass.
