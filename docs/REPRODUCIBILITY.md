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
does not identify its compiler. V100 results retain the explicit SM70
`reference-fallback` profile as historical evidence, not as a release gate.

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
  --modules-cache-dir /results/hub-smoke/rwkv7-g1d-0.1b-hf/modules-cache \
  --require-empty-cache \
  --force-download \
  --require-package-free \
  --output /results/hub-smoke/rwkv7-g1d-0.1b-hf.json
```

Repeat for all six repositories. The smoke report retains the resolved tag
commit, weight metadata, `RWKV7ForCausalLM`, `RWKV7Cache`, finite logits, and
cached generation. `--require-package-free` additionally proves that neither
project wheel nor a local source checkout is importable in the smoke
environment; the loaded architecture therefore came entirely from the tagged
Hub repository.

The release run uses the checked-in sequential wrapper so no repository or
remote-code cache is accidentally reused:

```bash
python scripts/run_hub_release_smokes.py \
  --output-dir /results/hub-smoke-v1.0.0 \
  --revision v1.0.0 \
  --device cuda
```

Build the four distribution archives from a clean detached checkout with the
same pinned backend versions declared by both `pyproject.toml` files:

```bash
python3.12 -m venv /tmp/rwkv7-release-build
PY=/tmp/rwkv7-release-build/bin/python
$PY -m pip install --only-binary=:all: \
  "pip==26.1.2" "build==1.5.0" "setuptools==82.0.1" \
  "wheel==0.47.0" "twine==7.0.0" "packaging==26.3"
export TZ=UTC LC_ALL=C.UTF-8 PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct "$FINAL_SOURCE_SHA")"
$PY -m build --no-isolation --wheel --sdist \
  --outdir /artifacts/rwkv7-v1.0.0 .
$PY -m build --no-isolation --wheel --sdist \
  --outdir /artifacts/rwkv7-v1.0.0 kernels
$PY -m twine check --strict /artifacts/rwkv7-v1.0.0/*.{whl,tar.gz}
```

The GitHub release is prepared as a draft after the final wheel pair completes
the RTX 4080 and RTX 4090 gates. The exact wheel/source archives, two compact
device-evidence archives, `SHA256SUMS`, and `release-provenance.json` are
attached before that draft is published. The release-triggered PyPI workflow
downloads those assets, safely unpacks and revalidates both compact bundles,
rebuilds the provenance/checksum bytes, and verifies their source SHA, fixed
FLA commit, shared harness/wheel identities, and every required device gate.
Only then does it publish the downloaded distribution bytes; it deliberately
does not rebuild either distribution in GitHub Actions.

Before any GPU run, audit that the candidate wheel bytes contain the clean HF
model and every migrated NVIDIA payload:

```bash
python scripts/audit_release_wheels.py \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl
```

The final release verifier repeats this audit, including all 102 embedded
migration-manifest destination hashes, Git-blob identity for the 86 exact
transfers, the sixteen declared clean-boundary adaptations, the reconstructed
153-file historical tree, and the separate byte-identical v0.10 Graph/Triton
recurrent subtree. A locally complete source tree therefore cannot hide an
incomplete wheel.

The same final verifier opens both `.tar.gz` source distributions without
extracting them, rejects traversal, links, devices and duplicate members,
checks `PKG-INFO` plus `pyproject.toml`, and requires every packaged HF/kernel
file to be byte-identical to the corresponding already-audited wheel member.
It also compares every package-owned member of both wheels byte-for-byte with
the checked-out release tag, rejects wheel payload outside the single owned
package roots and `.dist-info` directory, and binds each install-relevant sdist
file to the checkout. Unowned top-level wheel modules, `.data` installs,
`setup.py` hooks, and modified `pyproject.toml`/`setup.cfg` files therefore fail
closed. Wheel `WHEEL`, `top_level.txt`, console entry points, the complete
`Requires-Dist`/extras contract, and every `RECORD` hash/size are also bound to
the checked-out projects. The kernel wheel must carry PEP 639
`License-Expression: MIT` and
`License-File: LICENSE` metadata plus the byte-exact `kernels/LICENSE` payload;
the kernel sdist must contain the same checkout-owned license. Thus PyPI cannot
receive a correct wheel paired with a stale or unsafe sdist, nor mutually
consistent archives built from a different source tree.

Generate the final provenance from the required compact bundles rather than
writing it by hand:

```bash
# Start this marker before the first GPU command on each release device. Finish
# the complete RTX 4080 run before starting RTX 4090.
python evaluation/record_device_acceptance.py start \
  --device rtx-4080 \
  --source-sha "$FINAL_SOURCE_SHA" \
  --harness-sha "$FINAL_HARNESS_SHA" \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl \
  --output /results/4080/device-acceptance.json

# Run once per device after all validators have completed.
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

python evaluation/record_device_acceptance.py finish \
  --run-report /results/4080/device-acceptance.json \
  --release-validation /results/4080/release-validation.json

python evaluation/build_backend_v2_compact_bundle.py \
  --input-dir /results/4080 \
  --output-dir /results/4080-final-compact \
  --device rtx-4080 \
  --harness-sha "$FINAL_HARNESS_SHA"

# Run after both required device summaries and compact bundles pass.
python scripts/build_release_provenance.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$(git rev-parse HEAD)" \
  --harness-sha "$FINAL_HARNESS_SHA" \
  --device-evidence rtx-4080=/results/4080-final-compact \
  --device-evidence rtx-4090=/results/4090-final-compact

python scripts/verify_release_assets.py \
  --directory /artifacts/rwkv7-v1.0.0 \
  --version 1.0.0 \
  --source-sha "$(git rev-parse HEAD)" \
  --require-validation-passed
```

Each compact bundle must contain manifest-covered `release-validation.json`
and `device-acceptance.json` files. The generator cryptographically binds the
run marker to the device summary and rejects overlapping or out-of-order
RTX 4080 -> RTX 4090 lifetimes, a missing gate, selector-only route
name, invalid compact manifest, different wheel byte hash, different
harness/source revision, or an FLA revision other than the pinned commit. It
then deterministically writes `release-provenance.json` and `SHA256SUMS` for
the four already-built distribution archives plus two deterministic compact
evidence archives; it never alters the distribution bytes. The PyPI workflow
uses `build_release_provenance.py --verify-existing` to re-run the compact
bundle validators and require a byte-for-byte metadata rebuild before upload.
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
  --speed rtx-4090=/results/4090/speed.json \
  --lm-eval rtx-4080=/results/4080/lm-eval/validation-three-way.json \
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

This final verifier repeats the required-device release-asset gate, requires the
six Hub tags and unchanged weight baselines, exact PyPI wheel bytes, a GitHub
tag contained in `main`, merged release PR, required architecture/evaluation
documentation, a comprehensive public validation Issue, and six genuine
empty-cache Hub redownload/load/cache/generation smokes.

The historical fused implementation remains on `perf/native-kernels-v0.8`.
The clean optional-package migration is reviewed separately on
`perf/optional-kernels-v1`; neither branch changes the readable reference
contract before its release gates pass.
