# Maintenance CI and lifecycle design

Lifecycle: **Active implementation design**. Keep this file as a historical
decision record after the implementation merges.

## Scope

This change closes three repository-maintenance gaps without changing RWKV-7
math, runtime routing, kernel defaults, checkpoint metadata, or benchmark
claims:

1. publish a release-versioned lifecycle for experimental backends, flags, and
   compatibility shims;
2. classify and enforce the existing pytest suite with stable execution/domain
   markers; and
3. test the supported Transformers/PEFT/TRL range through minimum and current
   clean-install lanes.

The work intentionally excludes a scheduled CUDA job, Hub publication, source
tree reorganization, and any GPU benchmark rerun.

## Chosen approach

The lifecycle contract belongs in `docs/BACKENDS.md`, the canonical backend
boundary document. Experimental flags receive an introduced/default/removal
table and a minimum two-minor-release warning window. Stable converted-model
entrypoints and serialized formats require at least one full minor release of
compatibility after a replacement ships. Removal before the stated release is
allowed only for a security or correctness emergency and must be documented in
release notes. A static documentation test keeps the table and policy language
from disappearing.

Pytest markers are registered under `[tool.pytest.ini_options]` with strict
marker validation. A centralized `tests/marker_policy.py` classifies collected
tests by path into `cpu`, `cuda`, `sm70`, `ada`, `blackwell`, `apple`, `slow`,
and `model_required`. `tests/conftest.py` applies the policy at collection time
and fails collection for invalid relationships (for example, a GPU-family
marker without `cuda`, or `model_required` without `slow`). Centralized rules
avoid editing more than 150 test modules while still making the classification
reviewable and testable. The `cpu` marker means the item is safe in the
CPU/offline collection lane; hardware-domain markers are additive, because many
kernel files also contain CPU fallback and policy tests.

## Dependency and CI contract

Package metadata declares the supported major range and the evidence-backed
minimums: Transformers `>=5.12.1,<6`, PEFT `>=0.19.1,<1`, and TRL `>=1.7,<2`.
The minimum lane installs exact lower-bound constraints. The current lane uses
normal dependency resolution inside the supported major range, so it detects a
new compatible release without editing the workflow first. Both lanes build a
non-editable wheel in a fresh virtual environment, print resolved versions, run
`pip check`, prove collection with strict markers, and execute a focused
CPU/offline compatibility suite covering native HF APIs, PEFT/Trainer/TRL
interfaces, remote-code packaging, and lifecycle/marker contracts.

The workflow runs on pull requests, `main`, weekly schedule, and manual
dispatch. A small script owns the profile so local reproduction and CI use the
same command. CI does not download model checkpoints and does not infer GPU
support from CPU compatibility.

## Rejected alternatives

- A full Cartesian matrix across every library version would be expensive and
  would not define which combinations are actually supported.
- Pinning only a single current environment would miss lower-bound regressions.
- Hand-adding module markers to every test file would create a noisy structural
  change and make future source moves harder.
- Treating filename markers as automatic skip conditions would hide CPU
  fallback coverage; selection stays explicit through `pytest -m`.

## Verification

- documentation freshness, repository-layout, and Markdown-link tests;
- marker-policy unit tests plus `pytest --collect-only` under strict markers;
- focused compatibility suite in the current local environment;
- isolated minimum and current clean-install runs where network access exists;
- full CPU/offline pytest regression before the implementation commit.
