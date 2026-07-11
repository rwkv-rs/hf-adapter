from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_end_to_end_release", ROOT / "scripts" / "verify_end_to_end_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture():
    version = "1.0.0"
    source_sha = "a" * 40
    harness_sha = "b" * 40
    artifacts = {
        f"rwkv7_hf-{version}-py3-none-any.whl": {
            "size": 10,
            "sha256": "1" * 64,
        },
        f"rwkv7_hf-{version}.tar.gz": {"size": 11, "sha256": "2" * 64},
        f"rwkv7_kernels-{version}-py3-none-any.whl": {
            "size": 12,
            "sha256": "3" * 64,
        },
        f"rwkv7_kernels-{version}.tar.gz": {
            "size": 13,
            "sha256": "4" * 64,
        },
    }
    release = {"harness_sha": harness_sha, "artifacts": artifacts}
    repositories = []
    smokes = {}
    for index, repo in enumerate(sorted(MODULE.HUB_REPOSITORIES)):
        revision = f"{index + 1:040x}"
        repositories.append(
            {
                "repo": repo,
                "status": "passed",
                "resolved_revision": revision,
                "code_sha256": {"modeling_rwkv7.py": {"match": True}},
            }
        )
        smokes[repo] = {
            "status": "passed",
            "model": repo,
            "revision": f"v{version}",
            "commit": revision,
            "model_class": "RWKV7ForCausalLM",
            "cache_class": "RWKV7Cache",
            "generated": [1, 2],
            "download": {
                "force_download": True,
                "require_empty_cache": True,
                "cache_was_empty": True,
                "cache_dir": f"/fresh/{index}",
            },
        }
    hub = {
        "schema": "rwkv7-hub-release-audit-v1",
        "status": "passed",
        "required_tag": f"v{version}",
        "revision": "main",
        "code_sha": source_sha,
        "weight_baseline": "/baseline.json",
        "repositories": repositories,
    }
    pypi = {
        "schema": "rwkv7-pypi-release-audit-v1",
        "status": "passed",
        "harness_sha": harness_sha,
        "distributions": [],
    }
    for project, wheel_name in (
        ("rwkv7-hf", f"rwkv7_hf-{version}-py3-none-any.whl"),
        ("rwkv7-kernels", f"rwkv7_kernels-{version}-py3-none-any.whl"),
    ):
        pypi["distributions"].append(
            {
                "project": project,
                "version": version,
                "status": "passed",
                "expected_artifact": {
                    "filename": wheel_name,
                    **artifacts[wheel_name],
                },
            }
        )
    github = {
        "schema": "rwkv7-github-release-audit-v1",
        "status": "passed",
        "repository": "rwkv-rs/hf-adapter",
        "tag": f"v{version}",
        "version": version,
        "source_sha": source_sha,
        "tag_commit": source_sha,
        "default_branch": "main",
        "pull_request": {"base": "main", "merged_at": "now"},
        "required_source_paths": {"missing": []},
        "issue": {"missing_terms": [], "url": "https://github/issue"},
        "release": {
            "url": "https://github/release",
            "assets": {
                name: {"match": True, "github": identity}
                for name, identity in artifacts.items()
            },
        },
    }
    return version, source_sha, release, hub, pypi, github, smokes


def test_end_to_end_evidence_accepts_all_release_surfaces():
    version, source_sha, release, hub, pypi, github, smokes = fixture()
    result = MODULE.validate_external_evidence(
        version=version,
        source_sha=source_sha,
        release=release,
        hub=hub,
        pypi=pypi,
        github=github,
        smokes=smokes,
    )
    assert result["repositories"] == sorted(MODULE.HUB_REPOSITORIES)


def test_end_to_end_evidence_rejects_cached_hub_smoke():
    version, source_sha, release, hub, pypi, github, smokes = fixture()
    next(iter(smokes.values()))["download"]["cache_was_empty"] = False
    with pytest.raises(ValueError, match="fresh Hub redownload"):
        MODULE.validate_external_evidence(
            version=version,
            source_sha=source_sha,
            release=release,
            hub=hub,
            pypi=pypi,
            github=github,
            smokes=smokes,
        )
