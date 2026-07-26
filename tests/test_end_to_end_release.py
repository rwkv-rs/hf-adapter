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
    evidence = {
        device: {
            "archive": f"rwkv7-evidence-{device}-{version}.tar.gz",
            "size": 20 + index,
            "sha256": str(5 + index) * 64,
            "compact_bundle_manifest_sha256": str(7 + index) * 64,
        }
        for index, device in enumerate(("rtx-4080", "rtx-4090"))
    }
    release = {
        "harness_sha": harness_sha,
        "artifacts": artifacts,
        "evidence": evidence,
    }
    repositories = []
    smokes = {}
    for index, repo in enumerate(sorted(MODULE.HUB_REPOSITORIES)):
        revision = f"{index + 1:040x}"
        repositories.append(
            {
                "repo": repo,
                "status": "passed",
                "resolved_revision": revision,
                "code_sha256": {
                    name: {"match": True}
                    for name in MODULE.HUB_CANONICAL_CODE
                },
                "release_file_sha256": {
                    name: {"match": True}
                    for name in MODULE.HUB_RELEASE_FILES
                },
                "weights": {
                    "model.safetensors": {"size": 1, "sha256": "8" * 64}
                },
                "tags": {f"v{version}": revision},
            }
        )
        smokes[repo] = {
            "schema": "rwkv7-hub-release-smoke-v1",
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
                "modules_cache_was_empty": True,
                "modules_cache_dir": f"/fresh/{index}/modules",
            },
            "package_free": {
                "required": True,
                "passed": True,
                "installed_distributions": {
                    "rwkv7-hf": None,
                    "rwkv7-kernels": None,
                },
                "local_import_origins": {
                    "rwkv7_hf": None,
                    "rwkv7_kernels": None,
                },
            },
        }
    hub = {
        "schema": "rwkv7-hub-release-audit-v1",
        "status": "passed",
        "required_tag": f"v{version}",
        "revision": "main",
        "code_sha": source_sha,
        "source_checkout": {"commit": source_sha},
        "weight_baseline": {"path": "/baseline.json", "sha256": "8" * 64},
        "release_manifest": {"path": "/stage.json", "sha256": "9" * 64},
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
                **{
                    name: {"match": True, "github": identity}
                    for name, identity in artifacts.items()
                },
                **{
                    row["archive"]: {
                        "match": True,
                        "github": {
                            "size": row["size"],
                            "sha256": row["sha256"],
                        },
                    }
                    for row in evidence.values()
                },
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


def test_end_to_end_evidence_rejects_missing_compact_release_asset():
    version, source_sha, release, hub, pypi, github, smokes = fixture()
    archive = release["evidence"]["rtx-4090"]["archive"]
    github["release"]["assets"].pop(archive)
    with pytest.raises(ValueError, match="validated evidence: rtx-4090"):
        MODULE.validate_external_evidence(
            version=version,
            source_sha=source_sha,
            release=release,
            hub=hub,
            pypi=pypi,
            github=github,
            smokes=smokes,
        )


def test_hub_stage_and_weight_files_are_reopened_and_cross_checked(tmp_path):
    version, source_sha, _release, hub, _pypi, _github, _smokes = fixture()
    stage_rows = []
    baseline_rows = []
    for row in hub["repositories"]:
        file_sha = {name: "7" * 64 for name in MODULE.HUB_RELEASE_FILES}
        row["release_file_sha256"] = {
            name: {"expected": digest, "match": True}
            for name, digest in file_sha.items()
        }
        stage_rows.append(
            {
                "repo_id": row["repo"],
                "source_sha": source_sha,
                "files": sorted(MODULE.HUB_RELEASE_FILES),
                "file_sha256": file_sha,
                "weights": row["weights"],
            }
        )
        baseline_rows.append({"repo": row["repo"], "weights": row["weights"]})
    stage = tmp_path / "stage.json"
    stage.write_text(
        MODULE.json.dumps(
            {
                "schema": "rwkv7-hub-release-stage-v1",
                "source_sha": source_sha,
                "tag": f"v{version}",
                "repositories": stage_rows,
            }
        )
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(MODULE.json.dumps({"repositories": baseline_rows}))
    hub["release_manifest"] = {
        "path": str(stage),
        "sha256": MODULE.sha256_file(stage),
    }
    hub["weight_baseline"] = {
        "path": str(baseline),
        "sha256": MODULE.sha256_file(baseline),
    }
    result = MODULE.verify_hub_provenance_files(
        hub, source_sha=source_sha, tag=f"v{version}"
    )
    assert result["stage_manifest_sha256"] == MODULE.sha256_file(stage)

    baseline.write_text("{}")
    with pytest.raises(ValueError, match="weight_baseline SHA256 differs"):
        MODULE.verify_hub_provenance_files(
            hub, source_sha=source_sha, tag=f"v{version}"
        )
