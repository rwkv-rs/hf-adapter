#!/usr/bin/env python3
"""Verify the complete GitHub, Hub, PyPI, and fresh-download release state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_release_assets import DEVICES  # noqa: E402
from scripts.verify_release_assets import verify as verify_release_assets  # noqa: E402


HUB_REPOSITORIES = {
    "wangyue114514/rwkv7-g1d-0.1b-hf",
    "wangyue114514/rwkv7-g1d-0.4b-hf",
    "wangyue114514/rwkv7-g1g-1.5b-hf",
    "wangyue114514/rwkv7-g1g-2.9b-hf",
    "wangyue114514/rwkv7-g1g-7.2b-hf",
    "wangyue114514/rwkv7-g1g-13.3b-hf",
}
HUB_CANONICAL_CODE = {
    "cache_rwkv7.py",
    "chat_template.jinja",
    "configuration_rwkv7.py",
    "modeling_rwkv7.py",
    "ops_rwkv7.py",
    "tokenization_rwkv7.py",
}
HUB_RELEASE_FILES = {"README.md", "config.json", *HUB_CANONICAL_CODE}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--hub-audit", type=Path, required=True)
    parser.add_argument("--pypi-audit", type=Path, required=True)
    parser.add_argument("--github-audit", type=Path, required=True)
    parser.add_argument(
        "--hub-smoke",
        action="append",
        required=True,
        help="repo=/path/to/fresh-download-smoke.json; repeat for all six repos",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe release evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"release evidence is not an object: {path}")
    return payload


def parse_smokes(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--hub-smoke must use repo=path: {value}")
        repo, raw_path = value.split("=", 1)
        if repo in result or repo not in HUB_REPOSITORIES or not raw_path:
            raise ValueError(f"invalid or duplicate Hub smoke: {value}")
        result[repo] = Path(raw_path)
    if set(result) != HUB_REPOSITORIES:
        raise ValueError("fresh-download smoke does not cover the six Hub repositories")
    return result


def require_report(report: dict[str, Any], schema: str, label: str) -> None:
    if report.get("schema") != schema or report.get("status") != "passed":
        raise ValueError(f"{label} audit did not pass")


def verify_hub_provenance_files(
    hub: dict[str, Any], *, source_sha: str, tag: str
) -> dict[str, Any]:
    """Open and cross-check the stage and pre-release weight manifests."""

    loaded = {}
    for field in ("release_manifest", "weight_baseline"):
        identity = hub.get(field) or {}
        path = Path(str(identity.get("path", "")))
        payload = safe_json(path)
        digest = sha256_file(path.expanduser().resolve())
        if identity.get("sha256") != digest:
            raise ValueError(f"Hub {field} SHA256 differs")
        loaded[field] = (path.expanduser().resolve(), payload, digest)

    _stage_path, stage, stage_sha = loaded["release_manifest"]
    if (
        stage.get("schema") != "rwkv7-hub-release-stage-v1"
        or stage.get("source_sha") != source_sha
        or stage.get("tag") != tag
    ):
        raise ValueError("Hub stage manifest source/tag/schema differs")
    stage_rows = {
        str(row.get("repo_id")): row for row in stage.get("repositories", [])
    }
    if len(stage.get("repositories", [])) != 6 or set(stage_rows) != HUB_REPOSITORIES:
        raise ValueError("Hub stage manifest repository set differs")

    _baseline_path, baseline, baseline_sha = loaded["weight_baseline"]
    baseline_rows = {
        str(row.get("repo")): row for row in baseline.get("repositories", [])
    }
    if set(baseline_rows) != HUB_REPOSITORIES:
        raise ValueError("Hub weight baseline repository set differs")
    audit_rows = {
        str(row.get("repo")): row for row in hub.get("repositories", [])
    }
    for repo in HUB_REPOSITORIES:
        staged = stage_rows[repo]
        audited = audit_rows[repo]
        if (
            staged.get("source_sha") != source_sha
            or set(staged.get("files") or []) != HUB_RELEASE_FILES
            or staged.get("weights") != audited.get("weights")
            or baseline_rows[repo].get("weights") != audited.get("weights")
        ):
            raise ValueError(f"Hub stage/weight provenance differs: {repo}")
        expected = staged.get("file_sha256") or {}
        actual = audited.get("release_file_sha256") or {}
        if set(expected) != HUB_RELEASE_FILES or any(
            (actual.get(name) or {}).get("expected") != digest
            or not (actual.get(name) or {}).get("match")
            for name, digest in expected.items()
        ):
            raise ValueError(f"Hub staged-file provenance differs: {repo}")
    return {
        "stage_manifest_sha256": stage_sha,
        "weight_baseline_sha256": baseline_sha,
    }


def validate_external_evidence(
    *,
    version: str,
    source_sha: str,
    release: dict[str, Any],
    hub: dict[str, Any],
    pypi: dict[str, Any],
    github: dict[str, Any],
    smokes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tag = f"v{version}"
    require_report(hub, "rwkv7-hub-release-audit-v1", "Hub")
    require_report(pypi, "rwkv7-pypi-release-audit-v1", "PyPI")
    require_report(github, "rwkv7-github-release-audit-v1", "GitHub")
    if hub.get("required_tag") != tag or hub.get("revision") != "main":
        raise ValueError("Hub audit does not bind main to the release tag")
    if (
        hub.get("code_sha") != source_sha
        or not (hub.get("weight_baseline") or {}).get("sha256")
        or not (hub.get("release_manifest") or {}).get("sha256")
        or (hub.get("source_checkout") or {}).get("commit") != source_sha
    ):
        raise ValueError(
            "Hub audit lacks source, stage-manifest, or weight-baseline provenance"
        )

    repositories = {str(row.get("repo")): row for row in hub.get("repositories", [])}
    if set(repositories) != HUB_REPOSITORIES:
        raise ValueError("Hub audit does not cover the six release repositories")
    for repo, row in repositories.items():
        if row.get("status") != "passed":
            raise ValueError(f"Hub repository audit failed: {repo}")
        resolved = str(row.get("resolved_revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise ValueError(f"Hub repository revision is missing: {repo}")
        code = row.get("code_sha256") or {}
        release_files = row.get("release_file_sha256") or {}
        weights = row.get("weights") or {}
        if set(code) != HUB_CANONICAL_CODE or not all(
            value.get("match") for value in code.values()
        ):
            raise ValueError(f"Hub canonical code was not byte-verified: {repo}")
        if set(release_files) != HUB_RELEASE_FILES or not all(
            value.get("match") for value in release_files.values()
        ):
            raise ValueError(f"Hub staged release files were not byte-verified: {repo}")
        if not weights or any(
            not value.get("sha256") or value.get("size") is None
            for value in weights.values()
        ):
            raise ValueError(f"Hub weight identities are incomplete: {repo}")
        if (row.get("tags") or {}).get(tag) != resolved:
            raise ValueError(f"Hub tag target differs from audited main: {repo}")

    if pypi.get("harness_sha") != release.get("harness_sha"):
        raise ValueError("PyPI audit harness SHA differs from GPU release evidence")
    distributions = {
        str(row.get("project")): row for row in pypi.get("distributions", [])
    }
    if set(distributions) != {"rwkv7-hf", "rwkv7-kernels"}:
        raise ValueError("PyPI audit does not cover both distributions")
    for project, wheel_name in (
        ("rwkv7-hf", f"rwkv7_hf-{version}-py3-none-any.whl"),
        ("rwkv7-kernels", f"rwkv7_kernels-{version}-py3-none-any.whl"),
    ):
        row = distributions[project]
        expected = row.get("expected_artifact") or {}
        identity = (release.get("artifacts") or {}).get(wheel_name) or {}
        if (
            row.get("status") != "passed"
            or row.get("version") != version
            or expected.get("filename") != wheel_name
            or expected.get("size") != identity.get("size")
            or expected.get("sha256") != identity.get("sha256")
        ):
            raise ValueError(f"PyPI bytes do not match the validated wheel: {project}")

    if (
        github.get("repository") != "rwkv-rs/hf-adapter"
        or github.get("tag") != tag
        or github.get("version") != version
        or github.get("source_sha") != source_sha
        or github.get("tag_commit") != source_sha
        or github.get("default_branch") != "main"
        or (github.get("pull_request") or {}).get("base") != "main"
        or not (github.get("pull_request") or {}).get("merged_at")
        or (github.get("required_source_paths") or {}).get("missing")
        or (github.get("issue") or {}).get("missing_terms")
    ):
        raise ValueError("GitHub tag, branch, PR, docs, or issue audit is incomplete")
    github_assets = (github.get("release") or {}).get("assets") or {}
    for name, identity in (release.get("artifacts") or {}).items():
        row = github_assets.get(name) or {}
        if not row.get("match") or row.get("github") != identity:
            raise ValueError(
                f"GitHub release bytes differ from validated asset: {name}"
            )
    release_evidence = release.get("evidence") or {}
    if set(release_evidence) != DEVICES:
        raise ValueError("validated release does not cover both compact evidence assets")
    for device, evidence in release_evidence.items():
        name = str(evidence.get("archive", ""))
        identity = {
            "size": evidence.get("size"),
            "sha256": evidence.get("sha256"),
        }
        row = github_assets.get(name) or {}
        if not name or not row.get("match") or row.get("github") != identity:
            raise ValueError(
                f"GitHub release bytes differ from validated evidence: {device}"
            )

    for repo, smoke in smokes.items():
        row = repositories[repo]
        download = smoke.get("download") or {}
        package_free = smoke.get("package_free") or {}
        if (
            smoke.get("schema") != "rwkv7-hub-release-smoke-v1"
            or smoke.get("status") != "passed"
            or smoke.get("model") != repo
            or smoke.get("revision") != tag
            or smoke.get("commit") != row.get("resolved_revision")
            or not download.get("force_download")
            or not download.get("require_empty_cache")
            or download.get("cache_was_empty") is not True
            or download.get("modules_cache_was_empty") is not True
            or not download.get("cache_dir")
            or not download.get("modules_cache_dir")
            or smoke.get("model_class") != "RWKV7ForCausalLM"
            or smoke.get("cache_class") != "RWKV7Cache"
            or not smoke.get("generated")
            or package_free.get("required") is not True
            or package_free.get("passed") is not True
            or any((package_free.get("installed_distributions") or {}).values())
            or any((package_free.get("local_import_origins") or {}).values())
        ):
            raise ValueError(
                f"fresh Hub redownload/load/cache/generation failed: {repo}"
            )

    return {
        "repositories": sorted(repositories),
        "pypi_distributions": sorted(distributions),
        "github_release": (github.get("release") or {}).get("url"),
        "github_issue": (github.get("issue") or {}).get("url"),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    release = verify_release_assets(
        SimpleNamespace(
            directory=args.directory,
            version=args.version,
            source_sha=args.source_sha,
            require_validation_passed=True,
        )
    )
    audit_paths = {
        "hub": args.hub_audit.expanduser().resolve(),
        "pypi": args.pypi_audit.expanduser().resolve(),
        "github": args.github_audit.expanduser().resolve(),
    }
    reports = {name: safe_json(path) for name, path in audit_paths.items()}
    hub_provenance = verify_hub_provenance_files(
        reports["hub"], source_sha=args.source_sha, tag=f"v{args.version}"
    )
    smoke_paths = parse_smokes(args.hub_smoke)
    smokes = {repo: safe_json(path) for repo, path in smoke_paths.items()}
    external = validate_external_evidence(
        version=args.version,
        source_sha=args.source_sha,
        release=release,
        hub=reports["hub"],
        pypi=reports["pypi"],
        github=reports["github"],
        smokes=smokes,
    )
    evidence = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in audit_paths.items()
    }
    evidence["hub_smokes"] = {
        repo: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for repo, path in smoke_paths.items()
    }
    evidence["hub_provenance"] = hub_provenance
    return {
        "schema": "rwkv7-end-to-end-release-verification-v1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": args.version,
        "source_sha": args.source_sha,
        "release": release,
        "external": external,
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
