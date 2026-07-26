#!/usr/bin/env python3
"""Audit the final GitHub tag, release bytes, and public validation issue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
import urllib.error
import urllib.request


MIGRATION_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "kernels"
    / "rwkv7_kernels"
    / "nvidia"
    / "MIGRATION_MANIFEST.json"
)
EXPECTED_MIGRATION_TRANSFER_SUMMARY = {
    "total": 102,
    "byte_identical": 86,
    "adapted_clean_boundary": 16,
}


def migration_transfer_summary(path: Path = MIGRATION_MANIFEST) -> dict[str, int]:
    """Derive the public migration denominator from its machine-readable source."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") or []
    if not isinstance(files, list):
        raise ValueError("NVIDIA migration manifest files must be a list")
    result = {"total": len(files), "byte_identical": 0, "adapted_clean_boundary": 0}
    for row in files:
        transfer = str((row or {}).get("transfer", ""))
        if transfer not in {"byte_identical", "adapted_clean_boundary"}:
            raise ValueError(f"unexpected NVIDIA migration transfer class: {transfer}")
        result[transfer] += 1
    if result != EXPECTED_MIGRATION_TRANSFER_SUMMARY:
        raise ValueError(
            "NVIDIA migration manifest canonical transfer counts differ: "
            f"expected={EXPECTED_MIGRATION_TRANSFER_SUMMARY} actual={result}"
        )
    return result


MIGRATION_TRANSFER_SUMMARY = migration_transfer_summary()


REQUIRED_ISSUE_TERMS = (
    "144",
    "lm_eval",
    "wikitext",
    "nll",
    "ppl",
    "sft",
    "dpo",
    "grpo",
    "trainer",
    "accelerate",
    "peft",
    "trl",
    "state",
    "cache",
    "greedy",
    "beam",
    "quantization",
    "route",
    "sha256",
    "rtx 4080",
    "rtx 4090",
    "fla",
    "reference",
    "optimized",
    "recurrent",
    "dense decode",
    "dplr",
    "self-chunk",
    "cuda graph",
    "sm70",
    "ada",
    "blackwell",
    "w8",
    "w4",
    "a8w8",
    "bn/tn",
    "bitsandbytes",
    "marlin",
    "torchao",
    "autograd",
    "153",
    "102-file",
    f"{MIGRATION_TRANSFER_SUMMARY['byte_identical']} are byte-identical",
    (
        f"{MIGRATION_TRANSFER_SUMMARY['adapted_clean_boundary']} are declared "
        "clean-boundary adaptations"
    ),
    "migration manifest",
    "source scope",
    "clean-boundary",
    "byte-identical",
    "v0.10",
    "sequentially",
    "non-overlapping",
)
REQUIRED_SOURCE_PATHS = (
    ".github/workflows/publish.yml",
    "docs/ARCHITECTURE.md",
    "docs/EVALUATION.md",
    "docs/NVIDIA_MIGRATION_AUDIT.md",
    "docs/REPRODUCIBILITY.md",
    "kernels/README.md",
    "kernels/rwkv7_kernels/protocol.py",
    "kernels/rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json",
    "kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json",
    "kernels/rwkv7_kernels/nvidia/RECURRENT_SOURCE_SCOPE.json",
    "kernels/rwkv7_kernels/nvidia/SOURCE_SCOPE.json",
    "rwkv7_hf/cache_rwkv7.py",
    "rwkv7_hf/configuration_rwkv7.py",
    "rwkv7_hf/modeling_rwkv7.py",
    "rwkv7_hf/ops_rwkv7.py",
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="rwkv-rs/hf-adapter")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--pull-request", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="https://api.github.com")
    parser.add_argument("--token-env", default="GH_TOKEN")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_assets(version: str) -> tuple[str, ...]:
    return (
        f"rwkv7_hf-{version}-py3-none-any.whl",
        f"rwkv7_hf-{version}.tar.gz",
        f"rwkv7_kernels-{version}-py3-none-any.whl",
        f"rwkv7_kernels-{version}.tar.gz",
        f"rwkv7-evidence-rtx-4080-{version}.tar.gz",
        f"rwkv7-evidence-rtx-4090-{version}.tar.gz",
        "SHA256SUMS",
        "release-provenance.json",
    )


def request_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rwkv7-release-audit/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str, *, token: str | None, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=request_headers(token))
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub response is not an object: {url}")
    return payload


def download_identity(url: str, *, token: str | None, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=request_headers(token))
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def resolve_tag_commit(
    *,
    repo_api: str,
    tag: str,
    get_json: Callable[[str], dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    ref = get_json(f"{repo_api}/git/ref/tags/{tag}")
    chain = []
    obj = ref.get("object") or {}
    for _ in range(8):
        kind = str(obj.get("type", ""))
        sha = str(obj.get("sha", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("GitHub tag object has no valid SHA")
        chain.append({"type": kind, "sha": sha})
        if kind == "commit":
            return sha, chain
        if kind != "tag":
            raise ValueError(f"GitHub tag resolves to unsupported object: {kind}")
        obj = get_json(f"{repo_api}/git/tags/{sha}").get("object") or {}
    raise ValueError("GitHub annotated tag chain is too deep")


def audit(
    args: argparse.Namespace,
    *,
    get_json: Callable[[str], dict[str, Any]],
    get_asset: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    if MIGRATION_TRANSFER_SUMMARY["total"] != 102:
        failures.append(
            "NVIDIA migration manifest does not contain the required 102 files"
        )
    root = args.release_dir.expanduser().resolve()
    repo_api = f"{args.api_url.rstrip('/')}/repos/{args.repo}"
    tag_commit, tag_chain = resolve_tag_commit(
        repo_api=repo_api, tag=args.tag, get_json=get_json
    )
    if tag_commit != args.source_sha:
        failures.append("GitHub tag commit differs from the release source SHA")

    repository = get_json(repo_api)
    default_branch = str(repository.get("default_branch", ""))
    if default_branch != "main":
        failures.append("GitHub default branch is not main")
    comparison = get_json(f"{repo_api}/compare/{tag_commit}...{default_branch}")
    if (
        comparison.get("status") not in {"ahead", "identical"}
        or (comparison.get("merge_base_commit") or {}).get("sha") != tag_commit
    ):
        failures.append("release tag commit is not contained in the default branch")
    pull = get_json(f"{repo_api}/pulls/{args.pull_request}")
    if not pull.get("merged_at") or (pull.get("base") or {}).get("ref") != "main":
        failures.append("release pull request is not merged into main")

    tree = get_json(f"{repo_api}/git/trees/{tag_commit}?recursive=1")
    if tree.get("truncated"):
        failures.append("GitHub source tree response is truncated")
    source_paths = {str(row.get("path")) for row in (tree.get("tree") or [])}
    missing_paths = sorted(set(REQUIRED_SOURCE_PATHS) - source_paths)
    if missing_paths:
        failures.append(
            "release source omits required files: " + ", ".join(missing_paths)
        )

    release = get_json(f"{repo_api}/releases/tags/{args.tag}")
    if release.get("draft"):
        failures.append("GitHub release is still a draft")
    if release.get("prerelease"):
        failures.append("GitHub release is marked as a prerelease")
    if not release.get("published_at"):
        failures.append("GitHub release has no publication timestamp")
    if release.get("tag_name") != args.tag:
        failures.append("GitHub release tag name mismatch")

    release_assets = {
        str(row.get("name")): row for row in (release.get("assets") or [])
    }
    assets: dict[str, dict[str, Any]] = {}
    for name in expected_assets(args.version):
        local = root / name
        remote = release_assets.get(name)
        if not local.is_file() or local.is_symlink():
            failures.append(f"local release asset is missing or unsafe: {name}")
            continue
        if remote is None or not remote.get("browser_download_url"):
            failures.append(f"GitHub release asset is missing: {name}")
            continue
        local_identity = {"size": local.stat().st_size, "sha256": sha256_file(local)}
        remote_identity = get_asset(str(remote["browser_download_url"]))
        match = local_identity == remote_identity
        assets[name] = {
            "local": local_identity,
            "github": remote_identity,
            "match": match,
        }
        if not match:
            failures.append(f"GitHub release asset bytes differ: {name}")

    issue = get_json(f"{repo_api}/issues/{args.issue}")
    if issue.get("pull_request") is not None:
        failures.append("requested validation issue is a pull request")
    body = str(issue.get("body") or "").lower().replace("lm-eval", "lm_eval")
    missing_terms = [term for term in REQUIRED_ISSUE_TERMS if term not in body]
    if args.version.lower() not in body:
        missing_terms.append(args.version.lower())
    if missing_terms:
        failures.append(
            "validation issue omits required release evidence: "
            + ", ".join(missing_terms)
        )

    return {
        "schema": "rwkv7-github-release-audit-v1",
        "status": "passed" if not failures else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": args.repo,
        "tag": args.tag,
        "version": args.version,
        "source_sha": args.source_sha,
        "tag_commit": tag_commit,
        "tag_chain": tag_chain,
        "default_branch": default_branch,
        "default_branch_comparison": {
            "status": comparison.get("status"),
            "merge_base_commit": (comparison.get("merge_base_commit") or {}).get("sha"),
        },
        "pull_request": {
            "number": args.pull_request,
            "url": pull.get("html_url"),
            "merged_at": pull.get("merged_at"),
            "merge_commit_sha": pull.get("merge_commit_sha"),
            "base": (pull.get("base") or {}).get("ref"),
            "head": (pull.get("head") or {}).get("ref"),
        },
        "required_source_paths": {
            "expected": list(REQUIRED_SOURCE_PATHS),
            "missing": missing_paths,
        },
        "release": {
            "id": release.get("id"),
            "url": release.get("html_url"),
            "published_at": release.get("published_at"),
            "assets": assets,
        },
        "issue": {
            "number": args.issue,
            "url": issue.get("html_url"),
            "state": issue.get("state"),
            "title": issue.get("title"),
            "missing_terms": missing_terms,
            "migration_transfer_summary": dict(MIGRATION_TRANSFER_SUMMARY),
        },
        "failures": failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    token = os.environ.get(args.token_env)

    def getter(url: str) -> dict[str, Any]:
        return fetch_json(url, token=token, timeout=args.timeout)

    def asset_getter(url: str) -> dict[str, Any]:
        return download_identity(url, token=token, timeout=args.timeout)

    try:
        report = audit(args, get_json=getter, get_asset=asset_getter)
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        report = {
            "schema": "rwkv7-github-release-audit-v1",
            "status": "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": args.repo,
            "tag": args.tag,
            "version": args.version,
            "source_sha": args.source_sha,
            "pull_request": args.pull_request,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    report["command"] = sys.argv
    report["api_url"] = args.api_url
    return report


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
