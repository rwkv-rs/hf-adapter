from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_github_release", ROOT / "evaluation" / "audit_github_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path: Path):
    version = "1.0.0"
    source_sha = "a" * 40
    assets = {}
    identities = {}
    for name in MODULE.expected_assets(version):
        path = tmp_path / name
        path.write_bytes(("payload:" + name).encode())
        url = "https://download.invalid/" + name
        assets[name] = {"name": name, "browser_download_url": url}
        identities[url] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    body = " ".join([*MODULE.REQUIRED_ISSUE_TERMS, version])
    payloads = {
        "/git/ref/tags/v1.0.0": {"object": {"type": "tag", "sha": "b" * 40}},
        "/git/tags/" + "b" * 40: {"object": {"type": "commit", "sha": source_sha}},
        "": {"default_branch": "main"},
        f"/compare/{source_sha}...main": {
            "status": "identical",
            "merge_base_commit": {"sha": source_sha},
        },
        "/pulls/146": {
            "html_url": "https://github.invalid/pull/146",
            "merged_at": "2026-08-28T00:00:00Z",
            "merge_commit_sha": "c" * 40,
            "base": {"ref": "main"},
            "head": {"ref": "perf/optional-kernels-v1"},
        },
        f"/git/trees/{source_sha}?recursive=1": {
            "truncated": False,
            "tree": [{"path": path} for path in MODULE.REQUIRED_SOURCE_PATHS],
        },
        "/releases/tags/v1.0.0": {
            "id": 1,
            "html_url": "https://github.invalid/release",
            "tag_name": "v1.0.0",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-08-28T00:00:00Z",
            "assets": list(assets.values()),
        },
        "/issues/123": {
            "number": 123,
            "html_url": "https://github.invalid/issue/123",
            "state": "open",
            "title": "RWKV7 release validation",
            "body": body,
        },
    }
    args = SimpleNamespace(
        release_dir=tmp_path,
        api_url="https://api.github.invalid",
        repo="rwkv-rs/hf-adapter",
        tag="v1.0.0",
        version=version,
        source_sha=source_sha,
        issue=123,
        pull_request=146,
    )

    def get_json(url):
        key = url.split("/repos/rwkv-rs/hf-adapter", 1)[1]
        return payloads[key]

    return args, get_json, lambda url: identities[url], payloads, identities


def test_github_release_audit_accepts_exact_tag_assets_and_issue(tmp_path: Path):
    args, getter, asset_getter, _, _ = fixture(tmp_path)
    report = MODULE.audit(args, get_json=getter, get_asset=asset_getter)
    assert report["status"] == "passed"
    assert report["tag_commit"] == args.source_sha
    assert len(report["release"]["assets"]) == 8


def test_github_release_audit_rejects_drift_and_incomplete_issue(tmp_path: Path):
    args, getter, asset_getter, payloads, identities = fixture(tmp_path)
    payloads["/issues/123"]["body"] = "1.0.0"
    url = next(iter(identities))
    identities[url] = {"size": 1, "sha256": "0" * 64}
    report = MODULE.audit(args, get_json=getter, get_asset=asset_getter)
    assert report["status"] == "failed"
    assert any("asset bytes differ" in row for row in report["failures"])
    assert any("omits required" in row for row in report["failures"])
