#!/usr/bin/env python3
"""Audit the six self-contained Hub repositories before and after release.

The audit never downloads model weights.  It compares the small canonical HF
source files with a local source-of-truth directory and records the Hub LFS
SHA256/size metadata so a later release audit can prove that weights did not
change while code, cards, and tags were updated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


CANONICAL_CODE = (
    "cache_rwkv7.py",
    "chat_template.jinja",
    "configuration_rwkv7.py",
    "modeling_rwkv7.py",
    "ops_rwkv7.py",
    "tokenization_rwkv7.py",
)
HUB_REPOSITORIES = {
    "wangyue114514/rwkv7-g1d-0.1b-hf",
    "wangyue114514/rwkv7-g1d-0.4b-hf",
    "wangyue114514/rwkv7-g1g-1.5b-hf",
    "wangyue114514/rwkv7-g1g-2.9b-hf",
    "wangyue114514/rwkv7-g1g-7.2b-hf",
    "wangyue114514/rwkv7-g1g-13.3b-hf",
}
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "conversion_manifest.json",
    "generation_config.json",
    "rwkv_vocab_v20230424.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    *CANONICAL_CODE,
)
FORBIDDEN_FILES = (
    "adapter_manifest.py",
    "kernel_bridge.py",
    "model_cache.py",
    "model_config.py",
    "native_model.py",
)
EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_rwkv7.RWKV7Config",
    "AutoModel": "modeling_rwkv7.RWKV7Model",
    "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--require-tag")
    parser.add_argument("--code-sha", required=True)
    parser.add_argument(
        "--weight-baseline",
        type=Path,
        help="Earlier audit JSON whose LFS weight hashes must remain unchanged.",
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Staging manifest whose exact small-file bytes must be on the Hub.",
    )
    return parser.parse_args()


def _weight_rows(siblings: list[Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        name = str(sibling.rfilename)
        if not name.endswith(".safetensors"):
            continue
        lfs = getattr(sibling, "lfs", None)
        rows[name] = {
            "size": getattr(lfs, "size", None) if lfs is not None else None,
            "sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
        }
    return rows


def load_weight_baseline(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {
        str(row["repo"]): dict(row.get("weights", {}))
        for row in payload.get("repositories", [])
    }


def load_release_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema") != "rwkv7-hub-release-stage-v1":
        raise ValueError("unexpected Hub release-stage manifest schema")
    rows = payload.get("repositories") or []
    repositories = {str(row.get("repo_id")): row for row in rows}
    if len(rows) != 6 or set(repositories) != HUB_REPOSITORIES:
        raise ValueError("Hub release-stage manifest does not cover the six repositories")
    return {**payload, "repositories_by_id": repositories}


def verify_source_checkout(source_dir: Path, code_sha: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("code SHA is not a full Git commit")
    git_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=source_dir, text=True
        ).strip()
    ).resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=git_root, text=True
    ).strip()
    if head != code_sha:
        raise ValueError(f"code SHA {code_sha} does not equal checkout HEAD {head}")
    relative_root = source_dir.relative_to(git_root)
    for name in CANONICAL_CODE:
        path = source_dir / name
        committed = subprocess.check_output(
            ["git", "show", f"{code_sha}:{relative_root.as_posix()}/{name}"],
            cwd=git_root,
        )
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"source file differs from checkout commit: {name}")
    return {"root": str(git_root), "commit": head}


@dataclass
class RepositoryAudit:
    repo: str
    revision: str
    resolved_revision: str | None
    files: list[str]
    code_sha256: dict[str, dict[str, str | bool]]
    release_file_sha256: dict[str, dict[str, str | bool]]
    weights: dict[str, dict[str, Any]]
    tags: dict[str, str]
    failures: list[str]

    @property
    def status(self) -> str:
        return "passed" if not self.failures else "failed"


def audit_repository(
    *,
    api: Any,
    downloader: Callable[..., str],
    repo: str,
    revision: str,
    source_dir: Path,
    required_tag: str | None,
    baseline_weights: dict[str, dict[str, Any]] | None,
    expected_release_files: dict[str, str] | None = None,
) -> RepositoryAudit:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    siblings = list(info.siblings or [])
    files = sorted(str(row.rfilename) for row in siblings)
    failures: list[str] = []
    missing = sorted(set(REQUIRED_FILES) - set(files))
    forbidden = sorted(set(FORBIDDEN_FILES) & set(files))
    if missing:
        failures.append(f"missing required files: {', '.join(missing)}")
    if forbidden:
        failures.append(f"forbidden files present: {', '.join(forbidden)}")

    weights = _weight_rows(siblings)
    if not weights:
        failures.append("no safetensors weights found")
    if any(
        not row.get("sha256") or row.get("size") is None for row in weights.values()
    ):
        failures.append("one or more weights lack Hub LFS SHA256/size metadata")
    if baseline_weights is not None and weights != baseline_weights:
        failures.append("weight LFS metadata differs from the pre-release baseline")

    code_sha256: dict[str, dict[str, str | bool]] = {}
    for name in CANONICAL_CODE:
        expected_path = source_dir / name
        if not expected_path.is_file():
            failures.append(f"local source-of-truth is missing {name}")
            continue
        if name not in files:
            continue
        downloaded = Path(
            downloader(
                repo_id=repo,
                filename=name,
                revision=revision,
                force_download=True,
            )
        ).resolve()
        expected = sha256_file(expected_path)
        actual = sha256_file(downloaded)
        match = actual == expected
        code_sha256[name] = {
            "expected": expected,
            "actual": actual,
            "match": match,
        }
        if not match:
            failures.append(f"canonical source differs: {name}")

    release_file_sha256: dict[str, dict[str, str | bool]] = {}
    for name, expected in sorted((expected_release_files or {}).items()):
        if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", expected):
            failures.append(f"invalid staged file identity: {name}")
            continue
        if name not in files:
            failures.append(f"staged release file is missing: {name}")
            continue
        downloaded = Path(
            downloader(
                repo_id=repo,
                filename=name,
                revision=revision,
                force_download=True,
            )
        ).resolve()
        actual = sha256_file(downloaded)
        match = actual == expected
        release_file_sha256[name] = {
            "expected": expected,
            "actual": actual,
            "match": match,
        }
        if not match:
            failures.append(f"staged release file differs: {name}")

    if "config.json" in files:
        config_path = Path(
            downloader(
                repo_id=repo,
                filename="config.json",
                revision=revision,
                force_download=True,
            )
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "rwkv7":
            failures.append("config model_type is not rwkv7")
        if config.get("auto_map") != EXPECTED_AUTO_MAP:
            failures.append("config auto_map does not match the canonical HF contract")

    refs = api.list_repo_refs(repo_id=repo, repo_type="model")
    tags = {str(tag.name): str(tag.target_commit) for tag in getattr(refs, "tags", [])}
    if required_tag is not None:
        target = tags.get(required_tag)
        if target is None:
            failures.append(f"required tag is missing: {required_tag}")
        elif target != str(info.sha):
            failures.append(
                f"tag {required_tag} targets {target}, not audited revision {info.sha}"
            )

    return RepositoryAudit(
        repo=repo,
        revision=revision,
        resolved_revision=str(info.sha) if info.sha is not None else None,
        files=files,
        code_sha256=code_sha256,
        release_file_sha256=release_file_sha256,
        weights=weights,
        tags=tags,
        failures=failures,
    )


def main() -> int:
    args = arguments()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for the Hub release audit"
        ) from exc

    source_dir = args.source_dir.expanduser().resolve()
    if set(args.repo) != HUB_REPOSITORIES or len(args.repo) != 6:
        raise SystemExit("the Hub release audit must cover each of the six repositories")
    source_checkout = verify_source_checkout(source_dir, args.code_sha)
    baseline = load_weight_baseline(args.weight_baseline)
    if args.weight_baseline is not None and set(baseline) != HUB_REPOSITORIES:
        raise SystemExit("the weight baseline must cover the six repositories")
    release_manifest = load_release_manifest(args.release_manifest)
    if release_manifest:
        if (
            release_manifest.get("source_sha") != args.code_sha
            or release_manifest.get("tag") != args.require_tag
        ):
            raise SystemExit("release-stage manifest source SHA/tag mismatch")
    api = HfApi()
    repositories = []
    for repo in args.repo:
        repositories.append(
            audit_repository(
                api=api,
                downloader=hf_hub_download,
                repo=repo,
                revision=args.revision,
                source_dir=source_dir,
                required_tag=args.require_tag,
                baseline_weights=baseline.get(repo)
                if args.weight_baseline is not None
                else None,
                expected_release_files=(
                    release_manifest["repositories_by_id"][repo].get("file_sha256")
                    if release_manifest
                    else None
                ),
            )
        )
    passed = all(row.status == "passed" for row in repositories)
    report = {
        "schema": "rwkv7-hub-release-audit-v1",
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha": args.code_sha,
        "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        "revision": args.revision,
        "required_tag": args.require_tag,
        "source_dir": str(source_dir),
        "source_checkout": source_checkout,
        "weight_baseline": (
            {
                "path": str(args.weight_baseline.expanduser().resolve()),
                "sha256": sha256_file(args.weight_baseline.expanduser().resolve()),
            }
            if args.weight_baseline is not None
            else None
        ),
        "release_manifest": (
            {
                "path": str(args.release_manifest.expanduser().resolve()),
                "sha256": sha256_file(args.release_manifest.expanduser().resolve()),
            }
            if args.release_manifest is not None
            else None
        ),
        "repositories": [{**asdict(row), "status": row.status} for row in repositories],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
