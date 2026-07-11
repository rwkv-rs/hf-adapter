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
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


CANONICAL_CODE = (
    "cache_rwkv7.py",
    "chat_template.jinja",
    "configuration_rwkv7.py",
    "modeling_rwkv7.py",
    "ops_rwkv7.py",
    "tokenization_rwkv7.py",
)
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
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
    parser.add_argument(
        "--weight-baseline",
        type=Path,
        help="Earlier audit JSON whose LFS weight hashes must remain unchanged.",
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


@dataclass
class RepositoryAudit:
    repo: str
    revision: str
    resolved_revision: str | None
    files: list[str]
    code_sha256: dict[str, dict[str, str | bool]]
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
            downloader(repo_id=repo, filename=name, revision=revision)
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

    if "config.json" in files:
        config_path = Path(
            downloader(repo_id=repo, filename="config.json", revision=revision)
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
    baseline = load_weight_baseline(args.weight_baseline)
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
            )
        )
    passed = all(row.status == "passed" for row in repositories)
    report = {
        "schema": "rwkv7-hub-release-audit-v1",
        "status": "passed" if passed else "failed",
        "revision": args.revision,
        "required_tag": args.require_tag,
        "source_dir": str(source_dir),
        "weight_baseline": (
            str(args.weight_baseline.expanduser().resolve())
            if args.weight_baseline is not None
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
