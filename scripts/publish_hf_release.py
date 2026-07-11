#!/usr/bin/env python3
"""Publish an audited staged commit and tag to the six HF repositories.

The command is a dry run unless ``--publish`` is supplied.  It uses each
staged repository's recorded parent commit, so an unexpected concurrent Hub
change aborts instead of being overwritten.  Existing release tags are
verified file-by-file, which makes an interrupted six-repository release safe
to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing_tag(repo_id: str, tag: str, target: Path, files: list[str]) -> str:
    api = HfApi()
    info = api.model_info(repo_id, revision=tag)
    for name in files:
        remote = Path(hf_hub_download(repo_id, name, revision=tag))
        local = target / name
        if sha256(remote) != sha256(local):
            raise SystemExit(f"{repo_id}@{tag}: staged file differs: {name}")
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads((args.stage_dir / "manifest.json").read_text(encoding="utf-8"))
    api = HfApi()
    report = []
    for row in rows:
        repo_id = row["repo_id"]
        target = args.stage_dir / repo_id.rsplit("/", 1)[-1]
        files = list(row["files"])
        if row.get("tag") != args.tag:
            raise SystemExit(
                f"{repo_id}: staged tag {row.get('tag')!r} does not match {args.tag!r}"
            )
        missing = [name for name in files if not (target / name).is_file()]
        if missing:
            raise SystemExit(f"{repo_id}: missing staged files: {missing}")
        if "release.json" in files or any(
            name.endswith(".safetensors") for name in files
        ):
            raise SystemExit(f"{repo_id}: unsafe release file list")

        tags = {ref.name: ref.ref for ref in api.list_repo_refs(repo_id).tags}
        if args.tag in tags:
            commit = verify_existing_tag(repo_id, args.tag, target, files)
            report.append(
                {"repo_id": repo_id, "status": "already-published", "commit": commit}
            )
            continue

        current = api.model_info(repo_id).sha
        if current != row["parent_commit"]:
            raise SystemExit(
                f"{repo_id}: Hub main moved from {row['parent_commit']} to {current}; restage first"
            )
        if not args.publish:
            report.append(
                {
                    "repo_id": repo_id,
                    "status": "dry-run",
                    "parent_commit": current,
                    "files": files,
                }
            )
            continue

        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            parent_commit=current,
            commit_message=f"Release RWKV-7 HF reference {args.tag}",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=target / name)
                for name in files
            ],
        )
        api.create_tag(
            repo_id=repo_id,
            repo_type="model",
            tag=args.tag,
            revision=commit.oid,
            tag_message=(
                "Readable Hugging Face reference implementation with an optional "
                "versioned kernel companion"
            ),
        )
        verified = verify_existing_tag(repo_id, args.tag, target, files)
        report.append({"repo_id": repo_id, "status": "published", "commit": verified})

    payload = {"tag": args.tag, "published": args.publish, "repositories": report}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
