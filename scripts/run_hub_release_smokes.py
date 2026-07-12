#!/usr/bin/env python3
"""Run all six package-free Hub release smokes from isolated empty caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORIES = (
    "wangyue114514/rwkv7-g1d-0.1b-hf",
    "wangyue114514/rwkv7-g1d-0.4b-hf",
    "wangyue114514/rwkv7-g1g-1.5b-hf",
    "wangyue114514/rwkv7-g1g-2.9b-hf",
    "wangyue114514/rwkv7-g1g-7.2b-hf",
    "wangyue114514/rwkv7-g1g-13.3b-hf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Hub smoke output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def smoke_command(
    *, verifier: Path, repo: str, revision: str, device: str, output_root: Path
) -> tuple[list[str], Path]:
    slug = repo.rsplit("/", 1)[-1]
    directory = output_root / slug
    report = output_root / f"{slug}.json"
    return (
        [
            sys.executable,
            str(verifier),
            "--model",
            repo,
            "--revision",
            revision,
            "--device",
            device,
            "--cache-dir",
            str(directory / "hub-cache"),
            "--modules-cache-dir",
            str(directory / "modules-cache"),
            "--require-empty-cache",
            "--force-download",
            "--require-package-free",
            "--output",
            str(report),
        ],
        report,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_root = prepare_output(args.output_dir)
    verifier = Path(__file__).with_name("verify_hf_release.py").resolve()
    rows = []
    for repo in REPOSITORIES:
        command, report = smoke_command(
            verifier=verifier,
            repo=repo,
            revision=args.revision,
            device=args.device,
            output_root=output_root,
        )
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=output_root, check=False)
        if completed.returncode != 0 or not report.is_file():
            raise SystemExit(f"Hub release smoke failed: {repo}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("status") != "passed" or payload.get("model") != repo:
            raise SystemExit(f"Hub release smoke report is invalid: {repo}")
        rows.append(
            {
                "repo": repo,
                "started_at": started,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "report": str(report),
                "sha256": sha256(report),
            }
        )
    manifest = {
        "schema": "rwkv7-hub-release-smokes-v1",
        "status": "passed",
        "revision": args.revision,
        "device": args.device,
        "repositories": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "repositories": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
